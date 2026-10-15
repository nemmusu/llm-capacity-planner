/**
 * The capacity command (`npm run plan -- <model> [--tokens N] …`): what NVIDIA hardware a
 * Hugging Face model needs with vLLM or SGLang, how many users it serves and how fast, without a GPU and
 * without downloading the weights.
 * Arguments and help: args.ts; output: render.ts; the numbers: src/planner and src/model.
 */
import { execFile } from 'node:child_process';
import { repoLayout } from '../hub/formats';
import { hardwareFileSchema, type GpuSpec } from '../contract/data';
import hardwareJson from '../../data/hardware.json';
import { HubClient } from '../hub/client';
import { DEFAULT_WORKLOAD, type Workload } from '../planner/capacity';
import { DEFAULT_COUNTS, gpuFromSpec, soldAsCards, type Gpu } from '../planner/hardware';
import { plannerModel } from '../planner/model';
import type { PlanOptions, PlanRow } from '../planner/plan';
import { planOverview } from './overview';
import { documentedMaxTokens } from './compat';
import { shortGpuName } from './figures';
import { cachedFetch, defaultCacheDir } from './http-cache';
import { checkContext, parseArgs, USAGE, type PlanArgs } from './args';
import { asciiOnly, renderCsv, renderJson, renderMarkdown, renderText, type Answer, type RenderModel } from './render';
import { readRemoteModel } from './model';
import reportJson from '../../data/report.json';
import { reportCsv, reportFileSchema, reportJsonText, reportLines, reportMarkdown, renderReport } from './report';

/** The GPU catalog bundled with the command (data/hardware.json, every value with a source). */
export function catalog(json: unknown = hardwareJson): Gpu[] {
  return hardwareFileSchema.parse(json).gpus.map((g: GpuSpec) => gpuFromSpec(g)).filter(soldAsCards);
}

export interface LocalGpu {
  name: string;
  totalMiB: number;
}

/** The first NVIDIA GPU of this PC (nvidia-smi), undefined without one (a Mac, a PC without NVIDIA). */
export function localGpu(run: typeof execFile = execFile): Promise<LocalGpu | undefined> {
  return new Promise((resolve) => {
    run('nvidia-smi', ['--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], { timeout: 5000, windowsHide: true }, (err, stdout) => {
      const [name, total] = (err ? '' : (String(stdout).split(/\r?\n/)[0] as string)).split(',').map((s) => s.trim());
      const mib = Number(total);
      resolve(name && Number.isFinite(mib) && mib > 0 ? { name, totalMiB: mib } : undefined);
    });
  });
}

/** Catalog cards matching what the user typed ("5090", "h100", "pro-6000"…). */
export function matchGpus(gpus: readonly Gpu[], query: string): Gpu[] {
  const q = query.toLowerCase().replace(/^((nvidia|geforce|rtx)[\s-]*)+/, '').replace(/\s+/g, '-');
  const name = (g: Gpu) => g.name.toLowerCase().replace(/^nvidia (geforce )?(rtx )?/, '').replace(/\s+/g, '-');
  // The name the tables show ("RTX PRO 6000 Server") matches too.
  const shown = (g: Gpu) => shortGpuName(g.name).toLowerCase().replace(/^rtx /, '').replace(/\s+/g, '-');
  const all = gpus.filter((g) => g.id.includes(q) || name(g).includes(q) || shown(g).includes(q));
  // "h200" means the H200s, not the GH200: cards whose id or name starts with it come first; "l4" means the L4, not
  // the L40S: before them, the cards that start with it as a whole word.
  const id = (g: Gpu) => g.id.replace(/^rtx-/, '');
  const starts = all.filter((g) => id(g).startsWith(q) || name(g).startsWith(q) || shown(g).startsWith(q));
  const word = (s: string) => s === q || s.startsWith(`${q}-`);
  const whole = starts.filter((g) => word(id(g)) || word(name(g)) || word(shown(g)));
  return whole.length > 0 ? whole : starts.length > 0 ? starts : all;
}

/** A card the catalog does not have: its memory only (no bandwidth or TFLOPS: no speed). */
export function customGpu(name: string, totalMiB: number): Gpu {
  return {
    id: 'custom',
    name,
    segment: 'consumer',
    architecture: 'blackwell',
    cc: 0,
    cudaMiB: { min: totalMiB - 520, max: totalMiB - 16 },
    totalMiB,
    nominalGB: Math.round(totalMiB / 1024),
    bandwidthGBs: 0,
    tflops: { bf16: null, fp8: null, fp4: null },
    linkGBs: 1,
    link: 'pcie-host',
    formats: { fp8: true, nvfp4: false, mxfp4: false, mxfp8: false },
  };
}

export interface PlanDeps {
  fetch: typeof fetch;
  env: Record<string, string | undefined>;
  localGpu: () => Promise<LocalGpu | undefined>;
  out: (text: string) => void;
  writeFile: (file: string, content: string) => Promise<void>;
  /** https://huggingface.co (tests: a fake hub). */
  baseUrl: string;
  cacheDir?: string;
  isTTY: boolean;
  columns?: number;
  platform: NodeJS.Platform;
  gpus?: Gpu[];
  /** The ready-made reports (default: data/report.json bundled with the command). */
  report?: unknown;
  /** Models shipped with the command (a data pack): read without the network unless --refresh. */
  builtin?: BuiltinModels;
}

/** Built-in model descriptions: `fetch` answers the same hub endpoints as Hugging Face (info, tree, files, Range reads). */
export interface BuiltinModels {
  /** When the pack was made (said in the output). */
  date: string;
  has(repo: string, revision: string): boolean;
  fetch: typeof fetch;
}

/** Runs the command: 0 = answer printed, 1 = error, 2 = model not found (suggestions printed). */
export async function runPlan(argv: readonly string[], deps: PlanDeps): Promise<number> {
  let args: PlanArgs | undefined;
  try {
    args = parseArgs(argv, { env: deps.env, isTTY: deps.isTTY, platform: deps.platform });
  } catch (e) {
    deps.out(`Error: ${(e as Error).message}\n\n${USAGE}`);
    return 1;
  }
  if (!args) {
    deps.out(USAGE);
    return 0;
  }
  const a = args;
  const say = (text: string) => deps.out(a.ascii ? asciiOnly(text) : text);
  // The terminal width (COLUMNS when the output goes to a file or a pipe), 120 by default.
  const columns = deps.columns ?? (Number(deps.env.COLUMNS) || 120);
  if (a.report) {
    const file = reportFileSchema.parse(deps.report ?? reportJson);
    const lines = reportLines(file, a.reportFilter);
    if (lines.length === 0) {
      say(`No model of the reports matches "${a.reportFilter ?? ''}". Models: ${file.models.map((m) => m.name).join(', ')}.`);
      return 2;
    }
    say(renderReport(file, lines, { color: a.color, ascii: a.ascii, columns }));
    for (const [what, target, content] of [
      ['CSV', a.csv, () => reportCsv(file, lines, a.csvExcelEu)],
      ['JSON', a.json, () => reportJsonText(lines)],
      ['Markdown', a.md, () => reportMarkdown(file, lines)],
    ] as const) {
      if (!target) continue;
      await deps.writeFile(target, content());
      say(`${what} written to ${target}`);
    }
    return 0;
  }
  const token = deps.env.HF_TOKEN ?? deps.env.HUGGING_FACE_HUB_TOKEN;
  // What was read, to say it (the weights never are).
  const read = { network: 0, cache: 0 };
  const fetchFn = cachedFetch(deps.fetch, {
    dir: deps.cacheDir ?? defaultCacheDir(deps.env, deps.platform),
    offline: a.offline,
    refresh: a.refresh,
    onRead: (bytes, from) => {
      read[from] += bytes;
    },
  });
  // A built-in model needs no network (and no cache), even --offline; --refresh reads Hugging Face again.
  const builtin = deps.builtin && !a.refresh && deps.builtin.has(a.repo, a.revision) ? deps.builtin : undefined;
  const hub = new HubClient({ fetch: builtin ? builtin.fetch : fetchFn, baseUrl: deps.baseUrl, getToken: () => token });
  try {
    const repo = a.repo;
    const found = /^[\w.-]+\/[\w.-]+$/.test(repo) ? await hub.info(repo, a.revision === 'main' ? undefined : a.revision).catch(notFound) : undefined;
    if (!found) {
      const page = await hub.search(repo.replace(/\//g, ' '), { format: 'safetensors', limit: 8 });
      // MLX conversions and the like are listed as safetensors too: vLLM and SGLang cannot load them.
      const models = page.models.filter((m) => m.formats.includes('safetensors'));
      if (models.length === 0) {
        say(`"${repo}" does not exist on Hugging Face and the search finds no safetensors model.`);
        return 2;
      }
      say(
        [
          `"${repo}" is not an exact repo id. Safetensors models found:`,
          ...models.map((m) => `  ${m.repo}${m.paramsB ? `  (${m.paramsB.toLocaleString('en-US', { maximumFractionDigits: 1 })} B parameters)` : ''}`),
          '',
          'Run again with one of these ids.',
        ].join('\n'),
      );
      return 2;
    }
    const files = await hub.files(repo, a.revision);
    const variants = repoLayout(files, { name: repo, tags: found.tags }).safetensors;
    const variant = variants.find((v) => v.dir === a.variant);
    if (!variant) {
      const dirs = variants.map((v) => (v.dir === '' ? '(root)' : v.dir));
      say(
        dirs.length === 0
          ? `${repo} has no safetensors model vLLM or SGLang can load (config.json + weights).`
          : `${repo}: no safetensors model in "${a.variant || '(root)'}". Variants: ${dirs.join(', ')} (use --variant).`,
      );
      return 1;
    }
    // Mistral-format checkpoints (params.json, no config.json): vLLM loads them with --config-format mistral, this
    // calculator reads config.json only.
    const inDir = (name: string) => files.some((f) => f.path === (variant.dir ? `${variant.dir}/${name}` : name));
    if (!inDir('config.json') && inDir('params.json')) {
      say(`${repo} ships only the Mistral format (params.json, no config.json): the calculator reads config.json, so it cannot plan this checkpoint yet. A version of the model with a config.json works.`);
      return 1;
    }
    const remote = await readRemoteModel(hub, repo, a.revision, variant);
    // With the repo, the compatibility matrix is read for this exact checkpoint, not another model of the same class.
    const model = plannerModel(remote.config, remote.facts, remote.shape, remote.tensors, repo);
    const maxContext = a.tokens ?? remote.facts.contextLength ?? 8192;
    if (a.tokens === undefined) checkContext(maxContext, a);
    const workload: Workload = {
      maxContext,
      typicalContext: Math.min(maxContext, a.typicalTokens ?? Math.max(1, Math.floor(maxContext / 2))),
      outputTokens: a.outputTokens,
      minTokS: a.minTokS,
      maxTtftMs: a.maxTtftS * 1000,
      activeShare: a.activeShare,
      safety: DEFAULT_WORKLOAD.safety,
      cachedShare: a.cachedShare,
      ttftScales: !a.ttftSet,
    };
    const opts: PlanOptions = {
      engine: a.engine,
      kvCache: a.cache,
      workload,
      textOnly: a.textOnly,
      ...(a.util !== undefined ? { util: a.util } : {}),
      ...(a.maxBatchedTokens !== undefined ? { maxBatchedTokens: a.maxBatchedTokens } : {}),
      ...(a.maxSeqs !== undefined ? { maxSeqs: a.maxSeqs } : {}),
    };
    const all = deps.gpus ?? catalog();
    let gpus = all;
    if (a.vramGB !== undefined) gpus = [customGpu(`${a.vramGB} GiB GPU`, Math.round(a.vramGB * 1024))];
    else if (a.gpu) {
      gpus = matchGpus(all, a.gpu);
      if (gpus.length === 0) {
        say(`No card of the catalog matches "${a.gpu}". Cards: ${all.map((g) => g.id).join(', ')}.`);
        return 1;
      }
    }
    const counts = a.count ? [a.count] : DEFAULT_COUNTS;
    const users = a.users !== undefined ? { total: a.users, concurrent: Math.max(1, Math.ceil(a.users * workload.activeShare)), active: workload.activeShare } : undefined;
    // This PC's card is one line of the overview (planned apart when the list leaves it out).
    const local = a.thisPc ? await deps.localGpu() : undefined;
    let localCard: Gpu | undefined;
    if (local) {
      const plain = local.name.replace(/^NVIDIA (GeForce )?/i, '');
      localCard = all.find((g) => g.name.replace(/^NVIDIA (GeForce )?/, '').toLowerCase() === plain.toLowerCase()) ?? { ...customGpu(plain, local.totalMiB), id: 'this-pc' };
    }
    const localApart = localCard && !gpus.some((g) => g.id === localCard.id) ? localCard : undefined;
    // The same overview as the ready-made reports (overview.ts).
    const { rows, cards, general, fig, compatChecked } = planOverview({ model, repo, opts, gpus, counts, ...(users ? { users } : {}), ...(localApart ? { apart: localApart } : {}), ...(localCard ? { localId: localCard.id } : {}) });
    const choice = (c: NonNullable<typeof general.fit>) => ({
      figures: fig(c.row),
      ...(c.alternative ? { alternative: fig(c.alternative) } : {}),
      ...(c.servers !== undefined ? { servers: c.servers } : {}),
      ...(c.alternativeServers !== undefined ? { alternativeServers: c.alternativeServers } : {}),
      ...(c.usersPerServer !== undefined ? { usersPerServer: c.usersPerServer } : {}),
      ...(c.likelyUsersPerServer !== undefined ? { likelyUsersPerServer: c.likelyUsersPerServer } : {}),
      ...(c.alternativeUsersPerServer !== undefined ? { alternativeUsersPerServer: c.alternativeUsersPerServer } : {}),
    });
    const picked = a.gpu !== undefined || a.vramGB !== undefined;
    const answer: Answer = {
      cards: cards.map((c) => c.line),
      rows: rows.map(fig),
      ...(general.fit ? { fit: choice(general.fit) } : {}),
      ...(general.attempt ? { attempt: fig(general.attempt) } : {}),
      ...(general.blocked ? { blocked: fig(general.blocked) } : {}),
      ...(users ? { users: { ...users, ...(general.users ? { choice: choice(general.users) } : {}) } } : {}),
      ...(a.allHardware || picked ? { layouts: rows.map(fig) } : {}),
    };
    // --details: the layout the answer names (the users' one, the chosen card's, the simplest), else this
    // PC's; --gpu / --vram always show the chosen card's.
    const detailed = picked ? (cards[0] as (typeof cards)[number]).row : general.users?.row ?? general.fit?.row ?? cards.find((c) => c.line.local)?.row ?? (rows[0] as PlanRow);
    if (a.details || picked) answer.details = { figures: picked ? (cards[0] as (typeof cards)[number]).line.figures : fig(detailed), row: detailed };
    // What the answer covers: the cards and the GPU count asked for.
    const selection = picked || a.count !== undefined ? { cards: picked ? gpus.map((g) => shortGpuName(g.name)) : [], ...(a.count !== undefined ? { count: a.count } : {}) } : undefined;
    const documented = documentedMaxTokens(repo);
    const render: RenderModel = {
      ...(selection ? { selection } : {}),
      ...(documented !== undefined ? { documentedMaxContext: documented } : {}),
      repo,
      variant: variant.dir,
      format: model.parallel.weights.toUpperCase(),
      family: FAMILY_TEXT[model.family],
      verification: model.verification,
      engine: a.engine,
      cache: a.cache,
      workload,
      contextIsModelMax: a.tokens === undefined,
      ...(remote.facts.contextLength !== undefined ? { modelMaxContext: remote.facts.contextLength } : {}),
      textOnly: a.textOnly,
      hasVision: remote.facts.visionEncoder !== undefined,
      compatChecked,
      source: builtin
        ? { kind: 'built-in', date: builtin.date }
        : read.network > 0
          ? { kind: 'network', bytes: read.network, cachedBytes: read.cache }
          : { kind: 'cache', bytes: read.cache },
    };
    deps.out(renderText(render, answer, { color: a.color, ascii: a.ascii, columns }));
    for (const [what, file, content] of [
      ['CSV', a.csv, () => renderCsv(render, answer, a.csvExcelEu)],
      ['JSON', a.json, () => renderJson(render, answer)],
      ['Markdown', a.md, () => renderMarkdown(render, answer)],
    ] as const) {
      if (!file) continue;
      await deps.writeFile(file, content());
      say(`${what} written to ${file}`);
    }
    return 0;
  } catch (e) {
    const code = (e as { code?: unknown }).code;
    if (code === 'HF_GATED' || code === 'HF_AUTH_REQUIRED')
      say(`Error: ${a.repo} is gated or private: accept its license on huggingface.co and set HF_TOKEN (${(e as Error).message}).`);
    else if (builtin) say(`Error: the built-in data of ${a.repo} is incomplete (${(e as Error).message}); --refresh reads the model from Hugging Face.`);
    else if (code === 'NETWORK' && a.offline) say(`Error: ${a.repo} has not been read on this computer yet (--offline uses only the local cache): run once without --offline.`);
    else if (code === 'NETWORK') say(`Error: Hugging Face does not answer (${(e as Error).message}). Try again, or use --offline for the models read before.`);
    else say(`Error: ${(e as Error).message}`);
    return 1;
  }
}

const FAMILY_TEXT = { dense: 'dense', moe: 'MoE', mla: 'MLA', hybrid: 'hybrid', multimodal: 'multimodal' } as const;

/** A repo that does not exist (HF answers 401 for those too when no token is sent). */
function notFound(e: unknown): undefined {
  const code = (e as { code?: unknown }).code;
  if (code === 'HF_NOT_FOUND' || code === 'HF_AUTH_REQUIRED') return undefined;
  throw e;
}

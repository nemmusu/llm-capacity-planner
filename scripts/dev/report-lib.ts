// Ready-made capacity reports for the recent open models of data/models.json, computed offline from the model
// fixtures written by scripts/dev/update-models.mjs. scripts/dev/report-entry.ts writes them to data/report.json;
// test/docs/report.test.ts recomputes them and compares, so the file never falls behind the code.
// For each model, its base checkpoint and official quantized versions, at the native maximum context and at 1M tokens
// (natively or with YaRN), every catalog layout of 1–8 GPUs is planned through the command's own overview
// (src/cli/overview.ts); `npm run plan -- report` shows and exports the result.
import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { gunzipSync } from 'node:zlib';
import { hardwareFileSchema } from '../../src/contract/data';
import { factsFromHfConfig, loadShapeFromHfConfig } from '../../src/model/hf-config';
import { tensorsFromHeader, type TensorInfo } from '../../src/model/weights-exact';
import { calibrationFor } from '../../src/planner/calibration';
import { DEFAULT_WORKLOAD, type Workload } from '../../src/planner/capacity';
import { DEFAULT_COUNTS, gpuFromSpec, soldAsCards } from '../../src/planner/hardware';
import { plannerModel } from '../../src/planner/model';
import { planLayout } from '../../src/planner/plan';
import { documentedMaxTokens } from '../../src/cli/compat';
import { planOverview } from '../../src/cli/overview';
import { packReport } from '../../src/cli/report';
import { sourceHash } from '../source-hash.mjs';
import type * as Pack from '../pack-models.mjs';

interface ModelsFile {
  models: {
    id: string;
    repo: string;
    name: string;
    context: { configMaxPositions: number | null; cardMaxTokens: number | null };
    quantizations: { format: string; repo: string; kind: string; weightsVariantDir: string | null; gated?: boolean }[];
    longContext: { case: string; to1M?: { maxModelLen?: number; hfOverrides?: Record<string, unknown>; vllmArgs?: string[]; sglangArgs?: string[] } | null };
  }[];
}

const readJson = (file: string): unknown => {
  if (existsSync(file)) return JSON.parse(readFileSync(file, 'utf8'));
  if (existsSync(`${file}.gz`)) return JSON.parse(gunzipSync(readFileSync(`${file}.gz`)).toString('utf8'));
  return undefined;
};

/** Every tensor of a checkpoint: from the full headers, else from the per-pattern summary ({n} = layers, {e} = experts). */
function tensorsOf(dir: string): TensorInfo[] | undefined {
  const headers = readJson(path.join(dir, 'safetensors-headers.json')) as { shards: Record<string, { header: unknown }> } | undefined;
  if (headers) return Object.values(headers.shards).flatMap((s) => tensorsFromHeader(s.header));
  const summary = readJson(path.join(dir, 'tensor-summary.json')) as { entries: { pattern: string; dtype: string; shape: number[]; count: number; layers: string | null }[] } | undefined;
  if (!summary) return undefined;
  const out: TensorInfo[] = [];
  for (const e of summary.entries) {
    const layers = e.layers
      ? e.layers.split(',').flatMap((part) => {
          const [a, b] = part.split('-').map(Number);
          return b === undefined ? [a as number] : Array.from({ length: (b as number) - (a as number) + 1 }, (_, i) => (a as number) + i);
        })
      : [undefined];
    const perLayer = Math.max(1, Math.round(e.count / layers.length));
    for (const n of layers)
      for (let k = 0; k < perLayer; k++) {
        let name = e.pattern.replace('{n}', String(n));
        name = /\{\w+\}/.test(name) ? name.replace(/\{\w+\}/, String(k)) : perLayer > 1 ? `${name}#${k}` : name;
        out.push({ name, dtype: e.dtype, shape: e.shape });
      }
  }
  return out;
}

/** config.json with the 1M overrides merged (vLLM --hf-overrides merges a text_config dict into the sub-config). */
function merge(base: unknown, over: Record<string, unknown>): unknown {
  const b = (base ?? {}) as Record<string, unknown>;
  const out: Record<string, unknown> = { ...b };
  for (const [k, v] of Object.entries(over)) out[k] = v && typeof v === 'object' && !Array.isArray(v) && typeof b[k] === 'object' ? merge(b[k], v as Record<string, unknown>) : v;
  return out;
}

const LABEL: Record<string, string> = {
  'native>=1M': 'native',
  'yarn-official': 'official YaRN',
  'yarn-beyond-declared': 'YaRN beyond the declared factor: quality not guaranteed',
  'not-applicable': 'not applicable',
};

const CHAT_CACHED_SHARE = 0.8;


/**
 * A fingerprint of what the report is computed from: the command's sources and data (without data/report.json), the
 * model fixtures and this file. data/report.json carries it; test/docs/report.test.ts compares.
 */
export async function reportInputs(root: string): Promise<string> {
  // Loaded at run time, not bundled: run directly, it would start its own command line.
  const { modelsHash } = (await import(pathToFileURL(path.join(root, 'scripts', 'pack-models.mjs')).href)) as typeof Pack;
  const lib = readFileSync(path.join(root, 'scripts', 'dev', 'report-lib.ts'), 'utf8').replace(/\r\n/g, '\n');
  return createHash('sha256').update(sourceHash(root, ['data/report.json'])).update(modelsHash(path.join(root, 'data', 'models'))).update(lib).digest('hex').slice(0, 12);
}

/** data/report.json's content, computed from the model fixtures under `root` and the current code; `only`: these model ids. */
export async function buildReport(root: string, only?: readonly string[]): Promise<{ file: Record<string, unknown>; models: Record<string, unknown>[]; missing: string[] }> {
  const fixtures = path.join(root, 'data/models');
  const hw = hardwareFileSchema.parse(JSON.parse(readFileSync(path.join(root, 'data/hardware.json'), 'utf8')));
  const gpus = hw.gpus.map(gpuFromSpec).filter(soldAsCards);
  const models = JSON.parse(readFileSync(path.join(root, 'data/models.json'), 'utf8')) as ModelsFile;
  const report: Record<string, unknown>[] = [];
  const missing: string[] = [];
  for (const m of models.models.filter((x) => !only || only.includes(x.id))) {
    const base = path.join(fixtures, m.repo.replace('/', '__'));
    const variants: Record<string, unknown>[] = [];
    for (const q of m.quantizations.filter((x) => x.kind === 'base' || x.kind === 'official')) {
      if (q.weightsVariantDir === null) continue;
      const dir = q.weightsVariantDir === '' ? base : path.join(base, q.weightsVariantDir);
      const config = readJson(path.join(dir, 'config.json'));
      const tensors = tensorsOf(dir);
      if (!config || !tensors || tensors.length === 0) {
        missing.push(`${q.repo} (${q.format})`);
        continue;
      }
      const native = m.context.configMaxPositions ?? m.context.cardMaxTokens ?? 32768;
      const contexts: { tokens: number; label: string; config: unknown; args?: { vllm?: string[]; sglang?: string[] } }[] = [{ tokens: native, label: 'native maximum context', config }];
      const lc = m.longContext;
      if (native < 1_000_000 && lc.case !== 'not-applicable' && lc.to1M?.hfOverrides)
        contexts.push({
          tokens: lc.to1M.maxModelLen ?? 1_048_576,
          label: LABEL[lc.case] ?? lc.case,
          config: merge(config, lc.to1M.hfOverrides),
          args: { ...(lc.to1M.vllmArgs ? { vllm: lc.to1M.vllmArgs } : {}), ...(lc.to1M.sglangArgs ? { sglang: lc.to1M.sglangArgs } : {}) },
        });
      const out: Record<string, unknown>[] = [];
      for (const c of contexts) {
        const facts = factsFromHfConfig(c.config, { name: q.repo });
        const model = plannerModel(c.config, facts, loadShapeFromHfConfig(c.config), tensors, q.repo);
        const workload: Workload = { ...DEFAULT_WORKLOAD, maxContext: c.tokens, typicalContext: Math.floor(c.tokens / 2) };
        const opts = { engine: 'vllm' as const, kvCache: 'fp8' as const, workload, textOnly: false };
        // The command's own overview: the compatibility matrix applied, one line per card, and the general answer.
        const { fig, cards: lines, general } = planOverview({ model, repo: q.repo, opts, gpus, counts: DEFAULT_COUNTS });
        const cards = lines.map((x) => x.line);
        // The answer's layout for a chat: 80 % of each request's context already in the prefix cache.
        const chatCap = general.fit ? planLayout(model, general.fit.row.layout, { ...opts, workload: { ...workload, cachedShare: CHAT_CACHED_SHARE } }).capacity : undefined;
        out.push({
          tokens: c.tokens,
          label: c.label,
          // The model's own maximum and the one its lab documents: beyond both, the command recommends nothing
          // (render.ts outsideLimits), and the report says the same.
          modelMaxContext: native,
          ...(documentedMaxTokens(q.repo) !== undefined ? { documentedMaxContext: documentedMaxTokens(q.repo) } : {}),
          ...(c.args ? { args: c.args } : {}),
          format: model.parallel.weights,
          family: model.family,
          verification: model.verification.status,
          ...(general.fit ? { fit: fig(general.fit.row), ...(general.fit.alternative ? { alternative: fig(general.fit.alternative) } : {}) } : {}),
          ...(general.attempt ? { attempt: fig(general.attempt) } : {}),
          ...(general.blocked ? { blocked: fig(general.blocked) } : {}),
          ...(chatCap ? { chat: { guaranteed: chatCap.totalUsers, likely: chatCap.probableUsers } } : {}),
          cards,
        });
      }
      variants.push({ format: q.format, repo: q.repo, contexts: out });
    }
    report.push({ id: m.id, repo: m.repo, name: m.name, toMillion: LABEL[m.longContext.case] ?? m.longContext.case, variants });
  }
  const cal = calibrationFor('vllm');
  /** Numbers with 6 significant digits (the file is bundled with the command): enough that a figure printed with one
   * decimal reads the same as the live command (4 digits turned 89.4499 into 89.45, printed 89.5). */
  const round = (_k: string, v: unknown) => (typeof v === 'number' && !Number.isInteger(v) ? Number(v.toPrecision(6)) : v);
  const rounded = JSON.parse(JSON.stringify({ models: report, missing }, round)) as { models: Record<string, unknown>[]; missing: string[] };
  const file = { $comment: 'Written by npm run dev:report (scripts/dev/report-lib.ts): one line per catalog card for each model version and context (vLLM, fp8 KV cache, default workload), as the command shows it.', inputs: await reportInputs(root), engine: 'vllm', engineVersion: cal.engineVersion, workload: DEFAULT_WORKLOAD, chatCachedShare: CHAT_CACHED_SHARE, packed: packReport(rounded) };
  return { file, ...rounded };
}

/** The file's text, as written. */
export const reportText = (file: Record<string, unknown>) => `${JSON.stringify(file)}\n`;

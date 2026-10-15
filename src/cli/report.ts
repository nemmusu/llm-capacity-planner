/**
 * `report [name]`: the ready-made reports of data/report.json (written by
 * scripts/dev/report.mjs) for the recent open models, at their native maximum context and at 1M
 * tokens. Each model version and context holds the same overview as the command: the general answer
 * and one line per catalog card, every number with its range and accuracy (figures.ts).
 * The file is bundled with the command, so it is packed: every object as its key list's index plus the
 * values, every string once (packReport / unpackReport, lossless).
 */
import { z } from 'zod';
import { DEFAULT_WORKLOAD, type Workload } from '../planner/capacity';
import { asciiOnly, int, wrap } from './format';
import { cardLineSchema, figuresSchema, ktok, type CardLine } from './figures';
import { COMMAND } from './args';
import { cardColumns, csvText, headline, outsideLimits, mdTable, notEnoughText, Notes, table, USERS_LEGEND, type Answer, type CsvModel, type RenderModel, type TextStyle } from './render';

const contextSchema = z.object({
  tokens: z.number(),
  label: z.string(),
  /** The model's own maximum context and the one its lab documents: beyond both, nothing is recommended. */
  modelMaxContext: z.number().optional(),
  documentedMaxContext: z.number().optional(),
  args: z.object({ vllm: z.array(z.string()).optional(), sglang: z.array(z.string()).optional() }).optional(),
  format: z.string(),
  family: z.string(),
  verification: z.string(),
  /** The fewest GPUs the model fits on (and serves someone), and the first option of the other market. */
  fit: figuresSchema.optional(),
  alternative: figuresSchema.optional(),
  /** Nothing fits: the layout closest to fitting. */
  attempt: figuresSchema.optional(),
  /** Nothing runs but the memory fits: the layout the engine refuses. */
  blocked: figuresSchema.optional(),
  /** Users of `fit` for a chat (most of each context already in the prefix cache). */
  chat: z.object({ guaranteed: z.number(), likely: z.number() }).optional(),
  cards: z.array(cardLineSchema),
});

const unpackedSchema = z.object({
  engine: z.string(),
  engineVersion: z.string(),
  workload: z.object({ outputTokens: z.number(), minTokS: z.number(), maxTtftMs: z.number(), activeShare: z.number(), safety: z.number(), cachedShare: z.number() }),
  chatCachedShare: z.number(),
  models: z.array(
    z.object({
      id: z.string(),
      repo: z.string(),
      name: z.string(),
      toMillion: z.string(),
      variants: z.array(z.object({ format: z.string(), repo: z.string(), contexts: z.array(contextSchema) })),
    }),
  ),
  missing: z.array(z.string()),
});

/** Packed values: objects as [shape index, …values], arrays as ["~a", …items], strings as "~<index>". */
type Packed = number | boolean | null | string | Packed[];

/** Every object as its key list's index and its values, every string once (lossless; see the module comment). */
export function packReport(value: unknown): { shapes: string[][]; strings: string[]; data: Packed } {
  const shapes: string[][] = [];
  const shapeIds = new Map<string, number>();
  const strings: string[] = [];
  const stringIds = new Map<string, number>();
  const pack = (v: unknown): Packed => {
    if (typeof v === 'string') {
      if (!stringIds.has(v)) stringIds.set(v, strings.push(v) - 1);
      return `~${stringIds.get(v) as number}`;
    }
    if (Array.isArray(v)) return ['~a', ...v.map(pack)];
    if (v !== null && typeof v === 'object') {
      const entries = Object.entries(v).filter(([, x]) => x !== undefined);
      const key = entries.map(([k]) => k).join('\u0000');
      if (!shapeIds.has(key)) shapeIds.set(key, shapes.push(entries.map(([k]) => k)) - 1);
      return [shapeIds.get(key) as number, ...entries.map(([, x]) => pack(x))];
    }
    return v as Packed;
  };
  return { shapes, strings, data: pack(value) };
}

/** The inverse of packReport. */
export function unpackReport(p: { shapes: string[][]; strings: string[]; data: Packed }): unknown {
  const unpack = (v: Packed): unknown => {
    if (typeof v === 'string') return p.strings[Number(v.slice(1))];
    if (!Array.isArray(v)) return v;
    if (v[0] === '~a') return v.slice(1).map(unpack);
    const keys = p.shapes[v[0] as number] as string[];
    return Object.fromEntries(keys.map((k, i) => [k, unpack(v[i + 1] as Packed)]));
  };
  return unpack(p.data);
}

/** The bundled file (packed) or an already unpacked one (tests). */
export const reportFileSchema = z.preprocess(
  (raw) => {
    if (!raw || typeof raw !== 'object' || !('packed' in raw)) return raw;
    const { packed, ...rest } = raw as { packed: Parameters<typeof unpackReport>[0] };
    return { ...rest, ...(unpackReport(packed) as object) };
  },
  unpackedSchema,
);
export type ReportFile = z.infer<typeof unpackedSchema>;
export type ReportContext = z.infer<typeof contextSchema>;

/** One line of the report: a model version at one context. */
export interface ReportLine {
  model: string;
  repo: string;
  version: string;
  versionRepo: string;
  context: ReportContext;
}

/** The report's lines, optionally only the models whose id, name or repo contains `filter`. */
export function reportLines(file: ReportFile, filter?: string): ReportLine[] {
  const f = filter?.toLowerCase();
  const out: ReportLine[] = [];
  for (const m of file.models) {
    if (f && ![m.id, m.name, m.repo].some((s) => s.toLowerCase().includes(f))) continue;
    for (const v of m.variants) for (const c of v.contexts) out.push({ model: m.name, repo: m.repo, version: v.format, versionRepo: v.repo, context: c });
  }
  return out;
}

/** The workload the report was computed with, at a line's context. */
function workloadOf(file: ReportFile, l: ReportLine): Workload {
  return { ...DEFAULT_WORKLOAD, ...file.workload, maxContext: l.context.tokens, typicalContext: Math.floor(l.context.tokens / 2) };
}

/**
 * The cards that do not fit, grouped by the GPUs tried in one server: "even with 8: H100 PCIe, A100 40GB SXM4; even
 * with 4: RTX 3090, RTX 4090; even with 2: GH200 96GB" (packed cards without serverGpus: the most GPUs tried).
 */
export function doNotFit(cards: readonly CardLine[]): string {
  const groups = new Map<number, string[]>();
  for (const c of cards) {
    const n = c.serverGpus ?? c.maxGpus;
    groups.set(n, [...(groups.get(n) ?? []), c.gpu]);
  }
  return [...groups.entries()]
    .sort((a, b) => b[0] - a[0])
    .map(([n, names]) => `even with ${n}: ${names.join(', ')}`)
    .join('; ');
}

/** A context beyond what the model is made for: the figures follow, but nothing is recommended (as the command says). */
const outsideText = (limit: number) => `Outside the model's limits: it is made for up to ${ktok(limit)} tokens. The figures below are computed with the context extension, not a recommendation.`;

/** What headline() needs of a report line. */
function asAnswer(file: ReportFile, l: ReportLine): { model: RenderModel; answer: Answer } {
  const c = l.context;
  const model: RenderModel = {
    repo: l.versionRepo,
    variant: '',
    format: c.format.toUpperCase(),
    family: c.family,
    // headline() reads the repo, the cache and the workload only.
    verification: { status: c.verification === 'gap' || c.verification === 'unverified' ? c.verification : 'verified', gapShare: 0, gapFixedMiB: 0, evidence: [] },
    engine: file.engine === 'sglang' ? 'sglang' : 'vllm',
    cache: 'fp8',
    workload: workloadOf(file, l),
    contextIsModelMax: false,
    ...(c.modelMaxContext !== undefined ? { modelMaxContext: c.modelMaxContext } : {}),
    ...(c.documentedMaxContext !== undefined ? { documentedMaxContext: c.documentedMaxContext } : {}),
    textOnly: false,
    hasVision: false,
  };
  const answer: Answer = {
    cards: c.cards,
    rows: c.cards.map((x) => x.figures),
    ...(c.fit ? { fit: { figures: c.fit, ...(c.alternative ? { alternative: c.alternative } : {}) } } : {}),
    ...(c.attempt ? { attempt: c.attempt } : {}),
    ...(c.blocked ? { blocked: c.blocked } : {}),
  };
  return { model, answer };
}

/** "RTX 5090: does not fit even with 8" / "RTX 5090: fits with 1, 60 users guaranteed, 70 likely". */
export function cardSummary(l: CardLine | undefined): string {
  if (!l) return 'not evaluated';
  const f = l.figures;
  const users = f.users ? `, ${int(f.users.guaranteed)} users guaranteed, ${int(f.users.likely)} likely` : '';
  if (l.status === 'no-fit') return `does not fit even with ${l.maxGpus}`;
  if (l.status === 'no-start') return `the engine does not start (${l.note as string})`;
  if (l.status === 'no-speed') return `fits with ${f.count}, speed unknown`;
  return `${l.status === 'at-limit' ? 'at the limit' : l.status === 'slow' ? 'fits, too slow,' : 'fits'} with ${f.count}${users}`;
}

/** Above this many lines the report lists the answers only (a filter shows every card). */
export const OVERVIEW_MAX_LINES = 6;

/**
 * The report as text: per model version and context the general answer, the RTX 5090's line and, when
 * few lines are selected, the overview per card (the cards that do not fit summed up in one line).
 */
export function renderReport(file: ReportFile, lines: readonly ReportLine[], style: TextStyle = { color: false, ascii: false, columns: 120 }): string {
  const width = Math.max(60, style.columns);
  const out: string[] = [];
  const para = (t: string, indent = 0) => out.push(...wrap(t, width - indent).map((x) => `${' '.repeat(indent)}${x}`));
  para(
    `Ready-made reports: ${file.engine === 'vllm' ? 'vLLM' : 'SGLang'} ${file.engineVersion}, fp8 KV cache. Each request: half of the context as new text to read (documents), ${file.workload.outputTokens} tokens generated, ` +
      `≥${file.workload.minTokS} tok/s per user, ${file.workload.activeShare * 100}% of users active at once; "chat" = the same with ${file.chatCachedShare * 100}% of the context already in the prefix cache.`,
  );
  const full = lines.length <= OVERVIEW_MAX_LINES;
  let last = '';
  for (const l of lines) {
    if (l.repo !== last) {
      out.push('', `${l.model}  (${l.repo})`);
      last = l.repo;
    }
    const { model, answer } = asAnswer(file, l);
    const c = l.context;
    out.push(`  ${l.version} · ${ktok(c.tokens)} (${c.label})`);
    const limit = outsideLimits(model);
    if (limit !== undefined) para(outsideText(limit), 4);
    para(headline(model, answer, style.ascii), 4);
    if (c.chat) para(`As a chat: ${int(c.chat.guaranteed)} users guaranteed, ${int(c.chat.likely)} likely.`, 4);
    const not = notEnoughText(c.cards);
    // Cards in the table that fit but guarantee nobody count too: say what "not enough" means.
    if (not) para(`Not enough (does not fit, does not start, or nobody guaranteed ${model.workload.minTokS} tok/s): ${not}.`, 4);
    if (full) {
      const fits = c.cards.filter((x) => x.status !== 'no-fit');
      const not = c.cards.filter((x) => x.status === 'no-fit');
      const notes = new Notes();
      if (fits.length > 0) {
        for (const t of table(fits, cardColumns(model.workload, notes, style.ascii), width - 4, style)) out.push(`    ${t}`);
        para(USERS_LEGEND, 4);
      }
      if (not.length > 0) para(`Do not fit ${doNotFit(not)}.`, 4);
      for (const [i, t] of notes.texts.entries()) para(`(${i + 1}) ${t}`, 4);
    }
    if (c.tokens >= 1_000_000 && c.args?.vllm) para(`for 1M with vLLM: ${c.args.vllm.join(' ')}`, 4);
  }
  if (!full) out.push('', `Every card for one model: add its name (e.g. ${COMMAND} report ${((lines[0] as ReportLine).model.split(/[-\s]/)[0] as string).toLowerCase()}).`);
  if (file.missing.length > 0) out.push('', ...wrap(`Versions without enough data: ${file.missing.join(', ')}.`, width));
  out.push(
    '',
    ...wrap(
      'At 1M tokens the memory is computed exactly; the quality beyond the context the lab declares is not guaranteed. Static YaRN in vLLM applies to every length and can make short texts worse.',
      width,
    ),
  );
  const text = out.join('\n');
  return style.ascii ? asciiOnly(text) : text;
}

/** One CSV line per model version × card × context, with the same columns as the command's --csv. */
export function reportCsv(file: ReportFile, lines: readonly ReportLine[], excelEu: boolean): string {
  return csvText(
    lines.flatMap((l) => {
      const m: CsvModel = {
        model: l.repo,
        version: l.versionRepo,
        format: l.context.format.toUpperCase(),
        family: l.context.family,
        engine: file.engine === 'sglang' ? 'SGLang' : 'vLLM',
        engineVersion: file.engineVersion,
        cache: 'fp8',
        workload: workloadOf(file, l),
        contextLabel: l.context.label,
      };
      return l.context.cards.map((c) => [m, { figures: c.figures, status: c.status, ...(c.note ? { note: c.note } : {}) }] as const);
    }),
    excelEu,
  );
}

/** Markdown of the report: per model version and context, the answer and the overview per card. */
export function reportMarkdown(file: ReportFile, lines: readonly ReportLine[]): string {
  const out = [`# Capacity reports (${file.engine === 'vllm' ? 'vLLM' : 'SGLang'} ${file.engineVersion}, fp8 KV cache)`, ''];
  for (const l of lines) {
    const { model, answer } = asAnswer(file, l);
    const notes = new Notes();
    const cols = cardColumns(model.workload, notes, false);
    const limit = outsideLimits(model);
    out.push(`## ${l.model} · ${l.version} · ${ktok(l.context.tokens)} (${l.context.label})`, '', ...(limit !== undefined ? [outsideText(limit), ''] : []), `**${headline(model, answer)}**`, '');
    out.push(...mdTable(l.context.cards, cols), '', `_${USERS_LEGEND}_`);
    if (notes.texts.length > 0) out.push('', ...notes.texts.map((t, i) => `${i + 1}. ${t}`));
    out.push('');
  }
  return `${out.join('\n')}`;
}

/** The figures of a report line's answer, for the JSON export. */
export function reportJsonText(lines: readonly ReportLine[]): string {
  return `${JSON.stringify(lines, null, 1)}\n`;
}


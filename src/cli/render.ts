/**
 * What the capacity command prints and exports:
 *  1. the general answer first (the fewest GPUs the model fits on, and with --users what serves them),
 *     then the scenario in one line and what was read (the weights are never downloaded);
 *  2. one line per catalog card: verdict, fewest GPUs, memory per GPU, margin, users, speed, each number
 *     with its absolute range and its accuracy label (figures.ts); this PC's card is marked;
 *  3. the notes, how to read the results, and with --details or --gpu every number of one layout with its
 *     roofline check.
 * Text fits the terminal (long forms when wide, short ones and fewer columns when narrow; colours and
 * ASCII-only consoles); CSV is standard (",", decimal point, UTF-8) or, with --csv-excel-eu, for Excel in
 * comma-decimal locales (";", decimal comma, UTF-8 with BOM).
 */
import type { MiBRange } from '../model/serving-fit';
import { calibrationFor } from '../planner/calibration';
import type { Workload } from '../planner/capacity';
import { FORMAT_WARNING } from '../planner/hardware';
import type { PlanRow } from '../planner/plan';
import type { Verification } from '../planner/weights-verification';
import { asciiOnly, GB, gb, int, measureLong, measureShort, num, plusMinus, precisionShort, precisionText, SECONDS, secs, TOKS, wordsAndPages, wordsPerSecond, WORDS_PER_PAGE, WORDS_PER_TOKEN, wrap } from './format';
import { DECIDE_MAX_ERROR, SATURATION, gpuCount, gpuFamily, nobodyReason, ktok, layoutShort, layoutText, serversOfFigures, type CardLine, type CardStatus, type Figures, type Measure } from './figures';

export { asciiOnly } from './format';
export { hardwareLabel } from './figures';

/** Where the model's description came from: the network (bytes read), the local cache, or built-in data. */
export type ModelSource = { kind: 'network'; bytes: number; cachedBytes: number } | { kind: 'cache'; bytes: number } | { kind: 'built-in'; date: string };

export interface RenderModel {
  repo: string;
  variant: string;
  /** Weight format and family ("FP8", "MoE"). */
  format: string;
  family: string;
  verification: Verification;
  engine: 'vllm' | 'sglang';
  cache: 'fp8' | 'auto';
  workload: Workload;
  /** The context is the model's own maximum (no --tokens given). */
  contextIsModelMax: boolean;
  modelMaxContext?: number;
  textOnly: boolean;
  hasVision: boolean;
  /** What was read to plan it. */
  source?: ModelSource;
  /** The model is in the compatibility matrix (engine × card checked with sources); false: the engine's own checks only. */
  compatChecked?: boolean;
  /** The longest context the lab documents (an official YaRN factor beyond the config's maximum), from models.json. */
  documentedMaxContext?: number;
  /** --gpu / --vram / --count: the cards and the GPU count asked for (the answer covers only those). */
  selection?: { cards: string[]; count?: number };
}

export interface TextStyle {
  color: boolean;
  ascii: boolean;
  /** Terminal width in characters. */
  columns: number;
}

/** An option of the general answer: the layout and the first one of the other market. */
export interface ChoiceFigures {
  figures: Figures;
  alternative?: Figures;
  /** --users: servers of `figures` (and of `alternative`) that serve them. */
  servers?: number;
  alternativeServers?: number;
  /** --users: users one server serves (plan.ts serverPlan), of `figures` and of `alternative`. */
  usersPerServer?: number;
  likelyUsersPerServer?: number;
  alternativeUsersPerServer?: number;
}

export interface Answer {
  /** One line per card (catalog order; this PC's card marked). */
  cards: CardLine[];
  /** Figures of every layout planned, ranked (CSV, JSON, --all-hw). */
  rows: Figures[];
  /** The fewest GPUs the model fits on (and serves someone). */
  fit?: ChoiceFigures;
  /** Nothing fits: the layout closest to fitting. */
  attempt?: Figures;
  /** Nothing runs, but the memory fits: the engine refuses (the fewest GPUs). */
  blocked?: Figures;
  /** --users: how many, how many at once, and the option that serves them. */
  users?: { total: number; concurrent: number; choice?: ChoiceFigures };
  /** --details / --gpu: one layout, every number (the row for the engine's memory items). */
  details?: { figures: Figures; row?: PlanRow };
  /** --gpu / --all-hw: every layout of the cards asked, as a table. */
  layouts?: Figures[];
}

const MIB = 1024 * 1024;
const GIB = 1024;


/** "1× RTX 3090", "4× H200 SXM (1 model on 4)": from the layout, also for the reports packed with older labels. */
const option = (f: Figures, ascii = false) => {
  const how = layoutText(f, ascii);
  return `${f.count}${ascii ? 'x' : '×'} ${f.gpu}${how ? ` (${how})` : ''}`;
};
/** The model's name without the organisation ("GLM-5.3"). */
const modelName = (m: RenderModel) => `${m.repo.split('/').pop() as string}${m.variant ? ` (${m.variant})` : ''}`;
const cacheText = (m: RenderModel) => (m.cache === 'fp8' ? 'fp8 KV cache' : 'full-precision KV cache (auto)');
const KIND_TAG: Record<Measure['precision']['kind'], string> = { exact: 'exact', measured: 'measured', calibrated: 'calibrated', estimated: 'estimated', imprecise: 'not precise enough to decide' };

/** "10 users guaranteed, 12 likely, at 32 tok/s each (27 to 38, ±16%, estimated)". */
function usersText(f: Figures): string {
  const u = f.users as NonNullable<Figures['users']>;
  const t = f.tokS as Measure;
  return `${int(u.guaranteed)} users guaranteed, ${int(u.likely)} likely, at ${measureLong(t, TOKS, KIND_TAG[t.precision.kind])} each`;
}

/** "200 users (10% active ≈ 20 concurrent requests)". */
export function usersHead(total: number, concurrent: number, activeShare: number): string {
  return `${int(total)} users (${num(activeShare * 100, 0)}% active ≈ ${int(concurrent)} concurrent ${concurrent === 1 ? 'request' : 'requests'})`;
}

/** The technical answer in one or two sentences. */
export function headline(m: RenderModel, a: Answer, ascii = false): string {
  const what = `${modelName(m)} at ${ktok(m.workload.maxContext)} tokens, ${cacheText(m)}`;
  // An alternative only when both speeds can be relied on (I-B: a caveat like "far below the estimate" leaves it out).
  const usable = (c: ChoiceFigures) => (c.alternative && !unreliableSpeed(c.figures) && !unreliableSpeed(c.alternative) ? c.alternative : undefined);
  const or = (c: ChoiceFigures) => `${option(c.figures, ascii)}${usable(c) ? `, or ${option(usable(c) as Figures, ascii)}` : ''}`;
  if (!a.fit && a.blocked) {
    const b = a.blocked;
    return `${what}: fits in memory from ${option(b, ascii)}, but the engine does not start (${b.compat.notes[0] as string}); no layout of the catalog runs it.`;
  }
  if (!a.fit) {
    const t = a.attempt as Figures;
    const one = oneServer(t);
    return `${what}: does not fit in one server ${scopeText(m)} (with ${option(one, ascii)} it needs ${gb(t.memory.value)} GiB per GPU of ${t.cardGiB} GiB${one.count < t.count ? `; ${gpuCount(t.count)} are ${serversOfFigures(t)} such servers, each with its own copy` : ''}).`;
  }
  const f = a.fit.figures;
  const head0 =
    f.users !== undefined && f.users.guaranteed < 1
      ? `no layout serves anyone (${nobodyReason(f.requests?.limitedBy, `${m.workload.minTokS} tok/s`)}); the closest is`
      : f.verdict === 'at-limit'
        ? 'at the limit already with'
        : f.users === undefined
          ? 'fits in memory from'
          : 'serves users from';
  const parts = [`${what}: ${head0} ${or(a.fit)}.`];
  if (a.users) {
    const u = a.users;
    const head = `For ${usersHead(u.total, u.concurrent, m.workload.activeShare)}`;
    const n = u.choice?.servers ?? 1;
    const times = (k: number, x: Figures) => (k > 1 ? `${int(k)} servers of ${option(oneServer(x), ascii)}` : option(x, ascii));
    const c = u.choice;
    if (c) {
      const alt = usable(c) ? `, or ${times(c.alternativeServers ?? 1, usable(c) as Figures)}` : '';
      const served =
        n > 1 && c.usersPerServer !== undefined
          ? `${int(c.usersPerServer)} users guaranteed${c.likelyUsersPerServer !== undefined ? `, ${int(c.likelyUsersPerServer)} likely` : ''} per server, at ${measureLong(c.figures.tokS as Measure, TOKS, KIND_TAG[(c.figures.tokS as Measure).precision.kind])} each`
          : `${usersText(c.figures)}${n > 1 ? ' per server' : ''}`;
      parts.push(`${head} you need ${times(n, c.figures)}${alt}: ${served}.`);
    } else parts.push(`${head}: none — ${nobodyCause(m, a, ascii)}.`);
  } else if (f.users) {
    // A layout with users has every speed figure.
    const read = `reading ${ktok(m.workload.maxContext)} tokens: ${measureLong(f.read as Measure, SECONDS)}`;
    if (f.users.guaranteed < 1)
      parts.push(`With ${option(f, ascii)}, ${nobodyReason(f.requests?.limitedBy, `${m.workload.minTokS} tok/s`)} (${int(f.users.likely)} likely; one request alone writes ${measureLong(f.single as Measure, TOKS)}); ${read}.`);
    else parts.push(`With ${option(f, ascii)}: ${usersText(f)}; ${read}.`);
  } else parts.push(`With ${option(f, ascii)} the speed cannot be estimated (the card's bandwidth or TFLOPS are not published).`);
  return parts.join(' ');
}

/** One server of a layout spread over several: 16× H200 (2 copies × 8) → 8× H200 (1 model on 8). */
function oneServer(f: Figures): Figures {
  const n = serversOfFigures(f);
  return n === 1 ? f : { ...f, count: f.count / n, replicas: f.replicas / n, servers: 1 };
}

/** "on any card of the catalog", "on the selected card (H100 SXM5 80GB, 1 GPU)". */
export function scopeText(m: RenderModel): string {
  const s = m.selection;
  if (!s) return 'on any card of the catalog';
  if (s.cards.length === 0) return `on any card of the catalog with ${gpuCount(s.count as number)}`;
  const what = `${s.cards.join(', ')}${s.count !== undefined ? `, ${gpuCount(s.count)}` : ''}`;
  return `on the selected ${s.cards.length === 1 ? 'card' : 'cards'} (${what})`;
}

/** The caveat of a speed that is estimated but not reliable ("far below the estimate"), not of one that cannot be estimated. */
const unreliableSpeed = (f: Figures) => (f.users !== undefined ? f.speedNote : undefined);

/** "15 words per second (20 tokens/s)": the reader's unit first, the engine's in brackets. */
export const wordsAndTokS = (tokS: number) => `${wordsPerSecond(tokS)} words per second (${int(tokS)} tokens/s)`;

/** "0.5 s", "10 s": a wait limit as the user wrote it (--max-wait, 0.05 s at least), never rounded to whole seconds. */
export const waitLimitText = (ms: number) => `${(ms / 1000).toLocaleString('en-US', { maximumFractionDigits: 2 })} s`;

/**
 * Why no layout serves anyone at the minimum speed, with the figures behind it: the memory (no full-length request
 * fits with its safety margin), the wait for the first words, or the speed under load (and how much of the GPU
 * time reading other users' prompts takes); where it was checked (the whole catalog or the cards selected), and
 * the layout that comes closest. Starts with "too slow…" or "at N tokens…"; "none — " goes in front.
 */
export function nobodyCause(m: RenderModel, a: Pick<Answer, 'rows' | 'fit'>, ascii = false): string {
  const w = m.workload;
  const k = ktok(w.maxContext);
  const fitted = a.rows.filter((r) => r.verdict !== 'no-fit' && r.verdict !== 'no-start' && r.users && r.tokS);
  if (fitted.length === 0) return `at ${k} tokens ${scopeText(m)} the speed cannot be estimated (the card's figures are not published), so no user count can be given`;
  if (fitted.every((r) => r.requests?.limitedBy === 'memory'))
    return `at ${k} tokens ${scopeText(m)}, the memory left after the model holds no full-length request with its safety margin, so no user is guaranteed, whatever the number of servers; use more GPUs per server or shorter requests`;
  // The closest layout is the one the answer chose (figures.ts closestRow): the headline, the report and each card's
  // line name the same one. Without a speed of its own, the first of the ranked rows that has one.
  const chosen = a.fit?.figures;
  const best = chosen?.users && chosen.tokS ? chosen : (fitted[0] as Figures);
  const t = best.tokS as Measure;
  const likely = (best.users as NonNullable<Figures['users']>).likely;
  const reading = best.readingShare ?? 0;
  // Writing (the answer, tokens/s per user) and reading (the prompt, time to the first words) are told apart.
  const wait = best.requests?.limitedBy === 'wait';
  const fresh = ktok(Math.round(w.typicalContext * (1 - w.cachedShare)));
  // Nobody served by the wait: even one request alone may wait longer than allowed for its first words (the guarantee
  // is the slow end); a wait limit that grows with the context never stops a request alone, so it is the one set.
  const first = best.wait as Measure;
  const why = wait
    ? `even one request alone may wait up to ${secs(first.max)} s for its first words (about ${secs(first.value)} s), more than the ${waitLimitText(w.maxTtftMs)} allowed (--max-wait): reading its ${fresh}-token prompt takes that long`
    : `no user is guaranteed a writing speed of ${wordsAndTokS(w.minTokS)} for the answer (the guarantee is the low end of the estimate)`;
  const writing = `writing about ${wordsPerSecond(t.value)} words per second per user under load (${wordsPerSecond(t.min)} to ${wordsPerSecond(t.max)}; ${int(t.value)} tokens/s)${likely > 0 ? `, ${int(likely)} users likely` : ''}, one request alone ${wordsAndTokS((best.single as Measure).value)}`;
  const read = `${reading >= 0.5 ? `reading other users' ${fresh}-token prompts takes ${int(Math.round(reading * 100))}% of the GPU time, and ` : ''}a full ${k}-token request alone is read in about ${secs((best.read as Measure).value)} s`;
  const help = reading >= 0.5 && w.cachedShare === 0 ? '. Shorter requests, or a chat that keeps its history in the prompt cache (--prompt-cache 80), serve more' : '';
  return `${wait ? `the first words come too late at ${k} tokens` : `too slow at ${k} tokens`} ${scopeText(m)}: ${why}, whatever the number of servers. Closest: ${serverText(oneServer(best), ascii)}, ${writing}; ${read}${help}`;
}

/**
 * The longest context the model is made for: its own maximum, or the longer one its lab documents (with an official
 * YaRN factor; labs write "1M" loosely, so 5% above it still counts). Beyond it nothing is recommended.
 */
export function outsideLimits(m: RenderModel): number | undefined {
  const own = m.modelMaxContext;
  if (own === undefined || m.workload.maxContext <= own) return undefined;
  const doc = m.documentedMaxContext;
  if (doc !== undefined && m.workload.maxContext <= doc * 1.05) return undefined;
  return Math.max(own, doc ?? 0);
}

/** The workload in one line: what every request brings (the default is heavy: all new text). */
export function scenario(w: Workload): string {
  const fresh = Math.round(w.typicalContext * (1 - w.cachedShare));
  const target = `at least ${w.minTokS} tok/s per user`;
  if (w.cachedShare > 0)
    return `Scenario "chat": each request has ${ktok(w.typicalContext)} tokens of context, ${num(w.cachedShare * 100, 0)}% already in the prefix cache (${ktok(fresh)} new tokens to read), and generates ${ktok(w.outputTokens)}; ${target}.`;
  return `Scenario "documents": each request brings ${ktok(w.typicalContext)} new tokens to read and generates ${ktok(w.outputTokens)}; ${target}. For a chat (the conversation stays in the cache) use --prompt-cache 80.`;
}

/** A context longer than the model's own maximum: the engine needs a RoPE override (YaRN), quality not guaranteed. */
export function beyondNative(m: RenderModel): string | undefined {
  const max = m.modelMaxContext;
  if (max === undefined || m.workload.maxContext <= max) return undefined;
  return `Note: ${ktok(m.workload.maxContext)} tokens is beyond the model's own maximum (${ktok(max)}): the engine refuses it without a RoPE override (YaRN, e.g. vLLM --hf-overrides), and the quality beyond the context the lab declares is not guaranteed.`;
}

/** "Read config + tensor headers (412 KB) from Hugging Face — weights not downloaded". */
export function sourceLine(s: ModelSource): string {
  const kb = (b: number) => (b >= 1024 * 1024 ? `${num(b / 1024 / 1024)} MB` : `${int(Math.max(1, b / 1024))} KB`);
  if (s.kind === 'built-in') return `Model description from built-in data (${s.date}) — weights not downloaded; --refresh reads it again from Hugging Face.`;
  if (s.kind === 'cache') return `Read config + tensor headers (${kb(s.bytes)}) from the local cache — weights not downloaded.`;
  return `Read config + tensor headers (${kb(s.bytes)}) from Hugging Face${s.cachedBytes > 0 ? `, ${kb(s.cachedBytes)} from the local cache` : ''} — weights not downloaded.`;
}

/** "Qwen/Qwen3-8B · BF16 · dense · vLLM 0.31.0 · fp8 KV cache · context 32k tokens (model maximum 40k)". */
export function modelLine(m: RenderModel): string {
  const cal = calibrationFor(m.engine);
  const max = m.contextIsModelMax ? ' (the model maximum)' : m.modelMaxContext ? ` (model maximum ${ktok(m.modelMaxContext)})` : '';
  return `${m.repo}${m.variant ? ` (${m.variant})` : ''} · ${m.format} · ${m.family} · ${m.engine === 'vllm' ? 'vLLM' : 'SGLang'} ${cal.engineVersion} · ${cacheText(m)} · context ${ktok(m.workload.maxContext)} tokens${max}`;
}

/** The verdicts in narrow tables. */
const SHORT_STATUS: Record<CardStatus, string> = { fits: 'fits', 'no-speed': 'speed unknown', slow: 'too slow', short: 'too slow', 'at-limit': 'at the limit', 'no-start': "won't start", 'no-fit': 'does not fit' };
export const STATUS: Record<CardStatus, string> = { fits: 'fits', 'no-speed': 'fits, speed unknown', slow: 'fits, too slow', short: 'too slow for anyone', 'at-limit': 'at the limit', 'no-start': 'engine does not start', 'no-fit': 'does not fit' };
const COLORS = { green: '\u001b[32m', yellow: '\u001b[33m', red: '\u001b[31m', bold: '\u001b[1m', reset: '\u001b[0m' };
const STATUS_COLOR: Record<CardStatus, keyof typeof COLORS> = { fits: 'green', 'no-speed': 'yellow', slow: 'yellow', short: 'yellow', 'at-limit': 'yellow', 'no-start': 'red', 'no-fit': 'red' };

/** A table column: the long form, the short one for narrow terminals, and when to drop it. */
export interface Column<T> {
  head: string;
  long: (item: T) => string;
  short?: (item: T) => string;
  /** The last resort before dropping columns (a value without its range, which --details and the CSV keep). */
  tiny?: (item: T) => string;
  /** Dropped first when the terminal is narrow (higher = earlier; 0 = never). */
  drop: number;
  /** Order in which long forms give way to short ones (lower = earlier; default 9). */
  shorten?: number;
  right?: boolean;
  /** ANSI colour of a cell. */
  color?: (item: T) => keyof typeof COLORS | undefined;
}

/**
 * A text table that fits `width`: long forms first, then the short ones (by `shorten`), then the tiny ones,
 * then columns dropped by priority. Colours are added after padding, so they never shift the columns.
 */
export function table<T>(items: readonly T[], cols: readonly Column<T>[], width: number, style: { color?: boolean; ascii?: boolean } = {}): string[] {
  let use = [...cols];
  const shortened = new Set<Column<T>>();
  const tinied = new Set<Column<T>>();
  // ASCII before measuring: "±" becomes "+-", one character more.
  const plainText = (s: string) => (style.ascii ? asciiOnly(s) : s);
  const color = style.color === true;
  const build = () => {
    const form = (c: Column<T>, x: T) => (tinied.has(c) && c.tiny ? c.tiny(x) : shortened.has(c) && c.short ? c.short(x) : c.long(x));
    const cells = items.map((x) => use.map((c) => plainText(form(c, x))));
    // Headers may take two lines ("Tok/s\nper user"): the cells set the width, not a long header.
    const heads = use.map((c) => [...plainText(c.head).split('\n'), ''].slice(0, 2) as [string, string]);
    const two = heads.some((h) => h[1] !== '');
    const widths = use.map((_c, j) => Math.max(...(heads[j] as string[]).map((h) => h.length), ...cells.map((row) => (row[j] as string).length)));
    const pad = (v: string, j: number) => ((use[j] as Column<T>).right ? v.padStart(widths[j] as number) : v.padEnd(widths[j] as number));
    const line = (vals: string[], item?: T) =>
      vals
        .map((v, j) => {
          const code = item !== undefined && color ? (use[j] as Column<T>).color?.(item) : undefined;
          return code ? `${COLORS[code]}${pad(v, j)}${COLORS.reset}` : pad(v, j);
        })
        .join('  ')
        .trimEnd();
    const headLines = [heads.map((h) => h[0]), ...(two ? [heads.map((h) => h[1])] : [])].map((vals) => line(vals));
    const body = cells.map((vals) => line(vals));
    return { plain: [...headLines, ...body], out: [...headLines, ...cells.map((vals, i) => line(vals, items[i]))] };
  };
  let t = build();
  const tooWide = () => Math.max(...t.plain.map((l) => l.length)) > width;
  while (tooWide()) {
    const order = (c: Column<T>) => c.shorten ?? 9;
    const next = use.filter((c) => c.short && !shortened.has(c)).sort((x, y) => order(x) - order(y))[0];
    const tiny = use.find((c) => c.tiny && !tinied.has(c));
    if (next) shortened.add(next);
    else if (tiny) tinied.add(tiny);
    else if (use.some((c) => c.drop > 0)) {
      const worst = Math.max(...use.map((c) => c.drop));
      use = use.filter((c) => c.drop !== worst);
    } else break;
    t = build();
  }
  return t.out;
}

/** "4 (1 model on 4)", "8 (2 copies × 4)", "2 (split in 2 stages)", "1"; short: "4 (1×4)", "8 (2×4)", "2 (2 stages)". */
export function splitText(f: Pick<Figures, 'count' | 'tp' | 'pp' | 'replicas'>, short = false): string {
  const how = short ? layoutShort(f) : layoutText(f);
  return how ? `${f.count} (${how})` : String(f.count);
}

/** The sentences of a note: after a full stop and a space, or after ")" before a capital ("v0.31.0 the" stays whole). */
const sentences = (t: string) => t.split(/(?<=\.)\s+|(?<=\))\s+(?=[A-Z])/);

/**
 * Footnotes, numbered in order of use; cells carry "(1)". A note already printed gets its number again; a sentence already printed
 * in an earlier note (the matrix repeats its general caveats per card) is left out of the later one.
 */
export class Notes {
  readonly texts: string[] = [];
  private readonly ids = new Map<string, number>();
  mark(texts: readonly (string | undefined)[]): string {
    const ids = [...new Set(texts.filter((t): t is string => !!t))].map((t) => {
      const known = this.ids.get(t);
      if (known !== undefined) return known;
      const seen = new Set(this.texts.flatMap(sentences));
      const kept = sentences(t).filter((x, i, all) => !seen.has(x) && all.indexOf(x) === i);
      // Every sentence already printed: the note that has its first one.
      const id = kept.length > 0 ? this.texts.push(kept.join(' ')) : this.texts.findIndex((x) => sentences(x).includes(sentences(t)[0] as string)) + 1;
      this.ids.set(t, id);
      return id;
    });
    return ids.length > 0 ? ` (${[...new Set(ids)].join(',')})` : '';
  }
}

/** The notes of a card line: why it has its verdict, the engine's caveats, speed warnings. */
export function lineNotes(l: CardLine): string[] {
  const f = l.figures;
  if (l.status === 'no-fit') return [];
  const compat = f.compat.status === 'warnings' ? f.compat.notes : [];
  // A speed estimated on pipeline stages: no run measured them (calibration.ts PIPELINE_WIDENING).
  const pipeline = f.pp > 1 && f.tokS ? PIPELINE_NOTE : undefined;
  // "not enough for N users" says it already.
  return [l.status === 'short' ? undefined : l.note, ...compat, f.speedNote, pipeline, f.saturation, f.roofline?.warning].filter((t): t is string => !!t);
}

/** The note of a layout split in pipeline stages that has a speed. */
export const PIPELINE_NOTE = 'pipeline speed not measured: no run with pipeline stages was measured, so the speed and user ranges are widened by half';

/** "71.4 GiB (69.8 to 73.1 GiB, ±2.3%) of 79.6 GiB". */
export const memoryText = (f: Figures) => `${measureLong(f.memory, GB)} of ${f.cardGiB} GiB`;

/** The overview's columns (per card). */
export function cardColumns(w: Workload, notes: Notes, ascii: boolean, users?: { total: number }): Column<CardLine>[] {
  const dash = ascii ? '-' : '–';
  const speed = (l: CardLine) => l.status !== 'no-fit' && l.status !== 'no-start' && l.figures.tokS !== undefined;
  const fits = (l: CardLine) => l.status !== 'no-fit';
  const marks = new Map<CardLine, string>();
  const mark = (l: CardLine) => marks.get(l) ?? (marks.set(l, notes.mark(lineNotes(l))).get(l) as string);
  const status = (l: CardLine, long: boolean) => {
    const slowToo = l.status === 'at-limit' && l.figures.users !== undefined && l.figures.users.guaranteed < 1;
    // Serving nobody, the binding limit in the verdict: memory, the first-token wait or the writing speed.
    const by = l.figures.requests?.limitedBy;
    const nobody = slowToo || l.status === 'slow' || l.status === 'short';
    const cause = by === 'memory' ? (long ? 'no room for a request' : 'no room') : by === 'wait' ? (long ? 'first words too late' : 'wait too long') : undefined;
    const text =
      l.status === 'no-fit'
        ? `does not fit${long ? ` even with ${l.serverGpus ?? l.figures.count}` : ''}`
        : nobody && cause
          ? `${slowToo ? (long ? 'at the limit, ' : 'limit, ') : long ? 'fits, ' : ''}${cause}`
          : slowToo
            ? long
              ? 'at the limit, too slow'
              : 'limit, slow'
            : long
              ? STATUS[l.status]
              : SHORT_STATUS[l.status];
    return `${text}${mark(l)}`;
  };
  const u = (l: CardLine, k: 'guaranteed' | 'likely') => (speed(l) ? int(users && l.usersPerServer ? l.usersPerServer[k] : l.figures.users![k]) : fits(l) ? 'n/a' : '');
  return [
    { head: 'Card', long: (l) => `${l.gpu}${l.local ? ' (this PC)' : ''}`, drop: 0 },
    { head: 'Verdict', long: (l) => status(l, true), short: (l) => status(l, false), drop: 0, shorten: 6, color: (l) => STATUS_COLOR[l.status] },
    { head: 'GPUs', long: (l) => (fits(l) ? splitText(l.figures) : '-'), short: (l) => (fits(l) ? splitText(l.figures, true) : '-'), drop: 0, shorten: 0 },
    ...(users ? [{ head: 'Servers\nneeded', long: (l: CardLine) => (l.serversNeeded !== undefined ? int(l.serversNeeded) : ''), drop: 0, right: true }] : []),
    {
      head: 'Memory per GPU\n(GiB)',
      long: (l) => `${fits(l) ? '' : `with ${l.figures.count}: `}${memoryText(l.figures)}`,
      short: (l) => `${fits(l) ? '' : `with ${l.figures.count}: `}${measureShort(l.figures.memory, GB, dash)} of ${l.figures.cardGiB}`,
      tiny: (l) => `${fits(l) ? '' : `with ${l.figures.count}: `}${gb(l.figures.memory.value)} of ${l.figures.cardGiB}`,
      drop: 0,
      shorten: 1,
    },
    { head: 'Margin (GiB,\nworst case)', long: (l) => (fits(l) ? gb(l.figures.margin.min) : ''), drop: 2, right: true },
    { head: 'Users\n(guaranteed)', long: (l) => u(l, 'guaranteed'), drop: 0, right: true },
    { head: 'Users\n(likely)', long: (l) => u(l, 'likely'), drop: 0, right: true },
    { head: 'Tok/s\nper user', long: (l) => (speed(l) ? measureLong(l.figures.tokS!, TOKS) : ''), short: (l) => (speed(l) ? measureShort(l.figures.tokS!, TOKS, dash) : ''), drop: 3, shorten: 2 },
    {
      head: `Read ${ktok(w.maxContext)}\n(s)`,
      long: (l) => (speed(l) ? measureLong(l.figures.read!, SECONDS) : ''),
      short: (l) => (speed(l) ? measureShort(l.figures.read!, SECONDS, dash) : ''),
      drop: 4,
      shorten: 3,
    },
    { head: 'Accuracy\n(users, tok/s)', long: (l) => accuracy(l.figures, precisionText), short: (l) => accuracy(l.figures, precisionShort), drop: 1, shorten: 4 },
  ];
}

/** The accuracy of users and speed: the wider of the two; n/a when nobody is guaranteed (no error on 0). */
export function accuracy(f: Figures, text: (p: Measure['precision']) => string): string {
  if (!f.tokS || !f.users) return '';
  if (f.users.guaranteed < 1) return 'n/a';
  return text(f.users.precision.errorPct > f.tokS.precision.errorPct ? f.users.precision : f.tokS.precision);
}

const VERDICT: Record<Figures['verdict'], string> = { fits: 'fits', 'at-limit': 'at the limit', 'no-fit': 'does not fit', 'no-start': 'engine does not start' };

/** The per-layout columns (--gpu, --all-hw); with `notes`, a layout the engine refuses gets its reason as a footnote. */
export function layoutColumns(w: Workload, ascii: boolean, notes?: Notes): Column<Figures>[] {
  const dash = ascii ? '-' : '–';
  return [
    { head: 'Layout', long: (f) => f.hardware, drop: 0 },
    {
      head: 'Verdict',
      // The reason (and its workaround) stays visible even when the Engine column is dropped for width.
      long: (f) => (f.verdict === 'fits' && f.users && f.users.guaranteed < 1 ? 'fits, too slow' : `${VERDICT[f.verdict]}${f.verdict === 'no-start' && notes ? notes.mark([f.compat.notes[0]]) : ''}`),
      drop: 0,
    },
    { head: 'Memory per GPU\n(GiB)', long: memoryText, short: (f) => `${measureShort(f.memory, GB, dash)} of ${f.cardGiB}`, drop: 0, shorten: 1 },
    { head: 'Margin (GiB,\nworst case)', long: (f) => gb(f.margin.min), drop: 2, right: true },
    { head: 'Users\n(guaranteed)', long: (f) => (f.users ? int(f.users.guaranteed) : ''), drop: 0, right: true },
    { head: 'Users\n(likely)', long: (f) => (f.users ? int(f.users.likely) : ''), drop: 0, right: true },
    { head: 'Tok/s\nper user', long: (f) => (f.tokS ? measureLong(f.tokS, TOKS) : ''), short: (f) => (f.tokS ? measureShort(f.tokS, TOKS, dash) : ''), drop: 3, shorten: 2 },
    { head: `Read ${ktok(w.maxContext)}\n(s)`, long: (f) => (f.read ? measureLong(f.read, SECONDS) : ''), short: (f) => (f.read ? measureShort(f.read, SECONDS, dash) : ''), drop: 4, shorten: 3 },
    { head: 'Accuracy\n(users, tok/s)', long: (f) => accuracy(f, precisionText), short: (f) => accuracy(f, precisionShort), drop: 1, shorten: 4 },
    { head: 'Engine', long: (f) => (f.compat.status === 'does-not-start' ? `does not start: ${f.compat.notes[0] as string}` : f.compat.status === 'warnings' ? 'starts, with warnings' : 'starts'), drop: 5 },
  ];
}

/** "guaranteed = …; likely = …" (under every table with users). */
export const USERS_LEGEND = 'guaranteed = holds even if the real speed is at the low end of the estimate; likely = at the central estimate.';
/** What counts as a user (with --users). */
export const AGENTS_NOTE =
  'Users include AI agents. Splitting work across parallel agents or subagents gets it done sooner, but each one is a concurrent client: 5 subagents count as 5 users. Agents are usually active all the time and send long contexts, so raise --active and --tokens for agent workloads.';

/** Said when the model is not in the compatibility matrix. */
export const COMPAT_UNCHECKED =
  'Engine compatibility: this model is not in the compatibility matrix (recent open models only): only the engine’s own checks (tensor-parallel splits, kernels per architecture) are applied.';

/** How to read the overview and the accuracy labels (a few lines); `shown`: the rows whose speed bases to name. */
export function legend(m: RenderModel, withUsers = false, shown: readonly (Figures | undefined)[] = []): string[] {
  const w = m.workload;
  const cal = calibrationFor(m.engine);
  const bases = shownBases(shown);
  return [
    `Memory per GPU with one request of ${ktok(w.maxContext)} tokens (${wordsAndPages(w.maxContext)}): the model's weights and its conversation memory (KV cache) are computed exactly, the engine's fixed costs are estimated. Margin = room left for more conversations after that request. Card memory = what CUDA reports (the engine's usable memory; nvidia-smi shows a little more).`,
    `Users: total = concurrent requests ÷ ${num(w.activeShare * 100, 0)}% active at once (--active); each gets ≥${w.minTokS} tok/s and the first token within ${waitLimitText(w.maxTtftMs)}${w.ttftScales ? ' (or, if longer, twice the wait for the first words of one typical request alone: half the context, less what the prefix cache holds)' : ''}.`,
    ...(withUsers ? [AGENTS_NOTE] : []),
    `Accuracy: exact = computed; measured = this model on this card; calibrated = fitted on measurements of this card or of the same model type (N points); estimated = calibrated on other cards, not verified, or extrapolated; not precise enough to decide = ±${DECIDE_MAX_ERROR}% or more, or a case not covered.`,
    ...(bases.length > 0 ? [`Speed basis (${m.engine === 'vllm' ? 'vLLM' : 'SGLang'} ${cal.engineVersion}) of the cards shown: ${bases.join('; ')}.`] : []),
  ];
}

/** A detail line: label, value, and the accuracy after it (or below it when the terminal is narrow). */
function detailLine(label: string, value: string, note: string, width: number, ascii: boolean): string[] {
  const head = `  ${(ascii ? asciiOnly(label) : label).padEnd(26)}${value}`;
  if (!note && head.length <= width) return [head];
  if (note && head.length + 3 + note.length <= width) return [`${head}   ${note}`];
  // Too long: the value and then the note wrapped under the value column.
  const indent = ' '.repeat(28);
  const under = (text: string) => wrap(text, Math.max(20, width - 28)).map((l) => `${indent}${l}`);
  const [first, ...rest] = under(value);
  return [`  ${(ascii ? asciiOnly(label) : label).padEnd(26)}${(first as string).slice(28)}`, ...rest, ...(note ? under(note) : [])];
}

const pct = (share: number) => `${Math.round(share * 100)}%`;
const LIMITED = { memory: 'memory', speed: 'speed', wait: 'first-token wait', none: 'none' } as const;

/** --details / --gpu: every number of one layout with its accuracy, and the roofline check. */
export function detailLines(m: RenderModel, f: Figures, width: number, ascii = false): string[] {
  const w = m.workload;
  const p = (x: Measure) => (x.precision.kind === 'exact' ? `exact (${[x.precision.basis, x.precision.reason].filter(Boolean).join('; ')})` : precisionText(x.precision));
  const detail = (label: string, value: string, note: string, wide: number) => detailLine(label, value, note, wide, ascii);
  const out = [`Details — ${option(f)}, one request of ${ktok(w.maxContext)} tokens:`];
  out.push(...detail('memory per GPU', memoryText(f), p(f.memory), width));
  out.push(...detail('  weights', measureLong(f.weights, GB), p(f.weights), width));
  out.push(...detail('  KV cache of one request', measureLong(f.kv, GB), p(f.kv), width));
  out.push(...detail('  fixed costs', measureLong(f.fixed, GB), p(f.fixed), width));
  out.push(...detail('held by the engine', `${measureLong(f.engineGiB, GB)} of ${f.cardGiB} GiB`, f.engineGiB.precision.basis, width));
  out.push(...detail('margin', measureLong(f.margin, GB), '', width));
  out.push(...detail('engine', f.compat.status === 'ok' ? 'starts' : f.compat.status === 'warnings' ? `starts, with warnings: ${f.compat.notes.join('; ')}` : `does not start: ${f.compat.notes[0] as string}`, '', width));
  if (!f.users) {
    out.push(...detail('speed', 'not estimable', f.speedNote ?? 'the card’s bandwidth or TFLOPS are not published', width));
    return out;
  }
  // A layout with users has every speed figure.
  const { tokS, single, read, wait } = f as Required<Figures>;
  const r = f.roofline as NonNullable<Figures['roofline']>;
  const req = f.requests as NonNullable<Figures['requests']>;
  const n = (k: number, one: string, many: string) => `${int(k)} ${k === 1 ? one : many}`;
  out.push(...detail('concurrent requests', `${n(req.guaranteed, 'guaranteed', 'guaranteed')}, ${int(req.likely)} likely (limited by ${LIMITED[req.limitedBy]}; the memory holds ${int(req.memory)} at ${ktok(w.typicalContext)})`, '', width));
  out.push(...detail('total users', `${int(f.users.guaranteed)} guaranteed, ${int(f.users.likely)} likely (at most ${int(f.users.memoryMax)} by memory)`, precisionText(f.users.precision), width));
  out.push(...detail('writing speed per user', `${measureLong(tokS, TOKS)} with ${n(Math.max(1, req.guaranteed), 'request', 'requests')} at once, including reading other users' prompts`, p(tokS), width));
  out.push(...detail('one request alone', `${measureLong(single, TOKS)} = ${pct(r.tokSShare)} of the physical limit (${int(r.tokSLimit)} tok/s at the datasheet bandwidth)`, p(single), width));
  out.push(...detail(`reading ${ktok(w.maxContext)} tokens`, `${measureLong(read, SECONDS)}; the physical limit (${SECONDS.fmt(r.readLimitS)} s at the ${r.tflopsDerived ? 'derived' : 'datasheet'} TFLOPS) = ${pct(r.readShare)} of it`, p(read), width));
  out.push(...detail('first token under load', measureLong(wait, SECONDS), p(wait), width));
  out.push(...detail('context validity', f.validity, '', width));
  if (r.warning) out.push(...detail('roofline check', r.warning, '', width));
  if (f.saturation) out.push(...detail('saturation', f.saturation, '', width));
  return out;
}

/** --details: the memory of one GPU as the engine counts it, item by item (worst … best case). */
export function memoryDetails(m: RenderModel, r: PlanRow, ascii = false): string[] {
  const dash = ascii ? '-' : '–';
  const w = r.memory.worst;
  const b = r.memory.best;
  const g = (x: MiBRange) => (gb(x.min / GIB) === gb(x.max / GIB) ? `${gb(x.min / GIB)} GiB` : `${gb(x.min / GIB)}${dash}${gb(x.max / GIB)} GiB`);
  const f = w.fixed;
  const wt = r.weights;
  const vllm = m.engine === 'vllm';
  const row = (label: string, value: string, note = '') => `  ${label.padEnd(36)}${value}${note ? `  (${note})` : ''}`;
  const parts = [
    wt.replicatedBytes >= 0.05 * GIB * MIB ? `replicated on every GPU ${gb(wt.replicatedBytes / MIB / GIB)} GiB` : '',
    wt.visionBytes >= 0.05 * GIB * MIB ? `vision encoder ${gb(wt.visionBytes / MIB / GIB)} GiB` : '',
    wt.ropeBytes >= 0.05 * GIB * MIB ? `RoPE cache ${gb(wt.ropeBytes / MIB / GIB)} GiB` : '',
  ].filter(Boolean);
  const out = [
    `Memory as the engine counts it (worst${dash}best case):`,
    row('memory visible to CUDA', g({ min: w.totalMiB, max: b.totalMiB })),
    row("the engine's share", g({ min: w.shareMiB.min, max: b.shareMiB.max }), vllm ? `--gpu-memory-utilization ${w.util}` : `--mem-fraction-static ${w.util}`),
    row('weights', g({ min: w.weightsMiB, max: w.weightsMiB }), `sum of the loaded tensors${parts.length ? `; of which ${parts.join(', ')}` : ''}`),
  ];
  if (f.weightsExtra) out.push(row('weights added by the engine', g(f.weightsExtra), 'MLA copies or the measured gap of the format'));
  out.push(row('CUDA context', g(f.cudaContext), vllm ? 'outside the share' : 'before the share'));
  out.push(row('workspaces (cuBLAS, attention)', g(f.workspaces)));
  out.push(row('activations (profile peak)', g(f.activations)));
  if (f.vision.max > 0) out.push(row('vision encoder profile', g(f.vision)));
  out.push(row('CUDA graphs', g(f.cudaGraphs)));
  if (f.nccl.max > 0) out.push(row('NCCL (GPU-to-GPU communication)', g(f.nccl)));
  out.push(
    row('fixed costs in total', g(f.total)),
    row('room for the KV cache', g({ min: w.kvRoomMiB.min, max: b.kvRoomMiB.max })),
    row(`KV cache of one ${ktok(m.workload.maxContext)} request`, g({ min: w.kvPerSeqMiB, max: w.kvPerSeqMiB })),
    row('requests at the full context', w.concurrency.min === w.concurrency.max ? String(Math.floor(w.concurrency.min)) : `${Math.floor(w.concurrency.min)}${dash}${Math.floor(w.concurrency.max)}`),
    row('margin after one request', g(r.marginMiB)),
  );
  return out;
}

/** The weights' accuracy in one line (for the notes). */
export function weightsNote(m: RenderModel): string {
  const v = m.verification;
  if (v.status === 'verified') return `Weights: exact (sum of the loaded tensors; ${m.format} × ${m.family} verified on vLLM measurements).`;
  if (v.status === 'gap') return `Weights: exact as the sum of the tensors; for ${m.format} × ${m.family} vLLM measured up to +${num(v.gapShare * 100)}% more${m.engine === 'sglang' ? ' (no SGLang measurement: the same gap is assumed)' : ''}, counted in the fixed costs.`;
  return `Weights: exact as the sum of the tensors; ${m.format} × ${m.family} not yet verified by a measurement: ${num(v.gapShare * 100, 0)}% margin in the fixed costs.`;
}

/** The whole text report. */
export function renderText(m: RenderModel, a: Answer, style: TextStyle): string {
  const width = Math.max(60, style.columns);
  const lines: string[] = [];
  const bold = (s: string) => (style.color ? `${COLORS.bold}${s}${COLORS.reset}` : s);
  const para = (text: string) => lines.push(...wrap(text, width));
  lines.push(bold('Summary'));
  for (const s of summaryLines(m, a, style.ascii)) lines.push(...wrap(s, width - 2, 4).map((l) => `  ${l}`));
  lines.push('', bold('Technical detail'));
  for (const l of wrap(headline(m, a, style.ascii), width)) lines.push(l);
  para(scenario(m.workload));
  const beyond = beyondNative(m);
  if (beyond) para(beyond);
  para(modelLine(m));
  if (m.source) para(sourceLine(m.source));
  lines.push('');
  const notes = new Notes();
  const users = a.users ? { total: a.users.total } : undefined;
  const nobody = a.fit?.figures.users !== undefined && a.fit.figures.users.guaranteed < 1;
  para(
    a.users
      ? a.users.choice
        ? `Per card: the layout with the fewest GPUs in total that serves ${usersHead(a.users.total, a.users.concurrent, m.workload.activeShare)}, and the servers it takes; users are per server.`
        : `None serves ${usersHead(a.users.total, a.users.concurrent, m.workload.activeShare)}; per card, the closest layout.`
      : nobody
        ? // The binding limit: the memory, the first-token wait or the writing speed (I-C).
          `No layout serves anyone: ${nobodyReason(a.fit?.figures.requests?.limitedBy, `${m.workload.minTokS} tok/s`)}; per card, the closest layout.`
        : 'Per card: the fewest GPUs that serve someone at the minimum writing speed (else the fewest the model fits on), and what that layout serves.',
  );
  for (const l of table(a.cards, cardColumns(m.workload, notes, style.ascii, users), width - 2, style)) lines.push(`  ${l}`);
  lines.push(...wrap(USERS_LEGEND, width - 2).map((l) => `  ${l}`));
  lines.push(...wrap(layoutLegend(a.cards), width - 2).map((l) => `  ${l}`));
  if (a.layouts) {
    lines.push('');
    para('Every layout:');
    for (const l of table(a.layouts, layoutColumns(m.workload, style.ascii, notes), width - 2, { ascii: style.ascii })) lines.push(`  ${l}`);
  }
  if (notes.texts.length > 0) {
    lines.push('');
    for (const [i, t] of notes.texts.entries()) lines.push(...wrap(`(${i + 1}) ${t}`, width, 4));
  }
  lines.push('');
  for (const l of legend(m, a.users !== undefined, [...a.cards.map((c) => c.figures), ...(a.layouts ?? []), a.details?.figures])) para(l);
  para(weightsNote(m));
  if (m.compatChecked === false) para(COMPAT_UNCHECKED);
  if (m.hasVision && !m.textOnly) para('Multimodal model: with --text-only (vLLM --language-model-only) the vision encoder is not loaded.');
  if (a.details) {
    lines.push('');
    lines.push(...detailLines(m, a.details.figures, width, style.ascii));
    if (a.details.row) {
      lines.push('');
      lines.push(...memoryDetails(m, a.details.row, style.ascii));
    }
  }
  const text = lines.join('\n');
  return style.ascii ? asciiOnly(text) : text;
}

// ---------------------------------------------------------------------------------------------------------
// The summary

/** The cards asked for: this PC's card (--this-pc) only when it is one of them, never in the summary or --md. */
export const plannedCards = (a: Answer) => a.cards.filter((c) => !c.local || a.rows.some((r) => r.gpuId === c.gpuId));

/** GPUs in each server of a layout: 16 GPUs are 2 servers of 8. */
const perServer = (f: Pick<Figures, 'count' | 'servers'>) => f.count / serversOfFigures(f);

/** "1 server with 4× NVIDIA B300 SXM", "2 servers with 8× NVIDIA H100 SXM5 80GB each" (16 GPUs). */
export function serverText(f: Pick<Figures, 'count' | 'gpu' | 'servers'>, ascii = false): string {
  const n = serversOfFigures(f);
  return n === 1 ? `1 server with ${f.count}${ascii ? 'x' : '×'} NVIDIA ${f.gpu}` : serversText(n, f, ascii);
}

/** "4 servers with 4× NVIDIA B300 SXM each": `n` servers in all (two or more), each with its share of the GPUs. */
export function serversText(n: number, f: Pick<Figures, 'count' | 'gpu' | 'servers'>, ascii = false): string {
  return `${int(n)} servers with ${perServer(f)}${ascii ? 'x' : '×'} NVIDIA ${f.gpu} each`;
}

/** The verb that goes with serverText: "1 server … serves", "2 servers … serve". */
const verb = (f: Pick<Figures, 'count' | 'servers'>, one: string, many: string) => (serversOfFigures(f) === 1 ? one : many);

/**
 * What each request brings, for a reader: "reading documents of up to 128k tokens (≈ 98,000 words ≈ 200
 * pages; 64k on average)". The capacity is computed at the average, the memory at the maximum.
 */
export function workText(w: Workload): string {
  const avg = w.typicalContext < w.maxContext ? `; ${ktok(w.typicalContext)} on average` : '';
  const what = w.cachedShare > 0 ? 'chatting, with conversations' : 'with requests';
  return `${what} of up to ${ktok(w.maxContext)} tokens (${wordsAndPages(w.maxContext)}${avg})`;
}

/** The statuses that make a card "not enough". */
const NOT_ENOUGH: readonly CardStatus[] = ['no-fit', 'no-start', 'short', 'slow'];

/**
 * What is not enough, with the GPUs tried in one server for each: "H100 and A100, even with 8 GPUs in one server;
 * RTX 6000 Ada and consumer cards, even with 4; GH200, even with 2" (families whose every card is not enough; a
 * family whose editions allow different counts is listed edition by edition), or undefined. `except`: the cards
 * recommended, never listed.
 */
export function notEnoughText(cards: readonly CardLine[], except: readonly string[] = []): string | undefined {
  const families = new Map<string, { segment: string; all: boolean; cards: { name: string; tried: number }[] }>();
  for (const c of cards) {
    // A card outside the catalog (this PC's, --vram) is no family to list.
    if (c.gpuId === 'this-pc' || c.gpuId === 'custom') continue;
    const name = gpuFamily(c.gpu);
    const prev = families.get(name) ?? { segment: c.figures.segment, all: true, cards: [] };
    // Too slow with the fewest GPUs but serving someone with more: enough; at the limit with nobody guaranteed: not enough.
    const nobody = c.status === 'at-limit' && c.figures.users !== undefined && c.figures.users.guaranteed < 1;
    const bad = (NOT_ENOUGH.includes(c.status) || nobody) && c.servingGpus === undefined && !except.includes(c.gpuId);
    families.set(name, { segment: prev.segment, all: prev.all && bad, cards: [...prev.cards, { name: c.gpu, tried: c.serverGpus ?? c.maxGpus }] });
  }
  if (families.size === 0) return undefined;
  const every = [...families.values()].every((v) => v.all);
  const same = (xs: readonly { tried: number }[]) => xs.every((x) => x.tried === (xs[0] as { tried: number }).tried);
  const consumer = [...families.entries()].filter(([, v]) => v.segment === 'consumer');
  const consumerCards = consumer.flatMap(([, v]) => v.cards);
  const allConsumer = consumer.length > 0 && consumer.every(([, v]) => v.all) && same(consumerCards);
  // Grouped by the GPUs tried, most first; consumer cards together.
  const groups = new Map<number, string[]>();
  const add = (n: number, name: string) => groups.set(n, [...(groups.get(n) ?? []), name]);
  for (const [k, v] of families) {
    if (!v.all || (allConsumer && v.segment === 'consumer')) continue;
    if (same(v.cards)) add((v.cards[0] as { tried: number }).tried, k);
    else for (const c of v.cards) add(c.tried, c.name);
  }
  if (allConsumer) add((consumerCards[0] as { tried: number }).tried, 'consumer cards');
  if (groups.size === 0) return undefined;
  const and = (xs: string[]) => (xs.length === 1 ? (xs[0] as string) : `${xs.slice(0, -1).join(', ')} and ${xs.at(-1) as string}`);
  const parts = [...groups.entries()].sort((x, y) => y[0] - x[0]).map(([n, xs], i) => `${and(xs)}, even with ${i === 0 ? `${gpuCount(n)} in one server` : n}`);
  return `${every ? 'every card checked: ' : ''}${parts.join('; ')}`;
}

/**
 * "high (±2.9%)", "medium (±25–48%)", "low, not precise enough to decide (more than ±50%)": the level of the
 * largest error; a label "not precise enough" under ±50% always says why (`why`).
 */
export function confidence(errors: readonly number[], why?: string): string {
  const [min, max] = [Math.min(...errors), Math.max(...errors)];
  // At ±50% or more the span is no figure to quote: the rule says why.
  if (max >= DECIDE_MAX_ERROR) return `low, not precise enough to decide (more than ±${DECIDE_MAX_ERROR}%)`;
  const span = Math.round(min) === Math.round(max) ? plusMinus(max) : `±${int(min)}–${int(max)}%`;
  return why !== undefined ? `low, not precise enough to decide (${span}, but ${why})` : `${max <= 15 ? 'high' : 'medium'} (${span})`;
}

/**
 * Why a recommended layout needs care, in plain words: close to its guaranteed capacity (`load` of it in use),
 * or the engine supports it only in part or without verification. A speed that is not reliable is no caution: nothing
 * is recommended on it (summaryLines, doNotPlan).
 */
function cautions(f: Figures, load: number | undefined): string {
  const out: string[] = [];
  if (f.verdict === 'at-limit')
    out.push(f.limitCause === 'sparse-reserve' ? 'it is at the limit of memory: the 21 GiB per GPU that sparse attention needs beyond 128K tokens while it reads a long prompt may not fit' : "it is at the limit of memory: it fits only if the engine's fixed costs are at the low end of their range");
  if (load !== undefined && load >= SATURATION)
    out.push(`each server runs at ${int(Math.round(load * 100))}% of its guaranteed capacity, so waits grow fast with more traffic: one more server leaves headroom`);
  else if (f.saturation) out.push('reading the prompts keeps the GPUs almost always busy, so the writing speed is less certain');
  // Only the engine's real limits: partial support, or support nobody verified; link and speed notes are not cautions.
  if (f.compat.notes.some((n) => n.startsWith('partly supported'))) out.push('the engine supports this setup only in part (see the notes under the per-card table)');
  else if (f.compat.notes.some((n) => n.startsWith('engine support on this card not verified') || FORMAT_WARNING.test(n)))
    out.push('support for this setup by the engine is not verified (see the notes under the per-card table)');
  return out.length > 0 ? ` Caution: ${out.join('; ')}.` : '';
}

/**
 * Why figures carry no plan (the summary's "Do not plan on these user counts"): the speed is not reliable here, or the
 * user count is not precise enough to decide; undefined when they can carry one.
 */
function doNotPlan(f: Figures): string | undefined {
  const note = unreliableSpeed(f);
  if (note) return `the speed is not reliable here (${note})`;
  const u = f.users;
  if (u && u.guaranteed > 0 && (u.precision.kind === 'imprecise' || u.precision.errorPct >= DECIDE_MAX_ERROR)) return 'the user count is not precise enough to decide';
  return undefined;
}

/** "Each server runs its own copy of the model", or "2 copies" when a server holds several (8 (2×4)). */
function copiesText(f: Figures): string {
  const copies = f.replicas / serversOfFigures(f);
  return copies > 1 ? `Each server runs ${int(copies)} copies of the model` : 'Each server runs its own copy of the model';
}

/**
 * The summary: 4 to 6 lines without jargon — the hardware that serves the workload, the alternative, what is
 * not enough, the speed in words, and how far to trust it. Every claim agrees with the tables:
 * a card recommended is never "not enough", nothing is recommended that guarantees nobody, and an alternative
 * serves someone too.
 */
export function summaryLines(m: RenderModel, a: Answer, ascii = false): string[] {
  const w = m.workload;
  const work = workText(w);
  const out: string[] = [];
  const users = (f: Figures) => (f.users ? `${int(f.users.guaranteed)} users guaranteed, ${int(f.users.likely)} likely` : 'speed unknown');
  const k = ktok(w.maxContext);
  // The layout the speed and confidence lines describe: the one recommended (none: no speed line).
  let pick: Figures | undefined;
  const c = a.users?.choice;
  // Requests in use per server of the recommendation, as a share of its guaranteed ones.
  const load = (f: Figures, n: number) => (a.users && f.requests && f.requests.guaranteed > 0 ? a.users.concurrent / ((n / serversOfFigures(f)) * f.requests.guaranteed) : undefined);
  const limit = outsideLimits(m);
  if (limit !== undefined)
    out.push(
      `Recommendation: none — ${k} tokens is outside the model's limits: it is made for up to ${ktok(limit)} tokens, and beyond that the quality of its answers is not known; plan for at most ${ktok(limit)} tokens, or choose a model made for longer requests.`,
    );
  else if (a.users && c) {
    pick = c.figures;
    const n = c.servers ?? serversOfFigures(c.figures);
    const per = int(c.usersPerServer ?? Math.ceil(a.users.total / n));
    // Figures marked "do not plan" make no recommendation (I-B): the estimate, said as one, with the reason.
    const why = doNotPlan(c.figures);
    if (why)
      out.push(
        `Recommendation: none reliable — ${why}. The estimate, not to plan on: ${n === 1 ? serverText(c.figures, ascii) : serversText(n, c.figures, ascii)} for ${int(a.users.total)} users ${work}${n > 1 ? `: ≈ ${per} users per server` : ''}.` +
          cautions(c.figures, load(c.figures, n)),
      );
    else
      out.push(
        (n === 1
          ? `Recommendation: ${serverText(c.figures, ascii)} serves ${int(a.users.total)} users ${work}.`
          : `Recommendation: ${serversText(n, c.figures, ascii)} serve ${int(a.users.total)} users ${work}: ≈ ${per} users per server. ${copiesText(c.figures)}, and a load balancer (a router in front of the servers) spreads the users across them.`) +
          cautions(c.figures, load(c.figures, n)),
      );
  } else if (a.fit && (a.users || (a.fit.figures.users !== undefined && a.fit.figures.users.guaranteed < 1))) out.push(`Recommendation: none — ${nobodyCause(m, a, ascii)}.`);
  else if (a.fit) {
    const f = a.fit.figures;
    pick = f;
    const hw = serverText(f, ascii);
    if (!f.users) out.push(`Recommendation: ${hw} ${verb(f, 'holds', 'hold')} the model ${work}; its speed cannot be estimated (the card's figures are not published).`);
    else if (doNotPlan(f)) out.push(`Recommendation: none reliable — ${doNotPlan(f) as string}. The estimate, not to plan on: ${hw} ${verb(f, 'serves', 'serve')} ${users(f)}, ${work}.${cautions(f, undefined)}`);
    else out.push(`Recommendation: ${hw} ${verb(f, 'serves', 'serve')} ${users(f)}, ${work}.${cautions(f, undefined)}`);
  } else if (a.blocked) out.push(`Recommendation: none — the model fits in memory on ${serverText(a.blocked, ascii)}, but the inference engine does not start there.`);
  else {
    const t = a.attempt as Figures;
    const one = oneServer(t);
    out.push(
      `Recommendation: none — the model does not fit in one server ${scopeText(m)} (with ${serverText(one, ascii)} it needs ${gb(t.memory.value)} GiB per GPU; the card has ${t.cardGiB} GiB${one.count < t.count ? `; more servers are more copies of the model, not more room for one` : ''}).`,
    );
  }
  const beyond = m.modelMaxContext !== undefined && w.maxContext > m.modelMaxContext ? m.modelMaxContext : undefined;
  if (beyond !== undefined && limit === undefined)
    out.push(`Warning: ${k} tokens is more than the model's own maximum (${ktok(beyond)}): the engine needs a context extension (YaRN) to start, which the lab documents up to ${int(m.documentedMaxContext as number)} tokens.`);
  // An alternative only next to a recommendation one can plan on, and one that serves someone at a speed that is reliable.
  const serves = (f: Figures) => f.users === undefined || f.users.guaranteed > 0;
  const alt0 = c?.alternative ?? (a.users ? undefined : a.fit?.alternative);
  const alt = pick && alt0 && serves(alt0) && !doNotPlan(pick) && !unreliableSpeed(alt0) ? alt0 : undefined;
  if (alt) {
    const n = c?.alternativeServers ?? serversOfFigures(alt);
    out.push(
      n > 1
        ? `Alternative: ${serversText(n, alt, ascii)} (${c?.alternativeUsersPerServer !== undefined ? `≈ ${int(c.alternativeUsersPerServer)} users per server` : users(alt)}).${cautions(alt, load(alt, n))}`
        : `Alternative: ${serverText(alt, ascii)} (${alt.verdict === 'at-limit' ? 'at the limit' : 'fits'}; ${users(alt)}).${cautions(alt, load(alt, n))}`,
    );
  }
  const not = limit === undefined ? notEnoughText(plannedCards(a), [pick, alt].filter((f): f is Figures => f !== undefined).map((f) => f.gpuId)) : undefined;
  // Only the cards asked for, all failing: the recommendation already says so.
  if (not && !(m.selection && not.startsWith('every card'))) out.push(`Not enough: ${not}.`);
  const guaranteed = pick?.users !== undefined && pick.users.guaranteed > 0;
  if (pick?.tokS && pick.wait && pick.read && guaranteed) {
    const words = (x: NonNullable<Figures['tokS']>) => `writing about ${wordsPerSecond(x.value)} words per second (${wordsPerSecond(x.min)} to ${wordsPerSecond(x.max)}; ${int(x.value)} tokens/s)`;
    out.push(
      `Speed for each user: ${words(pick.tokS)}; with everyone busy the first words appear after about ${secs(pick.wait.value)} s, and a full ${k}-token request alone is read in about ${secs(pick.read.value)} s (1 token ≈ ${WORDS_PER_TOKEN} words).`,
    );
  }
  const memory = pick ?? a.blocked ?? a.fit?.figures ?? (a.attempt as Figures);
  // No recommendation but a layout that fits: its figures, as the tables show them; one level per figure, from its own
  // error ("not precise enough" under ±50% always says why: a caveat of the speed model, a context beyond the model).
  const sp = pick ?? a.fit?.figures;
  // Where the speed caveat was said: in the recommendation itself when it makes none, else in its caution.
  const above = pick && doNotPlan(pick) ? 'see the recommendation above' : 'see the caution above';
  const level = (what: string, p: Measure['precision']) =>
    `${what} ${confidence([p.errorPct], limit !== undefined ? `${k} tokens is beyond the model's maximum of ${ktok(limit)}` : p.kind === 'imprecise' ? (pick && unreliableSpeed(pick) ? above : (p.reason ?? 'the speed model does not cover this case').split('; ')[0]) : undefined)}`;
  // No run measured pipeline stages: said whenever the layout described has them (calibration.ts PIPELINE_WIDENING).
  const pipeline = sp && sp.pp > 1 ? ['pipeline speed not measured (its ranges are widened by half)'] : [];
  const speedText =
    sp?.tokS && sp.read && sp.users
      ? [level('writing speed', sp.tokS.precision), level('reading time', sp.read.precision), sp.users.guaranteed > 0 ? level('user count', sp.users.precision) : 'user count: nobody guaranteed', ...pipeline].join('; ')
      : 'speed not estimated';
  // A plan on the guaranteed figures only when they can carry one: not when the calibration says the speed is not
  // reliable, nor when the user count is not precise enough to decide.
  const usersVague = sp?.users !== undefined && sp.users.guaranteed > 0 && (sp.users.precision.kind === 'imprecise' || sp.users.precision.errorPct >= DECIDE_MAX_ERROR);
  // The caveat in full once: in the recommendation's caution when there is one, else here.
  const note = sp ? unreliableSpeed(sp) : undefined;
  const unreliable = note ? `the speed is not reliable here (${pick ? above : note})` : usersVague ? 'the user count is not precise enough to decide' : undefined;
  const ending = !guaranteed ? 'Nothing is guaranteed here: do not rely on these figures alone.' : unreliable ? `Do not plan on these user counts: ${unreliable}.` : 'Plan on the guaranteed figures.';
  out.push(`Confidence: memory ${confidence([memory.memory.precision.errorPct])}; ${speedText}. ${ending}`);
  return out;
}

/** Under the per-card table: how to read "4 (1 model on 4)" (the server size: see layoutLegend). */
export const LAYOUT_LEGEND =
  'GPUs: "4 (1 model on 4)" or "4 (1×4)" = one copy of the model split across 4 GPUs that work together on every token (tensor parallel, TP); "8 (2 copies × 4)" or "8 (2×4)" = 2 independent copies, each on 4 GPUs and serving its own users; "split in 2 stages" = each GPU holds part of the layers (pipeline parallel, PP); "2 stages of 4" = each stage on 4 GPUs in tensor parallel; "2 copies × 2 stages of 2" or "2×2 stages of 2" = 2 independent copies, each split in 2 stages of 2 GPUs. One server holds up to 8 GPUs.';

/** LAYOUT_LEGEND with the server sizes of the cards shown: "One server holds up to 8 GPUs (RTX 5090: 4, GH200: 2)." */
export function layoutLegend(cards: readonly CardLine[]): string {
  const sizes = new Map<number, string[]>();
  for (const c of cards) if (c.serverSize !== undefined) sizes.set(c.serverSize, [...(sizes.get(c.serverSize) ?? []), c.gpu]);
  const most = Math.max(8, ...sizes.keys());
  const fewer = [...sizes.entries()].filter(([n]) => n < most).sort((x, y) => y[0] - x[0]);
  const all = sizes.size === 1 ? ([...sizes.keys()][0] as number) : undefined;
  const text =
    all !== undefined && all < 8 ? `One server holds up to ${gpuCount(all)} of these cards.` : `One server holds up to ${most} GPUs${fewer.length > 0 ? ` (${fewer.map(([n, xs]) => `${xs.join(', ')}: ${n}`).join('; ')})` : ''}.`;
  return LAYOUT_LEGEND.replace(/ One server holds up to 8 GPUs\.$/, ` ${text}`);
}

/** The speed bases of the rows shown (never a card that is not shown). */
export function shownBases(rows: readonly (Figures | undefined)[]): string[] {
  const out = new Set<string>();
  for (const f of rows) if (f?.tokS) out.add(`${f.tokS.precision.basis}${f.tokS.precision.points !== undefined ? ` (${f.tokS.precision.points} points)` : ''}`);
  return [...out];
}

// ---------------------------------------------------------------------------------------------------------
// CSV

/** What every CSV line says about the model and the workload. */
export interface CsvModel {
  model: string;
  version: string;
  format: string;
  family: string;
  engine: string;
  engineVersion: string;
  cache: string;
  workload: Pick<Workload, 'maxContext' | 'typicalContext' | 'outputTokens' | 'cachedShare' | 'minTokS' | 'activeShare'>;
  /** "chosen (--tokens)", "native maximum", "official YaRN"… */
  contextLabel: string;
}

/** One CSV line: a layout's figures, with the card's status when it comes from the overview. */
export interface CsvLine {
  figures: Figures;
  status?: CardStatus;
  /** --users: servers of this layout that serve them. */
  serversNeeded?: number;
  note?: string;
  local?: boolean;
}

type Cell = string | number | undefined;
const groups: [name: string, unit: string, get: (f: Figures) => Measure | undefined][] = [
  ['Memory per GPU', 'GiB', (f) => f.memory],
  ['Held by the engine per GPU', 'GiB', (f) => f.engineGiB],
  ['Weights per GPU', 'GiB', (f) => f.weights],
  ['KV cache per request', 'GiB', (f) => f.kv],
  ['Fixed costs', 'GiB', (f) => f.fixed],
  ['Margin', 'GiB', (f) => f.margin],
  ['Speed per user', 'tok/s', (f) => f.tokS],
  ['Speed of one request alone', 'tok/s', (f) => f.single],
  ['Context reading', 's', (f) => f.read],
  ['First token under load', 's', (f) => f.wait],
];

const baseText = (p: Measure['precision']) => `${precisionText(p)}: ${p.basis}`;

export const CSV_HEADERS = [
  'Model',
  'Version',
  'Weight format',
  'Family',
  'Engine',
  'Engine version',
  'KV cache',
  'Context (tokens)',
  'Context kind',
  'Typical context (tokens)',
  'New tokens per request',
  'Output tokens',
  'Minimum speed (tok/s)',
  'Active users (%)',
  'Card',
  'Card id',
  'Segment',
  'This PC',
  'Card memory (GiB)',
  'Layout',
  'GPUs',
  // One layout is 'Servers per layout' servers: the users and requests below are those of one layout, the servers
  // needed are whole servers (a 16-GPU layout is 2 servers of 8).
  'Servers per layout',
  'Servers needed for the users',
  'TP',
  'PP',
  'Copies',
  'Card verdict',
  'Memory verdict',
  'Engine compatibility',
  'Context validity',
  ...groups.flatMap(([n, u]) => [`${n} (${u})`, `${n} min (${u})`, `${n} max (${u})`, `${n} error (%)`, `${n} basis`]),
  'Concurrent requests guaranteed per layout',
  'Concurrent requests likely per layout',
  'Concurrent requests memory limit per layout',
  'Limited by',
  'Users likely per layout',
  'Users min (guaranteed) per layout',
  'Users max (memory limit) per layout',
  'Users error (%)',
  'Users basis',
  'Physical limit generation (tok/s)',
  'Generation share of the limit (%)',
  'Physical limit context reading (s)',
  'Reading share of the limit (%)',
  'Notes',
];

function csvCells(m: CsvModel, l: CsvLine): Cell[] {
  const f = l.figures;
  const w = m.workload;
  const r = f.roofline;
  return [
    m.model,
    m.version,
    m.format,
    m.family,
    m.engine,
    m.engineVersion,
    m.cache,
    w.maxContext,
    m.contextLabel,
    w.typicalContext,
    Math.round(w.typicalContext * (1 - w.cachedShare)),
    w.outputTokens,
    w.minTokS,
    w.activeShare * 100,
    f.gpu,
    f.gpuId,
    f.segment,
    l.local ? 'yes' : 'no',
    f.cardGiB,
    f.hardware,
    f.count,
    serversOfFigures(f),
    l.serversNeeded ?? f.serversNeeded,
    f.tp,
    f.pp,
    f.replicas,
    l.status ? STATUS[l.status] : undefined,
    VERDICT[f.verdict],
    f.compat.status === 'ok' ? 'starts' : f.compat.status === 'warnings' ? `starts, with warnings: ${f.compat.notes.join(' | ')}` : `does not start: ${f.compat.notes.join(' | ')}`,
    f.validity,
    ...groups.flatMap(([, , get]) => {
      const x = get(f);
      return x ? [x.value, x.min, x.max, x.precision.errorPct, baseText(x.precision)] : [undefined, undefined, undefined, undefined, undefined];
    }),
    f.requests?.guaranteed,
    f.requests?.likely,
    f.requests?.memory,
    f.requests?.limitedBy,
    f.users?.likely,
    f.users?.guaranteed,
    f.users?.memoryMax,
    f.users?.precision.errorPct,
    f.users ? baseText(f.users.precision) : undefined,
    r?.tokSLimit,
    r ? r.tokSShare * 100 : undefined,
    r?.readLimitS,
    r ? r.readShare * 100 : undefined,
    [l.note, f.speedNote, f.saturation, r?.warning].filter(Boolean).join(' | '),
  ];
}

/**
 * CSV: one line per model × layout × context. Standard by default (",", decimal point, UTF-8); `excelEu`
 * for Excel in comma-decimal locales (";", decimal comma, UTF-8 with BOM). CRLF either way; units are in the
 * headers, never in the cells.
 */
export function csvText(lines: readonly (readonly [CsvModel, CsvLine])[], excelEu: boolean): string {
  const sep = excelEu ? ';' : ',';
  const cell = (v: Cell) => {
    const s = v === undefined ? '' : typeof v === 'number' ? (Number.isInteger(v) ? String(v) : excelEu ? v.toFixed(2).replace('.', ',') : v.toFixed(2)) : v;
    return /["\n\r]/.test(s) || s.includes(sep) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const out = [CSV_HEADERS.map(cell).join(sep), ...lines.map(([m, l]) => csvCells(m, l).map(cell).join(sep))];
  return `${excelEu ? BOM : ''}${out.join('\r\n')}\r\n`;
}

/** The CSV model of a live answer. */
export function csvModel(m: RenderModel): CsvModel {
  return {
    model: m.repo,
    version: m.variant || m.repo,
    format: m.format,
    family: m.family,
    engine: m.engine === 'vllm' ? 'vLLM' : 'SGLang',
    engineVersion: calibrationFor(m.engine).engineVersion,
    cache: m.cache,
    workload: m.workload,
    contextLabel: m.contextIsModelMax ? 'model maximum' : 'chosen (--tokens)',
  };
}

/** Every layout planned, the overview's cards marked with their status. */
export function renderCsv(m: RenderModel, a: Answer, excelEu: boolean): string {
  const cm = csvModel(m);
  const byLayout = new Map(a.cards.map((c) => [c.figures.hardware, c]));
  return csvText(
    a.rows.map((f) => {
      const c = byLayout.get(f.hardware);
      return [cm, { figures: f, ...(c ? { status: c.status, ...(c.note ? { note: c.note } : {}), ...(c.local ? { local: true } : {}), ...(c.serversNeeded !== undefined ? { serversNeeded: c.serversNeeded } : {}) } : {}) }] as const;
    }),
    excelEu,
  );
}

// ---------------------------------------------------------------------------------------------------------
// Markdown and JSON

const mdEscape = (s: string) => s.replace(/\|/g, '\\|');

/** A Markdown table of a text table's columns (long forms). */
export function mdTable<T>(items: readonly T[], cols: readonly Column<T>[]): string[] {
  return [`| ${cols.map((c) => c.head.replace('\n', ' ')).join(' | ')} |`, `| ${cols.map((c) => (c.right ? '---:' : '---')).join(' | ')} |`, ...items.map((x) => `| ${cols.map((c) => mdEscape(c.long(x))).join(' | ')} |`)];
}

/**
 * A card's verdict for the Markdown summary, with the cause when nobody is guaranteed: the speed shown next to it is the
 * central estimate under load, while the guarantee rests on the low end.
 */
function mdVerdict(l: CardLine, minTokS: number): string {
  const u = l.figures.users;
  const slow = u !== undefined && u.guaranteed < 1 && (l.status === 'slow' || l.status === 'short' || l.status === 'at-limit');
  if (!slow) return STATUS[l.status];
  const by = l.figures.requests?.limitedBy;
  const why = by === 'memory' || by === 'wait' ? nobodyReason(by, '') : `nobody is guaranteed ${wordsAndTokS(minTokS)} of writing (low end of the estimate)`;
  return `${l.status === 'at-limit' ? 'at the limit of memory; ' : ''}serves nobody: ${why}${u.likely > 0 ? `; ${int(u.likely)} users likely` : ''}`;
}

/** The options the Markdown summary lists: every card that holds the model and starts, in plain words. */
function optionColumns(users: boolean, minTokS: number): Column<CardLine>[] {
  const speed = (l: CardLine) => l.figures.tokS !== undefined;
  const each = users ? ', per server' : '';
  const range = (x: Measure, f: (n: number) => string) => `about ${f(x.value)} (${f(x.min)} to ${f(x.max)})`;
  return [
    { head: users ? 'Each server' : 'Hardware', long: (l) => (users ? `${perServer(l.figures)}× NVIDIA ${l.figures.gpu}` : serverText(l.figures)), drop: 0 },
    ...(users ? [{ head: 'Servers needed', long: (l: CardLine) => (l.serversNeeded !== undefined ? int(l.serversNeeded) : 'n/a'), drop: 0, right: true }] : []),
    { head: 'How the GPUs hold the model', long: (l) => layoutText(l.figures) || 'one GPU', drop: 0 },
    { head: 'Verdict', long: (l) => mdVerdict(l, minTokS), drop: 0 },
    { head: `Users (guaranteed${each})`, long: (l) => (speed(l) ? int(users && l.usersPerServer ? l.usersPerServer.guaranteed : l.figures.users!.guaranteed) : 'n/a'), drop: 0, right: true },
    { head: `Users (likely${each})`, long: (l) => (speed(l) ? int(users && l.usersPerServer ? l.usersPerServer.likely : l.figures.users!.likely) : 'n/a'), drop: 0, right: true },
    { head: 'Words per second per user', long: (l) => (speed(l) ? range(l.figures.tokS!, wordsPerSecond) : 'n/a'), drop: 0, right: true },
    { head: 'Answer starts after (s)', long: (l) => (speed(l) ? range(l.figures.wait!, secs) : 'n/a'), drop: 0, right: true },
    { head: 'Accuracy (users, speed)', long: (l) => accuracy(l.figures, precisionShort) || 'n/a', drop: 0 },
  ];
}

/**
 * --md: the answer as a one-page Markdown summary — the scenario, the recommendation and the
 * alternatives, the options in plain words, the confidence, the assumptions and where the method is described.
 */
export function renderMarkdown(m: RenderModel, a: Answer): string {
  const w = m.workload;
  const cal = calibrationFor(m.engine);
  const options = plannedCards(a).filter((c) => !['no-fit', 'no-start'].includes(c.status));
  const summary = summaryLines(m, a);
  const confidenceLine = summary.find((s) => s.startsWith('Confidence:')) as string;
  const usersLine = a.users
    ? `${int(a.users.total)} (${num(w.activeShare * 100, 0)}% of them active at the same moment ≈ ${int(a.users.concurrent)} requests at once)`
    : `as many as the hardware serves (${num(w.activeShare * 100, 0)}% of them active at the same moment)`;
  const beyond = beyondNative(m);
  const out = [
    `# Summary: ${modelName(m)}${a.users ? ` for ${int(a.users.total)} users` : ''}`,
    '',
    '## Scenario',
    '',
    `- Model: ${m.repo}${m.variant ? ` (${m.variant})` : ''}, ${m.format} weights, served with ${m.engine === 'vllm' ? 'vLLM' : 'SGLang'} ${cal.engineVersion}.`,
    `- Users: ${usersLine}.`,
    `- Work: ${w.cachedShare > 0 ? `a chat: most of each conversation (${num(w.cachedShare * 100, 0)}%) is already held by the server` : 'documents: every request brings new text to read'}; each request up to ${ktok(w.maxContext)} tokens (${wordsAndPages(w.maxContext)}), answers of ${ktok(w.outputTokens)} tokens (${wordsAndPages(w.outputTokens).replace(/ ≈ [^≈]*pages?$/, '')}).`,
    ...(beyond ? [`- ${beyond}`] : []),
    '',
    '## Recommendation',
    '',
    ...summary.filter((s) => s !== confidenceLine).map((s) => `- ${s}`),
    '',
    '## Options',
    '',
  ];
  if (options.length > 0) out.push(...mdTable(options, optionColumns(a.users !== undefined, w.minTokS)), '', `_Users: ${USERS_LEGEND} ${layoutLegend(options).replace(/^.* (One server holds)/, '$1')}_`);
  else out.push('No card of the catalog holds the model and starts with up to 8 GPUs.');
  out.push(
    '',
    '## Confidence',
    '',
    confidenceLine,
    '',
    '## Assumptions',
    '',
    `- ${num(w.activeShare * 100, 0)}% of the users send a request at the same moment (--active).`,
    `- Every user gets at least ${wordsAndTokS(w.minTokS)} of writing and the first words within ${waitLimitText(w.maxTtftMs)}${w.ttftScales ? ', or, if longer, within twice the wait for the first words of one typical request alone (half the context, less what the prefix cache holds)' : ''}.`,
    `- The conversation memory (KV cache) is ${m.cache === 'fp8' ? 'stored in fp8, the usual production setting' : "kept at the model's full precision"}; 15% of it is kept free as a safety margin.`,
    `- Words and pages: ${WORDS_PER_TOKEN} words per token, ${WORDS_PER_PAGE} words per page.`,
    '',
    '## Method',
    '',
    'Memory is computed from the model files and the engine rules; speed is calibrated on measured runs and public benchmarks. Sources and limits: docs/METHODOLOGY.md.',
    '',
  );
  return out.join('\n');
}

/** Everything as JSON: the answer, the overview per card and every layout with all its figures. */
export function renderJson(m: RenderModel, a: Answer): string {
  return `${JSON.stringify(
    {
      model: m,
      summary: summaryLines(m, a),
      headline: headline(m, a),
      scenario: scenario(m.workload),
      fit: a.fit ?? null,
      users: a.users ?? null,
      cards: a.cards,
      ...(a.details ? { details: a.details.figures } : {}),
      rows: a.rows,
      units: { memory: 'GiB (2^30 bytes, as nvidia-smi and CUDA count)', speed: 'tokens/s', time: 's', context: 'tokens' },
      accuracy: { maxErrorToDecidePct: DECIDE_MAX_ERROR, legend: legend(m, a.users !== undefined, [...a.cards.map((c) => c.figures), a.details?.figures]) },
    },
    null,
    1,
  )}\n`;
}

/** UTF-8 byte-order mark: Excel reads the file as UTF-8. */
const BOM = String.fromCharCode(0xfeff);

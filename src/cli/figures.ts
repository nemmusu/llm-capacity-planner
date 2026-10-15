/**
 * The numbers the capacity command shows, each with its range, its error and how far it can be trusted,
 * derived once from the planner's rows so the text, CSV, JSON, Markdown and the ready-made reports say the
 * same thing:
 *  - every quantity is a value (the centre) with its min and max, the error ± (half the range over its
 *    centre) and an accuracy label: exact, measured, calibrated, estimated, not precise enough to decide;
 *  - the context range the speed calibration covers, the engine's compatibility, the roofline check (the
 *    estimate as a share of the card's datasheet bandwidth and TFLOPS) and the warning near saturation;
 *  - one line per catalog card: the fewest GPUs the model fits on, or why it does not.
 * Plain data (the zod schemas below): data/report.json stores it as is.
 */
import { z } from 'zod';
import type { MiBRange } from '../model/serving-fit';
import { calibrationFor, type SpeedBasis } from '../planner/calibration';
import { serverSize, type Gpu, type ParallelShape } from '../planner/hardware';
import { pipelineUnverified, pipelineWeight, serverPlan, serversOf, SPARSE_RESERVE_REASON, type PlannerModel, type PlanOptions, type PlanRow, type ServerPlan } from '../planner/plan';
import type { Verification } from '../planner/weights-verification';

const GIB = 1024;

export const precisionSchema = z.object({
  kind: z.enum(['exact', 'measured', 'calibrated', 'estimated', 'imprecise']),
  /** Half the range over its centre, %. */
  errorPct: z.number(),
  /** Where the number comes from. */
  basis: z.string(),
  /** Measured points the calibration is fitted on. */
  points: z.number().optional(),
  /** Why the number is only estimated, or not precise enough. */
  reason: z.string().optional(),
});
export type Precision = z.infer<typeof precisionSchema>;

/** A quantity: its centre, its range and how far it can be trusted. */
export const measureSchema = z.object({ value: z.number(), min: z.number(), max: z.number(), precision: precisionSchema });
export type Measure = z.infer<typeof measureSchema>;

/** The memory verdict of a layout: fits, at the limit, does not fit, the engine does not start. */
const verdictSchema = z.enum(['fits', 'at-limit', 'no-fit', 'no-start']);
/** The planner's verdict identifiers in English. */
export const VERDICT_ID = { fits: 'fits', 'borderline': 'at-limit', 'does-not-fit': 'no-fit', impossible: 'no-start' } as const;

export const figuresSchema = z.object({
  /** "4× H200 SXM TP4". */
  hardware: z.string(),
  /** Servers the layout takes (plan.ts serversOf: at most 8 GPUs, 4 workstation cards, 2 GH200 per server). */
  servers: z.number().optional(),
  gpuId: z.string(),
  /** The card's short name ("H200 SXM"). */
  gpu: z.string(),
  segment: z.string(),
  count: z.number(),
  tp: z.number(),
  pp: z.number(),
  replicas: z.number(),
  verdict: verdictSchema,
  /** The memory would fit at best (the engine's checks aside). */
  memoryFits: z.boolean(),
  /** Whether the engine starts: its checks (blockers) and caveats (warnings). */
  compat: z.object({ status: z.enum(['ok', 'warnings', 'does-not-start']), notes: z.array(z.string()) }),
  /** The card's whole memory as nvidia-smi reports it, GiB (the same unit as every other memory figure). */
  cardGiB: z.number(),
  /** Memory one GPU needs with one request at the full context (weights + fixed costs + its KV cache), GiB. */
  memory: measureSchema,
  /** Memory the engine process holds on each GPU: its whole share plus what it keeps outside it, GiB. */
  engineGiB: measureSchema,
  weights: measureSchema,
  /** KV cache of one request at the full context, per GPU, GiB. */
  kv: measureSchema,
  fixed: measureSchema,
  /** KV room left in the engine's share after one full-context request, GiB (negative: missing). */
  margin: measureSchema,
  /** Concurrent requests: guaranteed, likely, and what the memory holds at the typical context. */
  requests: z.object({ guaranteed: z.number(), likely: z.number(), memory: z.number(), limitedBy: z.enum(['memory', 'speed', 'wait', 'none']) }).optional(),
  /** Users in total: guaranteed, likely, and the most the memory alone would allow. */
  users: z.object({ guaranteed: z.number(), likely: z.number(), memoryMax: z.number(), precision: precisionSchema }).optional(),
  /** Tokens/s each user gets at the guaranteed load. */
  tokS: measureSchema.optional(),
  /** Tokens/s of one request alone. */
  single: measureSchema.optional(),
  /** Seconds to read a full-context prompt alone. */
  read: measureSchema.optional(),
  /** Seconds to the first token at the guaranteed load. */
  wait: measureSchema.optional(),
  /** The estimates as a share of the card's physical limits (bandwidth for generation, TFLOPS for reading). */
  roofline: z
    .object({
      tokSLimit: z.number(),
      tokSShare: z.number(),
      readLimitS: z.number(),
      readShare: z.number(),
      /** The card's TFLOPS are derived (hardware.json tflopsDerived), not a datasheet figure. */
      tflopsDerived: z.literal(true).optional(),
      warning: z.string().optional(),
    })
    .optional(),
  /** The context range the speed calibration covers ("context calibrated up to 1M"). */
  validity: z.string(),
  saturation: z.string().optional(),
  /** Share of the GPU time spent reading prompts at the guaranteed load (0–1): why a long context is slow. */
  readingShare: z.number().optional(),
  /** --users: servers of this layout that serve them (plan.ts serverPlan); absent when it serves nobody. */
  serversNeeded: z.number().optional(),
  speedNote: z.string().optional(),
  /** At the limit only because of the sparse-attention reserve (plan.ts limitCause). */
  limitCause: z.literal('sparse-reserve').optional(),
});
export type Figures = z.infer<typeof figuresSchema>;

export const cardStatusSchema = z.enum(['fits', 'no-speed', 'slow', 'short', 'at-limit', 'no-start', 'no-fit']);
/**
 * fits; fits but its speed cannot be estimated; fits but too slow for anyone; does not serve the users asked; at the
 * limit; the engine does not start; does not fit.
 */
export type CardStatus = z.infer<typeof cardStatusSchema>;

/** One catalog card in the overview: the layout it is described by and why. */
export const cardLineSchema = z.object({
  gpuId: z.string(),
  gpu: z.string(),
  status: cardStatusSchema,
  /** Fewest GPUs of this card the model fits on (undefined: none up to the largest count tried). */
  minGpus: z.number().optional(),
  /** Largest count tried (8, or --count). */
  maxGpus: z.number(),
  note: z.string().optional(),
  /** A slow card that serves someone with more GPUs: how many. */
  servingGpus: z.number().optional(),
  /** The most GPUs of this card tried in one server (plan.ts serverSize, or --count): what "not enough" refers to. */
  serverGpus: z.number().optional(),
  /** The most GPUs of this card one server holds (plan.ts serverSize). */
  serverSize: z.number().optional(),
  /** --users: servers of this layout that serve them (independent copies behind a load balancer). */
  serversNeeded: z.number().optional(),
  /** --users: users one server of this layout serves (plan.ts serverPlan), in the same unit as serversNeeded. */
  usersPerServer: z.object({ guaranteed: z.number(), likely: z.number() }).optional(),
  /** The layout the line describes: the fewest GPUs that fit (with --users: that serve them), else the largest attempt. */
  figures: figuresSchema,
  local: z.boolean().optional(),
});
export type CardLine = z.infer<typeof cardLineSchema>;

/** At or above this error a speed is not precise enough to decide on. */
export const DECIDE_MAX_ERROR = 50;
/** Above this share of the load's capacity the estimate degrades (queues grow fast). */
export const SATURATION = 0.85;
/** Above this share of a physical limit the estimate is suspect. */
export const ROOFLINE_MAX = 0.9;

/** The cards the public benchmarks ran on: elsewhere the public calibration is "calibrated on other cards". */
const PUBLIC_CARDS: Record<'vllm' | 'sglang', readonly string[]> = {
  vllm: ['h100-sxm5-80gb', 'h200-sxm-141gb', 'h200-nvl-141gb', 'b200-sxm-180gb', 'b300-sxm'],
  sglang: ['h100-sxm5-80gb', 'h200-sxm-141gb', 'b200-sxm-180gb', 'gb200-nvl72'],
};

/** A model measured on a card: its speed is "measured". */
export interface MeasuredRun {
  repo: string;
  gpu: string;
  engine: 'vllm' | 'sglang';
  weights: ParallelShape['weights'];
  kv: 'fp8' | 'auto';
  /** The longest context measured, tokens. */
  maxTokens: number;
}

/** The full models among the RTX 5090 runs (the others were cut to fewer layers). */
export const MEASURED_RUNS: readonly MeasuredRun[] = [
  { repo: 'Qwen/Qwen3-8B', gpu: 'rtx-5090-32gb', engine: 'vllm', weights: 'bf16', kv: 'fp8', maxTokens: 32_768 },
];

/** "32k", "208k", "1M", "1.5M". */
export function ktok(n: number): string {
  if (n >= 1_048_576) return `${(Math.round((n / 1_048_576) * 10) / 10).toLocaleString('en-US')}M`;
  if (n >= 10_240) return `${Math.round(n / 1024)}k`;
  // Below 10k one decimal: 1,536 is "1.5k", 1,500 "1.5k" (not "2k"), 1,024 "1k".
  if (n >= 1024) return `${Math.round((n / 1024) * 10) / 10}k`;
  return n.toLocaleString('en-US');
}

/** Half the range over its centre, % (0 for an exact value). */
export function errorPct(min: number, max: number): number {
  const sum = Math.abs(min + max);
  return sum > 0 ? (Math.abs(max - min) / sum) * 100 : 0;
}

export function measure(min: number, max: number, precision: Omit<Precision, 'errorPct'>): Measure {
  return { value: (min + max) / 2, min, max, precision: { ...precision, errorPct: errorPct(min, max) } };
}

/** What a figure needs besides the row: the model, the options and its repo. */
export interface FigureContext {
  model: PlannerModel;
  opts: PlanOptions;
  repo: string;
  verification: Verification;
}

const shortNames: [RegExp, string][] = [
  [/^NVIDIA (GeForce )?/, ''],
  [/\s*\(.*\)$/, ''],
  [/ Generation$/, ''],
  [/ Blackwell Workstation Edition$/, ' WS'],
  [/ Blackwell Max-Q Workstation Edition$/, ' Max-Q'],
  [/ Blackwell Server Edition$/, ' Server'],
  [/ Grace Hopper Superchip/, ''],
  [/ HBM3e?$/, ''],
];

/** A card's name short enough for a table ("RTX PRO 6000 WS", "GH200 144GB"). */
export function shortGpuName(name: string): string {
  return shortNames.reduce((s, [re, to]) => s.replace(re, to), name);
}

/** How a layout's GPUs hold the model, in words: "1 model on 4", "2 copies × 4", "split in 2 stages", "split in 2 stages of 4" ('' for one GPU). */
export function layoutText(l: { count: number; tp: number; pp: number; replicas?: number }, ascii = false): string {
  const x = ascii ? 'x' : '×';
  const copies = l.replicas ?? 1;
  if (l.count === 1) return '';
  const of = l.tp > 1 ? ` of ${l.tp}` : '';
  if (l.pp > 1) return copies > 1 ? `${copies} copies ${x} ${l.pp} stages${of}` : `split in ${l.pp} stages${of}`;
  return copies > 1 ? `${copies} copies ${x} ${l.tp}` : `1 model on ${l.tp}`;
}

/** The same in a few characters for narrow tables: "1×4", "2×4", "2 stages", "2 stages of 4". */
export function layoutShort(l: { count: number; tp: number; pp: number; replicas?: number }, ascii = false): string {
  const x = ascii ? 'x' : '×';
  const copies = l.replicas ?? 1;
  if (l.count === 1) return '';
  const of = l.tp > 1 ? ` of ${l.tp}` : '';
  if (l.pp > 1) return copies > 1 ? `${copies}${x}${l.pp} stages${of}` : `${l.pp} stages${of}`;
  return `${copies}${x}${l.tp}`;
}

/** Servers a figure's layout takes: the planner's (serversOf); reports packed before it count 8 GPUs per server. */
export const serversOfFigures = (f: { count: number; servers?: number }) => f.servers ?? Math.ceil(f.count / 8);

/** A card's family, as the summary names it ("H100" for every H100, "RTX PRO 6000" for its three editions). */
export function gpuFamily(shortName: string): string {
  let s = shortName;
  for (let prev = ''; prev !== s; ) {
    prev = s;
    s = s.replace(/ (\d+GB|PCIe|SXM\d?|NVL|WS|Max-Q|Server)$/, '');
  }
  return s;
}

/** "4× B300 SXM (1 model on 4)", "8× H100 SXM5 80GB (4 copies × 2)", "2× RTX 4090 (split in 2 stages)". */
export function hardwareLabel(r: Pick<PlanRow, 'layout'>, ascii = false): string {
  const { gpu, count } = r.layout;
  const how = layoutText(r.layout, ascii);
  return `${count > 1 ? `${count}${ascii ? 'x' : '×'} ` : ''}${shortGpuName(gpu.name)}${how ? ` (${how})` : ''}`;
}

const ENGINE = { vllm: 'vLLM', sglang: 'SGLang' } as const;

function weightsPrecision(v: Verification, engine: string): Omit<Precision, 'errorPct'> {
  if (v.status === 'verified') return { kind: 'exact', basis: `sum of the loaded tensors; format verified on ${engine} measurements` };
  if (v.status === 'gap') return { kind: 'exact', basis: 'sum of the loaded tensors', reason: `the engine measured up to +${Math.round(v.gapShare * 1000) / 10}% for this format: counted in the fixed costs` };
  return { kind: 'exact', basis: 'sum of the loaded tensors', reason: `format not yet verified by a measurement: ${Math.round(v.gapShare * 100)}% margin in the fixed costs` };
}

/** The speed's accuracy on a row: measured, calibrated on this card, estimated, or not precise enough. */
export function speedPrecision(ctx: FigureContext, row: PlanRow, range: MiBRange): Precision {
  const { gpu } = row.layout;
  const b = row.speedBasis as SpeedBasis;
  const err = errorPct(range.min, range.max);
  const engine = ctx.opts.engine;
  const tokens = ctx.opts.workload.maxContext;
  const own = b.kind === 'measured';
  // The planner's basis: its own models and points when measured, the public benchmarks (PUBLIC_POINTS) or an estimate.
  const basis = own ? b.base : `${b.base}, ${ENGINE[engine]} ${b.cal.engineVersion}`;
  const out = (kind: Precision['kind'], ...why: string[]): Precision => {
    // No run measured pipeline stages: their range is widened by half (calibration.ts PIPELINE_WIDENING).
    const pipeline = row.layout.pp > 1 ? ['pipeline speed not measured: range widened by half'] : [];
    const reasons = [...why, ...pipeline, ...(tokens > b.validUpTo ? [`extrapolated beyond the ${ktok(b.validUpTo)} tokens verified`] : [])];
    return { kind, errorPct: err, basis, points: b.points, ...(reasons.length ? { reason: reasons.join('; ') } : {}) };
  };
  if (row.speedNote) return out('imprecise', row.speedNote);
  if (err >= DECIDE_MAX_ERROR) return out('imprecise', `a range of ±${Math.round(err)}% is too wide to decide on`);
  const measured = MEASURED_RUNS.some(
    (m) => m.repo === ctx.repo && m.gpu === gpu.id && m.engine === engine && m.weights === ctx.model.parallel.weights && m.kv === ctx.opts.kvCache && row.layout.count === 1 && tokens <= m.maxTokens,
  );
  if (measured) return out('measured');
  if (b.kind === 'estimated') return out('estimated', ...(b.base.startsWith('estimated') ? [`${ctx.model.parallel.weights.toUpperCase()} ${ctx.model.family ?? 'dense'} not verified with ${ENGINE[engine]}: range doubled`] : []));
  if (!own && !PUBLIC_CARDS[engine].includes(gpu.id)) return out('estimated', 'calibrated on other cards');
  return out(tokens > b.validUpTo ? 'estimated' : 'calibrated');
}

/** "context calibrated up to 1M" or "context extrapolated beyond the 8k verified" (the speed's calibration). */
export function contextValidity(ctx: FigureContext, b: SpeedBasis): string {
  return ctx.opts.workload.maxContext <= b.validUpTo ? `context calibrated up to ${ktok(b.validUpTo)}` : `context extrapolated beyond the ${ktok(b.validUpTo)} verified`;
}

const pctText = (share: number) => `${Math.round(share * 100)}%`;

const LIMITED_BY = { memory: 'memory', speed: 'speed', wait: 'wait', none: 'none' } as const;

/** The figures of one row (module comment). */
export function figuresOf(ctx: FigureContext, row: PlanRow, users?: { concurrent: number; total?: number; active?: number }): Figures {
  const { layout } = row;
  const w = row.memory.worst;
  const best = row.memory.best;
  const engine = ENGINE[ctx.opts.engine];
  const engineVersion = calibrationFor(ctx.opts.engine).engineVersion;
  const exact = (gb: number, basis: string): Measure => measure(gb, gb, { kind: 'exact', basis });
  const weightsP = weightsPrecision(ctx.verification, engine);
  const memoryP = { kind: 'estimated' as const, basis: 'weights and KV cache exact + fixed costs estimated', reason: 'the fixed costs are a range' };
  const base = w.weightsMiB + w.kvPerSeqMiB;
  const blockers = row.check.blockers;
  const out: Figures = {
    hardware: hardwareLabel(row),
    servers: serversOf(layout),
    gpuId: layout.gpu.id,
    gpu: shortGpuName(layout.gpu.name),
    segment: layout.gpu.segment,
    count: layout.count,
    tp: layout.tp,
    pp: layout.pp,
    replicas: layout.replicas ?? 1,
    verdict: VERDICT_ID[row.verdict],
    memoryFits: best.verdict !== 'no',
    compat: { status: blockers.length > 0 ? 'does-not-start' : row.check.warnings.length > 0 ? 'warnings' : 'ok', notes: [...blockers, ...row.check.warnings] },
    // One scale everywhere: what CUDA reports on the smallest system, the memory the engine can use and the verdict uses.
    cardGiB: Math.round((layout.gpu.cudaMiB.min / GIB) * 10) / 10,
    memory: measure((base + best.fixed.total.min) / GIB, (base + w.fixed.total.max) / GIB, memoryP),
    engineGiB: measure(w.processMiB.min / GIB, best.processMiB.max / GIB, { kind: 'estimated', basis: `the engine's share (${ctx.opts.engine === 'sglang' ? '--mem-fraction-static' : '--gpu-memory-utilization'} × the CUDA memory), all of it reserved, plus what the process keeps outside it (CUDA context, communication buffers${ctx.opts.engine === 'sglang' ? ', activations and CUDA graphs' : ''}); what the share leaves after the weights and fixed costs is KV cache`, reason: 'the CUDA-visible memory is a range' }),
    weights: { ...exact(w.weightsMiB / GIB, weightsP.basis), precision: { ...weightsP, errorPct: 0 } },
    kv: exact(w.kvPerSeqMiB / GIB, `blocks and layer groups of the ${engine} ${engineVersion} KV cache`),
    fixed: measure(best.fixed.total.min / GIB, w.fixed.total.max / GIB, {
      kind: 'estimated',
      basis: 'measured vLLM runs and the engine code',
      reason: 'CUDA context, activations, CUDA graphs and workspaces: a measured range',
    }),
    margin: measure(row.marginMiB.min / GIB, row.marginMiB.max / GIB, memoryP),
    validity: row.speedBasis ? contextValidity(ctx, row.speedBasis) : 'speed not estimable',
  };
  if (row.speedNote) out.speedNote = row.speedNote;
  if (row.limitCause) out.limitCause = row.limitCause;
  const c = row.capacity;
  if (!c) return out;
  const decodeP = speedPrecision(ctx, row, c.atReliable.tokS);
  const prefillP = speedPrecision(ctx, row, c.single.fullPrefillMs);
  const asMeasure = (r: MiBRange, p: Omit<Precision, 'errorPct'>, scale = 1) => measure(r.min / scale, r.max / scale, p);
  const limitedBy = LIMITED_BY[c.limitedBy];
  out.requests = { guaranteed: c.reliable, likely: c.probable, memory: c.memoryTypical, limitedBy };
  const share = ctx.opts.workload.activeShare;
  const active = `${Math.round(share * 100)}% of users active at once`;
  const memErr = errorPct(w.kvRoomMiB.min, best.kvRoomMiB.max);
  // The same rule as the speed: a range of ±50% or more is not precise enough to decide on.
  const usersP: Precision =
    limitedBy === 'memory'
      ? memErr >= DECIDE_MAX_ERROR
        ? { kind: 'imprecise', errorPct: memErr, basis: 'KV cache memory limit (85% safety margin)', reason: `a range of ±${Math.round(memErr)}% is too wide to decide on; ${active}` }
        : { kind: 'estimated', errorPct: memErr, basis: 'KV cache memory limit (85% safety margin)', reason: active }
      : { ...decodeP, reason: [decodeP.reason, active].filter(Boolean).join('; ') };
  out.users = { guaranteed: c.totalUsers, likely: c.probableUsers, memoryMax: Math.floor(Math.floor(c.memoryTypical * ctx.opts.workload.safety) / share), precision: usersP };
  out.tokS = asMeasure(c.atReliable.tokS, decodeP);
  out.single = asMeasure(c.single.tokS, speedPrecision(ctx, row, c.single.tokS));
  out.read = asMeasure(c.single.fullPrefillMs, prefillP, 1000);
  out.wait = asMeasure(c.atReliable.ttftMs, speedPrecision(ctx, row, c.atReliable.ttftMs), 1000);
  // A row with a capacity has its physical limits (plan.ts).
  const limits = row.roofline as NonNullable<PlanRow['roofline']>;
  const tokSShare = out.single.value / limits.decodeTokS;
  const readShare = limits.fullPrefillMs / 1000 / out.read.value;
  const over = [tokSShare > ROOFLINE_MAX ? `generation = ${pctText(tokSShare)} of the datasheet bandwidth` : '', readShare > ROOFLINE_MAX ? `reading = ${pctText(readShare)} of the ${layout.gpu.tflopsDerived ? 'derived' : 'datasheet'} TFLOPS` : '']
    .filter(Boolean)
    .join(', ');
  out.roofline = {
    tokSLimit: limits.decodeTokS,
    tokSShare,
    readLimitS: limits.fullPrefillMs / 1000,
    readShare,
    ...(layout.gpu.tflopsDerived ? { tflopsDerived: true as const } : {}),
    ...(over ? { warning: `${over}: above ${pctText(ROOFLINE_MAX)} of the physical limit, suspect estimate` } : {}),
  };
  const busy = c.atReliable.prefillShare.max;
  out.readingShare = busy;
  if (users?.total !== undefined) {
    const plan = serverPlan(row, users.total, users.active ?? users.concurrent / users.total);
    if (plan) out.serversNeeded = plan.servers;
  }
  if (busy >= SATURATION) out.saturation = `near saturation: reading prompts takes ${pctText(busy)} of the GPU time, larger error`;
  else if (users && c.reliable > 0 && users.concurrent <= c.reliable && users.concurrent >= SATURATION * c.reliable)
    out.saturation = `near saturation: ${users.concurrent} concurrent requests of ${c.reliable} guaranteed (${pctText(users.concurrent / c.reliable)}), larger error`;
  return out;
}

/** A row that fits in memory but serves nobody at the target speed. */
export const servesNobody = (r: PlanRow) => r.capacity !== undefined && r.capacity.reliable < 1;

/** Rows grouped for "or …": data center against consumer and workstation cards. */
const market = (r: PlanRow) => (r.layout.gpu.segment === 'datacenter' ? 'dc' : 'desk');

export interface Choice {
  row: PlanRow;
  /** The first option of the other market (data center ↔ consumer / workstation), else of another card. */
  alternative?: PlanRow;
  /** --users: how many servers of `row` (and of `alternative`) serve them, and the users of one. */
  servers?: number;
  usersPerServer?: number;
  likelyUsersPerServer?: number;
  alternativeServers?: number;
  alternativeUsersPerServer?: number;
}

/**
 * Why a layout that fits serves nobody, one text for every place that says it (headline, card notes, --md): the
 * memory (no full-length request with its margin), the first-token wait, or the writing speed (`speed`: how the
 * minimum is written there, e.g. "20 tok/s").
 */
export function nobodyReason(by: 'memory' | 'speed' | 'wait' | 'none' | undefined, speed: string): string {
  if (by === 'memory') return 'the memory left after the model holds no full-length request with its safety margin';
  if (by === 'wait') return 'nobody gets the first words within the wait allowed (--max-wait)';
  return `nobody is guaranteed ${speed} of writing`;
}

/** "1 GPU", "8 GPUs". */
export const gpuCount = (n: number) => `${n} ${n === 1 ? 'GPU' : 'GPUs'}`;

/** The users one server serves, and the servers `users` take (plan.ts serverPlan): servers × users per server ≥ users. */
export interface UsersAsked {
  total: number;
  concurrent: number;
  /** Share active at once (--active); without it, concurrent ÷ total. */
  active?: number;
}
const activeOf = (u: UsersAsked) => u.active ?? u.concurrent / u.total;

export function serversNeeded(r: PlanRow, users: UsersAsked): ServerPlan | undefined {
  return serverPlan(r, users.total, activeOf(users));
}

/** The layouts that serve them, cheapest first: fewest GPUs in total, then fewest servers, then the planner's rank. */
export function byCost(rows: readonly PlanRow[], users: UsersAsked): { row: PlanRow; plan: ServerPlan }[] {
  const out: { row: PlanRow; plan: ServerPlan; gpus: number; i: number }[] = [];
  rows.forEach((row, i) => {
    const plan = serversNeeded(row, users);
    // One layout is serversOf(layout) servers: its GPUs spread over them.
    if (plan) out.push({ row, plan, gpus: (plan.servers * row.layout.count * pipelineWeight(row)) / serversOf(row.layout), i });
  });
  // A pipeline layout whose pipeline support the matrix does not confirm comes after every verified one; then the
  // fewest GPUs in all, fully supported first (m10), fewer servers, the planner's rank.
  const full = (r: PlanRow) => (fullySupported(r) ? 0 : 1);
  return out.sort((a, b) => pipelineUnverified(a.row) - pipelineUnverified(b.row) || a.gpus - b.gpus || full(a.row) - full(b.row) || a.plan.servers - b.plan.servers || a.i - b.i).map(({ row, plan }) => ({ row, plan }));
}

/** A layout the engine supports in full: no "partly supported" verdict of the matrix (open bugs, workarounds). */
export const fullySupported = (r: Pick<PlanRow, 'check'>) => !r.check.warnings.some((w) => w.startsWith('partly supported'));

/** The alternative to `row` among `ok` rows: the other market, else another card; a fully supported one first. */
function alternativeTo<T>(items: readonly T[], rowOf: (x: T) => PlanRow, first: PlanRow): T | undefined {
  const other = (x: T) => market(rowOf(x)) !== market(first);
  const card = (x: T) => rowOf(x).layout.gpu.id !== first.layout.gpu.id;
  const full = (x: T) => fullySupported(rowOf(x));
  return items.find((x) => other(x) && full(x)) ?? items.find((x) => card(x) && full(x)) ?? items.find(other) ?? items.find(card);
}

function choose(rows: readonly PlanRow[], ok: (r: PlanRow) => boolean): Choice | undefined {
  const first = rows.find(ok);
  if (!first) return undefined;
  // As useful (as many GPUs) and fully supported beats partly supported.
  const row = rows.find((r) => ok(r) && fullySupported(r) && r.layout.count === first.layout.count) ?? first;
  const alternative = alternativeTo(rows.filter(ok), (r) => r, row);
  return { row, ...(alternative ? { alternative } : {}) };
}

/** Nobody served anywhere: the closest layout (closestRow) and its alternative. */
function closestChoice(rows: readonly PlanRow[], ok: (r: PlanRow) => boolean): Choice | undefined {
  const row = closestRow(rows.filter(ok));
  if (!row) return undefined;
  const alternative = alternativeTo(rows.filter(ok), (r) => r, row);
  return { row, ...(alternative ? { alternative } : {}) };
}

/**
 * The general answer: the simplest option that fits and serves someone at a known speed (else one that fits with a
 * known speed, else any that fits, else a borderline one), with --users the one that serves them; when nothing fits,
 * the layout the engine refuses although the memory fits, or the closest attempt.
 */
export function generalAnswer(rows: readonly PlanRow[], users?: UsersAsked): { fit?: Choice; users?: Choice; attempt?: PlanRow; blocked?: PlanRow } {
  const fits = (r: PlanRow) => r.verdict === 'fits';
  const fit =
    choose(rows, (r) => fits(r) && r.capacity !== undefined && !servesNobody(r)) ??
    closestChoice(rows, (r) => fits(r) && r.capacity !== undefined) ??
    choose(rows, fits) ??
    choose(rows, (r) => r.verdict === 'borderline');
  const out: { fit?: Choice; users?: Choice; attempt?: PlanRow; blocked?: PlanRow } = {};
  const blocked = rows.find((r) => r.verdict === 'impossible' && r.memory.best.verdict !== 'no');
  if (fit) out.fit = fit;
  else if (blocked) out.blocked = blocked;
  else out.attempt = largestAttempt(rows);
  if (users) {
    const ranked = byCost(rows, users);
    const first = ranked[0];
    if (first) {
      const alt = alternativeTo(ranked, (x) => x.row, first.row);
      out.users = {
        row: first.row,
        servers: first.plan.servers,
        usersPerServer: first.plan.usersPerServer,
        likelyUsersPerServer: first.plan.likelyUsersPerServer,
        ...(alt ? { alternative: alt.row, alternativeServers: alt.plan.servers, alternativeUsersPerServer: alt.plan.usersPerServer } : {}),
      };
    }
  }
  return out;
}

/**
 * The layout closest to serving someone, when none does: the most likely users, then the fastest writing under
 * load (the summary's "Closest" and each card's row follow the same rule). Rows without a capacity come last.
 */
export function closestRow(rows: readonly PlanRow[]): PlanRow | undefined {
  const likely = (r: PlanRow) => r.capacity?.probableUsers ?? -1;
  const tokS = (r: PlanRow) => (r.capacity ? (r.capacity.atReliable.tokS.min + r.capacity.atReliable.tokS.max) / 2 : -1);
  return [...rows].sort((a, b) => pipelineUnverified(a) - pipelineUnverified(b) || likely(b) - likely(a) || tokS(b) - tokS(a))[0];
}

/** The layout closest to fitting: in one server (more servers are copies, they hold no more), the most GPUs, then the largest margin. */
export function largestAttempt(rows: readonly PlanRow[]): PlanRow {
  const one = rows.filter((r) => serversOf(r.layout) === 1);
  return [...(one.length > 0 ? one : rows)].sort((a, b) => b.layout.count - a.layout.count || b.marginMiB.max - a.marginMiB.max)[0] as PlanRow;
}

/**
 * One line per card (in the catalog's order) from the ranked rows: the fewest GPUs it fits on (with
 * --users: the fewest that serve them), the largest attempt when nothing fits, or the engine's reason
 * when only the engine's checks stand in the way.
 */
export function cardLines(ctx: FigureContext, rows: readonly PlanRow[], gpus: readonly Gpu[], users?: UsersAsked, localId?: string): { line: CardLine; row: PlanRow }[] {
  const out: { line: CardLine; row: PlanRow }[] = [];
  for (const gpu of gpus) {
    const mine = rows.filter((r) => r.layout.gpu.id === gpu.id);
    if (mine.length === 0) continue;
    const maxGpus = Math.max(...mine.map((r) => r.layout.count));
    // The most GPUs of this card in one server (more are copies on more servers): what "even with N" means.
    const tried = Math.min(maxGpus, serverSize(gpu));
    const fits = mine.filter((r) => r.verdict === 'fits');
    const minGpus = fits.length > 0 ? Math.min(...fits.map((r) => r.layout.count)) : undefined;
    let row: PlanRow;
    let status: CardStatus;
    let note: string | undefined;
    let servingGpus: number | undefined;
    let serversNeededHere: number | undefined;
    let perServerHere: { guaranteed: number; likely: number } | undefined;
    if (minGpus !== undefined) {
      // With --users: the cheapest layout of this card that serves them, with as many servers as it takes.
      const best = users ? byCost(fits, users)[0] : undefined;
      row = best?.row ?? (fits.find((r) => r.layout.count === minGpus) as PlanRow);
      status = 'fits';
      if (best) {
        serversNeededHere = best.plan.servers;
        perServerHere = { guaranteed: best.plan.usersPerServer, likely: best.plan.likelyUsersPerServer };
      }
      // Without a capacity the speed is unknown (the row's speedNote says why).
      if (!row.capacity) status = 'no-speed';
      else if (users && !best) {
        // Nobody served: the closest layout of this card, as the summary names it.
        row = closestRow(fits.filter((r) => r.capacity)) as PlanRow;
        status = 'short';
        note = `${nobodyReason((row.capacity as NonNullable<PlanRow['capacity']>).limitedBy, `${ctx.opts.workload.minTokS} tok/s`)}, even with ${gpuCount(tried)} in one server`;
      } else if (!users && servesNobody(row)) {
        const first = fits.find((r) => !servesNobody(r) && r.capacity);
        if (first) {
          // It serves someone with more GPUs: that layout is the row, the fewer-GPU fit (and why it serves nobody) in the note.
          const fewer = (row.capacity as NonNullable<PlanRow['capacity']>).limitedBy;
          row = first;
          servingGpus = first.layout.count;
          note = `with ${gpuCount(minGpus)} the model fits, but ${nobodyReason(fewer, `${ctx.opts.workload.minTokS} tok/s`)}: ${gpuCount(first.layout.count)} serve someone`;
        } else {
          row = closestRow(fits.filter((r) => r.capacity)) as PlanRow;
          status = 'slow';
          note = `${nobodyReason((row.capacity as NonNullable<PlanRow['capacity']>).limitedBy, `${ctx.opts.workload.minTokS} tok/s`)}, even with ${gpuCount(tried)} in one server`;
        }
      }
    } else {
      const edge = [...mine.filter((r) => r.verdict === 'borderline')].sort((a, b) => a.layout.count - b.layout.count || b.marginMiB.min - a.marginMiB.min)[0];
      const blocked = mine.find((r) => r.verdict === 'impossible' && r.memory.best.verdict !== 'no');
      if (edge) {
        row = edge;
        status = 'at-limit';
        // Both causes when both hold: the memory at its limit and nobody guaranteed the minimum speed.
        const slow = edge.capacity !== undefined && edge.capacity.reliable < 1 ? `; and ${nobodyReason(edge.capacity.limitedBy, `${ctx.opts.workload.minTokS} tok/s`)}` : '';
        // The sparse-attention reserve, when it is what puts the layout at the limit; else the memory's own reason.
        const why = edge.limitCause === 'sparse-reserve' ? SPARSE_RESERVE_REASON : (edge.memory.worst.reasons[0] ?? 'within the uncertainty of the fixed costs');
        note = `at the limit: ${why}${slow}`;
      } else if (blocked) {
        row = blocked;
        status = 'no-start';
        note = `fits in memory but the engine does not start: ${blocked.check.blockers[0] as string}`;
      } else {
        const tried = mine.filter((r) => r.verdict !== 'impossible');
        row = largestAttempt(tried.length > 0 ? tried : mine);
        status = 'no-fit';
      }
    }
    const line: CardLine = {
      gpuId: gpu.id,
      gpu: shortGpuName(gpu.name),
      status,
      ...(minGpus !== undefined ? { minGpus } : {}),
      maxGpus,
      ...(note ? { note } : {}),
      ...(servingGpus !== undefined ? { servingGpus } : {}),
      serverGpus: tried,
      serverSize: serverSize(gpu),
      ...(serversNeededHere !== undefined ? { serversNeeded: serversNeededHere } : {}),
      ...(perServerHere ? { usersPerServer: perServerHere } : {}),
      figures: figuresOf(ctx, row, users),
      ...(localId === gpu.id ? { local: true } : {}),
    };
    out.push({ line, row });
  }
  return out;
}

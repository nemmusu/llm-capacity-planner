/**
 * The capacity plan of one model on many hardware layouts: memory (exact weights and KV, fixed costs as a
 * range, on the card's CUDA-visible memory as a range), the layout's hard limits, speed and reliable users.
 * The memory verdict has three states and never says "fits" when the uncertainty could make the engine
 * refuse to start: "fits" holds with the largest fixed costs on the smallest memory, "does-not-fit" fails
 * even with the smallest costs on the largest memory, everything between is "borderline". Whether the engine
 * starts the model on the card at all (compat.ts) is kept apart from it: a row that does not start is
 * "impossible", with no speed and no users, and still says what the memory alone would give.
 */
import { kvElementBytes } from '../model/kv';
import { estimateServing, type MiBRange, type ServingEngine, type ServingEstimate } from '../model/serving-fit';
import type { KvLayout, ModelFactsEx } from '../model/types';
import { vllmKvPlan } from '../model/vllm-kv';
import { weightsPerGpu, type LoadShape, type TensorInfo, type WeightsPerGpu } from '../model/weights-exact';
import { capacity, type Capacity, type Workload } from './capacity';
import { speedBasis, speedCaveat, type SpeedBasis } from './calibration';
import { checkCompat, compatData, joinNeeds, needText, PIPELINE_FEATURE, type CompatResult } from './compat';
import { checkLayout, serverSize, type GpuLayout, type LayoutCheck, type ParallelShape } from './hardware';
import { kvReadBytes, type ModelProfile } from './model-profile';
import { decodeRooflineMs, decodeStepMs, prefillMs, prefillRooflineMs } from './perf';

const MIB = 1024 * 1024;

/** Everything the planner knows about a model (read once, used for every layout). */
export interface PlannerModel {
  facts: ModelFactsEx;
  layout: KvLayout;
  shape: LoadShape;
  tensors: readonly TensorInfo[];
  parallel: ParallelShape;
  profile: ModelProfile;
  /** Extra bytes per GPU a format may add after loading, measured but not yet explained (0 when verified). */
  formatGapShare: number;
  /** Fixed extra MiB per GPU the engine adds for this format × family (measured). */
  formatGapFixedMiB?: number;
  /** DeepSeek's sparse attention: keys each query reads (index_topk). */
  sparseTopk?: number;
  /** dense, moe, mla, hybrid, multimodal (what the speed calibration is verified for). */
  family?: string;
  /** The Hugging Face repo read, to find the checkpoint in the compatibility matrix (else the architecture class is used). */
  repo?: string;
}

export interface PlanOptions {
  engine: ServingEngine;
  /** 'fp8' (1 byte) or 'auto' (the model's bf16 / fp16). */
  kvCache: 'fp8' | 'auto';
  workload: Workload;
  /** --language-model-only: the vision tower is not loaded. */
  textOnly: boolean;
  util?: number;
  maxBatchedTokens?: number;
  maxSeqs?: number;
  /** Where the engine runs, for the compatibility matrix: Linux (default) or Windows through WSL 2. */
  platform?: 'linux' | 'windows-wsl';
}

export type Verdict = 'fits' | 'borderline' | 'does-not-fit' | 'impossible';

export interface PlanRow {
  layout: GpuLayout;
  check: LayoutCheck;
  /** Memory, the layout's limits and whether the engine starts the model on the card, together. */
  verdict: Verdict;
  /** Memory and the layout's limits alone, whether or not the engine starts the model. */
  memoryVerdict: Verdict;
  /** Whether the engine starts the model on this card (compat.ts); absent when the model is not in the matrix. */
  compat?: CompatResult;
  /** Memory with the smallest CUDA-visible memory (the verdict's worst case) and with the largest. */
  memory: { worst: ServingEstimate; best: ServingEstimate };
  weights: WeightsPerGpu;
  /** KV left after one full-context request, per GPU: worst … best. */
  marginMiB: MiBRange;
  capacity?: Capacity;
  /** Why the speed is missing or not to be trusted for this layout. */
  speedNote?: string;
  /** At the limit only because of the sparse-attention reserve (SPARSE_TRANSIENT_MIB): it fits otherwise. */
  limitCause?: 'sparse-reserve';
  /** What the speed rests on, its declared error and the context it is verified up to. */
  speedBasis?: SpeedBasis;
  /**
   * Physical limits (roofline, datasheet bandwidth and TFLOPS): one request's generation at the typical
   * context, tokens/s, and the time to read the full context, ms. No estimate passes them.
   */
  roofline?: { decodeTokS: number; fullPrefillMs: number };
}

const weightsMemo = new WeakMap<PlannerModel, Map<string, WeightsPerGpu>>();

/** Weights per GPU of a split, computed once per model (a large checkpoint has 100 000+ tensors). */
function weightsOf(model: PlannerModel, tp: number, pp: number, textOnly: boolean): WeightsPerGpu {
  const memo = weightsMemo.get(model) ?? new Map<string, WeightsPerGpu>();
  weightsMemo.set(model, memo);
  const key = `${tp}/${pp}/${textOnly}`;
  const hit = memo.get(key);
  if (hit) return hit;
  const w = weightsPerGpu(model.tensors, model.shape, { tp, pp, textOnly });
  memo.set(key, w);
  return w;
}

/**
 * The KV layout the fullest pipeline stage holds: ⌈layers / pp⌉ layers, shared among the layer kinds (attention
 * groups, recurrent layers) in proportion, by largest remainder, so a mixed stack is not rounded up kind by kind.
 */
export function stageLayout(layout: KvLayout, pp: number): KvLayout {
  if (pp <= 1) return layout;
  const counts = [...layout.groups.map((g) => g.layers), layout.recurrentLayers];
  const total = counts.reduce((x, y) => x + y, 0);
  if (total === 0) return layout;
  const stage = Math.ceil(total / pp);
  const exact = counts.map((n) => (n * stage) / total);
  const share = exact.map(Math.floor);
  const order = exact.map((x, i) => [x - Math.floor(x), i] as const).sort((x, y) => y[0] - x[0] || x[1] - y[1]);
  const missing = stage - share.reduce((x, y) => x + y, 0);
  for (let k = 0; k < missing; k++) {
    const i = (order[k] as readonly [number, number])[1];
    share[i] = (share[i] as number) + 1;
  }
  const recurrentLayers = share.at(-1) as number;
  return {
    ...layout,
    groups: layout.groups.map((g, i) => ({ ...g, layers: share[i] as number })),
    recurrentLayers,
    recurrentBytesPerSeq: layout.recurrentLayers > 0 ? (layout.recurrentBytesPerSeq * recurrentLayers) / layout.recurrentLayers : 0,
  };
}

/** One model on one layout (module comment). */
export function planLayout(model: PlannerModel, layout: GpuLayout, opts: PlanOptions): PlanRow {
  const { tp, pp, gpu } = layout;
  const layoutCheck = checkLayout(layout, model.parallel, opts.kvCache === 'fp8', opts.engine);
  const compat = compatOf(model, opts, gpu.id, pp > 1);
  const starts = compat?.status !== 'does-not-start';
  const blocked = starts ? layoutCheck : { ...layoutCheck, blockers: [...layoutCheck.blockers, startBlocker(opts.engine, compat as CompatResult)] };
  const sparseLong = model.sparseTopk !== undefined && opts.workload.maxContext > SPARSE_MEASURED;
  const extra = [...(sparseLong ? [SPARSE_LONG] : []), ...(pp > 1 && !compat ? [PP_NOT_VERIFIED] : [])];
  const check = extra.length ? { ...blocked, warnings: [...blocked.warnings, ...extra] } : blocked;
  const weights = weightsOf(model, tp, pp, opts.textOnly);
  const elem = kvElementBytes(opts.engine, opts.kvCache);
  const kv = stageLayout(model.layout, pp);
  const facts = opts.textOnly ? withoutVision(model.facts) : model.facts;
  const gap = model.formatGapShare * weights.perGpuBytes + (model.formatGapFixedMiB ?? 0) * MIB;
  const estimate = (totalMiB: number) =>
    estimateServing({
      facts,
      layout: kv,
      kvBytes: elem,
      weightsBytes: weights.checkpointBytes,
      weightsPerGpuBytes: weights.perGpuBytes,
      ...(weights.mlaAbsorbedBytes.max > 0 || gap > 0 ? { weightsExtraBytes: { min: weights.mlaAbsorbedBytes.min, max: weights.mlaAbsorbedBytes.max + gap } } : {}),
      engine: opts.engine,
      totalMiB,
      tokens: opts.workload.maxContext,
      tp,
      pp: layout.pp,
      ...(opts.util !== undefined ? { util: opts.util } : {}),
      ...(opts.maxBatchedTokens !== undefined ? { maxBatchedTokens: opts.maxBatchedTokens } : {}),
      ...(opts.maxSeqs !== undefined ? { maxSeqs: opts.maxSeqs } : {}),
    });
  const worst = estimate(gpu.cudaMiB.min);
  const best = estimate(gpu.cudaMiB.max);
  let memoryVerdict: Verdict;
  // Sparse attention beyond 128K: the transient buffer seen on the reference GPU must fit in what is left.
  const sparseReserve = sparseLong && worst.verdict === 'fits' && worst.marginMiB.min < SPARSE_TRANSIENT_MIB;
  if (layoutCheck.blockers.length > 0) memoryVerdict = 'impossible';
  else if (worst.verdict === 'fits') memoryVerdict = sparseReserve ? 'borderline' : 'fits';
  else if (best.verdict === 'no') memoryVerdict = 'does-not-fit';
  else memoryVerdict = 'borderline';
  const verdict: Verdict = starts ? memoryVerdict : 'impossible';
  const row: PlanRow = { layout, check, verdict, memoryVerdict, ...(compat ? { compat } : {}), memory: { worst, best }, weights, marginMiB: { min: worst.marginMiB.min, max: best.marginMiB.max }, ...(sparseReserve && memoryVerdict === 'borderline' ? { limitCause: 'sparse-reserve' as const } : {}) };
  const max = model.facts.contextLength;
  // Beyond the model's own maximum (no RoPE / YaRN override in its config) nothing is calibrated or measured.
  const beyond = max !== undefined && opts.workload.maxContext > max ? `beyond the model's maximum context (${max.toLocaleString('en-US')} tokens): the engine needs a RoPE / YaRN override, speed not reliable` : undefined;
  const note = speedCaveat(model.parallel.weights, model.parallel.moe, gpu) ?? beyond;
  if (note) row.speedNote = note;
  const speedKnown = gpu.bandwidthGBs > 0 && gpu.tflops.bf16 !== null;
  if ((verdict === 'fits' || verdict === 'borderline') && speedKnown) {
    const basis = speedBasis(opts.engine, gpu, model.parallel.weights, model.family ?? 'dense', opts.workload.maxContext, layout.pp);
    row.speedBasis = basis;
    row.capacity = scaled(layoutCapacity(model, layout, opts, worst, elem, kv, basis.cal), layout.replicas ?? 1, opts.workload.activeShare);
    const kvRead = kvReadBytes(model.layout, elem, tp, opts.workload.typicalContext, model.sparseTopk);
    row.roofline = {
      decodeTokS: 1000 / decodeRooflineMs({ profile: model.profile, layout, batch: 1, kvBytesPerSeq: kvRead, weights: model.parallel.weights }),
      fullPrefillMs: prefillRooflineMs({ profile: model.profile, layout, tokens: opts.workload.maxContext, batchedTokens: worst.limits.batchedTokens, weights: model.parallel.weights }),
    };
  }
  return row;
}

/**
 * The model in the compatibility matrix: by the repo read, else by its architecture class (a fine-tune or a
 * re-quantization of a catalog model) in the weight format read. Undefined when the matrix does not know it.
 */
function compatOf(model: PlannerModel, opts: PlanOptions, gpu: string, pipeline: boolean): CompatResult | undefined {
  const models = compatData().models;
  const repo = model.repo?.toLowerCase();
  const byRepo = repo === undefined ? undefined : models.find((m) => m.variants.some((v) => v.repo.toLowerCase() === repo));
  const arch = model.parallel.architecture;
  const entry = byRepo ?? (arch === undefined ? undefined : models.find((m) => m.architectureClass === arch));
  if (!entry) return undefined;
  const query = { model: entry.id, engine: opts.engine, gpu, kvCache: opts.kvCache, platform: opts.platform ?? 'linux', ...(pipeline ? { pipeline } : {}) } as const;
  const r = checkCompat({ ...query, ...(byRepo ? { repo: model.repo as string } : { format: model.parallel.weights }) });
  if (byRepo || r.status !== 'unknown') return r;
  // A format the catalog lacks (a fine-tune or a re-quantization): when every catalog checkpoint of the model fails
  // to start on the card, this one does not either (DeepSeek-V4.1 on sm120).
  const all = entry.variants.map((v) => checkCompat({ ...query, repo: v.repo }));
  if (!all.every((c) => c.status === 'does-not-start')) return r;
  // Why: the needs that stop each checkpoint, once each (the matrix lists at least one checkpoint per model).
  const wsl = query.platform === 'windows-wsl';
  const why = [...new Set(all.flatMap((c) => c.details.filter((d) => (wsl ? d.wslStatus : d.status) === 'does-not-start').map((d) => needText(d, wsl))))];
  return { ...(all[0] as CompatResult), repo: null, reason: `no checkpoint of ${entry.id} in the catalog starts: ${joinNeeds(why)}` };
}

const ENGINE_TEXT = { vllm: 'vLLM', sglang: 'SGLang' } as const;

/** Longest context sparse attention (DeepSeek's indexer, GLM's DSA) was measured at without surprises, tokens. */
const SPARSE_MEASURED = 131_072;
/** What was seen beyond it (test/fixtures/rtx5090/speed-long.json, long-glm-5.3-flash-4l-262k). */
const SPARSE_LONG =
  'sparse attention beyond 128K tokens: reading a 262K prompt, vLLM 0.31 used about 21 GiB more GPU memory than it had profiled on the reference GPU; a layout "fits" only with that much room left on each GPU, else it is at the limit';
/** That transient buffer (21 GiB seen on the reference GPU), MiB. */
export const SPARSE_TRANSIENT_MIB = 21 * 1024;
/** Why a layout is at the limit when only that reserve does not fit (limitCause 'sparse-reserve'). */
export const SPARSE_RESERVE_REASON = 'the request fits, but not the 21 GiB per GPU that sparse attention needs beyond 128K tokens while it reads a long prompt (seen on the reference GPU)';

/** The blocker of a model the engine does not start on the card. */
function startBlocker(engine: ServingEngine, c: CompatResult): string {
  // The reason carries each need's own workaround (compat.ts needText).
  return `does not start with ${ENGINE_TEXT[engine]} ${c.verifiedVersion} on this card${c.wslOnly ? ' under Windows / WSL 2' : ''}: ${c.reason}`;
}

/** Requests at the typical context that fit the KV room (vLLM: its block planner). */
function typicalFit(opts: PlanOptions, kv: KvLayout, elem: number, tp: number, roomMiB: number, batched: number, perSeqMiB: number): number {
  const typical = opts.workload.typicalContext;
  const plan = opts.engine === 'vllm' ? vllmKvPlan(kv, { elem, tp, batchedTokens: batched }) : undefined;
  if (plan) return Math.floor(plan.concurrency(Math.max(0, roomMiB) * MIB, typical));
  // SGLang: token slots, the full-context size scaled to the typical context.
  const per = (perSeqMiB * typical) / opts.workload.maxContext;
  return per > 0 ? Math.floor(Math.max(0, roomMiB) / per) : 0;
}

function layoutCapacity(model: PlannerModel, layout: GpuLayout, opts: PlanOptions, mem: ServingEstimate, elem: number, kv: KvLayout, cal: SpeedBasis['cal']): Capacity {
  const w = opts.workload;
  const kvRead = kvReadBytes(model.layout, elem, layout.tp, w.typicalContext, model.sparseTopk);
  const speed = {
    stepMs: (batch: number) => decodeStepMs({ profile: model.profile, layout, batch, kvBytesPerSeq: kvRead, weights: model.parallel.weights, cal }),
    prefillMs: (tokens: number) => prefillMs({ profile: model.profile, layout, tokens, batchedTokens: mem.limits.batchedTokens, weights: model.parallel.weights, cal }),
    batchedTokens: mem.limits.batchedTokens,
    requestOverheadMs: cal.requestOverheadMs,
    stages: layout.pp,
  };
  const typical = typicalFit(opts, kv, elem, layout.tp, mem.kvRoomMiB.min, mem.limits.batchedTokens, mem.kvPerSeqMiB);
  // The scheduler never runs more sequences than max_num_seqs.
  return capacity(speed, w, { worst: mem.concurrency.min, typical: Math.min(typical, mem.limits.maxSeqs) });
}

/**
 * Independent copies serve that many times the requests, each at the same speed. Users come from the concurrent
 * requests of all the copies together (floor(requests × copies ÷ active share)), not from a per-copy figure already
 * rounded down and then multiplied.
 */
function scaled(c: Capacity, replicas: number, activeShare: number): Capacity {
  if (replicas === 1) return c;
  const reliable = c.reliable * replicas;
  const probable = c.probable * replicas;
  return {
    ...c,
    memoryWorst: c.memoryWorst * replicas,
    memoryTypical: c.memoryTypical * replicas,
    speedLimit: c.speedLimit * replicas,
    reliable,
    probable,
    totalUsers: Math.floor(reliable / activeShare),
    probableUsers: Math.floor(probable / activeShare),
  };
}

/** Servers a layout occupies: its GPUs over the most cards of that model in one server. */
export function serversOf(layout: Pick<GpuLayout, 'gpu' | 'count'>): number {
  return Math.ceil(layout.count / serverSize(layout.gpu));
}

/** How many servers of a layout serve `users`, and how many users one of them serves. */
export interface ServerPlan {
  /** Servers in total: copies of the layout, each server independent, behind a load balancer. */
  servers: number;
  /** Users one server serves on the guaranteed basis: floor(guaranteed concurrent requests per server ÷ active share). */
  usersPerServer: number;
  /** The same on the likely basis. */
  likelyUsersPerServer: number;
}

/**
 * The servers that serve `users` with one layout, each an independent copy of the model behind a load balancer (no
 * traffic between servers, so users scale with the servers): servers = ceil(users ÷ users per server), and
 * servers × users per server ≥ users always. Undefined when the layout does not fit or serves nobody.
 */
export function serverPlan(row: Pick<PlanRow, 'verdict' | 'capacity' | 'layout'>, users: number, activeShare: number): ServerPlan | undefined {
  const c = row.capacity;
  if (row.verdict !== 'fits' || !c) return undefined;
  const perLayout = serversOf(row.layout);
  const usersPerServer = Math.floor(c.reliable / perLayout / activeShare);
  if (usersPerServer < 1) return undefined;
  return { servers: Math.ceil(users / usersPerServer), usersPerServer, likelyUsersPerServer: Math.floor(c.probable / perLayout / activeShare) };
}

function withoutVision(facts: ModelFactsEx): ModelFactsEx {
  const { visionEncoder: _, ...rest } = facts;
  return rest;
}

/**
 * How many GPUs a pipeline layout counts as when compared with tensor-parallel ones: its own, grown by its declared
 * writing-speed error (no pipeline run was measured), so pipeline stages win only by more than that error; 1 otherwise.
 */
export function pipelineWeight(r: Pick<PlanRow, 'layout' | 'speedBasis'>): number {
  return r.layout.pp > 1 ? 1 + (r.speedBasis?.decodePct ?? 0) / 100 : 1;
}

/** 1 for a pipeline layout the matrix does not confirm the engine supports for this model: never chosen over a verified one. */
export function pipelineUnverified(r: PlanRow): number {
  if (r.layout.pp <= 1) return 0;
  // A model outside the matrix: nobody checked that the engine implements its pipeline stages.
  if (!r.compat) return 1;
  return r.compat.details.some((d) => d.feature === PIPELINE_FEATURE && d.status !== 'ok') ? 1 : 0;
}

/** The warning of a pipeline layout for a model the compatibility matrix does not know. */
export const PP_NOT_VERIFIED = 'pipeline parallelism not verified for this model (it is not in the compatibility matrix): the engine may not implement its stages';

/** The order of verdicts: fits and serves at least one user at the target speed, fits, borderline… */
function tier(r: PlanRow): number {
  if (r.verdict === 'fits') return r.capacity && r.capacity.reliable < 1 ? 1 : 0;
  return { 'borderline': 2, 'does-not-fit': 3, impossible: 4 }[r.verdict];
}

/**
 * Rows in the order the output lists them: what fits and serves someone first, then the simplest (fewest GPUs), then
 * the least VRAM in total, then the most reliable users; among rows at the limit or that do not fit, the largest
 * worst-case margin (the one closest to fitting).
 */
export function rankRows(rows: readonly PlanRow[]): PlanRow[] {
  const vram = (r: PlanRow) => r.layout.count * r.layout.gpu.nominalGB * pipelineWeight(r);
  return [...rows].sort(
    (a, b) =>
      tier(a) - tier(b) ||
      pipelineUnverified(a) - pipelineUnverified(b) ||
      a.layout.count * pipelineWeight(a) - b.layout.count * pipelineWeight(b) ||
      vram(a) - vram(b) ||
      (b.capacity?.reliable ?? 0) - (a.capacity?.reliable ?? 0) ||
      (tier(a) >= 2 ? b.marginMiB.min - a.marginMiB.min : 0) ||
      a.layout.pp - b.layout.pp,
  );
}

/**
 * VRAM of vLLM and SGLang (the engines that reserve a share of the card up front), per GPU.
 *
 * What is exact and what is not:
 *  - weights: the bytes of the safetensors files, split over the tensor-parallel GPUs;
 *  - KV cache: from config.json (layers, KV heads per GPU = max(1, heads / TP), head sizes, sliding
 *    windows, recurrent layers, MLA latents replicated on every GPU), at the KV type's bytes;
 *  - fixed costs (CUDA context, NCCL, activations of the profile run, CUDA graphs, vision encoder): a
 *    min–max range from the model's shape (hidden size, FFN / expert sizes, vocabulary) and the
 *    scheduler limits (tokens per step, sequences), not a measurement.
 *
 * What each engine's share covers (read from their sources, vLLM ≥ 0.21 and SGLang ≥ 0.5.21):
 *  - vLLM `--gpu-memory-utilization` (default 0.9): the share util × total holds the weights, the
 *    activation peak of the profile run (max_num_batched_tokens tokens, the sampler's logits for
 *    max_num_seqs, the vision encoder), the CUDA graphs (profiled into the budget since v0.21) and
 *    the KV cache. The CUDA context and NCCL are set up before the share is measured: they sit
 *    outside it, and vLLM refuses to start unless the whole share is free at that point.
 *  - SGLang `--mem-fraction-static`: weights + KV pool = fraction × the memory free after the CUDA
 *    context and NCCL; activations and CUDA graphs use the rest. Its default fraction is the card minus
 *    a reserve that grows with the chunked-prefill size, the CUDA graph batch and TP.
 */
import type { KvLayout, ModelFactsEx } from './types';
import { kvHeadsPerGpu, vllmKvPlan } from './vllm-kv';

export { kvHeadsPerGpu } from './vllm-kv';

const MIB = 1024 * 1024;

/** A quantity known only within bounds, MiB. */
export interface MiBRange {
  min: number;
  max: number;
}

export type ServingEngine = 'vllm' | 'sglang';

export const VLLM_DEFAULT_GPU_UTIL = 0.9;
/** A fixed --mem-fraction-static some launchers use for SGLang instead of its own per-GPU default. */
export const SGLANG_APP_MEM_FRACTION = 0.85;
/** CUDA context + cuBLAS workspace of one GPU process: the upper end of its range. */
export const VLLM_NON_TORCH_MIB = 512;
/** vLLM allocates the KV cache in blocks of 16 tokens. */
export const VLLM_BLOCK_TOKENS = 16;

/** Defaults of the command line and of the estimate when a value is not given. */
export const SERVING_DEFAULTS = {
  /** No NVIDIA GPU found: an RTX 5090. */
  gpuMiB: 32 * 1024,
} as const;

/**
 * vLLM activation peak seen by its memory profiler when the model's shape is unknown: grows with the
 * model size (sampler, MLP). The upper end of the range; the lower end is half of it.
 */
export function vllmActivationMiB(paramsB: number | undefined): number {
  if (paramsB === undefined) return 2048;
  if (paramsB <= 4) return 1024;
  if (paramsB <= 16) return 1536;
  return 2048;
}


export interface KvSeqOptions {
  tp?: number;
  /** Tokens per cache block (vLLM 16, SGLang 1). */
  block?: number;
  /**
   * Tokens a sliding-window layer keeps beyond its window while a step is in flight (vLLM: the step's
   * tokens − 1, plus one block); 0 = exactly the window.
   */
  swaSlack?: number;
}

/**
 * KV bytes of one sequence of `len` tokens on one of `tp` GPUs. Sliding-window layers keep at most the
 * window (+ `swaSlack`); recurrent layers a fixed state split over the GPUs; an MLA latent is the same
 * on every GPU.
 */
export function servingKvBytesPerSeq(layout: KvLayout, len: number, elem: number, opts: KvSeqOptions = {}): number {
  const tp = opts.tp ?? 1;
  const block = opts.block ?? 1;
  let total = 0;
  for (const g of layout.groups) {
    const blocks = g.window ? Math.ceil(Math.min(len, g.window + (opts.swaSlack ?? 0)) / block) + (opts.swaSlack ? 1 : 0) : Math.ceil(len / block);
    const heads = layout.latent ? g.kvHeads : kvHeadsPerGpu(g.kvHeads, tp);
    total += g.layers * heads * (g.keyLength + g.valueLength) * blocks * block * elem;
  }
  return total + layout.recurrentBytesPerSeq / tp;
}

/** vLLM KV bytes for one sequence of `len` tokens on one GPU, SWA layers capped at the window. */
export function vllmKvBytesPerSeq(layout: KvLayout, len: number, elem: number): number {
  return servingKvBytesPerSeq(layout, len, elem);
}

export interface SchedulerLimits {
  /** Tokens of one forward step: vLLM max_num_batched_tokens, SGLang chunked_prefill_size. */
  batchedTokens: number;
  /** Sequences: vLLM max_num_seqs, SGLang's decode CUDA graph batch (cuda_graph_max_bs). */
  maxSeqs: number;
  /** CUDA graph sizes captured for decode, and (SGLang) for prefill. */
  decodeGraphs: number[];
  prefillGraphs: number[];
}

const range = (from: number, to: number, step: number) => Array.from({ length: Math.max(0, Math.floor((to - from) / step) + 1) }, (_, i) => from + i * step);

/** vLLM cudagraph_capture_sizes: 1, 2, 4, steps of 8 below 256, of 16 from 256, up to min(2 × max_num_seqs, 512, tokens per step). */
export function vllmGraphSizes(maxSeqs: number, batchedTokens: number): number[] {
  const max = Math.min(2 * maxSeqs, 512, batchedTokens);
  return [1, 2, 4, ...range(8, 248, 8), ...range(256, 512, 16)].filter((s) => s <= max);
}

/** SGLang decode CUDA graph batch sizes up to `maxBs` (cuda_graph_hook.py). */
export function sglangDecodeGraphSizes(maxBs: number): number[] {
  return [1, 2, 4, 8, 12, ...range(16, 256, 8), ...range(272, 496, 16), ...range(512, maxBs, 32)].filter((s) => s <= maxBs);
}

/** SGLang prefill CUDA graph token counts up to `maxTokens` (cuda_graph_hook.py). */
export function sglangPrefillGraphSizes(maxTokens: number): number[] {
  return [...range(4, 32, 4), ...range(48, 256, 16), ...range(288, 512, 32), ...range(576, 1024, 64), ...range(1280, 4096, 256), ...range(4608, maxTokens, 512)].filter(
    (s) => s <= maxTokens,
  );
}

/** Scheduler limits each engine picks for a card of `totalMiB` (vLLM arg_utils.py, SGLang memory_hook.py). */
export function schedulerDefaults(engine: ServingEngine, totalMiB: number, tp: number): SchedulerLimits {
  const gib = totalMiB / 1024;
  if (engine === 'vllm') {
    // OpenAI server: 16384 / 1024 from 160 GiB, 8192 / 1024 from 70 GiB (H100), else 2048 / 256.
    const [batchedTokens, maxSeqs] = gib >= 160 ? [16384, 1024] : gib >= 70 ? [8192, 1024] : [2048, 256];
    return { batchedTokens, maxSeqs, decodeGraphs: vllmGraphSizes(maxSeqs, batchedTokens), prefillGraphs: [] };
  }
  const small = tp < 4;
  let batchedTokens: number;
  let maxSeqs: number;
  if (gib < 20) [batchedTokens, maxSeqs] = [2048, 8];
  else if (gib < 35) [batchedTokens, maxSeqs] = [4096, small ? 48 : 160];
  else if (gib < 60) [batchedTokens, maxSeqs] = [4096, small ? 32 : 160];
  else if (gib < 160) [batchedTokens, maxSeqs] = [8192, small ? 256 : 512];
  else [batchedTokens, maxSeqs] = [16384, 512];
  return { batchedTokens, maxSeqs, decodeGraphs: sglangDecodeGraphSizes(maxSeqs), prefillGraphs: sglangPrefillGraphSizes(batchedTokens) };
}

/**
 * SGLang's own --mem-fraction-static (memory_hook.py): (card − reserve) / card, the reserve being
 * 512 MiB + 1.5 MiB per chunked-prefill token + TP / 8 GiB + 2 MiB per decode graph batch + 8 MiB per
 * prefill graph (1.5 GiB with MLA), at least 10 GiB on cards over 60 GiB; vision models less (ViT size).
 */
export function sglangDefaultMemFraction(totalMiB: number, tp: number, model: { latent?: boolean; vision?: { hiddenSize: number; layers: number } } = {}, limits?: SchedulerLimits): number {
  const s = limits ?? schedulerDefaults('sglang', totalMiB, tp);
  let reserve = 512 + Math.max(s.batchedTokens, 2048) * 1.5 + (tp / 8) * 1024 + Math.max(...s.decodeGraphs, 0) * 2 + (model.latent ? 1536 : s.prefillGraphs.length * 8);
  if (totalMiB > 60 * 1024) reserve = Math.max(reserve, 10 * 1024);
  let fraction = Math.round(((totalMiB - reserve) / totalMiB) * 1000) / 1000;
  if (model.vision) {
    const complexity = (model.vision.layers * model.vision.hiddenSize ** 2) / (24 * 1024 ** 2);
    fraction *= 0.95 * Math.max(0.8, Math.min(1.05, 1 - 0.1 * (complexity - 1)));
  }
  return fraction;
}

/** Bytes per token of the activations one forward step holds at its peak, on one GPU (bf16). */
function activationBytesPerToken(facts: ModelFactsEx, tp: number): number | undefined {
  const h = facts.embeddingLength;
  if (!h) return undefined;
  const ff = facts.feedForwardLength ?? 4 * h;
  // Dense layer: hidden + residual + QKV / attention output (~6 h) and the MLP's gate, up and activation (3 ff).
  const dense = 2 * (6 * h + (3 * ff) / tp);
  // MoE (fused kernel): per token, top-k × (gate+up, activation, down output) of one expert.
  const topk = facts.expertUsedCount;
  const ffe = facts.expertFeedForwardLength;
  const moe = topk && ffe ? 2 * (6 * h + topk * ((3 * ffe) / tp + h)) : 0;
  return Math.max(dense, moe);
}

export interface ServingInput {
  facts: ModelFactsEx;
  /** The exact KV layout (model/fit.ts kvLayoutOf). */
  layout: KvLayout;
  /** Bytes per KV element: 2 for bf16 / fp16 ('auto'), 1 for fp8. */
  kvBytes: number;
  /** Bytes of all weight files (every shard, every GPU together). */
  weightsBytes: number;
  /**
   * The weights one GPU holds, exact from the tensors vLLM loads (weights-exact.ts); without it, the
   * files' bytes divided by TP.
   */
  weightsPerGpuBytes?: number;
  /** Extra weight copies made after loading (MLA's absorbed kv_b_proj), per GPU, inside the share. */
  weightsExtraBytes?: { min: number; max: number };
  engine: ServingEngine;
  /** VRAM of each GPU, MiB. */
  totalMiB: number;
  /** Tokens of one request (max_model_len / --context-length). */
  tokens: number;
  tp?: number;
  /** Pipeline stages: each GPU holds 1/pp of the layers (its CUDA graphs too) and talks to the next stage. */
  pp?: number;
  /** --gpu-memory-utilization / --mem-fraction-static; default: vLLM 0.9, SGLang its own formula. */
  util?: number;
  /** Requests of `tokens` that should fit together. */
  seqs?: number;
  maxBatchedTokens?: number;
  maxSeqs?: number;
}

export interface ServingFixedCosts {
  /** MLA's absorbed kv_b_proj copies (bf16), made after loading; 0 otherwise. */
  weightsExtra?: MiBRange;
  /** CUDA context: created before either engine measures its share (outside it). */
  cudaContext: MiBRange;
  /**
   * Non-torch memory made while loading and profiling (cuBLAS / attention workspaces): inside the share.
   * Measured on an RTX 5090 with vLLM v0.31.0: 0.56 GiB ("consumed memory (weights + non-torch)").
   */
  workspaces: MiBRange;
  activations: MiBRange;
  vision: MiBRange;
  cudaGraphs: MiBRange;
  nccl: MiBRange;
  total: MiBRange;
}

export interface ServingEstimate {
  engine: ServingEngine;
  tp: number;
  util: number;
  /** `util` is the engine's default (not given). */
  utilDefault: boolean;
  totalMiB: number;
  /** The share (vLLM: util × total; SGLang: fraction × what is free after context and NCCL). */
  shareMiB: MiBRange;
  weightsMiB: number;
  kvPerSeqMiB: number;
  seqs: number;
  limits: { batchedTokens: number; maxSeqs: number; graphs: number; graphMaxBatch: number };
  fixed: ServingFixedCosts;
  /** Fixed costs inside the share (vLLM: activations, vision, CUDA graphs; SGLang: none). */
  insideFixedMiB: MiBRange;
  /** Room for the KV cache inside the share, MiB (min = with the largest fixed costs). */
  kvRoomMiB: MiBRange;
  /** Requests of `tokens` the KV room holds at once. */
  concurrency: MiBRange;
  /** What must fit outside the share (vLLM: context, NCCL; SGLang: activations, vision, CUDA graphs) and the room there. */
  outsideNeedMiB: MiBRange;
  outsideRoomMiB: MiBRange;
  /** VRAM the process holds on each GPU. */
  processMiB: MiBRange;
  /** KV room left after `seqs` requests (negative = missing). */
  marginMiB: MiBRange;
  verdict: 'fits' | 'uncertain' | 'no';
  /** Why the verdict is not "fits". */
  reasons: string[];
  notes: string[];
  /** The fixed costs come from the model's shape (false: fallback constants). */
  shaped: boolean;
}

const r = (min: number, max: number): MiBRange => ({ min: Math.round(min), max: Math.round(max) });
const add = (...rs: MiBRange[]): MiBRange => ({ min: rs.reduce((a, x) => a + x.min, 0), max: rs.reduce((a, x) => a + x.max, 0) });
const ZERO: MiBRange = { min: 0, max: 0 };
const larger = (a: MiBRange, b: MiBRange): MiBRange => ({ min: Math.max(a.min, b.min), max: Math.max(a.max, b.max) });

/** Fixed costs of one GPU as ranges (MiB), from the model's shape and the scheduler limits. */
export function servingFixedCosts(
  facts: ModelFactsEx,
  engine: ServingEngine,
  opts: { tp: number; limits: SchedulerLimits; layers: number; pp?: number },
): ServingFixedCosts & { shaped: boolean } {
  const { tp, limits } = opts;
  const pp = Math.max(1, opts.pp ?? 1);
  const cudaContext = r(250, 600);
  // Measured with vLLM 0.31 on an RTX 5090: 0.22 GiB (Qwen3-8B) to 0.56 GiB (Qwen3.8-27B).
  const workspaces = r(200, 800);
  const perToken = activationBytesPerToken(facts, tp);
  const vocab = facts.vocabSize;
  let activations: MiBRange;
  let shaped = false;
  if (perToken !== undefined && vocab) {
    shaped = true;
    // The profile run's peak: the forward at the step's tokens, or the fp32 logits it computes for every
    // one of those tokens (full vocabulary on every GPU), whichever is larger. Measured with vLLM 0.31 on
    // an RTX 5090: Qwen3-8B 1.24 GiB (logits 2048 × 151 936 × 4 B = 1.16 GiB), Qwen3.8-27B 1.91 GiB
    // (logits 1.89 GiB); with --enforce-eager only the forward (0.32 GiB at 2048 tokens).
    const fwd = (limits.batchedTokens * perToken) / MIB;
    const logits = (limits.batchedTokens * vocab * 4) / MIB;
    const peak = engine === 'vllm' ? Math.max(fwd, logits) : fwd + (Math.min(limits.maxSeqs, limits.batchedTokens) * vocab * 4) / MIB;
    activations = r(peak, peak * 1.1 + fwd * 0.5 + 64);
  } else {
    const top = vllmActivationMiB(facts.paramsB);
    activations = r(top / 2, top);
  }
  // Vision encoder: the profiler runs it on the largest image batch and keeps the embeddings.
  const v = facts.visionEncoder;
  // Measured: ~1.5 GiB for Qwen3.8's tower (1152 × 27) profiled at the default image budget.
  const vision = v ? r(128 + (v.hiddenSize * v.layers) / 64, 512 + (v.hiddenSize * v.layers) / 16) : ZERO;
  // CUDA graphs: one memory pool sized by the largest captured batch, plus per-graph bookkeeping (vLLM:
  // a piecewise graph per layer for each size plus full decode graphs; SGLang: one graph per decode
  // batch, and breakable prefill graphs whose non-torch part it counts as 8 MiB each).
  const graphMax = Math.max(...limits.decodeGraphs, 1);
  const graphAct = perToken !== undefined ? (graphMax * perToken) / MIB : 64;
  let cudaGraphs: MiBRange;
  if (engine === 'vllm') {
    const pieces = limits.decodeGraphs.length * (opts.layers + 2);
    // Measured: 0.04–0.10 GiB with 1–16 sequences (vLLM 0.31, RTX 5090).
    cudaGraphs = r(32 + pieces * 0.05 + graphAct, 256 + pieces * 0.3 + 2 * graphAct);
  } else {
    const n = limits.decodeGraphs.length;
    const prefill = limits.prefillGraphs.length * 8;
    cudaGraphs = r(64 + n * 4 + graphAct + prefill / 2, 192 + n * 12 + 2 * graphAct + prefill * 1.5);
  }
  // Tensor parallel: NCCL communicators and vLLM's custom all-reduce buffers (~35 MiB) on each GPU. Pipeline stages
  // send activations to the next one through a communicator of their own (the same range, not measured for PP).
  const nccl = add(tp > 1 ? r(100 + 35, 500 + 35 + 16 * tp) : ZERO, pp > 1 ? r(100, 500) : ZERO);
  const total = add(cudaContext, workspaces, activations, vision, cudaGraphs, nccl);
  return { cudaContext, workspaces, activations, vision, cudaGraphs, nccl, total, shaped };
}

/** The whole estimate for one launch (see the module comment for what each engine's share covers). */
export function estimateServing(input: ServingInput): ServingEstimate {
  const { facts, layout, engine, totalMiB } = input;
  const tp = Math.max(1, Math.floor(input.tp ?? 1));
  const seqs = Math.max(1, Math.floor(input.seqs ?? 1));
  const base = schedulerDefaults(engine, totalMiB, tp);
  const batchedTokens = input.maxBatchedTokens ?? base.batchedTokens;
  const maxSeqs = input.maxSeqs ?? base.maxSeqs;
  const limits: SchedulerLimits =
    engine === 'vllm'
      ? { batchedTokens, maxSeqs, decodeGraphs: vllmGraphSizes(maxSeqs, batchedTokens), prefillGraphs: [] }
      : { batchedTokens, maxSeqs, decodeGraphs: sglangDecodeGraphSizes(maxSeqs), prefillGraphs: sglangPrefillGraphSizes(batchedTokens) };
  const layers = facts.blockCount ?? layout.groups.reduce((a, g) => a + g.layers, 0) + layout.recurrentLayers;
  const utilDefault = input.util === undefined;
  const util =
    input.util ??
    (engine === 'vllm' ? VLLM_DEFAULT_GPU_UTIL : sglangDefaultMemFraction(totalMiB, tp, { ...(layout.latent ? { latent: true } : {}), ...(facts.visionEncoder ? { vision: facts.visionEncoder } : {}) }, limits));
  const weightsMiB = Math.round((input.weightsPerGpuBytes ?? input.weightsBytes / tp) / MIB);
  const extra = input.weightsExtraBytes ? r(input.weightsExtraBytes.min / MIB, input.weightsExtraBytes.max / MIB) : ZERO;
  // vLLM: its own KV planner (blocks, layer groups, recurrent pages), exact; SGLang: tokens × bytes (page size 1).
  const plan = engine === 'vllm' ? vllmKvPlan(layout, { elem: input.kvBytes, tp, batchedTokens }) : undefined;
  const kvOpts: KvSeqOptions = engine === 'vllm' ? { tp, block: VLLM_BLOCK_TOKENS, swaSlack: batchedTokens - 1 } : { tp, swaSlack: 0 };
  const kvPerSeqMiB = (plan ? plan.bytesPerRequest(input.tokens) : servingKvBytesPerSeq(layout, input.tokens, input.kvBytes, kvOpts)) / MIB;
  const pp = Math.max(1, Math.floor(input.pp ?? 1));
  // A stage captures CUDA graphs only for its own layers.
  const fixed = servingFixedCosts(facts, engine, { tp, limits, layers: Math.ceil(layers / pp), pp });
  const beforeShare = add(fixed.cudaContext, fixed.nccl);

  let shareMiB: MiBRange;
  let insideFixedMiB: MiBRange;
  let outsideNeedMiB: MiBRange;
  let outsideRoomMiB: MiBRange;
  if (engine === 'vllm') {
    const share = Math.ceil(util * totalMiB);
    shareMiB = { min: share, max: share };
    // The vision tower is profiled before the language model: the larger of the two peaks counts.
    insideFixedMiB = add(fixed.workspaces, larger(fixed.activations, fixed.vision), fixed.cudaGraphs, extra);
    outsideNeedMiB = beforeShare;
    outsideRoomMiB = { min: totalMiB - share, max: totalMiB - share };
  } else {
    // The fraction applies to what is free once the CUDA context and NCCL exist.
    const free = { min: totalMiB - beforeShare.max, max: totalMiB - beforeShare.min };
    shareMiB = r(util * free.min, util * free.max);
    // The KV pool also leaves room for the multimodal embedding cache (SGLANG_VLM_CACHE_SIZE_MB).
    insideFixedMiB = add(fixed.workspaces, facts.visionEncoder ? r(100, 100) : ZERO, extra);
    outsideNeedMiB = add(fixed.activations, fixed.vision, fixed.cudaGraphs);
    outsideRoomMiB = r(free.min - shareMiB.min, free.max - shareMiB.max);
  }
  const kvRoomMiB: MiBRange = { min: shareMiB.min - weightsMiB - insideFixedMiB.max, max: shareMiB.max - weightsMiB - insideFixedMiB.min };
  const fitting = (room: number) => (plan ? plan.concurrency(Math.max(0, room) * MIB, input.tokens) : Math.max(0, room) / kvPerSeqMiB);
  const concurrency: MiBRange = kvPerSeqMiB > 0 ? { min: Math.floor(fitting(kvRoomMiB.min)), max: Math.floor(fitting(kvRoomMiB.max)) } : { min: 0, max: 0 };
  const need = kvPerSeqMiB * seqs;
  const marginMiB: MiBRange = r(kvRoomMiB.min - need, kvRoomMiB.max - need);
  const processMiB: MiBRange =
    engine === 'vllm' ? add(shareMiB, beforeShare) : add(shareMiB, beforeShare, outsideNeedMiB);

  const reasons: string[] = [];
  let verdict: ServingEstimate['verdict'] = 'fits';
  const worse = (v: ServingEstimate['verdict']) => {
    if (v === 'no' || verdict === 'fits') verdict = v;
  };
  if (weightsMiB >= shareMiB.max) {
    worse('no');
    reasons.push(`the weights (${gbText(weightsMiB)}${tp > 1 ? ' per GPU' : ''}) do not fit the engine's share (${gbText(shareMiB.max)}): more GPUs (--tp), a larger card or quantized weights are needed`);
  } else if (marginMiB.max < 0) {
    worse('no');
    reasons.push(`the KV cache of ${seqs > 1 ? `${seqs} requests` : 'one request'} (${gbText(need)}) exceeds the room in the share even in the best case (${gbText(kvRoomMiB.max)})`);
  } else if (marginMiB.min < 0) {
    worse('uncertain');
    reasons.push(`the KV cache (${gbText(need)}) fits only if the fixed costs are at the low end of their range`);
  }
  const outsideWhat = engine === 'vllm' ? 'the CUDA context and NCCL' : 'activations and CUDA graphs';
  if (outsideNeedMiB.min > outsideRoomMiB.max) {
    worse('no');
    reasons.push(`outside the share ${outsideWhat} need at least ${gbText(outsideNeedMiB.min)} but only ${gbText(outsideRoomMiB.max)} is left: lower the share`);
  } else if (outsideNeedMiB.max > outsideRoomMiB.min) {
    worse('uncertain');
    reasons.push(`${gbText(outsideRoomMiB.min)} is left outside the share: enough for ${outsideWhat} only at the low end of their range`);
  }

  const notes: string[] = [];
  if (facts.contextLength !== undefined && input.tokens > facts.contextLength)
    notes.push(`${input.tokens} tokens exceed the model’s context (${facts.contextLength}): the engine refuses them without a RoPE override`);
  if (layout.groups.some((g) => g.window))
    notes.push(
      engine === 'vllm'
        ? `sliding-window layers: each holds the window + one step’s tokens (${batchedTokens}), not the whole context`
        : 'sliding-window layers: SGLang keeps them in a separate pool (swa_full_tokens_ratio), the concurrent requests are indicative',
    );
  if (layout.recurrentLayers > 0) notes.push(`${layout.recurrentLayers} recurrent layers: a fixed state per request, it does not grow with the tokens`);
  if (layout.latent && tp > 1) notes.push('MLA: the KV latent is replicated on every GPU, TP does not split it');
  if (!layout.latent && layout.groups.some((g) => g.kvHeads < tp)) notes.push(`fewer KV heads than GPUs: with TP ${tp} the heads are replicated`);
  if (engine === 'sglang' && utilDefault) notes.push(`--mem-fraction-static ${util.toFixed(3)} is what SGLang computes for this GPU`);
  if (engine === 'sglang' && input.kvBytes === 1) notes.push('in SGLang the fp8 cache is requested with --kv-cache-dtype fp8_e4m3 (or fp8_e5m2)');

  return {
    engine,
    tp,
    util,
    utilDefault,
    totalMiB,
    shareMiB,
    weightsMiB,
    kvPerSeqMiB,
    seqs,
    limits: { batchedTokens, maxSeqs, graphs: limits.decodeGraphs.length + limits.prefillGraphs.length, graphMaxBatch: Math.max(...limits.decodeGraphs, 0) },
    fixed: {
      ...(extra.max > 0 ? { weightsExtra: extra } : {}),
      cudaContext: fixed.cudaContext,
      workspaces: fixed.workspaces,
      activations: fixed.activations,
      vision: fixed.vision,
      cudaGraphs: fixed.cudaGraphs,
      nccl: fixed.nccl,
      total: add(fixed.total, extra),
    },
    insideFixedMiB,
    kvRoomMiB,
    concurrency,
    outsideNeedMiB,
    outsideRoomMiB,
    processMiB,
    marginMiB,
    verdict,
    reasons,
    notes,
    shaped: fixed.shaped,
  };
}

/** "12.3 GiB" (1 GiB = 1024 MiB, as nvidia-smi counts). */
export function gbText(mib: number): string {
  return `${(mib / 1024).toFixed(1)} GiB`;
}

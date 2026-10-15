/**
 * Speed of one GPU layout, from first principles with calibrated efficiencies (calibration.ts):
 *  - decode (generation) is bound by memory bandwidth: each step reads the active weights (MoE: the
 *    shared part + the experts the batch touches) and every sequence's KV cache (and recurrent state),
 *    plus a fixed per-step overhead and, with tensor parallel, two all-reduces per layer;
 *  - prefill (reading the context) is bound by compute: 2 × active parameters per token for the linear
 *    layers, plus attention, which grows with the square of the context (sliding windows cap it);
 *  - pipeline parallel runs the stages one after the other for a token, and overlaps them across
 *    requests.
 * Every result is a min–max range in milliseconds (best and worst calibrated efficiency).
 */
import type { MiBRange } from '../model/serving-fit';
import type { Calibration } from './calibration';
import { groupLink, tpLink, type GpuLayout, type ParallelShape } from './hardware';
import { distinctExperts, type ModelProfile } from './model-profile';

export type MsRange = MiBRange;

/** Dense TFLOPS a layout's linear layers run at for this weight format (falls back to bf16 kernels). */
export function linearTflops(layout: GpuLayout, weights: ParallelShape['weights']): number {
  const t = layout.gpu.tflops;
  const bf16 = t.bf16 ?? 100;
  if ((weights === 'nvfp4' || weights === 'mxfp4') && layout.gpu.cc >= 10 && t.fp4) return t.fp4;
  if (weights === 'fp8' && layout.gpu.cc >= 8.9 && t.fp8) return t.fp8;
  return bf16;
}

/** The calibration's scale on the linear layers' share of the peak for the format linearTflops picked. */
export function formatScale(layout: GpuLayout, weights: ParallelShape['weights'], cal: Calibration, of: 'formatMfuScale' | 'formatBwScale' = 'formatMfuScale'): number {
  const t = layout.gpu.tflops;
  if ((weights === 'nvfp4' || weights === 'mxfp4') && layout.gpu.cc >= 10 && t.fp4) return cal[of].fp4;
  if (weights === 'fp8' && layout.gpu.cc >= 8.9 && t.fp8) return cal[of].fp8;
  return 1;
}

/** All-reduce time of `bytes` over `tp` GPUs: per-call latency + ring traffic 2(tp−1)/tp × bytes at the link's speed. */
function allReduceMs(layout: GpuLayout, bytes: number, cal: Calibration, worst: boolean): number {
  if (layout.tp <= 1) return 0;
  // The pipeline stages of a TP × PP layout may cross PCIe; the TP group itself can stay on NVLink.
  const link = tpLink(layout);
  const lat = cal.allReduceLatencyUs[link.link] * (worst ? 1.5 : 1);
  return lat / 1000 + ((2 * (layout.tp - 1)) / layout.tp) * (bytes / (link.gbs * 1e9 * (worst ? cal.linkEff.min : cal.linkEff.max))) * 1000;
}

export interface DecodeInput {
  profile: ModelProfile;
  layout: GpuLayout;
  /** Sequences generating together. */
  batch: number;
  /** KV bytes one sequence reads per step on one GPU at its context (model-profile.ts kvReadBytes, with TP). */
  kvBytesPerSeq: number;
  weights: ParallelShape['weights'];
  cal: Calibration;
}

/** Time of one decode step (every sequence of the batch gets one token), ms. */
export function decodeStepMs(d: DecodeInput): MsRange {
  const { profile: p, layout, cal } = d;
  const batch = Math.max(1, d.batch);
  const { tp, pp } = layout;
  // With pipeline parallel, the batch is split into pp micro-batches that flow through the stages. vLLM never moves a
  // request already running to another micro-batch, so they balance only by chance: the best case is an even split,
  // the worst an upper estimate of the fullest one (microBatches).
  const micro = microBatches(batch, pp);
  // With stages, the slowest one sets the pace: its share of the layers plus the LM head (kept in BF16).
  const head = pp > 1 ? { bytes: Math.min(p.denseBytes, p.headParams * 2), params: p.headParams } : { bytes: 0, params: 0 };
  const bw = layout.gpu.bandwidthGBs * 1e9;
  const tf = linearTflops(layout, d.weights) * 1e12 * formatScale(layout, d.weights, cal);
  const layers = p.layers / pp;
  const stage = (worst: boolean) => {
    const m = worst ? micro.max : micro.min;
    const touched = p.experts > 0 ? distinctExperts(p.experts, p.expertsPerToken, m) / p.experts : 1;
    const weightBytes = (p.denseBytes - head.bytes + p.routedExpertBytes * touched) / tp / pp + head.bytes / tp;
    const kvBytes = (m * d.kvBytesPerSeq) / pp + (m * 2 * p.recurrentStateBytes) / tp / pp;
    const flops = (2 * (p.activeParams - head.params) * m) / tp / pp + (2 * head.params * m) / tp;
    const comm = (2 * p.layers * allReduceMs(layout, m * p.hidden * 2, cal, worst)) / pp;
    const w = (worst ? cal.decodeWeightEff.min : cal.decodeWeightEff.max) * formatScale(layout, d.weights, cal, 'formatBwScale');
    const k = worst ? cal.kvReadEff.min : cal.kvReadEff.max;
    const mem = (weightBytes / (bw * w) + kvBytes / (bw * k)) * 1000;
    const cmp = (flops / (tf * (worst ? cal.decodeMfu.min : cal.decodeMfu.max))) * 1000;
    const moe = p.experts > 0 ? (worst ? cal.moeLayerExtraUs.max : cal.moeLayerExtraUs.min) + touched * p.experts * (worst ? cal.moeExpertUs.max : cal.moeExpertUs.min) : 0;
    const perLayer = (worst ? cal.layerOverheadUs.max : cal.layerOverheadUs.min) + moe;
    const perSeq = m * (worst ? cal.seqLayerUs.max : cal.seqLayerUs.min);
    const launch = (layers * (perLayer + perSeq)) / 1000;
    return Math.max(mem, cmp) + launch + comm;
  };
  // A token crosses all stages; with pp micro-batches in flight every stage stays busy.
  const overhead = (worst: boolean) => (worst ? cal.stepOverheadMs.max : cal.stepOverheadMs.min);
  const pipeHop = (worst: boolean) => (pp > 1 ? (pp - 1) * (cal.allReduceLatencyUs[groupLink(layout).link] / 1000) * (worst ? 1.5 : 1) : 0);
  return widen({ min: pp * stage(false) + overhead(false) + pipeHop(false), max: pp * stage(true) + overhead(true) + pipeHop(true) }, cal.spread?.decode, decodeRooflineMs(d));
}

/**
 * Sequences in the fullest of pp micro-batches: an even split at best; at worst, since requests stay in the micro-batch
 * they started in, an upper estimate of the fullest of pp bins that batch requests fill at random (the usual bound
 * batch / pp + √(2 · batch / pp · ln pp), above its expected size), never more than the batch.
 */
export function microBatches(batch: number, pp: number): { min: number; max: number } {
  if (pp <= 1) return { min: batch, max: batch };
  const mean = batch / pp;
  return { min: Math.max(1, Math.ceil(mean)), max: Math.min(batch, Math.max(1, Math.ceil(mean + Math.sqrt(2 * mean * Math.log(pp))))) };
}

/**
 * Cards sustain clocks above the datasheet's boost clock (the RTX 5090 ran its SMs at 2.9 GHz against 2.41):
 * the compute floor allows that much more; memory bandwidth has no boost.
 */
export const BOOST_HEADROOM = 1.25;

/**
 * The physical floor of a decode step (roofline): the bytes it reads at the card's full datasheet bandwidth,
 * or its multiply-adds at the datasheet TFLOPS (with BOOST_HEADROOM), whichever is longer; no overhead. No
 * estimate is faster.
 */
export function decodeRooflineMs(d: Omit<DecodeInput, 'cal'>): number {
  const { profile: p, layout } = d;
  const batch = Math.max(1, d.batch);
  const { tp, pp } = layout;
  const micro = pp > 1 ? Math.max(1, Math.ceil(batch / pp)) : batch;
  const touched = p.experts > 0 ? distinctExperts(p.experts, p.expertsPerToken, micro) / p.experts : 1;
  const bytes = (p.denseBytes + p.routedExpertBytes * touched) / tp / pp + (micro * d.kvBytesPerSeq) / pp + (micro * 2 * p.recurrentStateBytes) / tp / pp;
  const flops = (2 * p.activeParams * micro) / tp / pp;
  const mem = (bytes / (layout.gpu.bandwidthGBs * 1e9)) * 1000;
  const cmp = (flops / (linearTflops(layout, d.weights) * 1e12 * BOOST_HEADROOM)) * 1000;
  return pp * Math.max(mem, cmp);
}

/** The physical floor of reading a prompt (roofline): its multiply-adds at the datasheet TFLOPS (with BOOST_HEADROOM). */
export function prefillRooflineMs(d: Omit<PrefillInput, 'cal'>): number {
  const { profile: p, layout, tokens } = d;
  const linear = (2 * ((p.activeParams - p.headParams) * tokens + p.headParams * Math.ceil(tokens / Math.max(1, d.batchedTokens)))) / layout.tp;
  const attention = attentionFlops(p, tokens) / layout.tp;
  return ((linear / (linearTflops(layout, d.weights) * 1e12) + attention / ((layout.gpu.tflops.bf16 ?? 100) * 1e12)) * 1000) / BOOST_HEADROOM;
}

export interface PrefillInput {
  profile: ModelProfile;
  layout: GpuLayout;
  /** Tokens to read (the prompt). */
  tokens: number;
  /** Tokens per prefill chunk (max_num_batched_tokens / chunked_prefill_size). */
  batchedTokens: number;
  weights: ParallelShape['weights'];
  cal: Calibration;
}

/** Attention multiply-adds ×2 of a causal prefill of `n` tokens: Σ layers × heads·(dk+dv) × n² (windows: 2·n·w). */
export function attentionFlops(profile: ModelProfile, n: number): number {
  let flops = 0;
  for (const a of profile.attention) {
    // Sliding windows and sparse attention (DeepSeek's index_topk) read at most `window` keys per query.
    const span = a.window !== undefined && n > a.window ? 2 * n * a.window - a.window * a.window : n * n;
    flops += a.layers * a.headDims * span;
  }
  return flops;
}

/** GPU time to read a prompt of `tokens` alone, ms (the first-token wait adds the request overhead). */
export function prefillMs(d: PrefillInput): MsRange {
  const { profile: p, layout, cal, tokens } = d;
  const { tp } = layout;
  // The LM head runs only for the sampled token of each chunk, not for every prompt token.
  const linear = (2 * ((p.activeParams - p.headParams) * tokens + p.headParams * Math.ceil(tokens / Math.max(1, d.batchedTokens)))) / tp;
  const attention = attentionFlops(p, tokens) / tp;
  const tfLinear = linearTflops(layout, d.weights) * 1e12 * formatScale(layout, d.weights, cal);
  const tfAttn = (layout.gpu.tflops.bf16 ?? 100) * 1e12;
  const attnEff = cal.prefillAttnEff(tokens);
  const comm = (worst: boolean) => 2 * p.layers * allReduceMs(layout, tokens * p.hidden * 2, cal, worst);
  // Chunked prefill reads the prompt in slices of the step's tokens, each a forward pass with its fixed
  // per-layer cost; a single request crosses the pipeline stages one after the other (no speed-up).
  const chunks = Math.ceil(tokens / Math.max(1, d.batchedTokens));
  const t = (worst: boolean) =>
    (linear / (tfLinear * (worst ? cal.prefillMfu.min : cal.prefillMfu.max)) + attention / (tfAttn * (worst ? attnEff.min : attnEff.max))) * 1000 +
    comm(worst) +
    (tokens * p.recurrentLayers * (worst ? (cal.recurrentTokenUs?.max ?? 0) : (cal.recurrentTokenUs?.min ?? 0))) / tp / 1000 +
    (chunks * p.layers * (worst ? cal.layerOverheadUs.max : cal.layerOverheadUs.min)) / 1000;
  return widen({ min: t(false), max: t(true) }, cal.spread?.prefill, prefillRooflineMs(d));
}

/** The fit error around the computed time (calibration.ts `spread`), never below the physical floor. */
function widen(r: MsRange, spread = 0, floor = 0): MsRange {
  return { min: Math.max(r.min * (1 - spread), floor), max: Math.max(r.max * (1 + spread), floor) };
}

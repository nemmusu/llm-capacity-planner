/**
 * vLLM v0.31's KV cache planner (v1/core/kv_cache_utils.py, kv_cache_interface.py), replicated:
 *  - every layer has a spec: full attention, sliding window, or a recurrent (Mamba / gated-delta) state;
 *  - pages are made uniform: the attention block (16 tokens) grows until one page holds a recurrent
 *    layer's state (rounded up to a multiple of 16 tokens), the smaller attention specs get larger
 *    blocks, and recurrent pages are padded to the attention page;
 *  - layers of the same spec form buckets, split into groups of `group_size` layers (the smallest
 *    bucket, or the largest when it is under 1.5× the smallest; short buckets padded); one block of
 *    the shared pool costs page × group_size;
 *  - a request needs, per group: full attention cdiv(len, block) blocks, sliding window
 *    cdiv(min(window − 1 + step tokens, len), block) + 1, a recurrent state 2 blocks with prefix
 *    caching ("align", the default) or 1;
 *  - start-up refuses a max_model_len whose one request needs more than the KV memory; the
 *    concurrency it prints is pool blocks ÷ blocks per request.
 * Checked on an RTX 5090 (Qwen3.8-27B-NVFP4, 16 attention + 48 gated-delta layers): the five
 * "X GiB KV cache is needed" errors and "Maximum concurrency … 10.00x" come out exactly.
 */
import type { KvLayout } from './types';

/** KV head count on one GPU: tensor parallelism splits the heads, replicating them when there are fewer than GPUs. */
export function kvHeadsPerGpu(kvHeads: number, tp: number): number {
  return Math.max(1, Math.floor(kvHeads / tp));
}


export const VLLM_BASE_BLOCK = 16;

export interface VllmKvOptions {
  /** Bytes per KV element (2 bf16/fp16, 1 fp8). */
  elem: number;
  tp: number;
  /** max_num_batched_tokens: the sliding-window layers keep one step beyond the window. */
  batchedTokens: number;
  /** Prefix caching on (vLLM's default): a recurrent layer keeps 2 states per request. */
  prefixCaching?: boolean;
}

interface Bucket {
  kind: 'full' | 'window' | 'state';
  layers: number;
  /** Attention: bytes per token per layer on one GPU. */
  perToken: number;
  window?: number;
  /** Recurrent: one state per layer on one GPU. */
  stateBytes?: number;
  blockTokens: number;
}

export interface VllmKvPlan {
  pageBytes: number;
  groupSize: number;
  /** Bytes of one block of the pool (page × group size). */
  blockBytes: number;
  buckets: { kind: Bucket['kind']; layers: number; groups: number; blockTokens: number; window?: number }[];
  /** Pool blocks one request of `len` tokens takes. */
  blocksPerRequest: (len: number) => number;
  /** Bytes vLLM requires for one request of `len` tokens (its start-up check). */
  bytesPerRequest: (len: number) => number;
  /** Requests of `len` tokens a pool of `bytes` holds (vLLM's "Maximum concurrency"). */
  concurrency: (bytes: number, len: number) => number;
  /** Padding layers added to even out the groups. */
  paddingLayers: number;
}

const cdiv = (a: number, b: number) => Math.ceil(a / b);
const alignUp = (a: number, b: number) => cdiv(a, b) * b;

/** The plan of a model's KV cache on one GPU (module comment); undefined without any cached layer. */
export function vllmKvPlan(layout: KvLayout, opts: VllmKvOptions): VllmKvPlan | undefined {
  const buckets: Bucket[] = [];
  for (const g of layout.groups) {
    const heads = layout.latent ? g.kvHeads : kvHeadsPerGpu(g.kvHeads, opts.tp);
    const perToken = heads * (g.keyLength + g.valueLength) * opts.elem;
    const kind = g.window ? 'window' : 'full';
    const same = buckets.find((b) => b.kind === kind && b.perToken === perToken && b.window === g.window);
    if (same) same.layers += g.layers;
    else buckets.push({ kind, layers: g.layers, perToken, blockTokens: VLLM_BASE_BLOCK, ...(g.window ? { window: g.window } : {}) });
  }
  const rs = layout.recurrentState;
  if (layout.recurrentLayers > 0 && rs) {
    const stateBytes = (rs.convElems * rs.convBytes + rs.ssmElems * rs.ssmBytes) / opts.tp;
    buckets.push({ kind: 'state', layers: layout.recurrentLayers, perToken: 0, stateBytes, blockTokens: 1 });
  }
  if (buckets.length === 0) return undefined;

  // Uniform page: the widest attention spec sets it (16 tokens, or enough tokens to hold a state).
  const attention = buckets.filter((b) => b.kind !== 'state');
  const widest = Math.max(...attention.map((b) => b.perToken), 0);
  const state = Math.max(...buckets.map((b) => b.stateBytes ?? 0));
  let pageBytes: number;
  if (widest === 0) pageBytes = state;
  else {
    const base = state > 0 ? alignUp(cdiv(state, widest), VLLM_BASE_BLOCK) : VLLM_BASE_BLOCK;
    pageBytes = base * widest;
  }
  for (const b of attention) b.blockTokens = pageBytes / b.perToken;

  const counts = buckets.map((b) => b.layers);
  const min = Math.min(...counts);
  const max = Math.max(...counts);
  const groupSize = max < min * 1.5 ? max : min;
  const plan = buckets.map((b) => ({ ...b, groups: cdiv(b.layers, groupSize) }));
  const paddingLayers = plan.reduce((a, b) => a + b.groups * groupSize - b.layers, 0);
  const blockBytes = pageBytes * groupSize;
  const stateBlocks = opts.prefixCaching === false ? 1 : 2;

  const blocksPerRequest = (len: number) =>
    plan.reduce((a, b) => {
      let blocks: number;
      if (b.kind === 'state') blocks = stateBlocks;
      else if (b.kind === 'window') blocks = cdiv(Math.min((b.window as number) - 1 + opts.batchedTokens, len), b.blockTokens) + 1;
      else blocks = cdiv(len, b.blockTokens);
      return a + b.groups * blocks;
    }, 0);
  return {
    pageBytes,
    groupSize,
    blockBytes,
    buckets: plan.map((b) => ({ kind: b.kind, layers: b.layers, groups: b.groups, blockTokens: b.blockTokens, ...(b.window ? { window: b.window } : {}) })),
    blocksPerRequest,
    bytesPerRequest: (len) => blocksPerRequest(len) * blockBytes,
    concurrency: (bytes, len) => Math.floor(bytes / blockBytes) / blocksPerRequest(len),
    paddingLayers,
  };
}

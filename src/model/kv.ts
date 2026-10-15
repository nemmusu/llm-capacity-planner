import type { ServingEngine } from './serving-fit';
import type { KvLayout, ModelFactsEx } from './types';

/** A uniform stack assumed when config.json leaves a value out. */
const FALLBACK = { layers: 32, kvHeads: 8, headDim: 128 };

/** KV cache bytes per element: 1 for the fp8 cache, 2 for 'auto' (the model's BF16 / FP16). Same rule for vLLM and SGLang. */
export function kvElementBytes(_engine: ServingEngine, type: 'fp8' | 'auto'): number {
  return type === 'fp8' ? 1 : 2;
}

/** The exact layout when config.json gave one, else a uniform stack built from the scalar facts. */
export function kvLayoutOf(facts: ModelFactsEx): { layout: KvLayout; assumed: boolean } {
  if (facts.kv) return { layout: facts.kv, assumed: false };
  const n = facts.blockCount ?? FALLBACK.layers;
  const kvHeads = facts.headCountKv ?? facts.headCount ?? FALLBACK.kvHeads;
  const k = facts.keyLength ?? (facts.embeddingLength && facts.headCount ? Math.floor(facts.embeddingLength / facts.headCount) : FALLBACK.headDim);
  const v = facts.valueLength ?? k;
  const swa = facts.slidingWindow ? Math.min(facts.slidingWindowLayers ?? 0, n) : 0;
  const full = Math.min(facts.fullAttentionLayers ?? n - swa, n - swa);
  const groups: KvLayout['groups'] = [];
  if (full > 0) groups.push({ layers: full, kvHeads, keyLength: k, valueLength: v });
  if (swa > 0) groups.push({ layers: swa, kvHeads, keyLength: k, valueLength: v, window: facts.slidingWindow as number });
  return { layout: { groups, recurrentLayers: 0, recurrentBytesPerSeq: 0 }, assumed: facts.blockCount === undefined };
}

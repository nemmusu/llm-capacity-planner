import type { AttentionGroup } from './types';

/** One decoder layer's attention, as config.json describes it. */
export interface LayerShape {
  attn: boolean;
  recurrent: boolean;
  kvHeads: number;
  k: number;
  v: number;
  window?: number;
}

/** Layers with the same KV heads, head sizes and window, counted together. */
export function groupLayers(shapes: LayerShape[]): AttentionGroup[] {
  const groups = new Map<string, AttentionGroup>();
  for (const s of shapes) {
    if (!s.attn) continue;
    const key = `${s.kvHeads}|${s.k}|${s.v}|${s.window ?? ''}`;
    const g = groups.get(key);
    if (g) g.layers++;
    else groups.set(key, { layers: 1, kvHeads: s.kvHeads, keyLength: s.k, valueLength: s.v, ...(s.window ? { window: s.window } : {}) });
  }
  return [...groups.values()];
}

/** Rounded to 3 decimals. */
export function round3(n: number): number {
  return Math.round(n * 1000) / 1000;
}

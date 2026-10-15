// Exported helpers of serving-fit.ts that the planner reaches only on some paths.
import { describe, expect, it } from 'vitest';
import { estimateServing, servingKvBytesPerSeq, vllmActivationMiB, vllmKvBytesPerSeq } from '../../src/model/serving-fit';

describe('vLLM helpers', () => {
  it('activation peak by model size when the shape is unknown', () => {
    expect(vllmActivationMiB(undefined)).toBe(2048);
    expect(vllmActivationMiB(3)).toBe(1024);
    expect(vllmActivationMiB(8)).toBe(1536);
    expect(vllmActivationMiB(70)).toBe(2048);
  });
  it('KV bytes per sequence: the serving formula with one-token blocks', () => {
    const layout = { groups: [{ layers: 2, kvHeads: 4, keyLength: 64, valueLength: 64 }], recurrentLayers: 0, recurrentBytesPerSeq: 0 };
    expect(vllmKvBytesPerSeq(layout, 10, 1)).toBe(servingKvBytesPerSeq(layout, 10, 1));
    expect(vllmKvBytesPerSeq(layout, 10, 1)).toBe(2 * 4 * 128 * 10);
  });
  it('without a layer count in the config, the layers of the KV layout', () => {
    const layout = { groups: [{ layers: 2, kvHeads: 4, keyLength: 64, valueLength: 64 }], recurrentLayers: 1, recurrentBytesPerSeq: 1024 };
    const withCount = estimateServing({ facts: { blockCount: 3 }, layout, kvBytes: 1, weightsBytes: 1e9, engine: 'vllm', totalMiB: 32_768, tokens: 1024 });
    const fromLayout = estimateServing({ facts: {}, layout, kvBytes: 1, weightsBytes: 1e9, engine: 'vllm', totalMiB: 32_768, tokens: 1024 });
    expect(fromLayout).toEqual(withCount);
  });
});

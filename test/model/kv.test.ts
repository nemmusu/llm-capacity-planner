import { describe, expect, it } from 'vitest';
import { AppError, errorMessage } from '../../src/contract/errors';
import { kvElementBytes, kvLayoutOf } from '../../src/model/kv';
import { groupLayers, round3 } from '../../src/model/layers';

describe('kvElementBytes', () => {
  it('1 byte for the fp8 cache, 2 for the model precision, on both engines', () => {
    expect(kvElementBytes('vllm', 'fp8')).toBe(1);
    expect(kvElementBytes('sglang', 'auto')).toBe(2);
  });
});

describe('kvLayoutOf', () => {
  it('the exact layout when config.json gave one', () => {
    const kv = { groups: [{ layers: 2, kvHeads: 4, keyLength: 64, valueLength: 64 }], recurrentLayers: 1, recurrentBytesPerSeq: 9 };
    expect(kvLayoutOf({ kv })).toEqual({ layout: kv, assumed: false });
  });
  it('a uniform stack from the scalar facts: full and sliding-window layers', () => {
    expect(kvLayoutOf({ blockCount: 10, headCountKv: 2, headCount: 8, keyLength: 128, valueLength: 64, slidingWindow: 512, slidingWindowLayers: 6 }).layout.groups).toEqual([
      { layers: 4, kvHeads: 2, keyLength: 128, valueLength: 64 },
      { layers: 6, kvHeads: 2, keyLength: 128, valueLength: 64, window: 512 },
    ]);
    // Head size from the hidden size; KV heads default to the attention heads; hybrid stacks count their attention layers.
    expect(kvLayoutOf({ blockCount: 4, headCount: 8, embeddingLength: 1024, fullAttentionLayers: 1 })).toEqual({
      layout: { groups: [{ layers: 1, kvHeads: 8, keyLength: 128, valueLength: 128 }], recurrentLayers: 0, recurrentBytesPerSeq: 0 },
      assumed: false,
    });
    // A window without the count of its layers: every layer is full attention.
    expect(kvLayoutOf({ blockCount: 2, slidingWindow: 64 }).layout.groups).toEqual([{ layers: 2, kvHeads: 8, keyLength: 128, valueLength: 128 }]);
    // Nothing known: a 32-layer stack, said to be assumed.
    expect(kvLayoutOf({})).toMatchObject({ assumed: true, layout: { groups: [{ layers: 32 }] } });
    // Every layer sliding-window.
    expect(kvLayoutOf({ blockCount: 2, slidingWindow: 64, slidingWindowLayers: 5 }).layout.groups).toEqual([{ layers: 2, kvHeads: 8, keyLength: 128, valueLength: 128, window: 64 }]);
  });
});

describe('groupLayers and round3', () => {
  it('counts layers with the same attention shape; recurrent-only layers have none', () => {
    expect(
      groupLayers([
        { attn: true, recurrent: false, kvHeads: 2, k: 64, v: 64 },
        { attn: true, recurrent: false, kvHeads: 2, k: 64, v: 64 },
        { attn: true, recurrent: false, kvHeads: 2, k: 64, v: 64, window: 128 },
        { attn: false, recurrent: true, kvHeads: 0, k: 0, v: 0 },
      ]),
    ).toEqual([
      { layers: 2, kvHeads: 2, keyLength: 64, valueLength: 64 },
      { layers: 1, kvHeads: 2, keyLength: 64, valueLength: 64, window: 128 },
    ]);
    expect(round3(1.23456)).toBe(1.235);
  });
});

describe('AppError', () => {
  it('keeps the code, params, detail and cause', () => {
    const cause = new Error('inner');
    const e = new AppError('NETWORK', 'down', { params: { status: 500 }, detail: 'x', cause });
    expect(e).toMatchObject({ name: 'AppError', code: 'NETWORK', params: { status: 500 }, detail: 'x', cause });
    expect(new AppError('VALIDATION', 'bad').cause).toBeUndefined();
    expect(errorMessage(cause)).toBe('inner');
    expect(errorMessage(42)).toBe('42');
  });
});

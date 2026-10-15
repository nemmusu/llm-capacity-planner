import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { kvLayoutOf } from '../../src/model/kv';
import { factsFromHfConfig } from '../../src/model/hf-config';
import type { KvLayout } from '../../src/model/types';
import { vllmKvPlan } from '../../src/model/vllm-kv';

const GiB = 1024 ** 3;
const facts = (name: string) => factsFromHfConfig(JSON.parse(readFileSync(path.join('test', 'fixtures', 'hf', 'configs', `${name}.json`), 'utf8')));
/** vLLM prints GiB with two decimals. */
const gib2 = (bytes: number) => Math.round((bytes / GiB) * 100) / 100;

describe('vLLM KV planner against real RTX 5090 logs (test/fixtures/rtx5090)', () => {
  const qwen = kvLayoutOf(facts('qwen3.8-27b-nvfp4')).layout;
  const log = readFileSync(path.join('test', 'fixtures', 'rtx5090', 'qwen3.8-27b-nvfp4.log'), 'utf8');

  it('Qwen3.8-27B: 16 attention layers (4 × 256 KV heads) and 48 gated-delta states (conv bf16, SSM f32)', () => {
    expect(qwen.groups).toEqual([{ layers: 16, kvHeads: 4, keyLength: 256, valueLength: 256 }]);
    expect(qwen.recurrentLayers).toBe(48);
    // conv: 3 × (2 × 16 × 128 + 48 × 128) = 30720 values; SSM: 48 × 128 × 128.
    expect(qwen.recurrentState).toEqual({ convElems: 30720, ssmElems: 786432, convBytes: 2, ssmBytes: 4 });
  });

  it('block of 784 tokens (bf16) / 1568 (fp8) so one page holds a 3.06 MiB state; groups of 16 layers', () => {
    const bf16 = vllmKvPlan(qwen, { elem: 2, tp: 1, batchedTokens: 2048 });
    expect(bf16).toMatchObject({ pageBytes: 784 * 4096, groupSize: 16, blockBytes: 16 * 784 * 4096, paddingLayers: 0 });
    expect(bf16?.buckets).toEqual([
      { kind: 'full', layers: 16, groups: 1, blockTokens: 784 },
      { kind: 'state', layers: 48, groups: 3, blockTokens: 1 },
    ]);
    expect(vllmKvPlan(qwen, { elem: 1, tp: 1, batchedTokens: 2048 })?.pageBytes).toBe(1568 * 2048);
  });

  it('"To serve at least one request with the model’s max seq len (N), (X GiB KV cache is needed…": all 13, bf16 or fp8 cache', () => {
    const errors = [...log.matchAll(/max seq len \((\d+)\), \(([\d.]+) GiB KV cache is needed/g)].map((m) => [Number(m[1]), Number(m[2])] as const);
    expect(errors).toHaveLength(13);
    const bf16 = vllmKvPlan(qwen, { elem: 2, tp: 1, batchedTokens: 2048 });
    const fp8 = vllmKvPlan(qwen, { elem: 1, tp: 1, batchedTokens: 2048 });
    // The tunes tried both caches: every message is exactly one of the two.
    for (const [len, gib] of errors) {
      const both = [gib2(bf16?.bytesPerRequest(len) as number), gib2(fp8?.bytesPerRequest(len) as number)];
      expect([len, both.filter((x) => x === gib).length]).toEqual([len, 1]);
    }
    expect(gib2(bf16?.bytesPerRequest(155648) as number)).toBe(9.81);
    expect(gib2(fp8?.bytesPerRequest(262144) as number)).toBe(8.33);
  });

  it('"Available KV cache memory: 6.24 GiB" → "Maximum concurrency for 5,120 tokens per request: 10.00x"', () => {
    expect(log).toContain('Available KV cache memory: 6.24 GiB');
    expect(log).toContain('Maximum concurrency for 5,120 tokens per request: 10.00x');
    const plan = vllmKvPlan(qwen, { elem: 2, tp: 1, batchedTokens: 2048 });
    // 130 blocks of 16 × 3 211 264 B; a request takes 7 attention blocks + 3 groups × 2 states.
    expect(plan?.blocksPerRequest(5120)).toBe(13);
    expect(plan?.concurrency(6.24 * GiB, 5120)).toBe(10);
  });
});

describe('vLLM KV planner: other layouts', () => {
  it('dense GQA: one group of every layer, blocks of 16', () => {
    const llama: KvLayout = { groups: [{ layers: 80, kvHeads: 8, keyLength: 128, valueLength: 128 }], recurrentLayers: 0, recurrentBytesPerSeq: 0 };
    const p = vllmKvPlan(llama, { elem: 2, tp: 4, batchedTokens: 2048 });
    expect(p).toMatchObject({ pageBytes: 16 * 2 * 256 * 2, groupSize: 80, paddingLayers: 0 });
    expect(p?.bytesPerRequest(1000)).toBe(63 * 16 * 80 * 2 * 256 * 2);
  });

  it('sliding window + full (Gemma 3: 52 + 10): groups of 10, the window layers padded to 60; Gemma 4: the wider SWA page sets the block', () => {
    const gemma3 = kvLayoutOf(facts('gemma-3-27b-it')).layout;
    const p = vllmKvPlan(gemma3, { elem: 2, tp: 1, batchedTokens: 2048 });
    expect(p).toMatchObject({ groupSize: 10, paddingLayers: 8 });
    // 1 full group: cdiv(32768, 16); 6 window groups: cdiv(min(1023 + 2048, 32768), 16) + 1.
    expect(p?.blocksPerRequest(32768)).toBe(2048 + 6 * (192 + 1));
    const gemma4 = vllmKvPlan(kvLayoutOf(facts('gemma-4-31b-it')).layout, { elem: 2, tp: 1, batchedTokens: 2048 });
    // SWA: 16 × 512 values per token; global: 4 × 1024, half as wide → blocks of 32 tokens.
    expect(gemma4?.buckets).toEqual([
      { kind: 'window', layers: 50, groups: 5, blockTokens: 16, window: 1024 },
      { kind: 'full', layers: 10, groups: 1, blockTokens: 32 },
    ]);
  });

  it('groups with the same bytes per token share one bucket (8 heads × 128 and 4 heads × 256)', () => {
    const mixed: KvLayout = {
      groups: [
        { layers: 3, kvHeads: 8, keyLength: 128, valueLength: 128 },
        { layers: 5, kvHeads: 4, keyLength: 256, valueLength: 256 },
      ],
      recurrentLayers: 0,
      recurrentBytesPerSeq: 0,
    };
    expect(vllmKvPlan(mixed, { elem: 2, tp: 1, batchedTokens: 2048 })?.buckets).toEqual([{ kind: 'full', layers: 8, groups: 1, blockTokens: 16 }]);
  });

  it('MLA keeps its latent on every GPU; 1 state per request without prefix caching; nothing cached → no plan', () => {
    const ds = kvLayoutOf(facts('deepseek-v3')).layout;
    expect(vllmKvPlan(ds, { elem: 2, tp: 8, batchedTokens: 8192 })?.pageBytes).toBe(16 * 576 * 2);
    const ssm: KvLayout = { groups: [], recurrentLayers: 4, recurrentBytesPerSeq: 0, recurrentState: { convElems: 10, ssmElems: 100, convBytes: 2, ssmBytes: 4 } };
    const p = vllmKvPlan(ssm, { elem: 2, tp: 2, batchedTokens: 2048, prefixCaching: false });
    expect(p).toMatchObject({ pageBytes: 210, groupSize: 4 });
    expect(p?.bytesPerRequest(100000)).toBe(210 * 4);
    expect(vllmKvPlan({ groups: [], recurrentLayers: 0, recurrentBytesPerSeq: 0 }, { elem: 2, tp: 1, batchedTokens: 2048 })).toBeUndefined();
    // Recurrent layers without a state shape (GGUF facts) are left out.
    expect(vllmKvPlan({ groups: [], recurrentLayers: 4, recurrentBytesPerSeq: 1 }, { elem: 2, tp: 1, batchedTokens: 2048 })).toBeUndefined();
  });
});

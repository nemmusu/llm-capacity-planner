import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { kvLayoutOf } from '../../src/model/kv';
import { factsFromHfConfig, loadShapeFromHfConfig } from '../../src/model/hf-config';
import type { KvLayout, ModelFactsEx } from '../../src/model/types';
import type { LoadShape, TensorInfo } from '../../src/model/weights-exact';
import { distinctExperts, kvReadBytes, modelProfile, tensorParams } from '../../src/planner/model-profile';

const readJson = (...p: string[]) => JSON.parse(readFileSync(path.join('test', 'fixtures', 'hf', ...p), 'utf8')) as unknown;
const header = (name: string): TensorInfo[] =>
  Object.entries((readJson('headers', `${name}.json`) as { tensors: Record<string, [string, number[]]> }).tensors).map(([n, [dtype, shape]]) => ({ name: n, dtype, shape }));
const t = (name: string, dtype: string, shape: number[]): TensorInfo => ({ name, dtype, shape });
const shape0: LoadShape = { layers: 2, mtpLayers: 0, tiedEmbeddings: false, rope: [] };
const noKv: KvLayout = { groups: [], recurrentLayers: 0, recurrentBytesPerSeq: 0 };

describe('tensorParams', () => {
  it('plain tensors: the product of the shape', () => {
    expect(tensorParams(t('a.weight', 'BF16', [4, 8]))).toBe(32);
    expect(tensorParams(t('a.qzeros', 'I32', [4, 8]))).toBe(32);
    expect(tensorParams(t('a.weight', 'I8', [4, 8]))).toBe(32);
    expect(tensorParams(t('a.weight_scale', 'U8', [4, 8]))).toBe(32);
  });

  it('packed 4-bit: eight per int32 qweight, two per byte for U8 weights / blocks and I8 blocks', () => {
    expect(tensorParams(t('a.qweight', 'I32', [4, 8]))).toBe(256);
    expect(tensorParams(t('a.weight', 'U8', [4, 8]))).toBe(64);
    expect(tensorParams(t('a.weight_packed', 'U8', [4, 8]))).toBe(64);
    expect(tensorParams(t('a.gate_up_proj_blocks', 'U8', [4, 8]))).toBe(64);
    expect(tensorParams(t('a.blocks', 'I8', [4, 8]))).toBe(64);
  });
});

describe('modelProfile', () => {
  it('Qwen3-8B (real header): every BF16 parameter, the input embedding not multiplied', () => {
    const config = readJson('configs', 'qwen3-8b.json');
    const facts = factsFromHfConfig(config);
    const shape = loadShapeFromHfConfig(config);
    const { layout } = kvLayoutOf(facts);
    const p = modelProfile(header('qwen3-8b'), shape, facts, layout);
    expect(p.totalParams).toBe(8_190_735_360);
    // The 151 936 × 4096 embedding is a lookup.
    expect(p.activeParams).toBe(8_190_735_360 - 151_936 * 4096);
    // A decode step reads one embedding row per token, not the table; the LM head is 151 936 × 4096.
    expect(p.denseBytes).toBe((8_190_735_360 - 151_936 * 4096) * 2);
    expect(p.headParams).toBe(151_936 * 4096);
    expect(p).toMatchObject({ layers: 36, hidden: 4096, experts: 0, expertsPerToken: 0, routedExpertBytes: 0, recurrentStateBytes: 0 });
    // 32 query heads × (128 + 128).
    expect(p.attention).toEqual([{ layers: 36, headDims: 32 * 256 }]);
  });

  it('dense synthetic: scales are bytes not parameters, MTP and vision skipped, unknown dtypes are 0 bytes', () => {
    const tensors = [
      t('model.embed_tokens.weight', 'BF16', [100, 8]),
      t('model.layers.0.self_attn.q_proj.weight', 'BF16', [8, 8]),
      t('model.layers.0.self_attn.q_proj.weight_scale', 'F32', [2]),
      t('model.layers.1.mlp.down_proj.weight', 'X9', [8, 8]),
      t('model.layers.2.mlp.down_proj.weight', 'BF16', [8, 8]),
      t('vision_tower.blocks.0.weight', 'BF16', [8, 8]),
      t('lm_head.weight', 'BF16', [100, 8]),
    ];
    const shape = { ...shape0, mtpLayers: 1 };
    const p = modelProfile(tensors, shape, { headCount: 2 }, noKv);
    // 800 + 64 + 64 (unknown dtype: parameters, 0 bytes) + 800; the scale counts 0.
    expect(p.totalParams).toBe(1728);
    expect(p.activeParams).toBe(1728 - 800);
    // (64 + 800) × 2 B + 2 × 4 B of scale: the embedding table is not read by a decode step.
    expect(p.denseBytes).toBe(1728 + 8);
    expect(p.headParams).toBe(800);
    expect(p.hidden).toBe(0);
    // Tied: the embedding is the LM head, multiplied and read whole.
    const tied = modelProfile(tensors.filter((x) => x.name !== 'lm_head.weight'), { ...shape, tiedEmbeddings: true }, {}, noKv);
    expect(tied).toMatchObject({ activeParams: 928, headParams: 800, denseBytes: 1728 + 8 });
  });

  it('MoE: per-expert and fused routed tensors, shared experts stay dense, top-k of the routed parameters active', () => {
    const tensors = [
      ...[0, 1, 2, 3].map((e) => t(`model.layers.0.mlp.experts.${e}.up_proj.weight`, 'BF16', [16, 8])),
      // Fused: first dimension = 4 experts, no ".experts." in the name.
      t('model.layers.1.block_sparse_moe.w1.weight', 'BF16', [4, 16, 8]),
      // A shared expert stays dense even when shaped like a fused one.
      t('model.layers.0.mlp.shared_expert.up_proj.weight', 'BF16', [4, 4, 8]),
      t('model.layers.0.self_attn.o_proj.weight', 'BF16', [8, 8]),
    ];
    const facts: ModelFactsEx = { expertCount: 4, expertUsedCount: 2, embeddingLength: 8 };
    const p = modelProfile(tensors, shape0, facts, noKv);
    // Routed: 4 × 128 + 512 = 1024 parameters, 2048 B; dense: 128 + 64 = 192 parameters, 384 B.
    expect(p.routedExpertBytes).toBe(2048);
    expect(p.denseBytes).toBe(384);
    expect(p.totalParams).toBe(1216);
    // 192 + 1024 × 2 / 4.
    expect(p.activeParams).toBe(704);
    expect(p).toMatchObject({ experts: 4, expertsPerToken: 2, hidden: 8 });
    // Expert count without top-k: no routed share is added.
    expect(modelProfile(tensors, shape0, { expertCount: 4 }, noKv).activeParams).toBe(192);
  });

  it('attention per group: heads × (k + v), MLA dims, windows and sparse top-k', () => {
    const layout: KvLayout = {
      groups: [
        { layers: 10, kvHeads: 8, keyLength: 128, valueLength: 128 },
        { layers: 20, kvHeads: 8, keyLength: 128, valueLength: 128, window: 1024 },
      ],
      recurrentLayers: 0,
      recurrentBytesPerSeq: 0,
    };
    expect(modelProfile([], shape0, { headCount: 32 }, layout).attention).toEqual([
      { layers: 10, headDims: 32 * 256 },
      { layers: 20, headDims: 32 * 256, window: 1024 },
    ]);
    // No head count: the KV heads stand for them; a sparse top-k caps every group, the smaller cap wins.
    expect(modelProfile([], shape0, {}, layout, 512).attention).toEqual([
      { layers: 10, headDims: 8 * 256, window: 512 },
      { layers: 20, headDims: 8 * 256, window: 512 },
    ]);
    expect(modelProfile([], shape0, {}, layout, 2048).attention[1]?.window).toBe(1024);
    // DeepSeek-V3 MLA: 128 heads × (nope 128 + rope 576 − 512 + v 128) = 128 × 320.
    const mla: KvLayout = { groups: [{ layers: 61, kvHeads: 1, keyLength: 576, valueLength: 0 }], recurrentLayers: 0, recurrentBytesPerSeq: 0, latent: true };
    const mlaShape = { ...shape0, mla: { kvLoraRank: 512, qkNope: 128, vHead: 128 } };
    expect(modelProfile([], mlaShape, { headCount: 128 }, mla).attention).toEqual([{ layers: 61, headDims: 128 * 320 }]);
    // Latent without MLA dims: heads × key length.
    expect(modelProfile([], shape0, { headCount: 128 }, mla).attention).toEqual([{ layers: 61, headDims: 128 * 576 }]);
  });

  it('recurrent state per sequence: layers × (conv + ssm bytes)', () => {
    const layout: KvLayout = { groups: [], recurrentLayers: 36, recurrentBytesPerSeq: 0, recurrentState: { convElems: 100, ssmElems: 1000, convBytes: 2, ssmBytes: 4 } };
    // 36 × (100 × 2 + 1000 × 4) = 36 × 4200.
    expect(modelProfile([], shape0, {}, layout).recurrentStateBytes).toBe(151_200);
  });
});

describe('distinctExperts', () => {
  it('uniform routing: E × (1 − (1 − k/E)^batch)', () => {
    expect(distinctExperts(0, 8, 4)).toBe(0);
    expect(distinctExperts(128, 8, 1)).toBeCloseTo(8, 12);
    // 128 × (1 − (120/128)²) = 128 × 0.12109375 = 15.5.
    expect(distinctExperts(128, 8, 2)).toBeCloseTo(15.5, 12);
    expect(distinctExperts(128, 8, 10_000)).toBeCloseTo(128, 6);
  });
});

describe('kvReadBytes', () => {
  const layout: KvLayout = {
    groups: [
      { layers: 10, kvHeads: 8, keyLength: 128, valueLength: 128 },
      { layers: 20, kvHeads: 8, keyLength: 128, valueLength: 128, window: 1024 },
    ],
    recurrentLayers: 0,
    recurrentBytesPerSeq: 0,
  };

  it('heads split by TP, windows capped', () => {
    // TP 2: 4 heads × 256 × 2 B = 2048 B per token and layer; 10 layers × 4096 tokens + 20 × 1024.
    expect(kvReadBytes(layout, 2, 2, 4096)).toBe(2048 * (10 * 4096 + 20 * 1024));
    // Below the window both groups read the whole context.
    expect(kvReadBytes(layout, 1, 1, 512)).toBe(8 * 256 * 30 * 512);
  });

  it('sparse top-k caps every group; a latent cache is not split by TP', () => {
    expect(kvReadBytes(layout, 2, 2, 4096, 256)).toBe(2048 * 30 * 256);
    const mla: KvLayout = { groups: [{ layers: 61, kvHeads: 1, keyLength: 576, valueLength: 0 }], recurrentLayers: 0, recurrentBytesPerSeq: 0, latent: true };
    // 61 × 576 × 1 B × min(100 000, 2048) on each of 8 GPUs.
    expect(kvReadBytes(mla, 1, 8, 100_000, 2048)).toBe(61 * 576 * 2048);
  });
});

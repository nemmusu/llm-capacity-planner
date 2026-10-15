import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { factsFromHfConfig, loadShapeFromHfConfig } from '../../src/model/hf-config';
import type { LoadShape, TensorInfo } from '../../src/model/weights-exact';
import { indexerDimsOf, modelFamily, parallelShapeFrom, plannerModel, sparseTopkOf, weightFormat } from '../../src/planner/model';

const readJson = (...p: string[]) => JSON.parse(readFileSync(path.join('test', 'fixtures', 'hf', ...p), 'utf8')) as unknown;
const header = (name: string): TensorInfo[] =>
  Object.entries((readJson('headers', `${name}.json`) as { tensors: Record<string, [string, number[]]> }).tensors).map(([n, [dtype, shape]]) => ({ name: n, dtype, shape }));
const t = (name: string, dtype: string, shape: number[]): TensorInfo => ({ name, dtype, shape });
/** 20 000 B of bf16 embedding: every quantized format must hold more than a quarter of it (5000 B). */
const embed = t('model.embed_tokens.weight', 'BF16', [1000, 10]);
const shape0: LoadShape = { layers: 2, mtpLayers: 0, tiedEmbeddings: false, rope: [] };

describe('weightFormat', () => {
  it('a real BF16 checkpoint (Qwen3-8B)', () => {
    expect(weightFormat(header('qwen3-8b'))).toBe('bf16');
    expect(weightFormat([])).toBe('bf16');
  });

  it('FP8 (E4M3 / E5M2), unless it is under a quarter of the bf16 bytes', () => {
    const fp8 = [embed, t('model.layers.0.mlp.up_proj.weight', 'F8_E4M3', [100, 50]), t('model.layers.0.mlp.down_proj.weight', 'F8_E5M2', [100, 50]), t('model.layers.0.mlp.up_proj.weight_scale_inv', 'F32', [1, 1])];
    // 10 000 B of FP8 > 5000.
    expect(weightFormat(fp8)).toBe('fp8');
    // 4000 B ≤ 5000: the bulk is bf16.
    expect(weightFormat([embed, t('model.layers.0.mlp.up_proj.weight', 'F8_E4M3', [40, 100])])).toBe('bf16');
  });

  it('NVFP4: U8 weights with a second-level or global scale', () => {
    expect(weightFormat([embed, t('l.0.up.weight', 'U8', [100, 60]), t('l.0.up.weight_scale_2', 'F32', [1])])).toBe('nvfp4');
    expect(weightFormat([embed, t('l.0.up.weight_packed', 'U8', [100, 60]), t('l.0.up.weight_global_scale', 'F32', [1])])).toBe('nvfp4');
  });

  it('MXFP4: U8 blocks (gpt-oss), I8 blocks, I8 weights with an E8M0 scale (DeepSeek V4)', () => {
    expect(weightFormat([embed, t('model.layers.0.mlp.experts.gate_up_proj_blocks', 'U8', [4, 100, 50]), t('model.layers.0.mlp.experts.gate_up_proj_scales', 'U8', [4, 100, 2])])).toBe('mxfp4');
    expect(weightFormat([embed, t('l.0.w1.blocks', 'I8', [100, 60])])).toBe('mxfp4');
    const v4 = [
      embed,
      t('layers.0.ffn.experts.0.w1.weight', 'I8', [100, 20]),
      t('layers.0.ffn.experts.0.w1.scale', 'F8_E8M0', [100, 1]),
      t('layers.0.ffn.experts.0.w2.weight', 'I8', [100, 20]),
      t('layers.0.ffn.experts.0.w2.weight_scale', 'F8_E8M0', [100, 1]),
      t('layers.0.ffn.experts.0.w3.weight', 'I8', [100, 20]),
      t('layers.0.ffn.experts.0.w3.scales', 'F8_E8M0', [100, 1]),
    ];
    expect(weightFormat(v4)).toBe('mxfp4');
  });

  it('INT4 (AWQ / GPTQ qweight, compressed-tensors int32 weight_packed) and INT8', () => {
    // 10 × 200 int32 = 8000 B.
    expect(weightFormat([embed, t('l.0.up.qweight', 'I32', [10, 200]), t('l.0.up.qzeros', 'I32', [1, 200])])).toBe('int4');
    expect(weightFormat([embed, t('l.0.up.weight_packed', 'I32', [10, 200])])).toBe('int4');
    // I8 without an E8M0 scale: W8A8.
    expect(weightFormat([embed, t('l.0.up.weight', 'I8', [100, 60]), t('l.0.up.weight_scale', 'F32', [100, 1])])).toBe('int8');
  });

  it('F16 and F32 count as unquantized; biases and unknown dtypes are not weights; the largest quantized format wins', () => {
    // F32 400 × 4 = 1600 B + F16 2000 × 2 = 4000 B → 5600 B unquantized (F64 and the bias left out);
    // FP8 1500 B > 5600 / 4 = 1400 → fp8.
    expect(weightFormat([t('a.weight', 'F32', [400]), t('b.weight', 'F16', [2000]), t('c.weight', 'F64', [100_000]), t('d.bias', 'F8_E4M3', [100_000]), t('e.weight', 'F8_E4M3', [1500])])).toBe('fp8');
    expect(weightFormat([embed, t('a.weight', 'F8_E4M3', [100, 100]), t('b.weight', 'I8', [100, 60])])).toBe('fp8');
  });
});

describe('modelFamily', () => {
  it('MLA, hybrid, MoE, multimodal (by its encoder or its tensors), dense', () => {
    expect(modelFamily({ expertCount: 256 }, { ...shape0, mla: { kvLoraRank: 512, qkNope: 128, vHead: 128 } }, [])).toBe('mla');
    expect(modelFamily({ expertCount: 512, kv: { groups: [], recurrentLayers: 36, recurrentBytesPerSeq: 1 } }, shape0, [])).toBe('hybrid');
    expect(modelFamily({ expertCount: 128, kv: { groups: [], recurrentLayers: 0, recurrentBytesPerSeq: 0 } }, shape0, [])).toBe('moe');
    expect(modelFamily({ visionEncoder: { hiddenSize: 1152, layers: 27 } }, shape0, [])).toBe('multimodal');
    expect(modelFamily({}, shape0, [t('model.visual.blocks.0.attn.qkv.weight', 'BF16', [1])])).toBe('multimodal');
    expect(modelFamily({}, shape0, [t('vision_tower.x.weight', 'BF16', [1])])).toBe('multimodal');
    expect(modelFamily({}, shape0, [embed])).toBe('dense');
  });
});

describe('parallelShapeFrom', () => {
  const real = (name: string) => {
    const c = readJson('configs', `${name}.json`);
    return { c, facts: factsFromHfConfig(c), shape: loadShapeFromHfConfig(c) };
  };

  it('DeepSeek-V3: MLA has one KV head per query head for TP, FP8 blocks of 128', () => {
    const { c, facts, shape } = real('deepseek-v3');
    expect(parallelShapeFrom(c, facts, shape, 'fp8')).toEqual({ heads: 128, kvHeads: 128, layers: 61, weights: 'fp8', moe: true, intermediate: 18_432, expertIntermediate: 2048, quantBlock: 128, architecture: 'DeepseekV3ForCausalLM' });
  });

  it('Qwen3-Next (top level) and Qwen3.8 (text_config): gated-delta heads', () => {
    const next = real('qwen3-next-80b-a3b-instruct');
    expect(parallelShapeFrom(next.c, next.facts, next.shape, 'bf16')).toMatchObject({ heads: 16, kvHeads: 2, linearKeyHeads: 16, linearValueHeads: 32, moe: true });
    const q38 = real('qwen3.8-27b-nvfp4');
    expect(parallelShapeFrom(q38.c, q38.facts, q38.shape, 'nvfp4')).toEqual({ heads: 24, kvHeads: 4, layers: 64, weights: 'nvfp4', moe: false, linearKeyHeads: 16, linearValueHeads: 48, intermediate: 17_408, architecture: 'Qwen3_5ForConditionalGeneration' });
  });

  it('quantization_config inside text_config; a block size that is not a list is ignored', () => {
    expect(parallelShapeFrom({ text_config: { quantization_config: { weight_block_size: [64, 64] } } }, {}, shape0, 'fp8').quantBlock).toBe(64);
    expect(parallelShapeFrom({ quantization_config: { weight_block_size: 128 } }, {}, shape0, 'fp8').quantBlock).toBeUndefined();
  });

  it('heads from the shape, else from the facts, else 1', () => {
    expect(parallelShapeFrom(null, {}, shape0, 'bf16')).toEqual({ heads: 1, kvHeads: 1, layers: 2, weights: 'bf16', moe: false });
    expect(parallelShapeFrom({}, { headCount: 40 }, { ...shape0, heads: 32 }, 'bf16')).toMatchObject({ heads: 32, kvHeads: 32 });
    expect(parallelShapeFrom({}, { headCount: 40 }, shape0, 'bf16')).toMatchObject({ heads: 40, kvHeads: 1 });
    expect(parallelShapeFrom({}, {}, { ...shape0, mla: { kvLoraRank: 512, qkNope: 128, vHead: 128 } }, 'bf16').kvHeads).toBe(1);
  });
});

describe('sparse attention fields', () => {
  it('index_topk at the top or in text_config', () => {
    expect(sparseTopkOf({ index_topk: 2048 })).toBe(2048);
    expect(sparseTopkOf({ index_topk: 5, text_config: { index_topk: 1024 } })).toBe(1024);
    expect(sparseTopkOf({})).toBeUndefined();
    expect(sparseTopkOf(null)).toBeUndefined();
  });

  it('indexer dims: heads × head dim, only with both', () => {
    expect(indexerDimsOf({ index_n_heads: 64, index_head_dim: 128 })).toBe(8192);
    expect(indexerDimsOf({ text_config: { index_n_heads: 32, index_head_dim: 64 } })).toBe(2048);
    expect(indexerDimsOf({ index_n_heads: 64 })).toBeUndefined();
    expect(indexerDimsOf(undefined)).toBeUndefined();
  });
});

describe('plannerModel', () => {
  it('Qwen3-8B: BF16 dense, verified within 0.8 %, no sparse attention', () => {
    const c = readJson('configs', 'qwen3-8b.json');
    const m = plannerModel(c, factsFromHfConfig(c), loadShapeFromHfConfig(c), header('qwen3-8b'));
    expect(m.family).toBe('dense');
    expect(m.parallel).toEqual({ heads: 32, kvHeads: 8, layers: 36, weights: 'bf16', moe: false, intermediate: 12_288, architecture: 'Qwen3ForCausalLM' });
    expect(m.verification.status).toBe('verified');
    expect(m.formatGapShare).toBe(0.008);
    expect('sparseTopk' in m).toBe(false);
    expect(m.profile.attention).toEqual([{ layers: 36, headDims: 32 * 256 }]);
  });

  it('sparse attention: keys capped at index_topk, the indexer scores every key as one more attention term', () => {
    const base = readJson('configs', 'qwen3-8b.json') as Record<string, unknown>;
    const c = { ...base, index_topk: 2048, index_n_heads: 64, index_head_dim: 128 };
    const m = plannerModel(c, factsFromHfConfig(base), loadShapeFromHfConfig(base), header('qwen3-8b'));
    expect(m.sparseTopk).toBe(2048);
    expect(m.profile.attention).toEqual([
      { layers: 36, headDims: 32 * 256, window: 2048 },
      // 64 × 128 per layer, no window.
      { layers: 36, headDims: 8192 },
    ]);
  });
});

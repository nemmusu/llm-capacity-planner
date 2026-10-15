import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { loadShapeFromHfConfig } from '../../src/model/hf-config';
import { tensorBytes, tensorRole, tensorsFromHeader, weightsPerGpu, type LoadShape, type TensorInfo } from '../../src/model/weights-exact';

/** Compact real headers (test/fixtures/hf/headers): { files: { name: { fileBytes, headerBytes } }, tensors: { name: [dtype, shape] } }. */
function real(name: string): { tensors: TensorInfo[]; tensorDataBytes: number } {
  const h = JSON.parse(readFileSync(path.join('test', 'fixtures', 'hf', 'headers', `${name}.json`), 'utf8')) as {
    files: Record<string, { fileBytes: number; headerBytes: number }>;
    tensors: Record<string, [string, number[]]>;
  };
  return {
    tensors: Object.entries(h.tensors).map(([n, [dtype, shape]]) => ({ name: n, dtype, shape })),
    // A safetensors file is 8 bytes of length, the JSON header, then the tensors' data.
    tensorDataBytes: Object.values(h.files).reduce((a, f) => a + f.fileBytes - f.headerBytes - 8, 0),
  };
}
const config = (name: string) => loadShapeFromHfConfig(JSON.parse(readFileSync(path.join('test', 'fixtures', 'hf', 'configs', `${name}.json`), 'utf8')));

describe('real checkpoints (headers read from Hugging Face)', () => {
  it('Qwen3-8B: every byte of the files is a tensor; norms replicated; K/V heads replicated beyond 8 GPUs', () => {
    const { tensors, tensorDataBytes } = real('qwen3-8b');
    const shape = config('qwen3-8b');
    expect(shape).toEqual({ layers: 36, mtpLayers: 0, tiedEmbeddings: false, rope: [{ positions: 40960, rotaryDim: 128 }], heads: 32, kvHeads: 8 });
    const one = weightsPerGpu(tensors, shape, { tp: 1 });
    expect(one.checkpointBytes).toBe(tensorDataBytes);
    expect(one.checkpointBytes).toBe(8_190_735_360 * 2);
    // RoPE cache: 40960 positions × 128 × 2 B.
    expect(one.ropeBytes).toBe(10_485_760);
    expect(one.perGpuBytes).toBe(16_381_470_720 + 10_485_760);
    // Norms: 36 × (4096 + 4096 + 128 + 128) + 4096 values × 2 B.
    expect(one.replicatedBytes).toBe(616_448);
    expect(weightsPerGpu(tensors, shape, { tp: 2 }).perGpuBytes).toBe((16_381_470_720 - 616_448) / 2 + 616_448 + 10_485_760);
    // TP 16: one of the 8 KV heads per GPU (K and V: 36 × 2 × 1024 × 4096 × 2 B in all), the rest ÷ 16.
    const kv = 36 * 2 * 1024 * 4096 * 2;
    expect(weightsPerGpu(tensors, shape, { tp: 16 }).perGpuBytes).toBe((16_381_470_720 - 616_448 - kv) / 16 + kv / 8 + 616_448 + 10_485_760);
  });

  it('Llama-3.1-8B: 8 030 261 248 BF16 parameters + a 131072-position RoPE cache', () => {
    const { tensors, tensorDataBytes } = real('llama-3.1-8b-instruct');
    const w = weightsPerGpu(tensors, config('meta-llama-3.1-8b-instruct'), { tp: 1 });
    expect(w.checkpointBytes).toBe(tensorDataBytes);
    expect(w.perGpuBytes).toBe(8_030_261_248 * 2 + 131072 * 128 * 2);
    expect(w.vocabPaddingBytes).toBe(0);
  });

  it('Gemma 3 27B: tied embeddings, the SigLIP tower counted apart and dropped with --language-model-only, two RoPE caches', () => {
    const { tensors, tensorDataBytes } = real('gemma-3-27b-it');
    const shape = config('gemma-3-27b-it');
    expect(shape.tiedEmbeddings).toBe(true);
    expect(shape.rope).toEqual([
      { positions: 131072, rotaryDim: 128 },
      { positions: 131072, rotaryDim: 128 },
    ]);
    const all = weightsPerGpu(tensors, shape, { tp: 1 });
    expect(all.checkpointBytes).toBe(tensorDataBytes);
    expect(all.visionBytes).toBeGreaterThan(800 * 1024 ** 2);
    const text = weightsPerGpu(tensors, shape, { tp: 1, textOnly: true });
    expect(text.perGpuBytes).toBe(all.perGpuBytes - all.visionBytes);
    expect(text.skippedBytes.vision).toBe(all.visionBytes);
    expect(text.visionBytesIfLoaded).toBe(all.visionBytes);
  });
});

describe('loading rules on synthetic checkpoints', () => {
  const t = (name: string, dtype: string, ...shape: number[]): TensorInfo => ({ name, dtype, shape });
  const base: LoadShape = { layers: 2, heads: 8, kvHeads: 2, mtpLayers: 0, tiedEmbeddings: false, rope: [] };

  it('vocabulary padded to 64 and split; a tied LM head is the embedding', () => {
    const tensors = [t('model.embed_tokens.weight', 'BF16', 1000, 16), t('lm_head.weight', 'BF16', 1000, 16)];
    const w = weightsPerGpu(tensors, base, { tp: 2 });
    expect(w.perGpuBytes).toBe(2 * (1024 / 2) * 16 * 2);
    expect(w.vocabPaddingBytes).toBe(2 * (24 / 2) * 16 * 2);
    expect(weightsPerGpu(tensors, { ...base, tiedEmbeddings: true }, { tp: 1 }).perGpuBytes).toBe(1024 * 16 * 2);
    // A shapeless (scalar) vocabulary tensor counts as one row.
    expect(weightsPerGpu([t('lm_head.weight', 'F32')], base, { tp: 1 }).perGpuBytes).toBe(64 * 4);
    // Mistral's consolidated layout names the head "output".
    expect(tensorRole('output.weight', base)).toBe('vocab');
    expect(tensorRole('model.layers.0.attention.output.weight', base)).toBe('split');
  });

  it('K / V and fused QKV: heads per GPU never below one', () => {
    const tensors = [t('model.layers.0.self_attn.k_proj.weight', 'BF16', 2 * 4, 32), t('model.layers.0.self_attn.qkv_proj.weight', 'BF16', (8 + 2 * 2) * 4, 32)];
    // TP 8 with 2 KV heads: one head per GPU (half of K); fused QKV: q ÷ 8 + kv × 1/2.
    const w = weightsPerGpu(tensors, base, { tp: 8 });
    const k = 8 * 32 * 2;
    const qkv = 48 * 32 * 2;
    expect(w.perGpuBytes).toBe(k / 2 + (qkv * (8 / 12)) / 8 + qkv * (4 / 12) * (1 / 2));
    // Without head counts: everything K/V is replicated when one head is all there is.
    expect(weightsPerGpu([tensors[0] as TensorInfo], { ...base, heads: undefined, kvHeads: undefined }, { tp: 4 }).perGpuBytes).toBe(k);
  });

  it('replicated: norms, row-parallel biases, routers, MLA down-projections; split: the rest', () => {
    const roles = {
      'model.layers.0.input_layernorm.weight': 'replicated',
      'model.norm.weight': 'replicated',
      'model.layers.0.self_attn.o_proj.bias': 'replicated',
      'model.layers.0.self_attn.q_proj.bias': 'split',
      'model.layers.0.mlp.gate.weight': 'replicated',
      'model.layers.0.mlp.gate.e_score_correction_bias': 'replicated',
      'model.layers.0.mlp.gate_proj.weight': 'split',
      'model.layers.0.self_attn.q_a_proj.weight': 'replicated',
      'model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale_inv': 'replicated',
      'model.layers.0.self_attn.kv_b_proj.weight': 'split',
      'model.layers.0.mlp.experts.7.down_proj.weight': 'split',
      'model.layers.0.block_sparse_moe.gate.weight': 'replicated',
      'model.layers.0.self_attn.v_proj.weight_scale': 'kv',
      'transformer.h.0.attn.c_attn.weight': 'qkv',
      'vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight': 'vision',
      'model.visual.blocks.0.attn.qkv.weight': 'vision',
      'mtp.layers.0.mlp.gate_proj.weight': 'mtp',
      'model.mtp.fc.weight': 'mtp',
    };
    for (const [name, role] of Object.entries(roles)) expect([name, tensorRole(name, base)]).toEqual([name, role]);
  });

  it('pipeline stages: layers as get_pp_indices splits them, embedding first, norm and head last, a tied head copied there', () => {
    const shape: LoadShape = { ...base, layers: 5, tiedEmbeddings: true };
    const tensors = [
      t('model.embed_tokens.weight', 'BF16', 64, 8),
      ...[0, 1, 2, 3, 4].map((i) => t(`model.layers.${i}.mlp.up_proj.weight`, 'BF16', 10, 10)),
      t('model.norm.weight', 'BF16', 8),
    ];
    // 5 layers over 2 stages: [2, 3]? get_pp_indices gives 5 // 2 = 2 each, the remainder to the stage before the last: [3, 2].
    const w = weightsPerGpu(tensors, shape, { tp: 1, pp: 2 });
    const embed = 64 * 8 * 2;
    expect(w.stageBytes).toEqual([embed + 3 * 200, 2 * 200 + 16 + embed]);
    expect(w.perGpuBytes).toBe(Math.max(...w.stageBytes));
  });

  it('MTP layers (num_nextn_predict_layers) are not loaded; MLA keeps bf16 copies of kv_b_proj', () => {
    const shape: LoadShape = { ...base, layers: 1, mtpLayers: 1, heads: 4, mla: { kvLoraRank: 8, qkNope: 4, vHead: 4 }, rope: [{ positions: 10, rotaryDim: 4 }] };
    const tensors = [t('model.layers.0.mlp.up_proj.weight', 'F8_E4M3', 64, 64), t('model.layers.0.mlp.up_proj.weight_scale_inv', 'F32', 1, 1), t('model.layers.1.mlp.up_proj.weight', 'F8_E4M3', 64, 64)];
    const w = weightsPerGpu(tensors, shape, { tp: 2 });
    expect(w.skippedBytes.mtp).toBe(64 * 64);
    expect(w.perGpuBytes).toBe((64 * 64 + 4) / 2 + 10 * 4 * 2);
    // 1 layer × 8 × (4 heads / 2) × (4 + 4) × 2 B, kept once or twice.
    expect(w.mlaAbsorbedBytes).toEqual({ min: 256, max: 512 });
    // On a pipeline stage only that stage's layers keep the copies: 4 layers in 2 stages, 2 each.
    const four = weightsPerGpu([], { ...shape, layers: 4, mtpLayers: 0 }, { tp: 2, pp: 2 });
    expect(four.mlaAbsorbedBytes).toEqual({ min: 2 * 256, max: 2 * 512 });
    // The same layer index without MTP in the config is a normal layer.
    expect(tensorRole('model.layers.1.mlp.up_proj.weight', { ...shape, mtpLayers: 0 })).toBe('split');
    // A vision tower's own "layers.N" is never mistaken for an MTP layer.
    expect(tensorRole('vision_tower.encoder.layers.5.mlp.fc1.weight', shape)).toBe('vision');
  });

  it('vision towers: split with replicated norms; unknown dtypes are reported, not counted', () => {
    const tensors = [t('vision_tower.layers.0.mlp.fc1.weight', 'BF16', 8, 8), t('vision_tower.layers.0.layer_norm1.weight', 'F32', 8), t('x.weight', 'F4', 4), t('y.weight', 'F4', 4)];
    const w = weightsPerGpu(tensors, base, { tp: 2 });
    expect(w.visionBytes).toBe(64 + 32);
    expect(w.unknownDtypes).toEqual(['F4']);
    expect(w.tensors).toBe(4);
    expect(tensorBytes({ dtype: 'F8_E8M0', shape: [3, 2] })).toBe(6);
    expect(tensorBytes({ dtype: 'F4', shape: [3] })).toBeUndefined();
  });
});

describe('loadShapeFromHfConfig', () => {
  it('YaRN covers original × factor positions; partial rotary; MLA rope dims; Gemma 4 global head; defaults', () => {
    const ds = config('deepseek-v3');
    expect(ds).toMatchObject({ layers: 61, mtpLayers: 1, tiedEmbeddings: false, mla: { kvLoraRank: 512, qkNope: 128, vHead: 128 }, rope: [{ positions: 163840, rotaryDim: 64 }] });
    expect(config('qwen3-next-80b-a3b-instruct').rope).toEqual([{ positions: 262144, rotaryDim: 64 }]);
    expect(config('gemma-4-31b-it').rope).toEqual([
      { positions: 262144, rotaryDim: 128 },
      { positions: 262144, rotaryDim: 256 },
    ]);
    expect(loadShapeFromHfConfig({ max_position_embeddings: 4096, head_dim: 64, rope_scaling: { rope_type: 'yarn', factor: 4, original_max_position_embeddings: 32768 } }).rope).toEqual([{ positions: 131072, rotaryDim: 64 }]);
    expect(loadShapeFromHfConfig({ hidden_size: 512, num_attention_heads: 8, max_position_embeddings: 100, rope_scaling: { type: 'yarn' } }).rope).toEqual([{ positions: 100, rotaryDim: 64 }]);
    expect(loadShapeFromHfConfig({ kv_lora_rank: 8, n_layer: 2 })).toEqual({ layers: 2, mtpLayers: 0, tiedEmbeddings: true, rope: [], mla: { kvLoraRank: 8, qkNope: 128, vHead: 128 } });
    expect(loadShapeFromHfConfig(null)).toEqual({ layers: 0, mtpLayers: 0, tiedEmbeddings: true, rope: [] });
    expect(loadShapeFromHfConfig({ num_layers: 3, n_head: 4, tie_word_embeddings: false }).layers).toBe(3);
  });
});

describe('tensorsFromHeader', () => {
  it('keeps tensors with a dtype and a shape, skips __metadata__ and junk', () => {
    expect(tensorsFromHeader({ __metadata__: { format: 'pt' }, a: { dtype: 'BF16', shape: [2, 3], data_offsets: [0, 12] }, b: null, c: { dtype: 1 }, d: { dtype: 'F32', shape: 'x' } })).toEqual([{ name: 'a', dtype: 'BF16', shape: [2, 3] }]);
    expect(tensorsFromHeader('x')).toEqual([]);
    expect(tensorsFromHeader(null)).toEqual([]);
  });
});

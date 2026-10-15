import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { chatTemplateOf, factsFromHfConfig, quantFromHfConfig, safetensorsParamCount } from '../../src/model/hf-config';
import { writeSafetensors } from '../helpers/safetensors';
import { tempDir } from '../helpers/safetensors';

const qwen3 = {
  architectures: ['Qwen3ForCausalLM'],
  model_type: 'qwen3',
  hidden_size: 1024,
  num_hidden_layers: 28,
  num_attention_heads: 16,
  num_key_value_heads: 8,
  head_dim: 128,
  max_position_embeddings: 40960,
  intermediate_size: 3072,
  vocab_size: 151936,
  torch_dtype: 'bfloat16',
  use_sliding_window: false,
  sliding_window: null,
};

describe('factsFromHfConfig', () => {
  it('maps a dense transformers config', () => {
    const f = factsFromHfConfig(qwen3, { name: 'Qwen/Qwen3-0.6B', parameterCount: 751632384, fileBytes: 1503300328 });
    expect(f).toMatchObject({
      architecture: 'Qwen3ForCausalLM',
      name: 'Qwen/Qwen3-0.6B',
      paramsB: 0.752,
      quant: 'BF16',
      bitsPerWeight: 16,
      blockCount: 28,
      headCount: 16,
      headCountKv: 8,
      keyLength: 128,
      valueLength: 128,
      contextLength: 40960,
      embeddingLength: 1024,
      vocabSize: 151936,
      feedForwardLength: 3072,
      fullAttentionLayers: 28,
    });
    expect(f.kv).toEqual({ groups: [{ layers: 28, kvHeads: 8, keyLength: 128, valueLength: 128 }], recurrentLayers: 0, recurrentBytesPerSeq: 0 });
  });

  it('a vision part of its own; the chat template of tokenizer_config.json or chat_template.jinja', () => {
    expect(factsFromHfConfig({ architectures: ['Qwen2_5_VLForConditionalGeneration'], vision_config: { depth: 32 } }).vision).toBe(true);
    expect(factsFromHfConfig({ vision_tower_config: {} }).vision).toBe(true);
    expect(factsFromHfConfig({ architectures: ['Qwen3ForCausalLM'] })).not.toHaveProperty('vision');
    expect(chatTemplateOf({ chat_template: '{% if enable_thinking %}' })).toBe('{% if enable_thinking %}');
    expect(chatTemplateOf({ chat_template: [{ name: 'default', template: 'a' }, { name: 'tool_use' }] })).toBe('a\n');
    expect(chatTemplateOf({}, '<think>')).toBe('<think>');
    expect(chatTemplateOf(undefined)).toBeUndefined();
  });

  it('reads text_config of multimodal checkpoints and layer_types for SWA', () => {
    const f = factsFromHfConfig({
      architectures: ['Gemma3ForConditionalGeneration'],
      text_config: {
        model_type: 'gemma3_text',
        hidden_size: 2560,
        num_hidden_layers: 6,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        head_dim: 256,
        sliding_window: 1024,
        layer_types: ['sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'chunked_attention', 'weird'],
        torch_dtype: 'float16',
      },
    });
    expect(f.architecture).toBe('Gemma3ForConditionalGeneration');
    expect(f.quant).toBe('F16');
    expect(f.slidingWindow).toBe(1024);
    expect(f.slidingWindowLayers).toBe(4);
    expect(f.fullAttentionLayers).toBe(2);
  });

  it('treats sliding layers without a window as full attention, and uses numeric SWA patterns', () => {
    const noWindow = factsFromHfConfig({ num_hidden_layers: 2, num_attention_heads: 4, hidden_size: 256, layer_types: ['sliding_attention', 'full_attention'] });
    expect(noWindow.fullAttentionLayers).toBe(2);
    expect(noWindow.headCountKv).toBe(4);
    expect(noWindow.keyLength).toBe(64);
    const pattern = factsFromHfConfig({ num_hidden_layers: 12, num_attention_heads: 4, sliding_window: 512, sliding_window_pattern: 6, model_type: 'gemma3' });
    expect(pattern.slidingWindowLayers).toBe(10);
    const disabled = factsFromHfConfig({ num_hidden_layers: 4, sliding_window: 4096, use_sliding_window: false, sliding_window_pattern: 2 });
    expect(disabled.slidingWindowLayers).toBeUndefined();
    // Chunked attention (Llama 4) uses attention_chunk_size as the window.
    const chunked = factsFromHfConfig({ num_hidden_layers: 2, attention_chunk_size: 8192, layer_types: ['chunked_attention', 'full_attention'] });
    expect(chunked.slidingWindow).toBe(8192);
  });

  it('handles hybrid linear-attention (Qwen3-Next) and Mamba stacks', () => {
    const next = factsFromHfConfig({
      hidden_size: 2048,
      num_hidden_layers: 8,
      num_attention_heads: 16,
      num_key_value_heads: 2,
      head_dim: 256,
      full_attention_interval: 4,
      linear_num_value_heads: 32,
      linear_num_key_heads: 16,
      linear_key_head_dim: 128,
      linear_value_head_dim: 128,
      linear_conv_kernel_dim: 4,
    });
    expect(next.fullAttentionLayers).toBe(2);
    expect(next.kv?.recurrentLayers).toBe(6);
    expect(next.kv?.recurrentBytesPerSeq).toBe(6 * 4 * (32 * 128 * 128 + 3 * (2 * 16 * 128 + 32 * 128)));

    const defaults = factsFromHfConfig({ num_hidden_layers: 2, full_attention_interval: 2, linear_num_value_heads: 4 });
    expect(defaults.kv?.recurrentBytesPerSeq).toBe(4 * (4 * 128 * 128 + 3 * (2 * 4 * 128 + 4 * 128)));

    const jamba = factsFromHfConfig({ hidden_size: 256, num_hidden_layers: 8, num_attention_heads: 4, attn_layer_period: 4, attn_layer_offset: 2, mamba_d_state: 16, mamba_d_conv: 4, mamba_expand: 2 });
    expect(jamba.fullAttentionLayers).toBe(2);
    const dInner = 512;
    expect(jamba.kv?.recurrentBytesPerSeq).toBe(6 * 4 * (dInner * 16 + 3 * (dInner + 2 * 16)));

    const nemotron = factsFromHfConfig({ hidden_size: 128, num_hidden_layers: 5, num_attention_heads: 4, hybrid_override_pattern: 'M-M*?', ssm_state_size: 8, conv_kernel: 2, expand: 1, n_groups: 2 });
    expect(nemotron.fullAttentionLayers).toBe(2);
    expect(nemotron.kv?.recurrentLayers).toBe(2);

    const shortPattern = factsFromHfConfig({ num_hidden_layers: 3, num_attention_heads: 2, hybrid_override_pattern: 'M' });
    expect(shortPattern.fullAttentionLayers).toBe(2);
    // Defaults when nothing describes the state: d_state 128, conv 4, expand 2 × hidden 4096.
    const bare = factsFromHfConfig({ num_hidden_layers: 2, attn_layer_period: 2 });
    expect(bare.kv?.recurrentBytesPerSeq).toBe(4 * (8192 * 128 + 3 * (8192 + 2 * 128)));
    const mambaInner = factsFromHfConfig({ num_hidden_layers: 2, attn_layer_period: 2, mamba_d_inner: 100, state_size: 10, mamba_n_groups: 1 });
    expect(mambaInner.kv?.recurrentBytesPerSeq).toBe(4 * (100 * 10 + 3 * (100 + 20)));
  });

  it('uses MLA latent sizes (DeepSeek) and expert counts', () => {
    const f = factsFromHfConfig({ num_hidden_layers: 2, num_attention_heads: 128, kv_lora_rank: 512, qk_rope_head_dim: 64, n_routed_experts: 256, num_experts_per_tok: 8, n_positions: 4096 });
    expect(f).toMatchObject({ headCountKv: 1, keyLength: 576, valueLength: 0, expertCount: 256, expertUsedCount: 8, contextLength: 4096 });
    const noRope = factsFromHfConfig({ num_hidden_layers: 1, kv_lora_rank: 512 });
    expect(noRope.keyLength).toBe(576);
  });

  it('reads the vision encoder (vision_config, Qwen2-VL style embed_dim / depth) and the expert FFN size', () => {
    expect(factsFromHfConfig({ num_hidden_layers: 1, vision_config: { embed_dim: 1280, depth: 32 }, expert_intermediate_size: 704 })).toMatchObject({
      visionEncoder: { hiddenSize: 1280, layers: 32 },
      expertFeedForwardLength: 704,
    });
    expect(factsFromHfConfig({ num_hidden_layers: 1, vision_config: { hidden_size: 1280 } }).visionEncoder).toBeUndefined();
  });

  it('survives junk and alternative key names', () => {
    expect(factsFromHfConfig(null)).toEqual({});
    expect(factsFromHfConfig('x')).toEqual({});
    const gpt2 = factsFromHfConfig({ model_type: 'gpt2', n_embd: 768, n_layer: 12, n_head: 12, seq_length: 1024, num_local_experts: 8 });
    expect(gpt2).toMatchObject({ architecture: 'gpt2', embeddingLength: 768, blockCount: 12, headCount: 12, keyLength: 64, contextLength: 1024, expertCount: 8 });
    const bare = factsFromHfConfig({ d_model: 512, num_layers: 2 });
    expect(bare.kv?.groups).toEqual([{ layers: 2, kvHeads: 8, keyLength: 128, valueLength: 128 }]);
  });
});

describe('quantFromHfConfig', () => {
  it('names quantization methods and dtypes', () => {
    expect(quantFromHfConfig({ quantization_config: { quant_method: 'awq', bits: 4 } })).toBe('AWQ-4bit');
    expect(quantFromHfConfig({ quantization_config: { quant_method: 'GPTQ', w_bit: 8 } })).toBe('GPTQ-8bit');
    expect(quantFromHfConfig({ quantization_config: { quant_method: 'gptq' } })).toBe('GPTQ-4bit');
    expect(quantFromHfConfig({ quantization_config: { quant_method: 'bitsandbytes', load_in_4bit: true } })).toBe('BNB-4bit');
    expect(quantFromHfConfig({ quantization_config: { quant_method: 'bitsandbytes' } })).toBe('BNB-8bit');
    expect(quantFromHfConfig({ quantization_config: { quant_method: 'fp8' } })).toBe('FP8');
    expect(quantFromHfConfig({ quantization_config: {}, dtype: 'float32' })).toBe('F32');
    expect(quantFromHfConfig({ torch_dtype: 'int3' })).toBeUndefined();
    expect(quantFromHfConfig({})).toBeUndefined();
  });
});

describe('safetensorsParamCount', () => {
  let t: ReturnType<typeof tempDir>;
  beforeEach(() => {
    t = tempDir('gb-st-');
  });
  afterEach(() => t.cleanup());

  it('sums tensor shapes over shards, ignoring __metadata__', async () => {
    const a = writeSafetensors(path.join(t.dir, 'model-00001-of-00002.safetensors'), { tensors: { a: { dtype: 'BF16', shape: [4, 8] }, s: { dtype: 'F32', shape: [] } }, metadata: { format: 'pt' } });
    const b = writeSafetensors(path.join(t.dir, 'model-00002-of-00002.safetensors'), { tensors: { b: { dtype: 'F16', shape: [10] } } });
    expect(await safetensorsParamCount([a, b])).toBe(32 + 1 + 10);
  });

  it('rejects broken headers', async () => {
    const bad = path.join(t.dir, 'bad.safetensors');
    writeFileSync(bad, Buffer.alloc(16));
    await expect(safetensorsParamCount([bad])).rejects.toMatchObject({ code: 'MODEL_INVALID' });
    const notJson = path.join(t.dir, 'nj.safetensors');
    const buf = Buffer.alloc(8 + 4);
    buf.writeBigUInt64LE(4n, 0);
    buf.write('{{{{', 8);
    writeFileSync(notJson, buf);
    await expect(safetensorsParamCount([notJson])).rejects.toThrow();
    // A tensor entry without a shape counts as a scalar.
    const json = '{"x":{"dtype":"F32","data_offsets":[0,4]}}';
    const noShape = path.join(t.dir, 'ns.safetensors');
    const b2 = Buffer.alloc(8 + json.length + 4);
    b2.writeBigUInt64LE(BigInt(json.length), 0);
    b2.write(json, 8);
    writeFileSync(noShape, b2);
    expect(await safetensorsParamCount([noShape])).toBe(1);
  });
});

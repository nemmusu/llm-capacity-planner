import { describe, expect, it } from 'vitest';
import { isOtherFormat, precisionFromConfig, precisionFromText, repoLayout, type RepoFile } from '../../src/hub/formats';

const f = (path: string, sizeBytes = 1, kind: RepoFile['kind'] = path.endsWith('.safetensors') || path.endsWith('.bin') ? 'weights' : 'other'): RepoFile => ({ path, sizeBytes, kind });

describe('precisionFromText', () => {
  it('names the precision of a repo name, folder or tag, most specific first', () => {
    expect(precisionFromText('Qwen3-8B-NVFP4')).toBe('NVFP4');
    expect(precisionFromText('gpt-oss-mxfp4')).toBe('MXFP4');
    expect(precisionFromText('model', 'awq')).toBe('AWQ');
    expect(precisionFromText('Model-GPTQ-Int4')).toBe('GPTQ');
    expect(precisionFromText('INT4/W4A16')).toBe('INT4');
    expect(precisionFromText('Qwen3-8B-FP8')).toBe('FP8');
    expect(precisionFromText('m-w8a8')).toBe('INT8');
    expect(precisionFromText('m-bf16')).toBe('BF16');
    expect(precisionFromText('m-fp16')).toBe('FP16');
    expect(precisionFromText('m-f32')).toBe('FP32');
    expect(precisionFromText('Qwen3-8B')).toBeUndefined();
  });
});

describe('precisionFromConfig', () => {
  const q = (quantization_config: unknown) => precisionFromConfig({ quantization_config });
  it('reads quantization_config, then torch_dtype', () => {
    expect(q({ quant_method: 'awq' })).toBe('AWQ');
    expect(q({ quant_method: 'gptq' })).toBe('GPTQ');
    expect(q({ quant_method: 'fp8' })).toBe('FP8');
    expect(q({ quant_method: 'fbgemm_fp8' })).toBe('FP8');
    expect(q({ quant_method: 'mxfp4' })).toBe('MXFP4');
    expect(q({ quant_method: 'bitsandbytes', load_in_8bit: true })).toBe('INT8');
    expect(q({ quant_method: 'bitsandbytes' })).toBe('INT4');
    expect(q({ quant_method: 'modelopt', quant_algo: 'NVFP4' })).toBe('NVFP4');
    expect(q({ quant_method: 'compressed-tensors', format: 'nvfp4-pack-quantized' })).toBe('NVFP4');
    expect(q({ quant_method: 'compressed-tensors', config_groups: { g: { weights: { num_bits: 4 } } } })).toBe('INT4');
    expect(q({ quant_method: 'compressed-tensors', config_groups: { g: { weights: { num_bits: 8, type: 'float' } } } })).toBe('FP8');
    expect(q({ quant_method: 'compressed-tensors', config_groups: { g: { weights: { num_bits: 8, type: 'int' } } } })).toBe('INT8');
    expect(q({ quant_method: 'compressed-tensors', config_groups: {} })).toBeUndefined();
    expect(q({})).toBeUndefined();
    expect(precisionFromConfig({ torch_dtype: 'bfloat16' })).toBe('BF16');
    expect(precisionFromConfig({ dtype: 'float16' })).toBe('FP16');
    expect(precisionFromConfig({ torch_dtype: 'float32' })).toBe('FP32');
    expect(precisionFromConfig({})).toBeUndefined();
    expect(precisionFromConfig(null)).toBeUndefined();
    expect(precisionFromConfig({ quantization_config: [] })).toBeUndefined();
  });
});

describe('isOtherFormat', () => {
  it('formats vLLM and SGLang do not load, by extension or folder', () => {
    expect(isOtherFormat('model-Q4_K_M.gguf')).toBe(true);
    expect(isOtherFormat('onnx/model.onnx_data')).toBe(true);
    expect(isOtherFormat('mlx/model.safetensors')).toBe(true);
    expect(isOtherFormat('original/consolidated.00.pth')).toBe(true);
    expect(isOtherFormat('trt/config.json')).toBe(true);
    expect(isOtherFormat('FP8/model.safetensors')).toBe(false);
    expect(isOtherFormat('config.json')).toBe(false);
  });
});

describe('repoLayout', () => {
  it('one variant per folder with a config and weights, the root first, with the files the engines read', () => {
    const files = [
      f('README.md'),
      f('config.json', 2),
      f('model-00002-of-00002.safetensors', 10),
      f('model-00001-of-00002.safetensors', 10),
      f('model.safetensors.index.json', 3),
      f('tokenizer.json', 4),
      f('FP8/config.json'),
      f('FP8/model.safetensors', 5),
      f('model-Q4_K_M.gguf', 7),
      f('mlx/config.json'),
      f('mlx/model.safetensors'),
      f('lora/adapter_config.json'),
      f('lora/adapter_model.safetensors'),
      f('MTP-BF16/config.json'),
      f('MTP-BF16/model.safetensors', 1, 'mtp'),
      f('loose/model.safetensors'),
      f('docs/notes.txt'),
      f('legacy/config.json'),
      f('legacy/pytorch_model.bin', 6),
    ];
    const v = repoLayout(files, { name: 'org/Model-8B-BF16' }).safetensors;
    expect(v.map((x) => [x.dir, x.precision, x.sizeBytes])).toEqual([
      ['', 'BF16', 29],
      ['FP8', 'FP8', 6],
      ['legacy', undefined, 7],
    ]);
    expect(v[0]?.weights.map((w) => w.path)).toEqual(['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']);
    expect(v[0]?.files.map((w) => w.path)).toEqual(['config.json', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors', 'model.safetensors.index.json', 'tokenizer.json']);
  });
  it('a root precision from the tags; MLX repos and repos without models have none', () => {
    expect(repoLayout([f('config.json'), f('model.safetensors')], { tags: ['fp8'] }).safetensors[0]?.precision).toBe('FP8');
    expect(repoLayout([f('config.json'), f('model.safetensors')]).safetensors[0]).not.toHaveProperty('precision');
    expect(repoLayout([f('config.json'), f('model.safetensors')], { tags: ['MLX'] }).safetensors).toEqual([]);
    expect(repoLayout([f('params.json'), f('consolidated.safetensors')]).safetensors).toHaveLength(1);
    expect(repoLayout([]).safetensors).toEqual([]);
  });
});

/**
 * Safetensors models of a repo and their precision: each folder with a config.json and its weights is one
 * model (a "variant": the root, "FP8/", "INT4/W4A16/"). Other formats (GGUF, MLX, ONNX, LoRA adapters,
 * MTP / draft modules shipped as folders) are not models vLLM or SGLang load.
 */
import type { HubFile } from './types';

/** A repo file as the layout needs it. */
export type RepoFile = Pick<HubFile, 'path' | 'sizeBytes' | 'kind'>;

/** Precisions of safetensors weights. */
export const PRECISIONS = ['NVFP4', 'MXFP4', 'FP8', 'INT8', 'AWQ', 'GPTQ', 'INT4', 'BF16', 'FP16', 'FP32'] as const;
export type Precision = (typeof PRECISIONS)[number];

const word = (w: string) => new RegExp(`(^|[^a-z0-9])${w}([^a-z0-9]|$)`, 'i');

/** Name tokens of a precision, most specific first ("W4A16" before "FP16", "NVFP4" before "FP4"). */
const PRECISION_WORDS: [RegExp, Precision][] = [
  [/nvfp4/i, 'NVFP4'],
  [/mxfp4/i, 'MXFP4'],
  [word('awq'), 'AWQ'],
  [/gptq/i, 'GPTQ'],
  [/w4a(16|8)|int4|4-?bit|bnb|nf4/i, 'INT4'],
  [/fp8|f8[-_]?e4m3|w8a8[-_]?fp8/i, 'FP8'],
  [/w8a(16|8)|int8|8-?bit/i, 'INT8'],
  [word('bf16'), 'BF16'],
  [word('(fp16|f16)'), 'FP16'],
  [word('(fp32|f32)'), 'FP32'],
];

/** The precision named in a repo name, folder or tag list ("Qwen3-8B-FP8", "INT4/W4A16", tag "awq"); undefined when none is. */
export function precisionFromText(...texts: readonly string[]): Precision | undefined {
  const text = texts.join(' ');
  return PRECISION_WORDS.find(([re]) => re.test(text))?.[1];
}

function obj(v: unknown): Record<string, unknown> | undefined {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
}

/**
 * The precision a transformers config.json declares: its quantization_config (fp8, awq, gptq,
 * compressed-tensors, modelopt, mxfp4, bitsandbytes) or, unquantized, its torch_dtype.
 */
export function precisionFromConfig(config: unknown): Precision | undefined {
  const c = obj(config) ?? {};
  const q = obj(c.quantization_config);
  if (q) {
    const method = String(q.quant_method ?? '').toLowerCase();
    const named = precisionFromText(String(q.quant_algo ?? ''), String(q.format ?? ''));
    if (method === 'awq') return 'AWQ';
    if (method === 'gptq') return 'GPTQ';
    if (method === 'fp8' || method === 'fbgemm_fp8') return 'FP8';
    if (method === 'mxfp4') return 'MXFP4';
    if (method === 'bitsandbytes') return q.load_in_8bit === true ? 'INT8' : 'INT4';
    if (named) return named;
    // compressed-tensors: the bits and type of the first weights group.
    const group = Object.values(obj(q.config_groups) ?? {}).map(obj)[0];
    const weights = obj(group?.weights);
    if (weights?.num_bits === 4) return 'INT4';
    if (weights?.num_bits === 8) return weights.type === 'float' ? 'FP8' : 'INT8';
    return undefined;
  }
  const dtype = String(c.torch_dtype ?? c.dtype ?? '');
  return ({ bfloat16: 'BF16', float16: 'FP16', float32: 'FP32' } as Record<string, Precision>)[dtype];
}

/** Folders of formats vLLM / SGLang do not load. */
const OTHER_DIRS = [word('mlx'), /onnx/i, /openvino/i, /coreml/i, word('exl[23]'), /tensorrt|(^|[^a-z])trt([^a-z]|$)/i, /^original$/i];
/** Files of other formats. */
const OTHER_FILE = /\.(gguf|onnx|onnx_data|ort|mlmodel|mlpackage|tflite|pte|engine|plan|npz|pth|pt|h5|keras|msgpack)$/i;

/** The file belongs to a format vLLM / SGLang do not load ("mlx/model.safetensors", "model.gguf", "original/consolidated.pth"). */
export function isOtherFormat(path: string): boolean {
  const segs = path.split('/');
  return OTHER_FILE.test(segs.at(-1) as string) || segs.slice(0, -1).some((dir) => OTHER_DIRS.some((re) => re.test(dir)));
}

/** Files of a model folder vLLM / SGLang read besides the weights: configs, tokenizer, chat template, remote code. */
const RUNTIME_FILE =
  /^(config|generation_config|preprocessor_config|processor_config|video_preprocessor_config|params|special_tokens_map|added_tokens|tokenizer_config|tokenizer|chat_template|quantize_config|quant_config|hf_quant_config|tekken)\.json$|^tokenizer[\w.-]*$|^vocab\.\w+$|^merges\.txt$|\.(model|tiktoken|py)$|^chat_template\.jinja$|\.index\.json$/i;

const isSafetensors = (p: string) => /\.safetensors$/i.test(p);
const isBinWeights = (p: string) => /(^|\/)pytorch_model[\w.-]*\.bin$/i.test(p);
const dirOf = (p: string) => (p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '');
const baseOf = (p: string) => p.slice(p.lastIndexOf('/') + 1);

/** One safetensors model of a repo: a folder with its config.json and its weights. */
export interface SafetensorsVariant<F extends RepoFile = RepoFile> {
  /** Folder inside the repo, '' for the root. */
  dir: string;
  /** The weights (every shard), sorted by path. */
  weights: F[];
  /** Everything vLLM / SGLang read from that folder (weights, index, config, tokenizer…). */
  files: F[];
  /** Bytes of `files`. */
  sizeBytes: number;
  /** From the folder name, else the repo's name and tags (refined from config.json later). */
  precision?: Precision | undefined;
}

export interface RepoLayout<F extends RepoFile = RepoFile> {
  /** One entry per loadable folder, the root first. */
  safetensors: SafetensorsVariant<F>[];
}

/** Code-point order: the same on every machine, whatever its locale. */
const byPath = (a: RepoFile, b: RepoFile) => (a.path < b.path ? -1 : 1);

/**
 * The safetensors models of a repo. `hint` (the repo name and tags) gives the precision of root weights
 * ("Qwen3-8B-FP8") and tells MLX repos, whose safetensors vLLM cannot read.
 */
export function repoLayout<F extends RepoFile>(files: readonly F[], hint: { name?: string; tags?: readonly string[] } = {}): RepoLayout<F> {
  const tags = hint.tags ?? [];
  if (tags.some((t) => t.toLowerCase() === 'mlx')) return { safetensors: [] };
  const byDir = new Map<string, F[]>();
  for (const f of files) if (!isOtherFormat(f.path)) byDir.set(dirOf(f.path), [...(byDir.get(dirOf(f.path)) ?? []), f]);
  const variants: SafetensorsVariant<F>[] = [];
  // Code-point order puts the root ('') first.
  for (const [dir, inDir] of [...byDir].sort((a, b) => (a[0] < b[0] ? -1 : 1))) {
    const st = inDir.filter((f) => isSafetensors(f.path));
    const weights = (st.length > 0 ? st : inDir.filter((f) => isBinWeights(f.path))).sort(byPath);
    if (weights.length === 0) continue;
    const names = new Set(inDir.map((f) => baseOf(f.path)));
    // LoRA adapters, MTP / draft modules ("MTP-BF16/", "DFlash2-FP8/") and weights without a config are not models.
    if (names.has('adapter_config.json') || weights.every((w) => w.kind === 'mtp' || w.kind === 'draft')) continue;
    if (!names.has('config.json') && !names.has('params.json')) continue;
    const all = inDir.filter((f) => weights.includes(f) || isSafetensors(f.path) || RUNTIME_FILE.test(baseOf(f.path))).sort(byPath);
    const precision = precisionFromText(dir) ?? (dir === '' ? precisionFromText(hint.name ?? '', ...tags) : undefined);
    variants.push({ dir, weights, files: all, sizeBytes: all.reduce((a, f) => a + f.sizeBytes, 0), ...(precision ? { precision } : {}) });
  }
  return { safetensors: variants };
}

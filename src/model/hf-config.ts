import { open } from 'node:fs/promises';
import { AppError } from '../contract/errors';
import { groupLayers } from './layers';
import { round3 } from './layers';
import type { ModelFactsEx } from './types';
import type { LoadShape } from './weights-exact';

type Json = Record<string, unknown>;

function obj(v: unknown): Json | undefined {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Json) : undefined;
}
function num(v: unknown): number | undefined {
  return typeof v === 'number' && Number.isFinite(v) ? v : undefined;
}
function str(v: unknown): string | undefined {
  return typeof v === 'string' && v.length > 0 ? v : undefined;
}

const DTYPES: Record<string, { label: string; bytes: number }> = {
  float32: { label: 'F32', bytes: 4 },
  float16: { label: 'F16', bytes: 2 },
  bfloat16: { label: 'BF16', bytes: 2 },
};

/** Quant label of a transformers checkpoint: quantization method, else the weights dtype. */
export function quantFromHfConfig(c: Json): string | undefined {
  const q = obj(c.quantization_config);
  const method = str(q?.quant_method)?.toLowerCase();
  if (q && method) {
    const bits = num(q.bits) ?? num(q.w_bit) ?? 4;
    if (method === 'awq' || method === 'gptq') return `${method.toUpperCase()}-${bits}bit`;
    if (method === 'bitsandbytes') return q.load_in_4bit === true ? 'BNB-4bit' : 'BNB-8bit';
    return method.toUpperCase();
  }
  return DTYPES[str(c.torch_dtype) ?? str(c.dtype) ?? '']?.label;
}

type LayerKind = 'full' | 'sliding' | 'recurrent' | 'none';

const LAYER_TYPE_KIND: Record<string, LayerKind> = {
  full_attention: 'full',
  attention: 'full',
  sliding_attention: 'sliding',
  chunked_attention: 'sliding',
  linear_attention: 'recurrent',
  mamba: 'recurrent',
  conv: 'recurrent',
  mlp: 'none',
};

/** Per-layer kind from the many ways transformers configs describe hybrid / SWA stacks. */
function layerKinds(c: Json, n: number, window: number | undefined): LayerKind[] {
  const types = Array.isArray(c.layer_types) ? (c.layer_types as unknown[]) : undefined;
  const hybrid = str(c.hybrid_override_pattern);
  const period = num(c.attn_layer_period);
  const interval = num(c.full_attention_interval);
  const swaPattern = num(c.sliding_window_pattern);
  const out: LayerKind[] = [];
  for (let i = 0; i < n; i++) {
    let kind: LayerKind = 'full';
    if (types) kind = LAYER_TYPE_KIND[String(types[i])] ?? 'full';
    else if (hybrid) kind = ({ '*': 'full', M: 'recurrent', '-': 'none' } as Record<string, LayerKind>)[hybrid[i] ?? '*'] ?? 'full';
    else if (period) kind = i % period === (num(c.attn_layer_offset) ?? 0) ? 'full' : 'recurrent';
    else if (interval) kind = (i + 1) % interval === 0 ? 'full' : 'recurrent';
    else if (window && swaPattern) kind = i % swaPattern < swaPattern - 1 ? 'sliding' : 'full';
    out.push(kind);
  }
  return out;
}

/**
 * Recurrent state elements per layer per sequence (Mamba / Mamba2 or gated linear attention): the
 * convolution state ((kernel − 1) × conv channels) and the SSM / delta-rule state.
 */
function recurrentState(c: Json, hidden: number): { conv: number; ssm: number } {
  const nv = num(c.linear_num_value_heads);
  if (nv) {
    const dk = num(c.linear_key_head_dim) ?? 128;
    const dv = num(c.linear_value_head_dim) ?? 128;
    const nk = num(c.linear_num_key_heads) ?? nv;
    const conv = num(c.linear_conv_kernel_dim) ?? 4;
    return { ssm: nv * dk * dv, conv: (conv - 1) * (2 * nk * dk + nv * dv) };
  }
  const dState = num(c.state_size) ?? num(c.mamba_d_state) ?? num(c.ssm_state_size) ?? 128;
  const dConv = num(c.conv_kernel) ?? num(c.mamba_d_conv) ?? 4;
  const dInner = num(c.mamba_d_inner) ?? (num(c.mamba_expand) ?? num(c.expand) ?? 2) * hidden;
  const groups = num(c.mamba_n_groups) ?? num(c.n_groups) ?? 1;
  return { ssm: dInner * dState, conv: (dConv - 1) * (dInner + 2 * groups * dState) };
}

const DTYPE_SIZE: Record<string, number> = { float32: 4, float16: 2, bfloat16: 2 };

export interface HfConfigFactsOptions {
  name?: string;
  parameterCount?: number;
  fileBytes?: number;
}

/** The chat template of a checkpoint: tokenizer_config.json's chat_template (a string, or named templates), else chat_template.jinja. */
export function chatTemplateOf(tokenizerConfig: unknown, jinja?: string): string | undefined {
  const t = obj(tokenizerConfig)?.chat_template;
  if (typeof t === 'string') return t;
  if (Array.isArray(t)) return t.map((x) => str(obj(x)?.template) ?? '').join('\n');
  return jinja;
}

/** Facts for a safetensors checkpoint from its config.json (vLLM path). */
export function factsFromHfConfig(raw: unknown, opts: HfConfigFactsOptions = {}): ModelFactsEx {
  const top = obj(raw) ?? {};
  // Multimodal checkpoints (Gemma 3/4, Qwen-VL, Llava…) keep the LLM shape in text_config.
  const c: Json = { ...top, ...(obj(top.text_config) ?? {}) };
  const facts: ModelFactsEx = {};
  const set = <K extends keyof ModelFactsEx>(k: K, v: ModelFactsEx[K] | undefined) => {
    if (v !== undefined) facts[k] = v;
  };
  const archs = Array.isArray(top.architectures) ? top.architectures : [];
  set('architecture', str(archs[0]) ?? str(c.model_type));
  // A vision part of its own (vision_config, or the vision tower's settings): it reads images.
  if (obj(top.vision_config) || obj(top.vision_tower_config)) set('vision', true);
  set('name', opts.name);
  if (opts.parameterCount) set('paramsB', round3(opts.parameterCount / 1e9));
  set('quant', quantFromHfConfig(c));
  if (opts.parameterCount && opts.fileBytes) set('bitsPerWeight', Math.round(((opts.fileBytes * 8) / opts.parameterCount) * 100) / 100);

  const hidden = num(c.hidden_size) ?? num(c.n_embd) ?? num(c.d_model);
  const heads = num(c.num_attention_heads) ?? num(c.n_head);
  const layers = num(c.num_hidden_layers) ?? num(c.n_layer) ?? num(c.num_layers);
  const kvHeadsRaw = num(c.num_key_value_heads) ?? num(c.num_attention_groups) ?? num(c.multi_query_group_num) ?? heads;
  const headDim = num(c.head_dim) ?? (hidden && heads ? Math.floor(hidden / heads) : undefined);
  set('embeddingLength', hidden);
  set('headCount', heads);
  set('blockCount', layers);
  set('contextLength', num(c.max_position_embeddings) ?? num(c.n_positions) ?? num(c.seq_length));
  set('expertCount', num(c.num_experts) ?? num(c.num_local_experts) ?? num(c.n_routed_experts) ?? num(c.moe_num_experts));
  set('expertUsedCount', num(c.num_experts_per_tok) ?? num(c.moe_top_k));
  set('vocabSize', num(c.vocab_size));
  set('feedForwardLength', num(c.intermediate_size));
  set('expertFeedForwardLength', num(c.moe_intermediate_size) ?? num(c.expert_intermediate_size));
  const vision = obj(top.vision_config);
  const visionHidden = num(vision?.hidden_size) ?? num(vision?.embed_dim);
  const visionLayers = num(vision?.num_hidden_layers) ?? num(vision?.depth);
  if (visionHidden && visionLayers) set('visionEncoder', { hiddenSize: visionHidden, layers: visionLayers });

  // MLA (DeepSeek, Kimi): vLLM caches one latent vector of kv_lora_rank + rope dims per token.
  const lora = num(c.kv_lora_rank);
  const kvHeads = lora ? 1 : kvHeadsRaw;
  const keyLength = lora ? lora + (num(c.qk_rope_head_dim) ?? 64) : headDim;
  const valueLength = lora ? 0 : headDim;
  set('headCountKv', kvHeads);
  set('keyLength', keyLength);
  set('valueLength', valueLength);
  // Gemma 4: the full-attention (global) layers have their own KV heads and head size.
  const globalKvHeads = lora ? undefined : num(c.num_global_key_value_heads);
  const globalHeadDim = lora ? undefined : num(c.global_head_dim);

  if (layers) {
    const window = c.use_sliding_window === false ? undefined : (num(c.sliding_window) ?? num(c.attention_chunk_size));
    const kinds = layerKinds(c, layers, window);
    const shapes = kinds.map((kind) => ({
      attn: kind === 'full' || kind === 'sliding',
      kvHeads: (kind === 'full' ? globalKvHeads : undefined) ?? kvHeads ?? 8,
      k: (kind === 'full' ? globalHeadDim : undefined) ?? keyLength ?? 128,
      v: (kind === 'full' ? globalHeadDim : undefined) ?? valueLength ?? 128,
      window: kind === 'sliding' ? window : undefined,
      recurrent: kind === 'recurrent',
    }));
    const recurrentLayers = kinds.filter((k) => k === 'recurrent').length;
    const swaLayers = shapes.filter((s) => s.attn && s.window !== undefined).length;
    const attnLayers = shapes.filter((s) => s.attn).length;
    const state = recurrentState(c, hidden ?? 4096);
    // vLLM keeps the convolution state in the model dtype and the SSM state in mamba_ssm_dtype when the config sets it.
    const modelBytes = DTYPE_SIZE[str(c.torch_dtype) ?? str(c.dtype) ?? ''] ?? 2;
    const ssmBytes = DTYPE_SIZE[str(c.mamba_ssm_dtype) ?? str(c.mamba_ssm_cache_dtype) ?? ''] ?? modelBytes;
    facts.kv = {
      groups: groupLayers(shapes),
      recurrentLayers,
      recurrentBytesPerSeq: recurrentLayers > 0 ? recurrentLayers * (state.conv + state.ssm) * 4 : 0,
      ...(recurrentLayers > 0 ? { recurrentState: { convElems: state.conv, ssmElems: state.ssm, convBytes: modelBytes, ssmBytes } } : {}),
      ...(lora ? { latent: true } : {}),
    };
    if (swaLayers > 0) {
      set('slidingWindow', window);
      set('slidingWindowLayers', swaLayers);
    }
    set('fullAttentionLayers', attnLayers - swaLayers);
  }
  return facts;
}

/**
 * RoPE caches vLLM builds (one per distinct rotary embedding): YaRN covers original × factor positions,
 * the others max_position_embeddings; the rotary dims are the rope head size (MLA) or head_dim ×
 * partial_rotary_factor. Gemma's local (sliding) layers have a RoPE of their own.
 */
function ropeCaches(c: Json, heads: number | undefined): LoadShape['rope'] {
  const hidden = num(c.hidden_size);
  const headDim = num(c.head_dim) ?? (hidden && heads ? Math.floor(hidden / heads) : undefined);
  const maxPos = num(c.max_position_embeddings);
  if (!headDim || !maxPos) return [];
  const scaling = obj(c.rope_scaling);
  const yarn = scaling && str(scaling.rope_type ?? scaling.type) === 'yarn' ? (num(scaling.original_max_position_embeddings) ?? 0) * (num(scaling.factor) ?? 1) : 0;
  const positions = Math.max(maxPos, yarn);
  const params = obj(c.rope_parameters);
  const full = obj(params?.full_attention);
  const partial = num(full?.partial_rotary_factor) ?? num(c.partial_rotary_factor) ?? 1;
  const out: LoadShape['rope'] = [{ positions, rotaryDim: num(c.qk_rope_head_dim) ?? Math.floor((num(c.global_head_dim) ?? headDim) * partial) }];
  if (num(c.rope_local_base_freq) !== undefined || obj(params?.sliding_attention)) out.push({ positions, rotaryDim: headDim });
  return out;
}

/** What vLLM's loading rules need from config.json (weights-exact.ts). */
export function loadShapeFromHfConfig(raw: unknown): LoadShape {
  const top = obj(raw) ?? {};
  const c: Json = { ...top, ...(obj(top.text_config) ?? {}) };
  const heads = num(c.num_attention_heads) ?? num(c.n_head);
  const lora = num(c.kv_lora_rank);
  const shape: LoadShape = {
    layers: num(c.num_hidden_layers) ?? num(c.n_layer) ?? num(c.num_layers) ?? 0,
    mtpLayers: num(c.num_nextn_predict_layers) ?? 0,
    // transformers' default is tied embeddings; most configs say it explicitly.
    tiedEmbeddings: (c.tie_word_embeddings ?? top.tie_word_embeddings) !== false,
    rope: ropeCaches(c, heads),
  };
  if (heads) shape.heads = heads;
  const kv = num(c.num_key_value_heads) ?? num(c.num_attention_groups) ?? num(c.multi_query_group_num) ?? heads;
  if (kv) shape.kvHeads = kv;
  if (lora) shape.mla = { kvLoraRank: lora, qkNope: num(c.qk_nope_head_dim) ?? 128, vHead: num(c.v_head_dim) ?? 128 };
  return shape;
}

const MAX_SAFETENSORS_HEADER = 100 * 1024 * 1024;

/** Exact parameter count from safetensors headers (8-byte LE length + JSON with tensor shapes). */
export async function safetensorsParamCount(files: string[]): Promise<number> {
  let total = 0;
  for (const f of files) {
    const fh = await open(f, 'r');
    try {
      const lenBuf = Buffer.alloc(8);
      await fh.read(lenBuf, 0, 8, 0);
      const len = Number(lenBuf.readBigUInt64LE(0));
      if (len <= 1 || len > MAX_SAFETENSORS_HEADER) throw new AppError('MODEL_INVALID', `Bad safetensors header in ${f}`);
      const buf = Buffer.alloc(len);
      await fh.read(buf, 0, len, 8);
      const header = JSON.parse(buf.toString('utf8')) as Record<string, { shape?: number[] }>;
      for (const [name, t] of Object.entries(header)) {
        if (name === '__metadata__') continue;
        total += (t.shape ?? []).reduce((a, b) => a * b, 1);
      }
    } finally {
      await fh.close();
    }
  }
  return total;
}

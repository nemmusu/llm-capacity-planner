/**
 * Weights one GPU holds once vLLM has loaded a safetensors checkpoint, summed tensor by tensor from the
 * safetensors headers (dtype × shape), with the loading rules of vLLM v0.31 (the pinned version; SGLang shards the same way):
 *  - only what the engine loads: MTP / next-token layers (`num_nextn_predict_layers`, `mtp.*`) are
 *    skipped without speculative decoding; the vision / audio tower is skipped with
 *    `--language-model-only`;
 *  - the vocabulary of the embedding and the LM head is padded to a multiple of 64 and split over the
 *    tensor-parallel GPUs (vocab_parallel_embedding.py); a tied LM head is the embedding itself;
 *  - norms, row-parallel biases, MoE routers and MLA's q_a / kv_a projections are replicated on every
 *    GPU (ReplicatedLinear, "bias is not parallelized");
 *  - K and V projections: max(1, kv_heads / TP) heads per GPU, replicated when there are fewer heads than
 *    GPUs (QKVParallelLinear num_kv_head_replicas);
 *  - everything else is split evenly over the GPUs (column / row parallel linear layers, MoE experts).
 * MLA layers also keep kv_b_proj dequantized to bf16 for the absorbed attention (W_UK, W_UV): an estimate,
 * reported apart.
 */

/** One tensor of a safetensors header. */
export interface TensorInfo {
  name: string;
  dtype: string;
  shape: number[];
}

const DTYPE_BYTES: Record<string, number> = {
  F64: 8,
  I64: 8,
  U64: 8,
  F32: 4,
  I32: 4,
  U32: 4,
  F16: 2,
  BF16: 2,
  I16: 2,
  U16: 2,
  F8_E4M3: 1,
  F8_E5M2: 1,
  F8_E8M0: 1,
  I8: 1,
  U8: 1,
  BOOL: 1,
};

/** Bytes of one tensor; undefined for a dtype safetensors does not define here. */
export function tensorBytes(t: Pick<TensorInfo, 'dtype' | 'shape'>): number | undefined {
  const size = DTYPE_BYTES[t.dtype];
  return size === undefined ? undefined : t.shape.reduce((a, b) => a * b, 1) * size;
}

/** What the loading rules need from config.json. */
export interface LoadShape {
  layers: number;
  heads?: number;
  kvHeads?: number;
  /** num_nextn_predict_layers (DeepSeek, GLM): extra layers after `layers`, loaded only for MTP. */
  mtpLayers: number;
  tiedEmbeddings: boolean;
  /** MLA: kv_lora_rank, qk_nope_head_dim, v_head_dim (for the absorbed bf16 copies of kv_b_proj). */
  mla?: { kvLoraRank: number; qkNope: number; vHead: number };
  /** The RoPE cos/sin caches vLLM keeps beside the weights: positions × rotary dims, one per distinct RoPE. */
  rope: { positions: number; rotaryDim: number }[];
}

export interface WeightsOptions {
  tp: number;
  /** --language-model-only: the vision / audio tower is not loaded. */
  textOnly?: boolean;
  /**
   * Pipeline stages (default 1): layers split as vLLM's get_pp_indices does (ppStarts), the embedding and
   * the vision tower on the first stage, the final norm and the LM head on the last.
   */
  pp?: number;
}

export type TensorRole = 'mtp' | 'vision' | 'vocab' | 'replicated' | 'kv' | 'qkv' | 'split';

export interface WeightsPerGpu {
  /** Bytes the most loaded GPU holds (with tensor parallel alone every GPU holds the same). */
  perGpuBytes: number;
  /** Pipeline parallel: the bytes of each stage's GPUs. */
  stageBytes: number[];
  /** Of perGpuBytes: what every GPU holds in full (norms, routers, replicated projections). */
  replicatedBytes: number;
  /** Of perGpuBytes: the vision / audio tower (0 with textOnly). */
  visionBytes: number;
  /** The vision tower's bytes per GPU even when textOnly drops it (to show both options). */
  visionBytesIfLoaded: number;
  /** Of perGpuBytes: the vocabulary padding to a multiple of 64. */
  vocabPaddingBytes: number;
  /** Of perGpuBytes: the RoPE cos/sin caches (bf16, positions × rotary dims). */
  ropeBytes: number;
  /** MLA: bf16 copies of kv_b_proj vLLM keeps for the absorbed attention, per GPU (min–max), not in perGpuBytes. */
  mlaAbsorbedBytes: { min: number; max: number };
  /** Every tensor of the checkpoint files. */
  checkpointBytes: number;
  /** Not loaded: MTP layers, and the vision tower with textOnly. */
  skippedBytes: { mtp: number; vision: number };
  tensors: number;
  /** Tensors whose dtype is unknown here (counted as 0 bytes). */
  unknownDtypes: string[];
}

const LAYER_RE = /(?:^|\.)layers\.(\d+)\./;
const VISION_RE = /(^|\.)(vision_tower|vision_model|visual|vision_encoder|multi_modal_projector|mm_projector|vision_embed|image_newline|audio_tower|audio_model|embed_vision|embed_audio|vision)\./;
const VOCAB_RE = /((^|\.)(embed_tokens|lm_head|wte|word_embeddings|tok_embeddings)|^output)\.weight$/;
const NORM_RE = /(norm\d*|ln_\w*|ln\d*)\.(weight|bias)$/i;
/** Row-parallel layers: their bias is the full output, on every GPU. */
const ROW_BIAS_RE = /\.(o_proj|down_proj|out_proj|dense|c_proj|w2|fc2|wo|dense_4h_to_h)\.bias$/;
/** Replicated linear layers and small vectors: MoE routers, MLA's q_a / kv_a projections, gates. */
const REPLICATED_RE =
  /\.(mlp\.gate|block_sparse_moe\.gate|feed_forward\.router|mlp\.router|router|gate\.e_score_correction_bias|e_score_correction_bias|shared_expert_gate|q_a_proj|kv_a_proj_with_mqa|fused_qkv_a_proj)(\.|$)/;
const KV_RE = /\.(k_proj|v_proj)\.(weight|bias|weight_scale|weight_scale_inv|input_scale|qweight|qzeros|scales|weight_packed|weight_zero_point)$/;
const QKV_RE = /\.(qkv_proj|query_key_value|c_attn|Wqkv|qkv)\.(weight|bias|weight_scale|weight_scale_inv|qweight|qzeros|scales)$/;

/** The rule a tensor follows (see the module comment). */
export function tensorRole(name: string, shape: LoadShape): TensorRole {
  if (/^(model\.)?mtp\./.test(name) || name.includes('.mtp.')) return 'mtp';
  const layer = LAYER_RE.exec(name);
  if (layer && shape.mtpLayers > 0 && Number(layer[1]) >= shape.layers && !VISION_RE.test(name)) return 'mtp';
  if (VISION_RE.test(name)) return 'vision';
  if (VOCAB_RE.test(name)) return 'vocab';
  if (NORM_RE.test(name) || ROW_BIAS_RE.test(name) || REPLICATED_RE.test(name)) return 'replicated';
  if (KV_RE.test(name)) return 'kv';
  if (QKV_RE.test(name)) return 'qkv';
  return 'split';
}

const pad64 = (n: number) => Math.ceil(n / 64) * 64;

/**
 * First layer of each pipeline stage, as vLLM's get_pp_indices splits them: num_layers // pp each, the
 * remainder one more each to the stages before the last (the last keeps the final norm and LM head).
 */
export function ppStarts(layers: number, pp: number): number[] {
  const parts = new Array<number>(pp).fill(Math.floor(layers / pp));
  for (let i = 2; i < (layers % pp) + 2; i++) parts[pp - i] = (parts[pp - i] as number) + 1;
  const starts: number[] = [];
  let at = 0;
  for (const n of parts) {
    starts.push(at);
    at += n;
  }
  return starts;
}

/** The weights one of `tp` GPUs holds (module comment), from the tensors of every checkpoint file. */
export function weightsPerGpu(tensors: readonly TensorInfo[], shape: LoadShape, opts: WeightsOptions): WeightsPerGpu {
  const tp = Math.max(1, Math.floor(opts.tp));
  const kvHeads = shape.kvHeads ?? shape.heads ?? 1;
  const kvShare = Math.max(1, Math.floor(kvHeads / tp)) / kvHeads;
  const heads = shape.heads ?? kvHeads;
  const qShare = heads / (heads + 2 * kvHeads);
  const out: WeightsPerGpu = {
    perGpuBytes: 0,
    stageBytes: [],
    replicatedBytes: 0,
    visionBytes: 0,
    visionBytesIfLoaded: 0,
    vocabPaddingBytes: 0,
    ropeBytes: 0,
    mlaAbsorbedBytes: { min: 0, max: 0 },
    checkpointBytes: 0,
    skippedBytes: { mtp: 0, vision: 0 },
    tensors: tensors.length,
    unknownDtypes: [],
  };
  const pp = Math.max(1, Math.floor(opts.pp ?? 1));
  const firstLayer = ppStarts(shape.layers, pp);
  const stages = new Array<number>(pp).fill(0);
  const stageOf = (name: string): number => {
    if (VISION_RE.test(name)) return 0;
    const layer = LAYER_RE.exec(name);
    if (layer) return firstLayer.filter((start) => start <= Number(layer[1])).length - 1;
    // The last stage holds the final norm and the LM head; the first the embedding and the towers.
    return /(^|\.)(lm_head|norm)\.weight$|^output\.weight$/.test(name) ? pp - 1 : 0;
  };
  for (const t of tensors) {
    const bytes = tensorBytes(t);
    if (bytes === undefined) {
      if (!out.unknownDtypes.includes(t.dtype)) out.unknownDtypes.push(t.dtype);
      continue;
    }
    out.checkpointBytes += bytes;
    const role = tensorRole(t.name, shape);
    let gpu: number;
    switch (role) {
      case 'mtp':
        out.skippedBytes.mtp += bytes;
        continue;
      case 'vision': {
        // The towers use the same parallel layers: split, with their norms replicated.
        const v = NORM_RE.test(t.name) || ROW_BIAS_RE.test(t.name) ? bytes : bytes / tp;
        out.visionBytesIfLoaded += v;
        if (opts.textOnly) {
          out.skippedBytes.vision += bytes;
          continue;
        }
        out.visionBytes += v;
        gpu = v;
        break;
      }
      case 'vocab': {
        // A tied model loads the embedding once (vLLM skips a checkpoint's lm_head copy).
        if (shape.tiedEmbeddings && /(^|\.)lm_head\.weight$/.test(t.name)) continue;
        const rows = t.shape[0] ?? 1;
        const per = bytes / rows;
        gpu = (pad64(rows) / tp) * per;
        out.vocabPaddingBytes += ((pad64(rows) - rows) / tp) * per;
        break;
      }
      case 'replicated':
        gpu = bytes;
        out.replicatedBytes += bytes;
        break;
      case 'kv':
        gpu = bytes * kvShare;
        break;
      case 'qkv':
        gpu = (bytes * qShare) / tp + bytes * (1 - qShare) * kvShare;
        break;
      default:
        gpu = bytes / tp;
    }
    const stage = stageOf(t.name);
    stages[stage] = (stages[stage] as number) + gpu;
    // A tied LM head on its own last stage is a copy of the embedding there.
    if (pp > 1 && role === 'vocab' && shape.tiedEmbeddings) stages[pp - 1] = (stages[pp - 1] as number) + gpu;
  }
  // cos_sin_cache: [positions, rotary_dim] in the model dtype (bf16), registered as a buffer with the weights.
  out.ropeBytes = shape.rope.reduce((a, r) => a + r.positions * r.rotaryDim * 2, 0);
  // Every stage builds the RoPE caches its layers use.
  out.stageBytes = stages.map((b) => Math.round(b + out.ropeBytes));
  out.perGpuBytes = Math.max(...out.stageBytes);
  if (shape.mla) {
    // W_UK and W_UV: kv_b_proj per GPU in bf16 (kv_lora_rank × heads / TP × (nope + v)), kept once or twice, for the
    // layers of one pipeline stage.
    const absorbed = Math.ceil(shape.layers / pp) * shape.mla.kvLoraRank * (heads / tp) * (shape.mla.qkNope + shape.mla.vHead) * 2;
    out.mlaAbsorbedBytes = { min: absorbed, max: 2 * absorbed };
  }
  out.replicatedBytes = Math.round(out.replicatedBytes);
  out.visionBytes = Math.round(out.visionBytes);
  out.visionBytesIfLoaded = Math.round(out.visionBytesIfLoaded);
  out.vocabPaddingBytes = Math.round(out.vocabPaddingBytes);
  return out;
}

/** Parses the JSON header of a safetensors file into its tensors (`__metadata__` left out). */
export function tensorsFromHeader(header: unknown): TensorInfo[] {
  const out: TensorInfo[] = [];
  if (header === null || typeof header !== 'object') return out;
  for (const [name, v] of Object.entries(header as Record<string, unknown>)) {
    if (name === '__metadata__' || v === null || typeof v !== 'object') continue;
    const t = v as { dtype?: unknown; shape?: unknown };
    if (typeof t.dtype !== 'string' || !Array.isArray(t.shape)) continue;
    out.push({ name, dtype: t.dtype, shape: t.shape.map(Number) });
  }
  return out;
}

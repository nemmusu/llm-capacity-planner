/** Layers that share one attention shape (same KV heads, head sizes and window). */
export interface AttentionGroup {
  layers: number;
  kvHeads: number;
  keyLength: number;
  valueLength: number;
  /** Sliding-window size in tokens; undefined = full attention. */
  window?: number;
}

/**
 * Exact per-layer KV layout read from config.json. Scalar facts cannot describe models whose layers differ
 * (Gemma 4: 16 KV heads × 256 on sliding-window layers but 4 × 512 on global ones; Qwen3.5: 3 of 4 layers are
 * recurrent), so the planner uses this when present.
 */
export interface KvLayout {
  groups: AttentionGroup[];
  /** Layers that keep a recurrent (SSM / linear-attention) state instead of, or next to, a KV cache. */
  recurrentLayers: number;
  /** Bytes of recurrent state per sequence, summed over all recurrent layers (f32). */
  recurrentBytesPerSeq: number;
  /**
   * MLA (DeepSeek, Kimi): each layer caches one compressed latent (kv_lora_rank + rope dims, one "head",
   * no V), which tensor parallelism replicates on every GPU instead of splitting.
   */
  latent?: boolean;
  /**
   * One recurrent layer's state per sequence, split as vLLM sizes it: the convolution state in the model
   * dtype and the SSM / delta-rule state in mamba_ssm_dtype (both divided by TP).
   */
  recurrentState?: { convElems: number; ssmElems: number; convBytes: number; ssmBytes: number };
}

/** What config.json says about a model's shape. */
export interface ModelFacts {
  architecture?: string;
  name?: string;
  paramsB?: number;
  /** Weight precision named by the config (quantization_config or torch_dtype). */
  quant?: string;
  /** Bits per weight, fileBytes × 8 / params when both are known. */
  bitsPerWeight?: number;
  contextLength?: number;
  blockCount?: number;
  headCount?: number;
  headCountKv?: number;
  keyLength?: number;
  valueLength?: number;
  embeddingLength?: number;
  /** Sliding-window size and how many layers use it. */
  slidingWindow?: number;
  slidingWindowLayers?: number;
  /** Layers with full attention (hybrid SSM / attention models); defaults to blockCount. */
  fullAttentionLayers?: number;
  expertCount?: number;
  expertUsedCount?: number;
  /** MoE: the feed-forward size of one expert (moe_intermediate_size). */
  expertFeedForwardLength?: number;
  /** Multimodal checkpoints: the vision encoder, whose weights are in the files and whose activations the engines reserve. */
  visionEncoder?: { hiddenSize: number; layers: number };
  /** The checkpoint reads images itself (config.json has a vision part). */
  vision?: boolean;
}

/** ModelFacts plus what the memory estimate needs. Serializable as JSON. */
export interface ModelFactsEx extends ModelFacts {
  kv?: KvLayout;
  vocabSize?: number;
  feedForwardLength?: number;
}

/** Returns the Hugging Face token if one is set. Never log its value. */
export type TokenFn = () => string | undefined;

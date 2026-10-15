/**
 * What the speed model needs from a checkpoint, exact from its tensors and config.json: the weights a
 * decode step reads (MoE: shared part + the experts the batch touches), the multiply-adds per token
 * (active parameters), the attention cost per layer and the KV bytes read per token.
 */
import type { KvLayout, ModelFactsEx } from '../model/types';
import { kvHeadsPerGpu } from '../model/vllm-kv';
import { tensorBytes, tensorRole, type LoadShape, type TensorInfo } from '../model/weights-exact';

export interface ModelProfile {
  layers: number;
  hidden: number;
  /** Routed experts per MoE layer and how many a token uses (0 for dense models). */
  experts: number;
  expertsPerToken: number;
  /** Bytes of the routed experts (all layers, all GPUs together) and of everything else a decode step reads (not the input embedding table). */
  routedExpertBytes: number;
  denseBytes: number;
  /** Parameters a token multiplies through (routed experts: only its top-k; no embedding lookup, no vision). */
  activeParams: number;
  /** Of activeParams: the LM head (vocabulary × hidden), multiplied only for the tokens sampled, not for every prompt token. */
  headParams: number;
  /** Every parameter of the text model (embeddings and all experts included). */
  totalParams: number;
  /** Attention per layer group: query heads × (key + value dims), and the window of sliding layers. */
  attention: { layers: number; headDims: number; window?: number }[];
  /** Recurrent layers: state bytes per sequence (read and written every decode step). */
  recurrentStateBytes: number;
  /** Recurrent (linear attention, Gated DeltaNet, Mamba) layers: their scan has its own cost per prompt token. */
  recurrentLayers: number;
}

const EXPERT_RE = /\.experts\.(\d+\.)?/;
const SHARED_RE = /shared_expert/;

/**
 * Parameters a weight tensor holds: packed 4-bit formats store two values per byte (NVFP4 / MXFP4 U8
 * weights and blocks) or eight per int32 (AWQ / GPTQ qweight).
 */
export function tensorParams(t: TensorInfo): number {
  const n = t.shape.reduce((a, b) => a * b, 1);
  if (t.dtype === 'I32' && /qweight$/.test(t.name)) return 8 * n;
  // U8 weights are packed FP4 (NVFP4, MXFP4); I8 ones are int8 (W8A8) unless stored as MXFP4 blocks.
  if ((t.dtype === 'U8' && /(weight|weight_packed|blocks)$/.test(t.name)) || (t.dtype === 'I8' && /blocks$/.test(t.name))) return 2 * n;
  return n;
}

/** The speed-relevant profile of a model (text part only: the vision tower runs per image, not per token). */
export function modelProfile(tensors: readonly TensorInfo[], shape: LoadShape, facts: ModelFactsEx, layout: KvLayout, sparseTopk?: number): ModelProfile {
  let routedExpertBytes = 0;
  let denseBytes = 0;
  let activeParams = 0;
  let routedParams = 0;
  let totalParams = 0;
  let headParams = 0;
  let embedParams = 0;
  for (const t of tensors) {
    const role = tensorRole(t.name, shape);
    if (role === 'mtp' || role === 'vision') continue;
    const bytes = tensorBytes(t) ?? 0;
    // Scales and zero points are bytes to read, not parameters.
    const isWeight = /(weight|weight_packed|qweight|blocks)$/.test(t.name) && !/scale|zero/.test(t.name);
    const params = isWeight ? tensorParams(t) : 0;
    totalParams += params;
    // Routed experts: per-expert tensors (".experts.7.") or fused ones whose first dimension is the expert count.
    const fused = facts.expertCount !== undefined && t.shape.length === 3 && t.shape[0] === facts.expertCount;
    const routed = (EXPERT_RE.test(t.name) || fused) && !SHARED_RE.test(t.name);
    if (routed) {
      routedExpertBytes += bytes;
      routedParams += params;
    } else {
      // A decode step reads one row of the input embedding per token, not the table (a tied one is read
      // whole, as the LM head).
      if (!(/embed_tokens|wte|word_embeddings|tok_embeddings/.test(t.name) && !shape.tiedEmbeddings)) denseBytes += bytes;
      // The input embedding is a lookup, not a matmul (a tied one is still multiplied as the LM head).
      const embed = /embed_tokens|wte|word_embeddings|tok_embeddings/.test(t.name);
      if (embed) embedParams += params;
      if (/(^|\.)(lm_head|embed_out)\.weight$|^output\.weight$/.test(t.name)) headParams += params;
      if (!(embed && !shape.tiedEmbeddings)) activeParams += params;
    }
  }
  if (shape.tiedEmbeddings) headParams = embedParams;
  const experts = facts.expertCount ?? 0;
  const expertsPerToken = facts.expertUsedCount ?? 0;
  if (experts > 0 && expertsPerToken > 0) activeParams += (routedParams * expertsPerToken) / experts;
  const attention = layout.groups.map((g) => ({
    layers: g.layers,
    headDims: (facts.headCount ?? g.kvHeads) * (layout.latent ? (shape.mla ? shape.mla.qkNope + (g.keyLength - shape.mla.kvLoraRank) + shape.mla.vHead : g.keyLength) : g.keyLength + g.valueLength),
    // Sparse attention (index_topk) attends to at most topk keys, like a window.
    ...(g.window || sparseTopk ? { window: Math.min(g.window ?? Number.POSITIVE_INFINITY, sparseTopk ?? Number.POSITIVE_INFINITY) } : {}),
  }));
  const rs = layout.recurrentState;
  return {
    layers: shape.layers,
    hidden: facts.embeddingLength ?? 0,
    experts,
    expertsPerToken,
    routedExpertBytes,
    denseBytes,
    activeParams,
    headParams,
    totalParams,
    attention,
    recurrentStateBytes: rs ? layout.recurrentLayers * (rs.convElems * rs.convBytes + rs.ssmElems * rs.ssmBytes) : 0,
    recurrentLayers: rs ? layout.recurrentLayers : 0,
  };
}

/** Expected distinct experts a batch of `batch` tokens touches in one layer (uniform routing). */
export function distinctExperts(experts: number, perToken: number, batch: number): number {
  if (experts <= 0) return 0;
  return experts * (1 - Math.pow(1 - perToken / experts, batch));
}

/**
 * KV bytes one GPU reads per decode step for one sequence at `context` tokens: windows capped, and with
 * sparse attention (index_topk) only the selected keys.
 */
export function kvReadBytes(layout: KvLayout, elem: number, tp: number, context: number, sparseTopk = Number.POSITIVE_INFINITY): number {
  let bytes = 0;
  for (const g of layout.groups) {
    const heads = layout.latent ? g.kvHeads : kvHeadsPerGpu(g.kvHeads, tp);
    bytes += g.layers * heads * (g.keyLength + g.valueLength) * elem * Math.min(context, g.window ?? Number.POSITIVE_INFINITY, sparseTopk);
  }
  return bytes;
}

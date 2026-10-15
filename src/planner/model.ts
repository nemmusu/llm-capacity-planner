/**
 * A checkpoint as the planner sees it: its weight format and family, what tensor parallelism needs from
 * config.json, the speed profile, and how well its weight bytes are verified (weights-verification.ts).
 */
import { kvLayoutOf } from '../model/kv';
import type { ModelFactsEx } from '../model/types';
import type { LoadShape, TensorInfo } from '../model/weights-exact';
import type { ParallelShape } from './hardware';
import { modelProfile } from './model-profile';
import type { PlannerModel } from './plan';
import { verificationOf, type ModelFamily } from './weights-verification';

type Json = Record<string, unknown>;
const obj = (v: unknown): Json | undefined => (v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Json) : undefined);
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);

/** The weight format of a checkpoint, by the bytes of its linear weights (packed and scaled formats told apart by their tensors). */
export function weightFormat(tensors: readonly TensorInfo[]): ParallelShape['weights'] {
  const names = new Set(tensors.map((t) => t.name));
  const dtypes = new Map(tensors.map((t) => [t.name, t.dtype]));
  const bytes: Record<ParallelShape['weights'], number> = { bf16: 0, fp8: 0, nvfp4: 0, mxfp4: 0, int4: 0, int8: 0 };
  for (const t of tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (/qweight$/.test(t.name) || (t.dtype === 'I32' && /weight_packed$/.test(t.name))) bytes.int4 += n * 4;
    else if (/(\.weight|weight_packed|_blocks|\.blocks)$/.test(t.name)) {
      const base = t.name.replace(/(\.weight|\.weight_packed|_blocks|\.blocks)$/, '');
      // MXFP4 packs two values per byte with an E8M0 scale per 32 (I8 in DeepSeek V4, U8 / blocks elsewhere).
      const e8m0 = dtypes.get(`${base}.scales`) === 'F8_E8M0' || dtypes.get(`${base}.weight_scale`) === 'F8_E8M0' || dtypes.get(`${base}.scale`) === 'F8_E8M0';
      if (t.dtype === 'U8' || (t.dtype === 'I8' && (/blocks$/.test(t.name) || e8m0))) bytes[names.has(`${base}.weight_scale_2`) || names.has(`${base}.weight_global_scale`) ? 'nvfp4' : 'mxfp4'] += n;
      else if (t.dtype === 'F8_E4M3' || t.dtype === 'F8_E5M2') bytes.fp8 += n;
      else if (t.dtype === 'I8') bytes.int8 += n;
      else if (t.dtype === 'BF16' || t.dtype === 'F16' || t.dtype === 'F32') bytes.bf16 += n * (t.dtype === 'F32' ? 4 : 2);
    }
  }
  // The format of the bulk of the weights (embeddings and norms stay bf16 in every format).
  const quantized = (['nvfp4', 'mxfp4', 'int4', 'fp8', 'int8'] as const).filter((f) => bytes[f] > 0).sort((a, b) => bytes[b] - bytes[a])[0];
  return quantized !== undefined && bytes[quantized] > bytes.bf16 * 0.25 ? quantized : 'bf16';
}

/** The architecture family a checkpoint belongs to (what its loading rules depend on). */
export function modelFamily(facts: ModelFactsEx, shape: LoadShape, tensors: readonly TensorInfo[]): ModelFamily {
  if (shape.mla) return 'mla';
  if ((facts.kv?.recurrentLayers ?? 0) > 0) return 'hybrid';
  if ((facts.expertCount ?? 0) > 0) return 'moe';
  if (facts.visionEncoder || tensors.some((t) => /vision_tower|visual\./.test(t.name))) return 'multimodal';
  return 'dense';
}

/** What tensor parallelism and the format checks need from config.json. */
export function parallelShapeFrom(config: unknown, facts: ModelFactsEx, shape: LoadShape, weights: ParallelShape['weights']): ParallelShape {
  const top = obj(config) ?? {};
  const c: Json = { ...top, ...(obj(top.text_config) ?? {}) };
  const q = obj(c.quantization_config) ?? obj(top.quantization_config);
  const block = Array.isArray(q?.weight_block_size) ? num(q.weight_block_size[0]) : undefined;
  const out: ParallelShape = {
    heads: shape.heads ?? facts.headCount ?? 1,
    kvHeads: shape.mla ? (shape.heads ?? 1) : (shape.kvHeads ?? shape.heads ?? 1),
    layers: shape.layers,
    weights,
    moe: (facts.expertCount ?? 0) > 0,
  };
  const arch = Array.isArray(top.architectures) && typeof top.architectures[0] === 'string' ? top.architectures[0] : undefined;
  if (arch) out.architecture = arch;
  const lk = num(c.linear_num_key_heads);
  const lv = num(c.linear_num_value_heads);
  if (lk) out.linearKeyHeads = lk;
  if (lv) out.linearValueHeads = lv;
  if (facts.feedForwardLength) out.intermediate = facts.feedForwardLength;
  if (facts.expertFeedForwardLength) out.expertIntermediate = facts.expertFeedForwardLength;
  if (block) out.quantBlock = block;
  return out;
}

/** DeepSeek's sparse attention: each query attends to index_topk keys chosen by an indexer. */
export function sparseTopkOf(config: unknown): number | undefined {
  const top = obj(config) ?? {};
  return num(obj(top.text_config)?.index_topk) ?? num(top.index_topk);
}

/** The sparse-attention indexer scores every earlier key: index_n_heads × index_head_dim per layer. */
export function indexerDimsOf(config: unknown): number | undefined {
  const top = obj(config) ?? {};
  const c: Json = { ...top, ...(obj(top.text_config) ?? {}) };
  const heads = num(c.index_n_heads);
  const dim = num(c.index_head_dim);
  return heads && dim ? heads * dim : undefined;
}

/** The indexer's quadratic scoring as one more attention term of the prefill (no window). */
function withIndexer(profile: ReturnType<typeof modelProfile>, dims: number | undefined): ReturnType<typeof modelProfile> {
  return dims ? { ...profile, attention: [...profile.attention, { layers: profile.layers, headDims: dims }] } : profile;
}

/** The planner's model from what was read of the checkpoint. */
export function plannerModel(config: unknown, facts: ModelFactsEx, shape: LoadShape, tensors: readonly TensorInfo[], repo?: string): PlannerModel & { family: ModelFamily; verification: ReturnType<typeof verificationOf> } {
  const { layout } = kvLayoutOf(facts);
  const weights = weightFormat(tensors);
  const family = modelFamily(facts, shape, tensors);
  const verification = verificationOf(weights, family);
  return {
    facts,
    layout,
    shape,
    tensors,
    parallel: parallelShapeFrom(config, facts, shape, weights),
    profile: withIndexer(modelProfile(tensors, shape, facts, layout, sparseTopkOf(config)), indexerDimsOf(config)),
    ...(sparseTopkOf(config) !== undefined ? { sparseTopk: sparseTopkOf(config) as number } : {}),
    formatGapShare: verification.gapShare,
    formatGapFixedMiB: verification.gapFixedMiB,
    family,
    verification,
    ...(repo !== undefined ? { repo } : {}),
  };
}

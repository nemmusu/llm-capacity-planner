/**
 * The compatibility matrix of the vLLM / SGLang capacity planner (data/compat.json): does this model, in this
 * weight format, with this engine at this version, start on this generation of NVIDIA GPUs? A configuration that fits
 * in memory but on which the engine does not start must never be recommended, so every answer has a source (the
 * engine's code at the release tag, an issue or PR, documentation) and what cannot be verified is "unknown", never
 * "ok" by assumption. Limits seen only under Windows / WSL2 are kept apart from the general ones.
 *
 * Read by src/planner/compat.ts; checked by test/contract/compat.test.ts; the code anchors are
 * re-checked against a new engine release with scripts/dev/update-compat.mjs.
 */
import { z } from 'zod';

/** GPU generations by CUDA compute capability: sm100 covers 10.0 (B200, GB200) and 10.3 (B300, GB300). */
export const COMPAT_ARCHS = ['sm80', 'sm86', 'sm89', 'sm90', 'sm100', 'sm120'] as const;
export const compatArchSchema = z.enum(COMPAT_ARCHS);
export type CompatArch = z.infer<typeof compatArchSchema>;

/** ok: starts and serves; partial: starts with a limit worth knowing; does-not-start: refuses or crashes at start. */
export const COMPAT_STATUSES = ['ok', 'partial', 'does-not-start', 'unknown'] as const;
export const compatStatusSchema = z.enum(COMPAT_STATUSES);
export type CompatStatus = z.infer<typeof compatStatusSchema>;

export const compatEngineSchema = z.enum(['vllm', 'sglang']);
export type CompatEngine = z.infer<typeof compatEngineSchema>;

/** "all": everywhere (Linux servers included); "wsl": only under Windows / WSL2 (e.g. pinned memory, UVA). */
export const compatPlatformSchema = z.enum(['all', 'wsl']);

/**
 * What a rule is about. w:<format>:<dense|moe>: kernels for linear layers or fused-MoE experts in a weight format;
 * a:<mixer>: attention or another sequence mixer; kv:<dtype>: the KV cache; m:<id>: something of one model family.
 */
export const compatFeatureIdSchema = z.string().regex(/^(w:[a-z0-9-]+:(dense|moe)|a:[a-z0-9-]+|kv:[a-z0-9-]+|m:[a-z0-9-]+|p:[a-z0-9-]+)$/);

const repoSchema = z.string().regex(/^[\w.-]+\/[\w.-]+$/);

/** A line range of an engine's (or a kernel library's) code at a release tag, with a verbatim substring of it. */
export const compatCodeSourceSchema = z.strictObject({
  kind: z.literal('code'),
  repo: repoSchema,
  /** Release tag ("v0.31.0") or commit. */
  ref: z.string().min(1),
  path: z.string().regex(/^[\w./-]+$/),
  /** 1-based, inclusive. */
  lines: z.tuple([z.number().int().positive(), z.number().int().positive()]).refine(([a, b]) => a <= b, 'start > end'),
  /** Copied exactly from one of those lines: scripts/dev/update-compat.mjs checks it is still there. */
  anchor: z.string().min(3),
});
export const compatIssueSourceSchema = z.strictObject({
  kind: z.enum(['issue', 'pr']),
  repo: repoSchema,
  number: z.number().int().positive(),
  /** What it shows: engine version, GPU, outcome. */
  note: z.string().min(1),
});
export const compatDocsSourceSchema = z.strictObject({ kind: z.literal('docs'), url: z.string().regex(/^https:\/\//), note: z.string().min(1) });
export const compatSourceSchema = z.discriminatedUnion('kind', [compatCodeSourceSchema, compatIssueSourceSchema, compatDocsSourceSchema]);
export type CompatSource = z.infer<typeof compatSourceSchema>;

/** One verified fact: this feature, with this engine, on these architectures, has this status. */
export const compatRuleSchema = z.strictObject({
  id: z.string().regex(/^[a-z0-9][a-z0-9.-]*$/),
  engine: compatEngineSchema,
  feature: compatFeatureIdSchema,
  archs: z.array(compatArchSchema).min(1),
  status: compatStatusSchema,
  platform: compatPlatformSchema,
  /** What happens and why (kernel, backend, check). */
  reason: z.string().min(1),
  /** A flag or variable that makes it start, verified in the code or an issue. */
  workaround: z.string().min(1).optional(),
  /** The open PR or the later release that fixes it, when known. */
  fixedIn: z.string().min(1).optional(),
  /** Only for these models (ids of models.json); without it the rule applies to every model with the feature. */
  models: z.array(z.string().min(1)).min(1).optional(),
  sources: z.array(compatSourceSchema).min(1),
});
export type CompatRule = z.infer<typeof compatRuleSchema>;

/** The engine release the rules were verified on, and the kernel libraries it pins. */
export const compatEngineVersionSchema = z.strictObject({
  engine: compatEngineSchema,
  version: z.string().regex(/^\d+\.\d+\.\d+(\.post\d+)?$/),
  ref: z.string().min(1),
  commit: z.string().regex(/^[0-9a-f]{40}$/),
  dependencies: z.array(z.strictObject({ name: z.string().min(1), version: z.string().min(1), source: compatCodeSourceSchema })),
});

export const compatArchInfoSchema = z.strictObject({
  id: compatArchSchema,
  /** As in hardware.json ("8.9", "10.3"). */
  computeCapabilities: z.array(z.string().regex(/^\d+\.\d+$/)).min(1),
  name: z.string().min(1),
});

export const compatFeatureSchema = z.strictObject({
  id: compatFeatureIdSchema,
  kind: z.enum(['weights', 'attention', 'kv-cache', 'model', 'parallel']),
  description: z.string().min(1),
});

/** One checkpoint of a model (models.json quantizations) and the weight kernels it needs. */
export const compatVariantSchema = z.strictObject({
  repo: repoSchema,
  /** The catalog's precision label ("mixed:nvfp4+fp8", "fp8-block", "gguf"...). */
  format: z.string().min(1),
  /** w:* features; empty when the format is outside vLLM / SGLang (outOfScope) or could not be read (weightsUnknown). */
  weights: z.array(compatFeatureIdSchema),
  outOfScope: z.string().min(1).optional(),
  weightsUnknown: z.string().min(1).optional(),
});

/** The features of one model of models.json: written by scripts/dev/update-compat.mjs, re-derived by the test. */
export const compatModelSchema = z.strictObject({
  id: z.string().min(1),
  /** config.json architectures[0]; null when the config cannot be read (gated) or is in Mistral format. */
  architectureClass: z.string().nullable(),
  moe: z.boolean(),
  /** a:* and m:* features, from models.json architecture.attention and the config. */
  features: z.array(compatFeatureIdSchema),
  variants: z.array(compatVariantSchema).min(1),
});
export type CompatModel = z.infer<typeof compatModelSchema>;

export const compatFileSchema = z.strictObject({
  $comment: z.string(),
  schemaVersion: z.literal(1),
  checkedOn: z.iso.date(),
  engines: z.array(compatEngineVersionSchema).min(1),
  architectures: z.array(compatArchInfoSchema).length(COMPAT_ARCHS.length),
  features: z.array(compatFeatureSchema).min(1),
  models: z.array(compatModelSchema).min(1),
  rules: z.array(compatRuleSchema).min(1),
});
export type CompatFile = z.infer<typeof compatFileSchema>;

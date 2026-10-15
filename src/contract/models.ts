/**
 * The model catalog of the vLLM / SGLang capacity planner (data/models.json): recent open-weight models
 * with their license, architecture, context, quantized checkpoints, engine support and the exact settings for
 * 1M tokens, each fact with its source. "hf" and "derived" are written by scripts/dev/update-models.mjs from
 * the files saved in data/models/ (config.json, compact safetensors headers, tensor summaries);
 * the rest is curated by hand. Field by field: data/README.md.
 */
import { z } from 'zod';

const url = z.string().regex(/^https:\/\//);
const tokens = z.number().int().positive();
/** Bytes, tensors and elements of a group of tensors (a dtype, the language model, MTP layers, vision, audio). */
const tensorTotalsSchema = z.strictObject({ tensors: z.number().int().nonnegative(), bytes: z.number().int().nonnegative(), elements: z.number().int().nonnegative() });

/** Exact totals of a checkpoint's tensors, from its safetensors headers. */
export const WeightsSchema = z.strictObject({
  shards: z.number().int().positive(),
  tensors: z.number().int().positive(),
  tensorBytes: z.number().int().positive(),
  storageElements: z.number().int().positive(),
  fileBytes: z.number().int().nonnegative(),
  byDtype: z.record(z.string(), tensorTotalsSchema),
  groups: z.strictObject({ lm: tensorTotalsSchema, mtp: tensorTotalsSchema, vision: tensorTotalsSchema, audio: tensorTotalsSchema }),
  /** Tensors the shard index places in the wrong file (the headers are right). */
  indexWrongFile: z.number().int().nonnegative().optional(),
});

/** 4-bit weights packed two per byte (I8/U8), recognized from the headers by their scale tensors. */
const packedWeightsSchema = z.partialRecord(
  z.enum(['mxfp4', 'nvfp4']),
  z.strictObject({ tensors: z.number().int().positive(), bytes: z.number().int().positive(), scaleBytes: z.number().int().positive(), valuesPerScale: z.number(), scaleDtype: z.string() }),
);

const engineSchema = z.strictObject({
  /** Only when stated as a version ("0.19.1"); "nightly"/"main" end up in notes. */
  minVersion: z
    .string()
    .regex(/^\d+\.\d+(\.\d+)?(\.post\d+|rc\d+)?$/)
    .nullable(),
  versionStatement: z.string().nullable(),
  command: z.string().nullable(),
  flags: z.array(z.string()),
  sources: z.array(url),
  notes: z.string().nullable(),
});

/** What to pass to vLLM / SGLang for 1,048,576 tokens. */
const to1MSchema = z.strictObject({
  method: z.enum(['native', 'yarn', 'llama3', 'deepseek_yarn']),
  factor: z.number().positive().optional(),
  originalMaxPositions: tokens.optional(),
  maxModelLen: z.literal(1_048_576),
  ropeKey: z.literal('rope_scaling').optional(),
  /** The config's rope block with the new factor (every other key kept). */
  rope: z.record(z.string(), z.unknown()).optional(),
  /** The JSON passed as is to --hf-overrides (vLLM) and --json-model-override-args (SGLang). */
  hfOverrides: z.record(z.string(), z.unknown()).optional(),
  vllmArgs: z.array(z.string()),
  sglangArgs: z.array(z.string()),
  env: z.record(z.string(), z.string()),
  /** True only when the lab documents at least this factor (or the length is native). */
  qualityGuaranteed: z.boolean(),
  withinDeclaredTokens: z.boolean().optional(),
  sources: z.array(url),
  howItWorks: z.string().optional(),
});

export const LongContextCaseSchema = z.enum(['native>=1M', 'yarn-official', 'yarn-beyond-declared', 'not-applicable']);

const quantizationSchema = z.strictObject({
  /** bf16, fp8-block, fp8-channel, fp8-tensor, fp8, mxfp8, nvfp4, mxfp4, w4a16-int4, gptq-int4, int8, gguf, mlx, "mixed:a+b"… */
  format: z.string().min(1),
  repo: z.string(),
  /** base: the checkpoint itself; official: the lab or NVIDIA; third-party: Red Hat AI. */
  kind: z.enum(['base', 'official', 'third-party']),
  gated: z.union([z.boolean(), z.string()]).optional(),
  /** Folder of its fixtures under the model's: '' the model's own, 'variants/<org>__<repo>', null when not saved. */
  weightsVariantDir: z.string().nullable(),
});

const variantSchema = z.strictObject({
  repo: z.string(),
  precision: z.string(),
  revision: z.string().nullable(),
  readable: z.boolean(),
  fixtures: z.string(),
  weights: WeightsSchema.nullable(),
  packedWeights: packedWeightsSchema.nullable(),
});

export const ModelEntrySchema = z.strictObject({
  id: z.string().regex(/^[a-z0-9-]+$/),
  repo: z.string().regex(/^[\w.-]+\/[\w.-]+$/),
  lab: z.string().min(1),
  name: z.string().min(1),
  releaseDate: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  /** card/blog when the lab states the date (releaseDateUrl), else the Hugging Face repo's creation date. */
  releaseDateSource: z.enum(['card', 'blog', 'hf-createdAt']),
  releaseDateUrl: url.optional(),
  license: z.strictObject({
    name: z.string().min(1),
    hfTag: z.string().nullable(),
    commercialUse: z.union([z.boolean(), z.literal('conditional')]),
    conditions: z.string().nullable(),
    url,
  }),
  architecture: z.strictObject({
    kind: z.enum(['dense', 'moe']),
    attention: z.array(z.string()).min(1),
    layoutNotes: z.string().nullable(),
    modalities: z.array(z.string()).min(1),
  }),
  /** As the lab states them, in billions. */
  params: z.strictObject({ totalB: z.number().positive(), activeB: z.number().positive(), source: url }),
  context: z.strictObject({
    cardMaxTokens: tokens.nullable(),
    cardSource: z.strictObject({ url, quote: z.string().nullable() }).nullable(),
    configMaxPositions: tokens.nullable(),
    /** Where in config.json / params.json, e.g. "config.json:text_config.max_position_embeddings". */
    configSource: z.string().nullable(),
    note: z.string().nullable(),
  }),
  /** The config's RoPE: type, factor and original length as written (null: no rope block, e.g. NoPE). */
  rope: z
    .strictObject({
      type: z.string(),
      factor: z.number().nullable(),
      originalMaxPositions: tokens.nullable(),
      layers: z.record(z.string(), z.strictObject({ type: z.string(), factor: z.number().nullable() })).optional(),
      source: z.string(),
    })
    .nullable(),
  mtp: z.strictObject({ inCheckpoint: z.boolean().nullable(), tensors: z.number().int().nonnegative().nullable(), documented: z.boolean().nullable(), notes: z.string().nullable() }),
  visionEncoder: z.strictObject({ inCheckpoint: z.boolean().nullable(), documented: z.boolean().nullable(), notes: z.string().nullable() }),
  quantizations: z.array(quantizationSchema).min(1),
  cardNamedQuantizations: z.array(z.strictObject({ format: z.string(), repo: z.string(), notes: z.string().nullable() })),
  draftModels: z.array(z.strictObject({ repo: z.string(), kind: z.string(), architectures: z.array(z.string()).nullable() })),
  engines: z.strictObject({ vllm: engineSchema, sglang: engineSchema }),
  longContext: z.strictObject({
    case: LongContextCaseSchema,
    officialMaxTokens: tokens.nullable(),
    officialFactor: z.number().positive().nullable(),
    method: z.string().nullable(),
    /** The lab's own long-context instructions, verbatim. */
    documented: z.strictObject({ ropeScaling: z.record(z.string(), z.unknown()).nullable(), vllm: z.string().nullable(), sglang: z.string().nullable() }),
    statement: z.string().nullable(),
    source: url.nullable(),
    notes: z.string().nullable(),
    to1M: to1MSchema.nullable(),
    /** Why no arguments are given when to1M is null. */
    to1MUnavailable: z.string().nullable(),
  }),
  sources: z.array(z.strictObject({ field: z.string().min(1), url, quote: z.string().nullable() })).min(1),
  notes: z.string().nullable(),
  hf: z.strictObject({
    revision: z.string().regex(/^[0-9a-f]{40}$/),
    createdAt: z.string(),
    gated: z.union([z.boolean(), z.string()]),
    license: z.string().nullable(),
    licenseName: z.string().nullable(),
    readable: z.boolean(),
    weightFormat: z.enum(['hf', 'mistral']).nullable(),
    fixtures: z.string(),
    /** Why its files are not kept here nor in the models pack: its licence may not allow passing them on. */
    withheld: z.string().optional(),
  }),
  derived: z.strictObject({
    architectures: z.array(z.string()).nullable(),
    modelType: z.string().nullable(),
    textModelType: z.string().nullable(),
    numHiddenLayers: z.number().int().positive().nullable(),
    configMaxPositions: tokens.nullable(),
    configMaxPositionsPath: z.string().nullable(),
    configRope: z.record(z.string(), z.unknown()).nullable(),
    configRopePath: z.string().nullable(),
    configMaxPositionsKey: z.string().nullable(),
    configMtpLayers: z.number().int().nonnegative().nullable(),
    configHasVision: z.boolean(),
    configQuantization: z.record(z.string(), z.unknown()).nullable(),
    weights: WeightsSchema.nullable(),
    packedWeights: packedWeightsSchema.nullable(),
    precision: z.string().nullable(),
    variants: z.array(variantSchema),
    quantRepos: z
      .array(
        z.strictObject({
          repo: z.string(),
          kind: z.enum(['official', 'third-party']),
          draft: z.boolean(),
          precision: z.string().nullable(),
          architectures: z.array(z.string()).nullable(),
          gated: z.union([z.boolean(), z.string()]),
          revision: z.string().nullable(),
        }),
      )
      .nullable(),
  }),
});

export const ModelsFileSchema = z.strictObject({
  schemaVersion: z.literal(1),
  generatedAt: z.string(),
  models: z.array(ModelEntrySchema).min(1),
});

export type ModelEntry = z.infer<typeof ModelEntrySchema>;
export type ModelsFile = z.infer<typeof ModelsFileSchema>;
export type Weights = z.infer<typeof WeightsSchema>;
export type LongContextCase = z.infer<typeof LongContextCaseSchema>;

import { z } from 'zod';

/**
 * Reference data for the VRAM and capacity calculator of vLLM / SGLang (data/*.json):
 * a catalog of NVIDIA GPUs from official NVIDIA sources, and public measurements of the two engines
 * read from logs and benchmarks. Every value carries its source; a value that is not public is null
 * and listed in `gaps`, never estimated. Units are in the field names.
 *
 * Checked by test/contract/data.test.ts (`npm test -- test/contract`).
 */

const url = z.url({ protocol: /^https$/ });
const positive = z.number().positive();
const nullablePositive = positive.nullable();
const count = z.number().int().positive();

/** Where one or more fields come from: `fields` are dotted paths of the entry (e.g. "tflopsDense.fp8"). */
export const dataSourceSchema = z.strictObject({
  fields: z.array(z.string().min(1)).min(1),
  url,
  title: z.string().min(1),
  /** How the value was read, or a footnote that matters (e.g. "datasheet footnote: with sparsity"). */
  note: z.string().optional(),
});
export type DataSource = z.infer<typeof dataSourceSchema>;

/** A field that is not public (or not published exactly), and why it is null. */
export const dataGapSchema = z.strictObject({ field: z.string().min(1), reason: z.string().min(1) });

/** Tensor-core TFLOPS of one GPU, per format, in TFLOPS (10^12 op/s); null when NVIDIA does not publish it. */
export const tflopsSchema = z.strictObject({
  tf32: nullablePositive,
  fp16: nullablePositive,
  bf16: nullablePositive,
  fp8: nullablePositive,
  int8: nullablePositive,
  fp6: nullablePositive,
  fp4: nullablePositive,
});
export type Tflops = z.infer<typeof tflopsSchema>;
export const TFLOPS_FORMATS = ['tf32', 'fp16', 'bf16', 'fp8', 'int8', 'fp6', 'fp4'] as const;

export const gpuSegmentSchema = z.enum(['consumer', 'workstation', 'datacenter']);
export const gpuArchitectureSchema = z.enum(['ampere', 'ada-lovelace', 'hopper', 'blackwell', 'blackwell-ultra']);
export const p2pSupportSchema = z.enum(['supported', 'unsupported', 'unknown']);

export const gpuSpecSchema = z.strictObject({
  /** Stable id used by references.json, e.g. "h100-sxm5-80gb". */
  id: z.string().regex(/^[a-z0-9][a-z0-9-]*$/),
  name: z.string().min(1),
  vendor: z.literal('nvidia'),
  segment: gpuSegmentSchema,
  /** "sxm", "pcie", "nvl", "workstation", "max-q", "server", "superchip", "desktop"... */
  formFactor: z.string().min(1),
  architecture: gpuArchitectureSchema,
  /** CUDA compute capability, e.g. "9.0", "12.0". */
  computeCapability: z.string().regex(/^\d+\.\d+$/),
  memoryType: z.string().min(1),
  /** Marketing capacity in GB as NVIDIA states it (e.g. 80, 141, 32). */
  vramGBNominal: positive,
  /** Exact framebuffer bytes: nvidiaSmiTotalMiB × 2^20 when that is known; null otherwise. */
  vramBytes: z.number().int().positive().nullable(),
  /** "Total" of nvidia-smi (memory.total / -q -d MEMORY), in MiB, as measured on a real card. */
  nvidiaSmiTotalMiB: z.number().int().positive().nullable(),
  /** "Reserved" of nvidia-smi -q -d MEMORY (driver/firmware), in MiB: not usable by CUDA. */
  nvidiaSmiReservedMiB: z.number().int().nonnegative().nullable(),
  /**
   * Memory CUDA sees (cudaDeviceProp.totalGlobalMem = torch.cuda.get_device_properties().total_memory), the base of
   * vLLM's gpu-memory-utilization. It depends on the OS and driver (an RTX 5090: 31.82 GiB on Windows, 31.35 GiB on
   * Linux): cudaTotalBytes is an exact measurement (its source says where), cudaTotalGiB the lowest value PyTorch or
   * vLLM printed on Linux, where the engines run (2 decimals).
   */
  cudaTotalBytes: z.number().int().positive().nullable(),
  cudaTotalGiB: nullablePositive,
  /** Peak memory bandwidth in GB/s (10^9 B/s). */
  memoryBandwidthGBs: nullablePositive,
  /** Dense tensor TFLOPS, without sparsity. */
  tflopsDense: tflopsSchema,
  /** TFLOPS exactly as NVIDIA publishes them with 2:4 sparsity, when it does. */
  tflopsSparse: tflopsSchema,
  /**
   * Formats whose dense value is NVIDIA's sparse figure halved, as the datasheet footnote states
   * ("specifications are shown with sparsity; dense is one-half"), because only the sparse one is published.
   */
  denseHalvedFromSparse: z.array(z.enum(TFLOPS_FORMATS)),
  /**
   * How the TFLOPS were derived when NVIDIA publishes no exact figure for this card (e.g. another edition's figures scaled
   * by the boost clock); absent when they come from an NVIDIA document as they are.
   */
  tflopsDerived: z.string().min(1).optional(),
  /** Formats with hardware tensor-core support. */
  formats: z.strictObject({ fp8: z.boolean(), nvfp4: z.boolean(), mxfp4: z.boolean(), mxfp8: z.boolean() }),
  nvlink: z.strictObject({
    supported: z.boolean(),
    generation: z.number().int().positive().nullable(),
    /** Total bidirectional NVLink bandwidth per GPU in GB/s; null without NVLink. */
    bandwidthGBs: nullablePositive,
    /** GPUs one NVLink bridge or switch connects (2 for a bridged pair, 8 on an HGX board); null without NVLink. */
    domainGpus: z.number().int().positive().nullable(),
    note: z.string().optional(),
  }),
  /** Most GPUs of this model in one server or workstation as NVIDIA (or a certified OEM design) builds it; null when no figure is published. */
  server: z.strictObject({ maxGpus: z.number().int().positive().nullable() }),
  pcie: z.strictObject({
    gen: z.number().int().min(3).max(6),
    lanes: z.number().int().positive().nullable(),
    /** Host link bandwidth per direction in GB/s for the card's usual slot (NVIDIA's figure, else the PCIe rate). */
    bandwidthGBsPerDirection: nullablePositive,
  }),
  /** Peer-to-peer over PCIe between two cards of this model (GeForce: blocked by the driver unless noted). */
  p2p: p2pSupportSchema,
  p2pNote: z.string().optional(),
  /** Maximum board power (TDP/TGP) in W; configurableMinW when NVIDIA publishes a configurable range. */
  power: z.strictObject({ maxW: nullablePositive, configurableMinW: nullablePositive }),
  /** Superchip/system the per-GPU numbers refer to (e.g. "GB200 NVL72: values per Blackwell GPU"). */
  perGpuOf: z.string().optional(),
  sources: z.array(dataSourceSchema).min(1),
  gaps: z.array(dataGapSchema),
  notes: z.string().optional(),
});
export type GpuSpec = z.infer<typeof gpuSpecSchema>;

export const hardwareFileSchema = z.strictObject({
  $comment: z.string().optional(),
  schemaVersion: z.literal(1),
  /** Date the catalog was last checked against its sources (YYYY-MM-DD). */
  checkedOn: z.iso.date(),
  gpus: z.array(gpuSpecSchema).min(1),
});
export type HardwareFile = z.infer<typeof hardwareFileSchema>;

export const referenceEngineSchema = z.enum(['vllm', 'sglang']);
export const referenceSourceKindSchema = z.enum(['issue', 'discussion', 'pull-request', 'docs', 'recipe', 'blog', 'benchmark', 'paper']);

/** One run's performance at a known load (vllm bench serve / sglang.bench_serving or equivalent). */
export const referencePerfSchema = z.strictObject({
  /**
   * Max concurrent requests; null when the load is set by request rate only. With both null, all numPrompts
   * requests were sent at once (request rate inf, no cap).
   */
  concurrency: count.nullable(),
  requestRatePerS: nullablePositive,
  numPrompts: count.nullable(),
  dataset: z.string().min(1),
  /** Input/output tokens per request (mean for datasets with variable lengths). */
  inputLen: count.nullable(),
  outputLen: count.nullable(),
  outputTokPerS: nullablePositive,
  totalTokPerS: nullablePositive,
  /** Prefill (input) tokens per second when reported separately. */
  inputTokPerS: nullablePositive,
  requestPerS: nullablePositive,
  ttftMsMean: nullablePositive,
  ttftMsMedian: nullablePositive,
  ttftMsP99: nullablePositive,
  tpotMsMean: nullablePositive,
  tpotMsMedian: nullablePositive,
  itlMsMedian: nullablePositive,
  /** Single-request decode speed in tokens/s, when the source reports it as such. */
  decodeTokPerSPerUser: nullablePositive,
  note: z.string().optional(),
});
export type ReferencePerf = z.infer<typeof referencePerfSchema>;

export const referenceSchema = z.strictObject({
  id: z.string().regex(/^[a-z0-9][a-z0-9-]*$/),
  engine: referenceEngineSchema,
  /** Engine version as the source states it (e.g. "0.10.1", "v0.4.9.post2", "nightly 2025-06-10"). */
  engineVersion: z.string().min(1),
  source: z.strictObject({ url, title: z.string().min(1), kind: referenceSourceKindSchema, date: z.iso.date().nullable() }),
  model: z.strictObject({
    hfId: z.string().regex(/^[^/\s]+\/[^/\s]+$/),
    /** Weights as loaded: "bf16", "fp8", "nvfp4", "mxfp4", "awq-int4", "gptq-int4", "w4a16"... */
    quantization: z.string().min(1),
    moe: z.boolean(),
  }),
  hardware: z.strictObject({
    /** An id of hardware.json. */
    gpuId: z.string().min(1),
    gpuCount: count,
    tp: count,
    pp: count,
    /** vLLM: data-parallel replicas; SGLang: --dp-size (with --enable-dp-attention it splits attention inside tp). */
    dp: count,
    /** Expert parallelism enabled (vLLM --enable-expert-parallel, SGLang --ep-size > 1). */
    ep: z.boolean(),
    nodes: count,
  }),
  params: z.strictObject({
    gpuMemoryUtilization: z.number().gt(0).lte(1).nullable(),
    memFractionStatic: z.number().gt(0).lte(1).nullable(),
    maxModelLen: count.nullable(),
    maxNumSeqs: count.nullable(),
    maxNumBatchedTokens: count.nullable(),
    /** SGLang --chunked-prefill-size. */
    chunkedPrefillSize: z.number().int().nullable(),
    /** "auto", "bf16", "fp8", "fp8_e4m3", "fp8_e5m2"... */
    kvCacheDtype: z.string().min(1),
    blockSize: count.nullable(),
    enforceEager: z.boolean().nullable(),
    cudaGraphMaxBs: count.nullable(),
    /** Any other flag that changes memory or speed (e.g. "--enable-prefix-caching", "--speculative-config"). */
    other: z.array(z.string()),
  }),
  /** Numbers read from the engine's startup log. Memory values are per GPU (one rank), in GiB as printed. */
  log: z
    .strictObject({
      modelLoadingGiB: nullablePositive,
      modelLoadingSeconds: nullablePositive,
      availableKvCacheGiB: nullablePositive,
      /** vLLM "# GPU blocks" / "num_gpu_blocks" as printed. */
      gpuBlocks: count.nullable(),
      /** vLLM "GPU KV cache size: N tokens"; SGLang "KV Cache is allocated. #tokens: N". */
      kvCacheTokens: count.nullable(),
      /** vLLM "Maximum concurrency for N tokens per request: X". */
      maximumConcurrency: nullablePositive,
      maximumConcurrencyTokensPerRequest: count.nullable(),
      cudaGraphSeconds: nullablePositive,
      cudaGraphGiB: z.number().nonnegative().nullable(),
      /** vLLM memory profiling: total GPU memory, non-torch memory, PyTorch activation peak, in GiB as printed. */
      totalGpuMemoryGiB: nullablePositive,
      nonTorchGiB: z.number().nonnegative().nullable(),
      torchPeakIncreaseGiB: z.number().nonnegative().nullable(),
      /** SGLang "max_total_num_tokens". */
      maxTotalNumTokens: count.nullable(),
      /** SGLang "KV Cache is allocated. ... K size: X GB, V size: Y GB" (sum of K and V, GB as printed). */
      kvCacheAllocatedGB: nullablePositive,
      /** SGLang "Load weight end ... mem usage=X GB, avail mem=Y GB". */
      weightLoadMemUsageGB: nullablePositive,
      availMemAfterWeightsGB: z.number().nonnegative().nullable(),
      /** SGLang "Capture cuda graph end ... mem usage=X GB. avail mem=Y GB". */
      cudaGraphMemUsageGB: z.number().nonnegative().nullable(),
      /** SGLang "avail mem=X GB" after the KV cache / CUDA graphs, as printed. */
      availableMemAfterGB: z.number().nonnegative().nullable(),
      /** The log lines the numbers above were read from, verbatim. */
      rawLines: z.array(z.string().min(1)).min(1),
    })
    .nullable(),
  perf: z.array(referencePerfSchema),
  notes: z.string().optional(),
});
export type Reference = z.infer<typeof referenceSchema>;

export const referencesFileSchema = z.strictObject({
  $comment: z.string().optional(),
  schemaVersion: z.literal(1),
  checkedOn: z.iso.date(),
  entries: z.array(referenceSchema).min(1),
});
export type ReferencesFile = z.infer<typeof referencesFileSchema>;

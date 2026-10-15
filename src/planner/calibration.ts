/**
 * Efficiencies of the speed model (perf.ts), as ranges, per engine version. They are fitted by
 * scripts/dev/calibrate.mjs on measured runs: an RTX 5090 (vLLM runs with dummy weights, test/fixtures/rtx5090) and
 * the public vLLM / SGLang benchmarks in data/references.json; the measured errors the script
 * finds are what the planner declares (CALIBRATION_ERRORS, written by the script).
 */
import type { MiBRange } from '../model/serving-fit';
import type { Gpu, ParallelShape } from './hardware';

export type Range = MiBRange;

export interface Calibration {
  engine: 'vllm' | 'sglang';
  /** The engine version the numbers hold for (vllm/vllm-openai:v0.31.0, lmsysorg/sglang 0.5.21). */
  engineVersion: string;
  /** Share of the memory bandwidth a decode step gets reading weights. */
  decodeWeightEff: Range;
  /** Share of the bandwidth reading the KV cache (one long streaming read per sequence). */
  kvReadEff: Range;
  /** Share of the tensor TFLOPS when a large batch makes decode compute-bound. */
  decodeMfu: Range;
  /** Fixed time per decode step (scheduler, sampling), ms. */
  stepOverheadMs: Range;
  /** Fixed time per layer of every forward step (kernel launches, small kernels), µs. */
  layerOverheadUs: Range;
  /** Extra fixed time of a MoE layer (router, top-k, grouped GEMM launches), µs. */
  moeLayerExtraUs: Range;
  /** Time each expert the step touches adds to its MoE layer (one more small GEMM tile), µs. */
  moeExpertUs: Range;
  /** Time each sequence of the batch adds per layer and step (attention launch, sampling share), µs. */
  seqLayerUs: Range;
  /** Fixed time of every request before its first token (HTTP, tokenizer, scheduling, first step), ms. */
  requestOverheadMs: Range;
  /** Share of the tensor TFLOPS the prefill's linear layers get. */
  prefillMfu: Range;
  /**
   * The share a low-precision format's linear layers reach of that format's own peak, relative to the
   * bf16 share (prefillMfu, decodeMfu): fp8 and fp4 kernels stay further from their peaks.
   */
  formatMfuScale: { fp8: number; fp4: number };
  /** The same for the bandwidth a decode step gets reading weights of that format (its GEMV kernels). */
  formatBwScale: { fp8: number; fp4: number };
  /** Time a recurrent layer (Gated DeltaNet, linear attention, Mamba) spends per prompt token, µs (its scan kernels). */
  recurrentTokenUs?: Range;
  /** Share of the bf16 TFLOPS attention gets at `tokens` of context (it falls with very long contexts). */
  prefillAttnEff: (tokens: number) => Range;
  /** Latency of one all-reduce call per link kind, µs. */
  allReduceLatencyUs: Record<Gpu['link'], number>;
  /** Share of the link bandwidth an all-reduce gets. */
  linkEff: Range;
  /**
   * A card measured directly with this engine: the efficiencies above are single fitted values and the
   * time of a decode step and of a prompt read get this relative spread (the largest fit error).
   */
  spread?: { decode: number; prefill: number };
  /** The card the calibration was measured on (undefined: the generic one). */
  measuredOn?: string;
}

/** Attention efficiency by context: measured on the RTX 5090 up to 213K tokens, held flat beyond. */
const ATTN_POINTS: [tokens: number, min: number, max: number][] = [
  [4096, 0.45, 0.85],
  [32768, 0.45, 0.8],
  [131072, 0.35, 0.72],
  [212992, 0.2, 0.6],
];

export function attnEffAt(tokens: number): Range {
  const pts = ATTN_POINTS;
  const first = pts[0] as (typeof pts)[0];
  const last = pts[pts.length - 1] as (typeof pts)[0];
  if (tokens <= first[0]) return { min: first[1], max: first[2] };
  if (tokens >= last[0]) return { min: last[1], max: last[2] };
  const i = pts.findIndex((p) => p[0] >= tokens);
  const [t1, a1, b1] = pts[i - 1] as (typeof pts)[0];
  const [t2, a2, b2] = pts[i] as (typeof pts)[0];
  const f = Math.log(tokens / t1) / Math.log(t2 / t1);
  return { min: a1 + f * (a2 - a1), max: b1 + f * (b2 - b1) };
}

/** A single value as a range. */
const at = (x: number): Range => ({ min: x, max: x });

/** The attention curve (attnEffAt), scaled to `atFourK` at 4K tokens. */
const attnScaled = (atFourK: number) => (tokens: number) => {
  const r = attnEffAt(tokens);
  return at((atFourK * (r.min + r.max)) / 1.3);
};

/**
 * vLLM v0.31.0 (vllm/vllm-openai:v0.31.0) on every card without its own
 * measurements: single values fitted on the public vLLM benchmarks of data/references.json
 * (dense Llama 3.x 8B / 70B on H100, H200, B200 from TP1 to TP8 at 4–64 requests, SemiAnalysis
 * InferenceX and vLLM issues and PRs; MoE models), the spread the fit error: 90 % of them within ±12 %.
 * Compared at their concurrency: generation with pure decode, TPOT with the other requests' prompts read
 * between steps, TTFT at up to 8 requests (with more, the benchmarks send all prompts at once and the
 * first wait is a start-up queue the steady-state model does not describe).
 */
export const VLLM_CALIBRATION: Calibration = {
  engine: 'vllm',
  engineVersion: '0.31.0',
  decodeWeightEff: at(0.966),
  kvReadEff: at(0.519),
  decodeMfu: at(0.942),
  stepOverheadMs: at(0),
  layerOverheadUs: at(45),
  moeLayerExtraUs: at(85),
  moeExpertUs: at(1.33),
  seqLayerUs: at(0.41),
  requestOverheadMs: { min: 15, max: 30 },
  prefillMfu: at(0.75),
  formatMfuScale: { fp8: 0.666, fp4: 0.155 },
  formatBwScale: { fp8: 0.901, fp4: 0.333 },
  prefillAttnEff: attnScaled(1),
  allReduceLatencyUs: { nvlink: 8, 'pcie-p2p': 20, 'pcie-host': 45 },
  linkEff: at(0.65),
  spread: { decode: 0.12, prefill: 0.25 },
};

/**
 * SGLang (lmsysorg/sglang 0.5.21), fitted the same way on its public benchmarks
 * (the SGLang cookbook: large MoE models on H100, H200, B200; no dense model): 90 % within ±25 %, the
 * spread ±30 %.
 */
export const SGLANG_CALIBRATION: Calibration = {
  engine: 'sglang',
  engineVersion: '0.5.21',
  decodeWeightEff: at(1),
  kvReadEff: at(0.4),
  decodeMfu: at(0.906),
  stepOverheadMs: at(4),
  layerOverheadUs: at(42.6),
  moeLayerExtraUs: at(0),
  moeExpertUs: at(0),
  seqLayerUs: at(0.553),
  requestOverheadMs: { min: 60, max: 120 },
  prefillMfu: at(0.75),
  formatMfuScale: { fp8: 0.18, fp4: 0.135 },
  formatBwScale: { fp8: 0.685, fp4: 1 },
  prefillAttnEff: attnScaled(1),
  allReduceLatencyUs: { nvlink: 8, 'pcie-p2p': 20, 'pcie-host': 45 },
  linkEff: at(0.65),
  spread: { decode: 0.3, prefill: 0.3 },
};

/** The architecture families the speed calibration distinguishes (multimodal models count as their text family). */
export type SpeedFamily = 'dense' | 'moe' | 'hybrid' | 'mla';

/** A model family as the speed calibration groups it. */
export function speedFamily(family: string): SpeedFamily {
  return family === 'moe' || family === 'hybrid' || family === 'mla' ? family : 'dense';
}

/** A family measured on a card: its own fitted values, the formats and context it covers, how many points. */
interface MeasuredFamily {
  cal: Partial<Calibration>;
  formats: ParallelShape['weights'][];
  /** Longest context measured, tokens. */
  maxContext: number;
  points: number;
  /** The measured models (cut to a few layers where named). */
  models: string;
  /** Why nothing beyond maxContext was measured, said when a context goes past it. */
  beyond?: string;
}

/** Fitted values as a calibration: point ranges, the attention efficiency from 4K to 1M tokens (log-linear). */
function fitted(f: { w: number; k: number; L: number; M: number; E: number; S: number; O: number; P: number; A: number; AL: number; D: number; F8: number; F4: number; B8: number; B4: number; G: number }, spread: number): Partial<Calibration> {
  return {
    decodeWeightEff: at(f.w),
    kvReadEff: at(f.k),
    decodeMfu: at(f.D),
    stepOverheadMs: at(f.O),
    layerOverheadUs: at(f.L),
    moeLayerExtraUs: at(f.M),
    moeExpertUs: at(f.E),
    seqLayerUs: at(f.S),
    requestOverheadMs: { min: 25, max: 150 },
    prefillMfu: at(f.P),
    formatMfuScale: { fp8: f.F8 / f.P, fp4: f.F4 / f.P },
    formatBwScale: { fp8: f.B8 / f.w, fp4: f.B4 / f.w },
    recurrentTokenUs: at(f.G),
    prefillAttnEff: (tokens) => at(f.A * (tokens <= 4096 ? 1 : 1 + (f.AL - 1) * (Math.log(tokens / 4096) / Math.log(256)))),
    spread: { decode: spread, prefill: spread },
  };
}

/**
 * vLLM v0.31.0 on the RTX 5090, fitted per family (as Vidur and GenZ fit per operator: GEMM efficiency by
 * format, attention efficiency by context, fixed cost per step and per layer, MoE cost per touched expert,
 * recurrent scan cost per token) on clean runs only — no other program using the CPU (vLLM's scheduler needs
 * it), no GPU memory spilled to system RAM, the server as fast at the end of a run as at its start
 * (scripts/dev/speed.mjs; test/fixtures/rtx5090/speed.json and speed-long.json). The spread is the
 * family's worst fit error. Measured on Windows 11 with Docker Desktop / WSL 2: a Linux server may generate faster.
 */
const RTX5090_FAMILIES: Record<SpeedFamily, MeasuredFamily> = {
  dense: {
    // Worst ±20 %, mean ±6 % on 18 points.
    cal: fitted({ w: 1, k: 0.594, L: 51.9, M: 73.7, E: 6.19, S: 0, O: 1.434, P: 1.116, A: 0.509, AL: 1.314, D: 0.629, F8: 0.283, F4: 0.306, B8: 0.942, B4: 0.515, G: 13.7 }, 0.22),
    formats: ['bf16'],
    maxContext: 1_048_576,
    points: 18,
    models: 'Qwen3-8B BF16 (whole, and cut to 2 layers from 64K to 1M)',
  },
  moe: {
    // Mean ±6 % on 33 points; the spread covers every one (worst ±15 %).
    cal: fitted({ w: 0.928, k: 0.797, L: 116.5, M: 0, E: 1.684, S: 0, O: 0.476, P: 1.16, A: 0.576, AL: 1.416, D: 0.676, F8: 0.79, F4: 0.515, B8: 0.907, B4: 0.754, G: 8.69 }, 0.15),
    formats: ['fp8'],
    maxContext: 524_288,
    points: 33,
    models: 'Qwen3-30B-A3B FP8 (24 layers), Qwen3-235B-A22B FP8 (6 layers; 2 layers from 64K to 512K)',
  },
  hybrid: {
    // Worst ±7 %, mean ±3 % on 23 points.
    cal: fitted({ w: 0.993, k: 1, L: 13.5, M: 14.5, E: 2.86, S: 0.148, O: 2.475, P: 0.664, A: 0.642, AL: 0.986, D: 0.473, F8: 0.562, F4: 0.28, B8: 1, B4: 0.669, G: 0.127 }, 0.1),
    formats: ['fp8', 'nvfp4'],
    maxContext: 1_048_576,
    points: 23,
    models: 'Qwen3.8-27B FP8 (4 layers, 64K to 1M) and NVFP4 (16 layers, 1K to 128K)',
  },
  mla: {
    // Worst ±21 %, mean ±10 % on 13 points.
    cal: fitted({ w: 0.642, k: 1, L: 7.51, M: 47.5, E: 0, S: 0, O: 1.841, P: 1.127, A: 1.3, AL: 1.5, D: 0.458, F8: 1, F4: 0.632, B8: 1, B4: 0.68, G: 2.21 }, 0.23),
    formats: ['fp8'],
    maxContext: 131_072,
    points: 13,
    models: 'GLM-5.3-Flash FP8 (sparse MLA + linear attention, 4 layers, to 128K)',
    beyond: 'the 262K run on the RTX 5090 spilled GPU memory to system RAM under Windows, so it is not a valid measurement: a limit of that machine, not of the model',
  },
};

/** Cards measured directly, per engine. */
const MEASURED: Record<string, { name: string; families: Record<SpeedFamily, MeasuredFamily> }> = {
  'vllm:rtx-5090-32gb': { name: 'RTX 5090', families: RTX5090_FAMILIES },
};

/** The measured family of a card for this engine and format, if any. */
function measuredFamily(engine: 'vllm' | 'sglang', gpu: Pick<Gpu, 'id'> | undefined, weights: ParallelShape['weights'] | undefined, family: string): MeasuredFamily | undefined {
  const card = gpu ? MEASURED[`${engine}:${gpu.id}`] : undefined;
  const f = card?.families[speedFamily(family)];
  return f && weights && f.formats.includes(weights) ? f : undefined;
}

/** The calibration for an engine: the card's own for this family and format when measured, else the generic one. */
export function calibrationFor(engine: 'vllm' | 'sglang', gpu?: Pick<Gpu, 'id'>, weights?: ParallelShape['weights'], family = 'dense'): Calibration {
  const generic = engine === 'vllm' ? VLLM_CALIBRATION : SGLANG_CALIBRATION;
  const own = measuredFamily(engine, gpu, weights, family);
  return own ? { ...generic, ...own.cal, measuredOn: (gpu as Pick<Gpu, 'id'>).id } : generic;
}

/** How a speed estimate is founded. */
export interface SpeedBasis {
  /** The calibration to compute with: the declared spread already widened for an unverified case. */
  cal: Calibration;
  kind: 'measured' | 'fitted' | 'estimated';
  /** What the numbers rest on. */
  base: string;
  /** Declared error of generation and of prompt reading, in % (the spread applied). */
  decodePct: number;
  prefillPct: number;
  /** Context the calibration has seen, tokens; beyond it the estimate is an extrapolation. */
  validUpTo: number;
  /** Measured or public points behind it. */
  points: number;
  /** Why the interval was widened (unverified combination, extrapolated context). */
  widened?: string;
}

const ENGINE_NAME = { vllm: 'vLLM', sglang: 'SGLang' } as const;

/**
 * Public points each generic calibration was fitted on: the entries of test/fixtures/calibration-points.json on cards
 * with published bandwidth and TFLOPS, outliers left out (test/planner/calibration.test.ts recounts them).
 */
export const PUBLIC_POINTS = { vllm: 43, sglang: 24 } as const;

/**
 * The family × format combinations the public benchmarks hold points for: only these are "fitted", the rest is
 * estimated (test/planner/calibration.test.ts checks this list against the points). vLLM: dense models in BF16 / FP8
 * and MoE in MXFP4; SGLang: MoE in BF16 / FP8 / NVFP4 and MLA in FP8.
 */
export const FITTED: Record<'vllm' | 'sglang', string[]> = { vllm: ['dense bf16', 'dense fp8', 'moe mxfp4'], sglang: ['moe bf16', 'moe fp8', 'moe nvfp4', 'mla fp8'] };

export function publiclyFitted(engine: 'vllm' | 'sglang', family: string, weights: ParallelShape['weights']): boolean {
  return FITTED[engine].includes(`${speedFamily(family)} ${weights}`);
}
/** Longest input of those public benchmarks, tokens. */
export const PUBLIC_MAX_INPUT = { vllm: 8192, sglang: 4194 } as const;

/** How much wider the speed ranges of a pipeline layout are: no pipeline run was measured. */
export const PIPELINE_WIDENING = 1.5;

/**
 * The speed basis of an engine × card × weight format × family at a context: measured on the card for this
 * family and format, fitted on public benchmarks of the same family and format, or estimated (spread doubled);
 * beyond the context the calibration has seen, the spread grows by half again, and so does it with pipeline stages,
 * whose speed no run measured.
 */
export function speedBasis(engine: 'vllm' | 'sglang', gpu: Pick<Gpu, 'id' | 'name'>, weights: ParallelShape['weights'], family: string, contextTokens: number, stages = 1): SpeedBasis {
  const own = measuredFamily(engine, gpu, weights, family);
  const cal = calibrationFor(engine, gpu, weights, family);
  const text = speedFamily(family);
  let kind: SpeedBasis['kind'];
  let base: string;
  let validUpTo: number;
  let points: number;
  let factor = 1;
  const why: string[] = [];
  if (own) {
    kind = 'measured';
    validUpTo = own.maxContext;
    points = own.points;
    base = `measured on ${gpu.name} with ${ENGINE_NAME[engine]} ${cal.engineVersion}: ${own.models}; ${own.points} points, on Windows / WSL 2`;
  } else {
    const fit = publiclyFitted(engine, family, weights);
    validUpTo = PUBLIC_MAX_INPUT[engine];
    points = PUBLIC_POINTS[engine];
    if (fit) {
      kind = 'fitted';
      base = `fitted on ${PUBLIC_POINTS[engine]} public ${ENGINE_NAME[engine]} benchmarks (${engine === 'vllm' ? 'dense models, MoE in MXFP4' : 'MoE and MLA models'} on H100 / H200 / B200)`;
    } else {
      kind = 'estimated';
      factor = 2;
      base = `estimated: ${text} ${weights.toUpperCase()} not verified with ${ENGINE_NAME[engine]}`;
      why.push('combination not verified: interval doubled');
    }
  }
  // No run with pipeline stages was measured (one public multi-node benchmark only, METHODOLOGY §7.3): estimated.
  if (stages > 1) {
    // Said in every basis, an estimated one too (B-PP).
    base = `${base}; pipeline speed not measured`;
    kind = 'estimated';
    factor *= PIPELINE_WIDENING;
    why.push(`pipeline speed not measured: interval widened by half`);
  }
  if (contextTokens > validUpTo) {
    factor *= 1.5;
    why.push(`context beyond the ${Math.round(validUpTo / 1024)}K tokens verified: interval widened by half${own?.beyond ? ` (${own.beyond})` : ''}`);
  }
  // Every calibration of this file has a spread.
  const s = cal.spread as { decode: number; prefill: number };
  const cap = (x: number) => Math.min(0.95, x);
  const decode = cap(s.decode * factor);
  const prefill = cap(s.prefill * factor);
  return {
    cal: { ...cal, spread: { decode, prefill } },
    kind,
    base,
    decodePct: Math.round(decode * 100),
    prefillPct: Math.round(prefill * 100),
    validUpTo,
    points,
    ...(why.length ? { widened: why.join('; ') } : {}),
  };
}

/**
 * How well the speed model matches the measurements (test/fixtures/calibration-points.json,
 * checked by test/planner/calibration-points.test.ts): every RTX 5090 point and every public
 * point is inside the predicted range or within `tolerance` of it, except the outliers below.
 */
export const CALIBRATION_CHECK = {
  tolerance: 0.05,
  outliers: {
    'sglang-kimi-k25-h200x8-tp8-cookbook':
      'Kimi K2.5 (INT4 W4A16, MoE) on H200 with SGLang: up to 90 % below the prediction and inconsistent with each other; INT4 MoE on Hopper is not covered by the model',
    'vllm-gemma4-31b-h200x1-pr-39866': 'a vLLM PR with a 65 000-token prompt and 2 tok/s per user: anomalous, response length unclear',
    'vllm-qwen3-235b-2507-bf16-h200x4-tp4-issue-25540':
      'Qwen3-235B (MoE) with vLLM 0.10.2 on H200 TP4, 10 240 tokens: prompt reading 3–4 times the compute; the issue itself doubts it (FP8 slower than BF16, open question)',
    'vllm-qwen3-235b-2507-fp8-h200x4-tp4-issue-25540': 'the same run in FP8 (issue 25540): the same prompt-reading anomaly',
    'sglang-minimax-m25-b200x8-tp8-ep8-cookbook':
      'MiniMax-M2.5 on 8 B200 with expert parallel (EP8) in SGLang: generation up to 30 % below the prediction at high concurrency, while the same model on H100 is inside; expert parallel is not modelled',
    'sglang-step35-flash-h200x4-tp4-ep4-cookbook': 'Step-3.5-Flash on 4 H200 with expert parallel (EP4) in SGLang: with 100 requests generation is 12 % below the prediction; expert parallel is not modelled',
  } as Record<string, string>,
  /** "Model loading took" lines the weights check leaves out, with why. */
  weightOutliers: {
    'vllm-qwen36-27b-nvfp4-rtxpro6000-kvnvfp4-ctx1m-issue-49011':
      'Qwen3.6-27B NVFP4 with nvfp4 KV at 1M on a vLLM development build: +22 % on the weights, mixed NVFP4 + FP8 format; to be measured again',
  } as Record<string, string>,
} as const;

/** The caveat when public measurements are far below the speed model's estimate: no figure to rely on. */
export const SPEED_FAR_BELOW = 'INT4 MoE without Blackwell: public measurements (Kimi K2.5 on H200) are far below the estimate, speed not reliable';

/** Combinations whose speed the model cannot vouch for (said in the output). */
export function speedCaveat(weights: string, moe: boolean, gpu: Gpu): string | undefined {
  if (gpu.bandwidthGBs <= 0 || gpu.tflops.bf16 === null) return 'speed not estimable: card bandwidth or TFLOPS not published';
  if (weights === 'int4' && moe && gpu.cc < 10) return SPEED_FAR_BELOW;
  return undefined;
}

/**
 * Which weight formats × architecture families have their "weights per GPU" checked against what vLLM
 * itself reported ("Model loading took X GiB"), from data/references.json and RTX 5090
 * runs (scripts/dev/calibrate.mjs prints the comparison). The computation is deterministic:
 * once a format's and a family's loading rules are confirmed, they hold for every model using them.
 *  - verified: computed = reported within 0.06 GiB on every measurement;
 *  - gap: the engine holds more than the tensors (kernel repacking, scale layouts, extra copies) by up to
 *    `gapShare`; until a measurement explains it, the fit verdict counts it as a margin;
 *  - a combination outside this list is unverified and gets UNVERIFIED_GAP.
 */
import type { ParallelShape } from './hardware';

export type ModelFamily = 'dense' | 'moe' | 'mla' | 'hybrid' | 'multimodal';

export interface WeightsCheck {
  format: ParallelShape['weights'];
  family: ModelFamily;
  status: 'verified' | 'gap';
  /** Largest share of the weights the engine added beyond the tensors on these measurements. */
  gapShare: number;
  /** A fixed amount the engine adds whatever the size (measured with cut-down models), MiB; 0 when none. */
  gapFixedMiB?: number;
  /** The measurements: model, engine version, hardware, reported vs computed GiB (reference id). */
  evidence: string[];
}

/** Margin on the weights of a combination never measured (the largest measured gaps are 3–13 %). */
export const UNVERIFIED_GAP = 0.05;

export const WEIGHTS_CHECKS: WeightsCheck[] = [
  {
    format: 'bf16',
    family: 'dense',
    status: 'verified',
    gapShare: 0.008,
    evidence: [
      'Qwen3-8B, vLLM 0.26, RTX 5090: 15.27 GiB reported = 15.27 calculated (vllm-qwen3-8b-rtx5090-kvauto-issue-50705)',
      'Qwen3-8B, vLLM 0.11, RTX PRO 6000: 15.27 = 15.27 (vllm-qwen3-8b-rtxpro6000server-issue-27649)',
      'Qwen2.5-72B, vLLM 0.10, A100 TP2 PP2: 33.93 vs 33.89 (vllm-qwen25-72b-a100sxm80x4-tp2pp2-issue-22430)',
      'Qwen3-8B, vLLM 0.31.0, RTX 5090 (dummy weights): 15.27 = 15.27 (test/fixtures/rtx5090/measurements.json)',
      'Qwen3-4B / Qwen2.5-7B, vLLM 0.10–0.11: +0.05–0.06 GiB (RoPE cache in fp32 in older versions)',
    ],
  },
  {
    format: 'bf16',
    family: 'moe',
    status: 'verified',
    gapShare: 0.008,
    evidence: ['Qwen3-VL-30B-A3B, vLLM 0.11, RTX A6000 TP2: 29.24 vs 29.01 (+0.8 %, vision encoder) (vllm-qwen3vl-30b-a3b-a6000x2-tp2-issue-26420)'],
  },
  {
    format: 'bf16',
    family: 'multimodal',
    status: 'gap',
    gapShare: 0.018,
    evidence: ['Gemma 4 12B, vLLM 0.31.0, RTX 5090 (dummy weights, --language-model-only): 22.83 vs 22.43 (+0.40 GiB: part of the vision tower stays loaded)'],
  },
  {
    format: 'fp8',
    family: 'moe',
    status: 'verified',
    gapShare: 0.001,
    evidence: [
      'Qwen3-235B-A22B FP8, vLLM 0.10, H200 TP2: 110.21 vs 110.20 (vllm-qwen3-235b-fp8-h200x2-tp2-issue-19483)',
      'Qwen3-30B-A3B FP8, vLLM 0.10, L40S: 29.10 vs 29.09 (vllm-qwen3-30b-a3b-fp8-l40s-issue-20611)',
    ],
  },
  {
    format: 'fp8',
    family: 'dense',
    status: 'gap',
    gapShare: 0.015,
    evidence: ['Qwen3-4B FP8, vLLM 0.10, RTX 4090: 4.23 vs 4.17 (+0.06 GiB) (vllm-qwen3-4b-fp8-rtx4090-v1-issue-24436)'],
  },
  {
    format: 'fp8',
    family: 'hybrid',
    status: 'gap',
    gapShare: 0.033,
    gapFixedMiB: 110,
    evidence: [
      'Qwen3.8-27B FP8 cut to 4 and 8 layers, vLLM 0.31.0, RTX 5090: 6.29 vs 6.19 and 7.71 vs 7.61 (+0.10 GiB fixed, not per layer)',
      'Qwen3.5-397B-A17B FP8, vLLM 0.16, H200 TP4: 94.02 vs 93.33 (+0.7 %) (vllm-qwen35-397b-fp8-h200nvlx4-tp4-dcp2-issue-35057)',
      'Qwen3.5-122B-A10B FP8, vLLM 0.16, H200 TP2: 58.68 vs 58.19 (+0.8 %) (vllm-qwen35-122b-fp8-h200x2-tp2-issue-35191)',
      'Qwen3.5-397B-A17B FP8, vLLM 0.18, A100 TP2 PP4: 48.98 vs 47.46 (+3.2 %: on Ampere vLLM converts the FP8 for Marlin) (vllm-qwen35-397b-fp8-a100sxm80x8-tp2pp4-issue-37972)',
    ],
  },
  {
    format: 'fp8',
    family: 'mla',
    status: 'gap',
    gapShare: 0.026,
    evidence: ['GLM-5.2 FP8, vLLM dev, B200 TP8: 90.42 vs 88.14 (+2.6 %) (vllm-glm52-fp8-b200x8-tp8-issue-46367)'],
  },
  {
    format: 'nvfp4',
    family: 'hybrid',
    status: 'gap',
    gapShare: 0.0135,
    gapFixedMiB: 125,
    evidence: [
      'the same cut to 4 and 8 layers (dummy weights): 4.16 vs 4.05, 5.15 vs 5.03, with the vision tower 5.03 vs 4.91: +0.11–0.12 GiB fixed', 'Qwen3.8-27B NVFP4, vLLM 0.31.0, RTX 5090: 19.92 vs 19.66 (+1.3 %) (test/fixtures/rtx5090)'],
  },
  {
    format: 'nvfp4',
    family: 'mla',
    status: 'gap',
    gapShare: 0.036,
    evidence: ['GLM-5.2 NVFP4, vLLM 0.26, B200 TP8 EP: 56.13 vs 54.16 (+3.6 %) (vllm-glm52-nvfp4-b200x8-tp8-ep-issue-49475)'],
  },
  {
    format: 'int4',
    family: 'multimodal',
    status: 'gap',
    gapShare: 0.055,
    evidence: ['Gemma 4 31B AWQ, vLLM, 2× RTX 3090 TP2: +0.54 GiB (+5.4 %) (vllm-gemma4-31b-awq4-rtx3090x2-tp2-issue-39133)'],
  },
  {
    format: 'int4',
    family: 'dense',
    status: 'verified',
    gapShare: 0.005,
    evidence: ['Llama-3.1-405B AWQ, vLLM 0.8, H100 NVL TP4: 51.15 vs 51.09 (vllm-llama31-405b-awq-h100nvlx4-tp4-issue-16259)'],
  },
  {
    format: 'int4',
    family: 'moe',
    status: 'verified',
    gapShare: 0.003,
    evidence: [
      'Qwen3-Coder-30B AWQ, vLLM 0.12, RTX 3090 TP2: 7.92 vs 7.90 (vllm-qwen3coder30b-awq-rtx3090x2-tp2-ep-issue-30172)',
      'MiniMax-M3 W4A16, vLLM 0.25, H200 TP4: 52.35 vs 52.36 (vllm-minimax-m3-w4a16-h200nvlx4-tp4-issue-48493)',
    ],
  },
  {
    format: 'mxfp4',
    family: 'moe',
    status: 'gap',
    gapShare: 0.13,
    evidence: [
      'gpt-oss-120b, vLLM 0.11, A100 / H200 TP2: 34.38 vs 30.41 (+13 %); H100 TP2 (0.23): +3.1 %; GH200 (0.16): +5.9 %',
      'gpt-oss-20b: RTX 5090 TP2 (0.17) +10.6 %, L40S +6.9 %, RTX A6000 +0.4 %',
      'gpt-oss-20b, vLLM 0.31.0, RTX 5090 (dummy weights): 13.80 vs 12.83 (+7.5 %: MXFP4 experts realigned, hidden 2880 → 3072)',
      'DeepSeek-V4-Flash, vLLM 0.25, B200 TP8: 20.09 vs 18.29 (+9.9 %)',
    ],
  },
  {
    format: 'mxfp4',
    family: 'mla',
    status: 'gap',
    gapShare: 0.13,
    evidence: ['Kimi-K3, vLLM 0.27, B300 TP16: 103.94 vs 92.65 (+12.2 %) (vllm-kimi-k3-b300x16-tp16-ep-2nodes-issue-51986)'],
  },
];

export interface Verification {
  status: 'verified' | 'gap' | 'unverified';
  gapShare: number;
  gapFixedMiB: number;
  evidence: string[];
}

/** How the weights of a format × family are verified, and the margin to add. */
export function verificationOf(format: ParallelShape['weights'], family: ModelFamily): Verification {
  // A multimodal model loads like its text family plus the tower (counted from its own tensors).
  const hit = WEIGHTS_CHECKS.find((c) => c.format === format && c.family === family) ?? (family === 'multimodal' ? WEIGHTS_CHECKS.find((c) => c.format === format && c.family === 'dense') : undefined);
  if (!hit) return { status: 'unverified', gapShare: UNVERIFIED_GAP, gapFixedMiB: 0, evidence: [] };
  return { status: hit.status, gapShare: hit.gapShare, gapFixedMiB: hit.gapFixedMiB ?? 0, evidence: hit.evidence };
}

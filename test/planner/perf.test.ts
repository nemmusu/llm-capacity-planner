import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { hardwareFileSchema } from '../../src/contract/data';
import type { Calibration } from '../../src/planner/calibration';
import { gpuFromSpec, type Gpu, type GpuLayout } from '../../src/planner/hardware';
import type { ModelProfile } from '../../src/planner/model-profile';
import { attentionFlops, decodeStepMs, linearTflops, microBatches, prefillMs } from '../../src/planner/perf';

const hw = hardwareFileSchema.parse(JSON.parse(readFileSync(path.join('data', 'hardware.json'), 'utf8')));
const catalog = (id: string): Gpu => gpuFromSpec(hw.gpus.find((g) => g.id === id) as (typeof hw.gpus)[number]);
const one = (gpu: Gpu, tp = 1, pp = 1): GpuLayout => ({ gpu, count: tp * pp, tp, pp });

/** Round numbers so every expectation can be checked by hand: best case = the first value of each range. */
const cal: Calibration = {
  engine: 'vllm',
  engineVersion: 'test',
  decodeWeightEff: { min: 0.5, max: 1 },
  kvReadEff: { min: 0.5, max: 1 },
  decodeMfu: { min: 0.5, max: 1 },
  stepOverheadMs: { min: 1, max: 2 },
  layerOverheadUs: { min: 10, max: 20 },
  moeLayerExtraUs: { min: 5, max: 10 },
  seqLayerUs: { min: 1, max: 2 },
  requestOverheadMs: { min: 10, max: 100 },
  prefillMfu: { min: 0.5, max: 1 },
  formatMfuScale: { fp8: 1, fp4: 1 },
  formatBwScale: { fp8: 1, fp4: 1 },
  moeExpertUs: { min: 0, max: 0 },
  prefillAttnEff: () => ({ min: 0.5, max: 1 }),
  allReduceLatencyUs: { nvlink: 10, 'pcie-p2p': 20, 'pcie-host': 40 },
  linkEff: { min: 0.5, max: 1 },
};
/** 1 TB/s, 100 bf16 TFLOPS, 10 GB/s through host memory. */
const gpu: Gpu = {
  id: 'test',
  name: 'Test',
  segment: 'consumer',
  architecture: 'hopper',
  cc: 9,
  cudaMiB: { min: 1000, max: 1000 },
  nominalGB: 1,
  bandwidthGBs: 1000,
  tflops: { bf16: 100, fp8: null, fp4: null },
  linkGBs: 10,
  link: 'pcie-host',
  formats: { fp8: false, nvfp4: false, mxfp4: false, mxfp8: false },
};
/** 1 GB of dense weights, 0.5 B parameters, 10 layers of hidden 1000. */
const dense: ModelProfile = {
  layers: 10,
  hidden: 1000,
  experts: 0,
  expertsPerToken: 0,
  routedExpertBytes: 0,
  denseBytes: 1e9,
  activeParams: 5e8,
  headParams: 0,
  totalParams: 5e8,
  attention: [],
  recurrentStateBytes: 0,
  recurrentLayers: 0,
};

describe('linearTflops', () => {
  it('FP4 on Blackwell, FP8 from Ada, bf16 otherwise', () => {
    const b = one(catalog('rtx-5090-32gb'));
    expect(linearTflops(b, 'nvfp4')).toBe(1676);
    expect(linearTflops(b, 'mxfp4')).toBe(1676);
    expect(linearTflops(b, 'fp8')).toBe(419);
    expect(linearTflops(b, 'int4')).toBe(209.5);
    expect(linearTflops(one(catalog('rtx-4090-24gb')), 'fp8')).toBe(330.3);
  });

  it('falls back to the bf16 kernels: FP4 below Blackwell, FP8 on Ampere, a missing figure; 100 without bf16', () => {
    expect(linearTflops(one(catalog('h100-sxm5-80gb')), 'nvfp4')).toBe(989.4);
    expect(linearTflops(one(catalog('rtx-3090-24gb')), 'fp8')).toBe(71);
    expect(linearTflops(one({ ...catalog('b200-sxm-180gb'), tflops: { bf16: 2200, fp8: null, fp4: null } }), 'nvfp4')).toBe(2200);
    expect(linearTflops(one(gpu), 'fp8')).toBe(100);
    expect(linearTflops(one(catalog('gb200-nvl72')), 'bf16')).toBe(100);
  });
});

describe('decodeStepMs', () => {
  it('dense, one GPU: weights + KV at the bandwidth, launches per layer, step overhead', () => {
    const r = decodeStepMs({ profile: dense, layout: one(gpu), batch: 1, kvBytesPerSeq: 1e8, weights: 'bf16', cal });
    // Best: (1e9 + 1e8) B / 1e12 B/s = 1.1 ms; compute 1e9 / 1e14 = 0.01 ms (memory-bound);
    // 10 layers × (10 + 1 × 1) µs = 0.11 ms; + 1 ms → 2.21 ms.
    expect(r.min).toBeCloseTo(2.21, 9);
    // Worst: 2.2 ms at half the bandwidth; 10 × (20 + 2) µs = 0.22 ms; + 2 ms → 4.42 ms.
    expect(r.max).toBeCloseTo(4.42, 9);
    // An empty batch counts as one sequence.
    expect(decodeStepMs({ profile: dense, layout: one(gpu), batch: 0, kvBytesPerSeq: 1e8, weights: 'bf16', cal })).toEqual(r);
    // Two stages: each reads half of the body, and the one with the LM head (here 1e8 parameters, read in BF16) sets
    // the pace; without a head the stages are equal.
    const headed = { ...dense, headParams: 1e8 };
    const two = (p: ModelProfile) => decodeStepMs({ profile: p, layout: one(gpu, 1, 2), batch: 2, kvBytesPerSeq: 0, weights: 'bf16', cal });
    expect(two(headed).min).toBeGreaterThan(two(dense).min);
    // Best case: the head stage reads (1e9 − 2e8) / 2 + 2e8 = 6e8 bytes at 1 TB/s, the even split 5e8.
    expect(two(headed).min - two(dense).min).toBeCloseTo(2 * 0.1, 6);
    expect(decodeStepMs({ profile: headed, layout: one(gpu), batch: 2, kvBytesPerSeq: 0, weights: 'bf16', cal })).toEqual(decodeStepMs({ profile: dense, layout: one(gpu), batch: 2, kvBytesPerSeq: 0, weights: 'bf16', cal }));
  });

  it('a large batch becomes compute-bound', () => {
    const r = decodeStepMs({ profile: dense, layout: one(gpu), batch: 1000, kvBytesPerSeq: 0, weights: 'bf16', cal });
    // Best: compute 2 × 5e8 × 1000 / 1e14 = 10 ms > memory 1 ms; 10 × (10 + 1000 × 1) µs = 10.1 ms; + 1 → 21.1.
    expect(r.min).toBeCloseTo(21.1, 9);
    // Worst: 1e12 / (1e14 × 0.5) = 20 ms at 50 % MFU; 10 × (20 + 2000) µs = 20.2 ms; + 2 → 42.2.
    expect(r.max).toBeCloseTo(42.2, 9);
  });

  it('tensor parallel: weights split, two all-reduces per layer over the host link', () => {
    const r = decodeStepMs({ profile: dense, layout: one(gpu, 2), batch: 1, kvBytesPerSeq: 1e8, weights: 'bf16', cal });
    // All-reduce of 1 × 1000 × 2 B: 40 µs + 2·1/2 × 2000 B / 10 GB/s = 0.04 + 0.0002 ms; × 20 = 0.804 ms.
    // Memory (5e8 + 1e8) / 1e12 = 0.6 ms; launches 0.11; + 1 → 2.514.
    expect(r.min).toBeCloseTo(2.514, 9);
    // Worst: (60 µs + 2000 / 5e9 s) × 20 = 1.208; memory 1.2; launches 0.22; + 2 → 4.628.
    expect(r.max).toBeCloseTo(4.628, 9);
  });

  it('pipeline parallel: micro-batches through the stages, recurrent state read and written, a hop per stage', () => {
    const r = decodeStepMs({ profile: { ...dense, recurrentStateBytes: 1e7 }, layout: one(gpu, 1, 2), batch: 4, kvBytesPerSeq: 1e8, weights: 'bf16', cal });
    // Micro-batch 2; per stage: weights 5e8, KV 2 × 1e8 / 2 = 1e8, state 2 × 2 × 1e7 / 2 = 2e7 → 0.62 ms;
    // 5 layers × (10 + 2 × 1) µs = 0.06 ms → 0.68 per stage; 2 stages + 1 ms + one 40 µs hop = 2.4.
    expect(r.min).toBeCloseTo(2.4, 9);
    // Worst: the requests stay in the micro-batch they started in, so the fullest holds all 4 (microBatches):
    // weights 1.0 + KV 4e8 / 2 at half speed 0.4 + state 0.08 = 1.48, launches 5 × (20 + 4 × 2) µs = 0.14 → 1.62
    // per stage; × 2 + 2 + 0.06 = 5.30.
    expect(r.max).toBeCloseTo(5.3, 9);
  });

  it("micro-batches: even at best; at worst the expected fullest, since vLLM never moves a running request", () => {
    expect(microBatches(5, 1)).toEqual({ min: 5, max: 5 });
    // 16 requests over 4 stages: 4 each at best; 4 + √(2 · 4 · ln 4) = 7.33 → 8 at worst.
    expect(microBatches(16, 4)).toEqual({ min: 4, max: 8 });
    // Never more than the batch, never less than one.
    expect(microBatches(2, 4)).toEqual({ min: 1, max: 2 });
    expect(microBatches(400, 2).max).toBeLessThan(230);
  });

  it('MoE: the routed experts a batch touches, an extra cost per MoE layer', () => {
    const moe: ModelProfile = { ...dense, experts: 4, expertsPerToken: 2, denseBytes: 1e8, routedExpertBytes: 4e8, activeParams: 1e8 };
    const r = decodeStepMs({ profile: moe, layout: one(gpu), batch: 1, kvBytesPerSeq: 0, weights: 'bf16', cal });
    // 1 token touches 2 of 4 experts: 1e8 + 4e8 / 2 = 3e8 B → 0.3 ms; 10 × (10 + 5 + 1) µs = 0.16; + 1 → 1.46.
    expect(r.min).toBeCloseTo(1.46, 9);
    // Worst: 0.6 + 10 × (20 + 10 + 2) µs + 2 = 2.92.
    expect(r.max).toBeCloseTo(2.92, 9);
    // 2 tokens touch 4 × (1 − ½²) = 3 experts: 1e8 + 3e8 → 0.4 ms; 10 × (10 + 5 + 2) µs = 0.17; + 1 → 1.57.
    expect(decodeStepMs({ profile: moe, layout: one(gpu), batch: 2, kvBytesPerSeq: 0, weights: 'bf16', cal }).min).toBeCloseTo(1.57, 9);
    // A cost per touched expert: 2 experts × 3 µs × 10 layers = 0.06 ms more.
    const perExpert = { ...cal, moeExpertUs: { min: 3, max: 3 } };
    expect(decodeStepMs({ profile: moe, layout: one(gpu), batch: 1, kvBytesPerSeq: 0, weights: 'bf16', cal: perExpert }).min).toBeCloseTo(1.52, 9);
  });

  it('a measured card: its fitted values, the fit error as spread, the format scales on bandwidth and TFLOPS', () => {
    const fitted: Calibration = { ...cal, decodeWeightEff: { min: 1, max: 1 }, kvReadEff: { min: 1, max: 1 }, stepOverheadMs: { min: 1, max: 1 }, layerOverheadUs: { min: 10, max: 10 }, seqLayerUs: { min: 1, max: 1 }, spread: { decode: 0.1, prefill: 0.2 } };
    // 2.21 ms (the best case of the dense test) ± 10 %.
    const r = decodeStepMs({ profile: dense, layout: one(gpu), batch: 1, kvBytesPerSeq: 1e8, weights: 'bf16', cal: fitted });
    expect(r.min).toBeCloseTo(2.21 * 0.9, 9);
    expect(r.max).toBeCloseTo(2.21 * 1.1, 9);
    // FP8 weights read at twice the bandwidth share: 0.5 + 0.1 ms instead of 1 + 0.1.
    const b = { ...catalog('rtx-5090-32gb'), bandwidthGBs: 1000 };
    const fp8 = decodeStepMs({ profile: dense, layout: one(b), batch: 1, kvBytesPerSeq: 1e8, weights: 'fp8', cal: { ...fitted, formatBwScale: { fp8: 2, fp4: 1 }, spread: undefined } });
    expect(fp8.min).toBeCloseTo(0.6 + 0.11 + 1, 9);
    // FP4 at a quarter of its peak share: linear 1e9 × 2048 / (1676e12 × 0.25 × 1) s.
    const p = prefillMs({ profile: dense, layout: one(b), tokens: 2048, batchedTokens: 2048, weights: 'nvfp4', cal: { ...cal, formatMfuScale: { fp8: 1, fp4: 0.25 } } });
    expect(p.min).toBeCloseTo((1e9 * 2048) / (1676e12 * 0.25) * 1000 + 0.1, 9);
  });
});

describe('attentionFlops', () => {
  const p: ModelProfile = {
    ...dense,
    attention: [
      { layers: 2, headDims: 10 },
      { layers: 3, headDims: 20, window: 100 },
    ],
  };

  it('n² for full attention and for windows the context does not exceed', () => {
    // 2 × 10 × 50² + 3 × 20 × 50².
    expect(attentionFlops(p, 50)).toBe(200_000);
    expect(attentionFlops(p, 100)).toBe(80 * 10_000);
  });

  it('beyond the window: 2·n·w − w²', () => {
    // 2 × 10 × 1000² + 3 × 20 × (2 × 1000 × 100 − 100²) = 2e7 + 60 × 190 000.
    expect(attentionFlops(p, 1000)).toBe(20_000_000 + 11_400_000);
  });
});

describe('prefillMs', () => {
  const p: ModelProfile = { ...dense, attention: [{ layers: 10, headDims: 1000 }] };

  it('linear layers and attention at their efficiencies, a fixed cost per chunk and layer', () => {
    const r = prefillMs({ profile: p, layout: one(gpu), tokens: 1000, batchedTokens: 512, weights: 'bf16', cal });
    // Linear 2 × 5e8 × 1000 = 1e12 → 10 ms; attention 10 × 1000 × 1000² = 1e10 → 0.1 ms;
    // 2 chunks × 10 layers × 10 µs = 0.2 ms → 10.3 ms.
    expect(r.min).toBeCloseTo(10.3, 9);
    // Worst: 20.2 ms at half efficiency + 2 × 10 × 20 µs = 0.4 → 20.6.
    expect(r.max).toBeCloseTo(20.6, 9);
    // A chunk size of 0 reads one token per chunk: 1000 chunks × 10 × 10 µs = 100 ms.
    expect(prefillMs({ profile: p, layout: one(gpu), tokens: 1000, batchedTokens: 0, weights: 'bf16', cal }).min).toBeCloseTo(110.1, 9);
  });

  it('recurrent layers: a cost per prompt token and layer; a format with its own spread', () => {
    const hybrid: ModelProfile = { ...p, recurrentLayers: 4 };
    const scan = { ...cal, recurrentTokenUs: { min: 2, max: 3 } };
    // 1000 tokens × 4 layers × 2 µs = 8 ms more (best), × 3 µs = 12 ms (worst).
    const r = prefillMs({ profile: hybrid, layout: one(gpu), tokens: 1000, batchedTokens: 512, weights: 'bf16', cal: scan });
    expect(r.min).toBeCloseTo(10.3 + 8, 9);
    expect(r.max).toBeCloseTo(20.6 + 12, 9);
    // No recurrent cost in the calibration: none added.
    expect(prefillMs({ profile: hybrid, layout: one(gpu), tokens: 1000, batchedTokens: 512, weights: 'bf16', cal }).min).toBeCloseTo(10.3, 9);
    // ±50 % would go below the physical floor: 10.1 ms of compute at the datasheet TFLOPS × 1.25.
    const spread = { ...cal, spread: { decode: 0.5, prefill: 0.5 } };
    expect(prefillMs({ profile: p, layout: one(gpu), tokens: 1000, batchedTokens: 512, weights: 'bf16', cal: spread }).min).toBeCloseTo(10.1 / 1.25, 9);
    expect(prefillMs({ profile: p, layout: one(gpu), tokens: 1000, batchedTokens: 512, weights: 'bf16', cal: { ...cal, spread: { decode: 0.1, prefill: 0.1 } } }).min).toBeCloseTo(10.3 * 0.9, 9);
    expect(decodeStepMs({ profile: p, layout: one(gpu), batch: 1, kvBytesPerSeq: 1e8, weights: 'bf16', cal: spread }).min).toBeCloseTo(2.21 * 0.5, 9);
  });

  it('tensor parallel: compute split, two all-reduces of the prompt per layer', () => {
    const r = prefillMs({ profile: p, layout: one(gpu, 2), tokens: 1000, batchedTokens: 512, weights: 'bf16', cal });
    // 5.05 ms of compute; all-reduce of 1000 × 1000 × 2 B: 0.04 + 1 × 2e6 / 1e10 s = 0.24 ms, × 20 = 4.8; + 0.2 → 10.05.
    expect(r.min).toBeCloseTo(10.05, 9);
  });

  it('a card without bf16 TFLOPS: 100 for attention too', () => {
    const r = prefillMs({ profile: p, layout: one({ ...gpu, tflops: { bf16: null, fp8: null, fp4: null } }), tokens: 1000, batchedTokens: 1000, weights: 'bf16', cal });
    // Same as the 100-TFLOPS card, with one chunk: 10.1 + 0.1.
    expect(r.min).toBeCloseTo(10.2, 9);
  });
});

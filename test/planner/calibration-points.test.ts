import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { hardwareFileSchema } from '../../src/contract/data';
import { kvElementBytes } from '../../src/model/kv';
import { schedulerDefaults } from '../../src/model/serving-fit';
import type { KvLayout } from '../../src/model/types';
import { CALIBRATION_CHECK, calibrationFor } from '../../src/planner/calibration';
import { loadPoint } from '../../src/planner/capacity';
import { gpuFromSpec, type ParallelShape } from '../../src/planner/hardware';
import { kvReadBytes, type ModelProfile } from '../../src/planner/model-profile';
import { decodeStepMs, prefillMs } from '../../src/planner/perf';
import { WEIGHTS_CHECKS } from '../../src/planner/weights-verification';

/** Written by scripts/dev/calibrate.mjs from the safetensors headers of every reference model. */
const fixture = JSON.parse(readFileSync(path.join('test', 'fixtures', 'calibration-points.json'), 'utf8')) as {
  profiles: Record<string, { profile: ModelProfile; layout: KvLayout; weights: ParallelShape['weights']; family: string; sparseTopk?: number }>;
  points: {
    id: string;
    engine: 'vllm' | 'sglang';
    repo: string;
    gpu: string;
    tp: number;
    pp: number;
    concurrency: number;
    inputLen: number;
    outputLen: number;
    kv: 'fp8' | 'auto';
    decodeTokSPerUser?: number;
    ttftMs?: number;
    basis: 'decode' | 'tpot' | 'throughput';
  }[];
  weights: { id: string; format: ParallelShape['weights']; family: string; tp: number; pp: number; reported: number; calculated: number }[];
};
const hw = hardwareFileSchema.parse(JSON.parse(readFileSync(path.join('data', 'hardware.json'), 'utf8')));
const gpus = new Map(hw.gpus.map((g) => [g.id, gpuFromSpec(g)]));

/** Measured value against the predicted range: 0 inside, else the relative distance to the nearest bound. */
const distance = (v: number, r: { min: number; max: number }) => (v < r.min ? (r.min - v) / r.min : v > r.max ? (v - r.max) / r.max : 0);

function predict(p: (typeof fixture.points)[number]) {
  const m = fixture.profiles[p.repo];
  const gpu = gpus.get(p.gpu);
  if (!m || !gpu) throw new Error(`missing ${p.repo} / ${p.gpu}`);
  const layout = { gpu, count: p.tp * p.pp, tp: p.tp, pp: p.pp };
  const cal = calibrationFor(p.engine, gpu, m.weights, m.family);
  const elem = kvElementBytes(p.engine, p.kv);
  const kvRead = kvReadBytes(m.layout, elem, p.tp, p.inputLen + p.outputLen / 2, m.sparseTopk);
  const batched = schedulerDefaults(p.engine, gpu.cudaMiB.max, p.tp).batchedTokens;
  const speed = {
    stepMs: (b: number) => decodeStepMs({ profile: m.profile, layout, batch: b, kvBytesPerSeq: kvRead, weights: m.weights, cal }),
    prefillMs: (t: number) => prefillMs({ profile: m.profile, layout, tokens: t, batchedTokens: batched, weights: m.weights, cal }),
    batchedTokens: batched,
    requestOverheadMs: cal.requestOverheadMs,
  };
  const step = speed.stepMs(p.concurrency);
  const single = speed.prefillMs(p.inputLen);
  const lp = loadPoint(speed, { maxContext: p.inputLen + p.outputLen, typicalContext: p.inputLen, outputTokens: p.outputLen, minTokS: 0, maxTtftMs: 0, activeShare: 1, safety: 1, cachedShare: 0 }, p.concurrency);
  return {
    // A TPOT lies between generation with the others' prompts read between steps and pure generation.
    decode: p.basis === 'decode' ? { min: 1000 / step.max, max: 1000 / step.min } : p.basis === 'tpot' ? { min: lp.tokS.min, max: 1000 / step.min } : lp.tokS,
    ttft: p.concurrency > 1 ? lp.ttftMs : { min: single.min + cal.requestOverheadMs.min + step.min, max: single.max + cal.requestOverheadMs.max + step.max },
    speedKnown: gpu.tflops.bf16 !== null && gpu.bandwidthGBs > 0,
  };
}

describe('speed model against measured runs (offline)', () => {
  it('the RTX 5090, vLLM 0.31.0 runs (scripts/dev/speed.mjs): every TTFT and generation speed inside the range of its family; disturbed runs left out', () => {
    const runs = fixture.points.filter((p) => p.id.startsWith('5090-vllm-'));
    expect(runs.length).toBeGreaterThanOrEqual(8);
    for (const p of runs) {
      const pr = predict(p);
      if (p.decodeTokSPerUser !== undefined) expect([p.id, distance(p.decodeTokSPerUser, pr.decode)]).toEqual([p.id, 0]);
      if (p.ttftMs !== undefined) expect([p.id, distance(p.ttftMs, pr.ttft)]).toEqual([p.id, 0]);
    }
    // The declared error per family: dense ±22 %, MoE ±15 %, Gated DeltaNet hybrid ±10 %, sparse MLA ±23 %.
    expect(['dense', 'moe', 'hybrid', 'mla'].map((f) => calibrationFor('vllm', { id: 'rtx-5090-32gb' }, f === 'dense' ? 'bf16' : 'fp8', f).spread?.decode)).toEqual([0.22, 0.15, 0.1, 0.23]);
  });

  it(`public vLLM / SGLang benchmarks: inside the range or within ${CALIBRATION_CHECK.tolerance * 100} %, outliers named with their reason`, () => {
    let checked = 0;
    for (const p of fixture.points) {
      if (p.id.startsWith('5090-') || CALIBRATION_CHECK.outliers[p.id] !== undefined) continue;
      const pr = predict(p);
      // Cards without published TFLOPS (GB200) get no speed estimate at all.
      if (!pr.speedKnown) continue;
      checked++;
      if (p.decodeTokSPerUser !== undefined) expect([p.id, p.concurrency, distance(p.decodeTokSPerUser, pr.decode) <= CALIBRATION_CHECK.tolerance]).toEqual([p.id, p.concurrency, true]);
      if (p.ttftMs !== undefined) expect([p.id, distance(p.ttftMs, pr.ttft) <= CALIBRATION_CHECK.tolerance]).toEqual([p.id, true]);
    }
    expect(checked).toBeGreaterThanOrEqual(35);
    for (const id of Object.keys(CALIBRATION_CHECK.outliers)) expect(fixture.points.some((p) => p.id === id)).toBe(true);
  });
});

describe('weights per GPU against "Model loading took" (offline)', () => {
  it('every verified format × family is within its gap; every measured gap is recorded in the matrix', () => {
    for (const w of fixture.weights) {
      // A pipeline-parallel log reports one rank, not the most loaded: not comparable.
      if (w.pp > 1 && w.tp === 1) continue;
      // Data parallel with expert parallel splits the experts in a way the TP planner does not model.
      // Speculative decoding (EAGLE, MTP drafts) loads extra weights; outliers are named in CALIBRATION_CHECK.
      if (/dp8-ep/.test(w.id) || /eagle|mtp|dspark|spec/.test(w.id) || CALIBRATION_CHECK.weightOutliers[w.id] !== undefined) continue;
      const check = WEIGHTS_CHECKS.find((c) => c.format === w.format && c.family === w.family);
      expect([w.id, check !== undefined]).toEqual([w.id, true]);
      const gap = (w.reported - w.calculated) / w.calculated;
      expect([w.id, gap <= (check?.gapShare as number) + 0.002]).toEqual([w.id, true]);
      // Never more than 0.5 % above what the engine reported (Gemma 4 26B-A4B FP8: −0.45 %, on the safe side).
      expect([w.id, gap >= -0.005]).toEqual([w.id, true]);
    }
  });
});

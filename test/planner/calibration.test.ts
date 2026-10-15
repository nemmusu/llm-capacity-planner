import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { hardwareFileSchema } from '../../src/contract/data';
import { attnEffAt, CALIBRATION_CHECK, calibrationFor, PUBLIC_POINTS, publiclyFitted, SGLANG_CALIBRATION, speedBasis, speedCaveat, speedFamily, VLLM_CALIBRATION } from '../../src/planner/calibration';
import { gpuFromSpec, type Gpu, type ParallelShape } from '../../src/planner/hardware';

const hw = hardwareFileSchema.parse(JSON.parse(readFileSync(path.join('data', 'hardware.json'), 'utf8')));
const gpu = (id: string): Gpu => gpuFromSpec(hw.gpus.find((g) => g.id === id) as (typeof hw.gpus)[number]);

describe('attnEffAt', () => {
  it('held flat below the first point and above the last', () => {
    expect(attnEffAt(1024)).toEqual({ min: 0.45, max: 0.85 });
    expect(attnEffAt(4096)).toEqual({ min: 0.45, max: 0.85 });
    expect(attnEffAt(212_992)).toEqual({ min: 0.2, max: 0.6 });
    expect(attnEffAt(1_000_000)).toEqual({ min: 0.2, max: 0.6 });
  });

  it('log-linear between the points', () => {
    // 4096 → 32768 is 3 doublings: 8192 is one third of the way, max 0.85 − (0.05 / 3).
    const r = attnEffAt(8192);
    expect(r.min).toBeCloseTo(0.45, 12);
    expect(r.max).toBeCloseTo(0.85 - 0.05 / 3, 12);
    // 65 536 is halfway (in log) between 32 768 and 131 072: (0.45 + 0.35) / 2, (0.8 + 0.72) / 2.
    const mid = attnEffAt(65_536);
    expect(mid.min).toBeCloseTo(0.4, 12);
    expect(mid.max).toBeCloseTo(0.76, 12);
    // Exactly on an inner point.
    expect(attnEffAt(131_072).min).toBeCloseTo(0.35, 12);
  });
});

describe('calibrationFor', () => {
  it('one calibration per engine, each fitted on its own public benchmarks, with its fit error as spread', () => {
    expect(calibrationFor('vllm')).toBe(VLLM_CALIBRATION);
    expect(calibrationFor('sglang')).toBe(SGLANG_CALIBRATION);
    expect(VLLM_CALIBRATION).toMatchObject({ engine: 'vllm', engineVersion: '0.31.0', spread: { decode: 0.12, prefill: 0.25 }, prefillMfu: { min: 0.75, max: 0.75 } });
    expect(SGLANG_CALIBRATION).toMatchObject({ engine: 'sglang', engineVersion: '0.5.21', spread: { decode: 0.3, prefill: 0.3 } });
    // The attention curve at 4K tokens is the fitted value, falling with the context like attnEffAt.
    expect(VLLM_CALIBRATION.prefillAttnEff(4096)).toEqual({ min: 1, max: 1 });
    expect(SGLANG_CALIBRATION.prefillAttnEff(212_992).min).toBeCloseTo(0.8 / 1.3, 12);
  });

  it('the RTX 5090 with vLLM: one calibration per family for the formats measured; anything else: the generic one', () => {
    const dense = calibrationFor('vllm', gpu('rtx-5090-32gb'), 'bf16', 'dense');
    expect(dense).toMatchObject({ engineVersion: '0.31.0', measuredOn: 'rtx-5090-32gb', spread: { decode: 0.22, prefill: 0.22 }, decodeWeightEff: { min: 1, max: 1 } });
    expect(calibrationFor('vllm', gpu('rtx-5090-32gb'), 'fp8', 'moe').spread).toEqual({ decode: 0.15, prefill: 0.15 });
    expect(calibrationFor('vllm', gpu('rtx-5090-32gb'), 'nvfp4', 'hybrid').spread).toEqual({ decode: 0.1, prefill: 0.1 });
    expect(calibrationFor('vllm', gpu('rtx-5090-32gb'), 'fp8', 'mla').spread).toEqual({ decode: 0.23, prefill: 0.23 });
    // A multimodal model counts as dense; attention efficiency log-linear from 4K to 1M tokens.
    expect(calibrationFor('vllm', gpu('rtx-5090-32gb'), 'bf16', 'multimodal').decodeWeightEff).toEqual(dense.decodeWeightEff);
    expect(dense.prefillAttnEff(1024).min).toBeCloseTo(0.509, 12);
    expect(dense.prefillAttnEff(1_048_576).min).toBeCloseTo(0.509 * 1.314, 12);
    // A format a family was not measured in, another card, SGLang, no format: generic.
    expect(calibrationFor('vllm', gpu('rtx-5090-32gb'), 'fp8', 'dense')).toBe(VLLM_CALIBRATION);
    expect(calibrationFor('vllm', gpu('rtx-5090-32gb'))).toBe(VLLM_CALIBRATION);
    expect(calibrationFor('sglang', gpu('rtx-5090-32gb'), 'fp8', 'moe')).toBe(SGLANG_CALIBRATION);
    expect(calibrationFor('vllm', gpu('h100-sxm5-80gb'), 'bf16', 'dense')).toBe(VLLM_CALIBRATION);
  });
});

describe('speedCaveat', () => {
  it('no bandwidth or no bf16 TFLOPS: no speed', () => {
    const msg = 'speed not estimable: card bandwidth or TFLOPS not published';
    expect(speedCaveat('bf16', false, gpu('gb200-nvl72'))).toBe(msg);
    expect(speedCaveat('bf16', false, { ...gpu('h100-sxm5-80gb'), bandwidthGBs: 0 })).toBe(msg);
  });

  it('INT4 MoE below Blackwell is not trusted; INT4 dense or on Blackwell is', () => {
    expect(speedCaveat('int4', true, gpu('h200-sxm-141gb'))).toMatch(/^INT4 MoE without Blackwell/);
    expect(speedCaveat('int4', false, gpu('h200-sxm-141gb'))).toBeUndefined();
    expect(speedCaveat('int4', true, gpu('b200-sxm-180gb'))).toBeUndefined();
    expect(speedCaveat('bf16', true, gpu('h200-sxm-141gb'))).toBeUndefined();
  });
});

describe('speedBasis', () => {
  const g = (id: string) => gpu(id);
  it('measured on the card for the family and format: its spread, its points and the context measured', () => {
    const b = speedBasis('vllm', g('rtx-5090-32gb'), 'bf16', 'dense', 32_768);
    expect(b).toMatchObject({ kind: 'measured', decodePct: 22, prefillPct: 22, validUpTo: 1_048_576, points: 18 });
    expect(b.widened).toBeUndefined();
    expect(b.base).toMatch(/^measured on RTX 5090 with vLLM 0\.31\.0: Qwen3-8B BF16 .*; 18 points, on Windows \/ WSL 2$/);
    // MLA beyond 128K: half again.
    const mla = speedBasis('vllm', g('rtx-5090-32gb'), 'fp8', 'mla', 262_144);
    expect(mla).toMatchObject({ kind: 'measured', validUpTo: 131_072, decodePct: 35, prefillPct: 35 });
    expect(mla.widened).toBe('context beyond the 128K tokens verified: interval widened by half (the 262K run on the RTX 5090 spilled GPU memory to system RAM under Windows, so it is not a valid measurement: a limit of that machine, not of the model)');
    expect(speedBasis('vllm', g('rtx-5090-32gb'), 'nvfp4', 'hybrid', 131_072)).toMatchObject({ kind: 'measured', decodePct: 10, points: 23 });
  });

  it('fitted on public benchmarks: vLLM dense in BF16 / FP8, SGLang MoE / MLA; multimodal counts as dense; MoE on vLLM is estimated', () => {
    expect(speedBasis('vllm', g('h100-sxm5-80gb'), 'fp8', 'dense', 8192)).toMatchObject({ kind: 'fitted', decodePct: 12, prefillPct: 25, points: 43 });
    expect(speedBasis('vllm', g('h100-sxm5-80gb'), 'bf16', 'multimodal', 4096).kind).toBe('fitted');
    // The vLLM public points hold no MoE model in BF16 / FP8 (two MXFP4 points only): estimated, spread doubled.
    expect(speedBasis('vllm', g('h100-sxm5-80gb'), 'fp8', 'moe', 8192)).toMatchObject({ kind: 'estimated', decodePct: 24, prefillPct: 50 });
    expect(speedBasis('sglang', g('b200-sxm-180gb'), 'fp8', 'mla', 4096)).toMatchObject({ kind: 'fitted', decodePct: 30, prefillPct: 30, validUpTo: 4194, points: 24 });
    expect(speedBasis('sglang', g('b200-sxm-180gb'), 'nvfp4', 'mla', 4096).kind).toBe('estimated');
    expect(speedBasis('vllm', g('h100-sxm5-80gb'), 'bf16', 'dense', 32_768)).toMatchObject({ decodePct: 18, prefillPct: 38 });
    // Pipeline stages: no run measured them, so the speed is estimated and its range widened by half, declared.
    const pp = speedBasis('vllm', g('h100-sxm5-80gb'), 'fp8', 'dense', 8192, 2);
    expect(pp).toMatchObject({ kind: 'estimated', decodePct: 18, prefillPct: 38, widened: 'pipeline speed not measured: interval widened by half' });
    expect(pp.base).toMatch(/; pipeline speed not measured$/);
    expect(speedBasis('vllm', g('h100-sxm5-80gb'), 'fp8', 'moe', 8192, 4)).toMatchObject({ kind: 'estimated', decodePct: 36 });
    expect(speedBasis('vllm', g('h100-sxm5-80gb'), 'bf16', 'dense', 8192).base).toBe('fitted on 43 public vLLM benchmarks (dense models, MoE in MXFP4 on H100 / H200 / B200)');
    expect(speedBasis('sglang', g('h100-sxm5-80gb'), 'fp8', 'moe', 4096).base).toBe('fitted on 24 public SGLang benchmarks (MoE and MLA models on H100 / H200 / B200)');
  });

  it('the public point counts and the fitted families are those of the calibration points', () => {
    const fixture = JSON.parse(readFileSync(path.join('test', 'fixtures', 'calibration-points.json'), 'utf8')) as {
      profiles: Record<string, { weights: ParallelShape['weights']; family: string }>;
      points: { id: string; engine: 'vllm' | 'sglang'; repo: string; gpu: string }[];
    };
    const priced = new Set(hw.gpus.filter((x) => x.tflopsDense.bf16 !== null && x.memoryBandwidthGBs !== null).map((x) => x.id));
    const used = fixture.points.filter((p) => !p.id.startsWith('5090-') && priced.has(p.gpu) && CALIBRATION_CHECK.outliers[p.id] === undefined);
    for (const engine of ['vllm', 'sglang'] as const) {
      const mine = used.filter((p) => p.engine === engine);
      expect([engine, mine.length]).toEqual([engine, PUBLIC_POINTS[engine]]);
      // "Fitted" exactly where the public points are: no combination is fitted on another family's points.
      const families = new Set(mine.map((p) => `${speedFamily((fixture.profiles[p.repo] as { family: string }).family)} ${(fixture.profiles[p.repo] as { weights: string }).weights}`));
      for (const family of ['dense', 'moe', 'hybrid', 'mla'])
        for (const w of ['bf16', 'fp8', 'nvfp4', 'mxfp4', 'int4', 'int8'] as const)
          expect([engine, family, w, publiclyFitted(engine, family, w)]).toEqual([engine, family, w, families.has(`${family} ${w}`)]);
    }
  });

  it('not verified: estimated, spread doubled (and capped at 95 %)', () => {
    const b = speedBasis('vllm', g('h100-sxm5-80gb'), 'int4', 'mla', 8192);
    expect(b).toMatchObject({ kind: 'estimated', decodePct: 24, prefillPct: 50 });
    expect(b.base).toBe('estimated: mla INT4 not verified with vLLM');
    expect(b.widened).toBe('combination not verified: interval doubled');
    expect(speedBasis('sglang', g('h100-sxm5-80gb'), 'bf16', 'dense', 100_000)).toMatchObject({ kind: 'estimated', decodePct: 90, prefillPct: 90 });
    expect(speedBasis('vllm', g('h100-sxm5-80gb'), 'int4', 'hybrid', 1_000_000).prefillPct).toBe(75);
  });
});

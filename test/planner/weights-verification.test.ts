import { describe, expect, it } from 'vitest';
import { UNVERIFIED_GAP, verificationOf, WEIGHTS_CHECKS } from '../../src/planner/weights-verification';

describe('verificationOf', () => {
  it('a verified combination: its measured gap and evidence', () => {
    const v = verificationOf('bf16', 'dense');
    expect(v.status).toBe('verified');
    expect(v.gapShare).toBe(0.008);
    expect(v.evidence[0]).toMatch(/^Qwen3-8B, vLLM 0\.26, RTX 5090/);
  });

  it('a combination with a gap: the largest gap measured', () => {
    // gpt-oss-120b: 34.38 GiB reported vs 30.41 computed → +13 %.
    expect(verificationOf('mxfp4', 'moe')).toMatchObject({ status: 'gap', gapShare: 0.13 });
    // Plus a fixed part measured with cut-down models (RTX 5090, vLLM 0.31.0).
    expect(verificationOf('nvfp4', 'hybrid')).toMatchObject({ status: 'gap', gapShare: 0.0135, gapFixedMiB: 125 });
    expect(verificationOf('fp8', 'hybrid')).toMatchObject({ gapFixedMiB: 110 });
  });

  it('multimodal: its own entry, else the dense family of the same format', () => {
    expect(verificationOf('fp8', 'multimodal')).toMatchObject({ status: 'gap', gapShare: 0.015, gapFixedMiB: 0 });
    // BF16 multimodal has its own measurement (Gemma 4 12B).
    expect(verificationOf('bf16', 'multimodal')).toMatchObject({ status: 'gap', gapShare: 0.018 });
    // INT4 multimodal has its own measurement (Gemma 4 31B AWQ); INT8 has none at all.
    expect(verificationOf('int4', 'multimodal')).toMatchObject({ status: 'gap', gapShare: 0.055 });
    expect(verificationOf('bf16', 'multimodal').status).toBe('gap');
    // NVFP4 has no dense entry: unverified.
    expect(verificationOf('nvfp4', 'multimodal')).toEqual({ status: 'unverified', gapShare: UNVERIFIED_GAP, gapFixedMiB: 0, evidence: [] });
  });

  it('never measured: unverified with the default margin', () => {
    expect(verificationOf('int8', 'dense')).toEqual({ status: 'unverified', gapShare: 0.05, gapFixedMiB: 0, evidence: [] });
    expect(verificationOf('bf16', 'hybrid').status).toBe('unverified');
  });

  it('every entry is unique and has evidence; verified ones stay under 1 %', () => {
    const keys = WEIGHTS_CHECKS.map((c) => `${c.format}/${c.family}`);
    expect(new Set(keys).size).toBe(keys.length);
    for (const c of WEIGHTS_CHECKS) {
      expect(c.evidence.length).toBeGreaterThan(0);
      if (c.status === 'verified') expect(c.gapShare).toBeLessThan(0.01);
    }
  });
});

import { describe, expect, it } from 'vitest';
import { applyCompat, catalogModelId } from '../../src/cli/compat';
import { customGpu } from '../../src/cli/run';
import type { CompatResult } from '../../src/planner/compat';
import { planLayout } from '../../src/planner/plan';
import { at, opts, qwen } from './fixtures';

const o = opts(32_768, { kvCache: 'fp8' });
const h100 = planLayout(qwen, at('h100-sxm5-80gb'), o);
const r5090 = planLayout(qwen, at('rtx-5090-32gb'), o);
const result = (over: Partial<CompatResult>): CompatResult => ({
  status: 'ok',
  reason: '',
  sources: [],
  wslOnly: false,
  linuxStatus: 'ok',
  wslStatus: 'ok',
  arch: 'sm90',
  engine: 'vllm',
  verifiedVersion: '0.31.0',
  repo: 'a/b',
  workarounds: [],
  details: [],
  ...over,
});
const q = { model: 'm', repo: 'a/b', engine: 'vllm' as const, kvCache: 'fp8' as const };

describe('the compatibility matrix on the planned layouts', () => {
  it('finds the catalog id of a base checkpoint or a quantization, any case', () => {
    expect(catalogModelId('zai-org/GLM-5.3')).toBe('glm-5-3');
    expect(catalogModelId('NVIDIA/GLM-5.3-NVFP4')).toBe('glm-5-3');
    expect(catalogModelId('Qwen/Qwen3-8B')).toBeUndefined();
    expect(catalogModelId('a/x', { models: [{ id: 'x', repo: 'a/y', quantizations: [{ repo: 'a/x' }] }] })).toBe('x');
  });

  it('ok: the row as it is; the matrix asked once per card', () => {
    const asked: string[] = [];
    const check = (c: { gpu: string }) => {
      asked.push(c.gpu);
      return result({});
    };
    const out = applyCompat([h100, h100, r5090], q, check as never);
    expect(out[0]).toBe(h100);
    expect(asked).toEqual(['h100-sxm5-80gb', 'rtx-5090-32gb']);
    // A pipeline layout asks again, with p:pp.
    const pipes: unknown[] = [];
    applyCompat([h100, { ...h100, layout: { ...h100.layout, count: 2, tp: 1, pp: 2 } }], q, ((c: { pipeline?: boolean }) => {
      pipes.push(c.pipeline);
      return result({});
    }) as never);
    expect(pipes).toEqual([undefined, true]);
  });

  it('does not start: no verdict, no speed, the reason and its source first among the blockers', () => {
    const [r] = applyCompat([h100], q, () => result({ status: 'does-not-start', reason: 'No sparse MLA kernel on sm90.', sources: ['https://x/1', 'https://x/2'] }));
    expect(r?.verdict).toBe('impossible');
    expect(r?.check.blockers[0]).toBe('No sparse MLA kernel on sm90. (source: https://x/1)');
    expect(r?.capacity).toBeUndefined();
    expect(r?.speedBasis).toBeUndefined();
    expect(r?.roofline).toBeUndefined();
    expect(applyCompat([h100], q, () => result({ status: 'does-not-start', reason: 'No kernel.' }))[0]?.check.blockers[0]).toBe('No kernel.');
  });

  it('partial, unknown, Windows / WSL only: warnings', () => {
    // The matrix's reason carries its own workaround (planner/compat.ts needText).
    expect(applyCompat([h100], q, () => result({ status: 'partial', reason: 'Slow MoE kernel. Workaround: Use --enforce-eager.', workarounds: ['Use --enforce-eager.'], sources: ['https://x'] }))[0]?.check.warnings.at(-1)).toBe(
      'partly supported: Slow MoE kernel. Workaround: Use --enforce-eager. (source: https://x)',
    );
    expect(applyCompat([h100], q, () => result({ status: 'partial', reason: 'Slow.' }))[0]?.check.warnings.at(-1)).toBe('partly supported: Slow.');
    expect(applyCompat([h100], q, () => result({ status: 'unknown', reason: 'Not checked.' }))[0]?.check.warnings.at(-1)).toBe('engine support on this card not verified: Not checked.');
    expect(applyCompat([h100], q, () => result({ wslOnly: true, wslStatus: 'does-not-start', linuxStatus: 'ok' }))[0]?.check.warnings.at(-1)).toBe(
      'under Windows / WSL the engine is "does-not-start" (on Linux "ok")',
    );
  });

  it('a card outside the catalog is left alone; the real matrix answers for a catalog model', () => {
    const custom = planLayout(qwen, { gpu: customGpu('X', 16_000), count: 1, tp: 1, pp: 1 }, o);
    expect(applyCompat([custom], q, () => result({ status: 'does-not-start', reason: 'x' }))[0]).toBe(custom);
    const real = applyCompat([h100], { model: 'glm-5-3', repo: 'zai-org/GLM-5.3', engine: 'vllm', kvCache: 'fp8' });
    expect(real).toHaveLength(1);
  });
});

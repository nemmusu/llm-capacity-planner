import { describe, expect, it } from 'vitest';
import type { CompatFile, CompatRule } from '../../src/contract/compat';
import { archOf, checkCompat, compatData, compatRow, PIPELINE_FEATURE, sourceUrl } from '../../src/planner/compat';

const real = compatData();

/** A small matrix: one MoE model with two checkpoints, rules written for the test. */
function synthetic(rules: Omit<CompatRule, 'sources'>[]): CompatFile {
  const src = [{ kind: 'docs' as const, url: 'https://example.com/x', note: 'test' }];
  return {
    ...real,
    models: [
      {
        id: 'm',
        architectureClass: 'M',
        moe: true,
        features: ['a:gqa'],
        variants: [
          { repo: 'org/m', format: 'bf16', weights: ['w:bf16:dense', 'w:bf16:moe'] },
          { repo: 'org/m-nvfp4', format: 'mixed:nvfp4+fp8', weights: ['w:fp8-tensor:dense', 'w:nvfp4:moe'] },
          { repo: 'org/m-gguf', format: 'gguf', weights: [], outOfScope: 'GGUF' },
        ],
      },
    ],
    rules: rules.map((r) => ({ ...r, sources: src })),
  };
}
const rule = (id: string, feature: string, status: CompatRule['status'], extra: Partial<CompatRule> = {}): Omit<CompatRule, 'sources'> => ({
  id,
  engine: 'vllm',
  feature,
  archs: ['sm80', 'sm90', 'sm120'],
  status,
  platform: 'all',
  reason: `${feature} ${status}`,
  ...extra,
});
const BASE = [rule('bf16-d', 'w:bf16:dense', 'ok'), rule('bf16-m', 'w:bf16:moe', 'ok'), rule('gqa', 'a:gqa', 'ok')];

describe('archOf', () => {
  it('maps hardware.json ids, compute capabilities and architecture ids', () => {
    expect(archOf('rtx-5090-32gb')).toBe('sm120');
    expect(archOf('rtx-pro-6000-blackwell-server-96gb')).toBe('sm120');
    expect(archOf('b300-sxm')).toBe('sm100');
    expect(archOf('10.3')).toBe('sm100');
    expect(archOf('h200-sxm-141gb')).toBe('sm90');
    expect(archOf('l40s-48gb')).toBe('sm89');
    expect(archOf('rtx-3090-24gb')).toBe('sm86');
    expect(archOf('a100-sxm4-80gb')).toBe('sm80');
    expect(archOf('sm89')).toBe('sm89');
    expect(archOf('12.1')).toBeNull();
    expect(archOf('sm75')).toBeNull();
    expect(archOf('nonexistent')).toBeNull();
  });
});

describe('checkCompat: how statuses combine (synthetic matrix)', () => {
  it('every need ok → ok, with no reason', () => {
    const r = checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90' }, synthetic(BASE));
    expect(r).toMatchObject({ status: 'ok', reason: '', sources: [], wslOnly: false, repo: 'org/m', arch: 'sm90' });
    expect(r.details.map((d) => d.feature)).toEqual(['w:bf16:dense', 'w:bf16:moe', 'a:gqa']);
  });

  it('a need without any rule is unknown, never ok', () => {
    const r = checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', format: 'nvfp4' }, synthetic([...BASE, rule('fp8', 'w:fp8-tensor:dense', 'ok')]));
    expect(r.repo).toBe('org/m-nvfp4');
    expect(r.status).toBe('unknown');
    expect(r.reason).toContain('w:nvfp4:moe');
  });

  it('the worst wins: does-not-start > unknown > partial > ok', () => {
    const data = synthetic([...BASE, rule('fp8', 'w:fp8-tensor:dense', 'partial'), rule('nv', 'w:nvfp4:moe', 'does-not-start', { archs: ['sm80'] }), rule('nv2', 'w:nvfp4:moe', 'ok', { archs: ['sm90', 'sm120'] })]);
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm80', repo: 'org/m-nvfp4' }, data).status).toBe('does-not-start');
    const h = checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', repo: 'org/m-nvfp4' }, data);
    expect(h.status).toBe('partial');
    expect(h.reason).toBe('w:fp8-tensor:dense partial');
    expect(h.sources).toEqual(['https://example.com/x']);
  });

  it("a model's own rule replaces the general one for that feature, and adds its own needs", () => {
    const data = synthetic([...BASE, rule('own', 'a:gqa', 'does-not-start', { models: ['m'], archs: ['sm120'] }), rule('extra', 'm:model', 'partial', { models: ['m'], archs: ['sm90'] })]);
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm120' }, data)).toMatchObject({ status: 'does-not-start', reason: 'a:gqa does-not-start' });
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90' }, data)).toMatchObject({ status: 'partial', reason: 'm:model partial' });
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm80' }, data).status).toBe('ok');
  });

  it('overlapping rules of one feature: one reason, from the most specific rule (fewest architectures); every source kept', () => {
    const wide = { ...rule('wide', 'a:gqa', 'partial', { reason: 'broad text' }) };
    const narrow = { ...rule('narrow', 'a:gqa', 'partial', { archs: ['sm90'], reason: 'specific text' }) };
    const data = synthetic([...BASE.filter((r) => r.id !== 'gqa'), wide, narrow]);
    data.rules = data.rules.map((r) => (r.id === 'narrow' ? { ...r, sources: [{ kind: 'docs', url: 'https://example.com/y', note: 'test' }] } : r));
    const r = checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90' }, data);
    expect(r).toMatchObject({ status: 'partial', reason: 'specific text' });
    expect(r.sources).toEqual(['https://example.com/x', 'https://example.com/y']);
  });

  it('WSL-only rules: Linux keeps its status, wslOnly says why WSL differs', () => {
    const data = synthetic([...BASE, rule('wsl', 'a:gqa', 'does-not-start', { platform: 'wsl', workaround: 'use X' })]);
    const linux = checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90' }, data);
    expect(linux).toMatchObject({ status: 'ok', linuxStatus: 'ok', wslStatus: 'does-not-start', wslOnly: true, workarounds: ['use X'] });
    const wsl = checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', platform: 'windows-wsl' }, data);
    expect(wsl).toMatchObject({ status: 'does-not-start', wslOnly: true, reason: 'Windows/WSL2 only: a:gqa does-not-start Workaround: use X', workarounds: ['use X'] });
  });

  it("each workaround follows its own need: a pipeline blocker is not followed by the fp8 KV cache's workaround", () => {
    const data = synthetic([...BASE, rule('kv', 'kv:fp8', 'partial', { workaround: '--attention-backend TRITON_ATTN' }), rule('pp', PIPELINE_FEATURE, 'does-not-start')]);
    const both = checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', kvCache: 'fp8', pipeline: true }, data);
    expect(both).toMatchObject({ status: 'does-not-start', reason: 'p:pp does-not-start', workarounds: [] });
    const kv = checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', kvCache: 'fp8' }, data);
    expect(kv).toMatchObject({ status: 'partial', reason: 'kv:fp8 partial Workaround: --attention-backend TRITON_ATTN', workarounds: ['--attention-backend TRITON_ATTN'] });
    // Two needs that decide it, each with its own workaround right after its reason.
    const two = synthetic([...BASE, rule('kv', 'kv:fp8', 'partial', { workaround: 'A' }), rule('pp', PIPELINE_FEATURE, 'partial', { workaround: 'B' })]);
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', kvCache: 'fp8', pipeline: true }, two).reason).toBe('kv:fp8 partial Workaround: A. p:pp partial Workaround: B');
  });

  it('unknown version, GPU, model, checkpoint or an out-of-scope format → unknown with the reason', () => {
    const data = synthetic(BASE);
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', version: '0.30.0' }, data).reason).toContain('verified on vLLM 0.31.0 only');
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', version: 'v0.31.0' }, data).status).toBe('ok');
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'tesla-v100' }, data).reason).toContain('not recognized');
    expect(checkCompat({ model: 'x', engine: 'vllm', gpu: 'sm90' }, data).status).toBe('unknown');
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', repo: 'org/none' }, data).status).toBe('unknown');
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', format: 'gguf' }, data)).toMatchObject({ status: 'unknown', reason: 'GGUF', repo: 'org/m-gguf' });
  });

  it("a model's rules about the KV cache or a weight format apply only when the query or the checkpoint has them", () => {
    const data = synthetic([...BASE, rule('kv', 'kv:fp8', 'does-not-start', { models: ['m'] }), rule('w', 'w:nvfp4:moe', 'does-not-start', { models: ['m'] })]);
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90' }, data).status).toBe('ok');
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', kvCache: 'fp8' }, data).status).toBe('does-not-start');
  });

  it('an fp8 KV cache adds kv:fp8, or kv:fp8-mla for MLA models', () => {
    const data = synthetic([...BASE, rule('kv', 'kv:fp8', 'partial')]);
    expect(checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', kvCache: 'fp8' }, data).status).toBe('partial');
    const mla: CompatFile = { ...data, models: [{ ...data.models[0]!, features: ['a:gqa', 'a:mla'] }] };
    const r = checkCompat({ model: 'm', engine: 'vllm', gpu: 'sm90', kvCache: 'fp8' }, mla);
    expect(r.details.map((d) => d.feature)).toContain('kv:fp8-mla');
  });
});

describe('checkCompat: the real matrix', () => {
  it('pipeline layouts add p:pp: every catalog model has a verdict on both engines, with its sources', () => {
    for (const engine of ['vllm', 'sglang'] as const)
      for (const m of real.models) expect(real.rules.some((r) => r.engine === engine && r.feature === PIPELINE_FEATURE && r.models?.includes(m.id) && r.sources.length > 0), `${engine} ${m.id}`).toBe(true);
    const pp = (model: string, engine: 'vllm' | 'sglang') => checkCompat({ model, engine, gpu: 'sm90', pipeline: true }).details.find((d) => d.feature === PIPELINE_FEATURE)?.status;
    // GLM-5.3: on vLLM a stage boundary on a top-k-reusing layer reads stale indices (wrong output, counted as not
    // starting); SGLang crashes at start. MiniMax-M3 on vLLM: an open start-up issue on SM100 (partial).
    expect([pp('glm-5-3', 'vllm'), pp('glm-5-3', 'sglang'), pp('minimax-m3', 'vllm'), pp('gpt-oss-120b', 'vllm')]).toEqual(['does-not-start', 'does-not-start', 'partial', 'ok']);
    // Without a pipeline the feature is not asked.
    expect(checkCompat({ model: 'glm-5-3', engine: 'sglang', gpu: 'sm90' }).details.some((d) => d.feature === PIPELINE_FEATURE)).toBe(false);
  });

  it('DeepSeek-V4.1-Flash with vLLM 0.31 does not start on the RTX 5090, on Linux too (sm120 sparse-MLA pages)', () => {
    for (const gpu of ['rtx-5090-32gb', 'rtx-5080-16gb', 'rtx-pro-6000-blackwell-server-96gb']) {
      const r = checkCompat({ model: 'deepseek-v4-1-flash', engine: 'vllm', version: '0.31.0', gpu });
      expect(r.status, gpu).toBe('does-not-start');
      expect(r.linuxStatus).toBe('does-not-start');
      expect(r.wslOnly).toBe(false);
      expect(r.reason).toContain('64-token pages');
      expect(r.sources).toContain('https://github.com/vllm-project/vllm/issues/59203');
      expect(r.sources).toContain('https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/models/deepseek_v41/nvidia/flashinfer_sparse.py#L116-L118');
    }
  });

  it('DeepSeek-V4.1-Flash with SGLang 0.5.21 does not start on sm120 either (candidate indexer)', () => {
    const r = checkCompat({ model: 'deepseek-v4-1-flash', engine: 'sglang', gpu: 'rtx-5090-32gb' });
    expect(r.status).toBe('does-not-start');
    expect(r.reason).toContain('indexer');
    expect(r.sources).toContain('https://github.com/sgl-project/sglang/issues/42364');
  });

  it('Engram / PLE tables in pinned RAM: a Windows / WSL2-only limit of vLLM, with the flag that avoids it', () => {
    for (const model of ['deepseek-v4-1-flash', 'qwen3-8-flash-next']) {
      const r = checkCompat({ model, engine: 'vllm', gpu: 'h100-sxm5-80gb', platform: 'windows-wsl' });
      const engram = r.details.find((d) => d.feature === 'm:engram');
      expect(engram, model).toMatchObject({ status: 'ok', wslStatus: 'does-not-start' });
      // The flag belongs to the WSL2 rule, and follows its reason.
      expect(engram?.wslWorkaround).toContain('cpu_offload');
      expect(engram?.workaround).toBeUndefined();
      expect(r.status).toBe('does-not-start');
      expect(r.reason).toMatch(/Workaround: .*cpu_offload/);
    }
  });

  it('every answer of the real matrix is decided by rules with sources', () => {
    for (const m of real.models)
      for (const engine of ['vllm', 'sglang'] as const)
        for (const [arch, r] of Object.entries(compatRow(m.id, engine))) {
          expect(r.arch).toBe(arch);
          if (r.status !== 'ok' && r.details.length > 0) expect(r.sources.length, `${m.id} ${engine} ${arch}`).toBeGreaterThan(0);
        }
  });

  it('sources become links: code at the tag with its lines, issues, PRs', () => {
    expect(sourceUrl({ kind: 'code', repo: 'vllm-project/vllm', ref: 'v0.31.0', path: 'a/b.py', lines: [3, 9], anchor: 'abc' })).toBe('https://github.com/vllm-project/vllm/blob/v0.31.0/a/b.py#L3-L9');
    expect(sourceUrl({ kind: 'pr', repo: 'sgl-project/sglang', number: 1, note: 'x' })).toBe('https://github.com/sgl-project/sglang/pull/1');
    expect(sourceUrl({ kind: 'issue', repo: 'sgl-project/sglang', number: 2, note: 'x' })).toBe('https://github.com/sgl-project/sglang/issues/2');
  });
});

describe('checkCompat: inputs outside the matrix', () => {
  it('an engine the matrix does not have, or a coarse format with no label: unknown with the reason', () => {
    const r = checkCompat({ model: 'deepseek-v4-1-flash', engine: 'trtllm' as 'vllm', gpu: 'rtx-5090-32gb' });
    expect([r.status, r.reason, r.verifiedVersion]).toEqual(['unknown', 'Engine trtllm is not in the matrix.', '']);
    expect(checkCompat({ model: 'deepseek-v4-1-flash', engine: 'vllm', gpu: 'rtx-5090-32gb', format: 'int8' }).reason).toBe('No int8 checkpoint of deepseek-v4-1-flash in the catalog.');
  });
});

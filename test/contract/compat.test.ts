// The compatibility matrix of the capacity planner (data/compat.json): its schema, a source for every rule,
// and its consistency with the GPU and model catalogs. The "models" section is recomputed from models.json and the
// fixtures with the same functions as scripts/dev/update-compat.mjs, so a hand edit or a stale section fails.
// With CAPACITY_PLANNER_SRC_VLLM / CAPACITY_PLANNER_SRC_SGLANG pointing at local clones of the verified tags, every code
// anchor is also checked offline (online: npm run dev:update-compat -- --anchors).
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { checkAnchor, deriveCompatModels } from '../../scripts/dev/update-compat.mjs';
import { COMPAT_ARCHS, compatFileSchema, type CompatRule } from '../../src/contract/compat';
import { hardwareFileSchema } from '../../src/contract/data';
import { ModelsFileSchema } from '../../src/contract/models';

const dir = path.resolve(import.meta.dirname, '../../data');
const readJson = (name: string): unknown => JSON.parse(readFileSync(path.join(dir, name), 'utf8'));
const compat = compatFileSchema.parse(readJson('compat.json'));
const hardware = hardwareFileSchema.parse(readJson('hardware.json'));
const catalog = ModelsFileSchema.parse(readJson('models.json'));

const ENGINE_REPO = { vllm: 'vllm-project/vllm', sglang: 'sgl-project/sglang' } as const;
/** Kernel libraries a rule may cite besides the engines. */
const LIBRARY_REPOS = ['flashinfer-ai/flashinfer', 'vllm-project/DeepGEMM', 'deepseek-ai/DeepGEMM', 'deepseek-ai/FlashMLA', 'Dao-AILab/flash-attention', 'sgl-project/sgl-kernel', 'fla-org/flash-linear-attention', 'NVIDIA/cutlass'];

describe('compat.json: structure', () => {
  it('rule ids are unique and every rule names a known feature', () => {
    const ids = compat.rules.map((r) => r.id);
    expect(ids.length - new Set(ids).size).toBe(0);
    const features = new Set(compat.features.map((f) => f.id));
    expect(compat.rules.filter((r) => !features.has(r.feature)).map((r) => r.id)).toEqual([]);
    expect(compat.features.map((f) => f.id).length).toBe(features.size);
  });

  it('one entry per engine, with the tag and the commit the rules were read at', () => {
    expect(compat.engines.map((e) => e.engine).sort()).toEqual(['sglang', 'vllm']);
    for (const e of compat.engines) expect(e.ref).toBe(`v${e.version}`);
  });

  it('code sources of an engine are read at the verified tag; other repos are known kernel libraries', () => {
    const refOf = Object.fromEntries(compat.engines.map((e) => [ENGINE_REPO[e.engine], e.ref]));
    const wrong: string[] = [];
    for (const r of compat.rules)
      for (const s of r.sources) {
        if (s.kind === 'docs') continue;
        if (s.kind === 'code' && refOf[s.repo] !== undefined && s.ref !== refOf[s.repo]) wrong.push(`${r.id}: ${s.repo}@${s.ref}`);
        if (refOf[s.repo] === undefined && !LIBRARY_REPOS.includes(s.repo)) wrong.push(`${r.id}: ${s.repo}`);
      }
    expect(wrong).toEqual([]);
  });

  it('a rule is never "ok" from an issue alone: it cites code or documentation', () => {
    const issueOnly = compat.rules.filter((r) => r.status === 'ok' && r.sources.every((s) => s.kind === 'issue' || s.kind === 'pr'));
    expect(issueOnly.map((r) => r.id)).toEqual([]);
  });

  it('a WSL-only rule is about something worse than Linux, and model rules name catalog models', () => {
    expect(compat.rules.filter((r) => r.platform === 'wsl' && r.status === 'ok').map((r) => r.id)).toEqual([]);
    const ids = new Set(catalog.models.map((m) => m.id));
    expect(compat.rules.flatMap((r) => (r.models ?? []).filter((m) => !ids.has(m)).map((m) => `${r.id}: ${m}`))).toEqual([]);
  });
});

describe('compat.json: architectures and the GPU catalog', () => {
  it('every GPU of hardware.json falls in exactly one architecture, and every architecture has GPUs', () => {
    for (const g of hardware.gpus) {
      const archs = compat.architectures.filter((a) => a.computeCapabilities.includes(g.computeCapability));
      expect(archs.map((a) => a.id), g.id).toHaveLength(1);
    }
    for (const a of compat.architectures) expect(hardware.gpus.some((g) => a.computeCapabilities.includes(g.computeCapability)), a.id).toBe(true);
    expect(compat.architectures.map((a) => a.id)).toEqual([...COMPAT_ARCHS]);
  });

  it('FP8 and FP4 formats: what hardware.json says each architecture runs natively', () => {
    // A rule saying FP4 / FP8 runs natively must be about an architecture with those tensor cores (hardware.json formats).
    const native = (cc: string, f: 'fp8' | 'nvfp4' | 'mxfp4') => hardware.gpus.filter((g) => g.computeCapability === cc).every((g) => g.formats[f]);
    for (const a of compat.architectures) {
      const fp8 = a.computeCapabilities.every((cc) => native(cc, 'fp8'));
      const fp4 = a.computeCapabilities.every((cc) => native(cc, 'nvfp4') && native(cc, 'mxfp4'));
      expect(fp8, a.id).toBe(['sm89', 'sm90', 'sm100', 'sm120'].includes(a.id));
      expect(fp4, a.id).toBe(['sm100', 'sm120'].includes(a.id));
    }
  });
});

describe('compat.json: models and the model catalog', () => {
  it('the "models" section is what update-compat.mjs derives from models.json and the fixtures', () => {
    expect(compat.models).toEqual(deriveCompatModels(catalog));
  });

  it('every model and every checkpoint of models.json is in the matrix', () => {
    expect(compat.models.map((m) => m.id)).toEqual(catalog.models.map((m) => m.id));
    for (const m of catalog.models) {
      const c = compat.models.find((x) => x.id === m.id);
      expect(c?.variants.map((v) => v.repo), m.id).toEqual(m.quantizations.map((q) => q.repo));
      expect(c?.moe, m.id).toBe(m.architecture.kind === 'moe');
    }
  });

  it('every feature a model needs is a feature of the matrix', () => {
    const features = new Set(compat.features.map((f) => f.id));
    const used = compat.models.flatMap((m) => [...m.features, ...m.variants.flatMap((v) => v.weights)]);
    expect([...new Set(used)].filter((f) => !features.has(f))).toEqual([]);
  });

  it('no gap left implicit: each feature a model uses has a rule for every engine and architecture', () => {
    const missing: string[] = [];
    const general = (r: CompatRule) => r.models === undefined && r.platform === 'all';
    for (const engine of ['vllm', 'sglang'] as const)
      for (const m of compat.models) {
        const used = new Set([...m.features, ...m.variants.flatMap((v) => v.weights)]);
        for (const f of used)
          for (const arch of COMPAT_ARCHS) {
            const rules = compat.rules.filter((r) => r.engine === engine && r.feature === f && r.archs.includes(arch) && r.platform === 'all');
            if (!rules.some((r) => general(r) || r.models?.includes(m.id))) missing.push(`${engine} ${f} ${arch} (${m.id})`);
          }
      }
    expect([...new Set(missing.map((x) => x.replace(/ \(.*\)$/, '')))]).toEqual([]);
  });
});

const CLONES: Record<string, string | undefined> = {
  'vllm-project/vllm': process.env.CAPACITY_PLANNER_SRC_VLLM,
  'sgl-project/sglang': process.env.CAPACITY_PLANNER_SRC_SGLANG,
};
describe.skipIf(!CLONES['vllm-project/vllm'] && !CLONES['sgl-project/sglang'])('compat.json: code anchors in local clones', () => {
  it('every anchor is inside its cited lines', () => {
    const bad: string[] = [];
    for (const r of compat.rules)
      for (const s of r.sources) {
        if (s.kind !== 'code') continue;
        const root = CLONES[s.repo];
        if (!root) continue;
        const file = path.join(root, ...s.path.split('/'));
        const state = checkAnchor(s, existsSync(file) ? readFileSync(file, 'utf8') : null).state;
        if (state !== 'ok') bad.push(`${r.id}: ${s.path}:${s.lines.join('-')} ${state}`);
      }
    expect(bad).toEqual([]);
  });
});

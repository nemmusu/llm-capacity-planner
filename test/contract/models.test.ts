// The capacity planner's model catalog (data/models.json): its schema, and its consistency with the files
// saved from Hugging Face (data/models/). Everything is recomputed from the fixtures,
// independently of scripts/dev/update-models.mjs, so a hand edit or a stale "derived" block fails.
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';
import { ModelEntrySchema, ModelsFileSchema, type ModelEntry, type Weights } from '../../src/contract/models';

const root = path.resolve(__dirname, '..', '..');
const raw: unknown = JSON.parse(readFileSync(path.join(root, 'data', 'models.json'), 'utf8'));
const catalog = ModelsFileSchema.parse(raw);

type Json = Record<string, unknown>;
interface Headers {
  revision: string;
  shards: Record<string, { headerBytes: number; fileBytes: number | null; header: Record<string, { dtype: string; shape: number[] }> }>;
}
interface Summary {
  revision: string;
  shards: number;
  tensors: number;
  tensorBytes: number;
  entries: { pattern: string; dtype: string; shape: number[]; group: 'lm' | 'mtp' | 'vision' | 'audio'; count: number; bytes: number; layers: string | null }[];
}

const DTYPE_BYTES: Record<string, number> = {
  F64: 8, I64: 8, U64: 8, F32: 4, I32: 4, U32: 4, F16: 2, BF16: 2, I16: 2, U16: 2,
  F8_E4M3: 1, F8_E5M2: 1, F8_E8M0: 1, I8: 1, U8: 1, BOOL: 1,
};
const ONE_M = 1_048_576;
const fixtureDir = (m: ModelEntry) => path.join(root, 'data', 'models', m.repo.replace('/', '__'));
const readJson = (file: string): Json | null => (existsSync(file) ? (JSON.parse(readFileSync(file, 'utf8')) as Json) : null);
const readGz = <T>(file: string): T => JSON.parse(gunzipSync(readFileSync(file)).toString('utf8')) as T;
const numel = (shape: number[]) => shape.reduce((a, x) => a * x, 1);
const PATTERN = (name: string) => name.replace(/(^|\.)\d+(?=\.|$)/g, '$1{n}');

function configMax(cfg: Json | null, params: Json | null): number | null {
  const sub = (k: string) => (cfg?.[k] as Json | undefined)?.max_position_embeddings;
  for (const v of [cfg?.max_position_embeddings, sub('text_config'), sub('llm_config'), sub('language_config'), cfg?.max_sequence_length, cfg?.seq_length, params?.max_position_embeddings, params?.max_seq_len]) {
    if (typeof v === 'number') return v;
  }
  return null;
}

/**
 * One repo's compact headers against the catalog's totals: the script checked data_offsets when it read them
 * (source.json headerChecks), and every file is still exactly 8 bytes + header + the tensors' shape × dtype bytes.
 */
function checkHeaders(dir: string, revision: string, w: Weights | null | undefined, withIndex: boolean) {
  const src = readJson(path.join(dir, 'source.json'));
  expect(src?.headerChecks).toEqual({ ok: true, problems: [] });
  const payload = readGz<Headers>(path.join(dir, 'safetensors-headers.json.gz'));
  expect(payload.revision).toBe(revision);
  let tensors = 0;
  let bytes = 0;
  const byDtype: Record<string, number> = {};
  const names = new Set<string>();
  // Hundreds of thousands of tensors (Kimi K3): problems are collected, then checked once.
  const problems: string[] = [];
  for (const [file, shard] of Object.entries(payload.shards)) {
    let shardBytes = 0;
    for (const [name, t] of Object.entries(shard.header)) {
      if (name === '__metadata__') continue;
      const per = DTYPE_BYTES[t.dtype];
      if (per == null || !Array.isArray(t.shape) || 'data_offsets' in t) problems.push(`${file}: ${name}`);
      const size = numel(t.shape) * (per ?? 0);
      shardBytes += size;
      tensors++;
      bytes += size;
      byDtype[t.dtype] = (byDtype[t.dtype] ?? 0) + size;
      names.add(name);
    }
    if (shard.fileBytes != null && shard.fileBytes !== 8 + shard.headerBytes + shardBytes) problems.push(`${file}: ${shard.fileBytes} bytes, not 8 + header + tensors`);
  }
  expect(problems).toEqual([]);
  expect(w?.tensors).toBe(tensors);
  expect(w?.tensorBytes).toBe(bytes);
  expect(Object.fromEntries(Object.entries(w?.byDtype ?? {}).map(([k, v]) => [k, v.bytes]))).toEqual(byDtype);
  const g = w?.groups;
  expect((g?.lm.bytes ?? 0) + (g?.mtp.bytes ?? 0) + (g?.vision.bytes ?? 0) + (g?.audio.bytes ?? 0)).toBe(bytes);
  // The shard index lists exactly these tensors; the ones it places in the wrong file are counted (granite-4.2-8b).
  const indexFile = ['model.safetensors.index.json.gz', 'consolidated.safetensors.index.json.gz'].map((f) => path.join(dir, f)).find((f) => existsSync(f));
  if (withIndex && indexFile) {
    const index = readGz<{ weight_map: Record<string, string> }>(indexFile);
    expect(Object.keys(index.weight_map).sort()).toEqual([...names].sort());
    expect(w?.indexWrongFile).toBe(Object.entries(index.weight_map).filter(([name, file]) => !payload.shards[file]?.header[name]).length);
  } else if (withIndex) expect(Object.keys(payload.shards)).toHaveLength(1);
  return payload;
}

/** The per-kind summary: each entry is count × shape × dtype bytes, totals match, and it groups exactly the headers' tensors. */
function checkSummary(dir: string, revision: string, w: Weights | null | undefined, payload: Headers) {
  const s = readJson(path.join(dir, 'tensor-summary.json')) as unknown as Summary;
  expect(s.revision).toBe(revision);
  expect(s.tensorBytes).toBe(w?.tensorBytes);
  expect(s.tensors).toBe(w?.tensors);
  expect(s.entries.filter((e) => e.bytes !== e.count * numel(e.shape) * (DTYPE_BYTES[e.dtype] ?? NaN)).map((e) => e.pattern)).toEqual([]);
  for (const g of ['lm', 'mtp', 'vision', 'audio'] as const) expect(w?.groups[g].bytes).toBe(s.entries.filter((e) => e.group === g).reduce((a, e) => a + e.bytes, 0));
  const fromHeaders = new Map<string, number>();
  for (const shard of Object.values(payload.shards)) {
    for (const [name, t] of Object.entries(shard.header)) {
      if (name === '__metadata__') continue;
      const key = `${PATTERN(name)}|${t.dtype}|${t.shape.join('x')}`;
      fromHeaders.set(key, (fromHeaders.get(key) ?? 0) + 1);
    }
  }
  const fromSummary = new Map<string, number>();
  for (const e of s.entries) {
    const key = `${e.pattern}|${e.dtype}|${e.shape.join('x')}`;
    fromSummary.set(key, (fromSummary.get(key) ?? 0) + e.count);
  }
  expect(Object.fromEntries(fromSummary)).toEqual(Object.fromEntries(fromHeaders));
}

describe('the model catalog', () => {
  it('follows the schema, with 25 to 45 models of unique ids and repos', () => {
    expect(catalog.models.length).toBeGreaterThanOrEqual(25);
    expect(catalog.models.length).toBeLessThanOrEqual(45);
    expect(new Set(catalog.models.map((m) => m.id)).size).toBe(catalog.models.length);
    expect(new Set(catalog.models.map((m) => m.repo)).size).toBe(catalog.models.length);
  });

  it('rejects an entry with a field it does not know or a version that is not one', () => {
    const m = catalog.models[0];
    expect(ModelEntrySchema.safeParse({ ...m, extra: 1 }).success).toBe(false);
    expect(ModelEntrySchema.safeParse({ ...m, engines: { ...m?.engines, vllm: { ...m?.engines.vllm, minVersion: 'nightly' } } }).success).toBe(false);
  });

  describe.each(catalog.models.map((m) => [m.id, m] as const))('%s', (_id, m) => {
    it('states its facts with their sources', () => {
      if (m.releaseDateSource === 'hf-createdAt') expect(m.releaseDate).toBe(m.hf.createdAt.slice(0, 10));
      else expect(m.releaseDateUrl).toBeDefined();
      if (m.license.commercialUse === 'conditional') expect(m.license.conditions ?? '').not.toBe('');
      expect(m.architecture.modalities).toContain('text');
      expect(m.params.activeB).toBeLessThanOrEqual(m.params.totalB);
      if (m.architecture.kind === 'dense') expect(m.params.activeB).toBe(m.params.totalB);
      if (m.context.cardMaxTokens != null) expect(m.context.cardSource).not.toBeNull();
      for (const e of [m.engines.vllm, m.engines.sglang]) if (e.minVersion) expect(e.sources.length).toBeGreaterThan(0);
      for (const s of m.sources) if (s.quote) expect(s.quote.split(/\s+/).length).toBeLessThanOrEqual(40);
    });

    it('lists the checkpoint itself and only quantized repos the update script found, with their fixtures', () => {
      const base = m.quantizations.filter((q) => q.kind === 'base');
      expect(base.map((q) => q.repo)).toEqual([m.repo]);
      expect(base[0]?.weightsVariantDir).toBe(m.hf.readable ? '' : null);
      if (m.hf.readable) expect(base[0]?.format).toBe(m.derived.precision);
      const found = new Map((m.derived.quantRepos ?? []).map((q) => [q.repo, q]));
      for (const q of m.quantizations.filter((x) => x.kind !== 'base')) {
        expect(found.get(q.repo)?.draft, `${q.repo} not a non-draft repo of quant-repos.json`).toBe(false);
        expect(q.format).toBe(found.get(q.repo)?.precision ?? 'unknown');
        const v = m.derived.variants.find((x) => x.repo === q.repo);
        expect(q.weightsVariantDir).toBe(v?.readable ? `variants/${q.repo.replace('/', '__')}` : null);
        if (q.weightsVariantDir) expect(existsSync(path.join(fixtureDir(m), q.weightsVariantDir, 'safetensors-headers.json.gz'))).toBe(true);
      }
      for (const d of m.draftModels) expect(found.get(d.repo)?.draft).toBe(true);
    });

    it('records whether its Hugging Face files could be read, and why not', () => {
      const src = readJson(path.join(fixtureDir(m), 'source.json'));
      expect(src?.repo).toBe(m.repo);
      expect(src?.revision).toBe(m.hf.revision);
      expect(m.hf.readable).toBe(src?.readable);
      expect(m.hf.fixtures).toBe(path.relative(root, fixtureDir(m)).split(path.sep).join('/'));
      if (!m.hf.readable) {
        // Gated, or withheld because its licence may not allow passing its files on.
        expect(m.hf.gated || m.hf.withheld).toBeTruthy();
        expect(m.notes ?? '').toMatch(m.hf.withheld ? /licence/ : /HF_TOKEN/);
        expect(m.derived.weights).toBeNull();
      }
    });

    it('takes its context and RoPE from config.json and explains any difference from the model card', () => {
      if (!m.hf.readable) return;
      const cfg = readJson(path.join(fixtureDir(m), 'config.json'));
      const params = readJson(path.join(fixtureDir(m), 'params.json'));
      const fromConfig = configMax(cfg, params);
      expect(fromConfig).not.toBeNull();
      expect(m.derived.configMaxPositions).toBe(fromConfig);
      expect(m.context.configMaxPositions).toBe(fromConfig);
      expect(m.context.configSource).toBe(m.derived.configMaxPositionsPath);
      if (m.context.cardMaxTokens != null && m.context.cardMaxTokens !== fromConfig) expect(m.context.note ?? '').not.toBe('');
      const own = m.derived.configRope;
      if (!own) expect(m.rope).toBeNull();
      else if (m.rope?.type !== 'per-layer-type') {
        expect(m.rope?.type).toBe(own.rope_type ?? own.type ?? 'default');
        expect(m.rope?.factor).toBe(own.factor ?? null);
        expect(m.rope?.originalMaxPositions).toBe(own.original_max_position_embeddings ?? null);
      } else expect(Object.keys(m.rope.layers ?? {})).toEqual(Object.keys(own));
    });

    it('sums exactly the tensors of its compact safetensors headers', () => {
      if (!m.hf.readable) return;
      const payload = checkHeaders(fixtureDir(m), m.hf.revision, m.derived.weights, true);
      checkSummary(fixtureDir(m), m.hf.revision, m.derived.weights, payload);
    });

    it('sums exactly the tensors of its quantized variants too', () => {
      for (const v of m.derived.variants) {
        const dir = path.join(root, v.fixtures);
        const src = readJson(path.join(dir, 'source.json'));
        expect(src?.repo, v.repo).toBe(v.repo);
        expect(v.readable).toBe(src?.readable);
        if (!v.readable) {
          expect(v.weights).toBeNull();
          continue;
        }
        const payload = checkHeaders(dir, v.revision ?? '', v.weights, false);
        checkSummary(dir, v.revision ?? '', v.weights, payload);
      }
    });

    it('agrees with the checkpoint on MTP layers and the vision encoder', () => {
      const g = m.derived.weights?.groups;
      if (!g) {
        expect(m.mtp.inCheckpoint).toBeNull();
        expect(m.visionEncoder.inCheckpoint).toBeNull();
        return;
      }
      expect(m.mtp.inCheckpoint).toBe(g.mtp.tensors > 0);
      expect(m.mtp.tensors).toBe(g.mtp.tensors);
      expect(m.visionEncoder.inCheckpoint).toBe(g.vision.tensors > 0);
      // MTP declared in config.json but not shipped (MiniMax-M3) must be said.
      if ((m.derived.configMtpLayers ?? 0) > 0 && g.mtp.tensors === 0) expect(m.mtp.notes ?? '').toMatch(/not in the checkpoint/);
      if (m.visionEncoder.inCheckpoint) expect(m.architecture.modalities).toContain('image');
    });

    it('gives the exact settings for 1M tokens, or says why there are none', () => {
      const lc = m.longContext;
      if (lc.case !== 'yarn-beyond-declared') expect(lc.source).toMatch(/^https:\/\//);
      const t = lc.to1M;
      if (lc.case === 'not-applicable') {
        expect(t).toBeNull();
        expect(lc.statement ?? '').not.toBe('');
        return;
      }
      if (!t) {
        // Only YaRN cases may lack arguments, and then they say why.
        expect(lc.case).not.toBe('native>=1M');
        expect(lc.to1MUnavailable ?? '').not.toBe('');
        return;
      }
      expect(lc.to1MUnavailable).toBeNull();
      expect(t.vllmArgs[t.vllmArgs.indexOf('--max-model-len') + 1]).toBe(String(ONE_M));
      expect(t.sglangArgs[t.sglangArgs.indexOf('--context-length') + 1]).toBe(String(ONE_M));
      if (lc.case === 'native>=1M') {
        expect(t.method).toBe('native');
        expect(Math.max(m.context.configMaxPositions ?? 0, m.context.cardMaxTokens ?? 0)).toBeGreaterThanOrEqual(1_000_000);
        expect(t.qualityGuaranteed).toBe(true);
        // A config below 1M needs the engines' documented escape hatches.
        if ((m.context.configMaxPositions ?? ONE_M) < ONE_M) expect(Object.keys(t.env).sort()).toEqual(['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN', 'VLLM_ALLOW_LONG_MAX_MODEL_LEN']);
        else expect(t.env).toEqual({});
        return;
      }
      // YaRN (or the config's own scaling type) stretched to exactly 1,048,576 positions.
      const rope = t.rope ?? {};
      expect(t.method).toBe(rope.rope_type ?? rope.type);
      expect(rope.original_max_position_embeddings).toBe(t.originalMaxPositions);
      expect((t.factor ?? 0) * (t.originalMaxPositions ?? 0)).toBe(ONE_M);
      expect(rope.factor).toBe(t.factor);
      expect(t.env).toEqual({});
      expect(t.qualityGuaranteed).toBe(lc.case === 'yarn-official' && (lc.officialFactor ?? 0) >= (t.factor ?? Infinity));
      if (lc.case === 'yarn-official') expect(lc.officialFactor).toBeGreaterThan(1);
      // vLLM and SGLang get the same override: "rope_scaling" (alias of rope_parameters in transformers 5) and
      // max_position_embeddings at 1M, at the level where config.json has them.
      expect(JSON.parse(t.vllmArgs[t.vllmArgs.indexOf('--hf-overrides') + 1] ?? 'null')).toEqual(t.hfOverrides);
      expect(JSON.parse(t.sglangArgs[t.sglangArgs.indexOf('--json-model-override-args') + 1] ?? 'null')).toEqual(t.hfOverrides);
      // Without the saved config (gated or withheld), the overrides themselves say where the keys are.
      const nested = m.derived.configMaxPositionsKey ? m.derived.configMaxPositionsKey.startsWith('text_config.') : t.hfOverrides?.text_config !== undefined;
      const where = (nested ? t.hfOverrides?.text_config : t.hfOverrides) as Json;
      expect(where.rope_scaling).toEqual(rope);
      expect(where.max_position_embeddings).toBe(ONE_M);
      // Every key of the config's own rope block is kept (mrope sections, rope_theta, partial rotary factor...).
      for (const [k, v] of Object.entries(m.derived.configRope ?? {})) if (!['factor', 'rope_type', 'type', 'original_max_position_embeddings'].includes(k)) expect(rope[k], k).toEqual(v);
    });
  });
});

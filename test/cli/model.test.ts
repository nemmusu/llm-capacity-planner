import { describe, expect, it } from 'vitest';
import { repoLayout } from '../../src/hub/formats';
import { HubClient } from '../../src/hub/client';
import { loadedWeightFiles, mapLimit, readRemoteModel } from '../../src/cli/model';
import { configJson, fakeHub, HUB, qwenHeaders, qwenRepo, qwenTensors, type FakeRepo } from './fixtures';

const GiB = 1024 ** 3;
const hubOf = (repos: Record<string, FakeRepo>, seen: { url: string }[] = []) => new HubClient({ fetch: fakeHub(repos, [], seen), baseUrl: HUB, getToken: () => undefined });
async function variantOf(hub: HubClient, repo: string, dir = '') {
  const files = await hub.files(repo);
  return repoLayout(files, { name: repo }).safetensors.find((v) => v.dir === dir) as ReturnType<typeof repoLayout>['safetensors'][number];
}

describe('mapLimit', () => {
  it('keeps the order and never runs more than the limit at once', async () => {
    let running = 0;
    let peak = 0;
    const out = await mapLimit([5, 1, 4, 2, 3], 2, async (n) => {
      running++;
      peak = Math.max(peak, running);
      await new Promise((r) => setTimeout(r, n));
      running--;
      return n * 10;
    });
    expect(out).toEqual([50, 10, 40, 20, 30]);
    expect(peak).toBe(2);
    expect(await mapLimit([], 8, () => Promise.resolve(1))).toEqual([]);
  });
});

describe('loadedWeightFiles', () => {
  const files = [
    { path: 'config.json', size: 1 },
    { path: 'model-00001-of-00002.safetensors', size: GiB },
    { path: 'model-00002-of-00002.safetensors', size: GiB },
    { path: 'consolidated.safetensors', size: 2 * GiB },
  ];
  const withIndex = (index: unknown): FakeRepo => ({ files: [...files, { path: 'model.safetensors.index.json', size: 1 }], configs: { '': {} }, json: { 'model.safetensors.index.json': index } });

  it('without an index: every weight file', async () => {
    const hub = hubOf({ 'a/b': { files, configs: {} } });
    expect((await loadedWeightFiles(hub, 'a/b', 'main', await variantOf(hub, 'a/b'))).map((f) => f.path)).toEqual(['consolidated.safetensors', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']);
  });

  it('with an index: only the files it names (the consolidated copy is left out)', async () => {
    const hub = hubOf({ 'a/b': withIndex({ weight_map: { x: 'model-00001-of-00002.safetensors', y: 'model-00002-of-00002.safetensors', z: 'model-00002-of-00002.safetensors' } }) });
    expect((await loadedWeightFiles(hub, 'a/b', 'main', await variantOf(hub, 'a/b'))).map((f) => f.path)).toEqual(['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']);
  });

  it('an index naming none of the files, or with no weight_map: every weight file', async () => {
    for (const index of [{ weight_map: { x: 'other.safetensors' } }, {}]) {
      const hub = hubOf({ 'a/b': withIndex(index) });
      expect(await loadedWeightFiles(hub, 'a/b', 'main', await variantOf(hub, 'a/b'))).toHaveLength(3);
    }
  });
});

describe('readRemoteModel', () => {
  it('Qwen3-8B: the real config and every tensor header, read with Range', async () => {
    const seen: { url: string }[] = [];
    const hub = hubOf({ 'Qwen/Qwen3-8B': qwenRepo() }, seen);
    const m = await readRemoteModel(hub, 'Qwen/Qwen3-8B', 'main', await variantOf(hub, 'Qwen/Qwen3-8B'));
    expect(m.repo).toBe('Qwen/Qwen3-8B');
    expect(m.dir).toBe('');
    expect(m.config).toEqual(configJson('qwen3-8b'));
    expect(m.files).toHaveLength(5);
    expect(m.fileBytes).toBe(Object.values(qwenHeaders.files).reduce((a, f) => a + f.fileBytes, 0));
    expect(m.tensors).toHaveLength(qwenTensors.length);
    expect(m.facts.contextLength).toBe(40_960);
    expect(m.precision).toBe('BF16');
    expect(m.shape).toBeDefined();
    // Two Range reads per file.
    expect(seen.filter((s) => s.url.endsWith('.safetensors'))).toHaveLength(10);
  });

  it('a variant folder: its own config; no precision when the config names none', async () => {
    const repo: FakeRepo = {
      files: [
        { path: 'FP8/config.json', size: 1 },
        { path: 'FP8/model.safetensors', size: 2048 },
      ],
      configs: { FP8: { num_hidden_layers: 2, num_attention_heads: 4, hidden_size: 64, vocab_size: 100 } },
    };
    const hub = hubOf({ 'a/b': repo });
    const m = await readRemoteModel(hub, 'a/b', 'main', await variantOf(hub, 'a/b', 'FP8'));
    expect(m.dir).toBe('FP8');
    expect(m.tensors).toEqual([{ name: 'FP8/model.safetensors.w', dtype: 'BF16', shape: [1024] }]);
    expect(m.precision).toBeUndefined();
    expect('precision' in m).toBe(false);
  });
});

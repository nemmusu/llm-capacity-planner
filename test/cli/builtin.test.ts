import { mkdirSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { gzipSync } from 'node:zlib';
import { afterAll, describe, expect, it } from 'vitest';
import { packModels } from '../../scripts/pack-models.mjs';
import { builtinModels, readPack } from '../../src/cli/builtin';
import { readRemoteModel } from '../../src/cli/model';
import { HubClient } from '../../src/hub/client';
import { repoLayout } from '../../src/hub/formats';
import { tempDir } from '../helpers/safetensors';

const BASE = 'https://hub.test';
const t = tempDir('pack-');
afterAll(() => t.cleanup());

const header = { __metadata__: { format: 'pt' }, 'model.embed_tokens.weight': { dtype: 'BF16', shape: [16, 8] }, 'lm_head.weight': { dtype: 'BF16', shape: [16, 8] } };
const config = { architectures: ['LlamaForCausalLM'], hidden_size: 8, num_hidden_layers: 1, num_attention_heads: 2, num_key_value_heads: 1, vocab_size: 16, torch_dtype: 'bfloat16' };

/** A repo folder as data/models keeps it. */
function repo(dir: string, source: Record<string, unknown>, files: Record<string, unknown> = {}) {
  const d = path.join(t.dir, dir);
  mkdirSync(d, { recursive: true });
  writeFileSync(path.join(d, 'source.json'), JSON.stringify({ readable: true, gated: false, revision: 'abc123', fetchedAt: '2026-10-14T10:00:00.000Z', ...source }));
  for (const [name, v] of Object.entries(files)) writeFileSync(path.join(d, name), name.endsWith('.gz') ? gzipSync(JSON.stringify(v)) : JSON.stringify(v));
}
const headers = (r: string) => ({ repo: r, shards: { 'model.safetensors': { fileBytes: 1000, headerBytes: 200, header } } });
repo('org__small', { repo: 'org/small', hfSafetensorsTotal: 256, pipelineTag: 'text-generation' }, { 'config.json': config, 'safetensors-headers.json.gz': headers('org/small') });
repo('org__small/variants/org__small-FP8', { repo: 'org/small-FP8', fetchedAt: '2026-10-15T08:00:00.000Z' }, {
  'config.json.gz': config,
  'model.safetensors.index.json.gz': { weight_map: { 'lm_head.weight': 'model.safetensors' } },
  'safetensors-headers.json.gz': headers('org/small-FP8'),
});
repo('org__gated', { repo: 'org/gated', gated: 'manual' }, { 'config.json': config, 'safetensors-headers.json.gz': headers('org/gated') });
repo('org__unreadable', { repo: 'org/unreadable', readable: false });
repo('org__no-headers', { repo: 'org/no-headers' }, { 'config.json': config });

const { pack, count } = packModels(t.dir);
const hub = builtinModels(pack, `${BASE}/`);
const get = (p: string, init?: RequestInit) => hub.fetch(`${BASE}${p}`, init);

describe('the model pack', () => {
  it('keeps only readable, public repos with their headers', () => {
    expect(count).toBe(2);
    expect(Object.keys(readPack(pack).index.repos).sort()).toEqual(['org/small', 'org/small-FP8']);
    expect(hub.date).toBe('2026-10-15');
    expect(() => readPack(new Uint8Array(16))).toThrow('Not a model pack');
  });

  it('knows its repos at main or at their pinned commit', () => {
    expect(hub.has('org/small', 'main')).toBe(true);
    expect(hub.has('org/small', 'abc123')).toBe(true);
    expect(hub.has('org/small', 'other')).toBe(false);
    expect(hub.has('org/gated', 'main')).toBe(false);
  });

  it('answers the hub endpoints the command calls', async () => {
    expect(await (await get('/api/models/org/small')).json()).toEqual({ id: 'org/small', sha: 'abc123', gated: false, tags: ['safetensors'], safetensors: { total: 256 }, pipeline_tag: 'text-generation' });
    expect(await (await get('/api/models/org/small-FP8/revision/abc123')).json()).toEqual({ id: 'org/small-FP8', sha: 'abc123', gated: false, tags: ['safetensors'] });
    const tree = (await (await get('/api/models/org/small-FP8/tree/main?recursive=true')).json()) as { path: string }[];
    expect(tree.map((f) => f.path).sort()).toEqual(['config.json', 'model.safetensors', 'model.safetensors.index.json']);
    expect(await (await get('/org/small-FP8/resolve/main/config.json')).json()).toEqual(config);
    const head = await get('/org/small/resolve/abc123/model.safetensors', { headers: { range: 'bytes=0-7' } });
    expect(head.status).toBe(206);
    expect((await head.arrayBuffer()).byteLength).toBe(8);
    expect((await get('/org/small/resolve/main/model.safetensors')).status).toBe(200);
  });

  it('says not found for anything else', async () => {
    for (const p of ['/api/models/org/gated', '/api/models/org/small/revision/zzz', '/org/small/resolve/zzz/config.json', '/org/gated/resolve/main/config.json', '/org/small/resolve/main/README.md', '/elsewhere'])
      expect((await get(p)).status, p).toBe(404);
    expect((await hub.fetch('https://other.test/api/models/org/small')).status).toBe(404);
  });

  it('reads a model through the hub client exactly as from Hugging Face', async () => {
    const client = new HubClient({ fetch: hub.fetch, baseUrl: BASE, getToken: () => undefined });
    const files = await client.files('org/small-FP8');
    const variant = repoLayout(files).safetensors[0];
    if (!variant) throw new Error('no variant');
    const model = await readRemoteModel(client, 'org/small-FP8', 'main', variant);
    expect(model.tensors.map((x) => x.name)).toEqual(['model.embed_tokens.weight', 'lm_head.weight']);
    expect(model.fileBytes).toBe(1000);
    // A second read of the same repo comes from the unpacked copy.
    expect((await client.json('org/small-FP8', 'config.json')) as unknown).toEqual(config);
  });
});

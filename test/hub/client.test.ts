import { describe, expect, it } from 'vitest';
import { fileKind, HubClient, hubHttpError, readCapped, SAFETENSORS_HEADER_MAX, toHubFile, toHubModel } from '../../src/hub/client';

const BASE = 'https://hub.test';

/** A fetch answering from a table of URL → response factory, recording what was asked. */
function fakeFetch(routes: Record<string, (init?: RequestInit) => Response | Promise<Response>>) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fn = (async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, ...(init ? { init } : {}) });
    const route = routes[url];
    if (!route) return new Response('missing', { status: 404 });
    return route(init);
  }) as typeof fetch;
  return { fn, calls };
}
const json = (v: unknown, init: ResponseInit = {}) => new Response(JSON.stringify(v), { status: 200, ...init });
const client = (routes: Parameters<typeof fakeFetch>[0], token?: string) => {
  const f = fakeFetch(routes);
  return { hub: new HubClient({ fetch: f.fn, baseUrl: `${BASE}/`, getToken: () => token }), calls: f.calls };
};

/** A safetensors file's bytes: u64 LE header length, then the header. */
function safetensors(header: unknown): Uint8Array {
  const body = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + body.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(body.length), true);
  out.set(body, 8);
  return out;
}
/** A response that honours (206) or ignores (200) the Range header. */
function served(bytes: Uint8Array, honourRange: boolean) {
  return (init?: RequestInit) => {
    const m = /^bytes=(\d+)-(\d+)$/.exec(new Headers(init?.headers).get('range') ?? '');
    if (!honourRange || !m) return new Response(bytes as Uint8Array<ArrayBuffer>, { status: 200 });
    return new Response(bytes.slice(Number(m[1]), Number(m[2]) + 1) as Uint8Array<ArrayBuffer>, { status: 206 });
  };
}

describe('hubHttpError', () => {
  const err = (status: number, headers: Record<string, string> = {}) => hubHttpError(new Response(null, { status, headers }), 'org/m');
  it('maps the hub answers to codes', () => {
    expect(err(403)).toMatchObject({ code: 'HF_GATED', params: { status: 403, repo: 'org/m' } });
    expect(err(401, { 'x-error-code': 'GatedRepo', 'x-error-message': 'accept it' })).toMatchObject({ code: 'HF_GATED', detail: 'accept it' });
    expect(err(401)).toMatchObject({ code: 'HF_AUTH_REQUIRED' });
    expect(err(404, { 'x-error-code': 'RepoNotFound' }).message).toContain('RepoNotFound');
    expect(err(404).message).toContain('HTTP 404');
    expect(err(429)).toMatchObject({ code: 'HF_RATE_LIMITED' });
    expect(err(500)).toMatchObject({ code: 'NETWORK' });
    expect(hubHttpError(new Response(null, { status: 403 })).message).toContain('the repo');
    expect(hubHttpError(new Response(null, { status: 403 })).params).toEqual({ status: 403 });
  });
});

describe('fileKind and toHubFile', () => {
  it('tells weights, MTP and draft modules, configs, tokenizers and the rest apart', () => {
    expect(fileKind('model-00001-of-00002.safetensors')).toBe('weights');
    expect(fileKind('pytorch_model.bin')).toBe('weights');
    expect(fileKind('MTP-BF16/model.safetensors')).toBe('mtp');
    expect(fileKind('model_mtp.safetensors')).toBe('mtp');
    expect(fileKind('DFlash2-FP8/model.safetensors')).toBe('draft');
    expect(fileKind('eagle3/model.safetensors')).toBe('draft');
    expect(fileKind('draft-model.safetensors')).toBe('draft');
    expect(fileKind('FP8/config.json')).toBe('config');
    expect(fileKind('model.safetensors.index.json')).toBe('config');
    expect(fileKind('tokenizer_config.json')).toBe('tokenizer');
    expect(fileKind('merges.txt')).toBe('tokenizer');
    expect(fileKind('tokenizer.model')).toBe('tokenizer');
    expect(fileKind('README.md')).toBe('other');
  });
  it('takes the LFS size and sha256 when there is one', () => {
    const oid = 'a'.repeat(64);
    expect(toHubFile({ path: 'model.safetensors', size: 1, lfs: { size: 10, oid } })).toEqual({ path: 'model.safetensors', sizeBytes: 10, kind: 'weights', sha256: oid });
    expect(toHubFile({ path: 'config.json', size: 5, lfs: { oid: 'short' } })).toEqual({ path: 'config.json', sizeBytes: 5, kind: 'config' });
    expect(toHubFile({ path: 'x' })).toEqual({ path: 'x', sizeBytes: 0, kind: 'other' });
  });
});

describe('toHubModel', () => {
  it('reads the fields the command uses', () => {
    expect(
      toHubModel({ id: 'org/Model-8B', tags: ['safetensors', 'license:apache-2.0'], safetensors: { total: 8_190_735_360 }, gated: false, pipeline_tag: 'text-generation' }),
    ).toEqual({ repo: 'org/Model-8B', author: 'org', name: 'Model-8B', gated: false, formats: ['safetensors'], tags: ['safetensors', 'license:apache-2.0'], license: 'apache-2.0', paramsB: 8.191, pipelineTag: 'text-generation' });
  });
  it('the card license wins; gated kinds; MLX and missing safetensors are not models for the engines', () => {
    const m = toHubModel({ modelId: 'org/m-mlx', tags: ['safetensors', 'MLX', 'license:mit'], cardData: { license: 'other' }, gated: 'manual' });
    expect(m).toMatchObject({ license: 'other', gated: 'manual', formats: ['other'] });
    expect(toHubModel({ id: 'org/a', gated: 'auto', safetensors: {} })).toMatchObject({ gated: 'auto', formats: ['safetensors'] });
    expect(toHubModel({ id: 'org/a', gated: true })).toMatchObject({ gated: true, formats: ['other'] });
    expect(toHubModel({ id: 'noslash', tags: 'x' })).toMatchObject({ author: '', name: 'noslash', tags: [] });
    expect(toHubModel({ id: 'org/a' })).not.toHaveProperty('paramsB');
  });
});

describe('HubClient', () => {
  it('sends the token only to the hub itself', () => {
    const { hub } = client({}, 'secret');
    expect(hub.authHeaders(`${BASE}/api/models/x`)).toEqual({ authorization: 'Bearer secret' });
    expect(hub.authHeaders('https://cdn.example/x')).toEqual({});
    expect(client({}).hub.authHeaders(`${BASE}/x`)).toEqual({});
    expect(hub.resolveUrl('org/m', 'refs/pr/1', 'FP8/model 1.safetensors')).toBe(`${BASE}/org/m/resolve/refs%2Fpr%2F1/FP8/model%201.safetensors`);
  });

  it('maps network failures and HTTP errors, releasing the body', async () => {
    const down = new HubClient({ fetch: (() => Promise.reject(new Error('ECONNRESET'))) as typeof fetch, baseUrl: BASE, getToken: () => undefined });
    await expect(down.info('org/m')).rejects.toMatchObject({ code: 'NETWORK', message: expect.stringContaining('ECONNRESET') });
    const thrower = new HubClient({ fetch: (() => Promise.reject('plain')) as typeof fetch, baseUrl: BASE, getToken: () => undefined });
    await expect(thrower.info('org/m')).rejects.toMatchObject({ message: expect.stringContaining('plain') });
    // A body that cannot be cancelled any more is ignored.
    const odd = { ok: false, status: 404, headers: new Headers(), body: { cancel: () => Promise.reject(new Error('closed')) } } as unknown as Response;
    const closed = new HubClient({ fetch: (() => Promise.resolve(odd)) as typeof fetch, baseUrl: BASE, getToken: () => undefined });
    await expect(closed.info('org/m')).rejects.toMatchObject({ code: 'HF_NOT_FOUND' });
    const empty = { ok: false, status: 500, headers: new Headers(), body: null } as unknown as Response;
    await expect(new HubClient({ fetch: (() => Promise.resolve(empty)) as typeof fetch, baseUrl: BASE, getToken: () => undefined }).info('org/m')).rejects.toMatchObject({ code: 'NETWORK' });
  });

  it('searches safetensors models, most downloaded first', async () => {
    const url = `${BASE}/api/models?search=qwen+8b&filter=safetensors&sort=downloads&direction=-1&limit=100&expand%5B%5D=gated&expand%5B%5D=cardData&expand%5B%5D=safetensors&expand%5B%5D=pipeline_tag&expand%5B%5D=tags`;
    const { hub } = client({ [url]: () => json([{ id: 'Qwen/Qwen3-8B', tags: ['safetensors'] }, { id: 'loose' }, 7]) });
    expect((await hub.search(' qwen 8b ', { format: 'safetensors', limit: 500 })).models.map((m) => m.repo)).toEqual(['Qwen/Qwen3-8B']);
    const plain = `${BASE}/api/models?sort=downloads&direction=-1&limit=1&expand%5B%5D=gated&expand%5B%5D=cardData&expand%5B%5D=safetensors&expand%5B%5D=pipeline_tag&expand%5B%5D=tags`;
    const odd = client({ [plain]: () => json({ not: 'a list' }) });
    await expect(odd.hub.search('', { limit: 0 })).rejects.toThrow('Unexpected search response');
    const defaults = `${BASE}/api/models?sort=downloads&direction=-1&limit=30&expand%5B%5D=gated&expand%5B%5D=cardData&expand%5B%5D=safetensors&expand%5B%5D=pipeline_tag&expand%5B%5D=tags`;
    const bad = client({ [defaults]: () => new Response('{oops', { status: 200 }) });
    await expect(bad.hub.search('')).rejects.toMatchObject({ code: 'NETWORK', message: expect.stringContaining('invalid JSON') });
  });

  it('reads repo info at a revision', async () => {
    const { hub } = client({
      [`${BASE}/api/models/org/m`]: () => json({ sha: 'abc', private: true, tags: ['safetensors'] }),
      [`${BASE}/api/models/org/m/revision/v1`]: () => json([]),
    });
    expect(await hub.info('org/m')).toMatchObject({ repo: 'org/m', sha: 'abc', private: true, formats: ['safetensors'] });
    const at = await hub.info('org/m', 'v1');
    expect(at).toMatchObject({ repo: 'org/m', private: false });
    expect(at).not.toHaveProperty('sha');
  });

  it('lists every file, following the pages', async () => {
    const first = `${BASE}/api/models/org/m/tree/main?recursive=true`;
    const second = `${BASE}/api/models/org/m/tree/main?recursive=true&cursor=2`;
    const { hub } = client({
      [first]: () => json([{ type: 'file', path: 'config.json', size: 10 }, { type: 'directory', path: 'FP8' }, null], { headers: { link: `<${second}>; rel="next"` } }),
      [second]: () => json([{ type: 'file', path: 'model.safetensors', size: 99 }, { type: 'file', path: 3 }]),
    });
    expect((await hub.files('org/m')).map((f) => f.path)).toEqual(['config.json', 'model.safetensors']);
    const odd = client({ [`${BASE}/api/models/org/m/tree/v2?recursive=true`]: () => json({}) });
    await expect(odd.hub.files('org/m', 'v2')).rejects.toThrow('Unexpected file list');
  });

  it('reads a small JSON file', async () => {
    const { hub } = client({ [`${BASE}/org/m/resolve/main/config.json`]: () => json({ hidden_size: 8 }) });
    expect(await hub.json('org/m', 'config.json')).toEqual({ hidden_size: 8 });
  });

  it('reads a safetensors header with two Range requests, from servers that honour Range or not', async () => {
    const header = { __metadata__: { format: 'pt' }, w: { dtype: 'BF16', shape: [2, 2], data_offsets: [0, 8] } };
    const bytes = safetensors(header);
    for (const honour of [true, false]) {
      const { hub, calls } = client({ [`${BASE}/org/m/resolve/main/model.safetensors`]: served(bytes, honour) }, 't');
      expect(await hub.safetensorsHeader('org/m', 'model.safetensors')).toEqual({ header, headerBytes: bytes.length - 8 });
      expect(calls.map((c) => new Headers(c.init?.headers).get('range'))).toEqual(['bytes=0-7', `bytes=8-${bytes.length - 1}`]);
    }
  });

  it('refuses broken headers', async () => {
    const tooBig = new Uint8Array(8);
    new DataView(tooBig.buffer).setBigUint64(0, BigInt(SAFETENSORS_HEADER_MAX + 1), true);
    const notJson = new Uint8Array(12);
    new DataView(notJson.buffer).setBigUint64(0, 4n, true);
    notJson.set(new TextEncoder().encode('{{{{'), 8);
    const cases: [Uint8Array, string][] = [
      [new Uint8Array(4), 'Bad safetensors header'],
      [tooBig, 'Bad safetensors header'],
      [notJson, 'Bad safetensors header in org/m/x.safetensors:'],
    ];
    for (const [bytes, message] of cases) {
      const { hub } = client({ [`${BASE}/org/m/resolve/main/x.safetensors`]: served(bytes, true) });
      await expect(hub.safetensorsHeader('org/m', 'x.safetensors')).rejects.toMatchObject({ code: 'MODEL_INVALID', message: expect.stringContaining(message) });
    }
  });
});

describe('readCapped', () => {
  const stream = (...chunks: number[][]) =>
    new Response(
      new ReadableStream<Uint8Array>({
        start(c) {
          for (const ch of chunks) c.enqueue(new Uint8Array(ch));
          c.close();
        },
      }),
    );
  it('keeps bytes [skip, limit) across chunks and stops reading at the limit', async () => {
    expect([...(await readCapped(stream([0, 1, 2], [3, 4, 5], [6, 7]), 5, 2))]).toEqual([2, 3, 4]);
    expect([...(await readCapped(stream([0, 1], [2]), 10))]).toEqual([0, 1, 2]);
    expect([...(await readCapped(stream([0, 1, 2, 3]), 4, 4))]).toEqual([]);
  });
});

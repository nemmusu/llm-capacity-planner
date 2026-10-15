import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { cachedFetch, defaultCacheDir } from '../../src/cli/http-cache';

describe('defaultCacheDir', () => {
  it('the user cache folder of each system, or the one the environment names', () => {
    expect(defaultCacheDir({ CAPACITY_PLANNER_CACHE: '/x/cache' }, 'linux', '/home/u')).toBe('/x/cache');
    expect(defaultCacheDir({ LOCALAPPDATA: 'D:\\Local' }, 'win32', 'C:\\Users\\u')).toBe(`D:\\Local\\${__CLI_NAME__}\\cache`);
    expect(defaultCacheDir({}, 'win32', 'C:\\Users\\u')).toBe(`C:\\Users\\u\\AppData\\Local\\${__CLI_NAME__}\\cache`);
    expect(defaultCacheDir({}, 'darwin', '/Users/u')).toBe(`/Users/u/Library/Caches/${__CLI_NAME__}`);
    expect(defaultCacheDir({ XDG_CACHE_HOME: '/var/c' }, 'linux', '/home/u')).toBe(`/var/c/${__CLI_NAME__}`);
    expect(defaultCacheDir({}, 'linux', '/home/u')).toBe(`/home/u/.cache/${__CLI_NAME__}`);
    // The defaults: this process.
    expect(typeof defaultCacheDir()).toBe('string');
  });
});

describe('cachedFetch', () => {
  let dir: string;
  let now: number;
  let calls: { url: string; range: string | null }[];
  let answer: () => Promise<Response>;
  const inner = (async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
    calls.push({ url: String(input), range: new Headers(init?.headers).get('range') });
    return answer();
  }) as typeof fetch;
  const ok = (body: string, status = 200) => () => Promise.resolve(new Response(body, { status, headers: { 'content-type': 'text/plain', 'x-other': 'no', link: '<next>; rel="next"' } }));

  beforeEach(() => {
    dir = mkdtempSync(path.join(os.tmpdir(), 'vram-cache-'));
    now = 1_000_000;
    calls = [];
    answer = ok('uno');
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  const f = (over: { offline?: boolean; refresh?: boolean; maxAgeMs?: number; dir?: string } = {}) => cachedFetch(inner, { dir, now: () => now, ...over });

  it('a miss asks the network and stores the answer; a hit does not ask again', async () => {
    const r1 = await f()('https://hub.test/a');
    expect(await r1.text()).toBe('uno');
    expect(r1.headers.get('content-type')).toBe('text/plain');
    answer = ok('due');
    const r2 = await f()('https://hub.test/a');
    expect(await r2.text()).toBe('uno');
    // Only the kept headers are stored.
    expect(r2.headers.get('link')).toBe('<next>; rel="next"');
    expect(r2.headers.get('x-other')).toBeNull();
    expect(calls).toHaveLength(1);
  });

  it('the range and the token are part of the key; 206 answers are stored', async () => {
    answer = ok('parte', 206);
    const get = f();
    expect((await get('https://hub.test/w', { headers: { range: 'bytes=0-7' } })).status).toBe(206);
    expect((await get('https://hub.test/w', { headers: { range: 'bytes=0-7' } })).status).toBe(206);
    await get('https://hub.test/w', { headers: { range: 'bytes=8-9' } });
    await get('https://hub.test/w', { headers: { range: 'bytes=0-7', authorization: 'Bearer t' } });
    expect(calls.map((c) => c.range)).toEqual(['bytes=0-7', 'bytes=8-9', 'bytes=0-7']);
  });

  it('a stale entry is refreshed; searches go stale after a day, the rest after a week', async () => {
    await f()('https://hub.test/api/models?search=x');
    await f()('https://hub.test/a');
    now += 2 * 24 * 60 * 60 * 1000;
    answer = ok('nuovo');
    expect(await (await f()('https://hub.test/a')).text()).toBe('uno');
    expect(await (await f()('https://hub.test/api/models?search=x')).text()).toBe('nuovo');
    // A maxAge of its own.
    expect(await (await f({ maxAgeMs: 1000 })('https://hub.test/a')).text()).toBe('nuovo');
    expect(calls).toHaveLength(4);
  });

  it('a network error replays a stale entry; without one it throws', async () => {
    await f()('https://hub.test/a');
    now += 30 * 24 * 60 * 60 * 1000;
    answer = () => Promise.reject(new Error('rete giù'));
    expect(await (await f()('https://hub.test/a')).text()).toBe('uno');
    await expect(f()('https://hub.test/b')).rejects.toThrow('rete giù');
  });

  it('offline: an entry of any age, or an error without one', async () => {
    await f()('https://hub.test/a');
    now += 365 * 24 * 60 * 60 * 1000;
    expect(await (await f({ offline: true })('https://hub.test/a')).text()).toBe('uno');
    await expect(f({ offline: true })('https://hub.test/b')).rejects.toThrow('offline: https://hub.test/b is not in the local cache');
    expect(calls).toHaveLength(1);
  });

  it('refresh: the network even for a fresh entry, the entry when the network fails', async () => {
    await f()('https://hub.test/a');
    answer = ok('due');
    expect(await (await f({ refresh: true })('https://hub.test/a')).text()).toBe('due');
    expect(calls).toHaveLength(2);
    answer = () => Promise.reject(new Error('down'));
    expect(await (await f({ refresh: true })('https://hub.test/a')).text()).toBe('due');
  });

  it('says how many bytes each answer had and where it came from', async () => {
    const seen: [number, string][] = [];
    const onRead = (bytes: number, from: 'network' | 'cache') => seen.push([bytes, from]);
    await cachedFetch(inner, { dir, now: () => now, onRead })('https://hub.test/a');
    await cachedFetch(inner, { dir, now: () => now, onRead })('https://hub.test/a');
    expect(seen).toEqual([
      [3, 'network'],
      [3, 'cache'],
    ]);
  });

  it('errors are passed on and not stored', async () => {
    answer = ok('manca', 404);
    expect((await f()('https://hub.test/a')).status).toBe(404);
    answer = ok('c’è');
    expect(await (await f()('https://hub.test/a')).text()).toBe('c’è');
    expect(calls).toHaveLength(2);
  });

  it('a cache folder that cannot be written only costs the speed', async () => {
    // A file where the folder should be: mkdir fails.
    const blocked = path.join(dir, 'file');
    writeFileSync(blocked, 'x');
    const get = f({ dir: blocked });
    expect(await (await get('https://hub.test/a')).text()).toBe('uno');
    expect(await (await get('https://hub.test/a')).text()).toBe('uno');
    expect(calls).toHaveLength(2);
  });

  it('a corrupt entry counts as a miss', async () => {
    await f()('https://hub.test/a');
    const sub = readdirSync(dir)[0] as string;
    const file = readdirSync(path.join(dir, sub))[0] as string;
    writeFileSync(path.join(dir, sub, file), '{rotto');
    answer = ok('di nuovo');
    expect(await (await f()('https://hub.test/a')).text()).toBe('di nuovo');
  });
});

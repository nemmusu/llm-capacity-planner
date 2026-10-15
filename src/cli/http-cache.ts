/**
 * A disk cache in front of fetch for the command: Hugging Face answers (model info, file lists,
 * config.json, safetensors headers read with Range) are kept, so a model read once works offline and
 * the 300 Range requests of a large checkpoint are made once. Fresh entries are used as they are;
 * older ones are refreshed, and used anyway when the network fails.
 */
import { createHash } from 'node:crypto';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

/** Where the cache lives on each system (the user's cache folder, never the program's). */
export function defaultCacheDir(env: Record<string, string | undefined> = process.env, platform: NodeJS.Platform = process.platform, home = os.homedir()): string {
  if (env.CAPACITY_PLANNER_CACHE) return env.CAPACITY_PLANNER_CACHE;
  if (platform === 'win32') return path.win32.join(env.LOCALAPPDATA ?? path.win32.join(home, 'AppData', 'Local'), __CLI_NAME__, 'cache');
  if (platform === 'darwin') return path.posix.join(home, 'Library', 'Caches', __CLI_NAME__);
  return path.posix.join(env.XDG_CACHE_HOME ?? path.posix.join(home, '.cache'), __CLI_NAME__);
}

interface Entry {
  status: number;
  headers: Record<string, string>;
  /** base64 body. */
  body: string;
  savedAt: number;
}

export interface CacheOptions {
  dir: string;
  /** How long an answer is used without asking again, ms (default 7 days; searches 1 day). */
  maxAgeMs?: number;
  now?: () => number;
  /** Never touch the network (fail when an answer is not cached). */
  offline?: boolean;
  /** Ask the network even for fresh entries (the cache is still the fallback when it fails). */
  refresh?: boolean;
  /** Every answer's body size and where it came from (to say how much was read). */
  onRead?: (bytes: number, from: 'network' | 'cache') => void;
}

const KEPT_HEADERS = ['content-type', 'link', 'content-range', 'x-error-code'];
const DAY = 24 * 60 * 60 * 1000;

/** fetch with the disk cache (GET only; 200 and 206 answers are stored). */
export function cachedFetch(inner: typeof fetch, opts: CacheOptions): typeof fetch {
  const now = opts.now ?? Date.now;
  return (async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
    const url = String(input);
    const range = new Headers(init?.headers).get('range') ?? '';
    // The token changes what a gated repo answers: it is part of the key (hashed, never stored).
    const auth = new Headers(init?.headers).get('authorization') ?? '';
    const key = createHash('sha256').update(`${url}\n${range}\n${auth}`).digest('hex');
    const file = path.join(opts.dir, key.slice(0, 2), `${key}.json`);
    const maxAge = opts.maxAgeMs ?? (url.includes('/api/models?') ? DAY : 7 * DAY);
    const cached = await readFile(file, 'utf8')
      .then((t) => JSON.parse(t) as Entry)
      .catch(() => undefined);
    const replay = (e: Entry, from: 'network' | 'cache' = 'cache') => {
      const body = Buffer.from(e.body, 'base64');
      opts.onRead?.(body.length, from);
      return new Response(body, { status: e.status, headers: e.headers });
    };
    if (cached && (opts.offline || (!opts.refresh && now() - cached.savedAt < maxAge))) return replay(cached);
    if (opts.offline) throw new Error(`offline: ${url} is not in the local cache`);
    let res: Response;
    try {
      res = await inner(input, init);
    } catch (e) {
      if (cached) return replay(cached);
      throw e;
    }
    if (res.status !== 200 && res.status !== 206) return res;
    const body = Buffer.from(await res.arrayBuffer());
    const headers: Record<string, string> = {};
    for (const h of KEPT_HEADERS) {
      const v = res.headers.get(h);
      if (v !== null) headers[h] = v;
    }
    const entry: Entry = { status: res.status, headers, body: body.toString('base64'), savedAt: now() };
    // Written next to the target and renamed: a crash never leaves half a file. A cache that cannot be
    // written (read-only folder, full disk) only costs the next run its speed.
    const tmp = `${file}.${process.pid}.tmp`;
    await mkdir(path.dirname(file), { recursive: true })
      .then(() => writeFile(tmp, JSON.stringify(entry)))
      .then(() => rename(tmp, file))
      .catch(() => undefined);
    return replay(entry, 'network');
  }) as typeof fetch;
}

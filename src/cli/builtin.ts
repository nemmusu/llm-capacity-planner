/**
 * The models of the catalog, bundled with the command: their config.json, shard index and safetensors headers
 * (data/models/, packed by scripts/pack-models.mjs), served as the Hugging Face hub would serve them. For these
 * models the command needs no network; `--refresh` reads them from Hugging Face again.
 *
 * Pack layout: "MODELS01", the index's length (u32 LE), the index (JSON), then one gzip blob per repo.
 */
import { gunzipSync } from 'node:zlib';

export const PACK_MAGIC = 'MODELS01';

/** One repo of the pack, as stored in its blob. */
export interface PackedRepo {
  /** Files served as they are: config.json, generation_config.json, params.json, the shard index. */
  json: Record<string, unknown>;
  /** Each safetensors file: its size and its header (dtype and shape of every tensor, the metadata). */
  shards: Record<string, { fileBytes: number; header: Record<string, unknown> }>;
}

export interface PackIndex {
  /** When the data was read from Hugging Face (YYYY-MM-DD). */
  date: string;
  /** Fingerprint of the files the pack was made of (scripts/pack-models.mjs modelsHash). */
  source?: string;
  repos: Record<string, { revision: string; gated: boolean | string; pipelineTag: string | null; total: number | null; offset: number; length: number }>;
}

export interface BuiltinModels {
  date: string;
  has(repo: string, revision: string): boolean;
  fetch: typeof fetch;
}

/** The pack's index and a reader of one repo. */
export function readPack(pack: Uint8Array): { index: PackIndex; repo: (name: string) => PackedRepo } {
  const view = new DataView(pack.buffer, pack.byteOffset, pack.byteLength);
  if (new TextDecoder().decode(pack.subarray(0, 8)) !== PACK_MAGIC) throw new Error('Not a model pack');
  const length = view.getUint32(8, true);
  const index = JSON.parse(new TextDecoder().decode(pack.subarray(12, 12 + length))) as PackIndex;
  const start = 12 + length;
  const cache = new Map<string, PackedRepo>();
  return {
    index,
    repo: (name) => {
      const hit = cache.get(name);
      if (hit) return hit;
      const e = index.repos[name] as PackIndex['repos'][string];
      const repo = JSON.parse(gunzipSync(pack.subarray(start + e.offset, start + e.offset + e.length)).toString('utf8')) as PackedRepo;
      cache.set(name, repo);
      return repo;
    },
  };
}

const json = (v: unknown) => new Response(JSON.stringify(v), { status: 200, headers: { 'content-type': 'application/json' } });
const notFound = () => new Response(null, { status: 404, headers: { 'x-error-code': 'EntryNotFound' } });

/** The safetensors bytes [start, end] of a header: u64 LE length, then the JSON. */
function headerBytes(header: Record<string, unknown>, range: string | null): Response {
  const body = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(8 + body.length);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(body.length), true);
  bytes.set(body, 8);
  const m = /^bytes=(\d+)-(\d+)$/.exec(range ?? '');
  if (!m) return new Response(bytes, { status: 200 });
  return new Response(bytes.slice(Number(m[1]), Number(m[2]) + 1), { status: 206 });
}

/** The models of a pack as a hub: answers only for its repos at "main" or their pinned commit. */
export function builtinModels(pack: Uint8Array, baseUrl: string): BuiltinModels {
  const { index, repo } = readPack(pack);
  const base = baseUrl.replace(/\/+$/, '');
  const known = (name: string, revision: string) => {
    const e = index.repos[name];
    return e !== undefined && (revision === 'main' || revision === e.revision);
  };
  const fetchFn = (async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
    const url = new URL(String(input));
    const p = decodeURIComponent(url.pathname);
    const api = /^\/api\/models\/([^/]+\/[^/]+)(?:\/revision\/(.+)|\/tree\/(.+))?$/.exec(p);
    if (url.origin === new URL(base).origin && api) {
      const [, name, rev, treeRev] = api as unknown as [string, string, string | undefined, string | undefined];
      if (!known(name, rev ?? treeRev ?? 'main')) return notFound();
      const e = index.repos[name] as PackIndex['repos'][string];
      if (treeRev === undefined)
        return json({
          id: name,
          sha: e.revision,
          gated: e.gated,
          tags: ['safetensors'],
          ...(e.total ? { safetensors: { total: e.total } } : {}),
          ...(e.pipelineTag ? { pipeline_tag: e.pipelineTag } : {}),
        });
      const r = repo(name);
      return json([
        ...Object.entries(r.json).map(([path, v]) => ({ type: 'file', path, size: JSON.stringify(v).length })),
        ...Object.entries(r.shards).map(([path, s]) => ({ type: 'file', path, size: s.fileBytes })),
      ]);
    }
    const file = /^\/([^/]+\/[^/]+)\/resolve\/([^/]+)\/(.+)$/.exec(p);
    if (url.origin === new URL(base).origin && file) {
      const [, name, rev, path] = file as unknown as [string, string, string, string];
      if (!known(name, rev)) return notFound();
      const r = repo(name);
      if (path in r.json) return json(r.json[path]);
      const shard = r.shards[path];
      if (shard) return headerBytes(shard.header, new Headers(init?.headers).get('range'));
    }
    return notFound();
  }) as typeof fetch;
  return { date: index.date, has: known, fetch: fetchFn };
}

/**
 * Hugging Face Hub client: repo info, search, file listing, small JSON files and safetensors headers. The weights
 * are never downloaded: a safetensors header is read with two small HTTP Range requests.
 */
import path from 'node:path';
import { AppError, errorMessage } from '../contract/errors';
import { round3 } from '../model/layers';
import type { TokenFn } from '../model/types';
import type { HubFile, HubModel } from './types';

export interface SearchOptions {
  /** Only repos with safetensors weights. */
  format?: 'safetensors';
  limit?: number;
}

/** One page of a hub listing. */
export interface HubPage {
  models: HubModel[];
}

export interface HubRepoInfo extends HubModel {
  /** Commit sha of the requested revision. */
  sha?: string;
  private: boolean;
}

export interface HubClientOptions {
  fetch: typeof fetch;
  /** e.g. https://huggingface.co (tests: a fake hub). */
  baseUrl: string;
  getToken: TokenFn;
}

/** A safetensors JSON header larger than this is refused (real ones are a few MB at most). */
export const SAFETENSORS_HEADER_MAX = 100 * 1024 * 1024;

const SEARCH_EXPAND = ['gated', 'cardData', 'safetensors', 'pipeline_tag', 'tags'];

/**
 * Maps an HF error response to an AppError. HF sends `X-Error-Code` (GatedRepo, RepoNotFound,
 * EntryNotFound, RevisionNotFound); a missing or private repo without a token is a bare 401.
 */
export function hubHttpError(res: Response, repo?: string): AppError {
  const code = res.headers.get('x-error-code') ?? '';
  const detail = res.headers.get('x-error-message') ?? undefined;
  const params: Record<string, string | number> = { status: res.status };
  if (repo) params.repo = repo;
  const opts = { params, ...(detail ? { detail } : {}) };
  if (code === 'GatedRepo' || res.status === 403) return new AppError('HF_GATED', `Access to ${repo ?? 'the repo'} requires accepting its license`, opts);
  if (res.status === 401) return new AppError('HF_AUTH_REQUIRED', 'Hugging Face requires a valid token for this repo', opts);
  if (res.status === 404) return new AppError('HF_NOT_FOUND', `Not found on Hugging Face (${code || 'HTTP 404'})`, opts);
  if (res.status === 429) return new AppError('HF_RATE_LIMITED', 'Hugging Face rate limit reached', opts);
  return new AppError('NETWORK', `Hugging Face returned HTTP ${res.status}`, opts);
}

/** Releases the connection of a response whose body is not needed. */
async function discardBody(res: Response): Promise<void> {
  try {
    await res.body?.cancel();
  } catch {
    /* already closed */
  }
}

function obj(v: unknown): Record<string, unknown> | undefined {
  return v !== null && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : undefined;
}
function strs(v: unknown): string[] {
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

/** Speculative-decoding drafts: EAGLE heads, "draft-…" / "…-draft-…" models. */
const DRAFT_RE = /(^|[-_.])(draft|eagle\d*)([-_.]|$)/i;

/** File role inside a repo, from its path. */
export function fileKind(p: string): HubFile['kind'] {
  const base = path.posix.basename(p).toLowerCase();
  if (base.endsWith('.safetensors') || /^pytorch_model.*\.bin$/.test(base)) {
    // MTP heads and drafts shipped as their own folders or files ("MTP-BF16/", "model_mtp.safetensors",
    // "DFlash2-FP8/"): extras of a model, never one.
    const stem = base.replace(/\.(safetensors|bin)$/, '');
    for (const seg of [...path.posix.dirname(p).split('/').filter((x) => x !== '.'), stem]) {
      if (/(^|[-_.])mtp([-_.]|$)/i.test(seg)) return 'mtp';
      if (/^dflash/i.test(seg) || DRAFT_RE.test(seg)) return 'draft';
    }
    return 'weights';
  }
  if (['config.json', 'generation_config.json', 'params.json', 'preprocessor_config.json'].includes(base) || base.endsWith('.index.json')) return 'config';
  if (base.startsWith('tokenizer') || base.startsWith('chat_template') || ['vocab.json', 'merges.txt', 'special_tokens_map.json', 'added_tokens.json'].includes(base) || base.endsWith('.model')) return 'tokenizer';
  return 'other';
}

export function toHubFile(entry: Record<string, unknown>): HubFile {
  const p = String(entry.path);
  const lfs = obj(entry.lfs);
  const oid = typeof lfs?.oid === 'string' && /^[0-9a-f]{64}$/.test(lfs.oid) ? lfs.oid : undefined;
  const file: HubFile = { path: p, sizeBytes: Number(lfs?.size ?? entry.size ?? 0), kind: fileKind(p) };
  if (oid) file.sha256 = oid;
  return file;
}

export function toHubModel(raw: Record<string, unknown>): HubModel {
  const repo = String(raw.id ?? raw.modelId);
  const slash = repo.indexOf('/');
  const tags = strs(raw.tags);
  const card = obj(raw.cardData) ?? {};
  // MLX repos ship safetensors too, in a layout only MLX reads: not a model for vLLM / SGLang.
  const safetensors = (tags.includes('safetensors') || obj(raw.safetensors) !== undefined) && !tags.some((t) => t.toLowerCase() === 'mlx');
  const total = Number(obj(raw.safetensors)?.total ?? 0);
  const license = (typeof card.license === 'string' ? card.license : undefined) ?? tags.find((t) => t.startsWith('license:'))?.slice('license:'.length);
  const model: HubModel = {
    repo,
    author: slash > 0 ? repo.slice(0, slash) : '',
    name: repo.slice(slash + 1),
    gated: raw.gated === 'auto' || raw.gated === 'manual' ? raw.gated : raw.gated === true,
    formats: [safetensors ? 'safetensors' : 'other'],
    tags,
  };
  if (license) model.license = license;
  if (total > 0) model.paramsB = round3(total / 1e9);
  if (typeof raw.pipeline_tag === 'string') model.pipelineTag = raw.pipeline_tag;
  return model;
}

async function readJson(res: Response): Promise<unknown> {
  try {
    return await res.json();
  } catch (e) {
    throw new AppError('NETWORK', `Hugging Face sent invalid JSON: ${errorMessage(e)}`, { cause: e });
  }
}

/** Hugging Face Hub API client on the injected fetch (which caches on disk: cli/http-cache.ts). */
export class HubClient {
  readonly baseUrl: string;

  constructor(private readonly opts: HubClientOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, '');
  }

  /** Authorization header, sent only to the hub's own origin (never to CDN redirect targets). */
  authHeaders(url: string): Record<string, string> {
    const token = this.opts.getToken();
    if (!token || new URL(url).origin !== new URL(this.baseUrl).origin) return {};
    return { authorization: `Bearer ${token}` };
  }

  resolveUrl(repo: string, revision: string, file: string): string {
    const encoded = file.split('/').map(encodeURIComponent).join('/');
    return `${this.baseUrl}/${repo}/resolve/${encodeURIComponent(revision)}/${encoded}`;
  }

  /** GET with auth, network-error and HTTP-error mapping. */
  async request(url: string, repo?: string, init: RequestInit = {}): Promise<Response> {
    let res: Response;
    try {
      res = await this.opts.fetch(url, { ...init, headers: { ...this.authHeaders(url), ...(init.headers as Record<string, string>) } });
    } catch (e) {
      throw new AppError('NETWORK', `Cannot reach Hugging Face: ${errorMessage(e)}`, { cause: e });
    }
    if (!res.ok) {
      await discardBody(res);
      throw hubHttpError(res, repo);
    }
    return res;
  }

  /** One page of a hub search, most downloaded first. */
  async search(q: string, opts: SearchOptions = {}): Promise<HubPage> {
    const p = new URLSearchParams();
    if (q.trim()) p.set('search', q.trim());
    if (opts.format) p.append('filter', opts.format);
    p.set('sort', 'downloads');
    p.set('direction', '-1');
    p.set('limit', String(Math.min(100, Math.max(1, Math.floor(opts.limit ?? 30)))));
    for (const e of SEARCH_EXPAND) p.append('expand[]', e);
    const raw = await readJson(await this.request(`${this.baseUrl}/api/models?${p.toString()}`));
    if (!Array.isArray(raw)) throw new AppError('NETWORK', 'Unexpected search response from Hugging Face');
    return { models: raw.map((r) => toHubModel(obj(r) ?? {})).filter((m) => m.repo.includes('/')) };
  }

  async info(repo: string, revision?: string): Promise<HubRepoInfo> {
    const url = revision ? `${this.baseUrl}/api/models/${repo}/revision/${encodeURIComponent(revision)}` : `${this.baseUrl}/api/models/${repo}`;
    const raw = obj(await readJson(await this.request(url, repo))) ?? {};
    const info: HubRepoInfo = { ...toHubModel({ id: repo, ...raw }), private: raw.private === true };
    if (typeof raw.sha === 'string') info.sha = raw.sha;
    return info;
  }

  /** Every file of a revision with sizes and LFS sha256 (follows `Link: rel="next"` pages). */
  async files(repo: string, revision = 'main'): Promise<HubFile[]> {
    const out: HubFile[] = [];
    let url: string | undefined = `${this.baseUrl}/api/models/${repo}/tree/${encodeURIComponent(revision)}?recursive=true`;
    for (let page = 0; url && page < 20; page++) {
      const res = await this.request(url, repo);
      const raw = await readJson(res);
      if (!Array.isArray(raw)) throw new AppError('NETWORK', 'Unexpected file list from Hugging Face');
      for (const e of raw) {
        const entry = obj(e);
        if (entry?.type === 'file' && typeof entry.path === 'string') out.push(toHubFile(entry));
      }
      const next = /<([^>]+)>;\s*rel="next"/.exec(res.headers.get('link') ?? '')?.[1];
      url = next ? new URL(next, this.baseUrl).toString() : undefined;
    }
    return out;
  }

  /** A small JSON file of a repo (config.json, the shard index). */
  async json(repo: string, file: string, revision = 'main'): Promise<unknown> {
    return readJson(await this.request(this.resolveUrl(repo, revision, file), repo));
  }

  /**
   * The JSON header of a safetensors file (8-byte little-endian length, then the JSON with each tensor's
   * dtype and shape), read with two small Range requests: the weights are never downloaded.
   */
  async safetensorsHeader(repo: string, file: string, revision = 'main'): Promise<{ header: unknown; headerBytes: number }> {
    const url = this.resolveUrl(repo, revision, file);
    const head = await this.range(url, repo, 0, 7);
    const len = head.length === 8 ? Number(new DataView(head.buffer, head.byteOffset, 8).getBigUint64(0, true)) : 0;
    if (len <= 1 || len > SAFETENSORS_HEADER_MAX) throw new AppError('MODEL_INVALID', `Bad safetensors header in ${repo}/${file}`, { params: { repo } });
    const body = await this.range(url, repo, 8, 8 + len - 1);
    try {
      return { header: JSON.parse(new TextDecoder().decode(body)) as unknown, headerBytes: len };
    } catch (e) {
      throw new AppError('MODEL_INVALID', `Bad safetensors header in ${repo}/${file}: ${errorMessage(e)}`, { params: { repo }, cause: e });
    }
  }

  /** Bytes [start, end] of a file; a server that ignores Range is read only up to `end` (a 20 GB body is never buffered). */
  private async range(url: string, repo: string, start: number, end: number): Promise<Uint8Array> {
    const res = await this.request(url, repo, { headers: { range: `bytes=${start}-${end}` } });
    if (res.status === 206) return new Uint8Array(await res.arrayBuffer());
    return readCapped(res, end + 1, start);
  }
}

/** Reads body bytes [skip, limit) and cancels the rest of the stream. */
export async function readCapped(res: Response, limit: number, skip = 0): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  let pos = 0;
  const reader = (res.body as ReadableStream<Uint8Array>).getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    const from = Math.max(0, skip - pos);
    const to = Math.min(value.length, limit - pos);
    if (to > from) chunks.push(value.subarray(from, to));
    pos += value.length;
    if (pos >= limit) {
      await reader.cancel();
      break;
    }
  }
  const out = new Uint8Array(chunks.reduce((a, c) => a + c.length, 0));
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out;
}

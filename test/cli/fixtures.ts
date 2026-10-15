/** Shared helpers of the vram command tests: real models and rows, and a fake Hugging Face. */
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { factsFromHfConfig, loadShapeFromHfConfig } from '../../src/model/hf-config';
import type { TensorInfo } from '../../src/model/weights-exact';
import { DEFAULT_WORKLOAD, type Workload } from '../../src/planner/capacity';
import type { GpuLayout } from '../../src/planner/hardware';
import { plannerModel } from '../../src/planner/model';
import { planLayout, type PlanOptions, type PlanRow } from '../../src/planner/plan';
import { catalog } from '../../src/cli/run';
import type { RenderModel } from '../../src/cli/render';

export const HUB = 'https://hub.test';
export const readFixture = (...p: string[]) => JSON.parse(readFileSync(path.join('test', 'fixtures', 'hf', ...p), 'utf8')) as unknown;
export const configJson = (name: string) => readFixture('configs', `${name}.json`);

interface CompactHeaders {
  files: Record<string, { fileBytes: number; headerBytes: number }>;
  tensors: Record<string, [string, number[]]>;
}
export const qwenHeaders = readFixture('headers', 'qwen3-8b.json') as CompactHeaders;
export const qwenTensors: TensorInfo[] = Object.entries(qwenHeaders.tensors).map(([name, [dtype, shape]]) => ({ name, dtype, shape }));

/**
 * The catalog, with the RTX PRO 6000 Server standing for a card whose speed figures are not published (no bandwidth,
 * no TFLOPS): the tests of "speed unknown" need one, whatever the data says about that card.
 */
export const NO_SPEED = 'rtx-pro-6000-blackwell-server-96gb';
export const GPUS = catalog().map((g) => (g.id === NO_SPEED ? { ...g, bandwidthGBs: 0, tflops: { bf16: null, fp8: null, fp4: null } } : g));
export const gpu = (id: string) => GPUS.find((g) => g.id === id) as GpuLayout['gpu'];
export const at = (id: string, tp = 1, pp = 1): GpuLayout => ({ gpu: gpu(id), count: tp * pp, tp, pp });

export const qwen = (() => {
  const c = configJson('qwen3-8b');
  return plannerModel(c, factsFromHfConfig(c), loadShapeFromHfConfig(c), qwenTensors);
})();
export const workload = (maxContext = 40_960, over: Partial<Workload> = {}): Workload => ({ ...DEFAULT_WORKLOAD, maxContext, typicalContext: maxContext / 2, ...over });
export const opts = (maxContext = 40_960, over: Partial<PlanOptions> = {}): PlanOptions => ({ engine: 'vllm', kvCache: 'auto', textOnly: false, workload: workload(maxContext), ...over });
export const row = (layout: GpuLayout, maxContext = 40_960): PlanRow => planLayout(qwen, layout, opts(maxContext));

export const renderModel = (over: Partial<RenderModel> = {}): RenderModel => ({
  repo: 'Qwen/Qwen3-8B',
  variant: '',
  format: 'BF16',
  family: 'denso',
  verification: qwen.verification,
  engine: 'vllm',
  cache: 'auto',
  workload: workload(),
  contextIsModelMax: true,
  modelMaxContext: 40_960,
  textOnly: false,
  hasVision: false,
  ...over,
});

export interface FakeRepo {
  info?: Record<string, unknown>;
  files: { path: string; size: number }[];
  /** config.json per folder ('' = root). */
  configs: Record<string, unknown>;
  /** Other JSON files by path (a weights index). */
  json?: Record<string, unknown>;
  /** safetensors headers by path; a weight file without one gets a single BF16 tensor of its size. */
  headers?: Record<string, Record<string, { dtype: string; shape: number[] }>>;
  status?: number;
  configStatus?: number;
}

/** The first bytes of a safetensors file: 8-byte little-endian length + the JSON header. */
export function safetensorsStart(header: object): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  return out;
}

/** A fetch that answers like huggingface.co for a few repos and one search. */
export function fakeHub(repos: Record<string, FakeRepo>, search: unknown[] = [], seen: { url: string; auth?: string }[] = []): typeof fetch {
  return (async (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
    const url = new URL(String(input));
    const auth = new Headers(init?.headers).get('authorization') ?? undefined;
    seen.push({ url: url.toString(), ...(auth ? { auth } : {}) });
    const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
    if (url.pathname === '/api/models') return json(search);
    const tree = /^\/api\/models\/([^/]+\/[^/]+)\/tree\/[^/]+$/.exec(url.pathname);
    if (tree) return json(repos[tree[1] as string]?.files.map((f) => ({ type: 'file', path: f.path, size: f.size })) ?? []);
    const info = /^\/api\/models\/([^/]+\/[^/]+)(\/revision\/.+)?$/.exec(url.pathname);
    if (info) {
      const r = repos[info[1] as string];
      if (!r) return new Response('', { status: 404, headers: { 'x-error-code': 'RepoNotFound' } });
      if (r.status) return new Response('', { status: r.status });
      return json({ id: info[1], tags: ['safetensors'], ...r.info });
    }
    const resolve = /^\/([^/]+\/[^/]+)\/resolve\/[^/]+\/(.+)$/.exec(url.pathname);
    if (resolve) {
      const r = repos[resolve[1] as string] as FakeRepo;
      if (r.configStatus) return new Response('', { status: r.configStatus, headers: { 'x-error-code': 'GatedRepo' } });
      const file = decodeURIComponent(resolve[2] as string);
      if (file.endsWith('.safetensors')) {
        const size = r.files.find((f) => f.path === file)?.size ?? 2;
        const bytes = safetensorsStart({ __metadata__: { format: 'pt' }, ...(r.headers?.[file] ?? { [`${file}.w`]: { dtype: 'BF16', shape: [size / 2] } }) });
        const [, a, b] = /bytes=(\d+)-(\d+)/.exec(new Headers(init?.headers).get('range') ?? '') ?? [];
        return new Response(bytes.slice(Number(a), Number(b) + 1), { status: 206 });
      }
      if (r.json?.[file] !== undefined) return json(r.json[file]);
      const dir = path.posix.dirname(file);
      return json(r.configs[dir === '.' ? '' : dir]);
    }
    return new Response('', { status: 500 });
  }) as typeof fetch;
}

/** Qwen3-8B as the hub serves it: real config, real file sizes and real tensor headers. */
export function qwenRepo(): FakeRepo {
  const headers: Record<string, Record<string, { dtype: string; shape: number[] }>> = {};
  const names = Object.keys(qwenHeaders.files);
  // The real index is not in the fixture: the tensors are spread over the 5 files in order.
  const tensors = Object.entries(qwenHeaders.tensors);
  tensors.forEach(([name, [dtype, shape]], i) => {
    const file = names[Math.floor((i * names.length) / tensors.length)] as string;
    (headers[file] ??= {})[name] = { dtype, shape };
  });
  return {
    files: [{ path: 'config.json', size: 700 }, ...names.map((n) => ({ path: n, size: (qwenHeaders.files[n] as { fileBytes: number }).fileBytes }))],
    configs: { '': configJson('qwen3-8b') },
    headers,
  };
}

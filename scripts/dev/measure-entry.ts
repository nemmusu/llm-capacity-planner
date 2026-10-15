// Short calibration measurements on this PC's GPU (development only; run when vLLM's version changes):
//   npm run dev:measure -- [--only=<name>]
// Each one starts vLLM (Docker image vllm/vllm-openai:v0.31.0) with dummy weights (--load-format dummy: no
// weight download) and the model cut to a few layers (--hf-overrides), reads the memory lines of its log,
// stops the container, and compares them with the planner's numbers for the same layers. Before each run
// the GPU must be free (another program using > 4 GiB stops the script). Results are appended to
// test/fixtures/rtx5090/measurements.json.
import { execFileSync, spawn } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { repoLayout } from '../../src/hub/formats';
import { cachedFetch, defaultCacheDir } from '../../src/cli/http-cache';
import { findHfToken } from '../../src/cli/hf-token';
import { readRemoteModel } from '../../src/cli/model';
import { HubClient } from '../../src/hub/client';
import { weightsPerGpu } from '../../src/model/weights-exact';
import { cutModel } from './cut';

const IMAGE = 'vllm/vllm-openai:v0.31.0';
const root = path.resolve(import.meta.dirname, '..', '..');
const out = path.join(root, 'test/fixtures/rtx5090/measurements.json');
const token = findHfToken({ env: process.env });
const hub = new HubClient({ fetch: cachedFetch(globalThis.fetch, { dir: defaultCacheDir() }), baseUrl: 'https://huggingface.co', getToken: () => token });

interface Run {
  name: string;
  repo: string;
  /** Layers kept (undefined = the whole model). */
  layers?: number;
  eager: boolean;
  maxModelLen: number;
  /** --language-model-only: no vision tower, no image profiling (much faster). */
  textOnly?: boolean;
  extra?: string[];
}

const RUNS: Run[] = [
  { name: 'nvfp4-hybrid-8-text', repo: 'nvidia/Qwen3.8-27B-NVFP4', layers: 8, eager: true, maxModelLen: 8192, textOnly: true },
  { name: 'nvfp4-hybrid-4-text', repo: 'nvidia/Qwen3.8-27B-NVFP4', layers: 4, eager: true, maxModelLen: 8192, textOnly: true },
  { name: 'fp8-hybrid-4-text', repo: 'Qwen/Qwen3.8-27B-FP8', layers: 4, eager: true, maxModelLen: 8192, textOnly: true },
  { name: 'fp8-hybrid-8-text', repo: 'Qwen/Qwen3.8-27B-FP8', layers: 8, eager: true, maxModelLen: 8192, textOnly: true },
  { name: 'mxfp4-moe-full', repo: 'openai/gpt-oss-20b', eager: true, maxModelLen: 8192 },
  { name: 'bf16-dense-graphs', repo: 'Qwen/Qwen3-8B', eager: false, maxModelLen: 16384 },
  { name: 'bf16-multimodal-text', repo: 'google/gemma-4-12B-it', eager: true, maxModelLen: 8192, textOnly: true },
];

function gpuUsedMiB(): number {
  return Number(execFileSync('nvidia-smi', ['--query-gpu=memory.used', '--format=csv,noheader,nounits'], { encoding: 'utf8' }).split(/\r?\n/)[0]);
}

const LINES = /Model loading took|Available KV cache memory|Graph capturing finished|Free memory on device|GPU KV cache size|KV cache is needed|Error|error:|Traceback/;

/** Starts vLLM, collects the memory lines until it is ready (or fails), then removes the container. */
function runVllm(r: Run, overrides: unknown): Promise<{ lines: string[]; seconds: number; ok: boolean }> {
  const name = `vramcal-${r.name}-${process.pid}`;
  const args = [
    'run',
    '--rm',
    '--name',
    name,
    '--gpus',
    'all',
    '--ipc=host',
    '-v',
    'llm-capacity-planner-hf-cache:/root/.cache/huggingface',
    '-v',
    'llm-capacity-planner-vllm-cache:/root/.cache/vllm',
    ...(token ? ['-e', 'HF_TOKEN'] : []),
    IMAGE,
    r.repo,
    '--load-format',
    'dummy',
    '--max-model-len',
    String(r.maxModelLen),
    '--gpu-memory-utilization',
    '0.9',
    '--max-num-seqs',
    '16',
    ...(r.eager ? ['--enforce-eager'] : []),
    ...(r.textOnly ? ['--language-model-only'] : []),
    ...(overrides ? ['--hf-overrides', JSON.stringify(overrides)] : []),
    ...(r.extra ?? []),
  ];
  const started = Date.now();
  return new Promise((resolve) => {
    const p = spawn('docker', args, { env: { ...process.env, ...(token ? { HF_TOKEN: token } : {}) } });
    const lines: string[] = [];
    let done = false;
    const finish = (ok: boolean) => {
      if (done) return;
      done = true;
      try {
        execFileSync('docker', ['rm', '-f', name], { stdio: 'ignore' });
      } catch {
        // Already gone.
      }
      resolve({ lines, seconds: Math.round((Date.now() - started) / 1000), ok });
    };
    const timer = setTimeout(() => finish(false), 8 * 60 * 1000);
    const onData = (b: Buffer) => {
      for (const l of b.toString('utf8').split(/\r?\n/)) {
        if (LINES.test(l)) lines.push(l.replace(/^.*?(INFO|ERROR|WARNING)/, '$1').slice(0, 400));
        // The memory summary is the last line needed: no reason to wait for the API server.
        if (/Free memory on device|Application startup complete|Starting vLLM API server/.test(l)) {
          clearTimeout(timer);
          finish(true);
        }
        if (/Engine core initialization failed|RuntimeError|ValueError/.test(l)) {
          clearTimeout(timer);
          setTimeout(() => finish(false), 2000);
        }
      }
    };
    p.stdout.on('data', onData);
    p.stderr.on('data', onData);
    p.on('exit', () => {
      clearTimeout(timer);
      finish(false);
    });
  });
}

const only = process.argv.slice(2).find((a) => a.startsWith('--only='))?.slice(7);
const results = existsSync(out) ? (JSON.parse(readFileSync(out, 'utf8')) as { runs: Record<string, unknown>[] }).runs : [];
for (const r of RUNS) {
  if (only && !r.name.includes(only)) continue;
  const busy = gpuUsedMiB();
  if (busy > 4096) {
    console.log(`GPU busy (${busy} MiB in use): stopping before ${r.name}.`);
    break;
  }
  const files = await hub.files(r.repo);
  const variant = repoLayout(files, { name: r.repo }).safetensors[0];
  if (!variant) continue;
  const remote = await readRemoteModel(hub, r.repo, 'main', variant);
  const { overrides, shape, tensors, planner: model } = cutModel(remote, r.layers);
  const w = weightsPerGpu(tensors, shape, { tp: 1, textOnly: r.textOnly === true });
  console.log(`\n== ${r.name}: ${r.repo}${r.layers ? ` (${r.layers} layer)` : ''}, ${r.eager ? 'eager' : 'con CUDA graph'}`);
  const res = await runVllm(r, overrides);
  const loading = res.lines.map((l) => /Model loading took ([\d.]+) GiB/.exec(l)?.[1]).find(Boolean);
  const calc = w.perGpuBytes / 1024 ** 3;
  console.log(`   ${res.seconds} s, ${res.ok ? 'ready' : 'did NOT start'}; Model loading took ${loading ?? '-'} GiB, calculated ${calc.toFixed(3)} GiB (extra MLA ${(w.mlaAbsorbedBytes.max / 1024 ** 3).toFixed(3)})`);
  for (const l of res.lines) console.log(`   ${l}`);
  // A run measured again replaces the earlier one.
  const at = results.findIndex((x) => x.name === r.name);
  if (at >= 0) results.splice(at, 1);
  results.push({
    name: r.name,
    repo: r.repo,
    layers: r.layers ?? null,
    eager: r.eager,
    textOnly: r.textOnly === true,
    maxModelLen: r.maxModelLen,
    date: new Date().toISOString().slice(0, 10),
    seconds: res.seconds,
    ok: res.ok,
    format: model.parallel.weights,
    family: model.family,
    calculatedGiB: Math.round(calc * 1000) / 1000,
    lines: res.lines,
  });
  writeFileSync(out, `${JSON.stringify({ $comment: `vLLM ${IMAGE} on the GPU of the machine that ran it, dummy weights; written by scripts/dev/measure.mjs`, runs: results }, null, 1)}\n`);
}

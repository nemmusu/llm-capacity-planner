// Short vLLM speed measurements on this PC's GPU, to calibrate the planner's speed model (development only;
// run again when vLLM's version changes):   npm run dev:speed -- [--only=<name>]
// Each run starts vLLM (Docker image vllm/vllm-openai:v0.31.0) with dummy weights (--load-format dummy: no
// weight download), the model optionally cut to its first layers (--hf-overrides), and the fp8 KV cache;
// then it measures over the OpenAI API, with random token ids as prompts:
//  - TTFT of one request alone at 4K, 16K and 32K prompt tokens (median of 3, every prompt new);
//  - generation speed per user with 1, 8 and 32 requests together at 1K and 16K tokens of context: the
//    prompts are read once beforehand (prefix cache), then sent together so they all generate at once;
//    the speed is the tokens of the window where every request generates ÷ its length.
// Combinations whose KV does not fit the cache are skipped. The other programs' CPU is checked before and
// during each run (cpu-guard.ts): a busy PC stops the script, a disturbed run is measured again once and
// otherwise kept marked "disturbed". Each run is added to test/fixtures/rtx5090/speed.json;
// scripts/dev/calibrate.mjs turns them into calibration points.
import { execFileSync, spawn } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { repoLayout } from '../../src/hub/formats';
import { cachedFetch, defaultCacheDir } from '../../src/cli/http-cache';
import { findHfToken } from '../../src/cli/hf-token';
import { readRemoteModel } from '../../src/cli/model';
import { HubClient } from '../../src/hub/client';
import { loadBetween, sharedGpuMiB, snapshot, SPILL_MIB, type Load } from './cpu-guard';
import { cutModel } from './cut';

const IMAGE = 'vllm/vllm-openai:v0.31.0';
const PORT = 8011;
const root = path.resolve(import.meta.dirname, '..', '..');
// --long: long contexts (64K–1M tokens) on models cut to a few layers, whose KV cache is small enough.
const LONG = process.argv.includes('--long');
const out = path.join(root, LONG ? 'test/fixtures/rtx5090/speed-long.json' : 'test/fixtures/rtx5090/speed.json');
const token = findHfToken({ env: process.env });
const hub = new HubClient({ fetch: cachedFetch(globalThis.fetch, { dir: defaultCacheDir() }), baseUrl: 'https://huggingface.co', getToken: () => token });

interface Run {
  name: string;
  repo: string;
  /** Layers kept (undefined = the whole model). */
  layers?: number;
  /** More config.json overrides (merged into the text config). */
  overrides?: Record<string, unknown>;
  /** Pass the overrides at the top level: for some models vLLM replaces a given text_config instead of merging it. */
  flat?: boolean;
  /** More vLLM arguments. */
  args?: string[];
  /** --max-model-len (default MAX_LEN). */
  maxLen?: number;
  /** Long-context lengths to measure (default LONG_TOKENS). */
  longTokens?: number[];
}

const RUNS: Run[] = [
  { name: 'qwen3-8b-bf16', repo: 'Qwen/Qwen3-8B' },
  { name: 'qwen3-30b-a3b-fp8-24l', repo: 'Qwen/Qwen3-30B-A3B-FP8', layers: 24 },
  { name: 'qwen3-235b-a22b-fp8-6l', repo: 'Qwen/Qwen3-235B-A22B-FP8', layers: 6 },
  // MLA / sparse attention (DSA): GLM-5.3-Flash (3 linear-attention layers per DSA layer) and DeepSeek-V4.1-Flash.
  { name: 'glm-5.3-flash-4l', repo: 'zai-org/GLM-5.3-Flash', layers: 4 },
  // Without its Engram layers: vLLM keeps their tables in pinned host memory through UVA, which WSL 2 lacks;
  // the Engram lookup is not attention and is not measured.
  // 3 layers (2 uncompressed, 1 with compressed KV): 4 already take 30.7 GiB. vLLM's SM120 sparse-MLA decode
  // kernels exist only for KV blocks of 64 tokens.
  // NVFP4 (Gated DeltaNet hybrid) cut to 16 layers (4 full attention).
  { name: 'qwen3.8-27b-nvfp4-16l', repo: 'nvidia/Qwen3.8-27B-NVFP4', layers: 16 },
  { name: 'deepseek-v4.1-flash-3l', repo: 'deepseek-ai/DeepSeek-V4.1-Flash', layers: 3, overrides: { engram_layer_ids: [] }, flat: true, args: ['--block-size', '128'] },
];
/** Long-context runs: dense GQA, MoE GQA, sparse MLA (DSA) with linear attention, Gated DeltaNet hybrid. */
const LONG_RUNS: Run[] = [
  // max_position_embeddings sizes vLLM's RoPE cache: raised so positions up to 1M exist.
  { name: 'long-qwen3-8b-bf16-2l', repo: 'Qwen/Qwen3-8B', layers: 2, maxLen: 1_049_600, overrides: { max_position_embeddings: 1_049_600 } },
  { name: 'long-qwen3-235b-a22b-fp8-2l', repo: 'Qwen/Qwen3-235B-A22B-FP8', layers: 2, maxLen: 1_049_600, overrides: { max_position_embeddings: 1_049_600 } },
  { name: 'long-glm-5.3-flash-4l', repo: 'zai-org/GLM-5.3-Flash', layers: 4, maxLen: 1_049_600 },
  { name: 'long-qwen3.8-27b-fp8-4l', repo: 'Qwen/Qwen3.8-27B-FP8', layers: 4, maxLen: 1_049_600, overrides: { max_position_embeddings: 1_049_600 } },
  // NVFP4 (Gated DeltaNet hybrid) to 128K, and GLM-5.3 at 262K again in a fresh process (it slowed 40× once).
  { name: 'long-qwen3.8-27b-nvfp4-16l', repo: 'nvidia/Qwen3.8-27B-NVFP4', layers: 16, maxLen: 140_000, longTokens: [65_536, 131_072] },
  { name: 'long-glm-5.3-flash-4l-262k', repo: 'zai-org/GLM-5.3-Flash', layers: 4, maxLen: 270_000, longTokens: [262_144] },
];
const LONG_TOKENS = [65_536, 131_072, 262_144, 524_288, 1_048_576];
/** A length is skipped when the previous one's prompt reading, scaled by the square, would pass this. */
const LONG_MAX_TTFT_MS = 180_000;
const TTFT_TOKENS = [4096, 16384, 32768];
const DECODE = { contexts: [1024, 16384], batches: [1, 8, 32], outputTokens: 512 };
const MAX_LEN = 34816;

function gpuUsedMiB(): number {
  return Number(execFileSync('nvidia-smi', ['--query-gpu=memory.used', '--format=csv,noheader,nounits'], { encoding: 'utf8' }).split(/\r?\n/)[0]);
}

function gpuTotalMiB(): number {
  return Number(execFileSync('nvidia-smi', ['--query-gpu=memory.total', '--format=csv,noheader,nounits'], { encoding: 'utf8' }).split(/\r?\n/)[0]);
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const median = (xs: number[]) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] as number;
const randomPrompt = (n: number) => Array.from({ length: n }, () => 1000 + Math.floor(Math.random() * 99000));

/** One streamed completion: the arrival time of every generated token (ms, performance.now()). */
async function complete(model: string, prompt: number[], maxTokens: number): Promise<{ sent: number; tokens: number[] }> {
  const sent = performance.now();
  const res = await fetch(`http://127.0.0.1:${PORT}/v1/completions`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, prompt, max_tokens: maxTokens, ignore_eos: true, temperature: 0, stream: true }),
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const tokens: number[] = [];
  const decoder = new TextDecoder();
  let buf = '';
  for await (const chunk of res.body as unknown as AsyncIterable<Uint8Array>) {
    buf += decoder.decode(chunk, { stream: true });
    let i: number;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const ev = buf.slice(0, i);
      buf = buf.slice(i + 2);
      if (ev.startsWith('data: ') && !ev.includes('[DONE]') && ev.includes('"choices":[{')) tokens.push(performance.now());
    }
  }
  return { sent, tokens };
}

/** vLLM's inter-token latency histogram (sum in seconds, count). */
async function itlMetrics(): Promise<{ sum: number; count: number }> {
  const t = await (await fetch(`http://127.0.0.1:${PORT}/metrics`)).text();
  const total = (re: RegExp) => [...t.matchAll(re)].reduce((a, m) => a + Number(m[1]), 0);
  return { sum: total(/^vllm:inter_token_latency_seconds_sum\{[^}]*\} (\S+)/gm), count: total(/^vllm:inter_token_latency_seconds_count\{[^}]*\} (\S+)/gm) };
}

async function measure(model: string, kvTokens: number) {
  await complete(model, randomPrompt(256), 16);
  await complete(model, randomPrompt(2048), 16);
  const ttft: { promptTokens: number; ttftMs: number[]; medianMs: number }[] = [];
  for (const n of TTFT_TOKENS) {
    const ms: number[] = [];
    for (let k = 0; k < 3; k++) {
      const r = await complete(model, randomPrompt(n), 1);
      ms.push(Math.round(((r.tokens[0] as number) - r.sent) * 10) / 10);
    }
    ttft.push({ promptTokens: n, ttftMs: ms, medianMs: median(ms) });
    console.log(`   TTFT ${n} tokens: ${ms.join(' / ')} ms`);
  }
  const decode: { inputLen: number; outputLen: number; batch: number; tokSPerUser: number; windowMs: number; serverItlMs: number; tries: number[] }[] = [];
  for (const ctx of DECODE.contexts)
    for (const batch of DECODE.batches) {
      // Prompts and generated tokens of every request must stay in the cache (with a margin for its blocks).
      if (batch * (ctx + DECODE.outputTokens) > kvTokens * 0.9) {
        console.log(`   generation ${batch} × ${ctx}: does not fit the KV cache (${kvTokens} tokens), skipped`);
        continue;
      }
      const prompts = Array.from({ length: batch }, () => randomPrompt(ctx));
      // Read once (one at a time keeps it short), so the measured run starts generating together.
      for (const p of prompts) await complete(model, p, 1);
      // Twice: the first run of a batch size can include one-off warm-up; the faster one is kept.
      const tries: { tokS: number; windowMs: number; serverItlMs: number }[] = [];
      for (let k = 0; k < 2; k++) {
        const m0 = await itlMetrics();
        const runs = await Promise.all(prompts.map((p) => complete(model, p, DECODE.outputTokens)));
        const start = Math.max(...runs.map((r) => r.tokens[0] as number));
        const end = Math.min(...runs.map((r) => r.tokens[r.tokens.length - 1] as number));
        const inWindow = runs.reduce((a, r) => a + r.tokens.filter((t) => t > start && t <= end).length, 0);
        const m1 = await itlMetrics();
        // vLLM's own time between tokens (includes the steps where the prompts are still being read).
        const serverItlMs = Math.round(((m1.sum - m0.sum) / Math.max(1, m1.count - m0.count)) * 1e5) / 100;
        tries.push({ tokS: end > start ? (inWindow / batch / (end - start)) * 1000 : NaN, windowMs: Math.round(end - start), serverItlMs });
      }
      const best = tries.reduce((a, b) => (b.tokS > a.tokS ? b : a));
      decode.push({ inputLen: ctx, outputLen: DECODE.outputTokens, batch, tokSPerUser: Math.round(best.tokS * 100) / 100, windowMs: best.windowMs, serverItlMs: best.serverItlMs, tries: tries.map((t) => Math.round(t.tokS * 10) / 10) });
      console.log(`   generation ${batch} × ${ctx}: ${tries.map((t) => t.tokS.toFixed(1)).join(' / ')} tok/s per user (window ${best.windowMs} ms; server ITL ${best.serverItlMs} ms)`);
    }
  return { ttft, decode };
}

/**
 * Long contexts: for each length, the TTFT of a new prompt (once: it takes seconds to minutes), then
 * generation with that full context for 1 and 4 requests, the prompt read from the prefix cache.
 */
async function measureLong(model: string, kvTokens: number, maxLen: number, lengths = LONG_TOKENS) {
  await complete(model, randomPrompt(256), 16);
  const points: { promptTokens: number; ttftMs: number; decode: { batch: number; tokSPerUser: number; serverItlMs: number }[] }[] = [];
  let lastTtft = 0;
  let lastLen = 0;
  for (const n of lengths) {
    if (n + 128 > maxLen || n > kvTokens * 0.9) {
      console.log(`   ${n} tokens: beyond the KV cache or the maximum context, skipped`);
      continue;
    }
    if (lastTtft > 0 && lastTtft * (n / lastLen) ** 2 > LONG_MAX_TTFT_MS) {
      console.log(`   ${n} tokens: reading would exceed ${LONG_MAX_TTFT_MS / 1000} s, skipped`);
      break;
    }
    const prompt = randomPrompt(n);
    const first = await complete(model, prompt, 1);
    const ttftMs = Math.round((first.tokens[0] as number) - first.sent);
    lastTtft = ttftMs;
    lastLen = n;
    const decode: { batch: number; tokSPerUser: number; serverItlMs: number }[] = [];
    for (const batch of [1, 4]) {
      const m0 = await itlMetrics();
      const runs = await Promise.all(Array.from({ length: batch }, () => complete(model, prompt, 64)));
      const m1 = await itlMetrics();
      const start = Math.max(...runs.map((x) => x.tokens[0] as number));
      const end = Math.min(...runs.map((x) => x.tokens[x.tokens.length - 1] as number));
      const inWindow = runs.reduce((a, x) => a + x.tokens.filter((t) => t > start && t <= end).length, 0);
      const tokSPerUser = end > start ? Math.round((inWindow / batch / (end - start)) * 100_000) / 100 : NaN;
      const serverItlMs = Math.round(((m1.sum - m0.sum) / Math.max(1, m1.count - m0.count)) * 1e5) / 100;
      decode.push({ batch, tokSPerUser, serverItlMs });
    }
    points.push({ promptTokens: n, ttftMs, decode });
    console.log(`   ${n} tokens: reading ${(ttftMs / 1000).toFixed(1)} s; generation ${decode.map((d) => `${d.batch}× ${d.tokSPerUser} tok/s (ITL ${d.serverItlMs} ms)`).join(', ')}`);
  }
  return { long: points };
}

/** TTFT of a fresh 4K prompt (median of 3): measured at the start and end of a run to catch degradation. */
async function probeTtft(model: string): Promise<number> {
  const ms: number[] = [];
  for (let k = 0; k < 3; k++) {
    const r = await complete(model, randomPrompt(4096), 1);
    ms.push(Math.round((r.tokens[0] as number) - r.sent));
  }
  return median(ms);
}

/** Starts vLLM in the background; resolves with its KV cache size once the API answers. */
async function startVllm(r: Run, overrides: unknown, name: string): Promise<{ kvTokens: number; log: string[] }> {
  const args = [
    'run',
    '--rm',
    '--name',
    name,
    '--gpus',
    'all',
    '--ipc=host',
    '-p',
    `${PORT}:8000`,
    '-v',
    'llm-capacity-planner-hf-cache:/root/.cache/huggingface',
    '-v',
    'llm-capacity-planner-vllm-cache:/root/.cache/vllm',
    ...(token ? ['-e', 'HF_TOKEN'] : []),
    // Beyond the model's max_position_embeddings (dummy weights: the speed does not depend on RoPE's range).
    '-e',
    'VLLM_ALLOW_LONG_MAX_MODEL_LEN=1',
    IMAGE,
    r.repo,
    '--load-format',
    'dummy',
    '--max-model-len',
    String(r.maxLen ?? MAX_LEN),
    '--kv-cache-dtype',
    'fp8',
    // 0.85, not 0.9: the desktop's own programs grow during a run, and a full card makes Windows move
    // memory to the system RAM (a run slowed tenfold).
    '--gpu-memory-utilization',
    '0.85',
    '--max-num-seqs',
    '32',
    '--language-model-only',
    ...(overrides ? ['--hf-overrides', JSON.stringify(overrides)] : []),
    ...(r.args ?? []),
  ];
  const p = spawn('docker', args, { env: { ...process.env, ...(token ? { HF_TOKEN: token } : {}) } });
  const log: string[] = [];
  let kvTokens = 0;
  let exited = false;
  const onData = (b: Buffer) => {
    for (const l of b.toString('utf8').split(/\r?\n/)) {
      const m = /GPU KV cache size: ([\d,]+) tokens/.exec(l);
      if (m) kvTokens = Number((m[1] as string).replace(/,/g, ''));
      if (/GPU KV cache size|Maximum concurrency|Model loading took|Graph capturing finished|Error|Traceback/.test(l)) log.push(l.replace(/^.*?(INFO|ERROR|WARNING)/, '$1').slice(0, 300));
    }
  };
  p.stdout.on('data', onData);
  p.stderr.on('data', onData);
  p.on('exit', () => (exited = true));
  const deadline = Date.now() + 12 * 60 * 1000;
  while (Date.now() < deadline && !exited) {
    try {
      if ((await fetch(`http://127.0.0.1:${PORT}/health`)).ok && kvTokens > 0) return { kvTokens, log };
    } catch {
      // Not listening yet.
    }
    await sleep(3000);
  }
  throw new Error(`vLLM did not start:\n${log.join('\n')}`);
}

const only = process.argv.slice(2).find((a) => a.startsWith('--only='))?.slice(7);
const saved = existsSync(out) ? (JSON.parse(readFileSync(out, 'utf8')) as { $comment?: string; runs: Record<string, unknown>[] }) : undefined;
const results = saved?.runs ?? [];
const COMMENT =
  saved?.$comment ??
  `vLLM ${IMAGE} on an RTX 5090 (Windows 11, Docker Desktop/WSL 2), dummy weights, fp8 KV cache, random token prompts; written by scripts/dev/speed.mjs. Runs marked "disturbed" (other programs using the CPU, which vLLM's scheduler needs) are kept with the reason but left out of the calibration.`;
for (const r of LONG ? LONG_RUNS : RUNS) {
  if (only && !r.name.includes(only)) continue;
  const busy = gpuUsedMiB();
  if (busy > 4096) {
    console.log(`GPU busy (${busy} MiB in use): stopping before ${r.name}.`);
    break;
  }
  // The PC must be quiet before the GPU is used at all.
  const quietFrom = snapshot();
  await sleep(10_000);
  const before = loadBetween(quietFrom, snapshot(), '');
  if (before.disturbed) {
    console.log(`PC not free, stopping before ${r.name}: ${before.disturbed} (${before.top.join(', ')}).`);
    break;
  }
  const files = await hub.files(r.repo);
  const variant = repoLayout(files, { name: r.repo }).safetensors[0];
  if (!variant) continue;
  const cut = cutModel(await readRemoteModel(hub, r.repo, 'main', variant), r.layers).overrides;
  const inner = cut && 'text_config' in cut ? (cut.text_config as Record<string, unknown>) : cut;
  const merged = { ...inner, ...r.overrides };
  const overrides = r.flat ? merged : r.overrides ? (cut && 'text_config' in cut ? { text_config: merged } : merged) : cut;
  console.log(`\n== ${r.name}: ${r.repo}${r.layers ? ` (${r.layers} layer)` : ''}`);
  const name = `vramspeed-${r.name}-${process.pid}`;
  const started = Date.now();
  let entry: Record<string, unknown>;
  try {
    const { kvTokens, log } = await startVllm(r, overrides, name);
    const startupS = Math.round((Date.now() - started) / 1000);
    console.log(`   ready in ${startupS} s, KV cache ${kvTokens} tokens`);
    // Measured again once if other programs used the CPU meanwhile; still disturbed: kept, marked.
    let m: Awaited<ReturnType<typeof measure>> | Awaited<ReturnType<typeof measureLong>> | undefined;
    let load: Load | undefined;
    let checks: { sharedGpuGrowthMiB: number; probeTtftMs: number[] } | undefined;
    for (let attempt = 0; attempt < 2 && (!load || load.disturbed); attempt++) {
      const from = snapshot();
      const sharedBefore = sharedGpuMiB();
      const probeBefore = await probeTtft(r.repo);
      m = LONG ? await measureLong(r.repo, kvTokens, r.maxLen ?? MAX_LEN, r.longTokens) : await measure(r.repo, kvTokens);
      const probeAfter = await probeTtft(r.repo);
      const spilled = sharedGpuMiB() - sharedBefore;
      load = loadBetween(from, snapshot(), name);
      // Memory spilled to system RAM, or the server slower at the end than at the start: the run is not the card's.
      const extra = [
        spilled > SPILL_MIB ? `shared GPU memory grew by ${spilled} MiB (spill to system RAM)` : '',
        probeAfter > probeBefore * 1.3 ? `slowed during the run: 4K TTFT from ${probeBefore} to ${probeAfter} ms` : '',
      ].filter(Boolean);
      if (extra.length) load = { ...load, disturbed: [load.disturbed, ...extra].filter(Boolean).join('; ') };
      checks = { sharedGpuGrowthMiB: spilled, probeTtftMs: [probeBefore, probeAfter] };
      if (load.disturbed) console.log(`   disturbed run: ${load.disturbed} (${load.top.join(', ')})${attempt === 0 ? '; repeating it' : ''}`);
    }
    const usedAfter = gpuUsedMiB();
    // A card filled to the last 400 MiB may have spilled to system memory: the numbers are not kept.
    if (usedAfter > gpuTotalMiB() - 400) throw new Error(`GPU full during the measurement (${usedAfter} MiB in use): results discarded`);
    // Every run is kept (one name per start), the disturbed ones with the reason.
    entry = {
      name: `${r.name}-${new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '')}`,
      repo: r.repo,
      layers: r.layers ?? null,
      kvCacheDtype: 'fp8',
      maxModelLen: r.maxLen ?? MAX_LEN,
      maxNumSeqs: 32,
      date: new Date().toISOString().slice(0, 10),
      kvTokens,
      startupS,
      seconds: Math.round((Date.now() - started) / 1000),
      ...(load?.disturbed ? { disturbed: load.disturbed } : {}),
      cpuLoad: { before, during: load },
      checks,
      log,
      ...m,
    };
  } finally {
    try {
      execFileSync('docker', ['rm', '-f', name], { stdio: 'ignore' });
    } catch {
      // Already gone.
    }
  }
  console.log(`   ${entry.seconds as number} s in tutto${entry.disturbed ? ' (DISTURBATA: esclusa dalla taratura)' : ''}`);
  results.push(entry);
  writeFileSync(out, `${JSON.stringify({ $comment: COMMENT, engine: 'vllm', engineVersion: '0.31.0', gpuId: 'rtx-5090-32gb', runs: results }, null, 1)}\n`);
}

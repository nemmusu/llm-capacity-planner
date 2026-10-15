// Development check of the VRAM planner against measurements (run with: npm run dev:calibrate --).
// For every reference of data/references.json and an RTX 5090 runs it reads the model
// from Hugging Face (headers only, through the disk cache), computes weights per GPU, decode speed and
// prompt reading time, and prints them next to what was measured. Writes .qa/vram-calibration.json.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';
import { repoLayout } from '../../src/hub/formats';
import { hardwareFileSchema, referencesFileSchema, type Reference } from '../../src/contract/data';
import { cachedFetch, defaultCacheDir } from '../../src/cli/http-cache';
import { readRemoteModel } from '../../src/cli/model';
import { HubClient } from '../../src/hub/client';
import { kvElementBytes } from '../../src/model/kv';
import { weightsPerGpu } from '../../src/model/weights-exact';
import { calibrationFor } from '../../src/planner/calibration';
import { gpuFromSpec, type Gpu, type GpuLayout } from '../../src/planner/hardware';
import { plannerModel } from '../../src/planner/model';
import { kvReadBytes } from '../../src/planner/model-profile';
import { decodeStepMs, prefillMs } from '../../src/planner/perf';
import { loadPoint } from '../../src/planner/capacity';
import { schedulerDefaults } from '../../src/model/serving-fit';
import { cutModel } from './cut';

const root = path.resolve(import.meta.dirname, '..', '..');
const hw = hardwareFileSchema.parse(JSON.parse(readFileSync(path.join(root, 'data/hardware.json'), 'utf8')));
const refs = referencesFileSchema.parse(JSON.parse(readFileSync(path.join(root, 'data/references.json'), 'utf8')));
const gpus = new Map<string, Gpu>(hw.gpus.map((g) => [g.id, gpuFromSpec(g)]));
const token = process.env.HF_TOKEN;
const hub = new HubClient({ fetch: cachedFetch(globalThis.fetch, { dir: defaultCacheDir() }), baseUrl: 'https://huggingface.co', getToken: () => token });
const only = process.argv.slice(2).find((a) => a.startsWith('--only='))?.slice(7);

const models = new Map<string, Promise<Awaited<ReturnType<typeof load>> | undefined>>();
/** `repo` or `repo@<n>L` (the model cut to its first n layers, as the speed runs start it). */
async function load(key: string) {
  const [repo, cut] = key.split('@') as [string, string | undefined];
  const files = await hub.files(repo);
  const variant = repoLayout(files, { name: repo }).safetensors[0];
  if (!variant) throw new Error('no safetensors');
  const m = await readRemoteModel(hub, repo, 'main', variant);
  if (cut) return { remote: m, planner: cutModel(m, Number.parseInt(cut, 10)).planner };
  return { remote: m, planner: plannerModel(m.config, m.facts, m.shape, m.tensors, repo) };
}
const model = (repo: string) => {
  if (!models.has(repo))
    models.set(
      repo,
      load(repo).catch((e: unknown) => {
        console.log(`  ! ${repo}: ${(e as Error).message}`);
        return undefined;
      }),
    );
  return models.get(repo) as Promise<Awaited<ReturnType<typeof load>> | undefined>;
};

const f = (n: number, d = 2) => n.toFixed(d);
const profiles = new Map<string, unknown>();
const out: Record<string, unknown>[] = [];

interface PerfPoint {
  id: string;
  engine: 'vllm' | 'sglang';
  repo: string;
  gpu: string;
  tp: number;
  pp: number;
  concurrency: number;
  inputLen: number;
  outputLen: number;
  kv: 'fp8' | 'auto';
  decodeTokSPerUser?: number;
  ttftMs?: number;
  /** decode: pure generation speed (TPOT at one request, or a decode speed reported as such); throughput: output ÷ wall time per user (prompts included). */
  /**
   * decode: pure generation speed (one request, or a decode speed the source reports as such); tpot: the
   * time per output token at that concurrency, which includes the other requests' prompts read between
   * steps; throughput: output ÷ wall time per user.
   */
  basis: 'decode' | 'tpot' | 'throughput';
}
const points: PerfPoint[] = [];
const weightsRows: { ref: Reference; reported: number }[] = [];
for (const r of refs.entries) {
  if (only && !r.id.includes(only) && !r.model.hfId.includes(only)) continue;
  if (r.log?.modelLoadingGiB != null) weightsRows.push({ ref: r, reported: r.log.modelLoadingGiB });
  // Speculative decoding (EAGLE, NextN / MTP drafts) is outside the speed model.
  if (/eagle|nextn|mtp|spec/i.test(`${r.id} ${r.params.other.join(' ')}`)) continue;
  for (const p of r.perf) {
    if (!p.concurrency || !p.inputLen || !p.outputLen) continue;
    const tpot = p.tpotMsMedian ?? p.tpotMsMean;
    const pure = p.decodeTokPerSPerUser ?? (tpot && p.concurrency === 1 ? 1000 / tpot : undefined);
    const decode = pure ?? (tpot ? 1000 / tpot : p.outputTokPerS ? p.outputTokPerS / p.concurrency : undefined);
    const ttft = p.ttftMsMedian ?? p.ttftMsMean;
    points.push({
      id: r.id,
      engine: r.engine,
      repo: r.model.hfId,
      gpu: r.hardware.gpuId,
      tp: r.hardware.tp,
      pp: r.hardware.pp,
      concurrency: p.concurrency,
      inputLen: p.inputLen,
      outputLen: p.outputLen,
      kv: /fp8/.test(r.params.kvCacheDtype) ? 'fp8' : 'auto',
      ...(decode ? { decodeTokSPerUser: decode } : {}),
      // With more than 8 requests the benchmarks send every prompt at once: the first wait is a start-up
      // queue the steady-state model does not describe, so it is not compared.
      ...(ttft && p.concurrency <= 8 ? { ttftMs: ttft } : {}),
      basis: pure !== undefined ? 'decode' : tpot ? 'tpot' : 'throughput',
    });
  }
}
// The vLLM speed runs on an RTX 5090 (scripts/dev/speed.mjs).
const speedRuns = JSON.parse(readFileSync(path.join(root, 'test/fixtures/rtx5090/speed.json'), 'utf8')) as {
  runs: { name: string; repo: string; layers: number | null; disturbed?: string; ttft: { promptTokens: number; medianMs: number }[]; decode: { inputLen: number; outputLen: number; batch: number; tokSPerUser: number }[] }[];
};
for (const run of speedRuns.runs) {
  const repo = run.layers ? `${run.repo}@${run.layers}L` : run.repo;
  if (only && !repo.includes(only) && !run.name.includes(only)) continue;
  // A run disturbed by other programs using the CPU measures the PC, not the card: left out.
  if (run.disturbed) {
    console.log(`  - ${run.name}: left out (${run.disturbed})`);
    continue;
  }
  for (const t of run.ttft)
    points.push({ id: `5090-vllm-${run.name}-ttft-${t.promptTokens}`, engine: 'vllm', repo, gpu: 'rtx-5090-32gb', tp: 1, pp: 1, concurrency: 1, inputLen: t.promptTokens, outputLen: 1, kv: 'fp8', ttftMs: t.medianMs, basis: 'decode' });
  for (const d of run.decode)
    points.push({ id: `5090-vllm-${run.name}-dec-${d.batch}x${d.inputLen}`, engine: 'vllm', repo, gpu: 'rtx-5090-32gb', tp: 1, pp: 1, concurrency: d.batch, inputLen: d.inputLen, outputLen: d.outputLen, kv: 'fp8', decodeTokSPerUser: d.tokSPerUser, basis: 'decode' });
}
// Long contexts (64K–1M tokens) on models cut to a few layers (scripts/dev/speed.mjs --long).
const longRuns = JSON.parse(readFileSync(path.join(root, 'test/fixtures/rtx5090/speed-long.json'), 'utf8')) as {
  runs: { name: string; repo: string; layers: number | null; disturbed?: string; anomalous?: Record<string, string>; long: { promptTokens: number; ttftMs: number; decode: { batch: number; serverItlMs: number }[] }[] }[];
};
for (const run of longRuns.runs) {
  const repo = run.layers ? `${run.repo}@${run.layers}L` : run.repo;
  if (only && !repo.includes(only) && !run.name.includes(only)) continue;
  if (run.disturbed) {
    console.log(`  - ${run.name}: left out (${run.disturbed})`);
    continue;
  }
  for (const l of run.long) {
    const why = run.anomalous?.[String(l.promptTokens)];
    if (why) {
      console.log(`  - ${run.name} at ${l.promptTokens} tokens: left out (${why})`);
      continue;
    }
    points.push({ id: `5090-vllm-${run.name}-ttft-${l.promptTokens}`, engine: 'vllm', repo, gpu: 'rtx-5090-32gb', tp: 1, pp: 1, concurrency: 1, inputLen: l.promptTokens, outputLen: 1, kv: 'fp8', ttftMs: l.ttftMs, basis: 'decode' });
    // vLLM's inter-token latency at that full context, one request. With 4 the requests shared the prompt and
    // vLLM's cascade attention read their common KV once: not 4 users with their own contexts, left out.
    for (const d of l.decode.filter((x) => x.batch === 1))
      points.push({ id: `5090-vllm-${run.name}-dec-${d.batch}x${l.promptTokens}`, engine: 'vllm', repo, gpu: 'rtx-5090-32gb', tp: 1, pp: 1, concurrency: d.batch, inputLen: l.promptTokens, outputLen: 64, kv: 'fp8', decodeTokSPerUser: Math.round((1000 / d.serverItlMs) * 100) / 100, basis: 'decode' });
  }
}

console.log('\n== Weights per GPU: reported by the engine vs calculated (GiB)');
for (const { ref, reported } of weightsRows) {
  const m = await model(ref.model.hfId);
  if (!m) continue;
  const w = weightsPerGpu(m.remote.tensors, m.remote.shape, { tp: ref.hardware.tp, pp: ref.hardware.pp });
  const calc = w.perGpuBytes / 1024 ** 3;
  console.log(`${ref.id.padEnd(58)} ${m.planner.parallel.weights.padEnd(6)} ${m.planner.family.padEnd(10)} tp${ref.hardware.tp} pp${ref.hardware.pp}  reported ${f(reported).padStart(7)}  calculated ${f(calc).padStart(7)}  diff ${f(reported - calc).padStart(6)} (${f(((reported - calc) / calc) * 100, 1)}%)`);
  out.push({ kind: 'weights', id: ref.id, format: m.planner.parallel.weights, family: m.planner.family, tp: ref.hardware.tp, pp: ref.hardware.pp, reported, calculated: calc });
}

console.log('\n== Speed: measured vs predicted (min–max)');
for (const p of points) {
  const m = await model(p.repo);
  const gpu = gpus.get(p.gpu);
  if (!m || !gpu) continue;
  const layout: GpuLayout = { gpu, count: p.tp * p.pp, tp: p.tp, pp: p.pp };
  const cal = calibrationFor(p.engine, gpu, m.planner.parallel.weights, m.planner.family);
  const elem = kvElementBytes(p.engine, p.kv);
  const ctx = p.inputLen + p.outputLen / 2;
  const kvRead = kvReadBytes(m.planner.layout, elem, p.tp, ctx, m.planner.sparseTopk);
  const batched = schedulerDefaults(p.engine, gpu.cudaMiB.max, p.tp).batchedTokens;
  const speed = {
    stepMs: (b: number) => decodeStepMs({ profile: m.planner.profile, layout, batch: b, kvBytesPerSeq: kvRead, weights: m.planner.parallel.weights, cal }),
    prefillMs: (t: number) => prefillMs({ profile: m.planner.profile, layout, tokens: t, batchedTokens: batched, weights: m.planner.parallel.weights, cal }),
    batchedTokens: batched,
    requestOverheadMs: cal.requestOverheadMs,
  };
  // The benchmarks send a new request as soon as one ends: generation shares the GPU with prompt reading.
  const lp = loadPoint(speed, { maxContext: p.inputLen + p.outputLen, typicalContext: p.inputLen, outputTokens: p.outputLen, minTokS: 0, maxTtftMs: 0, activeShare: 1, safety: 1, cachedShare: 0 }, p.concurrency);
  // Pure decode: the step alone; a single request's TTFT: its prompt + the request overhead + one step.
  const step = speed.stepMs(p.concurrency);
  const single = speed.prefillMs(p.inputLen);
  // A TPOT lies between generation with the others' prompts read between steps (requests out of step) and
  // pure generation (benchmarks whose requests start and end together read the prompts inside the TTFT).
  const dec = p.basis === 'decode' ? { min: 1000 / step.max, max: 1000 / step.min } : p.basis === 'tpot' ? { min: lp.tokS.min, max: 1000 / step.min } : lp.tokS;
  // With other requests, the first-token wait under that load (queueing behind their prompts).
  const pre = p.concurrency > 1 ? lp.ttftMs : { min: single.min + cal.requestOverheadMs.min + step.min, max: single.max + cal.requestOverheadMs.max + step.max };
  const inD = p.decodeTokSPerUser !== undefined ? (p.decodeTokSPerUser >= dec.min && p.decodeTokSPerUser <= dec.max ? 'ok' : p.decodeTokSPerUser < dec.min ? 'BELOW' : 'ABOVE') : '';
  const inT = p.ttftMs !== undefined ? (p.ttftMs >= pre.min && p.ttftMs <= pre.max ? 'ok' : p.ttftMs < pre.min ? 'BELOW' : 'ABOVE') : '';
  console.log(
    `${p.id.slice(0, 44).padEnd(44)} ${p.gpu.slice(0, 14).padEnd(14)} tp${p.tp} c${String(p.concurrency).padEnd(4)} in${String(p.inputLen).padEnd(7)} ` +
      `dec ${p.decodeTokSPerUser !== undefined ? f(p.decodeTokSPerUser, 1).padStart(6) : '     -'} [${f(dec.min, 1)}–${f(dec.max, 1)}] ${inD.padEnd(5)} ` +
      `ttft ${p.ttftMs !== undefined ? f(p.ttftMs, 0).padStart(7) : '      -'} [${f(pre.min, 0)}–${f(pre.max, 0)}] ${inT}`,
  );
  out.push({ kind: 'perf', ...p, predictedDecode: dec, predictedTtftMs: pre });
  profiles.set(p.repo, { profile: m.planner.profile, layout: m.planner.layout, weights: m.planner.parallel.weights, family: m.planner.family, ...(m.planner.sparseTopk !== undefined ? { sparseTopk: m.planner.sparseTopk } : {}) });
}
const perf = out.filter((o) => o.kind === 'perf') as { id: string; basis: string; decodeTokSPerUser?: number; ttftMs?: number; predictedDecode: { min: number; max: number }; predictedTtftMs: { min: number; max: number } }[];
const outside = (v: number | undefined, r: { min: number; max: number }) => (v === undefined ? undefined : v < r.min ? (v - r.min) / r.min : v > r.max ? (v - r.max) / r.max : 0);
// The RTX 5090 vLLM runs: the measured value against the centre of the range, and the range's half-width.
const centre = (r: { min: number; max: number }) => (r.min + r.max) / 2;
for (const [what, key, range] of [
  ['generation', 'decodeTokSPerUser', 'predictedDecode'],
  ['TTFT', 'ttftMs', 'predictedTtftMs'],
] as const) {
  const local = perf.filter((p) => p.id.startsWith('5090-vllm-') && p[key] !== undefined);
  if (local.length === 0) continue;
  const errs = local.map((p) => ((p[key] as number) - centre(p[range])) / centre(p[range]));
  const halves = local.map((p) => (p[range].max - p[range].min) / 2 / centre(p[range]));
  console.log(`RTX 5090 vLLM, ${what}: measured vs centre from ${f(Math.min(...errs) * 100, 0)}% to ${f(Math.max(...errs) * 100, 0)}%; range half-width up to ±${f(Math.max(...halves) * 100, 0)}%`);
}
for (const [what, key, range] of [
  ['generation', 'decodeTokSPerUser', 'predictedDecode'],
  ['TTFT', 'ttftMs', 'predictedTtftMs'],
] as const) {
  const errs = perf.map((p) => outside(p[key], p[range])).filter((e): e is number => e !== undefined);
  const inside = errs.filter((e) => e === 0).length;
  const worst = Math.max(...errs.map(Math.abs));
  console.log(`${what}: ${inside}/${errs.length} points inside the predicted range; outside by at most ${f(worst * 100, 0)}%`);
}
// The offline regression fixture: every model's speed profile and every measured point (no tensors).
if (!only)
  writeFileSync(
    path.join(root, 'test/fixtures/calibration-points.json'),
    `${JSON.stringify(
      {
        $comment: 'Written by scripts/dev/calibrate.mjs: model speed profiles (from the safetensors headers) and the measured points of data/references.json and of the reference RTX 5090 runs. test/planner/calibration-points.test.ts recomputes the predictions offline.',
        profiles: Object.fromEntries(profiles),
        points: out.filter((o) => o.kind === 'perf').map(({ predictedDecode: _d, predictedTtftMs: _t, kind: _k, ...p }) => p),
        weights: out.filter((o) => o.kind === 'weights').map(({ kind: _k, ...w }) => w),
      },
      null,
      0,
    )}
`,
  );
mkdirSync(path.join(root, '.qa'), { recursive: true });
writeFileSync(path.join(root, '.qa/vram-calibration.json'), JSON.stringify(out, null, 1));

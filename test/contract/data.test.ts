// The reference data of the capacity planner (data/): schemas, units, and a source for
// every value. A GPU value that is not public is null and listed in gaps; a log number is one of its log lines.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  hardwareFileSchema,
  referencesFileSchema,
  TFLOPS_FORMATS,
  type GpuSpec,
  type Reference,
} from '../../src/contract/data';

const dir = path.resolve(import.meta.dirname, '../../data');
const readJson = (name: string): unknown => JSON.parse(readFileSync(path.join(dir, name), 'utf8'));
const hardware = hardwareFileSchema.parse(readJson('hardware.json'));
const references = referencesFileSchema.parse(readJson('references.json'));
const gpus = new Map(hardware.gpus.map((g) => [g.id, g]));

/** Dotted paths of the leaves of a value (arrays are leaves). */
function leaves(value: unknown, prefix = ''): [string, unknown][] {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return [[prefix, value]];
  return Object.entries(value).flatMap(([k, v]) => leaves(v, prefix ? `${prefix}.${k}` : k));
}
/** A source or gap naming "a.b" covers "a.b" and everything under it. */
const covers = (named: string, field: string) => field === named || field.startsWith(`${named}.`);

/** Fields that describe the entry itself rather than a specification value. */
const UNSOURCED = ['id', 'name', 'vendor', 'tflopsDerived', 'segment', 'formFactor', 'sources', 'gaps', 'notes', 'perGpuOf', 'p2pNote', 'nvlink.note', 'denseHalvedFromSparse'];
const specLeaves = (g: GpuSpec) => leaves(g).filter(([f]) => !UNSOURCED.some((u) => covers(u, f)));

const GIB = 2 ** 30;
const vramGiB = (g: GpuSpec) => (g.vramBytes ?? g.vramGBNominal * 1e9) / GIB;

describe('hardware.json', () => {
  it('has unique ids', () => {
    expect(new Set(hardware.gpus.map((g) => g.id)).size).toBe(hardware.gpus.length);
  });

  it.each(hardware.gpus.map((g) => [g.id, g] as const))('%s: every value has a source, every missing one a reason', (_, g) => {
    const sourced = g.sources.flatMap((s) => s.fields);
    const gapped = g.gaps.map((x) => x.field);
    for (const [field, value] of specLeaves(g)) {
      // A false flag ("no NVLink", "no FP4") or an unknown P2P needs no source; anything stated does.
      if (value === false || value === 'unknown') continue;
      if (value === null) expect(gapped.some((n) => covers(n, field)), `${field} is null: add a gaps entry`).toBe(true);
      else expect(sourced.some((n) => covers(n, field)), `${field} has no source`).toBe(true);
    }
    const known = new Set(specLeaves(g).map(([f]) => f));
    for (const f of [...sourced, ...gapped]) expect([...known].some((k) => covers(f, k)), `${f} is not a field`).toBe(true);
  });

  it.each(hardware.gpus.map((g) => [g.id, g] as const))('%s: units and figures are consistent', (_, g) => {
    if (g.vramBytes !== null && g.nvidiaSmiTotalMiB !== null) expect(g.vramBytes).toBe(g.nvidiaSmiTotalMiB * 2 ** 20);
    // CUDA sees the framebuffer less what the driver reserves.
    const smiBytes = g.nvidiaSmiTotalMiB === null ? null : g.nvidiaSmiTotalMiB * 2 ** 20;
    if (g.cudaTotalBytes !== null && smiBytes !== null) expect(g.cudaTotalBytes).toBeLessThanOrEqual(smiBytes);
    if (g.cudaTotalGiB !== null && smiBytes !== null) expect(g.cudaTotalGiB).toBeLessThanOrEqual(smiBytes / GIB + 0.005);
    // Linux (cudaTotalGiB) leaves CUDA less than Windows or the same, never more.
    if (g.cudaTotalGiB !== null && g.cudaTotalBytes !== null) expect(g.cudaTotalGiB).toBeLessThanOrEqual(g.cudaTotalBytes / GIB + 0.005);
    if (g.nvidiaSmiReservedMiB !== null) expect(g.nvidiaSmiReservedMiB).toBeLessThan((g.nvidiaSmiTotalMiB ?? Infinity) * 0.05);
    if (g.vramBytes !== null) {
      // A "32 GB" card exposes about 32 GiB (32 607 MiB on an RTX 5090): between the decimal GB and 1.1× the GiB.
      expect(g.vramBytes).toBeGreaterThanOrEqual(0.9 * g.vramGBNominal * 1e9);
      expect(g.vramBytes).toBeLessThanOrEqual(1.1 * g.vramGBNominal * GIB);
    }
    if (g.memoryBandwidthGBs !== null) {
      expect(g.memoryBandwidthGBs).toBeGreaterThan(100);
      expect(g.memoryBandwidthGBs).toBeLessThan(20_000);
    }
    for (const f of TFLOPS_FORMATS) {
      const dense = g.tflopsDense[f];
      const sparse = g.tflopsSparse[f];
      if (dense !== null) expect(dense).toBeLessThan(50_000);
      if (dense !== null && sparse !== null) expect(sparse).toBeGreaterThanOrEqual(dense);
      if (g.denseHalvedFromSparse.includes(f)) {
        expect(sparse, `${f}: halved from a sparse figure that is missing`).not.toBeNull();
        // NVIDIA rounds the halves (H100 SXM: 1979 sparse → 989 dense, not 989.5).
        expect(Math.abs((dense ?? 0) - (sparse ?? 0) / 2)).toBeLessThanOrEqual(1);
      }
    }
    if (g.power.maxW !== null) {
      expect(g.power.maxW).toBeGreaterThanOrEqual(50);
      expect(g.power.maxW).toBeLessThanOrEqual(2000);
    }
    if (g.power.configurableMinW !== null) expect(g.power.configurableMinW).toBeLessThanOrEqual(g.power.maxW ?? Infinity);
    expect(g.nvlink.supported || (g.nvlink.bandwidthGBs === null && g.nvlink.generation === null)).toBe(true);
  });

  it.each(hardware.gpus.map((g) => [g.id, g] as const))('%s: formats follow the architecture', (_, g) => {
    const cc = Number(g.computeCapability);
    const major = Math.floor(cc);
    const byArch: Record<GpuSpec['architecture'], number[]> = {
      ampere: [8],
      'ada-lovelace': [8],
      hopper: [9],
      blackwell: [10, 12],
      'blackwell-ultra': [10, 12],
    };
    expect(byArch[g.architecture]).toContain(major);
    // FP8 tensor cores arrived with Ada (8.9) and Hopper; FP4 / MX formats with Blackwell.
    if (g.formats.fp8) expect(cc).toBeGreaterThanOrEqual(8.9);
    if (g.formats.nvfp4 || g.formats.mxfp4 || g.formats.mxfp8) expect(major).toBeGreaterThanOrEqual(10);
    if (g.tflopsDense.fp4 !== null || g.tflopsSparse.fp4 !== null) expect(major).toBeGreaterThanOrEqual(10);
  });
});

/** Every number written in some log lines (thousands separators removed). */
function numbersIn(lines: readonly string[]): number[] {
  return (lines.join('\n').replace(/(\d),(?=\d{3})/g, '$1').match(/\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi) ?? []).map(Number);
}
const isMemoryEntry = (r: Reference) => r.log !== null;
const isSpeedEntry = (r: Reference) => r.perf.length > 0;

describe('references.json', () => {
  it('has unique ids', () => {
    expect(new Set(references.entries.map((r) => r.id)).size).toBe(references.entries.length);
  });

  it.each(references.entries.map((r) => [r.id, r] as const))('%s: complete enough to redo the calculation', (_, r) => {
    expect(isMemoryEntry(r) || isSpeedEntry(r), 'neither log numbers nor perf').toBe(true);
    const gpu = gpus.get(r.hardware.gpuId);
    expect(gpu, `unknown gpuId ${r.hardware.gpuId}`).toBeDefined();
    const h = r.hardware;
    // SGLang's dp is the attention data parallelism inside tp (--enable-dp-attention), not more GPUs.
    if (r.engine === 'vllm') expect(h.tp * h.pp * h.dp).toBeLessThanOrEqual(h.gpuCount);
    else expect(Math.max(h.tp * h.pp, h.dp)).toBeLessThanOrEqual(h.gpuCount);
    expect(h.gpuCount % h.nodes).toBe(0);
    if (r.engine === 'vllm') expect(r.params.memFractionStatic).toBeNull();
    else expect(r.params.gpuMemoryUtilization).toBeNull();
    if (isMemoryEntry(r)) {
      // Without --max-model-len the model's own context applies: the notes must say so, and vLLM's maximum
      // concurrency (which depends on it) can't be checked. SGLang sizes its KV pool by memory alone.
      if (r.engine === 'vllm' && r.params.maxModelLen === null) {
        expect(r.notes ?? '', 'max-model-len unknown and not explained').toMatch(/max-model-len/i);
        expect(r.log?.maximumConcurrency ?? null).toBeNull();
      }
      expect(r.params.gpuMemoryUtilization ?? r.params.memFractionStatic, 'memory entry without a memory fraction').not.toBeNull();
    }
  });

  it.each(references.entries.filter(isMemoryEntry).map((r) => [r.id, r] as const))('%s: log numbers are in its log lines', (_, r) => {
    const log = r.log!;
    const printed = numbersIn(log.rawLines);
    const vllmOnly = [log.gpuBlocks, log.maximumConcurrency, log.totalGpuMemoryGiB, log.nonTorchGiB, log.torchPeakIncreaseGiB];
    const sglangOnly = [log.maxTotalNumTokens, log.kvCacheAllocatedGB, log.availableMemAfterGB, log.weightLoadMemUsageGB, log.availMemAfterWeightsGB, log.cudaGraphMemUsageGB];
    for (const v of r.engine === 'vllm' ? sglangOnly : vllmOnly) expect(v, `${r.engine} does not print this`).toBeNull();
    const { rawLines: _lines, ...numbers } = log;
    const values = Object.entries(numbers).filter((e): e is [string, number] => e[1] !== null);
    expect(values.length, 'a log with no number').toBeGreaterThan(0);
    for (const [field, v] of values) {
      // kvCacheAllocatedGB is K + V: either printed as such or as its two halves.
      if (field === 'kvCacheAllocatedGB' && !printed.includes(v)) {
        const pairs = printed.flatMap((a, i) => printed.slice(i + 1).map((b) => a + b));
        expect(pairs.some((s) => Math.abs(s - v) < 0.011), `${field} ${v} is not K + V of the log`).toBe(true);
      } else expect(printed, `${field} ${v} is not in the log lines`).toContain(v);
    }
    // Per-GPU memory can't exceed the card.
    const gpu = gpus.get(r.hardware.gpuId)!;
    for (const v of [log.modelLoadingGiB, log.availableKvCacheGiB, log.cudaGraphGiB, log.kvCacheAllocatedGB, log.totalGpuMemoryGiB, log.weightLoadMemUsageGB]) {
      if (v !== null) expect(v).toBeLessThan(vramGiB(gpu) * 1.08);
    }
  });

  it.each(references.entries.filter(isSpeedEntry).map((r) => [r.id, r] as const))('%s: perf rows have a known load', (_, r) => {
    for (const p of r.perf) {
      expect(p.concurrency !== null || p.requestRatePerS !== null || p.numPrompts !== null, 'no known load').toBe(true);
      const measured = [p.outputTokPerS, p.totalTokPerS, p.inputTokPerS, p.requestPerS, p.ttftMsMedian, p.ttftMsMean, p.tpotMsMedian, p.tpotMsMean, p.itlMsMedian, p.decodeTokPerSPerUser];
      expect(measured.some((v) => v !== null), 'a perf row with no number').toBe(true);
      if (p.outputTokPerS !== null && p.totalTokPerS !== null) expect(p.totalTokPerS).toBeGreaterThanOrEqual(p.outputTokPerS);
    }
  });
});

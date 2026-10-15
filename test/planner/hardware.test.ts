import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { hardwareFileSchema, type GpuSpec } from '../../src/contract/data';
import { checkLayout, gpuFromSpec, GPUS_PER_SERVER, groupLink, groupSize, layoutsFor, MAX_STAGES, nvlinkDomain, serverSize, soldAsCards, tpLink, type Gpu, type ParallelShape } from '../../src/planner/hardware';

const hw = hardwareFileSchema.parse(JSON.parse(readFileSync(path.join('data', 'hardware.json'), 'utf8')));
const spec = (id: string): GpuSpec => {
  const s = hw.gpus.find((g) => g.id === id);
  if (!s) throw new Error(`missing ${id}`);
  return s;
};
const gpu = (id: string): Gpu => gpuFromSpec(spec(id));

describe('gpuFromSpec: CUDA-visible memory', () => {
  const bare = { cudaTotalBytes: null, cudaTotalGiB: null };
  it('measured values span the systems: RTX 5090 31.35 GiB on Linux … 34 162 016 256 B on Windows', () => {
    // 31.35 × 1024 = 32 102 MiB; 34 162 016 256 / 2^20 = 32 579 MiB.
    const g = gpu('rtx-5090-32gb');
    expect(g.cudaMiB).toEqual({ min: 32_102, max: 32_579 });
    expect(g).toMatchObject({ id: 'rtx-5090-32gb', name: 'RTX 5090', segment: 'consumer', architecture: 'blackwell', cc: 12, nominalGB: 32, bandwidthGBs: 1792 });
    expect(g.tflops).toEqual({ bf16: 209.5, fp8: 419, fp4: 1676 });
    // One measurement only: exact.
    expect(gpuFromSpec({ ...spec('rtx-5090-32gb'), cudaTotalGiB: null }).cudaMiB).toEqual({ min: 32_579, max: 32_579 });
  });

  it('no measurement: nvidia-smi total minus 560 MiB or 0.7 % … minus 16 MiB', () => {
    // 32 607 − max(560, 228) … 32 607 − 16.
    expect(gpuFromSpec({ ...spec('rtx-5090-32gb'), ...bare }).cudaMiB).toEqual({ min: 32_047, max: 32_591 });
    // H200 SXM: 143 771 − 1 006 (0.7 %).
    expect(gpuFromSpec({ ...spec('h200-sxm-141gb'), ...bare }).cudaMiB).toEqual({ min: 142_765, max: 143_755 });
  });

  it('only the framebuffer bytes: their MiB stand for nvidia-smi', () => {
    // 34 190 917 632 B / 2^20 = 32 607 MiB.
    expect(gpuFromSpec({ ...spec('rtx-5090-32gb'), ...bare, nvidiaSmiTotalMiB: null }).cudaMiB).toEqual({ min: 32_047, max: 32_591 });
  });

  it('nothing measured nor listed (an RTX 6000 Ada without its nvidia-smi total): 97 % of the nominal GB', () => {
    // floor(48 × 1024 × 0.97) = 47 677 MiB, minus 560 … 16.
    expect(gpuFromSpec({ ...spec('rtx-6000-ada-48gb'), ...bare, nvidiaSmiTotalMiB: null, vramBytes: null }).cudaMiB).toEqual({ min: 47_117, max: 47_661 });
    expect(gpu('rtx-6000-ada-48gb').name).toBe('RTX 6000 Ada Generation');
  });

  it('no published bandwidth: 0 (the speed is then not estimated)', () => {
    expect(gpuFromSpec({ ...spec('rtx-5090-32gb'), memoryBandwidthGBs: null }).bandwidthGBs).toBe(0);
  });
});

describe('gpuFromSpec: GPU-to-GPU link', () => {
  it('NVLink: half the bidirectional total per direction', () => {
    // H100 SXM: 900 GB/s total → 450 per direction; RTX 3090 bridge: 112.5 → 56.25.
    expect(gpu('h100-sxm5-80gb')).toMatchObject({ link: 'nvlink', linkGBs: 450 });
    expect(gpu('rtx-3090-24gb')).toMatchObject({ link: 'nvlink', linkGBs: 56.25 });
  });

  it('NVLink "supported" without a figure counts as PCIe', () => {
    const g = gpuFromSpec({ ...spec('rtx-a6000-48gb'), nvlink: { supported: true, generation: 3, bandwidthGBs: null, domainGpus: 2 } });
    // A6000: P2P supported, the catalog's 31.5 GB/s per direction.
    expect(g).toMatchObject({ link: 'pcie-p2p', linkGBs: 31.5 });
    // NVLink with a figure but no published reach: one HGX board, 8.
    const h = gpuFromSpec({ ...spec('h100-sxm5-80gb'), nvlink: { ...spec('h100-sxm5-80gb').nvlink, domainGpus: null } });
    expect(h).toMatchObject({ link: 'nvlink', nvlinkDomain: GPUS_PER_SERVER });
  });

  it('PCIe P2P at the catalog rate; without P2P half of the PCIe rate (through host memory)', () => {
    expect(gpu('rtx-6000-ada-48gb')).toMatchObject({ link: 'pcie-p2p', linkGBs: 31.5 });
    // RTX 5090: PCIe 5 with no catalog figure → 64 GB/s, through the CPU → 32.
    expect(gpu('rtx-5090-32gb')).toMatchObject({ link: 'pcie-host', linkGBs: 32 });
    // RTX 4090: PCIe 4 → 32 GB/s → 16.
    expect(gpu('rtx-4090-24gb')).toMatchObject({ link: 'pcie-host', linkGBs: 16 });
  });

  it('a PCIe generation outside the table falls back to 32 GB/s', () => {
    const s = { ...spec('rtx-6000-ada-48gb'), pcie: { gen: 7, lanes: null, bandwidthGBsPerDirection: null } } as unknown as GpuSpec;
    expect(gpuFromSpec(s).linkGBs).toBe(32);
  });
});

describe('layoutsFor', () => {
  it('each count as one group and as copies of every smaller group, each group in TP, in stages (up to 4) and both; no group above one server', () => {
    const show = (l: { count: number; tp: number; pp: number; replicas?: number }) => `${l.count} tp${l.tp} pp${l.pp} x${l.replicas ?? '-'}`;
    // An HGX board: every split of 8 GPUs.
    expect(layoutsFor([gpu('h100-sxm5-80gb')], [8]).map(show)).toEqual([
      '8 tp8 pp1 x-',
      '8 tp4 pp2 x-',
      '8 tp2 pp4 x-',
      '8 tp4 pp1 x2',
      '8 tp2 pp2 x2',
      '8 tp1 pp4 x2',
      '8 tp2 pp1 x4',
      '8 tp1 pp2 x4',
      '8 tp1 pp1 x8',
    ]);
    // 4 consumer cards per machine: 8 is two machines of 4, never one group of 8.
    expect(layoutsFor([gpu('rtx-5090-32gb')], [1, 2, 8]).map(show)).toEqual(['1 tp1 pp1 x-', '2 tp2 pp1 x-', '2 tp1 pp2 x-', '2 tp1 pp1 x2', '8 tp4 pp1 x2', '8 tp2 pp2 x2', '8 tp1 pp4 x2', '8 tp2 pp1 x4', '8 tp1 pp2 x4', '8 tp1 pp1 x8']);
    expect(MAX_STAGES).toBe(4);
    const out = layoutsFor([gpu('rtx-5090-32gb'), gpu('h100-sxm5-80gb')]);
    // tp × pp × replicas = count, always.
    for (const l of out) expect(groupSize(l) * (l.replicas ?? 1)).toBe(l.count);
    // A count that does not halve to 1: the halving stops at the first fraction.
    expect(layoutsFor([gpu('rtx-5090-32gb')], [3]).map((l) => [l.tp, l.pp, l.replicas])).toEqual([[3, 1, undefined]]);
    expect(layoutsFor([gpu('h100-sxm5-80gb')], [6]).map((l) => [l.tp, l.pp, l.replicas])).toEqual([
      [6, 1, undefined],
      [3, 2, undefined],
      [3, 1, 2],
    ]);
  });
});

describe('one server at most per group, and the NVLink domain', () => {
  it('more GPUs than one server holds are copies on several servers, never one group split across them', () => {
    expect(layoutsFor([gpu('h100-sxm5-80gb')], [16]).every((l) => groupSize(l) <= 8 && (l.replicas ?? 1) >= 2)).toBe(true);
    expect(layoutsFor([gpu('h100-sxm5-80gb')], [16]).filter((l) => l.pp === 1).map((l) => [l.tp, l.replicas])).toEqual([
      [8, 2],
      [4, 4],
      [2, 8],
      [1, 16],
    ]);
    expect(serverSize(gpu('h100-sxm5-80gb'))).toBe(GPUS_PER_SERVER);
    // A card sold two per server (a dual-superchip system): no group of 4, two servers of 2 instead.
    const two = { ...gpu('gh200-144gb'), maxPerServer: 2 };
    expect(layoutsFor([two], [4]).map((l) => [l.tp, l.pp, l.replicas])).toEqual([
      [2, 1, 2],
      [1, 2, 2],
      [1, 1, 4],
    ]);
    // Never more than 8 in a group, whatever the catalog says.
    expect(serverSize({ maxPerServer: 72 })).toBe(8);
    expect(serverSize({})).toBe(8);
  });

  it('a group larger than the NVLink domain talks over PCIe, and pipeline layouts are offered for it', () => {
    const qwen: ParallelShape = { heads: 32, kvHeads: 8, layers: 36, weights: 'bf16', moe: false };
    const at = (id: string, tp: number) => ({ gpu: gpu(id), count: tp, tp, pp: 1 });
    // H100 PCIe: NVLink bridges pairs only.
    const pcie = gpu('h100-pcie-80gb');
    expect(pcie).toMatchObject({ link: 'nvlink', nvlinkDomain: 2 });
    expect(groupLink({ gpu: pcie, tp: 2, pp: 1 })).toEqual({ link: 'nvlink', gbs: pcie.linkGBs });
    expect(groupLink({ gpu: pcie, tp: 4, pp: 1 }).link).toBe('pcie-p2p');
    // One copy on 4 or 8: TP 2 on each NVLink pair × 2 or 4 stages, among the other splits.
    expect(layoutsFor([pcie], [4, 8]).filter((l) => !l.replicas && l.pp > 1).map((l) => [l.count, l.tp, l.pp])).toEqual([[4, 2, 2], [4, 1, 4], [8, 4, 2], [8, 2, 4]]);
    expect(tpLink({ gpu: pcie, tp: 2 }).link).toBe('nvlink');
    expect(checkLayout({ gpu: pcie, count: 8, tp: 2, pp: 4 }, qwen, false).warnings).toEqual(['NVLink connects only 2 of these cards: each TP 2 group stays on NVLink, the 4 pipeline stages talk over PCIe']);
    expect(checkLayout(at('h100-pcie-80gb', 4), qwen, false).warnings).toEqual(['4 GPUs in one group, but NVLink connects only 2 of these cards: the group talks over PCIe (slower)']);
    expect(checkLayout(at('h100-pcie-80gb', 2), qwen, false).warnings).toEqual([]);
    // H200 NVL: a 4-way bridge; HGX boards: all 8.
    expect(groupLink({ gpu: gpu('h200-nvl-141gb'), tp: 4, pp: 1 }).link).toBe('nvlink');
    expect(groupLink({ gpu: gpu('h200-nvl-141gb'), tp: 8, pp: 1 }).link).toBe('pcie-p2p');
    expect(groupLink({ gpu: gpu('h100-sxm5-80gb'), tp: 8, pp: 1 }).link).toBe('nvlink');
    // A card without NVLink keeps its own link; a hand-made card without the new fields gets the defaults.
    expect(groupLink({ gpu: gpu('rtx-5090-32gb'), tp: 4, pp: 1 })).toEqual({ link: 'pcie-host', gbs: 32 });
    const bare = { ...gpu('h100-sxm5-80gb'), nvlinkDomain: undefined, pcie: undefined };
    expect(nvlinkDomain(bare)).toBe(8);
    expect(nvlinkDomain({ link: 'pcie-p2p' })).toBe(1);
    expect(groupLink({ gpu: { ...bare, nvlinkDomain: 2 }, tp: 4, pp: 1 })).toEqual({ link: 'pcie-p2p', gbs: bare.linkGBs });
    // Without P2P, PCIe through host memory.
    expect(gpuFromSpec({ ...spec('rtx-3090-24gb'), p2p: 'unsupported' }).pcie?.link).toBe('pcie-host');
    expect(gpu('h100-pcie-80gb').pcie?.link).toBe('pcie-p2p');
  });
});

describe('groupSize', () => {
  it('the GPUs of one copy: tp × pp, whatever the copies', () => {
    expect(groupSize({ gpu: gpu('h100-sxm5-80gb'), count: 8, tp: 2, pp: 1, replicas: 4 })).toBe(2);
    expect(groupSize({ gpu: gpu('rtx-5090-32gb'), count: 4, tp: 1, pp: 4 })).toBe(4);
    expect(groupSize({ gpu: gpu('rtx-5090-32gb'), count: 1, tp: 1, pp: 1 })).toBe(1);
  });
});

describe('checkLayout', () => {
  // Qwen3-8B-like: 32 heads, 8 KV heads, 36 layers.
  const qwen: ParallelShape = { heads: 32, kvHeads: 8, layers: 36, weights: 'bf16', moe: false };
  const at = (id: string, tp: number, pp = 1) => ({ gpu: gpu(id), count: tp * pp, tp, pp });

  it('a clean layout: no blockers, no warnings', () => {
    expect(checkLayout(at('h100-sxm5-80gb', 8), qwen, false)).toEqual({ blockers: [], warnings: [] });
  });

  it('whether the engine starts the model on the card is not a layout check (compat.ts)', () => {
    const dsv41: ParallelShape = { ...qwen, heads: 64, kvHeads: 64, architecture: 'DeepseekV41ForCausalLM' };
    expect(checkLayout(at('rtx-5090-32gb', 1), dsv41, true).blockers).toEqual([]);
  });

  it('heads and KV heads must divide by TP (fewer KV heads than GPUs are replicated)', () => {
    expect(checkLayout(at('h100-sxm5-80gb', 3), qwen, false).blockers).toEqual([
      '32 attention heads do not divide by TP 3',
      '8 KV heads do not divide by TP 3',
    ]);
    // TP 16 > 8 KV heads: replicated, allowed.
    expect(checkLayout(at('h100-sxm5-80gb', 16), qwen, false).blockers).toEqual([]);
  });

  it('linear-attention heads must divide by TP', () => {
    const m = { ...qwen, linearKeyHeads: 6, linearValueHeads: 32 };
    // 6 % 4 = 2 (blocker), 32 % 4 = 0.
    expect(checkLayout(at('h100-sxm5-80gb', 4), m, false).blockers).toEqual(['6 linear-attention key heads do not divide by TP 4']);
    expect(checkLayout(at('h100-sxm5-80gb', 4), { ...qwen, linearKeyHeads: 16, linearValueHeads: 6 }, false).blockers).toEqual(['6 linear-attention value heads do not divide by TP 4']);
  });

  it('FP8 blocks: the split FFN sizes must stay multiples of the block', () => {
    const m: ParallelShape = { ...qwen, weights: 'fp8', intermediate: 18_944, expertIntermediate: 1536, quantBlock: 128 };
    // TP 4: 18 944 / 4 = 4736 = 37 × 128, 1536 / 4 = 384 = 3 × 128 → fine.
    expect(checkLayout(at('h100-sxm5-80gb', 4), m, false).blockers).toEqual([]);
    // TP 8: 2368 / 128 = 18.5 and 192 / 128 = 1.5 → both refused.
    expect(checkLayout(at('h100-sxm5-80gb', 8), m, false).blockers).toEqual([
      'with TP 8 the FFN (18944/8) is not a multiple of the FP8 block 128',
      'with TP 8 the expert FFN (1536/8) is not a multiple of the FP8 block 128',
    ]);
    // Only the expert size known: only it is checked.
    expect(checkLayout(at('h100-sxm5-80gb', 8), { ...qwen, expertIntermediate: 1536, quantBlock: 128 }, false).blockers).toEqual([
      'with TP 8 the expert FFN (1536/8) is not a multiple of the FP8 block 128',
    ]);
  });

  it('more pipeline stages than layers', () => {
    expect(checkLayout(at('rtx-5090-32gb', 1, 4), { ...qwen, layers: 3 }, false).blockers).toEqual(['3 layers are not enough for 4 pipeline stages']);
  });

  it('format warnings: FP8 and FP8 KV on Ampere, NVFP4 / MXFP4 below Blackwell', () => {
    expect(checkLayout(at('a100-sxm4-80gb', 1), { ...qwen, weights: 'fp8' }, true).warnings).toEqual([
      'FP8 weights on Ampere: vLLM uses weight-only kernels (same memory, compute in bf16, slower)',
      'fp8 cache on Ampere: not every attention backend supports it (to be checked with the engine version)',
    ]);
    // Ada (8.9) runs FP8 natively.
    expect(checkLayout(at('l40s-48gb', 1), { ...qwen, weights: 'fp8' }, true).warnings).toEqual([]);
    expect(checkLayout(at('h100-sxm5-80gb', 1), { ...qwen, weights: 'nvfp4' }, false).warnings).toEqual([
      'NVFP4 weights without Blackwell: vLLM runs them with weight-only kernels, slower; support to be checked per architecture',
    ]);
    // The notes name the engine asked.
    expect(checkLayout(at('h100-sxm5-80gb', 1), { ...qwen, weights: 'mxfp4' }, false, 'sglang').warnings[0]).toMatch(/^MXFP4 weights without Blackwell: SGLang runs them/);
    expect(checkLayout(at('a100-sxm4-80gb', 1), { ...qwen, weights: 'fp8' }, false, 'sglang').warnings[0]).toMatch(/^FP8 weights on Ampere: SGLang uses/);
    expect(checkLayout(at('h100-sxm5-80gb', 1), { ...qwen, weights: 'mxfp4' }, false).warnings[0]).toMatch(/^MXFP4 weights without Blackwell/);
    expect(checkLayout(at('b200-sxm-180gb', 1), { ...qwen, weights: 'mxfp4' }, false).warnings).toEqual([]);
  });

  it('consumer cards without NVLink nor P2P: TP through host memory, PP needs slots and power', () => {
    expect(checkLayout(at('rtx-5090-32gb', 2), qwen, false).warnings).toEqual([
      'TP 2 on cards without NVLink or P2P: every all-reduce goes through CPU memory over PCIe (slower; needs x16 slots and an adequate power supply)',
    ]);
    expect(checkLayout(at('rtx-5090-32gb', 1, 2), qwen, false).warnings).toEqual(['2 consumer cards in a pipeline: needs PCIe slots and an adequate power supply']);
    // Copies of single GPUs share nothing: no warning; copies of a TP 2 group still warn about TP.
    expect(checkLayout({ gpu: gpu('rtx-5090-32gb'), count: 4, tp: 1, pp: 1, replicas: 4 }, qwen, false).warnings).toEqual([]);
    expect(checkLayout({ gpu: gpu('rtx-5090-32gb'), count: 4, tp: 2, pp: 1, replicas: 2 }, qwen, false).warnings[0]).toMatch(/^TP 2 on cards without NVLink/);
    expect(checkLayout(at('rtx-5090-32gb', 1), qwen, false).warnings).toEqual([]);
  });

  it('TP above 2 over PCIe P2P is slower than NVLink', () => {
    expect(checkLayout(at('rtx-6000-ada-48gb', 4), qwen, false).warnings).toEqual(['TP 4 over PCIe (no NVLink): all-reduces slower than over NVLink']);
    expect(checkLayout(at('rtx-6000-ada-48gb', 2), qwen, false).warnings).toEqual([]);
  });
});

describe('soldAsCards', () => {
  it('rack-scale NVL72 systems are not offered as 1–8 cards', () => {
    expect(soldAsCards({ id: 'gb200-nvl72' })).toBe(false);
    expect(soldAsCards({ id: 'h200-sxm-141gb' })).toBe(true);
  });
});

describe('derived values', () => {
  it('a card whose TFLOPS are derived, not published as they are, says so; whole memory derived too', () => {
    expect(gpu('rtx-pro-6000-blackwell-server-96gb').tflopsDerived).toBe(true);
    expect(gpu('rtx-pro-6000-blackwell-workstation-96gb').tflopsDerived).toBeUndefined();
    expect(gpu('gh200-144gb').totalDerived).toBe(true);
  });
});

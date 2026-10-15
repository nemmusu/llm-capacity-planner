/**
 * NVIDIA hardware for the capacity planner: the catalog's cards (data/hardware.json, each value
 * with its source) as the planner reads them, the multi-GPU layouts worth considering, and what makes a
 * layout impossible (vLLM refuses to start) or worth a warning.
 */
import type { GpuSpec } from '../contract/data';
import type { MiBRange } from '../model/serving-fit';

const MIB = 1024 * 1024;

export interface Gpu {
  id: string;
  name: string;
  segment: GpuSpec['segment'];
  architecture: GpuSpec['architecture'];
  /** Compute capability as a number (8.6, 8.9, 9.0, 12.0…). */
  cc: number;
  /**
   * Memory CUDA reports (torch total_memory), on which vLLM computes its share: exact when the catalog
   * has it, else nvidia-smi's total minus the driver's reserve (16–520 MiB measured: 23 MiB in WSL on an
   * RTX 5090, 470 MiB on an H100 under Linux).
   */
  cudaMiB: MiBRange;
  nominalGB: number;
  /** The whole memory as nvidia-smi reports it, MiB (absent on a hand-made card: then the CUDA-visible maximum). */
  totalMiB?: number;
  /** No nvidia-smi total or byte count was found: totalMiB comes from the CUDA-visible measurement or the size sold × 0.97. */
  totalDerived?: true;
  /** The TFLOPS are derived, not published as they are (data/hardware.json tflopsDerived says how). */
  tflopsDerived?: true;
  bandwidthGBs: number;
  /** Dense tensor TFLOPS (no sparsity) per format; null where NVIDIA publishes none. */
  tflops: { bf16: number | null; fp8: number | null; fp4: number | null };
  /** GPU-to-GPU link per direction, GB/s: NVLink when present, else PCIe (through host memory without P2P). */
  linkGBs: number;
  link: 'nvlink' | 'pcie-p2p' | 'pcie-host';
  /**
   * GPUs one NVLink fabric connects (2 with a bridge between two PCIe cards, 4 with the H200 NVL 4-way bridge, 8 on
   * an HGX board); 1 without NVLink. A tensor- or pipeline-parallel group larger than this talks over PCIe.
   */
  nvlinkDomain?: number;
  /** The PCIe path between cards of a group larger than the NVLink domain. Default: the card's own link. */
  pcie?: { link: 'pcie-p2p' | 'pcie-host'; gbs: number };
  /** Most cards of this model in one server (node); default GPUS_PER_SERVER. */
  maxPerServer?: number;
  formats: GpuSpec['formats'];
}

/** GPUs in one server (node) when the catalog gives no other figure: an HGX board, an 8-slot PCIe server. */
export const GPUS_PER_SERVER = 8;

/** Most cards of one model in one server: no tensor- or pipeline-parallel group is larger (multi-node is not modelled). */
export function serverSize(gpu: Pick<Gpu, 'maxPerServer'>): number {
  return Math.min(GPUS_PER_SERVER, gpu.maxPerServer ?? GPUS_PER_SERVER);
}

/** GPUs one NVLink fabric connects for this card (1 without NVLink). */
export function nvlinkDomain(gpu: Pick<Gpu, 'link' | 'nvlinkDomain'>): number {
  return gpu.nvlinkDomain ?? (gpu.link === 'nvlink' ? GPUS_PER_SERVER : 1);
}

/** The link a group of the layout uses: NVLink when the whole group is inside one NVLink domain, else PCIe. */
export function groupLink(layout: Pick<GpuLayout, 'gpu' | 'tp' | 'pp'>): { link: Gpu['link']; gbs: number } {
  const { gpu } = layout;
  if (gpu.link !== 'nvlink' || layout.tp * layout.pp <= nvlinkDomain(gpu)) return { link: gpu.link, gbs: gpu.linkGBs };
  return gpu.pcie ? { link: gpu.pcie.link, gbs: gpu.pcie.gbs } : { link: 'pcie-p2p', gbs: gpu.linkGBs };
}

/** The link the tensor-parallel all-reduces use: NVLink when the TP group is inside one NVLink domain (the pipeline
 * stages of a TP × PP layout may still cross PCIe: groupLink). */
export function tpLink(layout: Pick<GpuLayout, 'gpu' | 'tp'>): { link: Gpu['link']; gbs: number } {
  return groupLink({ ...layout, pp: 1 });
}

/** PCIe x16 per direction, GB/s (the rate when the catalog has no figure). */
const PCIE_GBS: Record<number, number> = { 3: 16, 4: 32, 5: 64, 6: 128 };

/** The planner's view of one catalog card. */
export function gpuFromSpec(spec: GpuSpec): Gpu {
  // Measured values span the systems (Linux lower, Windows / WSL higher: 31.35 vs 31.82 GiB on an RTX 5090).
  const measured = [
    ...(spec.cudaTotalBytes !== null ? [Math.floor(spec.cudaTotalBytes / MIB)] : []),
    ...(spec.cudaTotalGiB !== null ? [Math.round(spec.cudaTotalGiB * 1024)] : []),
  ];
  // nvidia-smi's total; without it, what CUDA was measured to see, and only without that the size sold × 0.97 (derived).
  const derived = spec.nvidiaSmiTotalMiB === null && spec.vramBytes === null;
  const smi = spec.nvidiaSmiTotalMiB ?? (spec.vramBytes !== null ? Math.floor(spec.vramBytes / MIB) : measured.length > 0 ? Math.max(...measured) : Math.floor(spec.vramGBNominal * 1024 * 0.97));
  // Without a measurement: nvidia-smi's total minus what CUDA never sees (450–730 MiB on the measured
  // cards, up to 0.7 %; 28 MiB under WSL).
  const cudaMiB: MiBRange =
    measured.length > 0 ? { min: Math.min(...measured), max: Math.max(...measured) } : { min: smi - Math.max(560, Math.round(0.007 * smi)), max: smi - 16 };
  const nvlink = spec.nvlink.supported && spec.nvlink.bandwidthGBs !== null;
  const pcie = spec.pcie.bandwidthGBsPerDirection ?? PCIE_GBS[spec.pcie.gen] ?? 32;
  return {
    id: spec.id,
    name: spec.name.replace(/^NVIDIA (GeForce )?/, ''),
    segment: spec.segment,
    architecture: spec.architecture,
    cc: Number(spec.computeCapability),
    cudaMiB,
    totalMiB: smi,
    ...(derived ? { totalDerived: true } : {}),
    ...(spec.tflopsDerived ? { tflopsDerived: true } : {}),
    nominalGB: spec.vramGBNominal,
    bandwidthGBs: spec.memoryBandwidthGBs ?? 0,
    tflops: { bf16: spec.tflopsDense.bf16, fp8: spec.tflopsDense.fp8, fp4: spec.tflopsDense.fp4 },
    // NVLink figures are bidirectional totals: half per direction.
    linkGBs: nvlink ? (spec.nvlink.bandwidthGBs as number) / 2 : spec.p2p === 'supported' ? pcie : pcie / 2,
    link: nvlink ? 'nvlink' : spec.p2p === 'supported' ? 'pcie-p2p' : 'pcie-host',
    nvlinkDomain: nvlink ? (spec.nvlink.domainGpus ?? GPUS_PER_SERVER) : 1,
    pcie: spec.p2p === 'supported' ? { link: 'pcie-p2p', gbs: pcie } : { link: 'pcie-host', gbs: pcie / 2 },
    // Without an NVIDIA figure: 8 in a data-center server, 4 in a workstation or desktop (data/hardware.json gaps).
    maxPerServer: spec.server.maxGpus ?? (spec.segment === 'datacenter' ? GPUS_PER_SERVER : 4),
    formats: spec.formats,
  };
}

/** Rack-scale systems (GB200 / GB300 NVL72) come as whole racks of 72 GPUs, never as 1–8 cards. */
export function soldAsCards(gpu: Pick<Gpu, 'id'>): boolean {
  return !/nvl72/.test(gpu.id);
}

/** How many cards and how the model is split over them. */
export interface GpuLayout {
  gpu: Gpu;
  count: number;
  /** Tensor parallel × pipeline parallel × replicas = count. */
  tp: number;
  pp: number;
  /**
   * Independent copies of the model (data parallel: one engine per tp × pp GPUs behind a load balancer);
   * default 1. Many small-model users scale this way, not with more TP.
   */
  replicas?: number;
}

/** GPUs per layout tried when none is asked (--count). */
export const DEFAULT_COUNTS: readonly number[] = [1, 2, 4, 8];

/** Most pipeline stages tried in one copy of the model. */
export const MAX_STAGES = 4;

/**
 * Layouts to compare for each card: 1, 2, 4 and 8 GPUs as one copy of the model or as copies of a smaller group
 * (8 GPUs: one group of 8, 2 copies × 4, 4 copies × 2, 8 copies). Each group is tried in tensor parallel, in pipeline
 * stages (up to MAX_STAGES) and as both (TP inside each stage): with MLA, tensor parallelism repeats the KV cache on
 * every GPU while pipeline stages split it, so a pipeline can fit where TP does not, on any link. Where a group is
 * larger than the NVLink domain (bridged PCIe cards), the TP part stays inside it and the stages cross PCIe (H200 NVL:
 * TP4 × 2 stages). No group is larger than one server (serverSize): more GPUs than one server holds are independent
 * copies on several servers, never one model split across them. The planner ranks them and each card keeps its best.
 */
export function layoutsFor(gpus: readonly Gpu[], counts: readonly number[] = DEFAULT_COUNTS): GpuLayout[] {
  const out: GpuLayout[] = [];
  for (const gpu of gpus) {
    const server = serverSize(gpu);
    for (const count of counts)
      for (let group = count; group >= 1; group /= 2) {
        if (!Number.isInteger(group)) break;
        if (group > server) continue;
        const replicas = count / group;
        for (let pp = 1; pp <= Math.min(group, MAX_STAGES); pp *= 2) if (group % pp === 0) out.push({ gpu, count, tp: group / pp, pp, ...(replicas > 1 ? { replicas } : {}) });
      }
  }
  return out;
}

/** GPUs of one copy of the model (tp × pp). */
export function groupSize(l: GpuLayout): number {
  return l.tp * l.pp;
}

/** What tensor parallelism and the weight format need from the model (config.json). */
export interface ParallelShape {
  heads: number;
  kvHeads: number;
  /** Gated-delta / linear attention heads (they are split by TP too). */
  linearKeyHeads?: number;
  linearValueHeads?: number;
  intermediate?: number;
  expertIntermediate?: number;
  /** FP8 block quantization (weight_block_size): split dims must stay multiples of it. */
  quantBlock?: number;
  layers: number;
  /** Weights format: bf16, fp8, nvfp4, mxfp4, int4 (AWQ / GPTQ), int8. */
  weights: 'bf16' | 'fp8' | 'nvfp4' | 'mxfp4' | 'int4' | 'int8';
  /** Mixture of experts (expert parallel is an option on 2+ GPUs). */
  moe: boolean;
  /** config.json architectures[0] (e.g. "DeepseekV41ForCausalLM"). */
  architecture?: string;
}

/** checkLayout's generic warnings about a weight or cache format on an older architecture: the compatibility matrix
 * replaces them with what the engine really does where it has a verdict for the model on the card (cli/compat.ts). */
export const FORMAT_WARNING = /^(FP8 weights on Ampere|NVFP4 weights without Blackwell|MXFP4 weights without Blackwell|fp8 cache on Ampere)/;

export interface LayoutCheck {
  /** The engine refuses to start with this layout (the reason). */
  blockers: string[];
  /** It starts, with a caveat worth knowing. */
  warnings: string[];
}

const ENGINE = { vllm: 'vLLM', sglang: 'SGLang' } as const;

/**
 * Why a layout cannot run the model, and what to know when it can. The head and block divisibility checks hold for
 * vLLM v0.31 and SGLang v0.5.21 alike (both shard attention heads and FFN blocks over TP); the kernel notes name the
 * engine asked. Whether the engine starts the model at all on the card is compat.ts.
 */
export function checkLayout(layout: GpuLayout, m: ParallelShape, kvFp8: boolean, engine: 'vllm' | 'sglang' = 'vllm'): LayoutCheck {
  const { tp, pp, gpu } = layout;
  const name = ENGINE[engine];
  const link = groupLink(layout).link;
  const blockers: string[] = [];
  const warnings: string[] = [];
  if (m.heads % tp !== 0) blockers.push(`${m.heads} attention heads do not divide by TP ${tp}`);
  if (m.kvHeads >= tp && m.kvHeads % tp !== 0) blockers.push(`${m.kvHeads} KV heads do not divide by TP ${tp}`);
  for (const [n, what] of [
    [m.linearKeyHeads, 'linear-attention key heads'],
    [m.linearValueHeads, 'linear-attention value heads'],
  ] as const)
    if (n !== undefined && n % tp !== 0) blockers.push(`${n} ${what} do not divide by TP ${tp}`);
  if (m.quantBlock)
    for (const [n, what] of [
      [m.intermediate, 'FFN'],
      [m.expertIntermediate, 'expert FFN'],
    ] as const)
      if (n !== undefined && (n / tp) % m.quantBlock !== 0) blockers.push(`with TP ${tp} the ${what} (${n}/${tp}) is not a multiple of the FP8 block ${m.quantBlock}`);
  if (pp > m.layers) blockers.push(`${m.layers} layers are not enough for ${pp} pipeline stages`);
  if (m.weights === 'fp8' && gpu.cc < 8.9) warnings.push(`FP8 weights on Ampere: ${name} uses weight-only kernels (same memory, compute in bf16, slower)`);
  if ((m.weights === 'nvfp4' || m.weights === 'mxfp4') && gpu.cc < 10)
    warnings.push(`${m.weights.toUpperCase()} weights without Blackwell: ${name} runs them with weight-only kernels, slower; support to be checked per architecture`);
  if (kvFp8 && gpu.cc < 8.9) warnings.push('fp8 cache on Ampere: not every attention backend supports it (to be checked with the engine version)');
  if (groupSize(layout) > 1 && link === 'pcie-host')
    warnings.push(
      tp > 1
        ? `TP ${tp} on cards without NVLink or P2P: every all-reduce goes through CPU memory over PCIe (slower; needs x16 slots and an adequate power supply)`
        : `${groupSize(layout)} consumer cards in a pipeline: needs PCIe slots and an adequate power supply`,
    );
  else if (gpu.link === 'nvlink' && link !== 'nvlink' && groupSize(layout) > 1)
    warnings.push(
      tp > 1 && tp <= nvlinkDomain(gpu)
        ? `NVLink connects only ${nvlinkDomain(gpu)} of these cards: each TP ${tp} group stays on NVLink, the ${pp} pipeline stages talk over PCIe`
        : `${groupSize(layout)} GPUs in one group, but NVLink connects only ${nvlinkDomain(gpu)} of these cards: the group talks over PCIe (slower)`,
    );
  else if (tp > 2 && link === 'pcie-p2p') warnings.push(`TP ${tp} over PCIe (no NVLink): all-reduces slower than over NVLink`);
  return { blockers, warnings };
}

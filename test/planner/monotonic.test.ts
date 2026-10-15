// A stronger card is never planned worse: when card A has at least the memory, bandwidth, compute, links and GPUs per
// server of card B of the same architecture, A's best layout is never a worse verdict than B's and never serves fewer
// users. (Review 3 found an H200 SXM "does not fit" where the H200 NVL fitted, because only the NVL got pipeline layouts.)
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { hardwareFileSchema } from '../../src/contract/data';
import { DEFAULT_WORKLOAD } from '../../src/planner/capacity';
import { gpuFromSpec, layoutsFor, nvlinkDomain, serverSize, soldAsCards, type Gpu } from '../../src/planner/hardware';
import { planLayout, rankRows, type PlanOptions, type PlanRow } from '../../src/planner/plan';
import { catalogModel } from '../helpers/catalog';

const hw = hardwareFileSchema.parse(JSON.parse(readFileSync(path.join('data', 'hardware.json'), 'utf8')));
const cards = hw.gpus.map(gpuFromSpec).filter(soldAsCards);

const atLeast = (a: number | null, b: number | null) => b === null || (a !== null && a >= b);
/** Within 1 %: the same memory measured on one card and derived on another, or 4,800 vs 4,813 GB/s, count as equal. */
const near = (a: number, b: number) => a >= 0.99 * b;
/** A has everything B has, on the same major compute capability and at least B's minor one (the same kernels). */
function dominates(a: Gpu, b: Gpu): boolean {
  return (
    a.id !== b.id &&
    Math.floor(a.cc) === Math.floor(b.cc) &&
    a.cc >= b.cc &&
    near(a.cudaMiB.min, b.cudaMiB.min) &&
    near(a.cudaMiB.max, b.cudaMiB.max) &&
    near(a.bandwidthGBs, b.bandwidthGBs) &&
    atLeast(a.tflops.bf16, b.tflops.bf16) &&
    atLeast(a.tflops.fp8, b.tflops.fp8) &&
    atLeast(a.tflops.fp4, b.tflops.fp4) &&
    nvlinkDomain(a) >= nvlinkDomain(b) &&
    a.linkGBs >= b.linkGBs &&
    serverSize(a) >= serverSize(b)
  );
}
const pairs = cards.flatMap((a) => cards.filter((b) => dominates(a, b)).map((b) => [a, b] as const));

/** The buyer's order of a row: fits and serves someone, fits, at the limit, does not fit, does not start. */
const tier = (r: PlanRow) => (r.verdict === 'fits' ? ((r.capacity?.reliable ?? 0) >= 1 ? 0 : 1) : { borderline: 2, 'does-not-fit': 3, impossible: 4 }[r.verdict]);
const best = (rows: PlanRow[]) => ({ tier: Math.min(...rows.map(tier)), users: Math.max(0, ...rows.map((r) => (r.verdict === 'fits' ? (r.capacity?.reliable ?? 0) : 0))) });

describe('a stronger card is never planned worse', { timeout: 300_000 }, () => {
  it('the catalog has such pairs (H200 SXM over H200 NVL, B300 over B200, SXM over PCIe)', () => {
    const names = pairs.map(([a, b]) => `${a.id} > ${b.id}`);
    expect(names).toEqual(expect.arrayContaining(['h200-sxm-141gb > h200-nvl-141gb', 'b300-sxm > b200-sxm-180gb', 'h100-sxm5-80gb > h100-pcie-80gb']));
  });

  it.each([
    // MLA at 1M (the case review 3 found), a MoE in MXFP4, a dense model, each on both engines.
    ['zai-org/GLM-5.3', 1_048_576],
    ['openai/gpt-oss-120b', 131_072],
    ['Qwen/Qwen3.8-27B', 262_144],
  ] as const)('%s at %i tokens, vLLM and SGLang', async (repo, tokens) => {
    const { model } = await catalogModel(repo);
    for (const engine of ['vllm', 'sglang'] as const) {
      const opts: PlanOptions = { engine, kvCache: 'fp8', textOnly: false, workload: { ...DEFAULT_WORKLOAD, maxContext: tokens, typicalContext: tokens / 2 } };
      const plans = new Map(cards.map((g) => [g.id, best(rankRows(layoutsFor([g]).map((l) => planLayout(model, l, opts))))]));
      const worse = pairs
        .map(([a, b]) => [a.id, b.id, plans.get(a.id), plans.get(b.id)] as const)
        .filter(([, , pa, pb]) => pa && pb && (pa.tier > pb.tier || pa.users < pb.users))
        .map(([a, b, pa, pb]) => `${engine}: ${a} ${JSON.stringify(pa)} < ${b} ${JSON.stringify(pb)}`);
      expect(worse).toEqual([]);
    }
  });
});

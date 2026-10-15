import { describe, expect, it } from 'vitest';
import {
  cardLineSchema,
  cardLines,
  contextValidity,
  errorPct,
  figuresOf,
  figuresSchema,
  generalAnswer,
  closestRow,
  nobodyReason,
  fullySupported,
  gpuFamily,
  hardwareLabel,
  layoutShort,
  serversNeeded,
  serversOfFigures,
  byCost,
  ktok,
  largestAttempt,
  MEASURED_RUNS,
  measure,
  servesNobody,
  shortGpuName,
  speedPrecision,
  type FigureContext,
} from '../../src/cli/figures';
import { PUBLIC_POINTS, type SpeedBasis } from '../../src/planner/calibration';
import { layoutsFor } from '../../src/planner/hardware';
import { planLayout, rankRows, type PlanOptions, type PlanRow } from '../../src/planner/plan';
import { at, gpu, GPUS, opts, qwen } from './fixtures';

const ctxOf = (over: Partial<PlanOptions> = {}, repo = 'Qwen/Qwen3-8B', maxContext = 32_768): FigureContext => ({
  model: qwen,
  opts: opts(maxContext, { kvCache: 'fp8', ...over }),
  repo,
  verification: qwen.verification,
});
const ctx = ctxOf();
const plan = (c: FigureContext, id: string, tp = 1, pp = 1) => planLayout(qwen, at(id, tp, pp), c.opts);
type Cap = NonNullable<PlanRow['capacity']>;
const withCap = (r: PlanRow, over: Partial<Cap>): PlanRow => ({ ...r, capacity: { ...(r.capacity as Cap), ...over } });
const withBasis = (r: PlanRow, over: Partial<SpeedBasis>): PlanRow => ({ ...r, speedBasis: { ...(r.speedBasis as SpeedBasis), ...over } });

describe('names and numbers', () => {
  it('short card names for tables', () => {
    const names = Object.fromEntries(GPUS.map((g) => [g.id, shortGpuName(g.name)]));
    expect(names['rtx-5090-32gb']).toBe('RTX 5090');
    expect(names['rtx-6000-ada-48gb']).toBe('RTX 6000 Ada');
    expect(names['rtx-pro-6000-blackwell-workstation-96gb']).toBe('RTX PRO 6000 WS');
    expect(names['rtx-pro-6000-blackwell-max-q-96gb']).toBe('RTX PRO 6000 Max-Q');
    expect(names['rtx-pro-6000-blackwell-server-96gb']).toBe('RTX PRO 6000 Server');
    expect(names['gh200-144gb']).toBe('GH200 144GB');
    expect(names['b200-sxm-180gb']).toBe('B200 SXM');
    expect(Math.max(...Object.values(names).map((n) => n.length))).toBeLessThanOrEqual(20);
  });

  it('the hardware: count, card and split (tensor, pipeline, copies)', () => {
    const l = (id: string, count: number, tp: number, pp = 1, replicas?: number) => hardwareLabel({ layout: { gpu: gpu(id), count, tp, pp, ...(replicas ? { replicas } : {}) } });
    expect(l('rtx-5090-32gb', 1, 1)).toBe('RTX 5090');
    expect(l('rtx-4090-24gb', 2, 2)).toBe('2× RTX 4090 (1 model on 2)');
    expect(l('rtx-4090-24gb', 2, 1, 2)).toBe('2× RTX 4090 (split in 2 stages)');
    expect(l('h200-nvl-141gb', 8, 4, 2)).toBe('8× H200 NVL 141GB (split in 2 stages of 4)');
    expect(l('h100-sxm5-80gb', 8, 2, 1, 4)).toBe('8× H100 SXM5 80GB (4 copies × 2)');
    expect(l('rtx-5090-32gb', 4, 1, 1, 4)).toBe('4× RTX 5090 (4 copies × 1)');
    expect(hardwareLabel({ layout: { gpu: gpu('rtx-4090-24gb'), count: 4, tp: 1, pp: 2, replicas: 2 } }, true)).toBe('4x RTX 4090 (2 copies x 2 stages)');
  });

  it('layouts in a few characters, servers, card families', () => {
    const s = (count: number, tp: number, pp = 1, replicas?: number) => layoutShort({ count, tp, pp, ...(replicas ? { replicas } : {}) });
    expect([s(1, 1), s(4, 4), s(8, 2, 1, 4), s(2, 1, 2), s(4, 1, 2, 2), s(8, 4, 2)]).toEqual(['', '1×4', '4×2', '2 stages', '2×2 stages', '2 stages of 4']);
    expect(layoutShort({ count: 8, tp: 8, pp: 1 }, true)).toBe('1x8');
    // The planner's servers when the figures carry them; packed reports without them: 8 GPUs per server.
    expect([serversOfFigures({ count: 1 }), serversOfFigures({ count: 8 }), serversOfFigures({ count: 16 }), serversOfFigures({ count: 8, servers: 2 })]).toEqual([1, 1, 2, 2]);
    expect(GPUS.map((g) => gpuFamily(shortGpuName(g.name)))).toEqual(expect.arrayContaining(['RTX PRO 6000', 'H100', 'H200', 'GH200', 'A100', 'B200', 'L40S', 'RTX 6000 Ada']));
    expect(gpuFamily('H100 SXM5 80GB')).toBe('H100');
  });

  it('tokens: k and M', () => {
    expect([ktok(500), ktok(32_768), ktok(212_848), ktok(1_048_576), ktok(1_572_864), ktok(5000), ktok(1536), ktok(1500), ktok(1024)]).toEqual(['500', '32k', '208k', '1M', '1.5M', '4.9k', '1.5k', '1.5k', '1k']);
  });

  it('the error: half the range over its centre, 0 for an exact value or a zero range', () => {
    expect(errorPct(90, 110)).toBe(10);
    expect(errorPct(5, 5)).toBe(0);
    expect(errorPct(0, 0)).toBe(0);
    expect(measure(65, 83, { kind: 'calibrated', basis: 'x' })).toEqual({ value: 74, min: 65, max: 83, precision: { kind: 'calibrated', basis: 'x', errorPct: (18 / 148) * 100 } });
  });
});

describe('what the speed rests on', () => {
  it('the measured runs are full models on the RTX 5090', () => {
    expect(MEASURED_RUNS.map((r) => r.repo)).toEqual(['Qwen/Qwen3-8B']);
  });
});

describe('the accuracy of the speed', () => {
  const r5090 = plan(ctx, 'rtx-5090-32gb');
  const range = (r: PlanRow) => (r.capacity as Cap).single.tokS;
  it('measured: this model on this card with this engine, cache and context', () => {
    expect(speedPrecision(ctx, r5090, range(r5090))).toMatchObject({ kind: 'measured', points: 18, basis: expect.stringMatching(/^measured on RTX 5090 with vLLM 0\.31\.0: Qwen3-8B BF16 .*; 18 points, on Windows \/ WSL 2$/) });
  });

  it('calibrated: the card is measured, the model is not (another repo, cache, count or a longer context)', () => {
    expect(speedPrecision(ctxOf({}, 'Qwen/Qwen3-14B'), r5090, range(r5090)).kind).toBe('calibrated');
    const full = ctxOf({ kvCache: 'auto' });
    expect(speedPrecision(full, plan(full, 'rtx-5090-32gb'), range(r5090)).kind).toBe('calibrated');
    const two = plan(ctx, 'rtx-5090-32gb', 2);
    expect(speedPrecision(ctx, two, range(two)).kind).toBe('calibrated');
    const long = ctxOf({}, 'Qwen/Qwen3-8B', 40_960);
    expect(speedPrecision(long, plan(long, 'rtx-5090-32gb'), range(r5090)).kind).toBe('calibrated');
    // Public benchmarks on this card within the context they cover.
    const short = ctxOf({}, 'x/y', 8192);
    const h100 = plan(short, 'h100-sxm5-80gb');
    expect(speedPrecision(short, h100, range(h100))).toMatchObject({ kind: 'calibrated', points: PUBLIC_POINTS.vllm, basis: `${h100.speedBasis?.base as string}, vLLM 0.31.0` });
  });

  it('estimated: beyond the context verified, calibrated on other cards, a combination not verified', () => {
    const h100 = plan(ctx, 'h100-sxm5-80gb');
    expect(speedPrecision(ctx, h100, { min: 10, max: 11 })).toMatchObject({ kind: 'estimated', reason: 'extrapolated beyond the 8k tokens verified' });
    const short = ctxOf({}, 'x/y', 8192);
    const l40s = plan(short, 'l40s-48gb');
    expect(speedPrecision(short, l40s, range(l40s))).toMatchObject({ kind: 'estimated', reason: 'calibrated on other cards' });
    expect(speedPrecision(short, withBasis(l40s, { kind: 'estimated', base: 'estimated: dense BF16 not verified with vLLM' }), { min: 10, max: 11 })).toMatchObject({ kind: 'estimated', reason: 'BF16 dense not verified with vLLM: range doubled' });
    const noFamily = { ...short, model: { ...qwen, family: undefined } } as unknown as FigureContext;
    expect(speedPrecision(noFamily, withBasis(l40s, { kind: 'estimated', base: 'estimated: dense BF16 not verified with vLLM' }), { min: 10, max: 11 }).reason).toBe('BF16 dense not verified with vLLM: range doubled');
    // A pipeline on a calibrated combination: estimated because no pipeline run was measured, and only that.
    const staged = { ...withBasis(l40s, { kind: 'estimated' }), layout: { ...l40s.layout, count: 2, tp: 1, pp: 2 } } as PlanRow;
    expect(speedPrecision(short, staged, { min: 10, max: 11 })).toMatchObject({ kind: 'estimated', reason: 'pipeline speed not measured: range widened by half' });
  });

  it('not precise enough: a caveat, a range of ±50 % or more; SGLang on a dense model', () => {
    const h100 = plan(ctx, 'h100-sxm5-80gb');
    expect(speedPrecision(ctx, { ...h100, speedNote: 'INT4 MoE' }, range(h100))).toMatchObject({ kind: 'imprecise', reason: 'INT4 MoE; extrapolated beyond the 8k tokens verified' });
    expect(speedPrecision(ctx, h100, { min: 5, max: 15 })).toMatchObject({ kind: 'imprecise', reason: 'a range of ±50% is too wide to decide on; extrapolated beyond the 8k tokens verified' });
    const sg = ctxOf({ engine: 'sglang' }, 'x/y', 4096);
    const sgRow = plan(sg, 'h100-sxm5-80gb');
    expect(speedPrecision(sg, sgRow, range(sgRow))).toMatchObject({ kind: 'imprecise', points: PUBLIC_POINTS.sglang });
  });

  it('the context range the calibration covers', () => {
    expect(contextValidity(ctx, r5090.speedBasis as SpeedBasis)).toBe('context calibrated up to 1M');
    expect(contextValidity(ctx, plan(ctx, 'h100-sxm5-80gb').speedBasis as SpeedBasis)).toBe('context extrapolated beyond the 8k verified');
  });
});

describe('figures of a row', () => {
  it('memory with its range, exact weights and KV cache, estimated fixed costs, the engine’s share, the margin', () => {
    const r = plan(ctx, 'rtx-5090-32gb');
    const f = figuresOf(ctx, r);
    expect(figuresSchema.parse(f)).toEqual(f);
    expect(f).toMatchObject({ hardware: 'RTX 5090', gpuId: 'rtx-5090-32gb', gpu: 'RTX 5090', segment: 'consumer', count: 1, tp: 1, pp: 1, replicas: 1, verdict: 'fits', memoryFits: true, servers: 1 });
    // The card's memory on one scale: what CUDA reports on the smallest system (the engine's usable memory, as the
    // verdict counts it), in GiB like every other memory figure; nvidia-smi's total plays no part.
    expect(f.cardGiB).toBe(Math.round((r.layout.gpu.cudaMiB.min / 1024) * 10) / 10);
    expect(figuresOf(ctx, { ...r, layout: { ...r.layout, gpu: { ...r.layout.gpu, totalMiB: 1 } } }).cardGiB).toBe(f.cardGiB);
    expect(f.compat).toEqual({ status: 'ok', notes: [] });
    expect(f.weights.precision).toMatchObject({ kind: 'exact', errorPct: 0 });
    expect(f.weights.precision.basis).toContain('verified');
    expect(f.kv.min).toBe(f.kv.max);
    expect(f.kv.precision.basis).toBe('blocks and layer groups of the vLLM 0.31.0 KV cache');
    expect(f.memory.min).toBeCloseTo(f.weights.value + f.fixed.min + f.kv.value, 6);
    expect(f.memory.max).toBeCloseTo(f.weights.value + f.fixed.max + f.kv.value, 6);
    expect(f.memory.precision.kind).toBe('estimated');
    expect(f.engineGiB.max).toBeCloseTo(r.memory.best.processMiB.max / 1024, 6);
    expect(f.validity).toBe('context calibrated up to 1M');
  });

  it('speed, users, reading and waiting with their accuracy; the roofline from the planner', () => {
    const r = plan(ctx, 'rtx-5090-32gb');
    const f = figuresOf(ctx, r);
    expect(f.tokS?.precision.kind).toBe('measured');
    expect(f.users?.guaranteed).toBeGreaterThan(0);
    expect(f.users?.memoryMax).toBeGreaterThanOrEqual(f.users?.likely as number);
    expect(f.read?.min).toBeLessThan(f.read?.max as number);
    expect(f.wait?.max).toBeGreaterThan(0);
    const roof = f.roofline as NonNullable<typeof f.roofline>;
    const limits = r.roofline as NonNullable<PlanRow['roofline']>;
    expect(roof.tokSLimit).toBe(limits.decodeTokS);
    expect(roof.readLimitS).toBe(limits.fullPrefillMs / 1000);
    expect(roof.tokSShare).toBeCloseTo((f.single?.value as number) / limits.decodeTokS, 9);
    expect(roof.readShare).toBeLessThan(1);
    expect(roof.warning).toBeUndefined();
  });

  it('every row of the catalog stays under its physical limits', () => {
    for (const r of layoutsFor(GPUS).map((l) => planLayout(qwen, l, ctx.opts))) {
      if (!r.capacity) continue;
      const f = figuresOf(ctx, r);
      expect(f.roofline?.tokSShare, f.hardware).toBeLessThan(1);
      expect(f.roofline?.readShare, f.hardware).toBeLessThan(1);
    }
  });

  it('users limited by memory: the error of the KV room; by speed: the speed accuracy', () => {
    const f = figuresOf(ctx, plan(ctx, 'rtx-3090-24gb'));
    expect(f.requests?.limitedBy).toBe('memory');
    expect(f.users?.precision).toMatchObject({ kind: 'estimated', basis: 'KV cache memory limit (85% safety margin)', reason: '10% of users active at once' });
    const h = figuresOf(ctx, plan(ctx, 'rtx-5090-32gb', 2));
    expect(h.requests?.limitedBy).not.toBe('memory');
    expect(h.users?.precision).toMatchObject({ kind: 'calibrated', reason: '10% of users active at once' });
    const l = figuresOf(ctx, plan(ctx, 'l40s-48gb'));
    expect(l.users?.precision.reason).toBe('calibrated on other cards; extrapolated beyond the 8k tokens verified; 10% of users active at once');
  });

  it('weights not verified or with a measured gap; SGLang', () => {
    const gap = figuresOf({ ...ctx, verification: { status: 'gap', gapShare: 0.026, gapFixedMiB: 0, evidence: [] } }, plan(ctx, 'h100-sxm5-80gb'));
    expect(gap.weights.precision.reason).toBe('the engine measured up to +2.6% for this format: counted in the fixed costs');
    const un = figuresOf({ ...ctx, verification: { status: 'unverified', gapShare: 0.05, gapFixedMiB: 0, evidence: [] } }, plan(ctx, 'h100-sxm5-80gb'));
    expect(un.weights.precision.reason).toBe('format not yet verified by a measurement: 5% margin in the fixed costs');
    const sg = ctxOf({ engine: 'sglang' });
    expect(figuresOf(sg, plan(sg, 'h100-sxm5-80gb')).kv.precision.basis).toBe('blocks and layer groups of the SGLang 0.5.21 KV cache');
  });

  it("the engine's checks and caveats", () => {
    const blocked = figuresOf(ctx, { ...plan(ctx, 'h100-sxm5-80gb'), check: { blockers: ['32 heads'], warnings: ['caveat'] }, verdict: 'impossible' });
    expect(blocked.compat).toEqual({ status: 'does-not-start', notes: ['32 heads', 'caveat'] });
    expect(figuresOf(ctx, plan(ctx, 'rtx-3090-24gb')).compat.status).toBe('warnings');
  });

  it('no speed on a card without bandwidth or TFLOPS: memory only', () => {
    const f = figuresOf(ctx, plan(ctx, 'rtx-pro-6000-blackwell-server-96gb'));
    expect(f.users).toBeUndefined();
    expect(f.tokS).toBeUndefined();
    expect(f.validity).toBe('speed not estimable');
    expect(f.speedNote).toBeDefined();
  });

  it('above 90 % of a physical limit: a suspect estimate', () => {
    const r = plan(ctx, 'h100-sxm5-80gb');
    const fast = withCap(r, { single: { ...(r.capacity as Cap).single, tokS: { min: 1e6, max: 1e6 }, fullPrefillMs: { min: 1, max: 1 } } });
    expect(figuresOf(ctx, fast).roofline?.warning).toMatch(/^generation = \d+% of the datasheet bandwidth, reading = \d+% of the datasheet TFLOPS: above 90% of the physical limit, suspect estimate$/);
    const read = withCap(r, { single: { ...(r.capacity as Cap).single, fullPrefillMs: { min: 1, max: 1 } } });
    expect(figuresOf(ctx, read).roofline?.warning).toMatch(/^reading = /);
    // A card whose TFLOPS are derived (no exact NVIDIA figure): said so, and kept in the figures.
    const derived = { ...read, layout: { ...read.layout, gpu: { ...read.layout.gpu, tflopsDerived: true as const } } };
    expect(figuresOf(ctx, derived).roofline).toMatchObject({ tflopsDerived: true, warning: expect.stringMatching(/^reading = \d+% of the derived TFLOPS/) as string });
    expect(figuresOf(ctx, read).roofline?.tflopsDerived).toBeUndefined();
  });

  it('near saturation: prompts take most of the GPU, or the users asked are close to the guaranteed', () => {
    const r = plan(ctx, 'h100-sxm5-80gb');
    const c = r.capacity as Cap;
    const busy = withCap(r, { atReliable: { ...c.atReliable, prefillShare: { min: 0.8, max: 0.9 } } });
    expect(figuresOf(ctx, busy).saturation).toBe('near saturation: reading prompts takes 90% of the GPU time, larger error');
    const calm = withCap(r, { atReliable: { ...c.atReliable, prefillShare: { min: 0.1, max: 0.2 } }, reliable: 20 });
    expect(figuresOf(ctx, calm, { concurrent: 18 }).saturation).toBe('near saturation: 18 concurrent requests of 20 guaranteed (90%), larger error');
    expect(figuresOf(ctx, calm, { concurrent: 10 }).saturation).toBeUndefined();
    expect(figuresOf(ctx, calm, { concurrent: 21 }).saturation).toBeUndefined();
    expect(figuresOf(ctx, withCap(calm, { reliable: 0 }), { concurrent: 0 }).saturation).toBeUndefined();
    expect(figuresOf(ctx, calm).saturation).toBeUndefined();
  });

  it('what limits the users: memory, speed, the first-token wait, nothing', () => {
    const r = plan(ctx, 'h100-sxm5-80gb');
    for (const [by, en] of [['speed', 'speed'], ['wait', 'wait'], ['none', 'none']] as const) expect(figuresOf(ctx, withCap(r, { limitedBy: by })).requests?.limitedBy).toBe(en);
  });
});

describe('the general answer and the overview per card', () => {
  const rows = rankRows(layoutsFor(GPUS).map((l) => planLayout(qwen, l, ctx.opts)));
  it('the fewest GPUs that fit and serve someone, and the other market; with users, the option that serves them', () => {
    const g = generalAnswer(rows, { total: 200, concurrent: 20, active: 0.1 });
    expect(hardwareLabel(g.fit?.row as PlanRow)).toBe('RTX 3090');
    expect(g.fit?.alternative?.layout.gpu.segment).toBe('datacenter');
    expect((g.users?.row.capacity as Cap).reliable).toBeGreaterThanOrEqual(20);
    expect(g.attempt).toBeUndefined();
    expect(g.users?.servers).toBe(1);
    expect(g.users?.usersPerServer).toBeGreaterThanOrEqual(200);
    // More users than one server holds: more servers of the cheapest layout, never none; they cover everyone.
    const many = generalAnswer(rows, { total: 1e7, concurrent: 1e6, active: 0.1 }).users;
    expect(many?.servers).toBeGreaterThan(1);
    expect((many?.servers as number) * (many?.usersPerServer as number)).toBeGreaterThanOrEqual(1e7);
    expect(many?.alternativeServers).toBeGreaterThan(1);
    expect((many?.alternativeServers as number) * (many?.alternativeUsersPerServer as number)).toBeGreaterThanOrEqual(1e7);
    expect(generalAnswer(rows.map((r) => (r.capacity ? withCap(r, { reliable: 0 }) : r)), { total: 200, concurrent: 20 }).users).toBeUndefined();
  });

  it('servers needed: ceil(users / users per server), servers of the layout; layouts that serve nobody or do not fit excluded', () => {
    const one = rows.find((r) => r.layout.gpu.id === 'h100-sxm5-80gb' && r.layout.count === 1) as PlanRow;
    // 10% active: 5 guaranteed requests serve 50 users.
    const at = (n: number) => withCap(one, { reliable: n });
    const u = (total: number) => ({ total, concurrent: Math.ceil(total / 10), active: 0.1 });
    expect(serversNeeded(at(5), u(200))).toEqual({ servers: 4, usersPerServer: 50, likelyUsersPerServer: expect.any(Number) as number });
    expect(serversNeeded(at(5), u(201))?.servers).toBe(5);
    expect(serversNeeded(at(5), u(50))?.servers).toBe(1);
    expect(serversNeeded(at(5), u(1))?.servers).toBe(1);
    // Without --active: the share is concurrent ÷ total.
    expect(serversNeeded(at(5), { total: 200, concurrent: 20 })?.servers).toBe(4);
    expect(serversNeeded(at(0), u(200))).toBeUndefined();
    expect(serversNeeded({ ...at(5), verdict: 'borderline' }, u(200))).toBeUndefined();
    expect(serversNeeded({ ...one, capacity: undefined }, u(200))).toBeUndefined();
    // 16 GPUs are two servers of 8: the guaranteed requests split over them.
    const sixteen = { ...at(10), layout: { ...one.layout, count: 16 } };
    expect(serversNeeded(sixteen, u(200))).toMatchObject({ servers: 4, usersPerServer: 50 });
    // Cheapest first: fewest GPUs in total, then fewest servers; nobody served: left out.
    const two = (n: number) => ({ ...withCap(one, { reliable: n }), layout: { ...one.layout, count: 2 } });
    expect(byCost([at(0), two(7), at(5)], u(200)).map((x) => [x.row.layout.count, x.plan.servers])).toEqual([[1, 4], [2, 3]]);
    // Same GPUs in total (4 × 1 and 2 × 2): fewer servers first.
    expect(byCost([at(5), two(12)], u(200)).map((x) => [x.row.layout.count, x.plan.servers])).toEqual([[2, 2], [1, 4]]);
    expect(byCost([at(5), withCap(one, { reliable: 10 })], u(200)).map((x) => x.plan.servers)).toEqual([2, 4]);
    // A pipeline the matrix does not confirm comes after every verified layout, even one that takes more GPUs.
    const pipe = { ...two(40), layout: { ...one.layout, count: 1, tp: 1, pp: 2 }, compat: { details: [{ feature: 'p:pp', status: 'partial' }] } } as unknown as PlanRow;
    expect(byCost([pipe, two(7)], u(200)).map((x) => x.row)).toEqual([two(7), pipe].map((r) => expect.objectContaining({ layout: r.layout })));
    expect(closestRow([pipe, two(7)])?.layout).toEqual(two(7).layout);
  });

  it('the alternative: another card of the same market when the other market has none', () => {
    const dc = rows.filter((r) => r.layout.gpu.segment === 'datacenter');
    const g = generalAnswer(dc);
    expect(g.fit?.alternative?.layout.gpu.id).not.toBe(g.fit?.row.layout.gpu.id);
    const one = dc.filter((r) => r.layout.gpu.id === 'h100-sxm5-80gb' && r.layout.count === 1);
    expect(generalAnswer(one).fit?.alternative).toBeUndefined();
  });

  it('a known speed first: a card whose speed cannot be estimated is the answer only when nothing else fits', () => {
    const slow = rows.map((r) => (r.capacity ? withCap(r, { reliable: 0 }) : r));
    const g = generalAnswer(slow);
    expect(g.fit?.row.capacity).toBeDefined();
    const onlyUnknown = rows.filter((r) => !r.capacity && r.verdict === 'fits');
    expect(generalAnswer(onlyUnknown).fit?.row.layout.gpu.id).toBe('rtx-pro-6000-blackwell-server-96gb');
  });

  it('nothing runs but the memory fits: the layout the engine refuses, never "does not fit"', () => {
    const refused = rows.map((r) => ({ ...r, verdict: 'impossible' as const, check: { blockers: ['no kernel'], warnings: [] } }));
    const g = generalAnswer(refused);
    expect(g.fit).toBeUndefined();
    expect(g.attempt).toBeUndefined();
    expect(g.blocked?.check.blockers[0]).toBe('no kernel');
  });

  it('nobody served: the first that fits; only borderline ones; nothing fits: the closest attempt', () => {
    const slow = rows.filter((r) => r.capacity).map((r) => withCap(r, { reliable: 0 }));
    // Nobody served anywhere: the closest layout (the most likely users, then the fastest), as the rows show it.
    expect(generalAnswer(slow).fit?.row).toBe(closestRow(slow.filter((r) => r.verdict === 'fits')));
    expect(closestRow([])).toBeUndefined();
    const noCap = { ...(slow[0] as PlanRow), capacity: undefined };
    expect(closestRow([noCap, slow[0] as PlanRow])).toBe(slow[0]);
    expect(closestRow([noCap, { ...noCap }])).toBe(noCap);
    expect(servesNobody(slow[0] as PlanRow)).toBe(true);
    const edge = rows.map((r) => ({ ...r, verdict: r.verdict === 'fits' ? ('borderline' as const) : r.verdict }));
    expect(generalAnswer(edge).fit?.row.verdict).toBe('borderline');
    const none = rows.map((r) => ({ ...r, verdict: 'does-not-fit' as const }));
    const g = generalAnswer(none);
    expect(g.fit).toBeUndefined();
    expect(g.attempt?.layout.count).toBe(8);
    expect(largestAttempt(none)).toBe(g.attempt);
  });

  it('one line per card in the catalog order; this PC marked; the fewest GPUs that fit', () => {
    const lines = cardLines(ctx, rows, GPUS, undefined, 'rtx-5090-32gb');
    expect(lines.map((l) => l.line.gpuId)).toEqual(GPUS.map((g) => g.id));
    for (const l of lines) expect(cardLineSchema.parse(l.line)).toEqual(l.line);
    const by = (id: string) => (lines.find((l) => l.line.gpuId === id) as (typeof lines)[number]).line;
    expect(by('rtx-5090-32gb')).toMatchObject({ status: 'fits', minGpus: 1, maxGpus: 8, local: true });
    expect(by('rtx-5080-16gb')).toMatchObject({ status: 'fits', minGpus: 2 });
    expect(by('rtx-4090-24gb').local).toBeUndefined();
    expect(by('rtx-pro-6000-blackwell-server-96gb')).toMatchObject({ status: 'no-speed' });
    expect(by('rtx-pro-6000-blackwell-server-96gb').note).toBeUndefined();
    // Cards not planned are left out.
    expect(cardLines(ctx, rows.filter((r) => r.layout.gpu.id === 'l4-24gb'), GPUS)).toHaveLength(1);
  });

  it('with users: the cheapest layout of each card with the servers it takes; too slow for anyone', () => {
    const lines = cardLines(ctx, rows, GPUS, { total: 20000, concurrent: 2000, active: 0.1 });
    const by = (id: string) => (lines.find((l) => l.line.gpuId === id) as (typeof lines)[number]).line;
    const l4 = by('l4-24gb');
    expect(l4.status).toBe('fits');
    const mine = rows.filter((r) => r.layout.gpu.id === 'l4-24gb');
    expect(l4.serversNeeded).toBe(byCost(mine, { total: 20000, concurrent: 2000, active: 0.1 })[0]?.plan.servers);
    expect(l4.serversNeeded).toBeGreaterThan(1);
    const slow = rows.filter((r) => r.layout.gpu.id === 'l4-24gb').map((r) => (r.capacity ? withCap(r, { reliable: 0 }) : r));
    const short = cardLines(ctx, slow, GPUS, { total: 200, concurrent: 20 })[0]?.line;
    expect(short).toMatchObject({ status: 'short', note: `${nobodyReason(slow.find((r) => r.capacity)?.capacity?.limitedBy, '20 tok/s')}, even with 8 GPUs in one server` });
    expect(short?.figures.hardware).toBe(cardLines(ctx, slow, GPUS)[0]?.line.figures.hardware);
    expect(short?.serversNeeded).toBeUndefined();
    // A card without a speed: no verdict on the users.
    expect(by('rtx-pro-6000-blackwell-server-96gb')).toMatchObject({ status: 'no-speed' });
  });

  it('too slow: fits but serves nobody (with more GPUs or never)', () => {
    const one = rows.filter((r) => r.layout.gpu.id === 'h100-sxm5-80gb').map((r) => (r.layout.count === 1 ? withCap(r, { reliable: 0 }) : r));
    // It serves someone with more GPUs: that layout is the row, the smaller fit in the note.
    expect(cardLines(ctx, one, [gpu('h100-sxm5-80gb')])[0]?.line).toMatchObject({ status: 'fits', note: `with 1 GPU the model fits, but ${nobodyReason(one.find((r) => r.layout.count === 1)?.capacity?.limitedBy, '20 tok/s')}: 2 GPUs serve someone`, servingGpus: 2 });
    // One text for the binding limit everywhere: memory, the first-token wait, the writing speed.
    expect(nobodyReason('memory', '20 tok/s')).toBe('the memory left after the model holds no full-length request with its safety margin');
    expect(nobodyReason('wait', '20 tok/s')).toBe('nobody gets the first words within the wait allowed (--max-wait)');
    expect(nobodyReason('speed', '20 tok/s')).toBe('nobody is guaranteed 20 tok/s of writing');
    expect(nobodyReason(undefined, '20 tok/s')).toBe('nobody is guaranteed 20 tok/s of writing');
    const never = one.map((r) => withCap(r, { reliable: 0 }));
    expect(cardLines(ctx, never, [gpu('h100-sxm5-80gb')])[0]?.line).toMatchObject({ status: 'slow', note: `${nobodyReason(never[0]?.capacity?.limitedBy, '20 tok/s')}, even with 8 GPUs in one server` });
    // At the limit of memory and too slow: both causes in the note; the server size of the card on the line.
    const edge = never.map((r) => ({ ...r, verdict: 'borderline' as const }));
    const l = cardLines(ctx, edge, [gpu('h100-sxm5-80gb')])[0]?.line;
    expect(l?.note).toBe(`at the limit: ${edge[0]?.memory.worst.reasons[0] ?? 'within the uncertainty of the fixed costs'}; and ${nobodyReason(edge[0]?.capacity?.limitedBy, '20 tok/s')}`);
    expect(l).toMatchObject({ status: 'at-limit', serverGpus: 8, serverSize: 8 });
  });

  it('the recommendation: fully supported first among layouts with as many GPUs; at the limit: the largest worst-case margin', () => {
    const partly = (r: PlanRow) => ({ ...r, check: { ...r.check, warnings: ['partly supported: open bug'] } });
    const first = generalAnswer(rows).fit?.row as PlanRow;
    const marked = rows.map((r) => (r === first ? partly(r) : r));
    const g = generalAnswer(marked);
    expect(g.fit?.row.layout.count).toBe(first.layout.count);
    expect(fullySupported(g.fit?.row as PlanRow)).toBe(true);
    // In users mode too: as many GPUs in all, fully supported first.
    const u = { total: 200, concurrent: 20, active: 0.1 };
    const usersFirst = generalAnswer(rows, u).users?.row as PlanRow;
    const same = byCost(rows, u).filter((x) => (x.plan.servers * x.row.layout.count) / Math.ceil(x.row.layout.count / 8) === (byCost(rows, u)[0]?.plan.servers ?? 0) * usersFirst.layout.count);
    if (same.length > 1) expect(fullySupported(generalAnswer(rows.map((r) => (r === usersFirst ? partly(r) : r)), u).users?.row as PlanRow)).toBe(true);
    // At the limit only: the fewest GPUs, then the largest margin at the worst.
    const mine = rows.filter((r) => r.layout.gpu.id === 'h100-sxm5-80gb').map((r) => ({ ...r, verdict: 'borderline' as const }));
    const n = Math.min(...mine.map((r) => r.layout.count));
    const fewest = mine.filter((r) => r.layout.count === n);
    const widest = [...fewest].sort((a, b) => b.marginMiB.min - a.marginMiB.min)[0] as PlanRow;
    expect(cardLines(ctx, [...mine].reverse(), [gpu('h100-sxm5-80gb')])[0]?.line.figures.hardware).toBe(hardwareLabel(widest));
  });

  it('the alternative: fully supported first; servers needed on every layout that serves the users', () => {
    const partly = (r: PlanRow) => ({ ...r, check: { ...r.check, warnings: ['partly supported: open bug'] } });
    const marked = rows.map((r) => (r.layout.gpu.segment === 'datacenter' && r.layout.gpu.id !== 'h100-sxm5-80gb' ? partly(r) : r));
    const g = generalAnswer(marked);
    expect(g.fit?.alternative?.layout.gpu.id).toBe('h100-sxm5-80gb');
    expect(fullySupported(g.fit?.alternative as PlanRow)).toBe(true);
    // Every option partly supported: one of them all the same.
    expect(generalAnswer(rows.map(partly)).fit?.alternative).toBeDefined();
    const one = rows.find((r) => r.layout.gpu.id === 'h100-sxm5-80gb' && r.layout.count === 1) as PlanRow;
    expect(figuresOf(ctx, withCap(one, { reliable: 5 }), { total: 200, concurrent: 20, active: 0.1 }).serversNeeded).toBe(4);
    expect(figuresOf(ctx, withCap(one, { reliable: 5 }), { total: 200, concurrent: 20 }).serversNeeded).toBe(4);
    expect(figuresOf(ctx, withCap(one, { reliable: 0 }), { total: 200, concurrent: 20 }).serversNeeded).toBeUndefined();
    expect(figuresOf(ctx, one, { concurrent: 20 }).serversNeeded).toBeUndefined();
    expect(figuresOf(ctx, one).readingShare).toBeGreaterThan(0);
  });

  it('borderline, the engine refuses to start, does not fit even with 8', () => {
    const mine = rows.filter((r) => r.layout.gpu.id === 'h100-sxm5-80gb');
    const edge = mine.map((r) => ({ ...r, verdict: 'borderline' as const }));
    expect(cardLines(ctx, edge, [gpu('h100-sxm5-80gb')])[0]?.line).toMatchObject({ status: 'at-limit', note: 'at the limit: within the uncertainty of the fixed costs' });
    const reason = edge.map((r) => ({ ...r, memory: { ...r.memory, worst: { ...r.memory.worst, reasons: ['the KV cache fits only if'] } } }));
    expect(cardLines(ctx, reason, [gpu('h100-sxm5-80gb')])[0]?.line.note).toBe('at the limit: the KV cache fits only if');
    const blocked = mine.map((r) => ({ ...r, verdict: 'impossible' as const, check: { blockers: ['40 heads do not divide by TP 3'], warnings: [] } }));
    expect(cardLines(ctx, blocked, [gpu('h100-sxm5-80gb')])[0]?.line).toMatchObject({ status: 'no-start', note: 'fits in memory but the engine does not start: 40 heads do not divide by TP 3' });
    const none = mine.map((r) => ({ ...r, verdict: 'does-not-fit' as const }));
    const l = cardLines(ctx, none, [gpu('h100-sxm5-80gb')])[0]?.line;
    expect(l).toMatchObject({ status: 'no-fit', maxGpus: 8 });
    expect(l?.minGpus).toBeUndefined();
    expect(l?.figures.count).toBe(8);
    // Every layout refused and too small: the largest attempt still.
    const worst = mine.map((r) => ({ ...r, verdict: 'impossible' as const, memory: { ...r.memory, best: { ...r.memory.best, verdict: 'no' as const } } }));
    expect(cardLines(ctx, worst, [gpu('h100-sxm5-80gb')])[0]?.line).toMatchObject({ status: 'no-fit' });
  });
});

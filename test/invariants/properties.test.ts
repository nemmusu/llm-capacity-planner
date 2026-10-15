/**
 * Properties the planner must keep for any catalog model and any parameters (fast-check, a fixed seed so a
 * failure replays): memory grows with the context, copies of a layout change nothing per GPU, no speed beats the
 * card's bandwidth, every range holds its value, guaranteed never exceeds likely, an engine that does not start
 * is never the answer, the memory items add up, and the text, CSV and JSON say the same numbers.
 * Only what the code really guarantees is asserted; each property's comment says why it must hold.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import fc from 'fast-check';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { catalog } from '../../src/cli/run';
import { figuresOf, type Figures, type Measure } from '../../src/cli/figures';
import { gb, int } from '../../src/cli/format';
import { CSV_HEADERS } from '../../src/cli/render';
import { DEFAULT_WORKLOAD, type Workload } from '../../src/planner/capacity';
import { layoutsFor, type Gpu, type GpuLayout } from '../../src/planner/hardware';
import { planLayout, type PlanOptions, type PlannerModel, type PlanRow } from '../../src/planner/plan';
import type { Verification } from '../../src/planner/weights-verification';
import { catalogModel, catalogPack, runCli, tempFolder } from '../helpers/catalog';

/** Catalog models of every kind: dense, MoE (MXFP4), hybrid, multimodal, MLA, FP8 and NVFP4 checkpoints. */
const REPOS = [
  'ibm-granite/granite-4.2-3b',
  'ibm-granite/granite-4.2-8b-fp8',
  'openai/gpt-oss-20b',
  'Qwen/Qwen3.5-9B',
  'google/gemma-4-31B-it',
  'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16',
  'inclusionAI/Ling-3.0-flash-int4',
  'nvidia/Qwen3.6-35B-A3B-NVFP4',
] as const;

type Loaded = PlannerModel & { family: string; verification: Verification };
const models = new Map<string, Loaded>();
const GPUS = catalog();
const cache = tempFolder('props-');

beforeAll(async () => {
  catalogPack();
  for (const repo of REPOS) models.set(repo, (await catalogModel(repo)).model);
}, 180_000);
afterAll(() => cache.cleanup());

/** A fixed seed: the same cases on every run (a failure prints its seed and path to replay it). */
const RUNS = { seed: 20261015, numRuns: 150 } as const;

const EPS = 1e-9;
const close = (a: number, b: number) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(a), Math.abs(b));

/** A workload anywhere in the command's accepted ranges. */
const workloadArb = (maxContext: fc.Arbitrary<number> = fc.integer({ min: 256, max: 1_048_576 })): fc.Arbitrary<Workload> =>
  fc.record({
    maxContext,
    typicalShare: fc.double({ min: 0.01, max: 1, noNaN: true }),
    outputTokens: fc.integer({ min: 1, max: 8192 }),
    minTokS: fc.double({ min: 0.1, max: 200, noNaN: true }),
    maxTtftMs: fc.double({ min: 50, max: 120_000, noNaN: true }),
    activeShare: fc.double({ min: 0.001, max: 1, noNaN: true }),
    cachedShare: fc.double({ min: 0, max: 0.99, noNaN: true }),
    ttftScales: fc.boolean(),
  }).map(({ typicalShare, ...w }) => ({ ...w, typicalContext: Math.max(1, Math.floor(w.maxContext * typicalShare)), safety: DEFAULT_WORKLOAD.safety }));

const optionsArb = (w: fc.Arbitrary<Workload> = workloadArb()): fc.Arbitrary<PlanOptions> =>
  fc
    .record({
      engine: fc.constantFrom('vllm' as const, 'sglang' as const),
      kvCache: fc.constantFrom('fp8' as const, 'auto' as const),
      textOnly: fc.boolean(),
      workload: w,
      util: fc.option(fc.double({ min: 0.5, max: 0.95, noNaN: true }), { nil: undefined }),
    })
    .map(({ util, ...o }) => ({ ...o, ...(util !== undefined ? { util } : {}) }));

const repoArb = fc.constantFrom(...REPOS);
const gpuArb = fc.constantFrom(...GPUS);
/** One of the layouts the command plans (1, 2, 4, 8 GPUs: TP, PP, copies) on a card. */
const layoutArb = (gpu: fc.Arbitrary<Gpu> = gpuArb): fc.Arbitrary<GpuLayout> => gpu.chain((g) => fc.constantFrom(...layoutsFor([g])));

const model = (repo: string) => models.get(repo) as Loaded;
const ctxOf = (repo: string, opts: PlanOptions) => ({ model: model(repo), opts, repo, verification: model(repo).verification });
/** Memory a GPU needs for one full-context request, worst case: weights + its KV cache + the largest fixed costs. */
const needMiB = (r: PlanRow) => r.memory.worst.weightsMiB + r.memory.worst.kvPerSeqMiB + r.memory.worst.fixed.total.max;
const RANK = { fits: 0, 'borderline': 1, 'does-not-fit': 2, impossible: 3 } as const;

describe('memory', () => {
  it('a longer context never needs less memory, never leaves more margin, never improves the verdict', () => {
    fc.assert(
      fc.property(repoArb, layoutArb(), optionsArb(), fc.integer({ min: 256, max: 1_048_576 }), fc.integer({ min: 256, max: 1_048_576 }), (repo, layout, opts, a, b) => {
        const [short, long] = a <= b ? [a, b] : [b, a];
        const at = (n: number) => planLayout(model(repo), layout, { ...opts, workload: { ...opts.workload, maxContext: n, typicalContext: Math.min(n, opts.workload.typicalContext) } });
        const s = at(short);
        const l = at(long);
        expect(l.memory.worst.kvPerSeqMiB).toBeGreaterThanOrEqual(s.memory.worst.kvPerSeqMiB - EPS);
        expect(needMiB(l)).toBeGreaterThanOrEqual(needMiB(s) - EPS);
        expect(l.marginMiB.min).toBeLessThanOrEqual(s.marginMiB.min + EPS);
        expect(l.marginMiB.max).toBeLessThanOrEqual(s.marginMiB.max + EPS);
        expect(RANK[l.verdict]).toBeGreaterThanOrEqual(RANK[s.verdict]);
      }),
      RUNS,
    );
  });

  it('copies of a layout (data parallel) hold the same per GPU: same verdict, memory and margin; copies × the requests', () => {
    // Each copy is an independent engine on tp × pp GPUs: per GPU nothing changes, the requests add up.
    const copiesArb = layoutArb().filter((l) => (l.replicas ?? 1) > 1);
    fc.assert(
      fc.property(repoArb, copiesArb, optionsArb(), (repo, layout, opts) => {
        const k = layout.replicas as number;
        const one = planLayout(model(repo), { gpu: layout.gpu, count: layout.tp * layout.pp, tp: layout.tp, pp: layout.pp }, opts);
        const many = planLayout(model(repo), layout, opts);
        expect(many.verdict).toBe(one.verdict);
        expect(many.memory).toEqual(one.memory);
        expect(many.marginMiB).toEqual(one.marginMiB);
        expect(many.capacity === undefined).toBe(one.capacity === undefined);
        if (one.capacity && many.capacity) {
          expect(many.capacity.reliable).toBe(k * one.capacity.reliable);
          expect(many.capacity.probable).toBe(k * one.capacity.probable);
          expect(many.capacity.memoryTypical).toBe(k * one.capacity.memoryTypical);
          // One request runs on one copy: its speed does not change.
          expect(many.capacity.single).toEqual(one.capacity.single);
        }
      }),
      RUNS,
    );
  });

  it('more GPUs of the same card as copies never turn a fit into a non-fit (1 → 2 → 4 → 8 at the same TP)', () => {
    fc.assert(
      fc.property(repoArb, gpuArb, fc.constantFrom(1, 2, 4, 8), optionsArb(), (repo, gpu, tp, opts) => {
        const verdicts = [1, 2, 4, 8].filter((n) => n >= tp).map((n) => planLayout(model(repo), { gpu, count: n, tp, pp: 1, ...(n / tp > 1 ? { replicas: n / tp } : {}) }, opts).verdict);
        expect(new Set(verdicts).size).toBe(1);
      }),
      RUNS,
    );
  });

  it('splitting the model over twice the GPUs (TP, or PP without NVLink) never turns a fit into a non-fit when both start', () => {
    // Doubling TP or PP halves the bulk of the weights and of the KV cache per GPU (MLA's latent and the logits
    // stay whole); the only cost that grows is NCCL (at most 0.5 GiB + 16 MiB per GPU), far below the weights
    // saved for every catalog model. A split the engine refuses ("impossible") is a different answer, not a
    // memory verdict: such pairs are skipped.
    const splitArb = fc.tuple(gpuArb, fc.constantFrom(1, 2, 4), fc.boolean()).filter(([g, , pp]) => !pp || g.link !== 'nvlink');
    fc.assert(
      fc.property(repoArb, splitArb, optionsArb(), (repo, [gpu, n, pp], opts) => {
        const at = (k: number) => planLayout(model(repo), { gpu, count: k, tp: pp ? 1 : k, pp: pp ? k : 1 }, opts);
        if (pp && 2 * n > 4) return;
        const few = at(n);
        const more = at(2 * n);
        if (few.verdict === 'impossible' || more.verdict === 'impossible') return;
        expect(RANK[more.verdict]).toBeLessThanOrEqual(RANK[few.verdict]);
        expect(more.memory.worst.weightsMiB).toBeLessThanOrEqual(few.memory.worst.weightsMiB);
      }),
      RUNS,
    );
  });

  it('the memory items add up: memory per GPU = weights + one request’s KV cache + the fixed costs; fixed costs = their parts', () => {
    fc.assert(
      fc.property(repoArb, layoutArb(), optionsArb(), (repo, layout, opts) => {
        const row = planLayout(model(repo), layout, opts);
        for (const est of [row.memory.worst, row.memory.best]) {
          const f = est.fixed;
          const parts = [f.cudaContext, f.workspaces, f.activations, f.vision, f.cudaGraphs, f.nccl, ...(f.weightsExtra ? [f.weightsExtra] : [])];
          expect(close(f.total.min, parts.reduce((a, p) => a + p.min, 0))).toBe(true);
          expect(close(f.total.max, parts.reduce((a, p) => a + p.max, 0))).toBe(true);
          expect(f.total.min).toBeLessThanOrEqual(f.total.max);
        }
        const fig = figuresOf(ctxOf(repo, opts), row);
        expect(close(fig.memory.min, fig.weights.value + fig.kv.value + fig.fixed.min)).toBe(true);
        expect(close(fig.memory.max, fig.weights.value + fig.kv.value + fig.fixed.max)).toBe(true);
      }),
      RUNS,
    );
  });
});

/** Every measure of a row's figures, by name. */
function measures(f: Figures): [string, Measure][] {
  const all: [string, Measure | undefined][] = [
    ['memory', f.memory],
    ['engineGB', f.engineGiB],
    ['weights', f.weights],
    ['kv', f.kv],
    ['fixed', f.fixed],
    ['margin', f.margin],
    ['tokS', f.tokS],
    ['single', f.single],
    ['read', f.read],
    ['wait', f.wait],
  ];
  return all.filter((m): m is [string, Measure] => m[1] !== undefined);
}

describe('speed and users', () => {
  it('no speed beats the card: one request’s generation ≤ the roofline, and ≤ bandwidth × TP ÷ the bytes a token must read', () => {
    fc.assert(
      fc.property(repoArb, layoutArb(gpuArb.filter((g) => g.bandwidthGBs > 0)), optionsArb(), (repo, layout, opts) => {
        const row = planLayout(model(repo), layout, opts);
        if (!row.capacity) return;
        const roof = (row.roofline as NonNullable<PlanRow['roofline']>).decodeTokS;
        const c = row.capacity;
        // The planner's own roofline (datasheet bandwidth, no overhead) bounds every speed it gives.
        expect(c.single.tokS.max).toBeLessThanOrEqual(roof * (1 + 1e-9));
        expect(c.atReliable.tokS.max).toBeLessThanOrEqual(roof * (1 + 1e-9));
        expect(c.atProbable.tokS.max).toBeLessThanOrEqual(roof * (1 + 1e-9));
        // Independently: a token reads at least the dense weights and one token's experts, split over the TP
        // GPUs reading in parallel (pipeline stages run one after the other, copies do not help one request).
        const p = model(repo).profile;
        const bytes = p.denseBytes + (p.experts > 0 ? (p.routedExpertBytes * p.expertsPerToken) / p.experts : p.routedExpertBytes);
        const bound = (layout.gpu.bandwidthGBs * 1e9 * layout.tp) / bytes;
        expect(c.single.tokS.max).toBeLessThanOrEqual(bound * (1 + 1e-9));
      }),
      RUNS,
    );
  });

  it('every range holds its value (min ≤ value ≤ max, error ≥ 0); guaranteed ≤ likely ≤ what the memory holds', () => {
    fc.assert(
      fc.property(repoArb, layoutArb(), optionsArb(), fc.option(fc.integer({ min: 1, max: 5000 }), { nil: undefined }), (repo, layout, opts, concurrent) => {
        const row = planLayout(model(repo), layout, opts);
        const f = figuresOf(ctxOf(repo, opts), row, concurrent !== undefined ? { concurrent } : undefined);
        for (const [name, m] of measures(f)) {
          expect([name, m.min <= m.value + EPS && m.value <= m.max + EPS]).toEqual([name, true]);
          expect([name, Number.isFinite(m.value) && m.precision.errorPct >= 0]).toEqual([name, true]);
        }
        if (f.requests && f.users) {
          expect(f.requests.guaranteed).toBeLessThanOrEqual(f.requests.likely);
          expect(f.users.guaranteed).toBeLessThanOrEqual(f.users.likely);
          // The likely count never passes the memory limit with its safety margin.
          expect(f.requests.likely).toBeLessThanOrEqual(Math.floor(f.requests.memory * opts.workload.safety));
          expect(f.users.likely).toBeLessThanOrEqual(f.users.memoryMax);
          expect(f.requests.guaranteed).toBeGreaterThanOrEqual(0);
        }
        if (f.roofline) {
          expect(f.roofline.tokSShare).toBeLessThanOrEqual(1 + 1e-9);
          expect(f.roofline.readShare).toBeLessThanOrEqual(1 + 1e-9);
        }
      }),
      RUNS,
    );
  });
});

/** The JSON of the command (the fields these properties read). */
interface JsonAnswer {
  fit: { figures: Figures; alternative?: Figures } | null;
  users: { choice?: { figures: Figures; alternative?: Figures } } | null;
  cards: { gpu: string; status: string; figures: Figures }[];
  rows: Figures[];
}

/** Catalog models in the compatibility matrix, with cards their engine does not start on. */
const MATRIX_REPOS = ['ibm-granite/granite-4.2-3b', 'openai/gpt-oss-20b', 'google/gemma-4-31B-it', 'microsoft/Phi-4-reasoning-vision-15B', 'nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16', 'deepseek-ai/DeepSeek-V4.1-Flash'] as const;

describe('the command', () => {
  it('a layout whose engine does not start is never the answer, nor a card the overview says fits', async () => {
    await fc.assert(
      fc.asyncProperty(
        fc.constantFrom(...MATRIX_REPOS),
        fc.constantFrom('vllm', 'sglang'),
        fc.constantFrom('fp8', 'auto'),
        fc.constantFrom('8k', '32k', '128k'),
        fc.option(fc.constantFrom('50', '500', '5000'), { nil: undefined }),
        async (repo, engine, kv, tokens, users) => {
          const r = await runCli([repo, '--engine', engine, '--cache', kv, '--tokens', tokens, ...(users ? ['--users', users] : []), '--json', 'a.json'], cache.dir);
          expect(r.code).toBe(0);
          const json = JSON.parse(r.written['a.json'] as string) as JsonAnswer;
          const answers = [json.fit?.figures, json.fit?.alternative, json.users?.choice?.figures, json.users?.choice?.alternative].filter((f): f is Figures => f !== undefined);
          for (const f of answers) {
            expect([f.hardware, f.compat.status]).not.toEqual([f.hardware, 'does-not-start']);
            expect(['fits', 'at-limit']).toContain(f.verdict);
          }
          for (const c of json.cards) if (c.status !== 'no-start' && c.status !== 'no-fit') expect([c.gpu, c.figures.compat.status]).not.toEqual([c.gpu, 'does-not-start']);
          // Every layout the engine refuses has no speed and no users.
          for (const f of json.rows) if (f.compat.status === 'does-not-start') expect([f.hardware, f.users, f.verdict]).toEqual([f.hardware, undefined, 'no-start']);
        },
      ),
      { seed: RUNS.seed, numRuns: 30 },
    );
  }, 120_000);

  /** The overview's line of a card in the text (its name, then two spaces). */
  const cardLine = (text: string, gpu: string) => text.split('\n').find((l) => new RegExp(`^  ${gpu.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}( \\(this PC\\))? {2,}`).test(l));

  it.each([
    ['ibm-granite/granite-4.2-3b', ['--tokens', '32k']],
    ['openai/gpt-oss-20b', ['--tokens', '128k', '--users', '300', '--engine', 'sglang']],
    ['google/gemma-4-31B-it', ['--tokens', '8k', '--prompt-cache', '80', '--cache', 'auto']],
    ['zai-org/GLM-5.3-Flash', ['--tokens', '64k']],
  ] as const)('%s %s: the text, the CSV and the JSON say the same numbers for every card', async (repo, args) => {
    // Wide enough for the long forms of every column.
    const r = await runCli([repo, ...args, '--csv', 'a.csv', '--json', 'a.json'], cache.dir, { columns: 400 });
    expect(r.code).toBe(0);
    const json = JSON.parse(r.written['a.json'] as string) as JsonAnswer;
    const csv = (r.written['a.csv'] as string).trimEnd().split('\r\n').map((l) => l.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/));
    const col = (name: string) => CSV_HEADERS.indexOf(name);
    for (const card of json.cards) {
      const f = card.figures;
      const line = cardLine(r.text, card.gpu);
      expect([card.gpu, line !== undefined]).toEqual([card.gpu, true]);
      expect(line).toContain(`${gb(f.memory.value)} GiB`);
      if (f.users && card.status !== 'no-fit' && card.status !== 'no-start') expect(line).toMatch(new RegExp(` ${int(f.users.guaranteed)} +${int(f.users.likely)} `));
      const row = csv.find((c) => c[col('Layout')]?.replace(/^"|"$/g, '') === f.hardware);
      expect([f.hardware, row !== undefined]).toEqual([f.hardware, true]);
      const cell = (name: string) => (row as string[])[col(name)] as string;
      expect(Number(cell('Memory per GPU (GiB)'))).toBeCloseTo(f.memory.value, 2);
      expect(Number(cell('Weights per GPU (GiB)'))).toBeCloseTo(f.weights.value, 2);
      if (f.users) {
        expect(Number(cell('Users min (guaranteed) per layout'))).toBe(f.users.guaranteed);
        expect(Number(cell('Users likely per layout'))).toBe(f.users.likely);
        expect(Number(cell('Speed per user (tok/s)'))).toBeCloseTo((f.tokS as Measure).value, 2);
      } else expect(cell('Users min (guaranteed) per layout')).toBe('');
    }
    // One CSV line per layout planned, in the JSON's order.
    expect(csv.length - 1).toBe(json.rows.length);
    expect(csv.slice(1).map((c) => c[col('Layout')]?.replace(/^"|"$/g, ''))).toEqual(json.rows.map((f) => f.hardware));
  });
});

describe('the built-in models', () => {
  /**
   * Repos scripts/pack-models.mjs packs (readable, public, with their headers) that have no config.json: the
   * command reads config.json first, so it cannot plan them.
   */
  function withoutConfig(dir = path.join('data', 'models')): string[] {
    const out: string[] = [];
    if (existsSync(path.join(dir, 'source.json'))) {
      const source = JSON.parse(readFileSync(path.join(dir, 'source.json'), 'utf8')) as { repo: string; readable: boolean; gated: unknown };
      const files = readdirSync(dir);
      if (source.readable && !source.gated && files.includes('safetensors-headers.json.gz') && !files.some((f) => /^config.json(.gz)?$/.test(f))) out.push(source.repo);
    }
    for (const e of readdirSync(dir, { withFileTypes: true })) if (e.isDirectory()) out.push(...withoutConfig(path.join(dir, e.name)));
    return out.sort();
  }

  // Three Mistral-format repos are packed with params.json but no config.json, which the calculator needs: it says
  // so and stops (exit 1) without asking for config.json. This test pins the list so a new one is caught.
  it('only the three known Mistral-format repos lack a config.json in the pack', () => {
    expect(withoutConfig()).toEqual(['mistralai/Mistral-Large-3-675B-Instruct-2512', 'mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4', 'mistralai/Mistral-Small-4-119B-2603-NVFP4']);
  });

  it('a Mistral-format built-in model: a clear limit, not a network error', async () => {
    const r = await runCli(['mistralai/Mistral-Large-3-675B-Instruct-2512', '--tokens', '8k', '--offline'], cache.dir);
    expect(r.code).toBe(1);
    expect(r.text).toMatch(/^mistralai\/Mistral-Large-3-675B-Instruct-2512 ships only the Mistral format \(params\.json, no config\.json\)/);
    expect(r.text).not.toContain('Hugging Face does not answer');
  });

  // When every layout fits in memory but the engine does not start, the headline says so, not "does not fit".
  it('the headline never says "does not fit" when the memory fits and only the engine refuses', async () => {
    const r = await runCli(['deepseek-ai/DeepSeek-V4.1-Flash', '--tokens', '32k', '--gpu', 'rtx-pro-6000-blackwell-server-96gb', '--count', '8', '--json', 'd.json'], cache.dir);
    expect(r.code).toBe(0);
    const json = JSON.parse(r.written['d.json'] as string) as JsonAnswer;
    // The premise: nothing starts, yet the memory fits on every card. The Server edition: 8 per server (the workstation
    // editions hold 4 per machine, where the model does not fit).
    expect(json.cards.every((c) => c.status === 'no-start' && c.figures.memoryFits)).toBe(true);
    expect(r.text.split('\nScenario')[0]).not.toMatch(/does not fit/);
  });
});

import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { hardwareFileSchema } from '../../src/contract/data';
import { kvLayoutOf } from '../../src/model/kv';
import { factsFromHfConfig, loadShapeFromHfConfig } from '../../src/model/hf-config';
import type { TensorInfo } from '../../src/model/weights-exact';
import { DEFAULT_WORKLOAD, type Workload } from '../../src/planner/capacity';
import { gpuFromSpec, type Gpu, type GpuLayout } from '../../src/planner/hardware';
import { parallelShapeFrom, plannerModel } from '../../src/planner/model';
import { modelProfile } from '../../src/planner/model-profile';
import { pipelineUnverified, pipelineWeight, planLayout, PP_NOT_VERIFIED, rankRows, SPARSE_TRANSIENT_MIB, stageLayout, type PlannerModel, type PlanOptions, type PlanRow } from '../../src/planner/plan';

const hw = hardwareFileSchema.parse(JSON.parse(readFileSync(path.join('data', 'hardware.json'), 'utf8')));
const gpu = (id: string): Gpu => gpuFromSpec(hw.gpus.find((g) => g.id === id) as (typeof hw.gpus)[number]);
const at = (id: string, tp = 1, pp = 1): GpuLayout => ({ gpu: gpu(id), count: tp * pp, tp, pp });
const readJson = (...p: string[]) => JSON.parse(readFileSync(path.join('test', 'fixtures', 'hf', ...p), 'utf8')) as unknown;
const header = (name: string): TensorInfo[] =>
  Object.entries((readJson('headers', `${name}.json`) as { tensors: Record<string, [string, number[]]> }).tensors).map(([n, [dtype, shape]]) => ({ name: n, dtype, shape }));
const real = (name: string) => {
  const c = readJson('configs', `${name}.json`);
  return plannerModel(c, factsFromHfConfig(c), loadShapeFromHfConfig(c), header(name));
};
const qwen = real('qwen3-8b');
const workload = (maxContext: number): Workload => ({ ...DEFAULT_WORKLOAD, maxContext, typicalContext: maxContext / 2 });
const vllm = (maxContext = 40_960, extra: Partial<PlanOptions> = {}): PlanOptions => ({ engine: 'vllm', kvCache: 'auto', textOnly: false, workload: workload(maxContext), ...extra });

describe('planLayout: verdicts (Qwen3-8B BF16, vLLM)', () => {
  it('entra: fits with the largest fixed costs on the smallest CUDA memory (RTX 5090, 40 960 tokens)', () => {
    const r = planLayout(qwen, at('rtx-5090-32gb'), vllm());
    expect(r.verdict).toBe('fits');
    // CUDA sees 31.35 GiB (Linux) … 31.82 GiB (Windows).
    expect(r.memory.worst.totalMiB).toBe(32_102);
    expect(r.memory.best.totalMiB).toBe(32_579);
    // KV of one request: 36 layers × 8 heads × 256 × 2 B × 40 960 tokens = 5760 MiB.
    expect(r.memory.worst.kvPerSeqMiB).toBe(5760);
    expect(r.marginMiB).toEqual({ min: r.memory.worst.marginMiB.min, max: r.memory.best.marginMiB.max });
    expect(r.weights.perGpuBytes).toBe(16_381_470_720 + 10_485_760);
    // Verified bf16 dense: 0.8 % of the weights as a margin, 0.008 × 16 391 956 480 B = 125 MiB.
    expect(r.memory.worst.fixed.weightsExtra).toEqual({ min: 0, max: 125 });
    expect(r.capacity).toBeDefined();
    expect(r.speedNote).toBeUndefined();
  });

  it('al limite: fits only with the smaller fixed costs (A100 40 GB, 131 072 tokens)', () => {
    const r = planLayout(qwen, at('a100-sxm4-40gb'), vllm(131_072));
    expect([r.memory.worst.verdict, r.memory.best.verdict]).toEqual(['uncertain', 'uncertain']);
    expect(r.verdict).toBe('borderline');
    // Speed is still estimated for a borderline fit.
    expect(r.capacity?.limitedBy).toBe('memory');
  });

  it('non fits: fails even with the smallest costs (RTX 5080 16 GB); no speed', () => {
    const r = planLayout(qwen, at('rtx-5080-16gb'), vllm());
    expect(r.verdict).toBe('does-not-fit');
    expect(r.capacity).toBeUndefined();
  });

  it('impossibile: a layout vLLM refuses, whatever the memory', () => {
    const r = planLayout(qwen, at('h100-sxm5-80gb', 3), vllm());
    expect(r.check.blockers).toEqual(['32 attention heads do not divide by TP 3', '8 KV heads do not divide by TP 3']);
    expect(r.verdict).toBe('impossible');
    expect(r.capacity).toBeUndefined();
  });

  it('a model the engine does not start on the card: impossibile, no speed, the memory verdict kept apart', () => {
    // Qwen3-8B's tensors read as DeepSeek-V4.1-Flash's checkpoint: the compatibility matrix decides by repo.
    const dsv41 = { ...qwen, repo: 'deepseek-ai/DeepSeek-V4.1-Flash' };
    const r = planLayout(dsv41, at('rtx-5090-32gb'), vllm());
    expect(r.compat?.status).toBe('does-not-start');
    expect(r.verdict).toBe('impossible');
    expect(r.memoryVerdict).toBe('fits');
    expect(r.capacity).toBeUndefined();
    expect(r.check.blockers).toHaveLength(1);
    expect(r.check.blockers[0]).toMatch(/^does not start with vLLM 0\.31\.0 on this card: On sm120, /);
    // On Hopper it starts on Linux (with weight-only MXFP4 experts) but not under WSL 2.
    const h100 = planLayout(dsv41, at('h100-sxm5-80gb'), vllm());
    expect([h100.compat?.status, h100.verdict, h100.check.blockers]).toEqual(['partial', 'fits', []]);
    const wsl = planLayout(dsv41, at('h100-sxm5-80gb'), vllm(40_960, { platform: 'windows-wsl' }));
    expect(wsl.verdict).toBe('impossible');
    expect(wsl.check.blockers[0]).toMatch(/^does not start with vLLM 0\.31\.0 on this card under Windows \/ WSL 2: /);
  });

  it('sparse attention beyond 128K tokens: it fits only with room for the transient buffer left on each GPU', () => {
    // A long-context model with sparse attention (the Qwen3-8B shape, its context raised to 1M).
    const long = { ...qwen, facts: { ...qwen.facts, contextLength: 1_048_576 } };
    const sparse = { ...long, sparseTopk: 2048 };
    const big = planLayout(sparse, at('b200-sxm-180gb'), vllm(262_144));
    expect(big.check.warnings.at(-1)).toMatch(/^sparse attention beyond 128K tokens: .* about 21 GiB more GPU memory/);
    // A B200 keeps far more than 21 GiB free: it fits.
    expect(big.memory.worst.marginMiB.min).toBeGreaterThan(SPARSE_TRANSIENT_MIB);
    expect(big.verdict).toBe('fits');
    // An L40S that fits the same kind of request without that room is only at the limit.
    const small = planLayout(sparse, at('l40s-48gb'), vllm(140_000));
    expect(small.memory.worst.verdict).toBe('fits');
    expect(small.memory.worst.marginMiB.min).toBeLessThan(SPARSE_TRANSIENT_MIB);
    expect([small.memoryVerdict, small.verdict]).toEqual(['borderline', 'borderline']);
    // The cause is named: the reserve, not the uncertainty of the fixed costs.
    expect([small.limitCause, big.limitCause]).toEqual(['sparse-reserve', undefined]);
    // Without sparse attention, or up to 128K, nothing changes.
    expect(planLayout(long, at('l40s-48gb'), vllm(140_000)).memoryVerdict).toBe('fits');
    expect(planLayout(sparse, at('b200-sxm-180gb'), vllm(131_072)).check.warnings).toEqual([]);
  });

  it("beyond the model's maximum context the speed is not reliable; up to it nothing changes", () => {
    // Qwen3-8B: 40,960 positions in its config.
    expect(qwen.facts.contextLength).toBe(40_960);
    expect(planLayout(qwen, at('h100-sxm5-80gb'), vllm(40_960)).speedNote).toBeUndefined();
    expect(planLayout(qwen, at('h100-sxm5-80gb'), vllm(81_920)).speedNote).toBe("beyond the model's maximum context (40,960 tokens): the engine needs a RoPE / YaRN override, speed not reliable");
    // The card's own caveat comes first.
    expect(planLayout(qwen, { ...at('h100-sxm5-80gb'), gpu: { ...gpu('h100-sxm5-80gb'), bandwidthGBs: 0 } }, vllm(81_920)).speedNote).toMatch(/^speed not estimable/);
  });

  it('the matrix by architecture class when the repo is not in it; nothing when the model is not in it', () => {
    // DeepSeek-V4.1 in a format the catalog lacks (BF16): no catalog checkpoint starts on sm120, so neither does this one.
    const byClass = { ...qwen, parallel: { ...qwen.parallel, architecture: 'DeepseekV41ForCausalLM' } };
    const sm120 = planLayout(byClass, at('rtx-5090-32gb'), vllm());
    expect([sm120.compat?.status, sm120.compat?.repo, sm120.verdict, sm120.memoryVerdict]).toEqual(['does-not-start', null, 'impossible', 'fits']);
    expect(sm120.compat?.reason).toMatch(/^no checkpoint of deepseek-v4-1-flash in the catalog starts: .*sm120/);
    expect(planLayout(byClass, at('rtx-5090-32gb'), vllm(40_960, { platform: 'windows-wsl' })).compat?.status).toBe('does-not-start');
    // On Hopper some checkpoint starts: the unknown format stays unknown.
    expect(planLayout(byClass, at('h100-sxm5-80gb'), vllm()).compat?.status).toBe('unknown');
    // A class whose catalog format matches is answered directly.
    const glm = { ...qwen, parallel: { ...qwen.parallel, architecture: 'Glm5NextForConditionalGeneration', weights: 'fp8' as const } };
    expect(planLayout(glm, at('rtx-5090-32gb'), vllm()).compat?.repo).toBe('zai-org/GLM-5.3-Flash');
    const r = planLayout(qwen, at('rtx-5090-32gb'), vllm());
    expect(r.compat).toBeUndefined();
    expect(r.memoryVerdict).toBe(r.verdict);
    // A model read without its architecture class nor a known repo.
    const { architecture: _, ...noArch } = qwen.parallel;
    expect(planLayout({ ...qwen, parallel: noArch, repo: 'someone/unknown' }, at('rtx-5090-32gb'), vllm()).compat).toBeUndefined();
    // plannerModel keeps the repo it is given.
    const c = readJson('configs', 'qwen3-8b.json');
    expect(plannerModel(c, factsFromHfConfig(c), loadShapeFromHfConfig(c), header('qwen3-8b'), 'Qwen/Qwen3-8B').repo).toBe('Qwen/Qwen3-8B');
  });
});

describe('planLayout: layouts and options', () => {
  it('pipeline parallel: each stage holds half the layers’ KV', () => {
    const r = planLayout(qwen, at('rtx-4090-24gb', 1, 2), vllm());
    // 18 of 36 layers: 2880 MiB per request.
    expect(r.memory.worst.kvPerSeqMiB).toBe(2880);
    expect(r.weights.stageBytes).toHaveLength(2);
    expect(r.verdict).toBe('fits');
  });

  it('data parallel: each copy holds the whole model; requests and users multiply by the copies, speed stays', () => {
    const one = planLayout(qwen, at('h100-sxm5-80gb'), vllm());
    const four = planLayout(qwen, { gpu: gpu('h100-sxm5-80gb'), count: 4, tp: 1, pp: 1, replicas: 4 }, vllm());
    // The memory of one GPU is the same as one card alone.
    expect(four.memory.worst.kvPerSeqMiB).toBe(one.memory.worst.kvPerSeqMiB);
    expect(four.marginMiB).toEqual(one.marginMiB);
    expect(four.weights.perGpuBytes).toBe(one.weights.perGpuBytes);
    const c1 = one.capacity as NonNullable<PlanRow['capacity']>;
    const c4 = four.capacity as NonNullable<PlanRow['capacity']>;
    expect(c1.reliable).toBeGreaterThan(0);
    expect(c4).toEqual({
      ...c1,
      memoryWorst: c1.memoryWorst * 4,
      memoryTypical: c1.memoryTypical * 4,
      speedLimit: c1.speedLimit * 4,
      reliable: c1.reliable * 4,
      probable: c1.probable * 4,
      totalUsers: c1.totalUsers * 4,
      probableUsers: c1.probableUsers * 4,
    });
    // Copies of a TP 2 group: the group's numbers × 2.
    const tp2 = planLayout(qwen, at('h100-sxm5-80gb', 2), vllm()).capacity as NonNullable<PlanRow['capacity']>;
    const tp2x2 = planLayout(qwen, { gpu: gpu('h100-sxm5-80gb'), count: 4, tp: 2, pp: 1, replicas: 2 }, vllm()).capacity as NonNullable<PlanRow['capacity']>;
    expect(tp2x2.reliable).toBe(tp2.reliable * 2);
    expect(tp2x2.atReliable).toEqual(tp2.atReliable);
    // replicas: 1 is the same as none.
    expect(planLayout(qwen, { ...at('h100-sxm5-80gb'), replicas: 1 }, vllm()).capacity).toEqual(c1);
    // No fit: no capacity to scale.
    expect(planLayout(qwen, { gpu: gpu('rtx-5080-16gb'), count: 2, tp: 1, pp: 1, replicas: 2 }, vllm()).capacity).toBeUndefined();
  });

  it('text only: the vision tower is neither loaded nor profiled (Gemma 3 27B)', () => {
    const gemma = real('gemma-3-27b-it');
    const full = planLayout(gemma, at('h100-sxm5-80gb'), vllm());
    const text = planLayout(gemma, at('h100-sxm5-80gb'), vllm(40_960, { textOnly: true }));
    expect(full.memory.worst.fixed.vision.max).toBeGreaterThan(0);
    expect(text.memory.worst.fixed.vision).toEqual({ min: 0, max: 0 });
    expect(full.weights.perGpuBytes - text.weights.perGpuBytes).toBe(full.weights.visionBytes);
    expect(text.weights.visionBytes).toBe(0);
  });

  it('MLA: the absorbed kv_b_proj copies are an extra on the weights; a verified format adds no gap', () => {
    const c = readJson('configs', 'deepseek-v3.json');
    const facts = factsFromHfConfig(c);
    const shape = loadShapeFromHfConfig(c);
    const { layout } = kvLayoutOf(facts);
    const tensors: TensorInfo[] = [{ name: 'model.layers.0.self_attn.kv_b_proj.weight', dtype: 'BF16', shape: [32_768, 512] }];
    const m: PlannerModel = { facts, layout, shape, tensors, parallel: parallelShapeFrom(c, facts, shape, 'bf16'), profile: modelProfile(tensors, shape, facts, layout), formatGapShare: 0 };
    const r = planLayout(m, at('b200-sxm-180gb'), vllm(8192));
    // 61 layers × 512 × 128 heads × (128 + 128) × 2 B = 1952 MiB, kept once or twice.
    expect(r.memory.worst.fixed.weightsExtra).toEqual({ min: 1952, max: 3904 });
    // Neither MLA nor a gap: no extra at all.
    const plain = planLayout({ ...qwen, formatGapShare: 0 }, at('rtx-5090-32gb'), vllm());
    expect(plain.memory.worst.fixed.weightsExtra).toBeUndefined();
  });

  it('engine options pass through: utilization, batched tokens, max sequences (which caps the typical fit)', () => {
    const r = planLayout(qwen, at('h100-sxm5-80gb'), vllm(40_960, { util: 0.8, maxBatchedTokens: 4096, maxSeqs: 2 }));
    expect(r.memory.worst.util).toBe(0.8);
    expect(r.memory.worst.limits).toMatchObject({ batchedTokens: 4096, maxSeqs: 2 });
    expect(r.capacity?.memoryTypical).toBe(2);
  });

  it('SGLang: token slots, the full-context size scaled to the typical context', () => {
    const r = planLayout(qwen, at('h100-sxm5-80gb'), { ...vllm(), engine: 'sglang' });
    const w = r.memory.worst;
    // Typical = half the context: half the full request's KV per request.
    expect(r.capacity?.memoryTypical).toBe(Math.min(Math.floor(w.kvRoomMiB.min / (w.kvPerSeqMiB / 2)), w.limits.maxSeqs));
    expect(r.capacity?.memoryWorst).toBe(w.concurrency.min);
  });

  it('a model without KV cache: no typical requests counted, memory limits it to zero', () => {
    const r = planLayout({ ...qwen, layout: { groups: [], recurrentLayers: 0, recurrentBytesPerSeq: 0 } }, at('h100-sxm5-80gb'), vllm());
    expect(r.memory.worst.kvPerSeqMiB).toBe(0);
    expect(r.capacity).toMatchObject({ memoryTypical: 0, reliable: 0, limitedBy: 'memory' });
  });

  it('no TFLOPS published (GB200): fits, no speed, said why', () => {
    const r = planLayout(qwen, at('gb200-nvl72'), vllm());
    expect(r.verdict).toBe('fits');
    expect(r.capacity).toBeUndefined();
    expect(r.speedNote).toBe('speed not estimable: card bandwidth or TFLOPS not published');
  });

  it('a speed caveat does not remove the estimate (INT4 MoE on Hopper)', () => {
    const r = planLayout({ ...qwen, parallel: { ...qwen.parallel, weights: 'int4', moe: true } }, at('h100-sxm5-80gb'), vllm());
    expect(r.speedNote).toMatch(/^INT4 MoE without Blackwell/);
    expect(r.capacity).toBeDefined();
  });

  it('FP8 KV cache: one byte per element halves the request', () => {
    expect(planLayout(qwen, at('rtx-5090-32gb'), vllm(40_960, { kvCache: 'fp8' })).memory.worst.kvPerSeqMiB).toBe(2880);
  });
});

describe('rankRows', () => {
  const row = (id: string, verdict: PlanRow['verdict'], count: number, nominalGB: number, reliable?: number, pp = 1, margin = 0): PlanRow =>
    ({ layout: { gpu: { id, nominalGB }, count, tp: count / pp, pp }, verdict, marginMiB: { min: margin, max: margin + 1000 }, ...(reliable !== undefined ? { capacity: { reliable } } : {}) }) as unknown as PlanRow;

  it('fits and serves first, then fewer GPUs, less VRAM, more users, tensor before pipeline', () => {
    const rows = [
      row('impossible', 'impossible', 1, 24),
      row('no', 'does-not-fit', 1, 24),
      row('limit', 'borderline', 1, 24, 5),
      row('fits-nobody', 'fits', 1, 24, 0),
      row('fits-2gpu', 'fits', 2, 24, 9),
      row('fits-80', 'fits', 1, 80, 9),
      row('fits-24-few', 'fits', 1, 24, 2),
      row('fits-24-many', 'fits', 1, 24, 7),
      row('fits-2gpu-pp', 'fits', 2, 24, 9, 2),
      row('fits-no-speed', 'fits', 1, 32),
    ];
    expect(rankRows(rows).map((r) => r.layout.gpu.id)).toEqual([
      'fits-24-many',
      'fits-24-few',
      'fits-no-speed',
      'fits-80',
      'fits-2gpu',
      'fits-2gpu-pp',
      'fits-nobody',
      'limit',
      'no',
      'impossible',
    ]);
    // The input is not reordered.
    expect(rows[0]?.layout.gpu.id).toBe('impossible');
  });

  it('rows without capacity count as zero users among equals', () => {
    const rows = [row('none', 'borderline', 1, 24), row('some', 'borderline', 1, 24, 1)];
    expect(rankRows(rows).map((r) => r.layout.gpu.id)).toEqual(['some', 'none']);
    expect(rankRows([...rows].reverse()).map((r) => r.layout.gpu.id)).toEqual(['some', 'none']);
    // Neither has a capacity: tensor before pipeline parallel.
    expect(rankRows([row('pp', 'does-not-fit', 2, 24, undefined, 2), row('tp', 'does-not-fit', 2, 24)]).map((r) => r.layout.gpu.id)).toEqual(['tp', 'pp']);
  });

  it('at the limit or not fitting: the largest worst-case margin first (the closest to fitting); fitting rows by users', () => {
    const tp = row('tp', 'borderline', 8, 94, undefined, 1, -3900);
    // A pipeline the matrix verified for the model (an unverified one would rank after, see below).
    const pp = { ...row('pp', 'borderline', 8, 94, undefined, 2, -1300), compat: { details: [{ feature: 'p:pp', status: 'ok' }] } } as unknown as PlanRow;
    expect(rankRows([tp, pp]).map((r) => r.layout.gpu.id)).toEqual(['pp', 'tp']);
    expect(rankRows([row('a', 'does-not-fit', 8, 94, undefined, 1, -9000), row('b', 'does-not-fit', 8, 94, undefined, 1, -100)]).map((r) => r.layout.gpu.id)).toEqual(['b', 'a']);
    // Rows that fit keep the users order whatever their margin.
    expect(rankRows([row('wide', 'fits', 8, 94, 1, 1, 9000), row('busy', 'fits', 8, 94, 5, 1, 10)]).map((r) => r.layout.gpu.id)).toEqual(['busy', 'wide']);
  });
});

describe('physical limits (roofline): no estimate passes the datasheet bandwidth and TFLOPS', () => {
  it('every card of the catalog, 1–8 GPUs, short and long contexts, vLLM and SGLang, fp8 and full cache', () => {
    let checked = 0;
    for (const spec of hw.gpus) {
      const g = gpuFromSpec(spec);
      if (g.bandwidthGBs <= 0 || g.tflops.bf16 === null) continue;
      for (const tp of [1, 2, 4, 8])
        for (const [ctx, engine, kvCache] of [
          [8192, 'vllm', 'fp8'],
          [40_960, 'sglang', 'auto'],
        ] as const) {
          const r = planLayout(qwen, { gpu: g, count: tp, tp, pp: 1 }, vllm(ctx, { engine, kvCache }));
          if (!r.capacity || !r.roofline) continue;
          checked++;
          // Generation of one request never faster than the bandwidth allows; prompt reading never faster than the TFLOPS.
          expect([spec.id, tp, r.capacity.single.tokS.max <= r.roofline.decodeTokS * 1.0001]).toEqual([spec.id, tp, true]);
          expect([spec.id, tp, r.capacity.single.fullPrefillMs.min >= r.roofline.fullPrefillMs * 0.9999]).toEqual([spec.id, tp, true]);
        }
    }
    expect(checked).toBeGreaterThan(50);
  });

  it('every row with a speed carries its basis and its limits', () => {
    const r = planLayout(qwen, at('h100-sxm5-80gb'), vllm(8192));
    expect(r.speedBasis).toMatchObject({ kind: 'fitted', decodePct: 12, prefillPct: 25, validUpTo: 8192 });
    expect(r.roofline?.decodeTokS).toBeGreaterThan(0);
    expect(planLayout(qwen, at('rtx-5080-16gb'), vllm()).speedBasis).toBeUndefined();
  });
});

describe('stageLayout: the fullest pipeline stage', () => {
  const group = (layers: number) => ({ layers, kvHeads: 8, keyLength: 128, valueLength: 128 });
  it('⌈layers / stages⌉ layers shared among the layer kinds in proportion, not rounded up kind by kind', () => {
    // 10 attention + 6 recurrent layers in 4 stages: 4 layers each (3 + 1), not ⌈10/4⌉ + ⌈6/4⌉ = 5.
    const mixed = stageLayout({ groups: [group(10)], recurrentLayers: 6, recurrentBytesPerSeq: 600 }, 4);
    expect([mixed.groups[0]?.layers, mixed.recurrentLayers, mixed.recurrentBytesPerSeq]).toEqual([3, 1, 100]);
    // A uniform stack: ⌈36 / 4⌉ = 9; one stage or no layers: unchanged.
    expect(stageLayout({ groups: [group(36)], recurrentLayers: 0, recurrentBytesPerSeq: 0 }, 4).groups[0]?.layers).toBe(9);
    const none = { groups: [], recurrentLayers: 0, recurrentBytesPerSeq: 0 };
    expect(stageLayout(none, 2)).toBe(none);
    const one = { groups: [group(36)], recurrentLayers: 0, recurrentBytesPerSeq: 0 };
    expect(stageLayout(one, 1)).toBe(one);
  });
});

describe('pipeline layouts and the engine', () => {
  const as = (repo: string) => ({ ...qwen, repo });
  const h200 = (tp: number, pp: number) => ({ gpu: gpu('h200-sxm-141gb'), count: tp * pp, tp, pp });
  it('a pipeline the engine refuses, or runs with wrong output, does not start; a partial one is never chosen over TP', () => {
    // GLM-5.3: SGLang crashes at start; vLLM reads stale top-k indices at a stage boundary (wrong output).
    expect(planLayout(as('zai-org/GLM-5.3'), h200(4, 2), vllm(32_768, { engine: 'sglang' })).verdict).toBe('impossible');
    expect(planLayout(as('zai-org/GLM-5.3'), h200(4, 2), vllm(32_768)).verdict).toBe('impossible');
    // MiniMax-M3 on vLLM: partial (an open start-up issue on SM100).
    const v = planLayout(as('MiniMaxAI/MiniMax-M3'), h200(4, 2), vllm(32_768));
    expect(v.compat?.details.find((d) => d.feature === 'p:pp')?.status).toBe('partial');
    expect(pipelineUnverified(v)).toBe(1);
    const tp = planLayout(as('MiniMaxAI/MiniMax-M3'), h200(8, 1), vllm(32_768));
    expect([tp.compat?.details.some((d) => d.feature === 'p:pp'), pipelineUnverified(tp)]).toEqual([false, 0]);
    // Ranked: the verified TP layout first, whatever the users of the partial pipeline.
    expect(rankRows([v, tp])[0]).toBe(tp);
  });
});

describe('pipeline layouts against tensor parallel ones', () => {
  it('a pipeline counts as its GPUs grown by its declared writing-speed error: it wins only by more than that', () => {
    // gpt-oss: vLLM implements its pipeline stages (the matrix says ok).
    const oss = { ...qwen, repo: 'openai/gpt-oss-120b' };
    const o = vllm(32_768, { kvCache: 'fp8' });
    const tp = planLayout(oss, at('h100-sxm5-80gb', 2), o);
    const pp = planLayout(oss, at('h100-sxm5-80gb', 1, 2), o);
    expect([pipelineWeight(tp), pipelineUnverified(pp)]).toEqual([1, 0]);
    expect(pipelineWeight(pp)).toBe(1 + (pp.speedBasis?.decodePct ?? 0) / 100);
    expect(pp.speedBasis?.widened).toContain('pipeline speed not measured');
    // Same GPUs: the tensor-parallel layout first.
    expect(rankRows([pp, tp])[0]).toBe(tp);
    // A pipeline on 2 GPUs still beats a tensor-parallel group on 4 (2 × 1.18 < 4).
    expect(rankRows([planLayout(oss, at('h100-sxm5-80gb', 4), o), pp])[0]).toBe(pp);
  });

  it('a model outside the matrix: its pipeline is not verified, says so, and ranks after tensor parallel', () => {
    const pp = planLayout(qwen, at('h100-sxm5-80gb', 1, 2), vllm(32_768, { kvCache: 'fp8' }));
    expect([pp.compat, pipelineUnverified(pp)]).toEqual([undefined, 1]);
    expect(pp.check.warnings).toContain(PP_NOT_VERIFIED);
    expect(rankRows([pp, planLayout(qwen, at('h100-sxm5-80gb', 4), vllm(32_768, { kvCache: 'fp8' }))])[0]?.layout.pp).toBe(1);
  });
});

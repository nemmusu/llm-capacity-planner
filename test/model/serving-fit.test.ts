import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { kvLayoutOf } from '../../src/model/kv';
import { factsFromHfConfig } from '../../src/model/hf-config';
import {
  estimateServing,
  gbText,
  kvHeadsPerGpu,
  schedulerDefaults,
  servingFixedCosts,
  servingKvBytesPerSeq,
  sglangDecodeGraphSizes,
  sglangDefaultMemFraction,
  sglangPrefillGraphSizes,
  vllmGraphSizes,
  type ServingInput,
} from '../../src/model/serving-fit';
import type { ModelFactsEx } from '../../src/model/types';

const MIB = 1024 * 1024;
/** Real config.json files (test/fixtures/hf/configs/SOURCES.txt). */
const config = (name: string) => factsFromHfConfig(JSON.parse(readFileSync(path.join('test', 'fixtures', 'hf', 'configs', `${name}.json`), 'utf8')));
const llama70b = config('llama-3.3-70b-instruct');
const gemma3 = config('gemma-3-27b-it');
const gemma4 = config('gemma-4-31b-it');
const deepseek = config('deepseek-v3');
const qwenNext = config('qwen3-next-80b-a3b-instruct');
const qwenMoe = config('qwen3-30b-a3b');
const layout = (f: ModelFactsEx) => kvLayoutOf(f).layout;
const kvMiB = (f: ModelFactsEx, len: number, elem: number, opts = {}) => servingKvBytesPerSeq(layout(f), len, elem, opts) / MIB;

describe('real config.json → KV layout', () => {
  it('dense GQA (Llama 3.3 70B): 80 layers × 8 KV heads × (128 + 128)', () => {
    expect(llama70b.kv).toEqual({ groups: [{ layers: 80, kvHeads: 8, keyLength: 128, valueLength: 128 }], recurrentLayers: 0, recurrentBytesPerSeq: 0 });
    expect(llama70b).toMatchObject({ embeddingLength: 8192, feedForwardLength: 28672, vocabSize: 128256 });
    expect(llama70b.visionEncoder).toBeUndefined();
  });

  it('sliding window (Gemma 3 27B): 52 layers keep 1024 tokens, 10 the whole context; SigLIP encoder', () => {
    expect(gemma3.kv?.groups).toEqual([
      { layers: 52, kvHeads: 16, keyLength: 128, valueLength: 128, window: 1024 },
      { layers: 10, kvHeads: 16, keyLength: 128, valueLength: 128 },
    ]);
    expect(gemma3.visionEncoder).toEqual({ hiddenSize: 1152, layers: 27 });
  });

  it('Gemma 4: the global layers have their own KV heads and head size (4 × 512)', () => {
    expect(gemma4.kv?.groups).toEqual([
      { layers: 50, kvHeads: 16, keyLength: 256, valueLength: 256, window: 1024 },
      { layers: 10, kvHeads: 4, keyLength: 512, valueLength: 512 },
    ]);
  });

  it('MLA (DeepSeek-V3): one latent of 512 + 64 per layer, no V; MoE expert size', () => {
    expect(deepseek.kv).toEqual({ groups: [{ layers: 61, kvHeads: 1, keyLength: 576, valueLength: 0 }], recurrentLayers: 0, recurrentBytesPerSeq: 0, latent: true });
    expect(deepseek).toMatchObject({ expertCount: 256, expertUsedCount: 8, expertFeedForwardLength: 2048, quant: 'FP8' });
  });

  it('hybrid (Qwen3-Next): 12 attention layers, 36 gated-DeltaNet states', () => {
    // State per layer: 32 × 128 × 128 + (4 − 1) × (2 × 16 × 128 + 32 × 128) = 548 864 values (f32 for llama.cpp);
    // vLLM keeps both parts in bf16 here (no mamba_ssm_dtype in the config).
    expect(qwenNext.kv).toEqual({
      groups: [{ layers: 12, kvHeads: 2, keyLength: 256, valueLength: 256 }],
      recurrentLayers: 36,
      recurrentBytesPerSeq: 36 * 548_864 * 4,
      recurrentState: { convElems: 24576, ssmElems: 524288, convBytes: 2, ssmBytes: 2 },
    });
  });

  it('MoE (Qwen3-30B-A3B): top-8 of experts of 768', () => {
    expect(qwenMoe).toMatchObject({ expertUsedCount: 8, expertFeedForwardLength: 768, expertCount: 128 });
  });
});

describe('KV per sequence (hand-computed)', () => {
  it('Llama 70B: 80 × 8 × 256 × 2 B = 320 KiB per token; tensor parallel splits the heads, replicating below one per GPU', () => {
    expect(kvMiB(llama70b, 32768, 2)).toBe(10240);
    expect(kvMiB(llama70b, 32768, 1)).toBe(5120);
    expect(kvMiB(llama70b, 32768, 2, { tp: 4 })).toBe(2560);
    expect(kvMiB(llama70b, 32768, 2, { tp: 16 })).toBe(1280);
    expect(kvHeadsPerGpu(8, 3)).toBe(2);
    // vLLM blocks of 16 tokens: 1000 tokens take 63 blocks.
    expect(servingKvBytesPerSeq(layout(llama70b), 1000, 2, { block: 16 })).toBe(327_680 * 1008);
  });

  it('Gemma 3: SGLang keeps the window, vLLM the window + one step − 1 tokens + one block', () => {
    // Full: 10 × 16 × 256 × 2 = 80 KiB per token → 2560 MiB; SWA: 52 × 16 × 256 × 2 × 1024 = 416 MiB.
    expect(kvMiB(gemma3, 32768, 2)).toBe(2560 + 416);
    // vLLM, 2048 tokens per step: min(32768, 1024 + 2047) = 3071 → 192 blocks + 1 = 3088 tokens × 0.40625 MiB.
    expect(kvMiB(gemma3, 32768, 2, { block: 16, swaSlack: 2047 })).toBe(2560 + 3088 * 0.40625);
    // Shorter than the window: every layer holds the context.
    expect(kvMiB(gemma3, 512, 2)).toBe((62 * 16 * 256 * 2 * 512) / MIB);
  });

  it('Gemma 4 at 256K: 10 × 4 × 1024 × 2 B per token on the global layers + 50 SWA layers × 1024 tokens', () => {
    expect(kvMiB(gemma4, 262144, 2)).toBe(20480 + 800);
  });

  it('DeepSeek-V3: 61 × 576 × 2 B per token, the same on every GPU', () => {
    expect(kvMiB(deepseek, 65536, 2)).toBe(4392);
    expect(kvMiB(deepseek, 65536, 2, { tp: 8 })).toBe(4392);
    expect(kvMiB(deepseek, 65536, 1, { tp: 8 })).toBe(2196);
  });

  it('Qwen3-Next: attention grows with the tokens, the recurrent state does not; both split by TP', () => {
    const state = (36 * 548_864 * 4) / MIB;
    expect(kvMiB(qwenNext, 131072, 2)).toBeCloseTo(3072 + state, 9);
    expect(kvMiB(qwenNext, 8, 2)).toBeCloseTo((12 * 2 * 512 * 2 * 8) / MIB + state, 9);
    expect(kvMiB(qwenNext, 131072, 2, { tp: 2 })).toBeCloseTo(1536 + state / 2, 9);
  });
});

describe('scheduler defaults and CUDA graph sizes (from the engines’ sources)', () => {
  it('vLLM: 2048 tokens / 256 sequences below 70 GiB, 8192 / 1024 from 70 GiB, 16384 / 1024 from 160 GiB', () => {
    expect(schedulerDefaults('vllm', 32607, 1)).toMatchObject({ batchedTokens: 2048, maxSeqs: 256 });
    expect(schedulerDefaults('vllm', 81559, 1)).toMatchObject({ batchedTokens: 8192, maxSeqs: 1024 });
    expect(schedulerDefaults('vllm', 180 * 1024, 1)).toMatchObject({ batchedTokens: 16384, maxSeqs: 1024 });
    expect(schedulerDefaults('vllm', 32607, 1).decodeGraphs).toHaveLength(51);
    expect(vllmGraphSizes(4, 2048)).toEqual([1, 2, 4, 8]);
    expect(vllmGraphSizes(256, 100)).toHaveLength(3 + 12);
  });

  it('SGLang: chunked prefill and graph batch by card size and TP', () => {
    expect(schedulerDefaults('sglang', 16 * 1024, 1)).toMatchObject({ batchedTokens: 2048, maxSeqs: 8 });
    expect(schedulerDefaults('sglang', 32607, 1)).toMatchObject({ batchedTokens: 4096, maxSeqs: 48 });
    expect(schedulerDefaults('sglang', 32607, 4)).toMatchObject({ batchedTokens: 4096, maxSeqs: 160 });
    expect(schedulerDefaults('sglang', 48 * 1024, 1)).toMatchObject({ batchedTokens: 4096, maxSeqs: 32 });
    expect(schedulerDefaults('sglang', 48 * 1024, 4)).toMatchObject({ maxSeqs: 160 });
    expect(schedulerDefaults('sglang', 81559, 1)).toMatchObject({ batchedTokens: 8192, maxSeqs: 256 });
    expect(schedulerDefaults('sglang', 81559, 8)).toMatchObject({ maxSeqs: 512 });
    expect(schedulerDefaults('sglang', 180 * 1024, 1)).toMatchObject({ batchedTokens: 16384, maxSeqs: 512 });
    expect(sglangDecodeGraphSizes(48)).toEqual([1, 2, 4, 8, 12, 16, 24, 32, 40, 48]);
    expect(sglangDecodeGraphSizes(600)).toHaveLength(5 + 31 + 15 + 3);
    expect(sglangPrefillGraphSizes(2048)).toHaveLength(42);
    expect(sglangPrefillGraphSizes(4096)).toHaveLength(50);
    expect(sglangPrefillGraphSizes(8192)).toHaveLength(58);
  });

  it("SGLang's own --mem-fraction-static", () => {
    // RTX 5090: 512 + 1.5 × 4096 + 1/8 GiB + 2 × 48 + 8 × 50 = 7280 MiB reserved → 0.777.
    expect(sglangDefaultMemFraction(32607, 1)).toBe(0.777);
    expect(sglangDefaultMemFraction(32607, 2)).toBe(0.773);
    expect(sglangDefaultMemFraction(32607, 4)).toBe(0.758);
    // MLA: 1.5 GiB instead of the prefill graphs → 8416 MiB.
    expect(sglangDefaultMemFraction(32607, 1, { latent: true })).toBe(0.742);
    // H100: 512 + 12288 + 128 + 512 + 464 = 13904 → 0.830; small limits on a big card: at least 10 GiB reserved.
    expect(sglangDefaultMemFraction(81559, 1)).toBe(0.83);
    expect(sglangDefaultMemFraction(81559, 1, {}, { batchedTokens: 2048, maxSeqs: 8, decodeGraphs: [8], prefillGraphs: [] })).toBe(0.874);
    // Vision: × 0.95 × (1 − 0.1 × (27 × 1152² / (24 × 1024²) − 1)), clamped to 0.8…1.05.
    expect(sglangDefaultMemFraction(32607, 1, { vision: { hiddenSize: 1152, layers: 27 } })).toBeCloseTo(0.777 * 0.95 * (1 - 0.1 * ((27 * 1152 ** 2) / (24 * 1024 ** 2) - 1)), 9);
    expect(sglangDefaultMemFraction(32607, 1, { vision: { hiddenSize: 512, layers: 2 } })).toBeCloseTo(0.777 * 0.95 * 1.05, 9);
    expect(sglangDefaultMemFraction(32607, 1, { vision: { hiddenSize: 4096, layers: 64 } })).toBeCloseTo(0.777 * 0.95 * 0.8, 9);
  });
});

describe('fixed costs', () => {
  const limits = schedulerDefaults('vllm', 32607, 1);
  it('from the model shape: dense, MoE, vision, NCCL with TP', () => {
    const dense = servingFixedCosts(llama70b, 'vllm', { tp: 1, limits, layers: 80 });
    expect(dense.shaped).toBe(true);
    // Forward 2 × (6 × 8192 + 3 × 28672) B per token × 2048 tokens = 528 MiB; the profile's fp32 logits for the
    // 2048 tokens, 2048 × 128256 × 4 B = 1002 MiB, are larger: 1002 … 1002 × 1.1 + 264 + 64.
    expect(dense.activations).toEqual({ min: 1002, max: Math.round(1002 * 1.1 + 264 + 64) });
    expect(dense.nccl).toEqual({ min: 0, max: 0 });
    expect(dense.vision).toEqual({ min: 0, max: 0 });
    expect(dense.total.min).toBe(dense.cudaContext.min + dense.workspaces.min + dense.activations.min + dense.cudaGraphs.min);
    // MoE: 2 × (6 × 2048 + 8 × (3 × 768 + 2048)) B per token beats the dense FFN of 6144.
    const moe = servingFixedCosts(qwenMoe, 'vllm', { tp: 1, limits, layers: 48 });
    const perToken = 2 * (6 * 2048 + 8 * (3 * 768 + 2048));
    // Its forward (2048 × perToken = 184 MiB) is below the logits: 2048 × 151936 × 4 B = 1187 MiB.
    expect((2048 * perToken) / MIB).toBeLessThan(1187);
    expect(moe.activations.min).toBe(1187);
    const tp4 = servingFixedCosts(llama70b, 'vllm', { tp: 4, limits, layers: 80 });
    expect(tp4.nccl).toEqual({ min: 135, max: 535 + 64 });
    // Pipeline stages talk to the next one through a communicator of their own, with or without TP.
    expect(servingFixedCosts(llama70b, 'vllm', { tp: 1, limits, layers: 80, pp: 2 }).nccl).toEqual({ min: 100, max: 500 });
    expect(servingFixedCosts(llama70b, 'vllm', { tp: 4, limits, layers: 80, pp: 2 }).nccl).toEqual({ min: 235, max: 1099 });
    expect(tp4.activations.max).toBeLessThan(dense.activations.max);
    const vision = servingFixedCosts(gemma3, 'sglang', { tp: 1, limits: schedulerDefaults('sglang', 32607, 1), layers: 62 });
    expect(vision.vision).toEqual({ min: Math.round(128 + 31104 / 64), max: Math.round(512 + 31104 / 16) });
  });

  it("falls back to the old size-based constants when the model's shape is unknown", () => {
    const blind = servingFixedCosts({ paramsB: 8 }, 'vllm', { tp: 1, limits, layers: 32 });
    expect(blind.shaped).toBe(false);
    expect(blind.activations).toEqual({ min: 768, max: 1536 });
    expect(servingFixedCosts({ embeddingLength: 4096 }, 'vllm', { tp: 1, limits, layers: 32 }).shaped).toBe(false);
  });
});

describe('estimateServing', () => {
  const base = (f: ModelFactsEx, over: Partial<ServingInput>): ServingInput => ({ facts: f, layout: layout(f), kvBytes: 2, weightsBytes: 0, engine: 'vllm', totalMiB: 32607, tokens: 32768, ...over });
  const llamaBytes = 141_107_412_992;

  it('vLLM: the share holds weights, activations, CUDA graphs and KV; context and NCCL sit outside it', () => {
    const e = estimateServing(base(llama70b, { weightsBytes: llamaBytes, totalMiB: 81920, tp: 4, seqs: 3, tokens: 131072 }));
    expect(e.shareMiB).toEqual({ min: 73728, max: 73728 });
    expect(e.weightsMiB).toBe(Math.round(llamaBytes / 4 / MIB));
    expect(e.kvPerSeqMiB).toBe(10240);
    const inside = [e.fixed.workspaces, e.fixed.activations, e.fixed.cudaGraphs];
    expect(e.insideFixedMiB).toEqual({ min: inside.reduce((a, x) => a + x.min, 0), max: inside.reduce((a, x) => a + x.max, 0) });
    expect(e.kvRoomMiB).toEqual({ min: 73728 - e.weightsMiB - e.insideFixedMiB.max, max: 73728 - e.weightsMiB - e.insideFixedMiB.min });
    expect(e.outsideNeedMiB).toEqual({ min: e.fixed.cudaContext.min + e.fixed.nccl.min, max: e.fixed.cudaContext.max + e.fixed.nccl.max });
    expect(e.outsideRoomMiB).toEqual({ min: 8192, max: 8192 });
    expect(e.concurrency.min).toBe(Math.floor(e.kvRoomMiB.min / 10240));
    expect(e.verdict).toBe('fits');
    expect(e.reasons).toEqual([]);
    expect(e.limits).toMatchObject({ batchedTokens: 8192, maxSeqs: 1024, graphMaxBatch: 512 });
    expect(e.marginMiB.min).toBe(e.kvRoomMiB.min - 30720);
    expect(e.processMiB).toEqual({ min: 73728 + e.outsideNeedMiB.min, max: 73728 + e.outsideNeedMiB.max });
    expect(e.shaped).toBe(true);
  });

  it('weights over the share, KV over the room, or only within the range', () => {
    const big = estimateServing(base(llama70b, { weightsBytes: llamaBytes }));
    // A pipeline stage captures CUDA graphs for its own layers only: fewer graph pieces than the whole stack.
    expect(estimateServing(base(llama70b, { pp: 4 })).fixed.cudaGraphs.max).toBeLessThan(estimateServing(base(llama70b, {})).fixed.cudaGraphs.max);
    expect(big.verdict).toBe('no');
    expect(big.reasons[0]).toMatch(/^the weights \(131\.4 GiB\) do not fit the engine's share/);
    expect(estimateServing(base(llama70b, { weightsBytes: llamaBytes, tp: 2, totalMiB: 32607 })).reasons[0]).toMatch(/per GPU\)/);
    const room = estimateServing(base(llama70b, { weightsBytes: 10 * 1024 * MIB, kvBytes: 1 }));
    const tooMany = estimateServing(base(llama70b, { weightsBytes: 10 * 1024 * MIB, kvBytes: 1, seqs: room.concurrency.max + 1 }));
    expect(tooMany.verdict).toBe('no');
    expect(tooMany.reasons[0]).toMatch(/requests .* exceeds the room/);
    // Pick weights so the KV fits only with the smaller fixed costs.
    const w = room.shareMiB.max - room.insideFixedMiB.min - room.kvPerSeqMiB - 1;
    const edge = estimateServing(base(llama70b, { weightsBytes: w * MIB, kvBytes: 1 }));
    expect(edge.verdict).toBe('uncertain');
    expect(edge.reasons[0]).toMatch(/low end/);
    const single = estimateServing(base(llama70b, { weightsBytes: 27 * 1024 * MIB, kvBytes: 1 }));
    expect(single.reasons[0]).toMatch(/of one request/);
  });

  it('vLLM with a share so large the CUDA context does not fit beside it', () => {
    const w = 8 * 1024 * MIB;
    expect(estimateServing(base(qwenMoe, { weightsBytes: w, util: 0.995, tokens: 4096 }))).toMatchObject({ verdict: 'no' });
    const tight = estimateServing(base(qwenMoe, { weightsBytes: w, util: 0.985, tokens: 4096 }));
    expect(tight.verdict).toBe('uncertain');
    expect(tight.reasons.at(-1)).toMatch(/the CUDA context and NCCL/);
  });

  it('SGLang: the fraction applies to what is free after context and NCCL; activations and graphs use the rest', () => {
    const e = estimateServing(base(qwenMoe, { engine: 'sglang', weightsBytes: 16 * 1024 * MIB, tokens: 40960 }));
    expect(e.util).toBe(0.777);
    expect(e.utilDefault).toBe(true);
    expect(e.shareMiB).toEqual({ min: Math.round(0.777 * (32607 - 600)), max: Math.round(0.777 * (32607 - 250)) });
    // Inside: the workspaces made before the pool.
    expect(e.insideFixedMiB).toEqual({ min: 200, max: 800 });
    expect(e.kvRoomMiB).toEqual({ min: e.shareMiB.min - 16384 - 800, max: e.shareMiB.max - 16384 - 200 });
    expect(e.outsideNeedMiB).toEqual({ min: e.fixed.activations.min + e.fixed.cudaGraphs.min, max: e.fixed.activations.max + e.fixed.cudaGraphs.max });
    expect(e.limits).toMatchObject({ batchedTokens: 4096, maxSeqs: 48, graphs: 10 + 50 });
    expect(e.notes).toContain('--mem-fraction-static 0.777 is what SGLang computes for this GPU');
    expect(e.processMiB.max).toBe(e.shareMiB.max + 600 + e.outsideNeedMiB.max);
    // An explicit fraction, fp8, custom scheduler limits.
    const fp8 = estimateServing(base(qwenMoe, { engine: 'sglang', weightsBytes: 16 * 1024 * MIB, util: 0.85, kvBytes: 1, maxBatchedTokens: 8192, maxSeqs: 8 }));
    expect(fp8.utilDefault).toBe(false);
    expect(fp8.limits).toMatchObject({ batchedTokens: 8192, maxSeqs: 8, graphMaxBatch: 8 });
    expect(fp8.notes).toContain('in SGLang the fp8 cache is requested with --kv-cache-dtype fp8_e4m3 (or fp8_e5m2)');
    // Too high a fraction: activations and graphs cannot fit what is left.
    expect(estimateServing(base(qwenMoe, { engine: 'sglang', weightsBytes: MIB, util: 0.99 })).verdict).toBe('no');
    expect(estimateServing(base(qwenMoe, { engine: 'sglang', weightsBytes: MIB, util: 0.95 })).verdict).toBe('uncertain');
    // Vision models: the embedding cache comes out of the pool, and the default fraction is lower.
    const vision = estimateServing(base(gemma3, { engine: 'sglang', weightsBytes: MIB }));
    expect(vision.insideFixedMiB).toEqual({ min: 300, max: 900 });
    expect(vision.util).toBeLessThan(0.777);
    expect(vision.notes.some((n) => n.includes('swa_full_tokens_ratio'))).toBe(true);
  });

  it('custom vLLM limits and notes: context over the model, SWA, recurrent layers, MLA and replicated heads under TP', () => {
    const custom = estimateServing(base(gemma3, { weightsBytes: MIB, maxBatchedTokens: 8192, maxSeqs: 64 }));
    expect(custom.limits).toMatchObject({ batchedTokens: 8192, maxSeqs: 64, graphMaxBatch: 128 });
    expect(custom.notes).toContain('sliding-window layers: each holds the window + one step’s tokens (8192), not the whole context');
    const over = estimateServing(base(qwenMoe, { weightsBytes: MIB, tokens: 65536 }));
    expect(over.notes[0]).toBe('65536 tokens exceed the model’s context (40960): the engine refuses them without a RoPE override');
    expect(estimateServing(base(qwenNext, { weightsBytes: MIB, tp: 4 })).notes).toEqual([
      '36 recurrent layers: a fixed state per request, it does not grow with the tokens',
      'fewer KV heads than GPUs: with TP 4 the heads are replicated',
    ]);
    expect(estimateServing(base(deepseek, { weightsBytes: MIB, tp: 8 })).notes).toEqual(['MLA: the KV latent is replicated on every GPU, TP does not split it']);
  });

  it('MLA under SGLang: its default fraction reserves 1.5 GiB instead of the prefill graphs; a failing launch stays "no"', () => {
    expect(estimateServing(base(deepseek, { engine: 'sglang', weightsBytes: MIB })).util).toBe(0.742);
    // Weights over the share and too little room outside it: still "no", with both reasons.
    const both = estimateServing(base(qwenMoe, { engine: 'sglang', weightsBytes: 40 * 1024 * MIB, util: 0.95 }));
    expect(both.verdict).toBe('no');
    expect(both.reasons).toHaveLength(2);
  });

  it('no attention at all and unknown layer counts', () => {
    const ssm: ModelFactsEx = { kv: { groups: [], recurrentLayers: 4, recurrentBytesPerSeq: 0 } };
    const e = estimateServing(base(ssm, { weightsBytes: MIB }));
    expect(e.kvPerSeqMiB).toBe(0);
    expect(e.concurrency).toEqual({ min: 0, max: 0 });
    expect(e.fixed.cudaGraphs.min).toBe(Math.round(32 + 51 * 6 * 0.05 + 64));
  });
});

it('gbText: GiB = 1024 MiB, one decimal', () => {
  expect(gbText(1536)).toBe('1.5 GiB');
});

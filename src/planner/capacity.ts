/**
 * How many users a layout serves reliably: the smaller of two limits, both reported.
 *  - memory: requests whose KV cache fits at once, at the full context (worst case) and at the typical
 *    one (vLLM allocates blocks as requests grow; prefix caching shares cached prompts);
 *  - speed: concurrent requests that still get `minTokS` tokens/s each and their first token within
 *    `maxTtftMs`. With chunked prefill a new request's prompt is read in slices of the step's tokens
 *    between decode steps, so prompts slow everyone's generation; overlapping prompts queue.
 * "Reliable" (guaranteed) uses the slow end of the speed ranges and keeps a safety margin (`safety`, 85 % by
 * default) on the memory limit; "probable" is the same count with the centre of the speed ranges (what the
 * measurements usually land on). Total users = concurrent requests ÷ the share of users active at once.
 */
import type { MsRange } from './perf';

export interface Workload {
  /** max_model_len: the longest request, tokens. */
  maxContext: number;
  /** A typical request's context (prompt + history), tokens; default half of maxContext. */
  typicalContext: number;
  /** Tokens generated per request. */
  outputTokens: number;
  /** Generation speed each user must keep, tokens/s. */
  minTokS: number;
  /** Longest acceptable wait for the first token under load, ms. */
  maxTtftMs: number;
  /**
   * The wait limit grows with the context: at least twice the time the typical prompt takes alone (a
   * 1M-token prompt takes minutes by itself). Off when the user set the limit.
   */
  ttftScales?: boolean;
  /** Share of users generating at the same moment (e.g. 0.1). */
  activeShare: number;
  /** Share of the memory limit used at most (e.g. 0.85). */
  safety: number;
  /** Share of each prompt already in the prefix cache (system prompt, shared documents), 0–1. */
  cachedShare: number;
}

export const DEFAULT_WORKLOAD = { outputTokens: 500, minTokS: 20, maxTtftMs: 10_000, activeShare: 0.1, safety: 0.85, cachedShare: 0, ttftScales: true } as const;

export interface SpeedModel {
  /** Decode step time at `batch` concurrent sequences of the typical context, ms. */
  stepMs: (batch: number) => MsRange;
  /** GPU time to read a prompt of `tokens` alone, ms. */
  prefillMs: (tokens: number) => MsRange;
  /** Wait of every request outside the GPU (HTTP, tokenizer, scheduling), ms. */
  requestOverheadMs: MsRange;
  /** Tokens of prompt read per scheduler step (max_num_batched_tokens). */
  batchedTokens: number;
  /**
   * Pipeline stages (1 without pipeline parallelism). prefillMs is one prompt crossing every stage; under load the
   * stages work on different requests at once, so each carries 1/stages of the reading.
   */
  stages?: number;
}

export interface LoadPoint {
  batch: number;
  /** Tokens/s each user sees while prompts are being read between steps. */
  tokS: MsRange;
  /** First-token wait under that load, ms. */
  ttftMs: MsRange;
  /** Share of GPU time spent reading prompts. */
  prefillShare: MsRange;
}

/** Speed and first-token wait at `batch` concurrent requests (best and worst case). */
export function loadPoint(speed: SpeedModel, w: Workload, batch: number): LoadPoint {
  const prompt = Math.max(1, Math.round(w.typicalContext * (1 - w.cachedShare)));
  const step = speed.stepMs(batch);
  const pre = speed.prefillMs(prompt);
  const at = (s: number, p: number) => {
    // Each request, once done, is followed by a new one: per decode step the scheduler also reads
    // batch × prompt / output tokens of prompts, spread over the pipeline stages.
    const prefillPerStep = (batch * p) / w.outputTokens / (speed.stages ?? 1);
    const stepEff = s + prefillPerStep;
    const share = prefillPerStep / stepEff;
    // The prompt is read in slices between steps; others' prompts ahead of it make it queue (M/D/1).
    const slices = Math.ceil(prompt / speed.batchedTokens);
    // Only the other users' prompts can be ahead (none with a single user); share < 1 always holds.
    const others = (share * (batch - 1)) / batch;
    // Each stage serves a prompt for p / stages of its time: that is the queue's service time.
    const queue = (others / (2 * (1 - others))) * (p / (speed.stages ?? 1));
    return { tokS: 1000 / stepEff, ttft: p + slices * s + queue, share };
  };
  const best = at(step.min, pre.min);
  const worst = at(step.max, pre.max);
  return {
    batch,
    tokS: { min: worst.tokS, max: best.tokS },
    ttftMs: { min: best.ttft + speed.requestOverheadMs.min, max: worst.ttft + speed.requestOverheadMs.max },
    prefillShare: { min: best.share, max: worst.share },
  };
}

export interface Capacity {
  /** Requests at the full context whose KV fits at once (vLLM's own count). */
  memoryWorst: number;
  /** Requests at the typical context whose KV fits at once. */
  memoryTypical: number;
  /** Concurrent requests that keep minTokS and maxTtftMs on the slow end. */
  speedLimit: number;
  /** min(memory × safety, speed): what to promise (guaranteed, slow end of the speed ranges). */
  reliable: number;
  /** The same with the centre of the speed ranges: what to expect. */
  probable: number;
  /** The load point at `probable` with the centre of the speed ranges. */
  atProbable: LoadPoint;
  limitedBy: 'memory' | 'speed' | 'wait' | 'none';
  /** The load point at `reliable` (or at 1 request when nothing is reliable). */
  atReliable: LoadPoint;
  /** One request alone: tokens/s and the time to read a full-context prompt. */
  single: { tokS: MsRange; fullPrefillMs: MsRange; typicalPrefillMs: MsRange };
  /** Users in total for `reliable` concurrent requests at the active share. */
  totalUsers: number;
  /** Users in total for `probable` concurrent requests. */
  probableUsers: number;
  /** The first-token wait limit applied, ms (maxTtftMs, or more for long contexts with ttftScales). */
  ttftLimitMs: number;
}

/** A prompt's reading time as a user waits for it: GPU time + the request overhead. */
function withOverhead(speed: SpeedModel, gpu: MsRange): MsRange {
  return { min: gpu.min + speed.requestOverheadMs.min, max: gpu.max + speed.requestOverheadMs.max };
}

/** The speed model at the centre of every range (the probable case). */
function central(speed: SpeedModel): SpeedModel {
  const mid = (r: MsRange): MsRange => ({ min: (r.min + r.max) / 2, max: (r.min + r.max) / 2 });
  return { stepMs: (b) => mid(speed.stepMs(b)), prefillMs: (t) => mid(speed.prefillMs(t)), requestOverheadMs: mid(speed.requestOverheadMs), batchedTokens: speed.batchedTokens, stages: speed.stages ?? 1 };
}

/** The largest batch up to `memCap` that keeps both targets (speed falls and waits grow with the batch). */
function speedLimitOf(speed: SpeedModel, w: Workload, memCap: number, limit: number): number {
  const ok = (b: number) => {
    const p = loadPoint(speed, w, b);
    return p.tokS.min >= w.minTokS && p.ttftMs.max <= limit;
  };
  let lo = 0;
  let hi = Math.max(1, memCap) + 1;
  if (ok(hi - 1)) lo = hi - 1;
  else
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (ok(mid)) lo = mid;
      else hi = mid;
    }
  return lo;
}

/** The reliable concurrency of a layout (module comment). */
export function capacity(speed: SpeedModel, w: Workload, memory: { worst: number; typical: number }): Capacity {
  const memCap = Math.floor(memory.typical * w.safety);
  const alone = loadPoint(speed, w, 1).ttftMs.max;
  const limit = w.ttftScales ? Math.max(w.maxTtftMs, 2 * alone) : w.maxTtftMs;
  const speedLimit = speedLimitOf(speed, w, memCap, limit);
  const reliable = Math.min(memCap, speedLimit);
  const mid = central(speed);
  const probable = Math.max(reliable, Math.min(memCap, speedLimitOf(mid, w, memCap, limit)));
  let limitedBy: Capacity['limitedBy'] = 'none';
  if (reliable === memCap && memCap <= speedLimit) limitedBy = 'memory';
  else {
    const p = loadPoint(speed, w, Math.max(1, speedLimit + 1));
    limitedBy = p.tokS.min < w.minTokS ? 'speed' : 'wait';
  }
  const step1 = speed.stepMs(1);
  return {
    memoryWorst: memory.worst,
    memoryTypical: memory.typical,
    speedLimit,
    reliable,
    limitedBy,
    atReliable: loadPoint(speed, w, Math.max(1, reliable)),
    probable,
    atProbable: loadPoint(mid, w, Math.max(1, probable)),
    single: { tokS: { min: 1000 / step1.max, max: 1000 / step1.min }, fullPrefillMs: withOverhead(speed, speed.prefillMs(w.maxContext)), typicalPrefillMs: withOverhead(speed, speed.prefillMs(w.typicalContext)) },
    totalUsers: Math.floor(reliable / w.activeShare),
    probableUsers: Math.floor(probable / w.activeShare),
    ttftLimitMs: limit,
  };
}

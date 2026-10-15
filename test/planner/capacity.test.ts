import { describe, expect, it } from 'vitest';
import { capacity, DEFAULT_WORKLOAD, loadPoint, type SpeedModel, type Workload } from '../../src/planner/capacity';

/** A fixed decode step of 10–20 ms, prompts read at 4 and 2 tokens/ms, 10–100 ms of request overhead. */
const speed: SpeedModel = {
  stepMs: () => ({ min: 10, max: 20 }),
  prefillMs: (t) => ({ min: t / 4, max: t / 2 }),
  requestOverheadMs: { min: 10, max: 100 },
  batchedTokens: 512,
};
const work: Workload = { maxContext: 4000, typicalContext: 2000, outputTokens: 100, minTokS: 1, maxTtftMs: 10_000, activeShare: 0.1, safety: 0.85, cachedShare: 0.5 };

describe('loadPoint', () => {
  it('prompts read between steps slow generation; the other users’ prompts queue (M/D/1)', () => {
    const p = loadPoint(speed, work, 4);
    // Prompt: 2000 × (1 − 0.5) = 1000 tokens, 2 slices of 512.
    // Best: prompt 250 ms; per step 4 × 250 / 100 = 10 ms of prompts → step 20 ms, 50 tok/s, share ½;
    // the 3 others' share ½ × ¾ = ⅜ → queue ⅜ / (2 × ⅝) × 250 = 75 ms; first token 250 + 2 × 10 + 75 + 10 = 355 ms.
    // Worst: prompt 500 ms; 20 ms of prompts → step 40 ms, 25 tok/s; queue 150; 500 + 2 × 20 + 150 + 100 = 790.
    expect(p).toEqual({ batch: 4, tokS: { min: 25, max: 50 }, ttftMs: { min: 355, max: 790 }, prefillShare: { min: 0.5, max: 0.5 } });
  });

  it('pipeline stages: each carries 1/stages of the reading under load; one prompt alone still crosses them all', () => {
    const p = loadPoint({ ...speed, stages: 2 }, work, 4);
    // Best: 4 × 250 / 100 / 2 = 5 ms of prompts per step → step 15 ms, 66.7 tok/s, share ⅓ (½ with one stage).
    expect(p.tokS.max).toBeCloseTo(1000 / 15, 9);
    expect(p.prefillShare.min).toBeCloseTo(1 / 3, 9);
    // The first token still waits for the whole prompt (250 ms best), plus less queueing than with one stage.
    expect(p.ttftMs.min).toBeGreaterThan(250);
    expect(p.ttftMs.min).toBeLessThan(loadPoint(speed, work, 4).ttftMs.min);
    // The queue's service time is a stage's share of the prompt: 2 stages, queue ⅙ / (2 × ⅚) × 250 / 2 = 12.5 ms best.
    const others = (1 / 3) * (3 / 4);
    expect(p.ttftMs.min).toBeCloseTo(250 + 2 * 10 + (others / (2 * (1 - others))) * (250 / 2) + 10, 6);
  });

  it('a fully cached prompt still reads one token', () => {
    // 1 token: ¼ ms best; per step 1 × ¼ / 100.
    const p = loadPoint(speed, { ...work, cachedShare: 1 }, 1);
    expect(p.tokS.max).toBeCloseTo(1000 / (10 + 0.0025), 9);
  });

  it('a single user never queues behind its own prompt', () => {
    const p = loadPoint(speed, work, 1);
    // 250 + 2 × 10 + 0 + 10 and 500 + 2 × 20 + 0 + 100.
    expect(p.ttftMs).toEqual({ min: 280, max: 640 });
  });
});

describe('capacity', () => {
  it('limited by memory: the KV room × safety, then users at the active share', () => {
    const fast: SpeedModel = { ...speed, prefillMs: (t) => ({ min: t / 1000, max: t / 500 }) };
    const c = capacity(fast, { ...work, cachedShare: 0 }, { worst: 6, typical: 10 });
    // floor(10 × 0.85) = 8 requests; the speed allows more.
    expect(c).toMatchObject({ memoryWorst: 6, memoryTypical: 10, speedLimit: 8, reliable: 8, limitedBy: 'memory', totalUsers: 80 });
    expect(c.atReliable.batch).toBe(8);
    // One request alone: 1000 / 20 … 1000 / 10 tok/s; full context 4000 / 1000 + 10 … 4000 / 500 + 100 ms.
    expect(c.single).toEqual({ tokS: { min: 50, max: 100 }, fullPrefillMs: { min: 14, max: 108 }, typicalPrefillMs: { min: 12, max: 104 } });
  });

  it('limited by speed: each user’s tokens/s', () => {
    const slow: SpeedModel = { ...speed, stepMs: (b) => ({ min: 10 * b, max: 20 * b }), prefillMs: () => ({ min: 0, max: 0 }) };
    const c = capacity(slow, { ...work, minTokS: 20 }, { worst: 100, typical: 100 });
    // Worst 1000 / (20 × b) ≥ 20 → b ≤ 2.5: 2 requests of the 85 memory allows.
    expect(c).toMatchObject({ speedLimit: 2, reliable: 2, limitedBy: 'speed', totalUsers: 20 });
    // Probable: the centre of the step, 15 × b ms: 1000 / (15 × b) ≥ 20 → b ≤ 3.33: 3 requests, 30 users.
    expect(c).toMatchObject({ probable: 3, probableUsers: 30 });
    expect(c.atProbable.tokS.min).toBeCloseTo(1000 / 45, 9);
  });

  it('limited by the first-token wait', () => {
    const c = capacity(speed, { ...work, typicalContext: 1000, cachedShare: 0, maxTtftMs: 1000 }, { worst: 100, typical: 100 });
    // Worst at b: prompt 500 ms, 5b ms of prompts per step; others' share 5(b − 1) / (20 + 5b);
    // b = 8: 35/60 → queue 0.7 × 500 = 350 → 640 + 350 = 990 ≤ 1000; b = 9: 40/65 → 400 → 1040.
    expect(c).toMatchObject({ speedLimit: 8, reliable: 8, limitedBy: 'wait', totalUsers: 80 });
    expect(c.atReliable.ttftMs.max).toBeCloseTo(990, 9);
  });

  it('a long context: the wait limit grows to twice the prompt read alone', () => {
    // Alone (worst): 500 + 2 × 20 + 100 = 640 ms → limit max(500, 1280) = 1280; fixed 500 ms would allow nobody.
    const scaled = capacity(speed, { ...work, typicalContext: 1000, cachedShare: 0, maxTtftMs: 500, ttftScales: true }, { worst: 100, typical: 100 });
    expect(scaled.ttftLimitMs).toBe(1280);
    expect(scaled.reliable).toBeGreaterThan(0);
    expect(capacity(speed, { ...work, typicalContext: 1000, cachedShare: 0, maxTtftMs: 500 }, { worst: 100, typical: 100 })).toMatchObject({ reliable: 0, ttftLimitMs: 500 });
  });

  it('no room for a single typical request: memory, zero users', () => {
    const c = capacity(speed, work, { worst: 0, typical: 1 });
    // floor(1 × 0.85) = 0.
    expect(c).toMatchObject({ reliable: 0, limitedBy: 'memory', totalUsers: 0 });
    expect(c.atReliable.batch).toBe(1);
  });

  it('defaults: 500 tokens out, 20 tok/s, 10 s of wait, 10 % active, 85 % of the memory', () => {
    expect(DEFAULT_WORKLOAD).toEqual({ outputTokens: 500, minTokS: 20, maxTtftMs: 10_000, activeShare: 0.1, safety: 0.85, cachedShare: 0, ttftScales: true });
  });
});

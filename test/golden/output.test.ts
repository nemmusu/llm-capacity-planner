/**
 * The command's whole text answer for eight real cases on the built-in models, against golden files
 * (test/golden/*.txt): any change to a number, a word or the layout shows up in review. 120 columns, no colour,
 * no GPU on this PC, no network. After an intended change: `npx vitest run test/golden -u`, then read the diff.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { catalogPack, runCli, tempFolder } from '../helpers/catalog';

const CASES: [file: string, argv: string[], what: string][] = [
  ['small-dense', ['ibm-granite/granite-4.2-3b', '--tokens', '32k'], 'a small dense model on every card'],
  // The pack's largest dense model (Mistral Medium 3.5 is not redistributed: scripts/pack-models.mjs).
  ['dense-31b', ['google/gemma-4-31B-it', '--tokens', '32k'], 'the largest dense model of the pack (31B)'],
  ['large-moe', ['tencent/Hy3-FP8', '--tokens', '32k'], 'a large MoE (299B, FP8)'],
  ['glm-5.3-128k', ['zai-org/GLM-5.3', '--tokens', '128k'], 'GLM-5.3 (MLA, 753B) at 128k tokens'],
  ['deepseek-v4.1-pro-6000', ['deepseek-ai/DeepSeek-V4.1-Flash', '--tokens', '32k', '--gpu', 'pro-6000', '--count', '8'], 'DeepSeek-V4.1 on the RTX PRO 6000 (sm120): the engine does not start'],
  ['users-200', ['Qwen/Qwen3.8-27B', '--tokens', '32k', '--users', '200'], 'which hardware serves 200 users'],
  ['chat-prompt-cache-80', ['Qwen/Qwen3.8-27B-FP8', '--tokens', '32k', '--prompt-cache', '80'], 'a chat: 80% of each request already in the prefix cache'],
  ['qwen3.6-1m-yarn', ['Qwen/Qwen3.6-35B-A3B', '--tokens', '1M'], '1M tokens, beyond the native 256k (YaRN)'],
  ['glm-5.3-268k-users-200', ['zai-org/GLM-5.3', '--tokens', '268k', '--users', '200'], 'GLM-5.3 at 268k tokens for 200 users: several servers'],
  ['glm-5.3-1m-users-200', ['zai-org/GLM-5.3', '--users', '200'], 'GLM-5.3 at 1M for 200 users: too slow, and why'],
  ['gpt-oss-1m-outside', ['openai/gpt-oss-120b', '--tokens', '1M'], 'gpt-oss-120b at 1M, 8 times its maximum: outside the model limits'],
];

/** More cases for the claims: a chat, one card selected, more GPUs than one server, a context the model was not made for. */
const CLAIMS: string[][] = [
  ...CASES.map(([, argv]) => argv),
  ['zai-org/GLM-5.3'],
  ['openai/gpt-oss-120b', '--tokens', '32k', '--prompt-cache', '80'],
  ['openai/gpt-oss-120b', '--tokens', '32k', '--prompt-cache', '80', '--gpu', 'h100-sxm', '--count', '1', '--users', '2000'],
  ['zai-org/GLM-5.3', '--tokens', '128k', '--users', '200', '--gpu', 'h200-sxm', '--count', '16'],
  ['zai-org/GLM-5.3', '--gpu', 'gh200', '--count', '16'],
  ['moonshotai/Kimi-K2.7-Code', '--tokens', '128k', '--users', '300', '--active', '30'],
  // The first-token wait as the binding limit; a speed the calibration does not trust; the memory as the limit.
  ['zai-org/GLM-5.3', '--tokens', '128k', '--max-wait', '0.5'],
  ['moonshotai/Kimi-K2.7-Code', '--gpu', 'h200', '--users', '300', '--active', '30'],
  ['moonshotai/Kimi-K3'],
];

const cache = tempFolder('golden-');
beforeAll(() => {
  catalogPack();
}, 120_000);
afterAll(() => cache.cleanup());

describe('the summary on every golden case', () => {
  it.each(CASES)('%s: 4 to 6 plain lines, then the technical detail', async (_file, argv) => {
    const r = await runCli(argv, cache.dir, { columns: 120, isTTY: false, localGpu: () => Promise.resolve(undefined) });
    const lines = r.text.split('\n');
    expect(lines[0]).toBe('Summary');
    const end = lines.indexOf('');
    const summary = lines.slice(1, end).join(' ').replace(/ {2,}/g, ' ');
    const items = summary.split(/(?=\b(?:Recommendation|Warning|Alternative|Not enough|Speed for each user|Confidence):)/).map((s) => s.trim()).filter(Boolean);
    expect(items.length).toBeGreaterThanOrEqual(2);
    expect(items.length).toBeLessThanOrEqual(6);
    // A recommendation, none (with the cause), or none reliable (figures marked "do not plan", with the reason).
    expect(items[0]).toMatch(/^Recommendation: (\d+ servers? with \d+× NVIDIA |none — |none reliable — )/);
    const users = argv.indexOf('--users');
    // With --users: as many servers as the users take, never a smaller scenario (or none, with the cause).
    if (users >= 0 && !items[0]?.startsWith('Recommendation: none')) {
      const total = Number(argv[users + 1]).toLocaleString('en-US');
      expect(items[0]).toMatch(new RegExp(`^Recommendation: (1 server with .* serves|\\d+ servers with .* each serve) ${total} users `));
      // Servers × users per server covers everyone; the copies and the load balancer explained.
      const many = /each serve [\d,]+ users .*: ≈ ([\d,]+) users per server\. Each server runs (its own copy|\d+ copies) of the model, and a load balancer \(a router in front of the servers\) spreads the users across them\./.exec(items[0] as string);
      const n = /^Recommendation: ([\d,]+) servers/.exec(items[0] as string);
      if (n) {
        expect(many).not.toBeNull();
        const num = (x: string) => Number(x.replace(/,/g, ''));
        expect(num(n[1] as string) * num(many?.[1] as string)).toBeGreaterThanOrEqual(Number(argv[users + 1]));
      }
    }
    expect(items.at(-1)).toMatch(/^Confidence: memory (high|medium|low, not precise enough to decide) \(±[\d.]+%\); (writing speed .+; reading time .+; user count.+|speed not estimated)\. (Plan on the guaranteed figures|Nothing is guaranteed here: do not rely on these figures alone|Do not plan on these user counts: .+)\.$/);
    // The size: the maximum and the average the capacity is computed at.
    if (items[0]?.includes('documents of up to') && !items[0].includes('the model maximum')) expect(items[0]).toMatch(/documents of up to [\d.]+[kM]? tokens \(.*; [\d.]+[kM]? on average\)/);
    expect(summary).toMatch(/≈ [\d,]+ words ≈ [\d,]+ pages?|does not start|does not fit|too slow at|the memory left|outside the model's limits/);
    // No jargon in the summary.
    expect(summary).not.toMatch(/\bTP\d|\bPP\d|KV cache|tok\/s|TTFT/);
    expect(lines[end + 1]).toBe('Technical detail');
  });
});

interface ClaimCard {
  gpuId: string;
  gpu: string;
  status: string;
  serverGpus?: number;
  maxGpus: number;
  figures: { segment: string; users?: { guaranteed: number; likely: number }; tokS?: { value: number; min: number }; verdict: string };
}
interface ClaimJson {
  summary: string[];
  headline: string;
  cards: ClaimCard[];
  rows: { gpuId: string; verdict: string; users?: { guaranteed: number } }[];
  fit?: { figures: { gpuId: string; gpu: string } };
  users?: { choice?: { figures: { gpuId: string; gpu: string } } };
}

/** "H100 SXM5 80GB" → "H100", "RTX PRO 6000 WS" → "RTX PRO 6000" (as the summary names families). */
const family = (gpu: string) => gpu.replace(/( (\d+GB|PCIe|SXM\d?|NVL|WS|Max-Q|Server))+$/, '');

describe('every claim of the summary agrees with the tables it sums up', () => {
  it.each(CLAIMS.map((argv) => [argv.join(' '), argv] as const))('%s', async (_name, argv) => {
    const r = await runCli([...argv, '--json', 'claims.json'], cache.dir, { columns: 120, isTTY: false, localGpu: () => Promise.resolve(undefined) });
    expect(r.code).toBe(0);
    const j = JSON.parse(r.written['claims.json'] as string) as ClaimJson;
    const line = (k: string) => j.summary.find((l) => l.startsWith(`${k}:`));
    const rec = line('Recommendation') as string;
    const not = line('Not enough')?.replace(/^Not enough: (every card checked: )?/, '').replace(/\.$/, '');
    // 1. The card recommended is never "not enough".
    const picked = rec.startsWith('Recommendation: none') ? undefined : (j.users?.choice ?? j.fit)?.figures;
    if (picked && not) {
      expect(not).not.toContain(picked.gpu);
      expect(not).not.toMatch(new RegExp(`(^|, |; | and )${family(picked.gpu)}(,|;| and|$)`));
    }
    // 2. "Too slow": no layout guarantees anyone, and the closest one quoted is below the minimum at the low end.
    if (/^Recommendation: none — too slow/.test(rec)) {
      expect(j.rows.filter((x) => x.verdict !== 'no-fit' && x.verdict !== 'no-start' && (x.users?.guaranteed ?? 0) > 0)).toEqual([]);
      // Words per second for the reader (0.75 per token): the low end below the 20 tokens/s minimum (15 words).
      const quoted = /writing about (\d+) words per second per user under load \((\d+) to (\d+); (\d+) tokens\/s\)/.exec(rec);
      expect(quoted).not.toBeNull();
      expect(Number(quoted?.[2])).toBeLessThanOrEqual(15);
    }
    // 3. The confidence label agrees with the ± it shows.
    const conf = line('Confidence') as string;
    for (const [, label, span] of conf.matchAll(/(high|medium|low, not precise enough to decide) \(([^)]*)\)/g)) {
      const pcts = [...(span as string).matchAll(/(\d+(?:\.\d+)?)%/g)].map((x) => Number(x[1]));
      const max = Math.max(...pcts);
      if (label === 'high') expect(max).toBeLessThanOrEqual(15);
      else if (label === 'medium') expect(max).toBeLessThan(50);
      else if (!(span as string).startsWith('more than')) expect(span).toMatch(/, but /);
    }
    // 4. "even with N" is the most GPUs of each card named in one server.
    for (const part of not?.split('; ') ?? []) {
      const [names, n] = part.split(/, even with /) as [string, string];
      const count = Number(/^\d+/.exec(n)?.[0]);
      for (const name of names.split(/, | and /)) {
        const cards = j.cards.filter((c) => (name === 'consumer cards' ? c.figures.segment === 'consumer' : c.gpu === name || family(c.gpu) === name));
        expect(cards.length, name).toBeGreaterThan(0);
        for (const c of cards) expect([c.gpu, c.serverGpus ?? c.maxGpus]).toEqual([c.gpu, count]);
      }
    }
    // 6. The "Closest" layout named is the one on its card's line.
    const near = /Closest: 1 server with (\d+)× NVIDIA (.+?), writing/.exec(rec);
    if (near) {
      const card = j.cards.find((c) => c.gpu === near[2]);
      expect(card, near[2]).toBeDefined();
      expect((card as ClaimCard & { figures: { count: number; servers?: number } }).figures.count / ((card as ClaimCard & { figures: { servers?: number } }).figures.servers ?? 1)).toBe(Number(near[1]));
    }
    // 7. The binding limit is named alike in the summary and the headline (memory, first-token wait, writing speed).
    if (/^Recommendation: none — the first words come too late/.test(rec)) {
      expect(j.headline).toContain('nobody gets the first words within the wait allowed');
      expect(j.headline).not.toMatch(/tok\/s of writing; the closest/);
    }
    if (/^Recommendation: none — at [\d.]+[kM]? tokens .*, the memory left/.test(rec)) expect(j.headline).toContain('the memory left after the model holds no full-length request');
    // 8. A plan on the guaranteed figures only when the user count and the speed can carry one.
    if (conf.endsWith('Plan on the guaranteed figures.')) {
      expect(conf).not.toMatch(/user count low, not precise enough/);
      expect(conf).not.toContain('not reliable');
    }
    if (/not reliable/.test(rec)) expect(line('Alternative')).toBeUndefined();
    // 5. An alternative serves someone when the recommendation does.
    const alt = line('Alternative');
    if (alt && !/≈ [\d,]+ users per server/.test(alt)) expect(alt).not.toMatch(/\(fits; 0 users guaranteed|\(at the limit; 0 users guaranteed/);
  });
});

describe('golden text output', () => {
  it.each(CASES)('%s: %s', async (file, argv) => {
    const r = await runCli(argv, cache.dir, { columns: 120, isTTY: false, localGpu: () => Promise.resolve(undefined) });
    expect(r.code).toBe(0);
    // The pack's date changes with every refresh of data/models, not with the planner: kept out of the files.
    const text = r.text.replace(/built-in data \(\d{4}-\d{2}-\d{2}\)/, 'built-in data (YYYY-MM-DD)');
    await expect(`$ ${__CLI_NAME__} ${argv.join(' ')}\n\n${text}\n`).toMatchFileSnapshot(`./${file}.txt`);
  });
});

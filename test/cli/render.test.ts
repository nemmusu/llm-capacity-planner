import { describe, expect, it } from 'vitest';
import { cardLines, figuresOf, generalAnswer, type CardLine, type FigureContext, type Figures } from '../../src/cli/figures';
import {
  AGENTS_NOTE,
  beyondNative,
  notEnoughText,
  serverText,
  serversText,
  scopeText,
  nobodyCause,
  outsideLimits,
  layoutLegend,
  confidence,
  plannedCards,
  accuracy,
  shownBases,
  workText,
  cardColumns,
  COMPAT_UNCHECKED,
  CSV_HEADERS,
  csvModel,
  csvText,
  detailLines,
  hardwareLabel,
  headline,
  layoutColumns,
  LAYOUT_LEGEND,
  legend,
  lineNotes,
  PIPELINE_NOTE,
  waitLimitText,
  mdTable,
  memoryDetails,
  memoryText,
  modelLine,
  Notes,
  renderCsv,
  renderJson,
  renderMarkdown,
  renderText,
  scenario,
  sourceLine,
  splitText,
  summaryLines,
  table,
  USERS_LEGEND,
  usersHead,
  weightsNote,
  type Answer,
  type Column,
} from '../../src/cli/render';
import { plusMinus } from '../../src/cli/format';
import { layoutsFor } from '../../src/planner/hardware';
import { planLayout, rankRows, type PlanRow } from '../../src/planner/plan';
import { at, GPUS, opts, qwen, renderModel, workload } from './fixtures';

// Qwen3-8B BF16 at 32k tokens, fp8 cache: the planner's real rows on every catalog card.
const W = workload(32_768);
const m = renderModel({ cache: 'fp8', workload: W, contextIsModelMax: false });
const ctx: FigureContext = { model: qwen, opts: { ...opts(32_768, { kvCache: 'fp8' }), workload: W }, repo: 'Qwen/Qwen3-8B', verification: qwen.verification };
const rows = rankRows(layoutsFor(GPUS).map((l) => planLayout(qwen, l, ctx.opts)));
const fig = (r: PlanRow, users?: { concurrent: number }) => figuresOf(ctx, r, users);
const cards = cardLines(ctx, rows, GPUS, undefined, 'rtx-5090-32gb');
const general = generalAnswer(rows);
const answer: Answer = {
  cards: cards.map((c) => c.line),
  rows: rows.map((r) => fig(r)),
  fit: { figures: fig(general.fit?.row as PlanRow), alternative: fig(general.fit?.alternative as PlanRow) },
};
const plain = { color: false, ascii: false, columns: 120 };
const line = (id: string) => answer.cards.find((c) => c.gpuId === id) as CardLine;
const f5090 = line('rtx-5090-32gb').figures;
const fH100 = fig(planLayout(qwen, at('h100-sxm5-80gb'), ctx.opts));
const noSpeed = line('rtx-pro-6000-blackwell-server-96gb').figures;
const withUsers = (f: Figures, over: Partial<NonNullable<Figures['users']>>): Figures => ({ ...f, users: { ...(f.users as NonNullable<Figures['users']>), ...over } });

describe('the general answer on top', () => {
  it('the fewest GPUs it fits on, the other market, the users and the speed with their range and accuracy, the reading', () => {
    expect(headline(m, answer)).toMatch(
      /^Qwen3-8B at 32k tokens, fp8 KV cache: serves users from 1× RTX 3090, or 1× A100 40GB PCIe\. With 1× RTX 3090: 10 users guaranteed, 10 likely, at \d+ tok\/s \(\d+ to \d+, ±\d+%, estimated\) each; reading 32k tokens: [\d.]+ s \([\d.]+ to [\d.]+ s, ±\d+%\)\.$/,
    );
    expect(headline(m, answer, true)).toContain('serves users from 1x RTX 3090, or 1x A100 40GB PCIe');
  });

  it('a variant, a full cache, borderline, more GPUs, no alternative', () => {
    const two = fig(planLayout(qwen, at('h100-sxm5-80gb', 2), ctx.opts));
    expect(headline(renderModel({ variant: 'FP8', cache: 'auto', workload: W }), { ...answer, fit: { figures: { ...two, verdict: 'at-limit' } } })).toMatch(
      /^Qwen3-8B \(FP8\) at 32k tokens, full-precision KV cache \(auto\): at the limit already with 2× H100 SXM5 80GB \(1 model on 2\)\. With 2× H100 SXM5 80GB \(1 model on 2\): /,
    );
  });

  it('nobody guaranteed at the target speed; the speed unknown; a speed not precise enough', () => {
    expect(headline(m, { ...answer, fit: { figures: withUsers(fH100, { guaranteed: 0, likely: 10 }) } })).toMatch(
      /With 1× H100 SXM5 80GB, nobody is guaranteed 20 tok\/s of writing \(10 likely; one request alone writes \d+ tok\/s \(\d+ to \d+, ±\d+%\)\); reading 32k tokens: /,
    );
    expect(headline(m, { ...answer, fit: { figures: noSpeed } })).toContain("With 1× RTX PRO 6000 Server the speed cannot be estimated (the card's bandwidth or TFLOPS are not published).");
    const vague = { ...fH100, tokS: { ...(fH100.tokS as NonNullable<Figures['tokS']>), precision: { kind: 'imprecise' as const, errorPct: 61, basis: 'x' } } };
    expect(headline(m, { ...answer, fit: { figures: vague } })).toMatch(/±\d+%, not precise enough to decide\) each/);
  });

  it('nothing fits: the closest attempt and the memory it would need', () => {
    const t = fig(planLayout(qwen, at('rtx-5080-16gb', 8), ctx.opts));
    const { fit: _, ...rest } = answer;
    expect(headline(m, { ...rest, attempt: t })).toMatch(/^Qwen3-8B at 32k tokens, fp8 KV cache: does not fit in one server on any card of the catalog \(with 4× RTX 5080 \(.*\) it needs [\d.]+ GiB per GPU of [\d.]+ GiB; 8 GPUs are 2 such servers, each with its own copy\)\.$/);
  });

  it('only the engine refuses: "fits in memory … but the engine does not start", not "does not fit"', () => {
    const { fit: _, ...rest } = answer;
    const b = { ...fH100, verdict: 'no-start' as const, compat: { status: 'does-not-start' as const, notes: ['no sparse MLA kernel on sm90'] } };
    expect(headline(m, { ...rest, blocked: b })).toBe('Qwen3-8B at 32k tokens, fp8 KV cache: fits in memory from 1× H100 SXM5 80GB, but the engine does not start (no sparse MLA kernel on sm90); no layout of the catalog runs it.');
  });

  it('a context beyond the model maximum: a note on YaRN and the quality', () => {
    expect(beyondNative(m)).toBeUndefined();
    expect(beyondNative(renderModel({ contextIsModelMax: false, modelMaxContext: 262_144, workload: workload(1_048_576) }))).toBe(
      "Note: 1M tokens is beyond the model's own maximum (256k): the engine refuses it without a RoPE override (YaRN, e.g. vLLM --hf-overrides), and the quality beyond the context the lab declares is not guaranteed.",
    );
    const { modelMaxContext: _, ...noMax } = renderModel();
    expect(beyondNative(noMax)).toBeUndefined();
    const long = renderModel({ contextIsModelMax: false, modelMaxContext: 32_768, workload: workload(65_536) });
    expect(renderText(long, answer, plain)).toContain("Note: 64k tokens is beyond the model's own maximum (32k)");
    expect(renderMarkdown(long, answer)).toContain("Note: 64k tokens is beyond the model's own maximum (32k)");
  });

  it('--users: the active share and the concurrent requests; the option that serves them; none, with the simplest one or without a speed', () => {
    expect(usersHead(200, 20, 0.1)).toBe('200 users (10% active ≈ 20 concurrent requests)');
    expect(usersHead(5, 1, 0.2)).toBe('5 users (20% active ≈ 1 concurrent request)');
    const u = { total: 200, concurrent: 20, choice: { figures: fH100, alternative: f5090 } };
    expect(headline(m, { ...answer, users: u })).toMatch(/ For 200 users \(10% active ≈ 20 concurrent requests\) you need 1× H100 SXM5 80GB, or 1× RTX 5090: \d+ users guaranteed, \d+ likely, at \d+ tok\/s \(.*\) each\.$/);
    // More users than one server holds: each option with its own number of servers, the users per server.
    const many = { total: 20_000, concurrent: 2000, choice: { figures: fH100, servers: 9, alternative: f5090, alternativeServers: 70 } };
    expect(headline(m, { ...answer, users: many })).toMatch(/ you need 9 servers of 1× H100 SXM5 80GB, or 70 servers of 1× RTX 5090: \d+ users guaranteed, \d+ likely, at .* each per server\.$/);
    // The planner's users per server (likely ones when known); a layout over two servers named by one server.
    const per = { ...many, choice: { ...many.choice, usersPerServer: 40, likelyUsersPerServer: 70 } };
    expect(headline(m, { ...answer, users: per })).toMatch(/ you need 9 servers of 1× H100 SXM5 80GB, .*: 40 users guaranteed, 70 likely per server, at /);
    expect(headline(m, { ...answer, users: { ...many, choice: { ...many.choice, usersPerServer: 40 } } })).toMatch(/: 40 users guaranteed per server, at /);
    const two = { ...fH100, count: 16, replicas: 16, servers: 2 };
    expect(headline(m, { ...answer, users: { ...many, choice: { figures: two, servers: 5, usersPerServer: 40 } } })).toContain('you need 5 servers of 8× H100 SXM5 80GB (8 copies × 1): 40 users guaranteed per server');
    expect(headline(m, { ...answer, users: { total: 5, concurrent: 1 } })).toMatch(/ For 5 users \(10% active ≈ 1 concurrent request\): none — too slow at 32k tokens on any card of the catalog: .* whatever the number of servers\. Closest: /);
    // Every layout that fits held back by memory: memory is the cause, said in the summary too.
    const full = answer.rows.map((r) => (r.users ? { ...withUsers(r, { guaranteed: 0 }), requests: { ...(r.requests as NonNullable<Figures['requests']>), limitedBy: 'memory' as const } } : r));
    // Without a layout chosen (a bare answer): the first of the ranked rows with a speed.
    const slowOnly = answer.rows.map((r) => (r.users ? { ...withUsers(r, { guaranteed: 0 }), requests: { ...(r.requests as NonNullable<Figures['requests']>), limitedBy: 'speed' as const } } : r));
    expect(nobodyCause(m, { rows: slowOnly })).toMatch(/whatever the number of servers\. Closest: 1 server with /);
    expect(nobodyCause(m, { rows: full })).toBe(
      'at 32k tokens on any card of the catalog, the memory left after the model holds no full-length request with its safety margin, so no user is guaranteed, whatever the number of servers; use more GPUs per server or shorter requests',
    );
    expect(summaryLines(m, { ...answer, rows: full, users: { total: 5, concurrent: 1 } })[0]).toMatch(/^Recommendation: none — at 32k tokens on any card of the catalog, the memory left /);
  });

  it('the scenario: documents by default (with the chat hint), a chat with the prompt in the cache', () => {
    expect(scenario(W)).toBe('Scenario "documents": each request brings 16k new tokens to read and generates 500; at least 20 tok/s per user. For a chat (the conversation stays in the cache) use --prompt-cache 80.');
    expect(scenario({ ...W, cachedShare: 0.8 })).toBe('Scenario "chat": each request has 16k tokens of context, 80% already in the prefix cache (3.2k new tokens to read), and generates 500; at least 20 tok/s per user.');
  });

  it('what was read: bytes from Hugging Face, from the local cache, or built-in data; never the weights', () => {
    expect(sourceLine({ kind: 'network', bytes: 421_888, cachedBytes: 0 })).toBe('Read config + tensor headers (412 KB) from Hugging Face — weights not downloaded.');
    expect(sourceLine({ kind: 'network', bytes: 3 * 1024 * 1024, cachedBytes: 2048 })).toBe('Read config + tensor headers (3.0 MB) from Hugging Face, 2 KB from the local cache — weights not downloaded.');
    expect(sourceLine({ kind: 'cache', bytes: 10 })).toBe('Read config + tensor headers (1 KB) from the local cache — weights not downloaded.');
    expect(sourceLine({ kind: 'built-in', date: '2026-10-15' })).toBe('Model description from built-in data (2026-10-15) — weights not downloaded; --refresh reads it again from Hugging Face.');
  });

  it('the model line: repo, format, family, engine, cache, context and the model maximum', () => {
    expect(modelLine(m)).toBe('Qwen/Qwen3-8B · BF16 · denso · vLLM 0.31.0 · fp8 KV cache · context 32k tokens (model maximum 40k)');
    expect(modelLine(renderModel({ variant: 'FP8', engine: 'sglang' }))).toBe('Qwen/Qwen3-8B (FP8) · BF16 · denso · SGLang 0.5.21 · full-precision KV cache (auto) · context 40k tokens (the model maximum)');
    const { modelMaxContext: _, ...noMax } = renderModel({ contextIsModelMax: false });
    expect(modelLine(noMax)).toMatch(/context 40k tokens$/);
  });
});

describe('the summary', () => {
  const find = (l: string[], start: string) => l.find((s) => s.startsWith(start));
  const DOCS = 'of up to 32k tokens (≈ 25,000 words ≈ 49 pages; 16k on average)';
  it('servers, the work in words and pages: the maximum, and the average the capacity is computed at', () => {
    expect(serverText(fH100)).toBe('1 server with 1× NVIDIA H100 SXM5 80GB');
    // 16 GPUs are two servers of 8: independent copies, never one model across servers.
    expect(serverText({ count: 16, gpu: 'H100 SXM5 80GB' }, true)).toBe('2 servers with 8x NVIDIA H100 SXM5 80GB each');
    expect(serversText(4, { count: 16, gpu: 'H100 SXM5 80GB' })).toBe('4 servers with 8× NVIDIA H100 SXM5 80GB each');
    expect(workText(W)).toBe(`with requests ${DOCS}`);
    expect(workText({ ...W, maxContext: 131_072, typicalContext: 131_072, cachedShare: 0.8 })).toBe('chatting, with conversations of up to 128k tokens (≈ 98,000 words ≈ 200 pages)');
  });

  it('confidence: the level of the largest error; at ±50% or more, or imprecise, not precise enough to decide', () => {
    expect(confidence([2.9])).toBe('high (±2.9%)');
    expect(confidence([10, 48])).toBe('medium (±10–48%)');
    // At ±50% or more no span is quoted (no ±452%): the rule says why.
    expect(confidence([10, 63])).toBe('low, not precise enough to decide (more than ±50%)');
    expect(confidence([18, 452])).toBe('low, not precise enough to decide (more than ±50%)');
    // Under ±50% the label "not precise enough" always says why.
    expect(confidence([10, 20], 'the speed model does not cover INT4 MoE')).toBe('low, not precise enough to decide (±10–20%, but the speed model does not cover INT4 MoE)');
  });

  it('the simplest option: recommendation, alternative, what is not enough, speed in words with a range, confidence', () => {
    const s = summaryLines(m, answer);
    expect(s.length).toBeGreaterThanOrEqual(4);
    expect(s.length).toBeLessThanOrEqual(6);
    expect(s[0]).toBe(`Recommendation: 1 server with 1× NVIDIA RTX 3090 serves 10 users guaranteed, 10 likely, with requests ${DOCS}. Caution: support for this setup by the engine is not verified (see the notes under the per-card table).`);
    expect(find(s, 'Alternative:')).toMatch(/^Alternative: 1 server with 1× NVIDIA A100 40GB PCIe \(fits; \d+ users guaranteed, \d+ likely\)\./);
    expect(find(s, 'Speed for each user:')).toMatch(
      /^Speed for each user: writing about \d+ words per second \(\d+ to \d+; \d+ tokens\/s\); with everyone busy the first words appear after about [\d.]+ s, and a full 32k-token request alone is read in about [\d.]+ s \(1 token ≈ 0\.75 words\)\.$/,
    );
    expect(find(s, 'Confidence:')).toMatch(/^Confidence: memory high \(±[\d.]+%\); writing speed (high|medium) \(±\d+%\); reading time (high|medium) \(±\d+%\); user count (high|medium) \(±\d+%\)\. Plan on the guaranteed figures\.$/);
    expect(find(s, 'Not enough')).toBeUndefined();
    expect(find(s, 'Warning:')).toBeUndefined();
    expect(summaryLines(m, answer, true)[0]).toContain('1x NVIDIA RTX 3090');
  });

  it('--users: the option that serves them and its alternative; how many servers, explained; near saturation; none at all', () => {
    const calm = { ...fH100, requests: { ...(fH100.requests as NonNullable<Figures['requests']>), guaranteed: 100 } };
    const u = summaryLines(m, { ...answer, users: { total: 200, concurrent: 20, choice: { figures: calm, alternative: { ...f5090, verdict: 'at-limit' } } } });
    expect(u[0]).toBe(`Recommendation: 1 server with 1× NVIDIA H100 SXM5 80GB serves 200 users with requests ${DOCS}.`);
    expect(find(u, 'Alternative:')).toMatch(/^Alternative: 1 server with 1× NVIDIA RTX 5090 \(at the limit; /);
    const many = summaryLines(m, {
      ...answer,
      users: { total: 100_000, concurrent: 10_000, choice: { figures: calm, servers: 400, usersPerServer: 251, alternative: f5090, alternativeServers: 4000, alternativeUsersPerServer: 25 } },
    });
    // The planner's users per server (serverPlan): servers × users per server covers everyone.
    expect(many[0]).toBe(
      `Recommendation: 400 servers with 1× NVIDIA H100 SXM5 80GB each serve 100,000 users with requests ${DOCS}: ≈ 251 users per server. Each server runs its own copy of the model, and a load balancer (a router in front of the servers) spreads the users across them.`,
    );
    expect(find(many, 'Alternative:')).toBe('Alternative: 4,000 servers with 1× NVIDIA RTX 5090 each (≈ 25 users per server).');
    // Without the planner's figure: total ÷ servers; an alternative of several servers without users asked: its own users.
    expect(summaryLines(m, { ...answer, users: { total: 100_000, concurrent: 10_000, choice: { figures: calm, servers: 400 } } })[0]).toContain(': ≈ 250 users per server.');
    const sixteen = { ...fH100, count: 16, servers: 2 };
    expect(find(summaryLines(m, { ...answer, fit: { figures: f5090, alternative: sixteen } }), 'Alternative:')).toMatch(/^Alternative: 2 servers with 8× NVIDIA H100 SXM5 80GB each \(\d+ users guaranteed, \d+ likely\)\./);
    expect(summaryLines(m, { ...answer, fit: { figures: sixteen } })[0]).toMatch(/^Recommendation: 2 servers with 8× NVIDIA H100 SXM5 80GB each serve \d+ users guaranteed/);
    expect(summaryLines(m, { ...answer, users: { total: 100_000, concurrent: 10_000, choice: { figures: calm, servers: 400 } } }, true)[0]).toContain('400 servers with 1x NVIDIA H100');
    // 20 requests on a server that guarantees 22: near its limit, said in the recommendation.
    const busy = { ...fH100, requests: { ...(fH100.requests as NonNullable<Figures['requests']>), guaranteed: 22 } };
    expect(summaryLines(m, { ...answer, users: { total: 200, concurrent: 20, choice: { figures: busy } } })[0]).toMatch(/ Caution: each server runs at 91% of its guaranteed capacity, so waits grow fast with more traffic: one more server leaves headroom\.$/);
    // Prompts keeping the GPUs busy, or an engine that supports the setup in part.
    const reading = { ...calm, saturation: 'near saturation: reading prompts takes 90% of the GPU time' };
    expect(summaryLines(m, { ...answer, users: { total: 200, concurrent: 20, choice: { figures: reading } } })[0]).toContain('Caution: reading the prompts keeps the GPUs almost always busy, so the writing speed is less certain.');
    // The engine's caution: partial support, or support not verified; a link note alone is no caution.
    const withNotes = (notes: string[]) => summaryLines(m, { ...answer, users: { total: 200, concurrent: 20, choice: { figures: { ...calm, compat: { status: 'warnings' as const, notes } } } } })[0];
    expect(withNotes(['partly supported: no FP8 kernel'])).toContain('Caution: the engine supports this setup only in part (see the notes under the per-card table).');
    expect(withNotes(['engine support on this card not verified: x'])).toContain('Caution: support for this setup by the engine is not verified (see the notes under the per-card table).');
    expect(withNotes(['TP 4 over PCIe (no NVLink): all-reduces slower than over NVLink'])).not.toContain('Caution');
    // Nobody served anywhere: the real cause and the closest layout, with its figures.
    const slowRows = answer.rows.map((r) => (r.users ? { ...withUsers(r, { guaranteed: 0 }), requests: { ...(r.requests as NonNullable<Figures['requests']>), limitedBy: 'speed' as const } } : r));
    const slowCards = answer.cards.map((c) => ({ ...c, figures: c.figures.users ? withUsers(c.figures, { guaranteed: 0 }) : c.figures }));
    const nobody = summaryLines(m, { ...answer, rows: slowRows, cards: slowCards, users: { total: 5, concurrent: 1 } });
    // Writing (the answer) and reading (the prompt) apart: which speed fails, then each with its own figure.
    expect(nobody[0]).toMatch(/^Recommendation: none — too slow at 32k tokens on any card of the catalog: no user is guaranteed a writing speed of 15 words per second \(20 tokens\/s\) for the answer \(the guarantee is the low end of the estimate\), whatever the number of servers\. Closest: 1 server with \d+× NVIDIA .*, writing about \d+ words per second per user under load \(\d+ to \d+; \d+ tokens\/s\).*, one request alone \d+ words per second \(\d+ tokens\/s\); (reading .*)?a full 32k-token request alone is read in about [\d.]+ s/);
    // The closest is the layout the answer chose (figures.ts closestRow), the one the headline names (I-1).
    const chosen = { ...withUsers(f5090, { guaranteed: 0 }), requests: { ...(f5090.requests as NonNullable<Figures['requests']>), limitedBy: 'speed' as const } };
    const named = summaryLines(m, { ...answer, rows: slowRows, cards: slowCards, fit: { figures: chosen }, users: { total: 5, concurrent: 1 } });
    expect(named[0]).toContain('Closest: 1 server with 1× NVIDIA RTX 5090,');
    expect(headline(m, { ...answer, rows: slowRows, cards: slowCards, fit: { figures: chosen } })).toContain('the closest is 1× RTX 5090');
    // No layout named: no speed of a layout the reader cannot see.
    expect(find(nobody, 'Speed for each user:')).toBeUndefined();
  });

  it('a speed unknown, nobody guaranteed (by speed or by memory), not precise enough; beyond the model maximum', () => {
    const unknown = summaryLines(m, { ...answer, fit: { figures: noSpeed } });
    expect(unknown[0]).toBe(`Recommendation: 1 server with 1× NVIDIA RTX PRO 6000 Server holds the model with requests ${DOCS}; its speed cannot be estimated (the card's figures are not published).`);
    expect(find(unknown, 'Alternative:')).toBeUndefined();
    expect(find(unknown, 'Speed for each user:')).toBeUndefined();
    expect(find(unknown, 'Confidence:')).toMatch(/speed not estimated\. Nothing is guaranteed here: do not rely on these figures alone\.$/);
    expect(summaryLines(m, { ...answer, fit: { figures: { ...noSpeed }, alternative: noSpeed } }).find((s) => s.startsWith('Alternative:'))).toContain('(fits; speed unknown)');
    // A layout that guarantees nobody is no recommendation: none, with the cause.
    const slowF = withUsers(fH100, { guaranteed: 0, likely: 3 });
    const asSlow = (f: Figures, by: 'speed' | 'wait' | 'memory', reading = 0.9) => ({ ...f, readingShare: reading, requests: { ...(f.requests as NonNullable<Figures['requests']>), limitedBy: by } });
    const tr = (by: 'speed' | 'wait' | 'memory', reading?: number) => (r: Figures) => (r.users ? asSlow(withUsers(r, { guaranteed: 0 }), by, reading) : r);
    const rowsBy = (by: 'speed' | 'wait' | 'memory', reading?: number) => answer.rows.map(tr(by, reading));
    // The rows and the per-card lines alike (the summary names the closest line).
    const slowAll = (by: 'speed' | 'wait' | 'memory', reading?: number) => ({ rows: rowsBy(by, reading), cards: answer.cards.map((c) => ({ ...c, figures: tr(by, reading)(c.figures) })) });
    const slow = summaryLines(m, { ...answer, ...slowAll('speed'), fit: { figures: asSlow(slowF, 'speed'), alternative: f5090 } });
    expect(slow[0]).toMatch(/^Recommendation: none — too slow at 32k tokens on any card of the catalog: no user is guaranteed a writing speed of 15 words per second \(20 tokens\/s\) .*; reading other users' 16k-token prompts takes 90% of the GPU time, and a full 32k-token request alone is read in about [\d.]+ s\. Shorter requests, or a chat that keeps its history in the prompt cache \(--prompt-cache 80\), serve more\.$/);
    // No recommendation, no alternative.
    expect(find(slow, 'Alternative:')).toBeUndefined();
    expect(summaryLines(m, { ...answer, ...slowAll('speed', 0.2), fit: { figures: asSlow(slowF, 'speed', 0.2) } })[0]).toMatch(/; a full 32k-token request alone is read in about [\d.]+ s\.$/);
    expect(summaryLines(m, { ...answer, ...slowAll('wait'), fit: { figures: asSlow(slowF, 'wait') } })[0]).toMatch(/: even one request alone may wait up to [\d.]+ s for its first words \(about [\d.]+ s\), more than the 10 s allowed \(--max-wait\): reading its 16k-token prompt takes that long, whatever the number of servers\./);
    // Figures packed before the reading share was kept: the speed under load, without the share.
    const strip = (r: Figures) => {
      const { readingShare: _, ...x } = r;
      return x;
    };
    const all = slowAll('speed');
    const old = { rows: all.rows.map(strip), cards: all.cards.map((c) => ({ ...c, figures: strip(c.figures) })), fit: { figures: strip(asSlow(slowF, 'speed')) } };
    expect(summaryLines(m, { ...answer, ...old, users: { total: 5, concurrent: 1 } })[0]).toMatch(/; a full 32k-token request alone is read in about [\d.]+ s\.$/);
    // No layout with a speed: said so.
    expect(nobodyCause(m, { rows: [noSpeed] })).toBe("at 32k tokens on any card of the catalog the speed cannot be estimated (the card's figures are not published), so no user count can be given");
    expect(find(slow, 'Speed for each user:')).toBeUndefined();
    expect(find(slow, 'Confidence:')).toMatch(/Nothing is guaranteed here: do not rely on these figures alone\.$/);
    const full = summaryLines(m, { ...answer, ...slowAll('memory'), fit: { figures: asSlow(slowF, 'memory') } });
    expect(full[0]).toContain('Recommendation: none — at 32k tokens on any card of the catalog, the memory left after the model holds no full-length request with its safety margin');
    const vague = { ...fH100, tokS: { ...(fH100.tokS as NonNullable<Figures['tokS']>), precision: { kind: 'imprecise' as const, errorPct: 30, basis: 'x', reason: 'INT4 MoE; extrapolated' } } };
    // One level per figure, each from its own error: the writing speed vague, the reading not.
    const vc = find(summaryLines(m, { ...answer, fit: { figures: vague } }), 'Confidence:');
    expect(vc).toMatch(/writing speed low, not precise enough to decide \(±30%, but INT4 MoE\); reading time (high|medium) \(±\d+%\); user count /);
    const { reason: _r, ...bare } = vague.tokS.precision;
    expect(find(summaryLines(m, { ...answer, fit: { figures: { ...vague, tokS: { ...vague.tokS, precision: bare } } } }), 'Confidence:')).toContain('but the speed model does not cover this case)');
    const at = (f: Figures, k: 'tokS' | 'read', e: number) => ({ ...f, [k]: { ...(f[k] as NonNullable<Figures['tokS']>), precision: { ...(f[k] as NonNullable<Figures['tokS']>).precision, errorPct: e } } });
    const sure = { ...at(at(fH100, 'tokS', 10), 'read', 10), users: { ...(fH100.users as NonNullable<Figures['users']>), precision: { ...(fH100.users as NonNullable<Figures['users']>).precision, errorPct: 10 } } };
    expect(find(summaryLines(m, { ...answer, fit: { figures: sure } }), 'Confidence:')).toContain('writing speed high (±10%); reading time high (±10%); user count high (±10%)');
    // The users' error is its own: low for them only.
    const users = { ...sure, users: { ...sure.users, precision: { ...sure.users.precision, errorPct: 55 } } };
    expect(find(summaryLines(m, { ...answer, fit: { figures: users } }), 'Confidence:')).toContain('writing speed high (±10%); reading time high (±10%); user count low, not precise enough to decide (more than ±50%)');
    // 49.7% is no "±50%": a decimal under the rule.
    expect(plusMinus(49.7)).toBe('±49.7%');
    expect(plusMinus(49.4)).toBe('±49%');
    expect(plusMinus(50.2)).toBe('±50%');
    // Beyond the model's own maximum, documented by the lab (official YaRN): a warning in the summary.
    const yarn = summaryLines({ ...m, workload: workload(65_536), modelMaxContext: 40_960, documentedMaxContext: 64_000 }, answer);
    expect(yarn[1]).toBe("Warning: 64k tokens is more than the model's own maximum (40k): the engine needs a context extension (YaRN) to start, which the lab documents up to 64,000 tokens.");
    // Beyond what the lab documents: outside the model's limits, nothing recommended, the reason in the confidence.
    const outside = summaryLines({ ...m, workload: workload(131_072), modelMaxContext: 40_960, documentedMaxContext: 64_000 }, answer);
    expect(outside[0]).toBe("Recommendation: none — 128k tokens is outside the model's limits: it is made for up to 63k tokens, and beyond that the quality of its answers is not known; plan for at most 63k tokens, or choose a model made for longer requests.");
    expect(outside.map((l) => l.split(':')[0])).toEqual(['Recommendation', 'Confidence']);
    expect(find(outside, 'Confidence:')).toContain("but 128k tokens is beyond the model's maximum of 63k)");
    expect(outsideLimits({ ...m, workload: workload(65_536), modelMaxContext: 40_960 })).toBe(40_960);
    expect(outsideLimits(m)).toBeUndefined();
  });

  it('nothing runs: the engine refuses, or nothing fits; every card checked is not enough; "1 GPU"', () => {
    const { fit: _, ...rest } = answer;
    const refusedCards = answer.cards.map((c) => ({ ...c, status: 'no-start' as const }));
    const b = summaryLines(m, { ...rest, cards: refusedCards, blocked: fH100 });
    expect(b[0]).toBe('Recommendation: none — the model fits in memory on 1 server with 1× NVIDIA H100 SXM5 80GB, but the inference engine does not start there.');
    // The GPUs tried in one server, per card: 8 data-center GPUs, 4 workstation or GeForce cards, 2 GH200.
    expect(b[1]).toMatch(/^Not enough: every card checked: .*H100.*, even with 8 GPUs in one server; .*consumer cards, even with 4; GH200, even with 2\.$/);
    expect(b.at(-1)).toMatch(/^Confidence: memory high .*; speed not estimated\./);
    expect(summaryLines(m, { ...rest, cards: refusedCards.map((c) => ({ ...c, maxGpus: 1, serverGpus: 1 })), blocked: fH100 })[1]).toMatch(/^Not enough: every card checked: .*, even with 1 GPU in one server\.$/);
    // Cards packed without serverGpus (older reports): the most GPUs tried.
    expect(notEnoughText(refusedCards.map(({ serverGpus: _, ...c }) => c))).toMatch(/^every card checked: .*, even with 8 GPUs in one server$/);
    const t = fig(planLayout(qwen, at('rtx-5080-16gb', 4), ctx.opts));
    const n = summaryLines(m, { ...rest, cards: refusedCards, attempt: t, users: { total: 9, concurrent: 1 } });
    expect(n[0]).toMatch(/^Recommendation: none — the model does not fit in one server on any card of the catalog \(with 1 server with 4× NVIDIA RTX 5080 it needs [\d.]+ GiB per GPU; the card has 15\.4 GiB\)\.$/);
    expect(n[1]).toMatch(/^Not enough: every card checked: /);
    // Only the cards asked for, all failing: the recommendation says it, with the scope.
    const sel = summaryLines({ ...m, selection: { cards: ['RTX 5080'], count: 4 } }, { ...rest, cards: refusedCards, attempt: t });
    expect(sel[0]).toContain('does not fit in one server on the selected card (RTX 5080, 4 GPUs)');
    expect(find(sel, 'Not enough')).toBeUndefined();
    expect(scopeText({ ...m, selection: { cards: ['H100 SXM5 80GB', 'H100 NVL 94GB'] } })).toBe('on the selected cards (H100 SXM5 80GB, H100 NVL 94GB)');
    expect(scopeText({ ...m, selection: { cards: [], count: 1 } })).toBe('on any card of the catalog with 1 GPU');
  });

  it("this PC's card (--this-pc) stays out of the summary unless it is one of the cards asked for", () => {
    const only = answer.rows.filter((r) => r.gpuId === 'h100-sxm5-80gb');
    const local = { ...line('rtx-5090-32gb'), local: true, status: 'no-fit' as const };
    const asked = { ...answer, rows: only, cards: [line('h100-sxm5-80gb'), local] };
    expect(plannedCards(asked).map((c) => c.gpuId)).toEqual(['h100-sxm5-80gb']);
    expect(plannedCards({ ...asked, rows: answer.rows }).map((c) => c.gpuId)).toEqual(['h100-sxm5-80gb', 'rtx-5090-32gb']);
    expect(summaryLines(m, { ...asked, fit: { figures: fH100 } }).join(' ')).not.toContain('consumer');
  });

  it('what is not enough: card families, consumer cards together, this PC left out', () => {
    const not = (ids: string[], status: CardLine['status'] = 'no-fit') => notEnoughText(answer.cards.map((c) => (ids.includes(c.gpuId) ? { ...c, status } : c)));
    expect(not([])).toBeUndefined();
    expect(not(['h100-pcie-80gb', 'h100-sxm5-80gb'])).toBeUndefined();
    expect(not(['h100-pcie-80gb', 'h100-sxm5-80gb', 'h100-nvl-94gb'])).toBe('H100, even with 8 GPUs in one server');
    const consumer = answer.cards.filter((c) => c.figures.segment === 'consumer').map((c) => c.gpuId);
    expect(not([...consumer, 'l4-24gb'], 'slow')).toBe('L4, even with 8 GPUs in one server; consumer cards, even with 4');
    expect(not(['rtx-3090-24gb'], 'short')).toBe('RTX 3090, even with 4 GPUs in one server');
    // At the limit with nobody guaranteed is not enough either (one rule with "too slow").
    const atLimit = answer.cards.map((c) => (c.gpuId === 'b200-sxm-180gb' ? { ...c, status: 'at-limit' as const, figures: withUsers(c.figures, { guaranteed: 0 }) } : c));
    expect(notEnoughText(atLimit)).toBe('B200, even with 8 GPUs in one server');
    expect(notEnoughText(answer.cards.map((c) => (c.gpuId === 'b200-sxm-180gb' ? { ...c, status: 'at-limit' as const } : c)))).toBeUndefined();
    // Too slow with the fewest GPUs but serving with more: enough.
    expect(notEnoughText(answer.cards.map((c) => (c.gpuId === 'l4-24gb' ? { ...c, status: 'slow' as const, servingGpus: 2 } : c)))).toBeUndefined();
    const local = { ...(answer.cards[0] as CardLine), gpuId: 'this-pc', gpu: 'Tesla T4', status: 'no-fit' as const };
    expect(notEnoughText([...answer.cards, local, { ...local, gpuId: 'custom' }])).toBeUndefined();
    const s = summaryLines(m, { ...answer, cards: answer.cards.map((c) => (c.figures.segment === 'consumer' ? { ...c, status: 'no-fit' as const } : c)) });
    // The card recommended (RTX 3090) is never "not enough": its family is no longer all failing, the others are named.
    expect(s[0]).toContain('NVIDIA RTX 3090');
    expect(s.find((l) => l.startsWith('Not enough'))).toBe('Not enough: RTX 4090, RTX 5080 and RTX 5090, even with 4 GPUs in one server.');
  });
});

describe('tables', () => {
  const items = [{ a: 'uno', b: 'lungo lungo lungo', c: 'tre' }];
  const cols: Column<(typeof items)[number]>[] = [
    { head: 'A', long: (x) => x.a, drop: 0, color: () => 'green' },
    { head: 'Bi\nsotto', long: (x) => x.b, short: (x) => x.b.slice(0, 5), drop: 1, shorten: 2 },
    { head: 'C', long: (x) => x.c, short: () => 'c', drop: 2, right: true, shorten: 1 },
  ];
  it('long forms when wide; two-line headers; colours after padding', () => {
    expect(table(items, cols, 80)).toEqual(['A    Bi                   C', '     sotto', 'uno  lungo lungo lungo  tre']);
    expect(table(items, cols, 80, { color: true })[2]).toBe('\u001b[32muno\u001b[0m  lungo lungo lungo  tre');
  });

  it('narrow: short forms by their order, then columns dropped by priority; never below the essentials', () => {
    expect(table(items, cols, 25)).toEqual(['A    Bi                 C', '     sotto', 'uno  lungo lungo lungo  c']);
    expect(table(items, cols, 15)).toEqual(['A    Bi     C', '     sotto', 'uno  lungo  c']);
    expect(table(items, cols, 10)).toEqual(['A    Bi', '     sotto', 'uno  lungo']);
    expect(table(items, cols, 2)).toEqual(['A', 'uno']);
    // A short form without an order comes last.
    const more: Column<(typeof items)[number]>[] = [...cols, { head: 'D', long: () => 'dddd', short: () => 'd', drop: 0 }];
    expect(table(items, more, 18)).toEqual(['A    Bi     C  D', '     sotto', 'uno  lungo  c  d']);
  });

  it('the tiny forms come after every short one, before dropping columns', () => {
    const more: Column<(typeof items)[number]>[] = [...cols, { head: 'D', long: () => 'dddd dddd', short: () => 'dddd', tiny: () => 'd', drop: 0 }];
    expect(table(items, more, 17)).toEqual(['A    Bi     C  D', '     sotto', 'uno  lungo  c  d']);
    expect(table(items, more, 21)).toEqual(['A    Bi     C  D', '     sotto', 'uno  lungo  c  dddd']);
  });

  it('a 100-column terminal with colours and many notes: no line is wider than 100 visible characters', () => {
    const noisy = answer.cards.map((c, i) => ({ ...c, status: 'slow' as const, note: `note ${i}` }));
    const t = renderText({ ...m, source: { kind: 'cache', bytes: 1 } }, { ...answer, cards: noisy, users: { total: 200, concurrent: 20 } }, { color: true, ascii: false, columns: 100 });
    const esc = new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*m`, 'g');
    const visible = t.split('\n').map((l) => l.replace(esc, ''));
    expect(Math.max(...visible.map((l) => l.length))).toBeLessThanOrEqual(100);
    expect(t).toMatch(/ of 31\.3 {2}/);
  });

  it('ASCII before measuring (± takes two characters)', () => {
    expect(table([{ x: '±', y: '1' }], [{ head: 'X', long: (i) => i.x, drop: 0 }, { head: 'Y', long: (i) => i.y, drop: 0 }], 80, { ascii: true })).toEqual(['X   Y', '+-  1']);
  });

  it('Markdown tables: one-line headers, right-aligned numbers, escaped bars', () => {
    expect(mdTable([{ a: 'x|y', b: '1', c: '3' }], cols)).toEqual(['| A | Bi sotto | C |', '| --- | --- | ---: |', '| x\\|y | 1 | 3 |']);
  });

  it('the split of a layout', () => {
    expect([
      splitText({ count: 1, tp: 1, pp: 1, replicas: 1 }),
      splitText({ count: 2, tp: 2, pp: 1, replicas: 1 }),
      splitText({ count: 8, tp: 2, pp: 1, replicas: 4 }),
      splitText({ count: 4, tp: 1, pp: 4, replicas: 1 }),
      splitText({ count: 4, tp: 1, pp: 1, replicas: 4 }),
    ]).toEqual(['1', '2 (1 model on 2)', '8 (4 copies × 2)', '4 (split in 4 stages)', '4 (4 copies × 1)']);
    expect(splitText({ count: 8, tp: 2, pp: 1, replicas: 4 }, true)).toBe('8 (4×2)');
    expect(splitText({ count: 1, tp: 1, pp: 1, replicas: 1 }, true)).toBe('1');
  });

  it('footnotes: one number per distinct text', () => {
    const n = new Notes();
    expect(n.mark(['a', 'b', 'a', undefined])).toBe(' (1,2)');
    expect(n.mark(['b', 'c'])).toBe(' (2,3)');
    expect(n.mark([])).toBe('');
    expect(n.texts).toEqual(['a', 'b', 'c']);
  });
});

describe('the overview per card', () => {
  const cols = cardColumns(W, new Notes(), false);
  const cell = (head: string, l: CardLine, form: 'long' | 'short' = 'long') => {
    const c = cols.find((x) => x.head.startsWith(head)) as Column<CardLine>;
    return (form === 'short' && c.short ? c.short : c.long)(l);
  };
  it('a card that fits: verdict, GPUs, memory "71.4 GiB (69.8 to 73.1 GiB, ±2.3%) of 79.6 GiB", margin, users guaranteed and likely, speed, reading, accuracy', () => {
    const l = line('rtx-5090-32gb');
    expect(cell('Card', l)).toBe('RTX 5090 (this PC)');
    expect(cell('Verdict', l)).toBe('fits');
    expect(cell('GPUs', l)).toBe('1');
    expect(cell('Memory', l)).toMatch(/^[\d.]+ GiB \([\d.]+ to [\d.]+ GiB, ±[\d.]+%\) of 31\.3 GiB$/);
    expect(cell('Memory', l, 'short')).toMatch(/^[\d.]+ \([\d.]+–[\d.]+\) of 31\.3$/);
    // The margin is the worst case: the fixed costs at the high end.
    expect(cell('Margin', l)).toBe(l.figures.margin.min.toFixed(1));
    expect(cell('Margin', l, 'short')).toMatch(/^[\d.]+$/);
    expect(cell('Users\n(guaranteed)', l)).toBe(String(l.figures.users?.guaranteed));
    expect(cell('Users\n(likely)', l)).toBe(String(l.figures.users?.likely));
    expect(cell('Tok/s', l)).toMatch(/^\d+ tok\/s \(\d+ to \d+, ±\d+%\)$/);
    expect(cell('Tok/s', l, 'short')).toMatch(/^\d+ \(\d+–\d+\)$/);
    expect(cell('Read', l)).toMatch(/^[\d.]+ s \([\d.]+ to [\d.]+ s, ±\d+%\)$/);
    expect(cell('Read', l, 'short')).toMatch(/^[\d.]+ \([\d.]+–[\d.]+\)$/);
    expect(cell('Accuracy', l)).toMatch(/^measured \(±\d+%\)$/);
    expect(cell('Accuracy', l, 'short')).toMatch(/^measured ±\d+%$/);
  });

  it('a card that does not fit: "does not fit even with 8" and the memory it would need with 8', () => {
    const t = fig(planLayout(qwen, at('rtx-5080-16gb', 8), { ...ctx.opts, workload: workload(4_000_000) }));
    const l: CardLine = { gpuId: 'x', gpu: 'RTX 5080', status: 'no-fit', maxGpus: 8, figures: t };
    expect(cell('Verdict', l)).toBe('does not fit even with 8');
    expect(cell('Verdict', l, 'short')).toBe('does not fit');
    expect(cell('GPUs', l)).toBe('-');
    expect(cell('GPUs', l, 'short')).toBe('-');
    expect(cell('Memory', l)).toMatch(/^with 8: .* of [\d.]+ GiB$/);
    expect(cell('Memory', l, 'short')).toMatch(/^with \d: .* of 15\.4$/);
    const memory = cols.find((x) => x.head.startsWith('Memory')) as Column<CardLine>;
    expect(memory.tiny?.(l)).toMatch(/^with \d: [\d.]+ of 15\.4$/);
    expect(memory.tiny?.(line('rtx-5090-32gb'))).toMatch(/^[\d.]+ of 31\.3$/);
    for (const h of ['Margin', 'Users\n(guaranteed)', 'Users\n(likely)', 'Tok/s', 'Read', 'Accuracy']) expect(cell(h, l)).toBe('');
    expect(cell('Margin', l, 'short')).toBe('');
    expect(lineNotes(l)).toEqual([]);
  });

  it('no speed, the engine refusing, slow, at the limit, not serving the users', () => {
    const server = line('rtx-pro-6000-blackwell-server-96gb');
    expect(cell('Users\n(guaranteed)', server)).toBe('n/a');
    expect(cell('Tok/s', server, 'short')).toBe('');
    expect(cell('Read', server, 'short')).toBe('');
    expect(cell('Accuracy', server, 'short')).toBe('');
    const refused: CardLine = { ...line('h100-sxm5-80gb'), status: 'no-start', note: 'fits in memory but the engine does not start: x' };
    const notes = new Notes();
    const c2 = cardColumns(W, notes, true, { total: 200 });
    const verdict = c2[1] as Column<CardLine>;
    expect(verdict.long(refused)).toBe('engine does not start (1)');
    expect(verdict.long(refused)).toBe('engine does not start (1)');
    expect(cell('Users\n(likely)', refused)).toBe('n/a');
    expect(verdict.long({ ...refused, status: 'slow', note: 'too slow' })).toBe('fits, too slow (2)');
    expect(verdict.long({ ...refused, status: 'at-limit', note: 'at the limit: x' })).toBe('at the limit (3)');
    expect(verdict.long({ ...refused, status: 'short', note: 'not enough' })).toBe('too slow for anyone');
    // Narrow tables: shorter verdicts.
    expect(verdict.short?.(refused)).toBe("won't start (1)");
    expect(verdict.short?.({ ...refused, status: 'short', note: 'not enough' })).toBe('too slow');
    expect(verdict.short?.({ ...refused, status: 'slow', note: 'too slow' })).toBe('too slow (2)');
    const { note: _, ...bare } = refused;
    expect((cols[1] as Column<CardLine>).long({ ...bare, status: 'short' })).toBe('too slow for anyone');
    expect(verdict.color?.(refused)).toBe('red');
  });

  it('the notes of a line: why, the engine caveats, speed caveats, saturation, the roofline', () => {
    const l = line('rtx-3090-24gb');
    expect(lineNotes(l)).toEqual(l.figures.compat.notes);
    const full: CardLine = { ...l, note: 'why', figures: { ...l.figures, speedNote: 'estimated', saturation: 'saturated', roofline: { ...(l.figures.roofline as NonNullable<Figures['roofline']>), warning: 'above' } } };
    expect(lineNotes(full)).toEqual(['why', ...l.figures.compat.notes, 'estimated', 'saturated', 'above']);
    expect(lineNotes({ ...full, status: 'short' })).not.toContain('why');
  });

  it('memory in plain words', () => {
    expect(memoryText(f5090)).toMatch(/^[\d.]+ GiB \([\d.]+ to [\d.]+ GiB, ±[\d.]+%\) of 31\.3 GiB$/);
  });
});

describe('every layout (--gpu, --all-hw)', () => {
  const cols = layoutColumns(W, true);
  const c = (head: string) => cols.find((x) => x.head.startsWith(head)) as Column<Figures>;
  it('verdict, memory, margin, users, speed, reading, accuracy and whether the engine starts', () => {
    expect(c('Layout').long(f5090)).toBe('RTX 5090');
    expect(c('Verdict').long(f5090)).toBe('fits');
    expect(c('Verdict').long(withUsers(f5090, { guaranteed: 0 }))).toBe('fits, too slow');
    expect(c('Verdict').long({ ...f5090, verdict: 'no-start' })).toBe('engine does not start');
    expect(c('Verdict').long(noSpeed)).toBe('fits');
    expect(c('Memory').short?.(f5090)).toMatch(/^[\d.]+ \([\d.]+-[\d.]+\) of 31\.3$/);
    expect(c('Margin').long({ ...f5090, margin: { ...f5090.margin, min: 1, max: 1 } })).toBe('1.0');
    expect(c('Margin').long(f5090)).toBe(f5090.margin.min.toFixed(1));
    // Users and speed: the wider error of the two; nobody guaranteed: n/a.
    const u = f5090.users as NonNullable<Figures['users']>;
    expect(accuracy({ ...f5090, users: { ...u, precision: { ...u.precision, kind: 'imprecise', errorPct: 452 } } }, (p) => p.kind)).toBe('imprecise');
    expect(accuracy({ ...f5090, users: { ...u, precision: { ...u.precision, kind: 'exact', errorPct: 0 } } }, (p) => p.kind)).toBe(f5090.tokS?.precision.kind);
    expect(accuracy(withUsers(f5090, { guaranteed: 0 }), (p) => p.kind)).toBe('n/a');
    expect(accuracy(noSpeed, (p) => p.kind)).toBe('');
    expect(c('Users\n(guaranteed)').long(noSpeed)).toBe('');
    expect(c('Users\n(likely)').long(noSpeed)).toBe('');
    expect(c('Users\n(likely)').long(f5090)).toBe(String(f5090.users?.likely));
    for (const h of ['Tok/s', 'Read', 'Accuracy']) {
      expect(c(h).long(noSpeed)).toBe('');
      expect(c(h).short?.(noSpeed)).toBe('');
      expect(c(h).long(f5090)).not.toBe('');
      expect(c(h).short?.(f5090)).not.toBe('');
    }
    expect(c('Engine').long(f5090)).toBe('starts');
    expect(c('Engine').long(line('rtx-3090-24gb').figures)).toBe('starts, with warnings');
    expect(c('Engine').long({ ...f5090, compat: { status: 'does-not-start', notes: ['32 heads'] } })).toBe('does not start: 32 heads');
  });
});

describe('how to read the accuracy', () => {
  it('the legend: memory, users, the labels and the calibration basis; the agents with --users', () => {
    const l = legend(m);
    expect(l).toHaveLength(3);
    expect(l[0]).toContain('Memory per GPU with one request of 32k tokens (≈ 25,000 words ≈ 49 pages)');
    expect(l[1]).toContain('Users: total = concurrent requests ÷ 10% active at once');
    expect(l[1]).toContain('≥20 tok/s and the first token within 10 s (or, if longer, twice');
    expect(l[2]).toContain('not precise enough to decide = ±50% or more');
    expect(legend(m, true)).toContain(AGENTS_NOTE);
    expect(AGENTS_NOTE).toContain('5 subagents count as 5 users');
    const s = legend(renderModel({ engine: 'sglang', workload: { ...W, ttftScales: false } }));
    expect(s[1]).not.toContain('twice');
    expect(USERS_LEGEND).toBe('guaranteed = holds even if the real speed is at the low end of the estimate; likely = at the central estimate.');
  });

  it('the speed basis names only the cards shown: the RTX 5090 only when one of its rows is shown', () => {
    const h100Only = legend(m, false, [fH100, noSpeed]);
    expect(h100Only.at(-1)).toBe(`Speed basis (vLLM 0.31.0) of the cards shown: ${fH100.tokS?.precision.basis} (${fH100.tokS?.precision.points} points).`);
    expect(h100Only.join(' ')).not.toContain('5090');
    expect(legend(renderModel({ engine: 'sglang' }), false, [fH100]).at(-1)).toContain('Speed basis (SGLang 0.5.21) of the cards shown: ');
    expect(legend(m, false, [f5090, fH100, f5090]).at(-1)).toContain('RTX 5090');
    expect(shownBases([fH100, fH100, undefined, noSpeed])).toHaveLength(1);
    expect(shownBases([{ ...fH100, tokS: { ...(fH100.tokS as NonNullable<Figures['tokS']>), precision: { kind: 'estimated', errorPct: 1, basis: 'b' } } }])).toEqual(['b']);
  });

  it('the weights: verified, with a measured gap, not verified', () => {
    expect(weightsNote(m)).toBe('Weights: exact (sum of the loaded tensors; BF16 × denso verified on vLLM measurements).');
    expect(weightsNote(renderModel({ verification: { status: 'gap', gapShare: 0.026, gapFixedMiB: 0, evidence: [] } }))).toBe(
      'Weights: exact as the sum of the tensors; for BF16 × denso vLLM measured up to +2.6% more, counted in the fixed costs.',
    );
    expect(weightsNote(renderModel({ verification: { status: 'unverified', gapShare: 0.05, gapFixedMiB: 0, evidence: [] } }))).toBe(
      'Weights: exact as the sum of the tensors; BF16 × denso not yet verified by a measurement: 5% margin in the fixed costs.',
    );
  });
});

describe('the details of one layout', () => {
  it('every number with its accuracy; the roofline as a share of the physical limits', () => {
    const d = detailLines(m, f5090, 300);
    expect(d[0]).toBe('Details — 1× RTX 5090, one request of 32k tokens:');
    const text = d.join('\n');
    // The writing speed under load says what it includes, next to one request alone.
    expect(text).toMatch(/writing speed per user +[\d.]+ tok\/s .* at once, including reading other users' prompts/);
    expect(text).toMatch(/memory per GPU +[\d.]+ GiB \([\d.]+ to [\d.]+ GiB, ±[\d.]+%\) of 31\.3 GiB {3}estimated \(±[\d.]+%, the fixed costs are a range\)/);
    expect(text).toMatch(/ {4}weights +[\d.]+ GiB {3}exact \(sum of the loaded tensors; format verified on vLLM measurements\)/);
    expect(text).toMatch(/held by the engine +[\d.]+ GiB \(.*\) of 31\.3 GiB\n +the engine's share \(--gpu-memory-utilization × the CUDA memory\), all of it reserved, plus what the process keeps outside it \(CUDA context, communication buffers\);/);
    expect(text).toMatch(/one request alone +\d+ tok\/s \(\d+ to \d+, ±\d+%\) = \d+% of the physical limit \(\d+ tok\/s at the datasheet bandwidth\) {3}measured/);
    expect(text).toMatch(/reading 32k tokens +[\d.]+ s \(.*\); the physical limit \([\d.]+ s at the datasheet TFLOPS\) = \d+% of it/);
    expect(text).toMatch(/concurrent requests +\d+ guaranteed, \d+ likely \(limited by (?:speed|memory); the memory holds \d+ at 16k\)/);
    expect(text).toContain('context validity          context calibrated up to 1M');
    expect(text).toContain('engine                    starts');
    expect(text).not.toContain('roofline check');
    expect(text).not.toContain('saturation');
  });

  it('narrow: the accuracy on the next line; ASCII labels; one request; caveats; warnings', () => {
    const one = { ...f5090, requests: { guaranteed: 1, likely: 1, memory: 2, limitedBy: 'wait' as const }, roofline: { ...(f5090.roofline as NonNullable<Figures['roofline']>), warning: 'above' }, saturation: 'saturated' };
    const d = detailLines(m, one, 60, true);
    expect(d.some((l) => /^ {28}measured/.test(l))).toBe(true);
    const flat = d.join(' ').replace(/ +/g, ' ');
    expect(flat).toMatch(/1 guaranteed, 1 likely \(limited by first-token wait/);
    expect(flat).toMatch(/with 1 request/);
    expect(Math.max(...d.map((l) => l.length))).toBeLessThanOrEqual(60);
    expect(d.join('\n')).toContain('roofline check            above');
    expect(d.join('\n')).toContain('saturation                saturated');
    expect(detailLines(m, line('rtx-3090-24gb').figures, 300).join('\n')).toMatch(/engine {20}starts, with warnings: /);
    // A card whose TFLOPS are derived: the roofline says "derived", not "datasheet".
    const r5 = f5090.roofline as NonNullable<Figures['roofline']>;
    expect(detailLines(m, { ...f5090, roofline: { ...r5, tflopsDerived: true } }, 300).join('\n')).toContain('at the derived TFLOPS)');
    expect(detailLines(m, f5090, 300).join('\n')).toContain('at the datasheet TFLOPS)');
    expect(detailLines(m, { ...f5090, compat: { status: 'does-not-start', notes: ['32 heads'] } }, 300).join('\n')).toContain('engine                    does not start: 32 heads');
  });

  it('long values and notes wrap under the value column at the terminal width', () => {
    const long = { ...line('rtx-3090-24gb').figures, compat: { status: 'warnings' as const, notes: ['a first caveat that is rather long and keeps going', 'a second caveat that is also quite long and keeps going too'] } };
    const d = detailLines(m, long, 80);
    expect(Math.max(...d.map((l) => l.length))).toBeLessThanOrEqual(80);
    const engine = d.findIndex((l) => l.startsWith('  engine '));
    expect(d[engine]).toMatch(/^ {2}engine {20}starts, with warnings: a first/);
    expect(d[engine + 1]).toMatch(/^ {28}\S/);
    // A value too long for the line with its note: the note below it, wrapped too.
    const narrow = detailLines(m, f5090, 70);
    expect(Math.max(...narrow.map((l) => l.length))).toBeLessThanOrEqual(70);
    expect(narrow.some((l) => /^ {28}estimated \(/.test(l))).toBe(true);
  });

  it('no speed: why', () => {
    const { speedNote: _, ...bare } = noSpeed;
    expect(detailLines(m, bare, 300).at(-1)).toBe('  speed                     not estimable   the card’s bandwidth or TFLOPS are not published');
    expect(detailLines(m, { ...bare, speedNote: 'why' }, 300).at(-1)).toBe('  speed                     not estimable   why');
  });

  it('the memory as the engine counts it, item by item', () => {
    const r5090 = planLayout(qwen, at('rtx-5090-32gb'), ctx.opts);
    const d = memoryDetails(m, r5090);
    expect(d[0]).toBe('Memory as the engine counts it (worst–best case):');
    expect(d.join('\n')).toMatch(/ weights +15.3 GiB {2}\(sum of the loaded tensors\)/);
    for (const item of ['memory visible to CUDA', "the engine's share", 'CUDA context', 'workspaces', 'activations', 'CUDA graphs', 'fixed costs in total', 'room for the KV cache', 'KV cache of one 32k request', 'requests at the full context', 'margin after one request'])
      expect(d.some((l) => l.includes(item))).toBe(true);
    expect(d.join('\n')).toContain('(--gpu-memory-utilization 0.9)');
    const single = { ...r5090, memory: { ...r5090.memory, worst: { ...r5090.memory.worst, concurrency: { min: 2.5, max: 2.5 } } } };
    expect(memoryDetails(m, single).join('\n')).toMatch(/requests at the full context +2$/m);
    const GB = 1024 ** 3;
    const worst = r5090.memory.worst;
    const rich: PlanRow = {
      ...r5090,
      weights: { ...r5090.weights, replicatedBytes: GB, visionBytes: 2 * GB, ropeBytes: GB / 2 },
      memory: { ...r5090.memory, worst: { ...worst, concurrency: { min: 1, max: 3 }, fixed: { ...worst.fixed, weightsExtra: { min: 100, max: 300 }, vision: { min: 200, max: 400 }, nccl: { min: 10, max: 20 } } } },
    };
    const s = memoryDetails(renderModel({ engine: 'sglang' }), rich, true).join('\n');
    expect(s).toMatch(/^Memory as the engine counts it \(worst-best case\):/);
    expect(s).toContain('(sum of the loaded tensors; of which replicated on every GPU 1.0 GiB, vision encoder 2.0 GiB, RoPE cache 0.5 GiB)');
    expect(s).toMatch(/weights added by the engine +0.1-0.3 GiB {2}\(MLA copies or the measured gap of the format\)/);
    expect(s).toMatch(/vision encoder profile +0.2-0.4 GiB/);
    expect(s).toContain('(before the share)');
    expect(s).toContain(`(--mem-fraction-static ${worst.util})`);
    expect(s).toContain('NCCL');
    expect(s).toMatch(/requests at the full context +1-3/);
    const { weightsExtra: _, ...bareFixed } = worst.fixed;
    expect(memoryDetails(m, { ...r5090, memory: { ...r5090.memory, worst: { ...worst, fixed: bareFixed } } }).join('\n')).not.toContain('weights added by the engine');
  });
});

describe('the whole text', () => {
  it('answer first, scenario, model, what was read, the overview per card with notes and its legend; fits 120 columns', () => {
    const t = renderText({ ...m, source: { kind: 'cache', bytes: 92_000 }, compatChecked: false }, answer, plain);
    const lines = t.split('\n');
    expect(lines[0]).toBe('Summary');
    expect(lines[1]?.startsWith('  Recommendation: 1 server with 1× NVIDIA RTX 3090 serves 10 users guaranteed, 10 likely, with requests of up to')).toBe(true);
    const detail = lines.indexOf('Technical detail');
    expect(detail).toBeGreaterThan(3);
    expect(detail).toBeLessThan(15);
    expect(lines[detail + 1]?.startsWith('Qwen3-8B at 32k tokens, fp8 KV cache: serves users from')).toBe(true);
    expect(t).toContain(`\n  ${LAYOUT_LEGEND.slice(0, 40)}`);
    expect(t).toContain('Scenario "documents"');
    expect(t).toContain('Read config + tensor headers (90 KB) from the local cache — weights not downloaded.');
    expect(t.replace(/\n/g, ' ')).toContain('Per card: the fewest GPUs that serve someone at the minimum writing speed (else the fewest the model fits on), and what that layout serves.');
    expect(t).toMatch(/\n {2}RTX 5090 \(this PC\) +fits +1 +/);
    expect(t).toContain(`\n  ${USERS_LEGEND.slice(0, 40)}`);
    expect(t).toMatch(/\n\(1\) /);
    expect(t).toContain('Weights: exact');
    expect(t).toContain(COMPAT_UNCHECKED.slice(0, 40));
    expect(t).not.toContain('Details');
    expect(t).not.toContain('Every layout');
    expect(t).not.toContain('Multimodal model');
    expect(t).not.toContain(AGENTS_NOTE.slice(0, 30));
    expect(Math.max(...lines.map((l) => l.length))).toBeLessThanOrEqual(120);
    expect(renderText({ ...m, compatChecked: true }, answer, plain)).not.toContain(COMPAT_UNCHECKED.slice(0, 40));
  });

  it('wide: the long forms; colours; ASCII; never narrower than 60 columns', () => {
    const wide = renderText(m, answer, { ...plain, columns: 300 });
    expect(wide).toMatch(/of 31\.3 GiB +[\d.]+ +\d+ +\d+ +\d+ tok\/s \(\d+ to \d+, ±\d+%\) +[\d.]+ s \([\d.]+ to/);
    const colour = renderText(m, answer, { ...plain, color: true });
    expect(colour.startsWith('\u001b[1mSummary')).toBe(true);
    expect(colour).toContain('\u001b[1mTechnical detail');
    expect(colour).toContain('\u001b[32mfits');
    const ascii = renderText(m, answer, { ...plain, ascii: true });
    expect(ascii).not.toMatch(/[–×±]/);
    expect(renderText(m, answer, { ...plain, columns: 10 }).split('\n')[0]?.length).toBeLessThanOrEqual(60);
  });

  it('--users (with the agents), --all-hw, --details (with and without the engine items), a multimodal model', () => {
    const u = renderText(m, { ...answer, users: { total: 200, concurrent: 20 }, layouts: answer.rows.slice(0, 3), details: { figures: f5090, row: planLayout(qwen, at('rtx-5090-32gb'), ctx.opts) } }, plain);
    // Nobody serves them: the closest layout per card, as the summary names it.
    expect(u).toContain('None serves 200 users (10% active ≈ 20 concurrent requests); per card, the closest layout.');
    const served = renderText(m, { ...answer, users: { total: 200, concurrent: 20, choice: { figures: fH100 } } }, plain);
    expect(served).toContain('Per card: the layout with the fewest GPUs in total that serves 200 users (10% active ≈ 20 concurrent requests)');
    const nobodyAll = renderText(m, { ...answer, fit: { figures: withUsers(fH100, { guaranteed: 0 }) } }, plain);
    // The binding limit, named (I-C): the writing speed, the first-token wait or the memory.
    expect(nobodyAll.replace(/\n/g, ' ')).toContain('No layout serves anyone: nobody is guaranteed 20 tok/s of writing; per card, the closest layout.');
    const bound = (limitedBy: 'memory' | 'wait') =>
      renderText(m, { ...answer, fit: { figures: { ...withUsers(fH100, { guaranteed: 0 }), requests: { ...(fH100.requests as NonNullable<Figures['requests']>), limitedBy } } } }, plain).replace(/\n/g, ' ');
    expect(bound('wait')).toContain('No layout serves anyone: nobody gets the first words within the wait allowed (--max-wait); per card, the closest layout.');
    expect(bound('memory')).toContain('No layout serves anyone: the memory left after the model holds no full-length request with its safety margin; per card, the closest layout.');
    expect(renderText(m, answer, plain).replace(/\n/g, ' ')).toContain('Per card: the fewest GPUs that serve someone at the minimum writing speed (else the fewest the model fits on), and what that layout serves.');
    expect(u).toContain('Users include AI agents.');
    expect(u).toContain('Every layout:');
    expect(u).toContain('Details — 1× RTX 5090');
    expect(u).toContain('Memory as the engine counts it');
    const d = renderText(renderModel({ ...m, hasVision: true }), { ...answer, details: { figures: f5090 } }, plain);
    expect(d).toContain('Details');
    expect(d).not.toContain('Memory as the engine counts it');
    expect(d).toContain('Multimodal model: with --text-only');
    expect(renderText(renderModel({ ...m, hasVision: true, textOnly: true }), answer, plain)).not.toContain('Multimodal model');
  });

  it('no notes: no footnotes', () => {
    const clean = answer.cards.filter((c) => lineNotes(c).length === 0);
    expect(renderText(m, { ...answer, cards: clean }, plain)).not.toMatch(/\n\(1\)/);
  });

  it('footnotes: a sentence already printed is left out of a later note; a note made only of those reuses its number', () => {
    const notes = new Notes();
    expect(notes.mark(['partly supported: A runs. Open bug here. Workaround: x.'])).toBe(' (1)');
    expect(notes.mark(['partly supported: B runs. Open bug here. Workaround: y.'])).toBe(' (2)');
    expect(notes.texts[1]).toBe('partly supported: B runs. Workaround: y.');
    expect(notes.mark(['partly supported: A runs. Open bug here.'])).toBe(' (1)');
    expect(notes.mark(['partly supported: A runs. Open bug here. Workaround: x.', undefined])).toBe(' (1)');
    expect(notes.mark([])).toBe('');
  });

  it('the binding limit in the verdict, the headline and --md: memory, the first-token wait, the writing speed', () => {
    const by = (limitedBy: 'memory' | 'wait' | 'speed', status: CardLine['status'] = 'short') => {
      const f = withUsers(fH100, { guaranteed: 0, likely: 2 });
      return { ...line('h100-sxm5-80gb'), status, figures: { ...f, requests: { ...(f.requests as NonNullable<Figures['requests']>), limitedBy } } };
    };
    const verdict = cardColumns(W, new Notes(), false, { total: 200 })[1] as Column<CardLine>;
    expect(verdict.long(by('wait'))).toMatch(/^fits, first words too late/);
    expect(verdict.short?.(by('wait'))).toMatch(/^wait too long/);
    expect(verdict.long(by('memory'))).toMatch(/^fits, no room for a request/);
    expect(verdict.short?.(by('memory', 'at-limit'))).toMatch(/^limit, no room/);
    expect(verdict.long(by('speed'))).toMatch(/^too slow for anyone/);
    const md = renderMarkdown(m, { ...answer, cards: [by('wait'), by('memory', 'at-limit')] });
    expect(md).toContain('| serves nobody: nobody gets the first words within the wait allowed (--max-wait); 2 users likely |');
    expect(md).toContain('| at the limit of memory; serves nobody: the memory left after the model holds no full-length request with its safety margin; 2 users likely |');
    const head = (limitedBy: 'memory' | 'wait') => headline(m, { ...answer, fit: { figures: by(limitedBy).figures } });
    expect(head('wait')).toContain(': no layout serves anyone (nobody gets the first words within the wait allowed (--max-wait)); the closest is');
    expect(head('wait')).toContain('With 1× H100 SXM5 80GB, nobody gets the first words within the wait allowed (--max-wait) (2 likely;');
    expect(head('memory')).toContain('no layout serves anyone (the memory left after the model holds no full-length request with its safety margin)');
  });

  it('pipeline stages said in the summary and the card notes; the copies in one server; the wait limit as written', () => {
    const find = (l: string[], start: string) => l.find((x) => x.startsWith(start));
    const staged = { ...fH100, count: 2, tp: 1, pp: 2 };
    expect(find(summaryLines(m, { ...answer, fit: { figures: staged } }), 'Confidence:')).toContain('; pipeline speed not measured (its ranges are widened by half)');
    expect(find(summaryLines(m, { ...answer, fit: { figures: fH100 } }), 'Confidence:')).not.toContain('pipeline');
    expect(lineNotes({ ...line('h100-sxm5-80gb'), figures: staged })).toContain(PIPELINE_NOTE);
    expect(lineNotes({ ...line('h100-sxm5-80gb'), figures: fH100 })).not.toContain(PIPELINE_NOTE);
    // Two copies in one server (8 (2×4)): not "its own copy".
    const two = { ...fH100, count: 8, tp: 4, replicas: 2, servers: 1 };
    const s = summaryLines(m, { ...answer, users: { total: 2000, concurrent: 200, choice: { figures: two, servers: 3, usersPerServer: 700 } } });
    expect(s[0]).toContain('Each server runs 2 copies of the model, and a load balancer');
    expect(summaryLines(m, { ...answer, users: { total: 2000, concurrent: 200, choice: { figures: fH100, servers: 3, usersPerServer: 700 } } })[0]).toContain('Each server runs its own copy of the model');
    // --max-wait 0.5 is 0.5 s, not "1 s".
    expect([waitLimitText(500), waitLimitText(10_000), waitLimitText(50)]).toEqual(['0.5 s', '10 s', '0.05 s']);
  });

  it('a speed the calibration does not trust: no recommendation but the estimate, no alternative, no plan on the user counts', () => {
    const find = (l: string[], start: string) => l.find((x) => x.startsWith(start));
    const note = 'INT4 MoE without Blackwell: public measurements are far below the estimate, speed not reliable';
    const shaky = { ...fH100, speedNote: note };
    const s = summaryLines(m, { ...answer, fit: { figures: shaky, alternative: f5090 } });
    // Figures marked "do not plan" make no recommendation (I-B): the estimate, said as one, with the reason once.
    expect(s[0]).toContain(`Recommendation: none reliable — the speed is not reliable here (${note}). The estimate, not to plan on: 1 server with 1× NVIDIA H100 SXM5 80GB serves `);
    expect(s[0]).not.toContain('Caution: the speed is not reliable');
    expect(find(s, 'Alternative:')).toBeUndefined();
    expect(find(s, 'Confidence:')).toMatch(/Do not plan on these user counts: the speed is not reliable here \(see the recommendation above\)\.$/);
    // With --users too: the servers as an estimate, not a plan.
    const forUsers = summaryLines(m, { ...answer, users: { total: 2000, concurrent: 200, choice: { figures: shaky, servers: 9, usersPerServer: 230 } } });
    expect(forUsers[0]).toMatch(/^Recommendation: none reliable — the speed is not reliable here .*\. The estimate, not to plan on: 9 servers with 1× NVIDIA H100 SXM5 80GB each for 2,000 users .*: ≈ 230 users per server\.( Caution: .*)?$/);
    expect(find(forUsers, 'Alternative:')).toBeUndefined();
    expect(headline(m, { ...answer, fit: { figures: shaky, alternative: f5090 } })).not.toContain(', or 1× RTX 5090');
    expect(headline(m, { ...answer, users: { total: 200, concurrent: 20, choice: { figures: shaky, alternative: f5090 } } })).not.toContain(', or 1× RTX 5090');
    // Nothing recommended: the caveat in full in the confidence line.
    const slowShaky = { ...withUsers(shaky, { guaranteed: 0 }) };
    const none = summaryLines(m, { ...answer, fit: { figures: slowShaky } });
    expect(find(none, 'Confidence:')).toMatch(/Nothing is guaranteed here/);
    // A user count not precise enough: no plan either.
    const u = fH100.users as NonNullable<Figures['users']>;
    const vagueUsers = { ...fH100, users: { ...u, precision: { ...u.precision, kind: 'imprecise' as const, errorPct: 30, reason: 'x' } } };
    const vs = summaryLines(m, { ...answer, fit: { figures: vagueUsers, alternative: f5090 } });
    expect(vs[0]).toMatch(/^Recommendation: none reliable — the user count is not precise enough to decide\. The estimate, not to plan on: /);
    expect(find(vs, 'Alternative:')).toBeUndefined();
    expect(find(vs, 'Confidence:')).toMatch(/Do not plan on these user counts: the user count is not precise enough to decide\.$/);
    // A card without a speed keeps its alternative (no speed is no unreliable speed).
    expect(summaryLines(m, { ...answer, fit: { figures: noSpeed, alternative: noSpeed } }).find((x) => x.startsWith('Alternative:'))).toContain('(fits; speed unknown)');
  });

  it('the legend: the server sizes of the cards shown; at the limit and too slow, both in the verdict', () => {
    const card = (id: string, serverSize: number) => ({ ...line(id), serverSize });
    expect(layoutLegend([card('h100-sxm5-80gb', 8)])).toMatch(/ One server holds up to 8 GPUs\.$/);
    // Every label explained, copies × stages included.
    expect(LAYOUT_LEGEND).toContain('"2 copies × 2 stages of 2" or "2×2 stages of 2" = 2 independent copies, each split in 2 stages of 2 GPUs');
    expect(layoutLegend([card('h100-sxm5-80gb', 8), card('rtx-5090-32gb', 4), card('rtx-4090-24gb', 4), card('gh200-96gb', 2)])).toMatch(/ One server holds up to 8 GPUs \(RTX 5090, RTX 4090: 4; GH200 96GB: 2\)\.$/);
    expect(layoutLegend([card('gh200-96gb', 2)])).toMatch(/ One server holds up to 2 GPUs of these cards\.$/);
    // Cards packed without their server size (older reports): 8.
    const { serverSize: _s, ...bare } = line('h100-sxm5-80gb');
    expect(layoutLegend([bare])).toMatch(/ One server holds up to 8 GPUs\.$/);
    const notes = new Notes();
    const verdict = cardColumns(W, notes, false)[1] as Column<CardLine>;
    const edge = { ...line('h100-sxm5-80gb'), status: 'at-limit' as const, figures: withUsers(fH100, { guaranteed: 0 }) };
    expect(verdict.long(edge)).toMatch(/^at the limit, too slow/);
    expect(verdict.short?.(edge)).toMatch(/^limit, slow/);
    expect(verdict.long({ ...edge, figures: fH100 })).toMatch(/^at the limit( \(\d+\))?$/);
    // Does not fit: the GPUs tried in one server, not the copies over more servers.
    expect(verdict.long({ ...line('rtx-5090-32gb'), status: 'no-fit', serverGpus: 4 })).toMatch(/^does not fit even with 4/);
  });
});

describe('exports', () => {
  it('CSV, standard: ",", decimal point, UTF-8 without BOM, CRLF; one line per layout; value, min, max, error and basis per group', () => {
    const csv = renderCsv(m, answer, false);
    expect(csv.startsWith('Model,')).toBe(true);
    expect(csv.endsWith('\r\n')).toBe(true);
    const lines = csv.trimEnd().split('\r\n');
    expect(lines).toHaveLength(rows.length + 1);
    expect(lines[0]).toBe(CSV_HEADERS.join(','));
    for (const g of ['Memory per GPU', 'KV cache per request', 'Fixed costs', 'Speed per user', 'Context reading', 'First token under load'])
      for (const suffix of [/ \((GiB|tok\/s|s)\)$/, / min \(/, / max \(/, / error \(%\)$/, / basis$/]) expect(CSV_HEADERS.some((h) => h.startsWith(g) && suffix.test(h)), `${g} ${suffix}`).toBe(true);
    for (const h of ['Users likely per layout', 'Users min (guaranteed) per layout', 'Users max (memory limit) per layout', 'Users error (%)', 'Users basis', 'Engine compatibility', 'Context validity']) expect(CSV_HEADERS).toContain(h);
    const at5090 = lines.find((l) => l.includes(',rtx-5090-32gb,') && l.includes(',RTX 5090,1,1,,1,1,')) as string;
    expect(at5090).toContain(',yes,31.30,');
    expect(at5090).toContain(',fits,fits,starts,context calibrated up to 1M,');
    expect(at5090).toMatch(/,\d+\.\d\d,\d+\.\d\d,\d+\.\d\d,\d+\.\d\d,"?estimated \(±/);
    expect(at5090).toContain('measured (±');
    // A layout not in the overview: no status, not this PC.
    expect(lines.find((l) => l.includes('8× RTX 5090 (8 copies × 1)'))).toMatch(/,no,31\.30,8× RTX 5090 \(8 copies × 1\),8,2,,1,1,8,,fits,/);
    expect(CSV_HEADERS.slice(CSV_HEADERS.indexOf('GPUs'), CSV_HEADERS.indexOf('GPUs') + 6)).toEqual(['GPUs', 'Servers per layout', 'Servers needed for the users', 'TP', 'PP', 'Copies']);
    // With --users: the servers each card needs.
    expect(renderCsv(m, { ...answer, cards: [{ ...line('h100-sxm5-80gb'), serversNeeded: 3 }] }, false)).toMatch(/,H100 SXM5 80GB,1,1,3,1,1,/);
    // A card's note goes with its layout.
    expect(renderCsv(m, { ...answer, cards: [{ ...line('h100-sxm5-80gb'), note: 'the reason' }] }, false)).toMatch(/,the reason\r\n/);
    // A card without speed: the speed groups empty.
    expect(lines.find((l) => l.includes(',rtx-pro-6000-blackwell-server-96gb,') && l.includes(',1,1,,1,1,'))).toContain(',,,,,,,,,,');
  });

  it('CSV for Excel in comma-decimal locales: BOM, ";", decimal comma; quotes only where needed; integers as they are', () => {
    const cm = { ...csvModel(m), model: 'a;"b"' };
    const csv = csvText([[cm, { figures: f5090, status: 'slow', note: 'line\nbreak', local: true }]], true);
    expect(csv.startsWith('﻿')).toBe(true);
    const body = csv.slice(1).split('\r\n')[1] as string;
    expect(body.startsWith('"a;""b""";Qwen/Qwen3-8B;BF16;denso;vLLM;0.31.0;fp8;32768;chosen (--tokens);16384;16384;500;20;10;RTX 5090;rtx-5090-32gb;consumer;yes;31,30;RTX 5090;1;1;;1;1;1;fits, too slow;fits;starts;')).toBe(true);
    expect(body).toMatch(/;\d+,\d\d;/);
    expect(body).toContain('"line\nbreak');
    expect(csvModel(renderModel({ variant: 'FP8', engine: 'sglang' }))).toMatchObject({ version: 'FP8', engine: 'SGLang', contextLabel: 'model maximum' });
    expect(csvText([[csvModel(m), { figures: line('rtx-3090-24gb').figures }]], false)).toContain(',"starts, with warnings: ');
    expect(csvText([[csvModel(m), { figures: { ...f5090, compat: { status: 'does-not-start', notes: ['a', 'b'] } } }]], false)).toContain(',does not start: a | b,');
  });

  it('Markdown: a one-page summary — scenario, recommendation, options in plain words, confidence, assumptions, method', () => {
    const md = renderMarkdown(m, { ...answer, cards: answer.cards.map((c) => (c.figures.gpuId === 'rtx-5090-32gb' ? { ...c, serversNeeded: 4 } : c)), users: { total: 200, concurrent: 20, choice: { figures: fH100 } } });
    const sections = ['# Summary: Qwen3-8B for 200 users', '## Scenario', '## Recommendation', '## Options', '## Confidence', '## Assumptions', '## Method'];
    const at = sections.map((s) => md.indexOf(s));
    expect(at.every((i) => i >= 0)).toBe(true);
    expect([...at].sort((a, b) => a - b)).toEqual(at);
    expect(md).toContain('- Users: 200 (10% of them active at the same moment ≈ 20 requests at once).');
    expect(md).toContain('each request up to 32k tokens (≈ 25,000 words ≈ 49 pages), answers of 500 tokens (≈ 380 words)');
    expect(md).toContain('- Recommendation: 1 server with 1× NVIDIA H100 SXM5 80GB serves 200 users with requests of up to 32k tokens (≈ 25,000 words ≈ 49 pages; 16k on average)');
    expect(md).toContain('| Each server | Servers needed | How the GPUs hold the model | Verdict | Users (guaranteed, per server) | Users (likely, per server) | Words per second per user | Answer starts after (s) | Accuracy (users, speed) |');
    expect(md).toMatch(/\| 1× NVIDIA RTX 5090 \| 4 \| one GPU \| fits \| \d+ \| \d+ \| about \d+ \(\d+ to \d+\) \| about [\d.]+ \([\d.]+ to [\d.]+\) \| (measured|calibrated|estimated) ±\d+% \|/);
    expect(md).toMatch(/\| 2× NVIDIA RTX 5080 \| n\/a \| 1 model on 2 \|/);
    expect(md).toContain('| 1× NVIDIA RTX PRO 6000 Server | n/a | one GPU | fits, speed unknown | n/a | n/a | n/a | n/a | n/a |');
    expect(md).toMatch(/\n## Confidence\n\nConfidence: memory high/);
    expect(md).toContain('- 10% of the users send a request at the same moment (--active).');
    expect(md).toContain('- Every user gets at least 15 words per second (20 tokens/s) of writing and the first words within 10 s, or, if longer, within twice the wait for the first words of one typical request alone (half the context, less what the prefix cache holds).');
    expect(md).toContain('- Words and pages: 0.75 words per token, 500 words per page.');
    expect(md).toContain('docs/METHODOLOGY.md');
    // A card that guarantees nobody: the cause next to the speed shown (the central estimate under load).
    const slowCard = (status: CardLine['status'], likely: number) => ({ ...line('h100-sxm5-80gb'), status, figures: withUsers(fH100, { guaranteed: 0, likely }) });
    const md2 = renderMarkdown(m, { ...answer, cards: [slowCard('short', 3), slowCard('at-limit', 0), slowCard('slow', 0)] });
    expect(md2).toContain('| serves nobody: nobody is guaranteed 15 words per second (20 tokens/s) of writing (low end of the estimate); 3 users likely |');
    expect(md2).toContain('| at the limit of memory; serves nobody: nobody is guaranteed 15 words per second (20 tokens/s) of writing (low end of the estimate) |');
    expect(md2).not.toContain('too slow for anyone');
    // No unexplained jargon: no "TP4", and "KV cache" only next to its explanation.
    expect(md).not.toMatch(/\bTP\d/);
    expect(md.match(/KV cache/g)?.length).toBe(md.match(/\(KV cache\)/g)?.length);
    const chat = renderMarkdown(renderModel({ ...m, cache: 'auto', workload: { ...W, cachedShare: 0.8, ttftScales: false }, variant: 'FP8' }), answer);
    expect(chat).toContain('# Summary: Qwen3-8B (FP8)\n');
    expect(chat).toContain('- Users: as many as the hardware serves (10% of them active at the same moment).');
    expect(chat).toContain('a chat: most of each conversation (80%) is already held by the server');
    expect(chat).toContain("kept at the model's full precision");
    expect(chat).toContain('the first words within 10 s.');
    expect(renderMarkdown(renderModel({ ...m, engine: 'sglang' }), answer)).toContain(', BF16 weights, served with SGLang 0.5.21.');
    const none = renderMarkdown(renderModel({ ...m, contextIsModelMax: false, modelMaxContext: 16_384 }), { ...answer, cards: answer.cards.map((c) => ({ ...c, status: 'no-fit' as const })) });
    expect(none).toContain('No card of the catalog holds the model and starts with up to 8 GPUs.');
    expect(none).toContain("- Note: 32k tokens is beyond the model's own maximum (16k)");
  });

  it('JSON: the model, the answer, the overview per card, every layout and the legend', () => {
    const j = JSON.parse(renderJson(m, { ...answer, details: { figures: f5090 } })) as Record<string, unknown>;
    expect(Object.keys(j)).toEqual(['model', 'summary', 'headline', 'scenario', 'fit', 'users', 'cards', 'details', 'rows', 'units', 'accuracy']);
    expect(j.summary).toEqual(summaryLines(m, answer));
    expect(j.headline).toBe(headline(m, answer));
    expect((j.cards as unknown[]).length).toBe(GPUS.length);
    expect((j.rows as unknown[]).length).toBe(rows.length);
    const { fit: _, ...rest } = answer;
    const bare = JSON.parse(renderJson(m, { ...rest, attempt: f5090 })) as Record<string, unknown>;
    expect(bare.fit).toBeNull();
    expect(bare.users).toBeNull();
    expect('details' in bare).toBe(false);
    expect(hardwareLabel({ layout: at('rtx-5090-32gb') })).toBe('RTX 5090');
  });
});

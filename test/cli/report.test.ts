import { describe, expect, it } from 'vitest';
import reportJson from '../../data/report.json';
import { cardLines, figuresOf, generalAnswer, type CardLine, type FigureContext } from '../../src/cli/figures';
import { CSV_HEADERS, USERS_LEGEND } from '../../src/cli/render';
import {
  cardSummary,
  doNotFit,
  OVERVIEW_MAX_LINES,
  packReport,
  renderReport,
  reportCsv,
  reportFileSchema,
  reportJsonText,
  reportLines,
  reportMarkdown,
  unpackReport,
  type ReportContext,
  type ReportFile,
} from '../../src/cli/report';
import { layoutsFor } from '../../src/planner/hardware';
import { planLayout, rankRows } from '../../src/planner/plan';
import { GPUS, opts, qwen, workload } from './fixtures';

// Qwen3-8B at 32k (every card) and a context that fits nowhere, as report.mjs writes them.
const ctx: FigureContext = { model: qwen, opts: { ...opts(32_768, { kvCache: 'fp8' }), workload: workload(32_768) }, repo: 'Qwen/Qwen3-8B', verification: qwen.verification };
const rows = rankRows(layoutsFor(GPUS).map((l) => planLayout(qwen, l, ctx.opts)));
const cards = cardLines(ctx, rows, GPUS).map((c) => c.line);
const g = generalAnswer(rows);
const fits: ReportContext = {
  tokens: 32_768,
  label: 'native maximum context',
  format: 'bf16',
  family: 'dense',
  verification: 'verified',
  fit: figuresOf(ctx, g.fit?.row as never),
  alternative: figuresOf(ctx, g.fit?.alternative as never),
  chat: { guaranteed: 30, likely: 40 },
  cards,
};
const none = rows.map((r) => ({ ...r, verdict: 'does-not-fit' as const }));
const nowhere: ReportContext = {
  tokens: 1_048_576,
  label: 'official YaRN',
  args: { vllm: ['--max-model-len', '1048576'] },
  format: 'bf16',
  family: 'dense',
  verification: 'gap',
  attempt: figuresOf(ctx, generalAnswer(none).attempt as never),
  cards: cardLines(ctx, none, GPUS).map((c) => c.line),
};
const { alternative: _a, chat: _c, ...lean } = fits;
const file: ReportFile = {
  engine: 'vllm',
  engineVersion: '0.31.0',
  workload: { outputTokens: 500, minTokS: 20, maxTtftMs: 10_000, activeShare: 0.1, safety: 0.85, cachedShare: 0 },
  chatCachedShare: 0.8,
  models: [
    { id: 'qwen3-8b', repo: 'Qwen/Qwen3-8B', name: 'Qwen3-8B', toMillion: 'YaRN', variants: [{ format: 'bf16', repo: 'Qwen/Qwen3-8B', contexts: [fits, nowhere] }] },
    { id: 'glm-5', repo: 'zai-org/GLM-5', name: 'GLM 5', toMillion: 'no', variants: [{ format: 'fp8', repo: 'zai-org/GLM-5-FP8', contexts: [{ ...lean, verification: 'unverified' }] }] },
  ],
  missing: ['Kimi K3 NVFP4'],
};
const plain = { color: false, ascii: false, columns: 120 };

describe('the packed file', () => {
  it('packs every object as its keys and values, every string once, and back without loss', () => {
    const value = { a: 'x', b: [1, 'x', { a: 'y', c: null, d: true }], e: undefined, f: '~a' };
    const p = packReport(value);
    expect(p.strings).toEqual(['x', 'y', '~a']);
    expect(p.shapes).toEqual([
      ['a', 'b', 'f'],
      ['a', 'c', 'd'],
    ]);
    expect(unpackReport(p)).toEqual({ a: 'x', b: [1, 'x', { a: 'y', c: null, d: true }], f: '~a' });
    expect(unpackReport(packReport(file))).toEqual(JSON.parse(JSON.stringify(file)));
  });

  it('the schema reads a packed or an unpacked file; the bundled one passes and is in English', () => {
    const { models, missing, ...rest } = file;
    expect(reportFileSchema.parse({ ...rest, packed: packReport({ models, missing }) })).toEqual(JSON.parse(JSON.stringify(file)));
    expect(reportFileSchema.parse(file)).toEqual(file);
    const bundled = reportFileSchema.parse(reportJson);
    expect(bundled.models.length).toBeGreaterThan(30);
    // The command's own texts are English (reasons and notes may quote the planner's checks as they are).
    const own: string[] = [];
    JSON.stringify(bundled, (k, v: unknown) => {
      if (['label', 'toMillion', 'basis', 'validity', 'hardware', 'saturation', 'warning'].includes(k) && typeof v === 'string') own.push(v);
      return v;
    });
    expect(own.length).toBeGreaterThan(1000);
    expect(own.join('\n')).not.toMatch(/ entra |non entra|richiest|velocità|contesto|garantit|probabil|copie\)/);
    expect(() => reportFileSchema.parse(null)).toThrow();
  });
});

describe('the lines of the report', () => {
  it('one per model version and context; a filter on id, name or repo, any case', () => {
    expect(reportLines(file).map((l) => `${l.model} ${l.version} ${l.context.tokens}`)).toEqual(['Qwen3-8B bf16 32768', 'Qwen3-8B bf16 1048576', 'GLM 5 fp8 32768']);
    expect(reportLines(file, 'ZAI').map((l) => l.versionRepo)).toEqual(['zai-org/GLM-5-FP8']);
    expect(reportLines(file, 'nothing')).toEqual([]);
  });

  it('the cards that do not fit, grouped by the GPUs tried in one server; packed cards without it: the most tried', () => {
    const by = (id: string) => cards.find((c) => c.gpuId === id) as CardLine;
    const at = (id: string, n?: number) => ({ ...by(id), status: 'no-fit' as const, ...(n !== undefined ? { serverGpus: n } : {}) });
    const { serverGpus: _, ...old } = at('h100-sxm5-80gb');
    expect(doNotFit([at('rtx-3090-24gb', 4), old, at('gh200-96gb', 2), at('rtx-4090-24gb', 4)])).toBe('even with 8: H100 SXM5 80GB; even with 4: RTX 3090, RTX 4090; even with 2: GH200 96GB');
  });

  it('a card in one line', () => {
    const by = (id: string) => cards.find((c) => c.gpuId === id) as CardLine;
    expect(cardSummary(undefined)).toBe('not evaluated');
    expect(cardSummary(by('rtx-5090-32gb'))).toMatch(/^fits with 1, \d+ users guaranteed, \d+ likely$/);
    expect(cardSummary(by('rtx-pro-6000-blackwell-server-96gb'))).toBe('fits with 1, speed unknown');
    expect(cardSummary({ ...by('h100-sxm5-80gb'), status: 'no-fit' })).toBe('does not fit even with 8');
    expect(cardSummary({ ...by('h100-sxm5-80gb'), status: 'no-start', note: 'heads' })).toBe('the engine does not start (heads)');
    expect(cardSummary({ ...by('h100-sxm5-80gb'), status: 'at-limit' })).toMatch(/^at the limit with 1/);
    expect(cardSummary({ ...by('h100-sxm5-80gb'), status: 'slow' })).toMatch(/^fits, too slow, with 1/);
  });
});

describe('the report as text', () => {
  it('a few lines: the answer, the chat, the RTX 5090 and the overview per card with its legend', () => {
    const text = renderReport(file, reportLines(file, 'qwen'), plain);
    expect(text.startsWith('Ready-made reports: vLLM 0.31.0, fp8 KV cache.')).toBe(true);
    expect(text).toContain('10% of users active at once');
    expect(text).toContain('\nQwen3-8B  (Qwen/Qwen3-8B)\n  bf16 · 32k (native maximum context)\n    Qwen3-8B at 32k tokens, fp8 KV cache: serves users from 1× RTX 3090');
    expect(text).toContain('    As a chat: 30 users guaranteed, 40 likely.');
    expect(text).not.toContain('RTX 5090:');
    expect(text).toMatch(/\n {4}Card +Verdict +GPUs/);
    expect(text).toContain(`    ${USERS_LEGEND.slice(0, 40)}`);
    expect(text).toMatch(/\n {4}\(1\) /);
    // The context that fits nowhere: the cards in one line, the 1M arguments.
    expect(text).toContain('does not fit in one server on any card of the catalog');
    // Grouped by the GPUs tried in one server (workstation and GeForce cards 4, GH200 2, the others 8).
    expect(text.replace(/\n {4}/g, ' ')).toMatch(/Do not fit even with 8: .*; even with 4: RTX 3090, RTX 4090, .*; even with 2: GH200 96GB, GH200 144GB\./);
    expect(text).toContain('    for 1M with vLLM: --max-model-len 1048576');
    expect(text).toContain('Versions without enough data: Kimi K3 NVFP4.');
    expect(text).not.toContain('Every card for one model');
    expect(Math.max(...text.split('\n').map((l) => l.length))).toBeLessThanOrEqual(120);
  });

  it('many lines: the answers only, with the hint; SGLang; ASCII; the default style; nothing missing', () => {
    const many = { ...file, engine: 'sglang', missing: [], models: Array.from({ length: OVERVIEW_MAX_LINES }, () => file.models[0] as ReportFile['models'][number]) };
    const text = renderReport(many, reportLines(many), { ...plain, ascii: true });
    expect(text.startsWith('Ready-made reports: SGLang 0.31.0')).toBe(true);
    expect(text).not.toContain('Card ');
    expect(text).toContain(`Every card for one model: add its name (e.g. ${__CLI_NAME__} report qwen3).`);
    expect(text).not.toContain('Versions without enough data');
    expect(text).not.toMatch(/[×–±]/);
    expect(renderReport(file, reportLines(file, 'glm'))).toContain('GLM 5  (zai-org/GLM-5)');
  });
});

describe('exports of the report', () => {
  it('CSV: one line per version × card × context, the same columns as the command', () => {
    const csv = reportCsv(file, reportLines(file), false);
    const lines = csv.trimEnd().split('\r\n');
    expect(lines[0]).toBe(CSV_HEADERS.join(','));
    expect(lines).toHaveLength(1 + 3 * GPUS.length);
    expect(lines[1]).toMatch(/^Qwen\/Qwen3-8B,Qwen\/Qwen3-8B,BF16,dense,vLLM,0\.31\.0,fp8,32768,native maximum context,16384,16384,500,20,10,RTX 3090,rtx-3090-24gb,consumer,no,[\d.]+,/);
    expect(lines.some((l) => l.includes(',official YaRN,') && l.includes(',does not fit,'))).toBe(true);
    const eu = reportCsv({ ...file, engine: 'sglang' }, reportLines(file, 'glm'), true);
    expect(eu.startsWith('﻿')).toBe(true);
    expect(eu.split('\r\n')[1]).toContain(';SGLang;0.31.0;');
    const noted = reportCsv(file, [{ ...(reportLines(file)[0] as ReturnType<typeof reportLines>[number]), context: { ...fits, cards: [{ ...(cards[0] as CardLine), note: 'the reason' }] } }], false);
    expect(noted).toContain('the reason');
  });

  it('Markdown: per version and context, the answer and the overview with its legend and notes', () => {
    const md = reportMarkdown(file, reportLines(file, 'qwen'));
    expect(md.startsWith('# Capacity reports (vLLM 0.31.0, fp8 KV cache)\n\n## Qwen3-8B · bf16 · 32k (native maximum context)\n\n**Qwen3-8B at 32k tokens')).toBe(true);
    expect(md).toContain('| Card | Verdict | GPUs | Memory per GPU (GiB) |');
    expect(md).toContain(`_${USERS_LEGEND}_`);
    expect(md).toMatch(/\n1\. /);
    expect(reportMarkdown({ ...file, engine: 'sglang' }, reportLines(file, 'glm'))).toContain('# Capacity reports (SGLang 0.31.0, fp8 KV cache)');
    const quiet = reportMarkdown(file, [{ ...(reportLines(file)[1] as ReturnType<typeof reportLines>[number]) }]);
    expect(quiet).not.toMatch(/\n1\. /);
  });

  it('JSON: the lines as they are', () => {
    expect(JSON.parse(reportJsonText(reportLines(file, 'glm')))).toEqual(JSON.parse(JSON.stringify(reportLines(file, 'glm'))));
  });
});

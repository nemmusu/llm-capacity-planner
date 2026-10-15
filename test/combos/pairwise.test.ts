/**
 * The command over all pairs of its options (test/helpers/pairwise.ts): engine, KV cache, card and count,
 * users, prompt cache, text-only, context, export, output style, --all-hw and --details, on four catalog
 * models of different kinds (small dense, MoE, MLA, multimodal), each crossed with every pair of the rest.
 * In-process on the built-in models: no network. Every case must answer (exit 0) with clean output: no
 * NaN / undefined / Infinity / null, a CSV with one column per header, JSON that parses with finite numbers,
 * the same headline in the text and the JSON, and ASCII only with --ascii.
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { CSV_HEADERS, asciiOnly } from '../../src/cli/render';
import { BAD_WORDS, catalogPack, italianHits, runCli, tempFolder, type CliRun } from '../helpers/catalog';
import { allPairs } from '../helpers/pairwise';

const MODELS = {
  'small dense': 'ibm-granite/granite-4.2-3b',
  MoE: 'openai/gpt-oss-20b',
  MLA: 'zai-org/GLM-5.3-Flash',
  multimodal: 'google/gemma-4-31B-it',
} as const;

const OPTIONS = {
  engine: ['vllm', 'sglang'],
  cache: ['fp8', 'auto'],
  gpu: ['none', 'h100/8', '5090/1', 'pro-6000/2', 'l40s/4'],
  users: ['none', '200/10', '1000/30'],
  promptCache: ['0', '80'],
  textOnly: [false, true],
  tokens: ['8k', '32k', '128k'],
  export: ['none', 'csv', 'json', 'md'],
  style: ['default', 'ascii', 'no-color'],
  allHw: [false, true],
  details: [false, true],
} as const;

type Options = { [K in keyof typeof OPTIONS]: (typeof OPTIONS)[K][number] };

/** The command line of a case. */
function argv(repo: string, o: Options): string[] {
  const a = [repo, '--engine', o.engine, '--cache', o.cache, '--tokens', o.tokens, '--prompt-cache', o.promptCache];
  if (o.gpu !== 'none') {
    const [gpu, count] = o.gpu.split('/') as [string, string];
    a.push('--gpu', gpu, '--count', count);
  }
  if (o.users !== 'none') {
    const [users, active] = o.users.split('/') as [string, string];
    a.push('--users', users, '--active', active);
  }
  if (o.textOnly) a.push('--text-only');
  if (o.export !== 'none') a.push(`--${o.export}`, `out.${o.export}`);
  if (o.style !== 'default') a.push(`--${o.style}`);
  if (o.allHw) a.push('--all-hw');
  if (o.details) a.push('--details');
  return a;
}

/** An ANSI colour code (ESC [ … m). */
const ANSI = new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*m`, 'g');
const plain = (s: string) => s.replace(ANSI, '');
const squash = (s: string) => s.replace(/\s+/g, ' ').trim();

/** RFC 4180 rows (quoted cells may hold the separator, quotes and line breaks). */
function parseCsv(text: string, sep = ','): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i] as string;
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === sep) {
      row.push(cell);
      cell = '';
    } else if (ch === '\r' && text[i + 1] === '\n') {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = '';
      i++;
    } else cell += ch;
  }
  if (cell !== '' || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

/** Every value of a JSON document with its path (to find a null or a non-finite number). */
function* leaves(v: unknown, at = '$'): Generator<[string, unknown]> {
  if (v !== null && typeof v === 'object') for (const [k, x] of Object.entries(v)) yield* leaves(x, `${at}.${k}`);
  else yield [at, v];
}

/** The headline as the text prints it: the lines after "Technical detail" and before the scenario, joined. */
const textHeadline = (text: string) => squash(((plain(text).split('\nTechnical detail\n')[1] as string).split('\nScenario ')[0] as string).replace(/\n/g, ' '));
/** The summary as the text prints it: the lines between "Summary" and the first blank line, joined. */
const textSummary = (text: string) => squash((plain(text).split('\n\n')[0] as string).replace(/^Summary\n/, '').replace(/\n/g, ' '));

const cache = tempFolder('combos-');
beforeAll(() => {
  catalogPack();
}, 120_000);
afterAll(() => cache.cleanup());

/** Italian found per case (a known bug, asserted at the end). */
const italian: { name: string; hits: string[] }[] = [];

describe.each(Object.entries(MODELS))('all pairs of options, %s model (%s)', (_kind, repo) => {
  const cases = allPairs(OPTIONS);

  it('covers every pair with a few dozen cases', () => {
    expect(cases.length).toBeGreaterThanOrEqual(20);
    expect(cases.length).toBeLessThanOrEqual(40);
  });

  it.each(cases.map((o) => [argv(repo, o).slice(1).join(' '), o] as const))('%s', async (_name, o) => {
    let r: CliRun | undefined;
    await expect(
      (async () => {
        r = await runCli(argv(repo, o), cache.dir, { isTTY: true });
      })(),
    ).resolves.toBeUndefined();
    const { code, text, written } = r as CliRun;
    expect(code, text).toBe(0);
    const shown = plain(text);
    expect(shown).not.toMatch(BAD_WORDS);
    expect(shown).not.toMatch(/\bError\b|\n\s+at /);
    // Colours on a terminal unless --no-color (--ascii keeps them: it only replaces the special characters).
    if (o.style === 'no-color') expect(text).not.toMatch(ANSI);
    else expect(text).toMatch(ANSI);

    const files = Object.keys(written);
    expect(files).toEqual(o.export === 'none' ? [] : [`out.${o.export}`]);
    const content = written[`out.${o.export}`] as string;
    if (o.export !== 'none') expect(shown).toContain(`written to out.${o.export}`);
    if (o.export === 'csv') {
      const rows = parseCsv(content);
      expect(rows[0]).toEqual(CSV_HEADERS);
      expect(rows.length).toBeGreaterThan(1);
      for (const row of rows) expect(row).toHaveLength(CSV_HEADERS.length);
      expect(content).not.toMatch(BAD_WORDS);
    }
    if (o.export === 'md') {
      expect(content).toMatch(/^# \S/);
      expect(content).not.toMatch(BAD_WORDS);
    }
    if (o.export === 'json') {
      const json = JSON.parse(content) as { headline: string; fit: unknown; users: unknown; rows: unknown[]; cards: unknown[] };
      for (const [at, v] of leaves(json)) {
        // A missing answer is null on purpose (nothing fits; no --users); nothing else may be.
        if (v === null) expect(['$.fit', '$.users']).toContain(at);
        if (typeof v === 'number') expect([at, Number.isFinite(v)]).toEqual([at, true]);
        if (typeof v === 'string') expect([at, v]).not.toEqual([at, expect.stringMatching(BAD_WORDS)]);
      }
      expect(json.rows.length).toBeGreaterThan(0);
      expect(json.cards.length).toBeGreaterThan(0);
      // The headline of the text is the JSON's, and so are its numbers.
      const head = squash(o.style === 'ascii' ? asciiOnly(json.headline) : json.headline);
      expect(textHeadline(text)).toBe(head);
      // The summary of the text is the JSON's.
      const summary = (json as unknown as { summary: string[] }).summary.map((l) => (o.style === 'ascii' ? asciiOnly(l) : l));
      expect(textSummary(text)).toBe(squash(summary.join(' ')));
      const fig = json as unknown as {
        fit: { figures: { users?: { guaranteed: number; likely: number } } } | null;
        users: { choice?: { figures: { users: { guaranteed: number; likely: number } } } } | null;
      };
      const u = fig.users?.choice?.figures.users ?? fig.fit?.figures.users;
      const m = /([\d,]+) users guaranteed, ([\d,]+) likely/.exec(textHeadline(text));
      if (m && u) expect([Number((m[1] as string).replace(/,/g, '')), Number((m[2] as string).replace(/,/g, ''))]).toEqual([u.guaranteed, u.likely]);
    }
    // --ascii: plain ASCII everywhere (old Windows consoles).
    if (o.style === 'ascii') {
      const odd = [...shown].filter((ch) => ch.charCodeAt(0) > 126);
      if (italianHits(shown).length === 0) expect(odd).toEqual([]);
    }
    const hits = italianHits([shown, o.export === 'json' ? '' : (content ?? '')].join('\n'));
    if (hits.length > 0) italian.push({ name: `${repo} ${argv(repo, o).slice(1).join(' ')}`, hits });
  });
});

// The planner's messages (layout checks, speed caveats, memory reasons) reach the output: all English.
it('no Italian words in any output', () => {
  expect(italian.slice(0, 5)).toEqual([]);
});

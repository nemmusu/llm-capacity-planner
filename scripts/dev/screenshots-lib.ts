// The examples of the README: images of the real command (docs/images/*.svg) and the README blocks around them
// (between <!-- generated:NAME --> markers), all from the command's output on the built-in models, offline.
// Shared by scripts/dev/screenshots-entry.ts (writes them) and test/docs/screenshots.test.ts (fails when they no
// longer match the command).
import type { BuiltinModels } from '../../src/cli/builtin';
import { runPlan } from '../../src/cli/run';
import { FITTED, PIPELINE_WIDENING, PUBLIC_MAX_INPUT, PUBLIC_POINTS, SGLANG_CALIBRATION, VLLM_CALIBRATION } from '../../src/planner/calibration';
import { DEFAULT_WORKLOAD } from '../../src/planner/capacity';
import { terminalSvg } from './ansi-svg';

/** Columns of the pictured terminal: wide enough for the tables, narrow enough for a README. */
export const COLUMNS = 100;
/** Longest image, in output lines. */
const MAX_LINES = 40;

export interface Screenshot {
  /** File in docs/images. */
  file: string;
  /** Scenario title in the README. */
  title: string;
  /** What the scenario assumes, in words (README). */
  scenario?: string;
  argv: string[];
  /** The part of the output shown: from the first line matching `from` (default: the top) to the line before `until`. */
  from?: RegExp;
  until?: RegExp;
  /** Rows hidden behind a dim "…": from the first line matching `from` to the line before `to`. */
  elide?: { from: RegExp; to: RegExp }[];
  /** Terminal width for this one (default COLUMNS). */
  columns?: number;
}

/** The README's first image: the strongest scenario. */
export const MAIN: Screenshot = {
  file: 'long-documents.svg',
  title: 'Long documents (128k tokens), 200 users',
  argv: ['zai-org/GLM-5.3', '--tokens', '128k', '--users', '200'],
  until: /^ {2}guaranteed = /,
  elide: [
    { from: /^GLM-5.3 at /, to: /^Per card:/ },
    { from: /^ {2}RTX 3090 /, to: /^ {2}GH200 144GB / },
  ],
};

/** The scenarios of the "Examples" section. */
export const EXAMPLES: Screenshot[] = [
  {
    file: 'coding-agents.svg',
    title: 'Coding with agents, 300 users at 30% active',
    scenario:
      'Each request carries up to 128k tokens of code and context, 64k on average, all of it read again: no prefix cache, the cautious case (agents that resend the same context gain from `--prompt-cache`). 30% of the users are active at the same moment, since agents keep working on their own.',
    argv: ['moonshotai/Kimi-K2.7-Code', '--tokens', '128k', '--users', '300', '--active', '30'],
    until: /^Technical detail/,
  },
  {
    file: 'knowledge-chat.svg',
    title: 'Chat, 2,000 users, 80% prefix cache',
    scenario: 'Conversations of up to 32k tokens; 80% of each one is already in the prefix cache (`--prompt-cache 80`), as in a chat that grows turn by turn. 10% of the users are active at the same moment.',
    argv: ['openai/gpt-oss-120b', '--tokens', '32k', '--users', '2000', '--prompt-cache', '80'],
    until: /^Technical detail/,
  },
  {
    file: 'h200-details.svg',
    title: 'One configuration in detail: 8× H200',
    scenario: 'A fixed configuration (`--gpu h200 --count 8`): documents of up to 64k tokens read without a prefix cache, and the memory item by item (`--details`).',
    argv: ['deepseek-ai/DeepSeek-V4-Pro-0813', '--tokens', '64k', '--gpu', 'h200', '--count', '8', '--details'],
    until: /^ {2}engine /,
    elide: [{ from: /^DeepSeek-V4-Pro-0813 at /, to: /^Details / }],
  },
];

export const HELP: Screenshot = { file: 'help.svg', title: 'The start of --help', argv: ['--help'] };

export const SCREENSHOTS: Screenshot[] = [MAIN, ...EXAMPLES, HELP];

/** The same documents at longer contexts (the README's table). */
export const CONTEXTS = { repo: 'zai-org/GLM-5.3', users: 200, tokens: ['128k', '268k', '1M'] };

const ANSI = new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*m`, 'g');
/** A line without its colour codes. */
const plain = (s: string) => s.replace(ANSI, '');
const HUB_FETCH = (() => Promise.reject(new Error('offline'))) as typeof fetch;

/** The command's output for one run (no network, no GPU of this computer) and the files it wrote. */
export async function run(argv: readonly string[], builtin: BuiltinModels, baseUrl: string, opts: { columns?: number; color?: boolean } = {}): Promise<{ text: string; files: Record<string, string> }> {
  const out: string[] = [];
  const files: Record<string, string> = {};
  await runPlan(argv, {
    fetch: HUB_FETCH,
    env: {},
    localGpu: () => Promise.resolve(undefined),
    out: (t) => out.push(t),
    writeFile: (f, c) => {
      files[f] = c;
      return Promise.resolve();
    },
    baseUrl,
    cacheDir: '.',
    isTTY: opts.color ?? true,
    columns: opts.columns ?? COLUMNS,
    platform: 'linux',
    builtin,
  });
  return { text: out.join('\n'), files };
}

/** The lines of a screenshot, cut and elided, with whether anything was left out after them. */
function cropped(s: Screenshot, text: string): { lines: string[]; cut: boolean } {
  const all = text.split('\n');
  const start = s.from ? Math.max(0, all.findIndex((l) => s.from?.test(plain(l)))) : 0;
  const rest = all.slice(start);
  const stop = s.until ? rest.findIndex((l, i) => i > 0 && s.until?.test(plain(l))) : -1;
  let shown = stop > 0 ? rest.slice(0, stop) : rest;
  for (const e of s.elide ?? []) {
    const a = shown.findIndex((x) => e.from.test(plain(x)));
    const b = shown.findIndex((x, i) => i > a && e.to.test(plain(x)));
    if (a > 0 && b > a) shown = [...shown.slice(0, a), '\u001b[2m  …\u001b[0m', ...shown.slice(b)];
  }
  shown = shown.slice(0, MAX_LINES);
  while (shown.length && plain(shown[shown.length - 1] as string).trim() === '') shown.pop();
  return { lines: shown, cut: stop > 0 || shown.length === MAX_LINES || start > 0 };
}

const command = (s: Screenshot) => `${__CLI_NAME__} ${s.argv.join(' ')}`;
/** The alt text of a README image: what it shows and the command it is the output of. */
const alt = (s: Screenshot) => `${s.title}: terminal output of ${command(s)}`;

/** The SVG of one screenshot. */
export async function screenshotSvg(s: Screenshot, builtin: BuiltinModels, baseUrl: string): Promise<string> {
  const columns = s.columns ?? COLUMNS;
  const { lines, cut } = cropped(s, (await run(s.argv, builtin, baseUrl, { columns })).text);
  return terminalSvg({
    command: command(s),
    output: lines.join('\n'),
    title: __CLI_NAME__,
    // Two cells of slack, as a terminal a little wider than the width the command was given.
    columns: columns + 2,
    cut,
  });
}

/** The first answer of a run: its Summary's first line, without the "Recommendation:" label. */
async function headline(s: Screenshot, builtin: BuiltinModels, baseUrl: string): Promise<string> {
  const { files } = await run([...s.argv, '--json', 'a.json'], builtin, baseUrl, { color: false });
  const summary = (JSON.parse(files['a.json'] as string) as { summary: string[] }).summary;
  return (summary[0] as string).replace(/^Recommendation: /, '');
}

/** The README's first image, the sentence under it and the same output as text. */
export async function readmeTop(builtin: BuiltinModels, baseUrl: string): Promise<string> {
  const text = cropped(MAIN, (await run(MAIN.argv, builtin, baseUrl, { color: false })).text).lines.map(plain).join('\n');
  return [
    `![${alt(MAIN)}](docs/images/${MAIN.file})`,
    '',
    `*${MAIN.title}: the summary comes first, the technical detail after it.*`,
    '',
    '<details>',
    '<summary>The same output as text</summary>',
    '',
    '```text',
    `$ ${command(MAIN)}`,
    text,
    '```',
    '',
    '</details>',
  ].join('\n');
}

/** The precision labels of the command, as its legend words them. */
const KIND: Record<string, string> = { measured: 'measured', calibrated: 'calibrated', estimated: 'estimated', imprecise: 'not precise enough to decide' };

const num = (n: number, digits = 1) => n.toLocaleString('en-US', { maximumFractionDigits: digits });

/** The README's table of the same documents at 128k, 268k and 1M tokens, and the contexts where nobody is guaranteed the speed. */
async function contextTable(builtin: BuiltinModels, baseUrl: string): Promise<{ table: string; unserved: string[] }> {
  type Figures = { hardware: string; verdict: string; users?: { guaranteed: number; likely: number }; read: { value: number; min: number; max: number }; tokS?: { value: number; precision: { kind: string; errorPct: number } } };
  type Answer = { summary: string[]; rows: Figures[]; fit?: { figures: Figures }; users?: { choice?: { servers: number; figures: Figures } } };
  const rows: string[][] = [];
  const unserved: string[] = [];
  for (const t of CONTEXTS.tokens) {
    const { files } = await run([CONTEXTS.repo, '--tokens', t, '--users', String(CONTEXTS.users), '--json', 'a.json'], builtin, baseUrl, { color: false });
    const a = JSON.parse(files['a.json'] as string) as Answer;
    const choice = a.users?.choice;
    // Nobody served: the layout the Summary calls closest (the most likely users, then the fastest under load).
    const closest = a.rows
      .filter((r) => r.verdict !== 'no-fit' && r.verdict !== 'no-start' && r.users && r.tokS)
      .sort((x, y) => (y.users?.likely ?? 0) - (x.users?.likely ?? 0) || (y.tokS?.value ?? 0) - (x.tokS?.value ?? 0))[0];
    if (!choice && closest) unserved.push(t);
    const f = choice?.figures ?? closest;
    const sec = (x: number) => x.toFixed(1);
    const read = f ? `${sec(f.read.value)} s (${sec(f.read.min)}–${sec(f.read.max)})` : '—';
    const label = f?.tokS ? `${KIND[f.tokS.precision.kind] ?? f.tokS.precision.kind}, ±${Math.round(f.tokS.precision.errorPct)} %` : '—';
    const users = (x: Figures) => `${num(x.users?.guaranteed ?? 0, 0)} / ${num(x.users?.likely ?? 0, 0)}`;
    rows.push(
      choice
        ? [t, choice.figures.hardware, users(choice.figures), String(choice.servers), read, label]
        : [t, closest ? `${closest.hardware}: fits, nobody guaranteed ${DEFAULT_WORKLOAD.minTokS} tok/s of writing` : 'does not fit', closest ? users(closest) : '—', '—', read, label],
    );
  }
  const table = [
    `| Context | One server | Users per server (guaranteed / likely) | Servers for ${CONTEXTS.users} | Reading the full context alone | Writing speed estimate |`,
    '| --- | --- | --- | --- | --- | --- |',
    ...rows.map((r) => `| ${r.join(' | ')} |`),
  ].join('\n');
  return { table, unserved };
}

/** The README's "Examples" section body. */
export async function readmeExamples(builtin: BuiltinModels, baseUrl: string): Promise<string> {
  const { table, unserved } = await contextTable(builtin, baseUrl);
  const slow = unserved.length
    ? `; at ${unserved.join(' and ')} the model fits, but no server guarantees anyone a writing speed of ${DEFAULT_WORKLOAD.minTokS} tokens/s for the answer (the low end of the estimate is below it; the likely users are in the table). Reading is a separate figure: the time to read a full context alone, in its own column`
    : '';
  const out: string[] = [
    `### The same documents, longer contexts`,
    '',
    `\`${__CLI_NAME__} ${CONTEXTS.repo} --tokens <context> --users ${CONTEXTS.users}\`: the longer each document, the fewer users one server serves; beyond 128k tokens the speed is extrapolated beyond the context verified${slow}.`,
    '',
    table,
  ];
  for (const s of EXAMPLES) {
    out.push('', `### ${s.title}`, '', `\`${command(s)}\`${s.scenario ? `: ${s.scenario}` : ''}`, '', `${await headline(s, builtin, baseUrl)}`, '', `![${alt(s)}](docs/images/${s.file})`);
  }
  return out.join('\n');
}

/** The README with its generated blocks replaced; throws when a marker is missing. */
export function withBlocks(readme: string, blocks: Record<string, string>): string {
  let out = readme;
  for (const [name, body] of Object.entries(blocks)) {
    const re = new RegExp(`(<!-- generated:${name} -->)[\\s\\S]*?(<!-- /generated:${name} -->)`);
    if (!re.test(out)) throw new Error(`README.md has no <!-- generated:${name} --> block`);
    out = out.replace(re, (_m, open: string, close: string) => `${open}\n${body}\n${close}`);
  }
  return out;
}

const FAMILY: Record<string, string> = { dense: 'dense', moe: 'MoE', mla: 'MLA', hybrid: 'hybrid' };
/** "moe mxfp4" → "MoE MXFP4" (the combinations of FITTED). */
const combos = (list: string[]) => list.map((c) => c.replace(/^(\w+) (\w+)$/, (_m, family: string, w: string) => `${FAMILY[family] ?? family} ${w.toUpperCase()}`)).join(', ');
const pct = (x: number) => `±${Math.round(x * 100)} %`;

/** The README's accuracy table: the declared errors and fitted combinations of the code, the memory ranges of the examples. */
export async function readmeAccuracy(builtin: BuiltinModels, baseUrl: string): Promise<string> {
  // The memory range of the configuration each README scenario recommends (with --users, the one that serves them).
  const memory: number[] = [];
  for (const s of [MAIN, ...EXAMPLES]) {
    const { files } = await run([...s.argv, '--json', 'a.json'], builtin, baseUrl, { color: false });
    type Memory = { figures: { memory: { precision: { errorPct: number } } } };
    const a = JSON.parse(files['a.json'] as string) as { fit?: Memory; users?: { choice?: Memory } };
    const pick = a.users?.choice ?? a.fit;
    if (pick) memory.push(pick.figures.memory.precision.errorPct);
  }
  const v = VLLM_CALIBRATION.spread as { decode: number; prefill: number };
  const g = SGLANG_CALIBRATION.spread as { decode: number; prefill: number };
  const k = (n: number) => `${Math.round(n / 1024)}K`;
  return [
    '| What is estimated | How precise | Basis |',
    '| --- | --- | --- |',
    '| Memory: weights and KV cache | exact | the loaded tensors and the engine\'s own cache rules |',
    `| Memory per GPU in total (fixed engine costs are a range) | ${pct(Math.min(...memory) / 100)} to ${pct(Math.max(...memory) / 100)} in the examples below | shown with every figure |`,
    `| Speed on vLLM: ${combos(FITTED.vllm)} | ${pct(v.decode)} generation, ${pct(v.prefill)} prompt reading | fitted on ${PUBLIC_POINTS.vllm} public benchmarks up to ${k(PUBLIC_MAX_INPUT.vllm)} tokens |`,
    `| Speed on SGLang: ${combos(FITTED.sglang)} | ${pct(g.decode)} | fitted on ${PUBLIC_POINTS.sglang} public benchmarks up to ${k(PUBLIC_MAX_INPUT.sglang)} tokens |`,
    `| Speed of any other model type or format | the range doubled (${pct(v.decode * 2)} / ${pct(v.prefill * 2)} on vLLM) | labelled "estimated" |`,
    '| Contexts longer than the benchmarks | the range widened by half again | labelled next to the number |',
    `| Layouts split in pipeline stages (no run measured them) | the range widened by half again (×${PIPELINE_WIDENING}) | labelled "pipeline speed not measured" |`,
    '| A range of ±50 % or more | not precise enough to decide | labelled so |',
  ].join('\n');
}

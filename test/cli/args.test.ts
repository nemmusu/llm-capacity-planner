import { describe, expect, it } from 'vitest';
import { BOOLEAN, checkContext, COMMAND, countArg, parseArgs, tokensArg, VALUED, USAGE, type ArgsEnv } from '../../src/cli/args';
import { matchGpus } from '../../src/cli/run';
import { GPUS } from './fixtures';

const ctx: ArgsEnv = { env: { WT_SESSION: '1' }, isTTY: false, platform: 'linux' };
const parse = (argv: string[], c: Partial<ArgsEnv> = {}) => parseArgs(argv, { ...ctx, ...c });

describe('countArg and checkContext', () => {
  it('people count in thousands: 2k = 2,000 (tokens count in 1,024s)', () => {
    expect(countArg('--users', '200')).toBe(200);
    expect(countArg('--users', '2k')).toBe(2000);
    expect(countArg('--users', '1.5M')).toBe(1_500_000);
    expect(() => countArg('--users', 'many')).toThrow('--users: expected a number (e.g. 200 or 2k), not "many"');
    expect(() => countArg('--users', '0')).toThrow('--users: must be at least 1');
    expect(parse(['a/b', '--users', '2k'])?.users).toBe(2000);
    // Whole people only (1.1k is 1,100 exactly).
    expect(countArg('--users', '1.1k')).toBe(1100);
    expect(() => countArg('--users', '10.5')).toThrow('--users: a whole number of people, not "10.5"');
    expect(() => countArg('--users', '1.0005k')).toThrow('a whole number of people');
  });

  it('the context once: after the model or with --tokens, the same if both', () => {
    expect(parse(['a/b', '32k', '--tokens', '32k'])?.tokens).toBe(32_768);
    expect(() => parse(['a/b', '32k', '--tokens', '64k'])).toThrow('the context is given twice: "32k" after the model and --tokens 64k; give one');
    expect(parse(['report', 'glm', '--tokens', '8k'])?.reportFilter).toBe('glm');
  });

  it('the answer shorter than the context, the typical request within it', () => {
    expect(() => checkContext(32_768, { outputTokens: 500 })).not.toThrow();
    expect(() => checkContext(32_768, { outputTokens: 500, typicalTokens: 32_768 })).not.toThrow();
    expect(() => checkContext(8192, { outputTokens: 8192 })).toThrow('--output-tokens (8,192) must be less than the context of 8,192 tokens: the answer is part of the request');
    expect(() => checkContext(8192, { outputTokens: 500, typicalTokens: 16_384 })).toThrow('--typical-context (16,384) must be at most the context of 8,192 tokens');
    expect(() => parse(['a/b', '--tokens', '8k', '--output-tokens', '16k'])).toThrow('--output-tokens (16,384) must be less than');
    expect(() => parse(['a/b', '--tokens', '8k', '--typical-context', '16k'])).toThrow('--typical-context (16,384) must be at most');
  });
});

describe('tokensArg', () => {
  it('reads plain numbers, k and M, a decimal point', () => {
    expect(tokensArg('--tokens', '8192')).toBe(8192);
    expect(tokensArg('--tokens', '32k')).toBe(32_768);
    expect(tokensArg('--tokens', '32K')).toBe(32_768);
    expect(tokensArg('--tokens', '1M')).toBe(1_048_576);
    expect(tokensArg('--tokens', '1.5k')).toBe(1536);
    expect(tokensArg('--tokens', ' 4k ')).toBe(4096);
  });

  it('says what is wrong', () => {
    expect(() => tokensArg('--tokens', 'tanti')).toThrow('--tokens: expected a number (e.g. 32k), not "tanti"');
    expect(() => tokensArg('--tokens', '1,5k')).toThrow('expected a number');
    expect(() => tokensArg('--tokens', '0')).toThrow('--tokens: must be at least 1');
    expect(() => tokensArg('--tokens', '0.1')).toThrow('must be at least 1');
  });
});

describe('the help', () => {
  const sections = ['Examples (free models', 'The three main options:', 'Sizing for a team', 'Hardware:', 'Engine:', 'Output:', 'How to read the results:', 'How to read the accuracy', 'Glossary:'];
  const at = sections.map((s) => USAGE.indexOf(s));
  const part = (i: number) => USAGE.slice(at[i], at[i + 1]);

  it('says what it does and that nothing is downloaded or run, then every section in order', () => {
    expect(USAGE.split('\n')[1]).toContain('Nothing is downloaded or run: only config.json and safetensors headers are read.');
    expect(at.every((i) => i > 0)).toBe(true);
    expect([...at].sort((a, b) => a - b)).toEqual(at);
  });

  it('examples to copy: free models only (checked on huggingface.co), every one valid for the parser', () => {
    // Checked on huggingface.co on 2026-10-15: none is gated (no license to accept, no token needed).
    // Large models of the built-in data, all free.
    const free = ['moonshotai/Kimi-K2.7-Code', 'openai/gpt-oss-120b', 'zai-org/GLM-5.3', 'deepseek-ai/DeepSeek-V4-Pro-0813'];
    const examples = [...USAGE.matchAll(new RegExp(`^ {2}${COMMAND.replace(/[-]/g, '\\-')} (.+)$`, 'gm'))].map((x) => (x[1] as string).trim().split(/\s+/));
    expect(examples.length).toBe(7);
    expect(new Set(examples.map((e) => e[0]).filter((r) => r !== 'report'))).toEqual(new Set(free));
    for (const e of examples) expect(parse(e), e.join(' ')).toBeDefined();
    const text = examples.map((e) => e.join(' ')).join('\n');
    for (const want of ['--users 200', '--users 300 --active 30', '--prompt-cache 80', '--gpu h200 --count 8', '--engine sglang', '--csv ', 'report ']) expect(text).toContain(want);
    for (const scenario of ['coding with agents, 300 users at 30% active', 'chat, 2,000 users, 80% of each request already in the prefix cache', 'long documents (128k tokens ≈ 200 pages), 200 users', 'one configuration in detail'])
      expect(USAGE).toContain(scenario);
    expect(USAGE).not.toMatch(/Qwen3-8B|Qwen3-32B/);
  });

  it('every option accepted is explained, one per line with its default', () => {
    for (const o of [...VALUED, ...BOOLEAN]) expect(USAGE, o).toMatch(new RegExp(`\\n {2}--${o}[ |,]`));
    for (const o of ['model ', '--tokens N', '--cache fp8|auto']) expect(part(1)).toContain(o);
    for (const o of ['--users', '--active', '--min-tok-s', '--max-wait', '--prompt-cache', '--typical-context', '--output-tokens']) expect(part(2)).toContain(o);
    expect(part(2)).toContain('(10)');
    expect(part(2)).toContain('(20)');
    expect(part(2)).toContain('(500)');
    expect(part(2)).toContain('one person running 5 subagents counts as 5');
    expect(part(2)).toContain('Users include AI agents.');
  });

  it('a card by the name the tables show', () => {
    expect(matchGpus(GPUS, 'RTX PRO 6000 Server').map((g) => g.id)).toEqual(['rtx-pro-6000-blackwell-server-96gb']);
    expect(matchGpus(GPUS, 'pro-6000-server').map((g) => g.id)).toEqual(['rtx-pro-6000-blackwell-server-96gb']);
  });

  it('a whole name first: "l4" is the L4, not the L40S; "h200" the H200s, not the GH200', () => {
    expect(matchGpus(GPUS, 'l4').map((g) => g.id)).toEqual(['l4-24gb']);
    expect(matchGpus(GPUS, 'l40').map((g) => g.id)).toEqual(['l40s-48gb']);
    expect(matchGpus(GPUS, 'h200').map((g) => g.id)).toEqual(['h200-sxm-141gb', 'h200-nvl-141gb']);
    expect(matchGpus(GPUS, 'h100').length).toBe(3);
    expect(matchGpus(GPUS, 'h100-sxm').map((g) => g.id)).toEqual(['h100-sxm5-80gb']);
  });

  it('the card names it lists are in the catalog', () => {
    const names = (/--gpu NAME +only the cards whose name contains NAME, in detail \([^)]*\): ([^(]+)\(/s.exec(USAGE)?.[1] as string).split(/[,\s]+/).filter(Boolean);
    expect(names.length).toBeGreaterThan(10);
    for (const n of names) expect(matchGpus(GPUS, n).length, n).toBeGreaterThan(0);
  });

  it('how to read the results and the accuracy, a glossary; 100 columns at most', () => {
    for (const k of ['fits', 'at the limit', 'does not fit', 'engine does not start', 'users (guaranteed)', 'users (likely)', 'concurrent requests', 'documents / chat']) expect(part(6)).toMatch(new RegExp(`\\n {2}${k.replace(/[()]/g, '\\$&')} `));
    for (const k of ['exact', 'measured', 'calibrated', 'estimated', 'not precise enough to decide']) expect(part(7)).toMatch(new RegExp(`\\n {2}${k} `));
    expect(part(7)).toContain('85%');
    expect(part(7)).toContain('docs/METHODOLOGY.md');
    for (const k of ['token', 'server', 'KV cache', 'TTFT', 'tok/s']) expect(USAGE.slice(at[8])).toMatch(new RegExp(`\\n {2}${k} `));
    for (const k of ['tensor parallel, TP', 'pipeline parallel, PP', '1 model on 4', '2 copies × 4', '1,000 tokens ≈ 750 English words']) expect(USAGE.slice(at[8])).toContain(k);
    expect(Math.max(...USAGE.split('\n').map((l) => l.length))).toBeLessThanOrEqual(100);
  });
});

describe('parseArgs', () => {
  it('--help, help, -h and no model: undefined (the help is printed)', () => {
    expect(parse([])).toBeUndefined();
    expect(parse(['--help'])).toBeUndefined();
    expect(parse(['a/b', '-h'])).toBeUndefined();
    expect(parse(['help'])).toBeUndefined();
    expect(parse(['--offline'])).toBeUndefined();
  });

  it('the defaults', () => {
    expect(parse(['Qwen/Qwen3-8B'])).toEqual({
      repo: 'Qwen/Qwen3-8B',
      report: false,
      cache: 'fp8',
      engine: 'vllm',
      activeShare: 0.1,
      minTokS: 20,
      maxTtftS: 10,
      ttftSet: false,
      outputTokens: 500,
      cachedShare: 0,
      textOnly: false,
      variant: '',
      revision: 'main',
      allHardware: false,
      details: false,
      thisPc: false,
      csvExcelEu: false,
      offline: false,
      refresh: false,
      color: false,
      ascii: false,
    });
  });

  it('every option, with a space or with =', () => {
    expect(
      parse([
        'a/b',
        '--tokens=64k',
        '--cache',
        'auto',
        '--engine',
        'sglang',
        '--users',
        '2k',
        '--active',
        '20%',
        '--min-tok-s',
        '30',
        '--max-wait',
        '5s',
        '--typical-context',
        '8k',
        '--output-tokens',
        '1k',
        '--prompt-cache',
        '80',
        '--gpu',
        'H100',
        '--count',
        '4',
        '--vram',
        '24',
        '--util',
        '0.85',
        '--variant',
        '/FP8/',
        '--revision',
        'v2',
        '--max-batched-tokens',
        '4k',
        '--max-seqs',
        '64',
        '--text-only',
        '--all-hw',
        '--this-pc',
        '--details',
        '--csv',
        'a.csv',
        '--csv-excel-eu',
        '--json',
        'a.json',
        '--md',
        'a.md',
        '--refresh',
        '--no-color',
        '--ascii',
      ]),
    ).toEqual({
      repo: 'a/b',
      report: false,
      tokens: 65_536,
      cache: 'auto',
      engine: 'sglang',
      users: 2000,
      activeShare: 0.2,
      minTokS: 30,
      maxTtftS: 5,
      ttftSet: true,
      typicalTokens: 8192,
      outputTokens: 1024,
      cachedShare: 0.8,
      gpu: 'h100',
      count: 4,
      vramGB: 24,
      util: 0.85,
      variant: 'FP8',
      revision: 'v2',
      maxBatchedTokens: 4096,
      maxSeqs: 64,
      textOnly: true,
      allHardware: true,
      details: true,
      thisPc: true,
      csv: 'a.csv',
      csvExcelEu: true,
      json: 'a.json',
      md: 'a.md',
      offline: false,
      refresh: true,
      color: false,
      ascii: true,
    });
  });

  it('the tokens after the model; bf16 and fp16 are refused (auto is the full cache); --gpu all; --offline', () => {
    expect(parse(['a/b', '32k'])?.tokens).toBe(32_768);
    expect(() => parse(['a/b', '--cache', 'bf16'])).toThrow("--cache: fp8 or auto (auto = the model's own precision, e.g. bf16), not \"bf16\"");
    expect(() => parse(['a/b', '--cache', 'fp16'])).toThrow('--cache: fp8 or auto');
    // --csv-excel-eu changes the --csv file: without one, an error.
    expect(() => parse(['a/b', '--csv-excel-eu'])).toThrow('--csv-excel-eu changes the --csv file: give --csv FILE too');
    // An option given twice: an error, not the last value.
    expect(() => parse(['a/b', '--tokens', '32k', '--tokens', '64k'])).toThrow('--tokens is given twice: give it once');
    expect(parse(['a/b', '--gpu', 'all'])).toMatchObject({ allHardware: true });
    expect(parse(['a/b', '--gpu', 'all'])?.gpu).toBeUndefined();
    expect(parse(['a/b', '--offline'])?.offline).toBe(true);
  });

  it('report and its filter (no tokens)', () => {
    expect(parse(['report'])).toMatchObject({ repo: 'report', report: true });
    expect(parse(['report', 'glm'])).toMatchObject({ report: true, reportFilter: 'glm' });
    expect(parse(['report', 'glm'])?.tokens).toBeUndefined();
  });

  it('colours only on a terminal without --no-color or NO_COLOR', () => {
    expect(parse(['a/b'], { isTTY: true })?.color).toBe(true);
    expect(parse(['a/b', '--no-color'], { isTTY: true })?.color).toBe(false);
    expect(parse(['a/b'], { isTTY: true, env: { NO_COLOR: '1', WT_SESSION: '1' } })?.color).toBe(false);
  });

  it('ASCII on the old Windows console (no Windows Terminal, no other terminal program)', () => {
    expect(parse(['a/b'], { platform: 'win32', env: {} })?.ascii).toBe(true);
    expect(parse(['a/b'], { platform: 'win32', env: { WT_SESSION: 'x' } })?.ascii).toBe(false);
    expect(parse(['a/b'], { platform: 'win32', env: { TERM_PROGRAM: 'vscode' } })?.ascii).toBe(false);
    expect(parse(['a/b'], { platform: 'darwin', env: {} })?.ascii).toBe(false);
  });

  it('says what is wrong', () => {
    expect(() => parse(['a/b', '--nope'])).toThrow('unknown option --nope (see --help)');
    expect(() => parse(['a/b', '--tokens'])).toThrow('--tokens: the value is missing');
    expect(() => parse(['a/b', 'c', 'd'])).toThrow('give one model');
    expect(() => parse(['a/b', '--cache', 'int4'])).toThrow('--cache: fp8 or auto (auto = the model\'s own precision, e.g. bf16), not "int4"');
    expect(() => parse(['a/b', '--engine', 'tgi'])).toThrow('--engine: vllm or sglang, not "tgi"');
    expect(() => parse(['a/b', '--active', '0'])).toThrow('--active: expected a number from 0.1 to 100, not "0"');
    expect(() => parse(['a/b', '--count', '3'])).toThrow('--count: 1, 2, 4, 8 or 16 GPUs');
    expect(() => parse(['a/b', '--util', '1'])).toThrow('--util');
    expect(() => parse(['a/b', '--offline', '--refresh'])).toThrow('--offline and --refresh exclude each other');
  });
});

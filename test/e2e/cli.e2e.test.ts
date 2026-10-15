/**
 * The real command end to end: the bundle `node scripts/build.mjs` makes (out/<name>.mjs, with the model pack
 * next to it) run as a child process, as a user runs it. The network is unreachable (HF_ENDPOINT points at a
 * closed port), the cache, the home folder and the working folder are temporary, and nvidia-smi is not on the
 * PATH (no GPU on "this PC"): the same answers on every machine. Every run is checked for its exit code, no stack
 * trace, no NaN / undefined / Infinity, and the files it was asked to write.
 */
import { execFile, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { sourceHash } from '../../scripts/source-hash.mjs';
import { BOOLEAN, USAGE, VALUED } from '../../src/cli/args';
import { tempFolder } from '../helpers/catalog';

const ROOT = process.cwd();
const BUNDLE = path.join(ROOT, 'out', `${__CLI_NAME__}.mjs`);
const PACK = path.join(ROOT, 'out', `${__CLI_NAME__}-models.pack`);
/** A small catalog model in the pack (3B, dense): every option is tried on it. */
const SMALL = 'ibm-granite/granite-4.2-3b';

const home = tempFolder('e2e-');
const work = path.join(home.dir, 'work');

beforeAll(() => {
  // The bundle and the pack (repacked only when data/models changed): built once for this file.
  const r = spawnSync(process.execPath, [path.join(ROOT, 'scripts', 'build.mjs')], { cwd: ROOT, encoding: 'utf8' });
  expect(r.status, `${r.stdout}\n${r.stderr}`).toBe(0);
  expect(existsSync(BUNDLE) && existsSync(PACK)).toBe(true);
  mkdirSync(work, { recursive: true });
}, 180_000);
afterAll(() => home.cleanup());

interface Run {
  code: number;
  stdout: string;
  stderr: string;
}

/** The command with `args`, offline, in the temporary working folder (where it writes its files). */
function cli(args: string[], extraEnv: Record<string, string> = {}): Promise<Run> {
  const env: Record<string, string> = {
    // Only node's folder: no nvidia-smi, so no local GPU line.
    PATH: path.dirname(process.execPath),
    ...(process.env.SystemRoot ? { SystemRoot: process.env.SystemRoot } : {}),
    HF_ENDPOINT: 'http://127.0.0.1:9',
    CAPACITY_PLANNER_CACHE: path.join(home.dir, 'cache'),
    // No Hugging Face login of this machine is read.
    HOME: home.dir,
    USERPROFILE: home.dir,
    HF_HOME: path.join(home.dir, 'hf'),
    LOCALAPPDATA: home.dir,
    COLUMNS: '120',
    // A modern terminal: no ASCII fallback for the legacy Windows console.
    WT_SESSION: '1',
    ...extraEnv,
  };
  return new Promise((resolve) => {
    execFile(process.execPath, [BUNDLE, ...args], { cwd: work, env, timeout: 60_000, maxBuffer: 64 * 1024 * 1024, windowsHide: true }, (err, stdout, stderr) => {
      const code = err ? (typeof (err as { code?: unknown }).code === 'number' ? ((err as { code: number }).code) : 1) : 0;
      resolve({ code, stdout: String(stdout), stderr: String(stderr) });
    });
  });
}

/** What every run must hold, whatever its exit code. */
function clean(r: Run): void {
  const all = `${r.stdout}\n${r.stderr}`;
  // No stack trace: errors are one readable line. A frame is "at name (file:line:col)" or "at file:line:col", not a
  // wrapped sentence that starts with "at 37 tok/s (22 to 52…".
  expect(all).not.toMatch(/\n\s+at (\S+ \(.*:\d+:\d+\)|\S+:\d+:\d+)|node:internal|TypeError|ReferenceError/);
  expect(r.stdout).not.toMatch(/\b(undefined|NaN|Infinity)\b/);
}

const written = (file: string) => readFileSync(path.join(work, file), 'utf8');

describe('help and version', () => {
  it.concurrent.each([[['--help']], [['-h']], [[]], [['help']]])('%j prints the help, exit 0', async (args) => {
    const r = await cli(args);
    clean(r);
    expect(r.code).toBe(0);
    expect(r.stdout.trimEnd()).toBe(USAGE);
  });

  it.concurrent('--version and -v print the name, the version and the fingerprint of the sources, exit 0', async () => {
    for (const flag of ['--version', '-v']) {
      const r = await cli([flag]);
      expect(r).toEqual({ code: 0, stdout: `${__CLI_NAME__} ${__CLI_VERSION__} (sources ${sourceHash('.')})\n`, stderr: '' });
    }
  });
});

/**
 * Every option alone on the small model (with --tokens 8k to keep it short), and what it must change. `code` is
 * the expected exit code when it is not 0, `files` the files it must write.
 */
const OPTION_CASES: { option: string; args: string[]; code?: number; files?: string[]; says?: RegExp }[] = [
  { option: 'tokens', args: ['--tokens', '32k'], says: /at 32k tokens/ },
  { option: 'cache', args: ['--cache', 'auto'], says: /full-precision KV cache \(auto\)/ },
  { option: 'engine', args: ['--engine', 'sglang'], says: /SGLang/ },
  { option: 'users', args: ['--users', '100'], says: /For\s+100\s+users\s+\(10%\s+active/ },
  { option: 'active', args: ['--active', '20'], says: /÷ 20% active at once/ },
  { option: 'min-tok-s', args: ['--min-tok-s', '30'], says: /at least 30 tok\/s per user/ },
  { option: 'max-wait', args: ['--max-wait', '5'], says: /first token within 5 s\./ },
  { option: 'typical-context', args: ['--typical-context', '2k'], says: /brings 2k new tokens/ },
  { option: 'output-tokens', args: ['--output-tokens', '1k'], says: /generates 1k/ },
  { option: 'prompt-cache', args: ['--prompt-cache', '80'], says: /Scenario "chat"/ },
  { option: 'gpu', args: ['--gpu', 'h100'], says: /Every layout:[\s\S]*H100/ },
  { option: 'count', args: ['--count', '2'], says: /2× / },
  { option: 'vram', args: ['--vram', '24'], says: /24 GiB GPU/ },
  { option: 'util', args: ['--util', '0.8'], says: /--gpu-memory-utilization 0.8|Margin/ },
  // The catalog model has only its root folder: another variant is refused with the list (exit 1).
  { option: 'variant', args: ['--variant', 'FP8'], code: 1, says: /no safetensors model in "FP8"\. Variants: \(root\)/ },
  { option: 'revision', args: ['--revision', 'main'], says: /fits/ },
  { option: 'csv', args: ['--csv', 'one.csv'], files: ['one.csv'], says: /CSV written to one\.csv/ },
  { option: 'json', args: ['--json', 'one.json'], files: ['one.json'], says: /JSON written to one\.json/ },
  { option: 'md', args: ['--md', 'one.md'], files: ['one.md'], says: /Markdown written to one\.md/ },
  { option: 'max-batched-tokens', args: ['--max-batched-tokens', '4096'], says: /fits/ },
  { option: 'max-seqs', args: ['--max-seqs', '64'], says: /fits/ },
  { option: 'text-only', args: ['--text-only'], says: /fits/ },
  // No NVIDIA GPU in the test machine: --this-pc adds nothing and the answer stands.
  { option: 'this-pc', args: ['--this-pc'], says: /^Summary/ },
  { option: 'all-hw', args: ['--all-hw'], says: /Every layout:/ },
  { option: 'details', args: ['--details'], says: /Details — .*\n[\s\S]*Memory as the engine counts it/ },
  { option: 'csv-excel-eu', args: ['--csv', 'eu.csv', '--csv-excel-eu'], files: ['eu.csv'], says: /CSV written to eu\.csv/ },
  // Built-in models need no network, even offline.
  { option: 'offline', args: ['--offline'], says: /built-in data/ },
  // --refresh reads Hugging Face again: unreachable here, and nothing is cached yet (exit 1).
  { option: 'refresh', args: ['--refresh'], code: 1, says: /^Error: Hugging Face does not answer .*Try again, or use --offline/ },
  { option: 'no-color', args: ['--no-color'], says: /fits/ },
  { option: 'ascii', args: ['--ascii'], says: /fits/ },
];

describe(`every option alone on ${SMALL}`, () => {
  it('covers every option the command accepts', () => {
    expect(new Set(OPTION_CASES.map((c) => c.option))).toEqual(new Set([...VALUED, ...BOOLEAN]));
  });

  it.concurrent.each(OPTION_CASES)('--$option', async ({ args, code = 0, files = [], says }) => {
    const r = await cli([SMALL, ...(args[0] === '--tokens' ? [] : ['--tokens', '8k']), ...args]);
    clean(r);
    expect(r.code, r.stdout).toBe(code);
    expect(r.stderr).toBe('');
    if (says) expect(r.stdout).toMatch(says);
    for (const f of files) expect(written(f).length).toBeGreaterThan(100);
    // ASCII only.
    if (args.includes('--ascii')) expect([...r.stdout].filter((ch) => ch.charCodeAt(0) > 126)).toEqual([]);
    // Not a terminal: never colours.
    expect(r.stdout).not.toContain(`${String.fromCharCode(27)}[`);
  });

  it.concurrent('the CSV, JSON and Markdown files are complete and consistent', async () => {
    const r = await cli([SMALL, '--tokens', '8k', '--csv', 'all.csv', '--json', 'all.json', '--md', 'all.md']);
    clean(r);
    expect(r.code).toBe(0);
    const json = JSON.parse(written('all.json')) as { headline: string; summary: string[]; rows: unknown[] };
    const csv = written('all.csv').trimEnd().split('\r\n');
    expect(csv.length - 1).toBe(json.rows.length);
    expect(written('all.md')).toContain(`- ${json.summary[0] as string}`);
    expect(r.stdout.replace(/\s+/g, ' ')).toContain(json.headline);
  });
});

describe('errors: exit 1, the reason and the help, no stack trace', () => {
  it.concurrent.each([
    [['--frobnicate'], /^Error: unknown option --frobnicate \(see --help\)/],
    [['--tokens'], /^Error: --tokens: the value is missing/],
    [['--active', '0'], /^Error: --active: expected a number from 0\.1 to 100, not "0"/],
    [['--count', '3'], /^Error: --count: 1, 2, 4, 8 or 16 GPUs/],
    [['--util', '2'], /^Error: --util: expected a number from 0\.05 to 0\.99, not "2"/],
    [['--prompt-cache', '100'], /^Error: --prompt-cache: expected a number from 0 to 99, not "100"/],
    [['--cache', 'int4'], /^Error: --cache: fp8 or auto \(auto = the model's own precision, e\.g\. bf16\), not "int4"/],
    [['--cache', 'bf16'], /^Error: --cache: fp8 or auto \(auto = the model's own precision, e\.g\. bf16\), not "bf16"/],
    [['--tokens', '32k', '--tokens', '64k'], /^Error: --tokens is given twice: give it once/],
    [['--engine', 'trt'], /^Error: --engine: vllm or sglang, not "trt"/],
    [['--offline', '--refresh'], /^Error: --offline and --refresh exclude each other/],
    [['--tokens', 'lots'], /^Error: --tokens: expected a number \(e\.g\. 32k\), not "lots"/],
  ])('%j', async (args, message) => {
    const r = await cli([SMALL, ...args]);
    clean(r);
    expect(r.code).toBe(1);
    expect(r.stdout).toMatch(message);
    expect(r.stdout).toContain(USAGE);
  });
});

describe('models that are not built in, without a network', () => {
  it.concurrent.each([
    // A repo that does not exist, and a gated catalog model the pack leaves out (Llama 4 Scout): nothing cached.
    ['nobody/no-such-model', '--offline'],
    ['meta-llama/Llama-4-Scout-17B-16E-Instruct', '--offline'],
    // A public model outside the catalog, never read before.
    ['Qwen/Qwen3-8B', '--offline'],
  ])('%s %s: not read on this computer yet, exit 1 with the reason', async (repo, flag) => {
    const r = await cli([repo, flag]);
    clean(r);
    expect(r.code).toBe(1);
    expect(r.stdout).toContain(`Error: ${repo} has not been read on this computer yet (--offline uses only the local cache): run once without --offline.`);
  });

  // With --offline a cache miss is not a network failure, and the error does not advise the option already given.
  it('--offline: the error does not advise --offline again', async () => {
    const r = await cli(['Qwen/Qwen3-8B', '--offline']);
    expect(r.code).toBe(1);
    expect(r.stdout).not.toMatch(/does not answer|use --offline/);
  });

  it.concurrent.each([['nobody/no-such-model'], ['meta-llama/Llama-4-Scout-17B-16E-Instruct'], ['not an exact id']])('%s online: Hugging Face does not answer, exit 1', async (repo) => {
    const r = await cli([repo]);
    clean(r);
    expect(r.code).toBe(1);
    expect(r.stdout).toMatch(/^Error: Hugging Face does not answer \(.*\)\. Try again, or use --offline/);
  });
});

describe('ready-made reports', () => {
  it.concurrent('report: every model, exit 0', async () => {
    const r = await cli(['report']);
    clean(r);
    expect(r.code).toBe(0);
    expect(r.stdout).toMatch(/GLM/);
    expect(r.stdout.split('\n').length).toBeGreaterThan(20);
  });

  it.concurrent('report glm --csv glm.csv: the GLM models only, and the CSV written', async () => {
    const r = await cli(['report', 'glm', '--csv', 'glm.csv']);
    clean(r);
    expect(r.code).toBe(0);
    expect(r.stdout).toMatch(/CSV written to glm\.csv/);
    const csv = written('glm.csv').trimEnd().split('\r\n');
    expect(csv.length).toBeGreaterThan(1);
    for (const line of csv.slice(1)) expect(line).toMatch(/GLM/i);
  });

  it.concurrent('report with a filter nothing matches: the models listed, exit 2', async () => {
    const r = await cli(['report', 'zzz-nothing']);
    clean(r);
    expect(r.code).toBe(2);
    expect(r.stdout).toMatch(/^No model of the reports matches "zzz-nothing"\. Models: /);
  });
});

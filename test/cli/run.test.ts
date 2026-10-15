import type { execFile } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { catalog, customGpu, localGpu, matchGpus, runPlan, type LocalGpu, type PlanDeps } from '../../src/cli/run';
import { USAGE } from '../../src/cli/args';
import { COMPAT_UNCHECKED } from '../../src/cli/render';
import { packReport, reportFileSchema } from '../../src/cli/report';
import reportJson from '../../data/report.json';
import { layoutsFor } from '../../src/planner/hardware';
import { configJson, fakeHub, GPUS, HUB, qwenRepo, type FakeRepo } from './fixtures';

const GiB = 1024 ** 3;
const LAYOUTS = layoutsFor(GPUS).length;
const tmp = mkdtempSync(path.join(os.tmpdir(), 'vram-run-'));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));
let n = 0;
const freshDir = () => path.join(tmp, `c${n++}`);

interface RunOpts {
  repos?: Record<string, FakeRepo>;
  search?: unknown[];
  local?: LocalGpu;
  env?: Record<string, string>;
  seen?: { url: string; auth?: string }[];
  fetch?: typeof fetch;
  deps?: Partial<PlanDeps>;
}

async function run(argv: string[], o: RunOpts = {}) {
  const out: string[] = [];
  const written: Record<string, string> = {};
  const code = await runPlan(argv, {
    fetch: o.fetch ?? fakeHub(o.repos ?? { 'Qwen/Qwen3-8B': qwenRepo() }, o.search, o.seen),
    env: o.env ?? { WT_SESSION: '1' },
    localGpu: () => Promise.resolve(o.local),
    out: (t) => out.push(t),
    writeFile: (file, content) => {
      written[file] = content;
      return Promise.resolve();
    },
    baseUrl: HUB,
    cacheDir: freshDir(),
    isTTY: false,
    columns: 300,
    platform: 'linux',
    ...o.deps,
  });
  return { code, text: out.join('\n'), out, written };
}

describe('runPlan: arguments', () => {
  it('--help and no arguments print the help; a bad argument the error and the help', async () => {
    expect(await run(['--help'])).toMatchObject({ code: 0, text: USAGE });
    expect(await run([])).toMatchObject({ code: 0, text: USAGE });
    expect(await run(['a/b', '--engine', 'tgi'])).toMatchObject({ code: 1, text: `Error: --engine: vllm or sglang, not "tgi"\n\n${USAGE}` });
  });
});

describe('runPlan: a model on Hugging Face', () => {
  /** The overview's card lines (two spaces, a card name, then its verdict). */
  const cardRows = (t: string) =>
    (t.split('\nEvery layout:')[0] as string).split('\n').filter((l) => /^ {2}\S.* {2}(fits|does not fit|engine does not start|won't start|at the limit|not enough|speed unknown|too slow)/.test(l));

  it('Qwen3-8B at its maximum context, no GPU here: the answer first, the scenario, what was read, one line per card, the legend', async () => {
    const r = await run(['Qwen/Qwen3-8B'], { deps: { columns: undefined } });
    expect(r.code).toBe(0);
    expect(r.out).toHaveLength(1);
    expect(r.text).toMatch(/^Summary\n {2}Recommendation: 1 server with 1× NVIDIA RTX 5090 serves \d+ users guaranteed/);
    expect(r.text).toMatch(/\nTechnical detail\nQwen3-8B at 40k tokens, fp8 KV cache: serves users from 1× RTX 5090, or 1× \S+.*\. With 1× RTX 5090: /);
    expect(r.text).toContain('\nScenario "documents": each request brings 20k new tokens to read');
    expect(r.text).toContain('\nQwen/Qwen3-8B · BF16 · dense · vLLM 0.31.0 · fp8 KV cache · context 40k tokens (the model maximum)\n');
    expect(r.text).toMatch(/\nRead config \+ tensor headers \(\d+ KB\) from Hugging Face — weights not downloaded\.\n/);
    expect(r.text).toContain('BF16 × dense verified on vLLM measurements');
    expect(r.text).toContain(COMPAT_UNCHECKED.slice(0, 60));
    expect(cardRows(r.text)).toHaveLength(GPUS.length);
    expect(r.text).not.toContain('(this PC)');
    expect(r.text).not.toContain('Details');
    // 120 columns when the terminal does not say (and COLUMNS is not set).
    expect(Math.max(...r.text.split('\n').map((l) => l.length))).toBeLessThanOrEqual(120);
    const wide = await run(['Qwen/Qwen3-8B'], { deps: { columns: undefined }, env: { WT_SESSION: '1', COLUMNS: '400' } });
    expect(wide.text).toMatch(/GiB \([\d.]+ to [\d.]+ GiB, ±[\d.]+%\) of 31\.3 GiB/);
  });

  it('what was read: the network first, then the local cache; --refresh asks the network again', async () => {
    const dir = freshDir();
    const first = await run(['Qwen/Qwen3-8B', '--tokens', '8k'], { deps: { cacheDir: dir } });
    expect(first.text).toMatch(/Read config \+ tensor headers \(\d+ KB\) from Hugging Face — weights not downloaded\./);
    const again = await run(['Qwen/Qwen3-8B', '--tokens', '8k'], { deps: { cacheDir: dir } });
    expect(again.text).toMatch(/Read config \+ tensor headers \(\d+ KB\) from the local cache — weights not downloaded\./);
    const seen: { url: string }[] = [];
    const fresh = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--refresh'], { seen, deps: { cacheDir: dir } });
    expect(fresh.text).toContain('from Hugging Face — weights not downloaded');
    expect(seen.length).toBeGreaterThan(0);
  });

  it('built-in models: read from the data pack without the network (even --offline); --refresh and other models use Hugging Face', async () => {
    const pack = fakeHub({ 'Qwen/Qwen3-8B': qwenRepo() });
    const builtin = { date: '2026-10-15', has: (repo: string, rev: string) => repo === 'Qwen/Qwen3-8B' && rev === 'main', fetch: pack };
    const down = () => Promise.reject(new Error('no network'));
    const r = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--offline'], { fetch: down, deps: { builtin } });
    expect(r.code).toBe(0);
    expect(r.text).toContain('Model description from built-in data (2026-10-15) — weights not downloaded; --refresh reads it again from Hugging Face.');
    const fresh = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--refresh'], { deps: { builtin } });
    expect(fresh.text).toContain('from Hugging Face — weights not downloaded');
    const other = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--revision', 'v2'], { deps: { builtin } });
    expect(other.text).toContain('from Hugging Face — weights not downloaded');
    // A pack that lacks a file: said as such, not blamed on Hugging Face.
    const broken = { ...builtin, fetch: (() => Promise.resolve(new Response('', { status: 500 }))) as typeof fetch };
    const bad = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--offline'], { fetch: down, deps: { builtin: broken } });
    expect(bad.code).toBe(1);
    expect(bad.text).toMatch(/^Error: the built-in data of Qwen\/Qwen3-8B is incomplete \(.*\); --refresh reads the model from Hugging Face\.$/);
  });

  it('only the engine refuses the card: "fits in memory … but the engine does not start" with its reason', async () => {
    const repo = qwenRepo();
    const dsv41 = { ...repo, configs: { '': { ...(repo.configs[''] as object), architectures: ['DeepseekV41ForCausalLM'] } } };
    const r = await run(['a/dsv41', '--tokens', '8k', '--gpu', '5090'], { repos: { 'a/dsv41': dsv41 } });
    expect(r.code).toBe(0);
    expect(r.text).toContain('Recommendation: none — the model fits in memory on 1 server with 1× NVIDIA RTX 5090, but the inference engine does not start there.');
    expect(r.text).toMatch(/\nTechnical detail\ndsv41 at 8k tokens, fp8 KV cache: fits in memory from 1× RTX 5090, but the engine does not start \(does not start with vLLM 0\.31\.0 on this card: no checkpoint of deepseek-v4-1-flash in the catalog starts: [\s\S]+\); no layout of the catalog runs it\./);
  });

  it('a Mistral-format checkpoint (params.json, no config.json): said clearly, nothing asked for config.json', async () => {
    const seen: { url: string }[] = [];
    const mistral: FakeRepo = {
      files: [
        { path: 'params.json', size: 1 },
        { path: 'consolidated-00001-of-00002.safetensors', size: GiB },
      ],
      configs: {},
    };
    const r = await run(['mistralai/Big'], { repos: { 'mistralai/Big': mistral }, seen });
    expect(r).toMatchObject({ code: 1, text: 'mistralai/Big ships only the Mistral format (params.json, no config.json): the calculator reads config.json, so it cannot plan this checkpoint yet. A version of the model with a config.json works.' });
    expect(seen.some((s) => s.url.endsWith('config.json'))).toBe(false);
  });

  it('a model of the compatibility matrix: its verdicts per card, without the "not in the matrix" line', async () => {
    const r = await run(['zai-org/GLM-5.3', '--tokens', '8k'], { repos: { 'zai-org/GLM-5.3': qwenRepo() } });
    expect(r.code).toBe(0);
    expect(r.text).not.toContain(COMPAT_UNCHECKED.slice(0, 60));
  });

  it('the GPU of this PC: its line in the overview, known to the catalog or not', async () => {
    // Without --this-pc nvidia-smi is never asked: nothing is run.
    let asked = 0;
    const quiet = await run(['Qwen/Qwen3-8B', '32k'], { deps: { localGpu: () => ((asked += 1), Promise.resolve({ name: 'NVIDIA GeForce RTX 5090', totalMiB: 32_607 })) } });
    expect(asked).toBe(0);
    expect(quiet.text).not.toContain('(this PC)');
    const known = await run(['Qwen/Qwen3-8B', '32k', '--this-pc'], { local: { name: 'NVIDIA GeForce RTX 5090', totalMiB: 32_607 } });
    expect(known.text).toContain('context 32k tokens (model maximum 40k)\n');
    expect(known.text).toMatch(/\n {2}RTX 5090 \(this PC\) +fits /);
    const unknown = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--this-pc'], { local: { name: 'NVIDIA Tesla T4', totalMiB: 15_360 } });
    // A card the catalog does not have: memory only, no speed, one more line.
    expect(cardRows(unknown.text)).toHaveLength(GPUS.length + 1);
    expect(unknown.text).toMatch(/\n {2}Tesla T4 \(this PC\) +fits, speed unknown \([\d,]+\) +2 \(1 model on 2\) +[\d.]+ GiB \(.*\) of 14\.5 GiB .* n\/a/);
    // --gpu leaves this PC's card out of the list: it is planned apart.
    const apart = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--gpu', 'h100-sxm5', '--this-pc'], { local: { name: 'NVIDIA GeForce RTX 5090', totalMiB: 32_607 } });
    expect(cardRows(apart.text).map((l) => l.trim().split('  ')[0])).toEqual(['H100 SXM5 80GB', 'RTX 5090 (this PC)']);
    // Planned apart: never in the summary.
    const summary = apart.text.split('\nTechnical detail')[0] as string;
    expect(summary).not.toMatch(/consumer|5090/);
  });

  it('--gpu: the card in detail (its layouts, every number, the roofline); --count; --all-hw; --vram; SGLang and the scheduler options', async () => {
    const one = await run(['Qwen/Qwen3-8B', '--gpu', 'RTX 5090', '--count', '2']);
    expect(one.code).toBe(0);
    expect(one.text).toContain('Every layout:');
    expect(one.text).toContain('2× RTX 5090 (1 model on 2)');
    expect(one.text).toMatch(/\nDetails — 2× RTX 5090 \(/);
    expect(one.text).toMatch(/one request alone .* of the physical limit \(\d+ tok\/s at the datasheet bandwidth\)/);
    expect(one.text).toContain('Memory as the engine counts it');
    const h200 = await run(['Qwen/Qwen3-8B', '--gpu', 'h200']);
    expect(cardRows(h200.text).map((l) => l.trim().split('  ')[0])).toEqual(['H200 SXM 141GB', 'H200 NVL 141GB']);
    const none = await run(['Qwen/Qwen3-8B', '--gpu', 'voodoo']);
    expect(none.code).toBe(1);
    expect(none.text).toMatch(/^No card of the catalog matches "voodoo"\. Cards: rtx-3090-24gb, rtx-4090-24gb, .*b300-sxm\.$/);
    expect(none.text).not.toContain('nvl72');
    const custom = await run(['Qwen/Qwen3-8B', '--vram', '48', '--engine', 'sglang', '--cache', 'auto', '--util', '0.8', '--max-batched-tokens', '4k', '--max-seqs', '32', '--text-only']);
    expect(custom.code).toBe(0);
    expect(custom.text).toContain('· SGLang 0.5.21 · full-precision KV cache (auto) ·');
    expect(custom.text).toMatch(/\n {2}48 GiB GPU +fits /);
    expect(custom.text).toContain('8× 48 GiB GPU (1 model on 8)');
    expect(custom.text).toContain('Details — 1× 48 GiB GPU');
    const all = await run(['Qwen/Qwen3-8B', '--all-hw']);
    const start = all.text.split('\n').findIndex((l) => l === 'Every layout:');
    expect(start).toBeGreaterThan(0);
    expect(all.text.split('\n').slice(start + 3, start + 3 + LAYOUTS).every((l) => /^ {2}\S/.test(l))).toBe(true);
    expect((await run(['Qwen/Qwen3-8B', '--gpu', 'all'])).text).toContain('Every layout:');
  });

  it('--users: the active share and the option that serves them on top, per card the fewest GPUs that do; the agents; none', async () => {
    const few = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--users', '50']);
    expect(few.code).toBe(0);
    expect(few.text).toMatch(/ For 50 users \(10% active ≈ 5 concurrent requests\) you need 1× \S+.*: \d+ users guaranteed, \d+ likely, at \d+ tok\/s/);
    expect(few.text).toContain('Per card: the layout with the fewest GPUs in total that serves 50 users (10% active ≈ 5 concurrent requests)');
    expect(few.text).toContain('Users include AI agents.');
    const busy = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--users', '50', '--active', '50']);
    expect(busy.text).toContain('For 50 users (50% active ≈ 25 concurrent requests)');
    const many = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--users', '10M']);
    // More users than one server holds: as many independent servers as they take (10M = 10,000,000 people).
    expect(many.text).toMatch(/For 10,000,000 users \(10% active ≈ 1,000,000 concurrent requests\) you need [\d,]+ servers of /);
    expect(many.text).toMatch(/\n {2}Recommendation: [\d,]+ servers with \d+× NVIDIA .* each serve 10,000,000 users /);
    expect(many.text).toMatch(/\n {2}RTX 3090 +fits +\S.* +[\d,]+ +/);
    // Nobody gets the minimum speed: too slow for anyone, whatever the servers.
    const none = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--users', '50', '--min-tok-s', '10000']);
    expect(none.text).toContain('Recommendation: none — too slow at 8k tokens on any card');
    expect(none.text).toMatch(/\n {2}RTX 3090 +too slow for anyone/);
    // --count alone: every card, that many GPUs, said in the scope.
    expect((await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--users', '50', '--min-tok-s', '10000', '--count', '1'])).text).toContain('too slow at 8k tokens on any card of the catalog with 1 GPU');
    // --count 16: two 8-GPU servers per layout; the summary and the detail say the same servers and users per server.
    const sixteen = (await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--users', '1M', '--gpu', 'h200-sxm', '--count', '16'])).text.replace(/\n {6}/g, ' ');
    const sum = /Recommendation: ([\d,]+) servers with 8× NVIDIA H200 SXM 141GB each serve 1,000,000 users .*: ≈ ([\d,]+) users per server\./.exec(sixteen);
    const tech = /you need ([\d,]+) servers of 8× H200 SXM 141GB \(\S+.*?\): ([\d,]+) users guaranteed, [\d,]+ likely per server/.exec(sixteen.replace(/\n/g, ' '));
    expect(sum).not.toBeNull();
    expect([sum?.[1], sum?.[2]]).toEqual([tech?.[1], tech?.[2]]);
    expect(sixteen).not.toMatch(/16× H200 SXM 141GB \(2 copies × 8\):/);
    // --details: the users' option; none serves them: the simplest one.
    expect((await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--users', '150', '--details'])).text).toMatch(/\nDetails — /);
    expect((await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--users', '50', '--min-tok-s', '10000', '--details'])).text).toMatch(/\nDetails — \S+.*, one request of 8k tokens:/);
  });

  it('only borderline fits: "at the limit already with"', async () => {
    // A card just large enough within the uncertainty of the fixed costs.
    const r = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--vram', '20', '--count', '1']);
    expect(r.text).toMatch(/\nTechnical detail\nQwen3-8B at 8k tokens, fp8 KV cache: at the limit already with 1× 20 GiB GPU\./);
  });

  it('a model too large for every card: the closest attempt; --details shows this PC or the first row', async () => {
    const huge: FakeRepo = {
      files: [
        { path: 'config.json', size: 1 },
        { path: 'model.safetensors', size: 4096 * GiB },
      ],
      configs: { '': { architectures: ['LlamaForCausalLM'], num_hidden_layers: 4, num_attention_heads: 8, num_key_value_heads: 8, hidden_size: 1024, vocab_size: 1000, torch_dtype: 'bfloat16' } },
    };
    const r = await run(['big/model'], { repos: { 'big/model': huge } });
    expect(r.code).toBe(0);
    // No max context in the config: 8192.
    expect(r.text).toMatch(/Recommendation: none — the model does not fit in one server on any card of the catalog/);
    expect(r.text).toMatch(/\nTechnical detail\nmodel at 8k tokens, fp8 KV cache: does not fit in one server on any card of the catalog \(with \d× \S+.* it needs [\d.,]+ GiB per GPU of [\d.]+ GiB[;)]/);
    expect(r.text).toMatch(/\n {2}RTX 3090 +does not fit even with 4 +- +with 4: /);
    const local = await run(['big/model', '--details', '--this-pc'], { repos: { 'big/model': huge }, local: { name: 'NVIDIA GeForce RTX 5090', totalMiB: 32_607 } });
    expect(local.text).toContain('\nDetails — ');
    const first = await run(['big/model', '--details'], { repos: { 'big/model': huge } });
    expect(first.text).toContain('\nDetails — ');
  });

  it('a multimodal model, a variant folder, a revision and the Hugging Face token', async () => {
    const gemma = configJson('gemma-3-27b-it');
    const repo: FakeRepo = {
      files: [
        { path: 'config.json', size: 1 },
        { path: 'model.safetensors', size: 2 * GiB },
        { path: 'FP8/config.json', size: 1 },
        { path: 'FP8/model.safetensors', size: GiB },
      ],
      configs: { '': gemma, FP8: gemma },
    };
    const seen: { url: string; auth?: string }[] = [];
    const r = await run(['g/gemma', '--variant', 'FP8', '--revision', 'v1', '--tokens', '8k'], { repos: { 'g/gemma': repo }, seen, env: { HUGGING_FACE_HUB_TOKEN: 'hf_x', WT_SESSION: '1' } });
    expect(r.code).toBe(0);
    expect(r.text).toMatch(/\nTechnical detail\ngemma \(FP8\) at 8k tokens/);
    expect(r.text).toContain('\ng/gemma (FP8) · ');
    expect(r.text).toContain('Multimodal model: with --text-only');
    expect(seen.some((s) => s.url === `${HUB}/api/models/g/gemma/revision/v1` && s.auth === 'Bearer hf_x')).toBe(true);
  });

  it('exports: CSV (standard, or for Excel in comma-decimal locales), JSON and Markdown through writeFile, each announced', async () => {
    const r = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--csv', 'a.csv', '--json', 'a.json', '--md', 'a.md', '--this-pc'], { local: { name: 'NVIDIA GeForce RTX 5090', totalMiB: 32_607 } });
    expect(r.code).toBe(0);
    expect(r.out.slice(1)).toEqual(['CSV written to a.csv', 'JSON written to a.json', 'Markdown written to a.md']);
    expect(r.written['a.csv']?.split('\r\n')[1]).toMatch(/^Qwen\/Qwen3-8B,Qwen\/Qwen3-8B,BF16,dense,vLLM,0\.31\.0,fp8,8192,chosen \(--tokens\),4096,4096,500,20,10,/);
    // Every layout in the CSV and the JSON, the overview's cards marked.
    expect(r.written['a.csv']?.split('\r\n').length).toBe(LAYOUTS + 2);
    expect(r.written['a.csv']).toMatch(/,rtx-5090-32gb,consumer,yes,31.30,RTX 5090,1,1,,1,1,1,fits,/);
    const json = JSON.parse(r.written['a.json'] as string) as { model: { repo: string; family: string; source: { kind: string } }; rows: unknown[]; cards: unknown[] };
    expect(json.model).toMatchObject({ repo: 'Qwen/Qwen3-8B', family: 'dense', source: { kind: 'network' } });
    expect(json.rows).toHaveLength(LAYOUTS);
    expect(json.cards).toHaveLength(GPUS.length);
    expect(r.written['a.md']).toMatch(/^# Summary: Qwen3-8B\n\n## Scenario\n/);
    const eu = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--csv', 'b.csv', '--csv-excel-eu']);
    expect(eu.written['b.csv']?.startsWith('﻿Model;Version;')).toBe(true);
  });

  it('ASCII on the old Windows console, colours on a terminal', async () => {
    const r = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--csv', 'x.csv'], { env: {}, deps: { platform: 'win32' } });
    expect(r.text).not.toMatch(/[–—×≥·±]/);
    const c = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--this-pc'], { local: { name: 'NVIDIA GeForce RTX 5090', totalMiB: 32_607 }, deps: { isTTY: true } });
    expect(c.text.startsWith('\u001b[1mSummary')).toBe(true);
    expect(c.text).toContain('\u001b[32mfits');
  });
});

describe('runPlan: not an exact repo, errors, offline', () => {
  const search = [
    { id: 'Qwen/Qwen3-8B', tags: ['safetensors'], safetensors: { total: 8_190_735_360 } },
    { id: 'mlx-community/Qwen3-8B-4bit', tags: ['safetensors', 'mlx'] },
    { id: 'someone/qwen3-raw', tags: ['safetensors'] },
  ];

  it('a search: the safetensors models found (MLX left out), or none', async () => {
    const r = await run(['qwen3'], { search, repos: {} });
    expect(r).toMatchObject({ code: 2, text: '"qwen3" is not an exact repo id. Safetensors models found:\n  Qwen/Qwen3-8B  (8.2 B parameters)\n  someone/qwen3-raw\n\nRun again with one of these ids.' });
    // A repo id that does not exist (404, or 401 without a token) searches its words.
    const seen: { url: string }[] = [];
    expect((await run(['Nobody/Qwen3'], { search, repos: {}, seen })).code).toBe(2);
    expect(seen.some((s) => s.url.includes('search=Nobody+Qwen3'))).toBe(true);
    expect((await run(['a/locked'], { search, repos: { 'a/locked': { files: [], configs: {}, status: 401 } } })).code).toBe(2);
    expect(await run(['Nobody/Nothing'], { repos: {} })).toMatchObject({ code: 2, text: '"Nobody/Nothing" does not exist on Hugging Face and the search finds no safetensors model.' });
  });

  it('repos vLLM and SGLang cannot load, and a missing variant', async () => {
    const gguf: FakeRepo = { files: [{ path: 'model-Q4_K_M.gguf', size: GiB }], configs: {} };
    expect(await run(['a/gguf'], { repos: { 'a/gguf': gguf } })).toMatchObject({ code: 1, text: 'a/gguf has no safetensors model vLLM or SGLang can load (config.json + weights).' });
    expect(await run(['Qwen/Qwen3-8B', '--variant', 'INT4'])).toMatchObject({ code: 1, text: 'Qwen/Qwen3-8B: no safetensors model in "INT4". Variants: (root) (use --variant).' });
    const onlyFp8: FakeRepo = {
      files: [
        { path: 'FP8/config.json', size: 1 },
        { path: 'FP8/model.safetensors', size: GiB },
      ],
      configs: {},
    };
    expect((await run(['a/b'], { repos: { 'a/b': onlyFp8 } })).text).toBe('a/b: no safetensors model in "(root)". Variants: FP8 (use --variant).');
  });

  it('gated repos, a network down, other hub errors', async () => {
    const gated = await run(['meta/gated'], { repos: { 'meta/gated': { ...qwenRepo(), configStatus: 403 } } });
    expect(gated.code).toBe(1);
    expect(gated.text).toBe('Error: meta/gated is gated or private: accept its license on huggingface.co and set HF_TOKEN (Access to meta/gated requires accepting its license).');
    const down = await run(['Qwen/Qwen3-8B'], { fetch: () => Promise.reject(new Error('ECONNREFUSED')) });
    expect(down).toMatchObject({ code: 1, text: 'Error: Hugging Face does not answer (Cannot reach Hugging Face: ECONNREFUSED). Try again, or use --offline for the models read before.' });
    expect(await run(['a/b'], { repos: { 'a/b': { files: [], configs: {}, status: 429 } } })).toMatchObject({ code: 1, text: 'Error: Hugging Face rate limit reached' });
  });

  it('--offline: what was read before works without the network; the rest fails', async () => {
    const dir = freshDir();
    const repos = { 'Qwen/Qwen3-8B': qwenRepo() };
    expect((await run(['Qwen/Qwen3-8B', '--tokens', '8k'], { repos, deps: { cacheDir: dir } })).code).toBe(0);
    const off = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--offline'], { fetch: () => Promise.reject(new Error('no network')), deps: { cacheDir: dir } });
    expect(off.code).toBe(0);
    expect(off.text).toContain('Qwen/Qwen3-8B · BF16');
    expect(off.text).toContain('from the local cache — weights not downloaded');
    const miss = await run(['Qwen/Qwen3-8B', '--offline'], { deps: { cacheDir: freshDir() } });
    expect(miss.code).toBe(1);
    expect(miss.text).toBe('Error: Qwen/Qwen3-8B has not been read on this computer yet (--offline uses only the local cache): run once without --offline.');
  });

  it('the default cache folder comes from the environment', async () => {
    const dir = freshDir();
    const r = await run(['Qwen/Qwen3-8B', '--tokens', '8k'], { env: { CAPACITY_PLANNER_CACHE: dir, WT_SESSION: '1' }, deps: { cacheDir: undefined } });
    expect(r.code).toBe(0);
    const off = await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--offline'], { env: { CAPACITY_PLANNER_CACHE: dir, WT_SESSION: '1' }, fetch: () => Promise.reject(new Error('x')), deps: { cacheDir: undefined } });
    expect(off.code).toBe(0);
  });
});

describe('runPlan: the ready-made reports', () => {
  const bundled = reportFileSchema.parse(reportJson);
  const one = { ...bundled, models: bundled.models.filter((m) => m.id.startsWith('glm-5-3') && !m.id.includes('flash')) };
  const { models, missing, ...rest } = one;
  const report = { ...rest, packed: packReport({ models, missing }) };

  it('the report, filtered, exported; ASCII when asked', async () => {
    const r = await run(['report', 'glm', '--csv', 'r.csv', '--json', 'r.json', '--md', 'r.md', '--ascii'], { deps: { report } });
    expect(r.code).toBe(0);
    expect(r.out[0]).toMatch(/\nGLM-5\.3 {2}\(zai-org\/GLM-5\.3\)\n/);
    // The GPUs tried per card, as the live command says it.
    expect((r.out[0] as string).replace(/\n {4,}/g, ' ')).toMatch(/Not enough \(does not fit, does not start, or nobody guaranteed 20 tok\/s\): .*consumer cards, even with \d+/);
    expect(r.out[0]).not.toContain('RTX 5090:');
    expect(r.out.slice(1)).toEqual(['CSV written to r.csv', 'JSON written to r.json', 'Markdown written to r.md']);
    const lines = (r.written['r.csv'] as string).split('\r\n');
    const contexts = (one.models[0] as (typeof one.models)[number]).variants.reduce((a, v) => a + v.contexts.length, 0);
    expect(lines.length).toBe(2 + contexts * GPUS.length);
    expect((JSON.parse(r.written['r.json'] as string) as unknown[]).length).toBe(contexts);
    expect(r.written['r.md']).toContain('## GLM-5.3 · ');
  });

  it('no model matches: the list of models, code 2', async () => {
    expect(await run(['report', 'zzz'], { deps: { report } })).toMatchObject({ code: 2, text: 'No model of the reports matches "zzz". Models: GLM-5.3.' });
    expect(await run(['report'], { deps: { report: { ...rest, models: [], missing: [] } } })).toMatchObject({ code: 2, text: 'No model of the reports matches "". Models: .' });
  });

  it('the bundled report', async () => {
    const r = await run(['report']);
    expect(r.code).toBe(0);
    expect(r.text).toMatch(/^Ready-made reports: vLLM /);
    expect(r.text).toContain('Every card for one model: add its name');
  });
});

describe('catalog, local GPU and options', () => {
  it('the catalog: cards sold as cards (no NVL72 racks)', () => {
    expect(catalog().length).toBe(GPUS.length);
    expect(catalog().some((g) => g.id.includes('nvl72'))).toBe(false);
    expect(catalog().find((g) => g.id === 'rtx-5090-32gb')?.name).toBe('RTX 5090');
  });

  it('nvidia-smi: the first GPU, undefined without one', async () => {
    const fake = (err: Error | null, stdout: string) => ((_cmd: string, _args: string[], _opts: object, cb: (e: Error | null, out: string) => void) => cb(err, stdout)) as unknown as typeof execFile;
    expect(await localGpu(fake(null, 'NVIDIA GeForce RTX 5090, 32607\r\nNVIDIA GeForce RTX 4090, 24564\n'))).toEqual({ name: 'NVIDIA GeForce RTX 5090', totalMiB: 32_607 });
    expect(await localGpu(fake(new Error('ENOENT'), ''))).toBeUndefined();
    expect(await localGpu(fake(null, 'NVIDIA X, [N/A]\n'))).toBeUndefined();
    expect(await localGpu(fake(null, ', 1000\n'))).toBeUndefined();
    expect(await localGpu(fake(null, 'NVIDIA X, 0\n'))).toBeUndefined();
  });

  it('matchGpus: what people type', () => {
    expect(matchGpus(GPUS, 'NVIDIA GeForce RTX 5090').map((g) => g.id)).toEqual(['rtx-5090-32gb']);
    expect(matchGpus(GPUS, 'GeForce RTX 5090').map((g) => g.id)).toEqual(['rtx-5090-32gb']);
    expect(matchGpus(GPUS, 'nvidia RTX-5090').map((g) => g.id)).toEqual(['rtx-5090-32gb']);
    expect(matchGpus(GPUS, '5090').map((g) => g.id)).toEqual(['rtx-5090-32gb']);
    expect(matchGpus(GPUS, 'h100').map((g) => g.id)).toEqual(['h100-pcie-80gb', 'h100-sxm5-80gb', 'h100-nvl-94gb']);
    // The H200s, not the GH200 (which only contains the name); without a card starting so, every match.
    expect(matchGpus(GPUS, 'h200').map((g) => g.id)).toEqual(['h200-sxm-141gb', 'h200-nvl-141gb']);
    expect(matchGpus(GPUS, '200-144').map((g) => g.id)).toEqual(['gh200-144gb']);
    expect(matchGpus(GPUS, 'pro 6000').length).toBe(3);
    expect(matchGpus(GPUS, 'none')).toEqual([]);
  });

  it('customGpu: memory only', () => {
    const g = customGpu('Tesla T4', 15_360);
    expect(g).toMatchObject({ id: 'custom', name: 'Tesla T4', cudaMiB: { min: 14_840, max: 15_344 }, nominalGB: 15, bandwidthGBs: 0, tflops: { bf16: null, fp8: null, fp4: null } });
  });

  it('--details: every number of the option the answer names, then the memory item by item', async () => {
    const r = await run(['Qwen/Qwen3-8B', '--tokens', '32k', '--gpu', '5090', '--details']);
    expect(r.code).toBe(0);
    expect(r.text).toContain('Details — 1× RTX 5090, one request of 32k tokens:');
    expect(r.text).toContain('Memory as the engine counts it (worst–best case):');
    expect(r.text).toMatch(/\n {2}weights {29}15\.3 GiB {2}\(sum of the loaded tensors\)\n/);
    expect(r.text).toMatch(/\n {2}KV cache of one 32k request {9}[\d.]+ GiB\n/);
    expect((await run(['Qwen/Qwen3-8B', '--tokens', '8k'])).text).not.toContain('Details');
    expect((await run(['Qwen/Qwen3-8B', '--tokens', '8k', '--details'])).text).toContain('\nDetails — 1× RTX 3090, one request of 8k tokens:');
  });
});

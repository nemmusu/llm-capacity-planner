/**
 * The command line of the capacity command: its options and the help,
 * written for a reader who is not an expert. COMMAND is the only place that names the command.
 */
import { AppError } from '../contract/errors';
import type { ServingEngine } from '../model/serving-fit';
import { DEFAULT_WORKLOAD } from '../planner/capacity';
import { DECIDE_MAX_ERROR, SATURATION } from './figures';

/** How the help and the hints call the command. */
export const COMMAND = __CLI_NAME__;

export interface PlanArgs {
  /** A Hugging Face repo id, a search, or "report" for the ready-made reports. */
  repo: string;
  report: boolean;
  /** report: only the models whose name contains this. */
  reportFilter?: string;
  /** Tokens of one request's context; undefined = the model's maximum. */
  tokens?: number;
  cache: 'fp8' | 'auto';
  engine: ServingEngine;
  /** Total users to serve: the command recommends hardware for them. */
  users?: number;
  activeShare: number;
  minTokS: number;
  maxTtftS: number;
  /** --max-wait was given: the limit does not grow with the context. */
  ttftSet: boolean;
  typicalTokens?: number;
  outputTokens: number;
  cachedShare: number;
  /** A catalog card ("5090", "h100", "pro-6000"…). */
  gpu?: string;
  count?: number;
  /** A card not in the catalog: GiB of VRAM (memory verdict only, no speed). */
  vramGB?: number;
  util?: number;
  maxBatchedTokens?: number;
  maxSeqs?: number;
  textOnly: boolean;
  variant: string;
  revision: string;
  allHardware: boolean;
  details: boolean;
  /** --this-pc: also this PC's NVIDIA card (asks nvidia-smi); off by default, nothing is run. */
  thisPc: boolean;
  csv?: string;
  /** --csv-excel-eu: ";" and decimal comma (standard CSV otherwise). */
  csvExcelEu: boolean;
  json?: string;
  md?: string;
  offline: boolean;
  /** Read the model again from Hugging Face, ignoring the local cache. */
  refresh: boolean;
  color: boolean;
  ascii: boolean;
}

const c = COMMAND;
const d = DEFAULT_WORKLOAD;

export const USAGE = `Which NVIDIA hardware a Hugging Face model needs with vLLM or SGLang, how many users it serves,
and how fast. Nothing is downloaded or run: only config.json and safetensors headers are read.
No GPU and no inference engine are needed on this computer.

Examples (free models, no license to accept; they work offline from the built-in data):
  ${c} zai-org/GLM-5.3 --tokens 128k --users 200
      long documents (128k tokens ≈ 200 pages), 200 users
  ${c} moonshotai/Kimi-K2.7-Code --tokens 128k --users 300 --active 30
      coding with agents, 300 users at 30% active (agents are busy)
  ${c} zai-org/GLM-5.3 --users 200
      the same 200 users at the model's maximum context (1M tokens)
  ${c} openai/gpt-oss-120b --tokens 32k --users 2000 --prompt-cache 80
      chat, 2,000 users, 80% of each request already in the prefix cache
  ${c} deepseek-ai/DeepSeek-V4-Pro-0813 --tokens 64k --gpu h200 --count 8 --details
      one configuration in detail: 8× H200, every number and its limits
  ${c} zai-org/GLM-5.3 --tokens 128k --users 200 --engine sglang
      the same question for the SGLang engine
  ${c} report glm --csv glm.csv
      ready-made results for recent open models (here: the GLM ones), saved for Excel

The three main options:
  model                the Hugging Face repo id, e.g. zai-org/GLM-5.3 (searched when not exact)
  --tokens N           tokens in one request: question, documents, conversation and answer;
                       e.g. 8k, 32k, 128k, 1M (default: the model's maximum); k = 1,024 tokens,
                       M = 1,048,576 (128k = 131,072 tokens); a token is about 0.75 words
  --cache fp8|auto     the KV cache is the memory the engine keeps for each open conversation;
                       fp8 = half the memory, the usual production choice (default);
                       auto = the model's full precision

Sizing for a team (defaults in brackets):
  --users N            total clients to serve: people, or AI agents/subagents working for them
                       (one person running 5 subagents counts as 5); e.g. --users 200, 2k = 2,000
  --active P           % of users sending a request at the same moment (${d.activeShare * 100}); e.g. --active 20
  --min-tok-s N        slowest acceptable answer speed per user, tokens/s (${d.minTokS}); e.g. --min-tok-s 30
  --max-wait S         longest acceptable wait for the first token under load, seconds
                       (${d.maxTtftMs / 1000}, or, if longer, twice the wait for the first words of one
                       typical request alone); e.g. --max-wait 5
  --prompt-cache P     % of each request already read and reused: system prompt, shared documents,
                       a chat's history (0); e.g. --prompt-cache 80 for a chat
  --typical-context N  tokens of a typical request, at most --tokens (half of --tokens);
                       e.g. --typical-context 8k
  --output-tokens N    tokens of one answer, less than --tokens (${d.outputTokens}); e.g. --output-tokens 1k
  Users include AI agents. Splitting work across parallel agents or subagents gets it done sooner,
  but each one is a concurrent client: 5 subagents count as 5 users. Agents are usually active all
  the time and send long contexts, so raise --active and --tokens for agent workloads.

Hardware:
  --gpu NAME           only the cards whose name contains NAME, in detail ("h100" = its three
                       editions): 3090, 4090, 5080, 5090, a6000, 6000-ada, pro-6000, l4, l40s, a100,
                       h100, gh200, h200, b200, b300 (all = --all-hw)
  --count N            only this many GPUs: 1, 2, 4, 8 or 16 (default: 1, 2, 4 and 8). A server
                       holds 8 data-center GPUs, 4 workstation or GeForce cards, 2 GH200: more
                       GPUs are copies of the model on more servers (16 = two 8-GPU servers,
                       eight GH200 servers)
  --vram GIB           a card outside the catalog, its memory in GiB as nvidia-smi shows it
                       (MiB / 1024): the memory verdict only
  --all-hw             also the table of every layout (tensor or pipeline parallel, copies)
  --this-pc            also this PC's NVIDIA card, marked "this PC" (runs nvidia-smi to read it;
                       without it nothing is run)

Engine:
  --engine vllm|sglang the inference engine (vllm)
  --util F             share of GPU memory the engine takes (vLLM --gpu-memory-utilization 0.9,
                       SGLang --mem-fraction-static)
  --text-only          multimodal models without the vision encoder (vLLM --language-model-only)
  --variant DIR        a sub-folder of the repo with another version, e.g. FP8 (none)
  --revision REV       a branch, tag or commit of the repo (main)
  --max-batched-tokens N  tokens the scheduler reads per step (the engine's default)
  --max-seqs N         requests the scheduler runs at most (the engine's default)

Output:
  --details            every number of the recommended layout, with the roofline check
  --csv FILE           a CSV file: one line per layout; value, min, max, error and basis per number
  --csv-excel-eu       with --csv: ";" and decimal comma, for Excel in comma-decimal locales
  --json FILE          everything as JSON
  --md FILE            the answer as a one-page Markdown summary
  --offline            only models read before (the local cache), no network
  --refresh            read the model again from Hugging Face, ignoring the local cache
  --no-color           no colours
  --ascii              no special characters (old Windows consoles)

How to read the results:
  fits                 the model and one full-context request fit, even with the largest fixed costs
  at the limit         fits only if the fixed costs are at the low end of their range
  does not fit         not enough memory, even with a full server of that card (8, 4 or 2 GPUs,
                       or --count)
  engine does not start  enough memory, but the engine refuses this card or split (reason shown)
  users (guaranteed)   served even if the real speed is at the low end of the estimate
  users (likely)       served at the central estimate (what measurements usually show)
  concurrent requests  requests generating at the same moment; total users = concurrent / --active
  documents / chat     by default each request brings all-new text (documents); with --prompt-cache
                       most of it is already cached (a chat), so more users fit

How to read the accuracy (every number is "value (min to max, ±%)"; memory in GiB = 2^30 bytes;
card memory = what CUDA reports, the engine's usable memory (nvidia-smi shows a little more);
speed in tok/s; time in s):
  exact                computed: weights (sum of the tensors) and KV cache (the engine's rules)
  measured             this model measured on this card with this engine
  calibrated           fitted on measurements of this card, or of the same model type (N points)
  estimated            calibrated on other cards, not verified for this format, or beyond the
                       context measured: the range is wider and the reason is shown
  not precise enough to decide  ±${DECIDE_MAX_ERROR}% or more, or a case the speed model does not cover
  Every number shows its own range. Memory is the most precise; speed and users are less, and at
  ±${DECIDE_MAX_ERROR}% or more they say "not precise enough to decide": rely on memory and on
  guaranteed users. Above ${SATURATION * 100}% of a layout's capacity queues grow fast: "near saturation"
  marks a larger error.
  Method and sources: docs/METHODOLOGY.md

Glossary:
  token                a piece of a word; 1,000 tokens ≈ 750 English words ≈ 1.5 pages
  server               one machine (one node): 8 data-center GPUs, 4 workstation cards or 2 GH200;
                       one copy of the model never spans two servers
  KV cache             memory the server keeps for each open conversation; it grows with the text
  GPUs "4 (1 model on 4)"  one copy of the model split across 4 GPUs that work together on every
                       token (tensor parallel, TP)
  GPUs "8 (2 copies × 4)"  two independent copies of the model, 4 GPUs each, each serving its
                       own users
  "split in 2 stages"  each GPU (or group of GPUs) holds part of the model's layers
                       (pipeline parallel, PP); it splits the conversation memory of MLA models,
                       which TP copies on every GPU, and it suits cards without a fast link (NVLink)
  TTFT                 time to first token: the wait before the answer starts
  tok/s                tokens per second: how fast an answer is written (× 0.75 = words per second)

Gated models: accept the license on huggingface.co, then log in (hf auth login) or set HF_TOKEN.`;

/** Options that take a value. */
export const VALUED = new Set([
  'tokens',
  'cache',
  'engine',
  'users',
  'active',
  'min-tok-s',
  'max-wait',
  'typical-context',
  'output-tokens',
  'prompt-cache',
  'gpu',
  'count',
  'vram',
  'util',
  'variant',
  'revision',
  'csv',
  'json',
  'md',
  'max-batched-tokens',
  'max-seqs',
]);
/** Options without a value. */
export const BOOLEAN = new Set(['text-only', 'all-hw', 'this-pc', 'details', 'csv-excel-eu', 'offline', 'refresh', 'no-color', 'ascii']);

/** "32k" → 32768, "1M" → 1048576, "1.5k" → 1536. */
export function tokensArg(name: string, v: string): number {
  const m = /^(\d+(?:\.\d+)?)([kKmM]?)$/.exec(v.trim());
  if (!m) throw new AppError('VALIDATION', `${name}: expected a number (e.g. 32k), not "${v}"`);
  const mult = m[2] === '' ? 1 : /k/i.test(m[2] as string) ? 1024 : 1024 * 1024;
  const n = Math.round(Number(m[1]) * mult);
  if (n < 1) throw new AppError('VALIDATION', `${name}: must be at least 1`);
  return n;
}

/** A count of people: "200", "2k" = 2,000, "1.5M" = 1,500,000 (k = 1,000, unlike tokens). */
export function countArg(name: string, v: string): number {
  const m = /^(\d+(?:\.\d+)?)([kKmM]?)$/.exec(v.trim());
  if (!m) throw new AppError('VALIDATION', `${name}: expected a number (e.g. 200 or 2k), not "${v}"`);
  const exact = Number(m[1]) * (m[2] === '' ? 1 : /k/i.test(m[2] as string) ? 1000 : 1_000_000);
  // 1.1k is 1,100 (not 1,100.0000000000002); 10.5 is no number of people.
  const n = Math.round(exact);
  if (Math.abs(exact - n) > 1e-6) throw new AppError('VALIDATION', `${name}: a whole number of people, not "${v}"`);
  if (n < 1) throw new AppError('VALIDATION', `${name}: must be at least 1`);
  return n;
}

/** A number within [min, max]; a trailing % is allowed (it is a share in %). */
function numArg(name: string, v: string, min: number, max: number): number {
  const n = Number(v.trim().replace(/%$/, ''));
  if (!Number.isFinite(n) || n < min || n > max) throw new AppError('VALIDATION', `${name}: expected a number from ${min} to ${max}, not "${v}"`);
  return n;
}

export interface ArgsEnv {
  env: Record<string, string | undefined>;
  isTTY: boolean;
  platform: NodeJS.Platform;
}

/** --output-tokens below the context, --typical-context within it (the context known: --tokens or the model's). */
export function checkContext(tokens: number, a: Pick<PlanArgs, 'outputTokens' | 'typicalTokens'>): void {
  const k = (n: number) => n.toLocaleString('en-US');
  if (a.outputTokens >= tokens) throw new AppError('VALIDATION', `--output-tokens (${k(a.outputTokens)}) must be less than the context of ${k(tokens)} tokens: the answer is part of the request`);
  if (a.typicalTokens !== undefined && a.typicalTokens > tokens) throw new AppError('VALIDATION', `--typical-context (${k(a.typicalTokens)}) must be at most the context of ${k(tokens)} tokens`);
}

/** The command line, or undefined for --help / no arguments. Throws AppError VALIDATION. */
export function parseArgs(argv: readonly string[], ctx: ArgsEnv): PlanArgs | undefined {
  const flags = new Map<string, string>();
  const positional: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i] as string;
    if (a === '-h' || a === '--help' || a === 'help') return undefined;
    if (!a.startsWith('--')) {
      positional.push(a);
      continue;
    }
    const eq = a.indexOf('=');
    const name = eq > 0 ? a.slice(2, eq) : a.slice(2);
    if (BOOLEAN.has(name)) {
      flags.set(name, 'true');
      continue;
    }
    if (!VALUED.has(name)) throw new AppError('VALIDATION', `unknown option --${name} (see --help)`);
    if (flags.has(name)) throw new AppError('VALIDATION', `--${name} is given twice: give it once`);
    const value = eq > 0 ? a.slice(eq + 1) : argv[++i];
    if (value === undefined) throw new AppError('VALIDATION', `--${name}: the value is missing`);
    flags.set(name, value);
  }
  if (positional.length === 0) return undefined;
  if (positional.length > 2) throw new AppError('VALIDATION', 'give one model (a Hugging Face repo), e.g. Qwen/Qwen3-8B');
  const get = (k: string) => flags.get(k) as string;
  const cache = flags.get('cache') ?? 'fp8';
  if (cache !== 'fp8' && cache !== 'auto') throw new AppError('VALIDATION', `--cache: fp8 or auto (auto = the model's own precision, e.g. bf16), not "${cache}"`);
  const engine = flags.get('engine') ?? 'vllm';
  if (engine !== 'vllm' && engine !== 'sglang') throw new AppError('VALIDATION', `--engine: vllm or sglang, not "${engine}"`);
  if (flags.has('offline') && flags.has('refresh')) throw new AppError('VALIDATION', '--offline and --refresh exclude each other');
  const repo = positional[0] as string;
  // The tokens may also follow the model: "Qwen/Qwen3-8B 32k".
  const report = repo === 'report';
  const tokens = flags.has('tokens') ? tokensArg('--tokens', get('tokens')) : positional[1] !== undefined && !report ? tokensArg('tokens', positional[1]) : undefined;
  if (flags.has('tokens') && positional[1] !== undefined && !report && tokensArg('tokens', positional[1]) !== tokens)
    throw new AppError('VALIDATION', `the context is given twice: "${positional[1]}" after the model and --tokens ${get('tokens')}; give one`);
  const winLegacy = ctx.platform === 'win32' && !ctx.env.WT_SESSION && !ctx.env.TERM_PROGRAM;
  const out: PlanArgs = {
    repo,
    report,
    cache: cache === 'fp8' ? 'fp8' : 'auto',
    engine,
    activeShare: flags.has('active') ? numArg('--active', get('active'), 0.1, 100) / 100 : DEFAULT_WORKLOAD.activeShare,
    minTokS: flags.has('min-tok-s') ? numArg('--min-tok-s', get('min-tok-s'), 0.1, 10000) : DEFAULT_WORKLOAD.minTokS,
    maxTtftS: flags.has('max-wait') ? numArg('--max-wait', get('max-wait').replace(/s$/, ''), 0.05, 36000) : DEFAULT_WORKLOAD.maxTtftMs / 1000,
    ttftSet: flags.has('max-wait'),
    outputTokens: flags.has('output-tokens') ? tokensArg('--output-tokens', get('output-tokens')) : DEFAULT_WORKLOAD.outputTokens,
    cachedShare: flags.has('prompt-cache') ? numArg('--prompt-cache', get('prompt-cache'), 0, 99) / 100 : 0,
    textOnly: flags.has('text-only'),
    variant: (flags.get('variant') ?? '').replace(/^\/+|\/+$/g, ''),
    revision: flags.get('revision') ?? 'main',
    allHardware: flags.has('all-hw') || flags.get('gpu') === 'all',
    details: flags.has('details'),
    thisPc: flags.has('this-pc'),
    csvExcelEu: flags.has('csv-excel-eu'),
    offline: flags.has('offline'),
    refresh: flags.has('refresh'),
    color: ctx.isTTY && !flags.has('no-color') && ctx.env.NO_COLOR === undefined,
    ascii: flags.has('ascii') || winLegacy,
  };
  if (tokens !== undefined) out.tokens = tokens;
  if (report && positional[1] !== undefined) out.reportFilter = positional[1];
  if (flags.has('users')) out.users = countArg('--users', get('users'));
  if (flags.has('typical-context')) out.typicalTokens = tokensArg('--typical-context', get('typical-context'));
  if (tokens !== undefined) checkContext(tokens, out);
  const gpu = flags.get('gpu');
  if (gpu !== undefined && gpu !== 'all') out.gpu = gpu.toLowerCase();
  if (flags.has('count')) {
    const n = numArg('--count', get('count'), 1, 64);
    if (![1, 2, 4, 8, 16].includes(n)) throw new AppError('VALIDATION', '--count: 1, 2, 4, 8 or 16 GPUs');
    out.count = n;
  }
  if (flags.has('vram')) out.vramGB = numArg('--vram', get('vram'), 1, 4096);
  if (flags.has('util')) out.util = numArg('--util', get('util'), 0.05, 0.99);
  if (flags.has('max-batched-tokens')) out.maxBatchedTokens = tokensArg('--max-batched-tokens', get('max-batched-tokens'));
  if (flags.has('max-seqs')) out.maxSeqs = tokensArg('--max-seqs', get('max-seqs'));
  for (const k of ['csv', 'json', 'md'] as const) if (flags.has(k)) out[k] = get(k);
  if (out.csvExcelEu && out.csv === undefined) throw new AppError('VALIDATION', '--csv-excel-eu changes the --csv file: give --csv FILE too');
  return out;
}

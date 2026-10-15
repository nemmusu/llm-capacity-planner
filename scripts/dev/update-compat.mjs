// Keeps the compatibility matrix of the capacity planner (data/compat.json) in step with the model catalog
// and with new releases of vLLM / SGLang. Plain Node >= 22 on Windows, macOS and Linux; no Python, vLLM or SGLang.
//
//   npm run dev:update-compat -- --models
//       rewrites the "models" section from data/models.json and the fixtures in data/models
//       (features from the architecture and config.json, weight kernels of every checkpoint from its tensor summary).
//   npm run dev:update-compat -- --anchors [--vllm v0.32.0] [--sglang v0.5.22] [--local-vllm DIR] [--local-sglang DIR] [--rebase]
//       checks that every code source still contains its anchor at the cited lines. With --vllm / --sglang the files are
//       read at the new tag instead: an anchor that moved is reported with its new lines, one that vanished means the
//       rule must be verified again. --rebase writes the new tag and lines of every anchor found exactly once (the
//       statuses are not touched: verify them, then update "engines"). Exit code 1 when something must be looked at.
//   npm run dev:update-compat -- --registry [--vllm TAG] [--sglang TAG] [--local-vllm DIR] [--local-sglang DIR]
//       checks that each model's architecture class is registered in each engine (vLLM's registry.py, SGLang's EntryClass).
//   npm run dev:update-compat -- --releases
//       prints the latest releases of vLLM, SGLang and FlashInfer next to the versions the matrix was verified on.
//
// Files are read from GitHub (raw.githubusercontent.com; api.github.com for listings, with GITHUB_TOKEN or GH_TOKEN
// when set) or from local clones (--local-vllm / --local-sglang: a checkout of the tag to check). Nothing else is written.
import { existsSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const dataDir = path.join(root, 'data');
const fixturesRoot = path.join(root, 'data', 'models');
const ENGINE_REPOS = { vllm: 'vllm-project/vllm', sglang: 'sgl-project/sglang' };

// ---------------------------------------------------------------------------------------------------------------
// Model features (pure functions: also used by test/contract/compat.test.ts)
// ---------------------------------------------------------------------------------------------------------------

/** models.json architecture.attention entries → a:* features ("hybrid ..." entries only describe the mix). */
export const ATTENTION_FEATURES = {
  gqa: ['a:gqa'],
  mla: ['a:mla'],
  'sparse (DSA/NSA)': ['a:dsa'],
  'sliding-window': ['a:sliding-window'],
  'linear (Gated DeltaNet)': ['a:gdn'],
  'linear (KDA)': ['a:kda'],
  'sparse (QSA - Qwen Sparse Attention, micro-block level)': ['a:qsa'],
  'sparse (MSA)': ['a:msa'],
  mamba: ['a:mamba2'],
  'hybrid (chunked local attention with RoPE + global attention without RoPE, 3:1)': ['a:chunked-local', 'a:gqa'],
  'hybrid (alternating banded window 128 / full dense)': ['a:sliding-window'],
  'hybrid (local sliding 1024 + global, 5:1, final layer global)': ['a:sliding-window'],
  'hybrid (local sliding 512 + global, 5:1, final layer global)': ['a:sliding-window'],
  'hybrid 6 SWA : 1 global': ['a:sliding-window'],
  'hybrid 3 SWA : 1 full': ['a:sliding-window'],
  'hybrid sliding-window + global': ['a:sliding-window'],
};
/** Classes whose attention is DeepSeek V4's own compressed sparse path (CSA/HCA + SWA, with sinks). */
export const DSV4_CLASSES = ['DeepseekV4ForCausalLM', 'DeepseekV41ForCausalLM'];
/** Weight formats the engines do not load as datacenter checkpoints (llama.cpp, Apple MLX, on-device QAT). */
export const OUT_OF_SCOPE_FORMATS = {
  gguf: 'GGUF is a llama.cpp format; not evaluated for vLLM / SGLang.',
  mlx: 'MLX is an Apple Silicon format, not for NVIDIA GPUs.',
  gemma: 'On-device QAT checkpoint (Gemma format), not for vLLM / SGLang.',
  'mixed:w4a8-int4+int8': 'On-device QAT checkpoint (W4A8); not evaluated for vLLM / SGLang.',
};
/** Precision label (or component of a mixed one) → w:<format>. */
const FORMAT_FEATURE = {
  bf16: 'bf16',
  'fp8-block': 'fp8-block',
  'fp8-channel': 'fp8-channel',
  'fp8-tensor': 'fp8-tensor',
  // quant_method "fp8" without weight_block_size: one scale per tensor.
  fp8: 'fp8-tensor',
  mxfp8: 'mxfp8',
  nvfp4: 'nvfp4',
  mxfp4: 'mxfp4',
  'w4a16-int4': 'int4-w4a16',
  'awq-int4': 'int4-w4a16',
  'gptq-int4': 'gptq-int4',
};
const FOUR_BIT = new Set(['nvfp4', 'mxfp4', 'int4-w4a16', 'gptq-int4']);
const EIGHT_BIT = new Set(['fp8-block', 'fp8-channel', 'fp8-tensor', 'mxfp8']);

function readJsonMaybeGz(file) {
  if (existsSync(file)) return JSON.parse(readFileSync(file, 'utf8'));
  if (existsSync(`${file}.gz`)) return JSON.parse(gunzipSync(readFileSync(`${file}.gz`)).toString('utf8'));
  return null;
}
const fixtureDir = (repo) => repo.replace('/', '__');
const textConfig = (cfg) => cfg?.text_config ?? cfg?.language_config ?? cfg;

/** Linear weights of a tensor summary, split into fused-MoE experts and the rest, with their stored dtypes. */
export function linearWeightDtypes(summary) {
  const out = { moe: new Set(), dense: new Set() };
  for (const e of summary.entries) {
    if (e.group !== 'lm') continue;
    // Step 3.5 stacks its experts in 3-D "moe.*_proj.weight" tensors.
    const expert = /\.experts\./.test(e.pattern) || (/\.moe\./.test(e.pattern) && e.shape.length === 3);
    // Fused experts are stored without ".weight" (gate_up_proj, down_proj; gpt-oss: *_blocks).
    const linear = /(^|\.)(weight|weight_packed|qweight)$/.test(e.pattern) || (expert && /(proj|w[123])(_blocks)?$/.test(e.pattern));
    if (!linear || e.shape.length < 2) continue;
    if (/embed|norm|lm_head|scale|zero|g_idx|bias|conv1d|router|\.gate\.weight$|\.indexer\./.test(e.pattern)) continue;
    out[expert ? 'moe' : 'dense'].add(e.dtype);
  }
  return out;
}

/**
 * The w:* features of one checkpoint. The label names the formats ("mixed:nvfp4+fp8"); the tensor summary, when saved,
 * says which part holds which: packed integers are the 4-bit format, F8_E4M3 the 8-bit one, BF16 stays bf16.
 */
export function variantWeights(format, moe, summary) {
  if (OUT_OF_SCOPE_FORMATS[format]) return { weights: [], outOfScope: OUT_OF_SCOPE_FORMATS[format] };
  const parts = (format.startsWith('mixed:') ? format.slice(6).split('+') : [format]).map((p) => FORMAT_FEATURE[p]);
  if (parts.some((p) => p === undefined)) return { weights: [], unknown: `precision label "${format}" not recognized: the precision could not be read from the config.` };
  const four = parts.find((p) => FOUR_BIT.has(p));
  const eight = parts.find((p) => EIGHT_BIT.has(p));
  const scopes = moe ? ['moe', 'dense'] : ['dense'];
  const found = new Set();
  if (summary) {
    const dtypes = linearWeightDtypes(summary);
    for (const s of scopes)
      for (const d of dtypes[s]) {
        if (['U8', 'I8', 'I32', 'U32'].includes(d)) found.add(`w:${four ?? parts[0]}:${s}`);
        else if (d.startsWith('F8_')) found.add(`w:${eight ?? parts[0]}:${s}`);
        else if (d === 'BF16' || d === 'F16') found.add(`w:${parts.length === 1 && !four && !eight ? parts[0] : 'bf16'}:${s}`);
      }
  } else if (parts.length === 1) {
    for (const s of scopes) found.add(`w:${parts[0]}:${s}`);
  } else {
    // Without the headers: the experts take the first (narrowest) format, the other layers the second.
    if (moe) {
      found.add(`w:${parts[0]}:moe`);
      found.add(`w:${parts[1]}:dense`);
    } else for (const p of parts) found.add(`w:${p}:dense`);
  }
  return { weights: [...found].sort() };
}

/** Compat features of one models.json entry, from its catalog fields, its config.json and its tensor summaries. */
export function deriveCompatModel(entry, fixtures = fixturesRoot) {
  const base = path.join(fixtures, fixtureDir(entry.repo));
  const cfg = readJsonMaybeGz(path.join(base, 'config.json'));
  const text = textConfig(cfg);
  const architectureClass = cfg?.architectures?.[0] ?? null;
  const moe = entry.architecture.kind === 'moe';
  const features = new Set();
  if (DSV4_CLASSES.includes(architectureClass)) features.add('a:dsv4');
  else
    for (const a of entry.architecture.attention) {
      if (a === 'linear (Gated DeltaNet/Lightning/KDA...)') {
        // Ling 3.0: the config says which linear attention (kda_* keys).
        if (Object.keys(text ?? {}).some((k) => k.startsWith('kda_') || k.includes('_kda'))) features.add('a:kda');
        continue;
      }
      for (const f of ATTENTION_FEATURES[a] ?? []) features.add(f);
    }
  if (text && !features.has('a:dsv4')) {
    const heads = [text.head_dim, text.global_head_dim].filter((x) => typeof x === 'number');
    if (!features.has('a:mla') && heads.some((h) => h > 256)) features.add('a:head-dim-512');
    if (architectureClass === 'GptOssForCausalLM' || text.use_sink === true) features.add('a:sinks');
  }
  // DeepSeek V4.1 Engram and Qwen3.8-Flash-Next PLE: n-gram embedding tables vLLM keeps in pinned host memory.
  if ((text?.engram_layer_ids?.length ?? 0) > 0 || (text?.ple_layer_ids?.length ?? 0) > 0) features.add('m:engram');
  const variants = entry.quantizations.map((q) => {
    let summary = null;
    if (q.weightsVariantDir !== null && q.weightsVariantDir !== undefined) {
      const f = path.join(base, q.weightsVariantDir, 'tensor-summary.json');
      if (existsSync(f)) summary = JSON.parse(readFileSync(f, 'utf8'));
    }
    const label = q.kind === 'base' ? (entry.derived?.precision ?? q.format) : q.format;
    const v = variantWeights(label ?? 'unknown', moe, summary);
    return { repo: q.repo, format: label ?? 'unknown', weights: v.weights, ...(v.outOfScope ? { outOfScope: v.outOfScope } : {}), ...(v.unknown ? { weightsUnknown: v.unknown } : {}) };
  });
  return { id: entry.id, architectureClass, moe, features: [...features].sort(), variants };
}

export function deriveCompatModels(catalog, fixtures = fixturesRoot) {
  return catalog.models.map((m) => deriveCompatModel(m, fixtures));
}

// ---------------------------------------------------------------------------------------------------------------
// Reading engine code at a tag (GitHub or a local clone)
// ---------------------------------------------------------------------------------------------------------------

const ghToken = process.env.GITHUB_TOKEN ?? process.env.GH_TOKEN ?? null;
const cache = new Map();

async function fetchText(url, api = false) {
  for (let attempt = 0; attempt < 3; attempt++) {
    const res = await fetch(url, { headers: api && ghToken ? { authorization: `Bearer ${ghToken}`, 'user-agent': 'llm-capacity-planner' } : { 'user-agent': 'llm-capacity-planner' } });
    if (res.ok) return res.text();
    if (res.status === 404) return null;
    await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
  }
  throw new Error(`${url}: cannot be read`);
}

/** The file at repo@ref, from a local clone when one is given for that repo. */
async function readEngineFile(repo, ref, file, locals) {
  const key = `${repo}@${ref}:${file}`;
  if (cache.has(key)) return cache.get(key);
  let text = null;
  const local = locals[repo];
  if (local) {
    const p = path.join(local, ...file.split('/'));
    text = existsSync(p) ? readFileSync(p, 'utf8') : null;
  } else text = await fetchText(`https://raw.githubusercontent.com/${repo}/${ref}/${file}`);
  cache.set(key, text);
  return text;
}

/** 1-based lines whose text contains the anchor. */
export function anchorLines(text, anchor) {
  return text.split(/\r?\n/).flatMap((l, i) => (l.includes(anchor) ? [i + 1] : []));
}

/**
 * One code source checked against a file's text: "ok" when the anchor is inside the cited lines, "moved" with the
 * lines where it now is (same span, shifted so the anchor keeps its offset), "missing" when it is gone.
 * `anchorAt` is the anchor's line in the cited version (null: assume the first line of the range).
 */
export function checkAnchor(source, text, anchorAt = null) {
  if (text === null) return { state: 'missing-file' };
  const hits = anchorLines(text, source.anchor);
  const [a, b] = source.lines;
  if (hits.some((h) => h >= a && h <= b)) return { state: 'ok', lines: [a, b] };
  if (hits.length === 0) return { state: 'missing' };
  const offset = (anchorAt ?? a) - a;
  return { state: 'moved', candidates: hits.map((h) => [Math.max(1, h - offset), Math.max(1, h - offset) + (b - a)]) };
}

function parseArgs(argv) {
  const opts = { flags: new Set(), values: {} };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (['--vllm', '--sglang', '--local-vllm', '--local-sglang'].includes(a)) opts.values[a.slice(2)] = argv[++i];
    else opts.flags.add(a.slice(2));
  }
  return opts;
}

function readCompat() {
  return JSON.parse(readFileSync(path.join(dataDir, 'compat.json'), 'utf8'));
}
function writeCompat(data) {
  writeFileSync(path.join(dataDir, 'compat.json'), `${JSON.stringify(data, null, 2)}\n`);
}

async function checkAnchors(opts) {
  const compat = readCompat();
  const newRef = { [ENGINE_REPOS.vllm]: opts.values.vllm, [ENGINE_REPOS.sglang]: opts.values.sglang };
  const locals = { [ENGINE_REPOS.vllm]: opts.values['local-vllm'], [ENGINE_REPOS.sglang]: opts.values['local-sglang'] };
  const sources = [
    ...compat.rules.flatMap((r) => r.sources.filter((s) => s.kind === 'code').map((s) => ({ s, where: r.id }))),
    ...compat.engines.flatMap((e) => e.dependencies.map((d) => ({ s: d.source, where: `${e.engine} ${d.name}` }))),
  ];
  let problems = 0;
  let rebased = 0;
  for (const { s, where } of sources) {
    const ref = newRef[s.repo] ?? s.ref;
    // A local clone is a checkout of the tag being checked; the cited version, when older, is read from GitHub.
    const text = await readEngineFile(s.repo, ref, s.path, locals);
    let anchorAt = null;
    if (ref !== s.ref) {
      const old = await readEngineFile(s.repo, s.ref, s.path, {});
      anchorAt = old ? (anchorLines(old, s.anchor).find((h) => h >= s.lines[0] && h <= s.lines[1]) ?? null) : null;
    }
    const r = checkAnchor(s, text, anchorAt);
    if (r.state === 'ok' && ref === s.ref) continue;
    if ((r.state === 'ok' || (r.state === 'moved' && r.candidates.length === 1)) && opts.flags.has('rebase') && ref !== s.ref) {
      s.lines = r.state === 'ok' ? r.lines : r.candidates[0];
      s.ref = ref;
      rebased++;
      continue;
    }
    if (r.state === 'ok') continue;
    problems++;
    const at = `${s.repo}@${ref} ${s.path}:${s.lines.join('-')}`;
    if (r.state === 'missing-file') console.log(`[file missing] ${where}: ${at}`);
    else if (r.state === 'missing') console.log(`[anchor gone] ${where}: ${at} — "${s.anchor}": verify the rule again`);
    else console.log(`[moved] ${where}: ${at} — "${s.anchor}" now at lines ${r.candidates.map((c) => c.join('-')).join(', ')}`);
  }
  if (rebased) {
    writeCompat(compat);
    console.log(`${rebased} sources moved to the new tag: update "engines" (version, commit), then verify the status and reason of their rules again.`);
  }
  console.log(`
${sources.length} code sources, ${problems} to look at.`);
  if (problems) process.exitCode = 1;
}

/** Architecture classes registered by an engine at a tag. */
async function registeredClasses(engine, ref, local) {
  if (engine === 'vllm') {
    const text = await readEngineFile(ENGINE_REPOS.vllm, ref, 'vllm/model_executor/models/registry.py', { [ENGINE_REPOS.vllm]: local });
    if (text === null) throw new Error('vLLM registry.py not found');
    return new Set([...text.matchAll(/^\s+"([A-Za-z0-9_]+)":\s*\(/gm)].map((m) => m[1]));
  }
  // SGLang: every module of python/sglang/srt/models exporting EntryClass.
  const dir = 'python/sglang/srt/models';
  let files;
  if (local) files = readdirSync(path.join(local, ...dir.split('/'))).filter((f) => f.endsWith('.py'));
  else {
    const list = await fetchText(`https://api.github.com/repos/${ENGINE_REPOS.sglang}/contents/${dir}?ref=${ref}`, true);
    files = JSON.parse(list).filter((f) => f.name.endsWith('.py')).map((f) => f.name);
  }
  const out = new Set();
  for (const f of files) {
    const text = await readEngineFile(ENGINE_REPOS.sglang, ref, `${dir}/${f}`, { [ENGINE_REPOS.sglang]: local });
    const m = text?.match(/^EntryClass\s*=\s*(\[[\s\S]*?\]|[A-Za-z0-9_]+)/m);
    if (m) for (const c of m[1].matchAll(/[A-Za-z_][A-Za-z0-9_]*/g)) out.add(c[0]);
  }
  return out;
}

async function checkRegistry(opts) {
  const compat = readCompat();
  for (const engine of ['vllm', 'sglang']) {
    const ev = compat.engines.find((e) => e.engine === engine);
    const ref = opts.values[engine] ?? ev.ref;
    const classes = await registeredClasses(engine, ref, opts.values[`local-${engine}`]);
    console.log(`
${engine} ${ref}: ${classes.size} registered classes`);
    for (const m of compat.models) {
      if (!m.architectureClass) {
        console.log(`  ${m.id}: class not read (gated config or Mistral format): check by hand`);
        continue;
      }
      // SGLang maps DeepSeek V4.1 to its V4 class in configs/deepseek_v41.py.
      const cls = engine === 'sglang' && m.architectureClass === 'DeepseekV41ForCausalLM' ? 'DeepseekV4ForCausalLM' : m.architectureClass;
      const rule = compat.rules.find((r) => r.engine === engine && r.feature === 'm:model' && r.platform === 'all' && r.models?.includes(m.id));
      if (!classes.has(cls)) console.log(`  ${m.id}: ${cls} NOT registered${rule ? ` (rule ${rule.id}: ${rule.status})` : ' — an m:model rule is missing'}`);
      else if (rule && rule.status === 'does-not-start') console.log(`  ${m.id}: ${cls} is now registered, but rule ${rule.id} says does-not-start: verify`);
    }
  }
}

async function releases() {
  const compat = readCompat();
  const repos = [
    ['vllm', ENGINE_REPOS.vllm],
    ['sglang', ENGINE_REPOS.sglang],
    ['flashinfer', 'flashinfer-ai/flashinfer'],
  ];
  for (const [name, repo] of repos) {
    const text = await fetchText(`https://api.github.com/repos/${repo}/releases?per_page=5`, true);
    const list = JSON.parse(text ?? '[]').filter((r) => !r.draft);
    const ev = compat.engines.find((e) => e.engine === name);
    const dep = compat.engines.flatMap((e) => e.dependencies.filter((d) => d.name.startsWith(name)).map((d) => `${e.engine}: ${d.version}`));
    console.log(`${name}: verified ${ev ? ev.version : dep.join(', ') || '-'}; latest: ${list.map((r) => `${r.tag_name}${r.prerelease ? ' (pre)' : ''} ${r.published_at.slice(0, 10)}`).join(', ')}`);
  }
}

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.flags.has('models')) {
    const catalog = JSON.parse(readFileSync(path.join(dataDir, 'models.json'), 'utf8'));
    const compat = readCompat();
    compat.models = deriveCompatModels(catalog);
    writeCompat(compat);
    console.log(`compat.json: ${compat.models.length} models, ${compat.models.reduce((n, m) => n + m.variants.length, 0)} checkpoints`);
  } else if (opts.flags.has('anchors')) await checkAnchors(opts);
  else if (opts.flags.has('registry')) await checkRegistry(opts);
  else if (opts.flags.has('releases')) await releases();
  else {
    console.log('Usage: npm run dev:update-compat -- --models | --anchors | --registry | --releases (see the header of this file)');
    process.exitCode = 2;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();

// Updates the capacity planner's model catalog (data/models.json) and its test fixtures
// (data/models/<org>__<repo>/) from Hugging Face. Development tool: plain Node (>= 22),
// no Python, no engines, runs the same on Windows, macOS and Linux.
//
//   npm run dev:update-models --                 every model of the catalog
//   npm run dev:update-models -- --only glm-5-3  one or more ids (comma separated)
//   npm run dev:update-models -- --offline       recompute the "hf"/"derived" blocks from the saved fixtures
//   npm run dev:update-models -- --quants-only   refresh only the list of quantized checkpoints (quant-repos.json)
//   npm run dev:update-models -- --missing       fetch only the models whose fixtures are missing or incomplete
//   npm run dev:update-models -- --variants      also the headers of the official quantized checkpoints
//                                                            (FP8, NVFP4, MXFP4, INT4...) in <model>/variants/<org>__<repo>/
//
// For each repo it saves config.json, generation_config.json, params.json (Mistral format) and the shard index
// as they are on HF at the pinned commit, and the JSON HEADER of every safetensors shard, read with HTTP Range
// requests (8 bytes of length, then the header): the weights are never downloaded. HF_TOKEN is sent when set,
// so gated repos can be read; a repo that cannot be read is written down as such ("readable": false). HTTPS_PROXY is
// honored. The headers are saved compact ({ dtype, shape } per tensor), after checking them against data_offsets.
//
// The curated fields of the catalog (license notes, context from the card, YaRN, engines, quantizations) are
// written by hand with their sources and are never touched here; only "hf" and "derived" are rewritten.
import { mkdirSync, readdirSync, readFileSync, writeFileSync, existsSync, rmSync } from 'node:fs';
import { gzipSync, gunzipSync } from 'node:zlib';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { NOT_REDISTRIBUTED } from '../pack-models.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const catalogPath = path.join(root, 'data', 'models.json');
const fixturesRoot = path.join(root, 'data', 'models');
const HF = 'https://huggingface.co';
/**
 * The Hugging Face token, found as huggingface_hub finds it: HF_TOKEN, then $HF_HOME/token, then
 * ~/.cache/huggingface/token (written by `hf auth login`). Never printed or saved.
 */
function hfToken() {
  if (process.env.HF_TOKEN) return process.env.HF_TOKEN.trim();
  const home = process.env.HF_HOME || path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), '.cache'), 'huggingface');
  for (const file of [path.join(home, 'token'), path.join(os.homedir(), '.cache', 'huggingface', 'token')]) {
    try {
      const t = readFileSync(file, 'utf8').trim();
      if (t) return t;
    } catch {}
  }
  return '';
}
const token = hfToken();

const args = process.argv.slice(2);
const offline = args.includes('--offline');
const quantsOnly = args.includes('--quants-only');
const variants = args.includes('--variants');
const missingOnly = args.includes('--missing');
const onlyIdx = args.indexOf('--only');
const only = onlyIdx >= 0 ? new Set(String(args[onlyIdx + 1] ?? '').split(',').filter(Boolean)) : null;

/** Bytes per element of the safetensors dtypes (packed 4-bit formats are stored as U8/I8). */
export const DTYPE_BYTES = {
  F64: 8, I64: 8, U64: 8, F32: 4, I32: 4, U32: 4, F16: 2, BF16: 2, I16: 2, U16: 2,
  F8_E4M3: 1, F8_E5M2: 1, F8_E8M0: 1, I8: 1, U8: 1, BOOL: 1,
};

export function fixtureDirName(repo) {
  return repo.replace('/', '__');
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Seconds until HF's rate-limit window resets, from its `RateLimit: "resolvers";r=0;t=22` header. */
function rateLimitWait(res) {
  const t = /;t=(\d+)/.exec(res.headers.get('ratelimit') ?? '')?.[1];
  const retry = Number(res.headers.get('retry-after'));
  return t ? Number(t) + 1 : Number.isFinite(retry) && retry > 0 ? retry : 30;
}

async function hfFetch(url, init = {}, attempt = 0) {
  const headers = { ...(init.headers ?? {}) };
  if (token && url.startsWith(HF)) headers.Authorization = `Bearer ${token}`;
  let res;
  try {
    res = await fetch(url, { ...init, headers, redirect: 'follow', signal: AbortSignal.timeout(120_000) });
  } catch (err) {
    if (attempt < 5) {
      await sleep(1000 * 2 ** attempt);
      return hfFetch(url, init, attempt + 1);
    }
    throw err;
  }
  // Anonymous clients get a few thousand file requests every 5 minutes: wait for the window to reset
  // (HF_TOKEN raises the limit).
  if (res.status === 429 && attempt < 30) {
    await res.body?.cancel();
    const wait = rateLimitWait(res);
    if (attempt === 0) process.stdout.write(`  rate limited by Hugging Face, waiting ${wait}s${token ? '' : ' (set HF_TOKEN for a higher limit)'}\n`);
    await sleep(wait * 1000);
    return hfFetch(url, init, attempt + 1);
  }
  if (res.status >= 500 && attempt < 6) {
    await res.body?.cancel();
    await sleep(1500 * 2 ** attempt);
    return hfFetch(url, init, attempt + 1);
  }
  return res;
}

async function fetchJson(url) {
  const res = await hfFetch(url);
  if (!res.ok) {
    await res.body?.cancel();
    return { ok: false, status: res.status };
  }
  return { ok: true, text: await res.text() };
}

async function readRange(url, start, end) {
  const res = await hfFetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (res.status !== 206) {
    // 200 would be the whole file, i.e. the weights: never read it.
    await res.body?.cancel();
    throw new Error(`HTTP ${res.status} for ${url} bytes ${start}-${end}`);
  }
  return Buffer.from(await res.arrayBuffer());
}

/**
 * The JSON header of one safetensors file, with its length. One Range request of `guess` bytes reads the length and,
 * usually, the whole header (shards of a repo have headers of similar size); a second request reads what is missing.
 */
async function readSafetensorsHeader(url, guess = 8) {
  const first = await readRange(url, 0, Math.max(8, guess) - 1);
  const n = Number(first.readBigUInt64LE(0));
  if (!Number.isFinite(n) || n <= 0 || n > 200_000_000) throw new Error(`bad header length ${n} in ${url}`);
  let body = first.subarray(8, 8 + n);
  if (body.length < n) body = Buffer.concat([body, await readRange(url, 8 + body.length, 8 + n - 1)]);
  return { headerBytes: n, header: JSON.parse(body.toString('utf8')) };
}

async function pool(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(
    Array.from({ length: Math.min(limit, items.length) }, async () => {
      while (next < items.length) {
        const i = next++;
        out[i] = await fn(items[i], i);
      }
    }),
  );
  return out;
}

/** Which weight files an engine loads: the HF index first, then a single file, then Mistral's consolidated format. */
export function chooseWeightSet(files, hasConfigJson) {
  const names = new Set(files);
  if (hasConfigJson && names.has('model.safetensors.index.json')) return { format: 'hf', index: 'model.safetensors.index.json' };
  if (hasConfigJson && names.has('model.safetensors')) return { format: 'hf', single: 'model.safetensors' };
  if (names.has('consolidated.safetensors.index.json')) return { format: 'mistral', index: 'consolidated.safetensors.index.json' };
  if (names.has('consolidated.safetensors')) return { format: 'mistral', single: 'consolidated.safetensors' };
  if (names.has('model.safetensors.index.json')) return { format: 'hf', index: 'model.safetensors.index.json' };
  if (names.has('model.safetensors')) return { format: 'hf', single: 'model.safetensors' };
  return null;
}

/** The text part of a multimodal config (text_config, llm_config, language_config) or the config itself. */
export function textConfig(cfg) {
  return cfg?.text_config ?? cfg?.llm_config ?? cfg?.language_config ?? cfg?.thinker_config?.text_config ?? cfg ?? {};
}

/** max_position_embeddings with the key path it was read from (top level first, then the text part). */
export function configMaxPositions(cfg, params) {
  if (cfg) {
    if (typeof cfg.max_position_embeddings === 'number') return { value: cfg.max_position_embeddings, path: 'config.json:max_position_embeddings' };
    for (const k of ['text_config', 'llm_config', 'language_config']) {
      const v = cfg[k]?.max_position_embeddings;
      if (typeof v === 'number') return { value: v, path: `config.json:${k}.max_position_embeddings` };
    }
    if (typeof cfg.max_sequence_length === 'number') return { value: cfg.max_sequence_length, path: 'config.json:max_sequence_length' };
    if (typeof cfg.seq_length === 'number') return { value: cfg.seq_length, path: 'config.json:seq_length' };
  }
  if (params) {
    if (typeof params.max_position_embeddings === 'number') return { value: params.max_position_embeddings, path: 'params.json:max_position_embeddings' };
    if (typeof params.max_seq_len === 'number') return { value: params.max_seq_len, path: 'params.json:max_seq_len' };
  }
  return null;
}

/**
 * rope_scaling / rope_parameters as written in the config, unchanged, with where it is: "rope_parameters" or
 * "text_config.rope_parameters" (vLLM's --hf-overrides and SGLang's --json-model-override-args must put the YaRN
 * block in the same place). Mistral-format checkpoints keep it in params.json under "yarn".
 */
export function configRope(cfg, params) {
  if (cfg) {
    for (const sub of [null, 'text_config', 'llm_config', 'language_config']) {
      const o = sub ? cfg[sub] : cfg;
      for (const key of ['rope_parameters', 'rope_scaling']) {
        if (o?.[key]) return { path: sub ? `${sub}.${key}` : key, value: o[key] };
      }
    }
  }
  if (params?.yarn) return { path: 'params.json:yarn', value: params.yarn };
  return null;
}

/** Where max_position_embeddings is in config.json: top level or the text part (the override goes there too). */
export function maxPositionsPath(cfg) {
  if (typeof cfg?.max_position_embeddings === 'number') return 'max_position_embeddings';
  for (const sub of ['text_config', 'llm_config', 'language_config']) {
    if (typeof cfg?.[sub]?.max_position_embeddings === 'number') return `${sub}.max_position_embeddings`;
  }
  return null;
}

const MTP_RE = /(^|\.)(mtp|nextn|mtp_layers?|multi_token_predict\w*)(\.|_|$)/i;
const VISION_RE = /(^|\.)(visual|vision_tower|vision_model|vision_encoder|vision_embed\w*|vit|siglip\w*|image_encoder|multi_modal_projector|mm_projector|mm_proj|vision_proj\w*|patch_embed|merger|connector|embed_vision)(\.|_|$)/i;
const AUDIO_RE = /(^|\.)(audio_tower|audio_model|audio_encoder|embed_audio|audio_proj\w*|whisper\w*|speech_encoder)(\.|_|$)/i;
const LAYER_RE = /(?:^|\.)layers\.(\d+)\./;

/**
 * Splits the tensors in four groups by name: MTP/draft layers (named so, or decoder layers numbered at or past
 * num_hidden_layers, the DeepSeek/GLM convention), vision encoder, audio encoder, and the language model.
 */
export function classifyTensor(name, numHiddenLayers) {
  if (MTP_RE.test(name)) return 'mtp';
  if (AUDIO_RE.test(name)) return 'audio';
  if (VISION_RE.test(name)) return 'vision';
  const m = LAYER_RE.exec(name);
  if (m && typeof numHiddenLayers === 'number' && Number(m[1]) >= numHiddenLayers && !/vision|visual|audio/i.test(name)) return 'mtp';
  return 'lm';
}

/** Totals from compact headers: exact bytes (shape × dtype, checked against data_offsets when read), elements, per dtype and per group. */
export function summarizeHeaders(shards, numHiddenLayers) {
  const byDtype = {};
  const groups = { lm: { tensors: 0, bytes: 0, elements: 0 }, mtp: { tensors: 0, bytes: 0, elements: 0 }, vision: { tensors: 0, bytes: 0, elements: 0 }, audio: { tensors: 0, bytes: 0, elements: 0 } };
  let tensors = 0;
  let bytes = 0;
  let elements = 0;
  let fileBytes = 0;
  for (const s of Object.values(shards)) {
    fileBytes += s.fileBytes ?? 0;
    for (const [name, t] of Object.entries(s.header)) {
      if (name === '__metadata__') continue;
      const b = tensorBytes(t);
      const e = t.shape.reduce((a, x) => a * x, 1);
      tensors++;
      bytes += b;
      elements += e;
      const d = (byDtype[t.dtype] ??= { tensors: 0, bytes: 0, elements: 0 });
      d.tensors++;
      d.bytes += b;
      d.elements += e;
      const g = groups[classifyTensor(name, numHiddenLayers)];
      g.tensors++;
      g.bytes += b;
      g.elements += e;
    }
  }
  return { shards: Object.keys(shards).length, tensors, tensorBytes: bytes, storageElements: elements, fileBytes, byDtype, groups };
}

/** "0-3,5,7-60" from sorted unique integers. */
function ranges(nums) {
  const out = [];
  for (let i = 0; i < nums.length; i++) {
    let j = i;
    while (j + 1 < nums.length && nums[j + 1] === nums[j] + 1) j++;
    out.push(j > i ? `${nums[i]}-${nums[j]}` : `${nums[i]}`);
    i = j;
  }
  return out.join(',');
}

/**
 * Exact per-kind summary of a checkpoint's tensors, small enough to keep for every quantized variant: tensors whose
 * names differ only by numbers (layer, expert) are one entry with their dtype, shape, count, total bytes, group and
 * the decoder layers they appear in. Sums are exact; only the per-tensor names are lost.
 */
export function tensorSummary(shards, numHiddenLayers) {
  const map = new Map();
  let headerBytes = 0;
  for (const s of Object.values(shards)) {
    headerBytes += s.headerBytes ?? 0;
    for (const [name, t] of Object.entries(s.header)) {
      if (name === '__metadata__') continue;
      const group = classifyTensor(name, numHiddenLayers);
      const pattern = name.replace(/(^|\.)\d+(?=\.|$)/g, '$1{n}');
      const key = `${pattern}|${t.dtype}|${t.shape.join('x')}|${group}`;
      let e = map.get(key);
      if (!e) map.set(key, (e = { pattern, dtype: t.dtype, shape: t.shape, group, count: 0, bytes: 0, layers: new Set() }));
      e.count++;
      e.bytes += tensorBytes(t);
      const m = LAYER_RE.exec(name);
      if (m) e.layers.add(Number(m[1]));
    }
  }
  const entries = [...map.values()]
    .map((e) => ({ ...e, layers: e.layers.size ? ranges([...e.layers].sort((a, b) => a - b)) : null }))
    .sort((a, b) => (a.pattern < b.pattern ? -1 : a.pattern > b.pattern ? 1 : a.dtype < b.dtype ? -1 : a.dtype > b.dtype ? 1 : a.shape.join('x') < b.shape.join('x') ? -1 : 1));
  const w = summarizeHeaders(shards, numHiddenLayers);
  return { shards: w.shards, tensors: w.tensors, tensorBytes: w.tensorBytes, fileBytes: w.fileBytes, headerBytes, entries };
}

/** The same totals as summarizeHeaders, from a tensor summary. */
export function summarizeEntries(summary) {
  const byDtype = {};
  const groups = { lm: { tensors: 0, bytes: 0, elements: 0 }, mtp: { tensors: 0, bytes: 0, elements: 0 }, vision: { tensors: 0, bytes: 0, elements: 0 }, audio: { tensors: 0, bytes: 0, elements: 0 } };
  let elements = 0;
  for (const e of summary.entries) {
    const n = e.shape.reduce((a, x) => a * x, 1) * e.count;
    elements += n;
    const d = (byDtype[e.dtype] ??= { tensors: 0, bytes: 0, elements: 0 });
    d.tensors += e.count;
    d.bytes += e.bytes;
    d.elements += n;
    const g = groups[e.group];
    g.tensors += e.count;
    g.bytes += e.bytes;
    g.elements += n;
  }
  return { shards: summary.shards, tensors: summary.tensors, tensorBytes: summary.tensorBytes, storageElements: elements, fileBytes: summary.fileBytes, byDtype, groups };
}

const SCALE_NAMES = [
  [/\.weight_packed$/, '.weight_scale'],
  [/\.weight$/, '.scale'],
  [/\.weight$/, '.weight_scale'],
  [/_blocks$/, '_scales'],
];

/**
 * 4-bit weights packed two per byte in I8/U8 tensors, recognized from the headers alone (DeepSeek V4 and MiMo
 * declare only "fp8" in config.json): a packed tensor next to a scale tensor with one scale every 32 values stored as
 * E8M0 (F8_E8M0 or U8) is MXFP4; one every 16 stored as F8_E4M3 is NVFP4.
 */
export function inferPackedFormats(summary) {
  const byKey = new Map(summary.entries.map((e) => [`${e.pattern}|${e.group}`, e]));
  const out = {};
  for (const e of summary.entries) {
    if (e.dtype !== 'I8' && e.dtype !== 'U8') continue;
    for (const [re, to] of SCALE_NAMES) {
      if (!re.test(e.pattern)) continue;
      const s = byKey.get(`${e.pattern.replace(re, to)}|${e.group}`);
      if (!s || s.count !== e.count) continue;
      const values = e.shape.reduce((a, x) => a * x, 1) * 2;
      const scales = s.shape.reduce((a, x) => a * x, 1);
      const per = values / scales;
      const format = per === 32 && (s.dtype === 'F8_E8M0' || s.dtype === 'U8') ? 'mxfp4' : per === 16 && s.dtype === 'F8_E4M3' ? 'nvfp4' : null;
      if (!format) continue;
      const o = (out[format] ??= { tensors: 0, bytes: 0, scaleBytes: 0, valuesPerScale: per, scaleDtype: s.dtype });
      o.tensors += e.count;
      o.bytes += e.bytes;
      o.scaleBytes += s.bytes;
      break;
    }
  }
  return out;
}

/** The checkpoint's precision label: its quantization config, plus 4-bit packed weights seen in the headers. */
export function precisionLabel(fromConfig, packed) {
  const parts = new Set(String(fromConfig ?? '').replace(/^mixed:/, '').split('+').filter(Boolean));
  for (const f of Object.keys(packed)) parts.add(f);
  // A 4-bit packed checkpoint stored with BF16 norms/embeddings is not "bf16".
  if (parts.size > 1) parts.delete('bf16');
  const order = ['nvfp4', 'mxfp4', 'w4a16-int4', 'w4a8-int4', 'awq-int4', 'gptq-int4', 'mxfp8', 'fp8-block', 'fp8-channel', 'fp8-tensor', 'fp8', 'int8', 'bf16', 'f16'];
  const list = [...parts].sort((a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99));
  return list.length > 1 ? `mixed:${list.join('+')}` : list[0] ?? null;
}

/** num_hidden_layers of a saved config (text part, or Mistral's params.json). */
function savedNumLayers(dir) {
  const read = (f) => (existsSync(path.join(dir, f)) ? JSON.parse(readFileSync(path.join(dir, f), 'utf8')) : null);
  const t = textConfig(read('config.json'));
  return t.num_hidden_layers ?? t.num_layers ?? t.n_layers ?? read('params.json')?.n_layers;
}

function writeJson(file, value) {
  writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

async function updateOne(entry, catalog) {
  const source = await fetchRepoFixtures(entry.repo, path.join(fixturesRoot, fixtureDirName(entry.repo)));
  await discoverQuantRepos(entry, catalog);
  return source;
}

/**
 * Checks every shard as read, before data_offsets are dropped: each tensor's bytes are shape × dtype size, the
 * tensors follow each other from offset 0 without gaps, and the file is 8 bytes + header + tensors. So the compact
 * headers ({ dtype, shape }) give the exact bytes.
 */
export function checkShards(shards) {
  const problems = [];
  for (const [file, s] of Object.entries(shards)) {
    const ts = Object.entries(s.header).filter(([n]) => n !== '__metadata__').sort((a, b) => a[1].data_offsets[0] - b[1].data_offsets[0]);
    let end = 0;
    for (const [name, t] of ts) {
      const per = DTYPE_BYTES[t.dtype];
      if (per == null) problems.push(`${file}: ${name} has unknown dtype ${t.dtype}`);
      else if (t.data_offsets[1] - t.data_offsets[0] !== t.shape.reduce((a, x) => a * x, 1) * per) problems.push(`${file}: ${name} bytes differ from shape × dtype`);
      if (t.data_offsets[0] !== end) problems.push(`${file}: ${name} starts at ${t.data_offsets[0]}, previous tensor ends at ${end}`);
      end = t.data_offsets[1];
    }
    if (s.fileBytes != null && s.fileBytes !== 8 + s.headerBytes + end) problems.push(`${file}: file is ${s.fileBytes} bytes, header + tensors ${8 + s.headerBytes + end}`);
  }
  return { ok: problems.length === 0, problems };
}

/** The headers without data_offsets: { fileBytes, headerBytes, header: { __metadata__, "<tensor>": { dtype, shape } } }. */
export function compactShards(shards) {
  const out = {};
  for (const [file, s] of Object.entries(shards)) {
    const header = {};
    for (const [name, t] of Object.entries(s.header)) header[name] = name === '__metadata__' ? t : { dtype: t.dtype, shape: t.shape };
    out[file] = { fileBytes: s.fileBytes ?? null, headerBytes: s.headerBytes, header };
  }
  return out;
}

/** Bytes of one tensor of a compact header. */
export function tensorBytes(t) {
  const per = DTYPE_BYTES[t.dtype];
  if (per == null) throw new Error(`unknown safetensors dtype ${t.dtype}`);
  return t.shape.reduce((a, x) => a * x, 1) * per;
}

/**
 * Saves config files, the shard index (not for quantized variants), the compact safetensors headers and the per-kind
 * tensor summary of one repo, at its current commit, in `dir`.
 */
async function fetchRepoFixtures(repo, dir, { variant = false } = {}) {
  mkdirSync(dir, { recursive: true });
  const info = await fetchJson(`${HF}/api/models/${repo}?blobs=true`);
  if (!info.ok) throw new Error(`${repo}: model info HTTP ${info.status}`);
  const j = JSON.parse(info.text);
  const sha = j.sha;
  const files = (j.siblings ?? []).map((s) => s.rfilename);
  const sizes = Object.fromEntries((j.siblings ?? []).map((s) => [s.rfilename, s.size ?? s.lfs?.size ?? null]));
  const source = {
    repo,
    revision: sha,
    fetchedAt: new Date().toISOString(),
    createdAt: j.createdAt ?? null,
    lastModified: j.lastModified ?? null,
    gated: j.gated ?? false,
    license: j.cardData?.license ?? null,
    licenseName: j.cardData?.license_name ?? null,
    licenseLink: j.cardData?.license_link ?? null,
    pipelineTag: j.pipeline_tag ?? null,
    hfSafetensorsTotal: j.safetensors?.total ?? null,
    readable: true,
    unreadable: [],
    files: [],
  };
  // A repo whose licence may not allow passing its files on keeps only this record: nothing of it is saved.
  if (repo in NOT_REDISTRIBUTED) {
    for (const f of readdirSync(dir)) if (f !== 'source.json' && f !== 'quant-repos.json' && f !== 'variants') rmSync(path.join(dir, f), { force: true });
    Object.assign(source, { readable: false, withheld: NOT_REDISTRIBUTED[repo] });
    writeJson(path.join(dir, 'source.json'), source);
    return source;
  }
  const resolve = (f) => `${HF}/${repo}/resolve/${sha}/${f.split('/').map(encodeURIComponent).join('/')}`;
  const saved = {};
  for (const f of ['config.json', 'generation_config.json', 'params.json']) {
    if (!files.includes(f)) {
      rmSync(path.join(dir, f), { force: true });
      continue;
    }
    const r = await fetchJson(resolve(f));
    if (!r.ok) {
      source.readable = false;
      source.unreadable.push({ file: f, status: r.status });
      continue;
    }
    saved[f] = JSON.parse(r.text);
    // ModelOpt variants list every quantized layer in config.json (up to 7 MB): those are saved gzipped.
    if (variant && r.text.length > 1_000_000) {
      rmSync(path.join(dir, f), { force: true });
      writeFileSync(path.join(dir, `${f}.gz`), gzipSync(Buffer.from(r.text), { level: 9 }));
      source.files.push(`${f}.gz`);
    } else {
      rmSync(path.join(dir, `${f}.gz`), { force: true });
      writeFileSync(path.join(dir, f), r.text.endsWith('\n') ? r.text : `${r.text}\n`);
      source.files.push(f);
    }
  }
  const set = chooseWeightSet(files, files.includes('config.json'));
  let shardFiles = [];
  if (set?.index) {
    const r = await fetchJson(resolve(set.index));
    if (!r.ok) {
      source.readable = false;
      source.unreadable.push({ file: set.index, status: r.status });
    } else {
      // Indexes of big MoE repos weigh megabytes: saved gzipped, byte for byte.
      if (!variant) {
        writeFileSync(path.join(dir, `${set.index}.gz`), gzipSync(Buffer.from(r.text), { level: 9 }));
        source.files.push(`${set.index}.gz`);
      }
      shardFiles = [...new Set(Object.values(JSON.parse(r.text).weight_map))].sort();
    }
  } else if (set?.single) shardFiles = [set.single];
  source.weightFormat = set?.format ?? null;
  // Safetensors files of the repo an engine does not load with this weight set (e.g. Mistral's duplicate format).
  source.otherSafetensors = shardFiles.length ? files.filter((f) => f.endsWith('.safetensors') && !shardFiles.includes(f)) : [];

  const shards = {};
  if (shardFiles.length && source.readable) {
    let done = 0;
    // The first shard is read alone to learn the header size; the others ask for a bit more than the largest seen.
    let largest = 0;
    const one = async (f) => {
      try {
        const h = await readSafetensorsHeader(resolve(f), largest ? Math.ceil(largest * 1.25) + 4096 : 8);
        largest = Math.max(largest, h.headerBytes + 8);
        done++;
        if (done % 25 === 0) process.stdout.write(`  ${repo}: ${done}/${shardFiles.length} headers\n`);
        return [f, { headerBytes: h.headerBytes, fileBytes: sizes[f] ?? null, header: h.header }];
      } catch (err) {
        return [f, { error: String(err.message ?? err) }];
      }
    };
    const results = [await one(shardFiles[0]), ...(await pool(shardFiles.slice(1), 8, one))];
    for (const [f, v] of results) {
      if (v.error) {
        source.readable = false;
        source.unreadable.push({ file: f, error: v.error });
      } else shards[f] = v;
    }
  }
  const headersFile = path.join(dir, 'safetensors-headers.json.gz');
  const summaryFile = path.join(dir, 'tensor-summary.json');
  if (Object.keys(shards).length && source.readable) {
    source.headerChecks = checkShards(shards);
    const compact = compactShards(shards);
    writeFileSync(headersFile, gzipSync(Buffer.from(JSON.stringify({ repo, revision: sha, weightFormat: source.weightFormat, shards: compact })), { level: 9 }));
    source.files.push('safetensors-headers.json.gz');
    writeJson(summaryFile, { repo, revision: sha, weightFormat: source.weightFormat, ...tensorSummary(compact, savedNumLayers(dir)) });
    source.files.push('tensor-summary.json');
  } else {
    rmSync(headersFile, { force: true });
    rmSync(summaryFile, { force: true });
  }
  writeJson(path.join(dir, 'source.json'), source);
  return source;
}

/**
 * The quantized checkpoints whose headers are saved too (in variants/<org>__<repo>/): the official ones (lab, NVIDIA)
 * for vLLM/SGLang, not drafts, GGUF, MLX, or other models published under the same name (fine-tunes, VL, preview).
 */
export function selectVariants(quants) {
  return (quants?.repos ?? []).filter(
    (r) => r.kind === 'official' && !r.draft && !r.gguf && !r.mlx && r.precision != null && !/-(Fin|VL|singprobe|GenRM|preview|unquantized|mobile)/i.test(r.repo),
  );
}

async function updateVariants(entry) {
  const base = path.join(fixturesRoot, fixtureDirName(entry.repo));
  const quants = existsSync(path.join(base, 'quant-repos.json')) ? JSON.parse(readFileSync(path.join(base, 'quant-repos.json'), 'utf8')) : null;
  for (const v of selectVariants(quants)) {
    const dir = path.join(base, 'variants', fixtureDirName(v.repo));
    if (missingOnly && existsSync(path.join(dir, 'source.json'))) {
      const s = JSON.parse(readFileSync(path.join(dir, 'source.json'), 'utf8'));
      if (s.readable ? existsSync(path.join(dir, 'safetensors-headers.json.gz')) : s.gated && !token) continue;
    }
    process.stdout.write(`  variant ${v.repo}\n`);
    try {
      const s = await fetchRepoFixtures(v.repo, dir, { variant: true });
      process.stdout.write(`    ${s.readable ? 'ok' : 'NOT READABLE'}\n`);
    } catch (err) {
      process.stdout.write(`    ERROR ${err.message ?? err}\n`);
    }
  }
}

/** Organizations whose checkpoints count as official: the lab itself and NVIDIA (ModelOpt NVFP4/FP8). */
const OFFICIAL_QUANT_ORGS = ['nvidia'];
/** Third parties listed for reference, never as official (Red Hat AI publishes vLLM-validated checkpoints). */
const PARTNER_QUANT_ORGS = ['RedHatAI'];
// Draft models for speculative decoding (Gemma's "-assistant" checkpoints are Gemma4AssistantForCausalLM drafters).
const DRAFT_RE = /(eagle|dspark|dflash|speculator|mtp|draft|assistant)/i;

/** One weight scheme of a quantization group: nvfp4 (fp4, groups of 16), mxfp4 (groups of 32), fp8-block/channel/tensor, int4, int8. */
function weightScheme(w, inputs) {
  if (!w) return null;
  if (w.type === 'float' && w.num_bits === 4) return w.group_size === 32 ? 'mxfp4' : 'nvfp4';
  if (w.type === 'float' && w.num_bits === 8) {
    if (w.strategy === 'block' || w.block_structure) return 'fp8-block';
    if (w.strategy === 'channel') return 'fp8-channel';
    if (w.strategy === 'tensor') return 'fp8-tensor';
    return 'fp8';
  }
  if (w.type === 'int' && w.num_bits === 4) return inputs?.num_bits === 8 ? 'w4a8-int4' : 'w4a16-int4';
  if (w.type === 'int' && w.num_bits === 8) return 'int8';
  return null;
}

/**
 * The precision of a checkpoint from its quantization_config (config.json, or params.json for the Mistral format)
 * and hf_quant_config.json (ModelOpt). Every quantization group counts: a checkpoint with NVFP4 experts and FP8
 * attention is "mixed:nvfp4+fp8-…". Without a quantization config, the stored dtype (bf16/f16).
 */
export function classifyQuantization(qc, hfQuant, dtypes) {
  const found = new Set();
  const algos = [hfQuant?.quantization?.quant_algo, qc?.quant_algo, qc?.moe_quant_algo, qc?.format, qc?.fmt]
    .filter(Boolean)
    .map((x) => String(x).toUpperCase());
  // ModelOpt MIXED_PRECISION lists the algorithm of every layer; Ling's config names the experts' own one.
  for (const l of Object.values(hfQuant?.quantization?.quantized_layers ?? qc?.quantized_layers ?? {})) if (l?.quant_algo) algos.push(String(l.quant_algo).toUpperCase());
  // (DeepSeek's "expert_dtype": "fp4" does not say which FP4: inferPackedFormats reads it from the headers.)
  if (qc?.routed_experts_quant_method) algos.push(String(qc.routed_experts_quant_method).toUpperCase());
  for (const g of Object.values(qc?.config_groups ?? {})) {
    const sch = weightScheme(g?.weights, g?.input_activations);
    if (sch) found.add(sch);
    else if (String(g?.format ?? '').includes('nvfp4')) found.add('nvfp4');
  }
  for (const a of algos) {
    if (a.includes('NVFP4')) found.add('nvfp4');
    else if (a.includes('MXFP4')) found.add('mxfp4');
    else if (a.includes('MXFP8')) found.add('mxfp8');
    else if (a === 'FP8_PB_WO' || a.includes('BLOCK')) found.add('fp8-block');
    else if (a === 'FP8' && !found.size) found.add('fp8');
  }
  const method = String(qc?.quant_method ?? '').toLowerCase();
  // quant_method "fp8" covers the layers not given another scheme (e.g. attention next to FP4 experts).
  if (method === 'fp8' && !(algos.includes('MIXED_PRECISION') && found.size)) found.add(qc.weight_block_size ? 'fp8-block' : 'fp8');
  if (method === 'mxfp4') found.add('mxfp4');
  if (method === 'awq') found.add(`awq-int${qc.bits ?? qc.w_bit ?? 4}`);
  if (method === 'gptq') found.add(`gptq-int${qc.bits ?? 4}`);
  if (method === 'bitsandbytes') found.add('bnb');
  // "fp8" without a strategy is redundant next to a more precise fp8 variant.
  if (found.has('fp8') && [...found].some((x) => x.startsWith('fp8-'))) found.delete('fp8');
  if (found.size === 1) return [...found][0];
  if (found.size > 1) {
    const order = ['nvfp4', 'mxfp4', 'w4a16-int4', 'w4a8-int4', 'awq-int4', 'gptq-int4', 'mxfp8', 'fp8-block', 'fp8-channel', 'fp8-tensor', 'fp8', 'int8'];
    return `mixed:${[...found].sort((x, y) => order.indexOf(x) - order.indexOf(y)).join('+')}`;
  }
  if (method) return method;
  const d = Object.keys(dtypes ?? {});
  if (d.includes('BF16')) return 'bf16';
  if (d.includes('F16')) return 'f16';
  return null;
}

/**
 * Finds the quantized checkpoints of a model published by the lab, by NVIDIA, and (as third parties) by Red Hat AI:
 * repos whose name is the model's name plus a suffix. Saves them, with their precision read from their config, in
 * quant-repos.json beside the fixtures.
 */
async function discoverQuantRepos(entry, catalog) {
  const [org, name] = entry.repo.split('/');
  const stem = name.replace(/-BF16$/i, '');
  const otherStems = catalog.models.filter((m) => m.repo !== entry.repo).map((m) => m.repo.split('/')[1].replace(/-BF16$/i, ''));
  const res = await fetchJson(`${HF}/api/models?search=${encodeURIComponent(stem)}&limit=500&full=true`);
  if (!res.ok) return;
  const found = [];
  for (const m of JSON.parse(res.text)) {
    const [o, n] = m.id.split('/');
    if (m.id === entry.repo) continue;
    if (!n.toLowerCase().startsWith(stem.toLowerCase())) continue;
    const rest = n.slice(stem.length);
    if (rest && !rest.startsWith('-')) continue;
    if (/base/i.test(rest)) continue;
    const kind = o === org || OFFICIAL_QUANT_ORGS.includes(o) ? 'official' : PARTNER_QUANT_ORGS.includes(o) ? 'third-party' : null;
    if (!kind) continue;
    // Another model of the catalog (GLM-5.3-Flash is not a variant of GLM-5.3).
    if (otherStems.some((o) => o.length > stem.length && n.toLowerCase().startsWith(o.toLowerCase()))) continue;
    found.push({ repo: m.id, kind, draft: DRAFT_RE.test(rest), gguf: /gguf/i.test(rest), mlx: /mlx/i.test(rest) });
  }
  const out = [];
  for (const q of found.sort((a, b) => a.repo.localeCompare(b.repo))) {
    const info = await fetchJson(`${HF}/api/models/${q.repo}`);
    if (!info.ok) {
      out.push({ ...q, readable: false, status: info.status });
      continue;
    }
    const j = JSON.parse(info.text);
    const files = (j.siblings ?? []).map((s) => s.rfilename);
    let qc = j.config?.quantization_config ?? null;
    let hfQuant = null;
    if (files.includes('config.json')) {
      const c = await fetchJson(`${HF}/${q.repo}/resolve/${j.sha}/config.json`);
      if (c.ok) {
        const cfg = JSON.parse(c.text);
        qc = cfg.quantization_config ?? textConfig(cfg).quantization_config ?? null;
      }
    }
    if (!qc && files.includes('params.json')) {
      const pj = await fetchJson(`${HF}/${q.repo}/resolve/${j.sha}/params.json`);
      if (pj.ok) qc = JSON.parse(pj.text).quantization_config ?? JSON.parse(pj.text).quantization ?? null;
    }
    if (files.includes('hf_quant_config.json')) {
      const h = await fetchJson(`${HF}/${q.repo}/resolve/${j.sha}/hf_quant_config.json`);
      if (h.ok) hfQuant = JSON.parse(h.text);
    }
    out.push({
      ...q,
      revision: j.sha,
      createdAt: j.createdAt ?? null,
      gated: j.gated ?? false,
      architectures: j.config?.architectures ?? null,
      precision: q.gguf ? 'gguf' : q.mlx ? 'mlx' : q.draft ? null : classifyQuantization(qc, hfQuant, j.safetensors?.parameters),
      safetensorsTotal: j.safetensors?.total ?? null,
      safetensorsDtypes: j.safetensors?.parameters ?? null,
      quantizationConfig: qc ? { quant_method: qc.quant_method ?? null, format: qc.format ?? qc.fmt ?? null, weight_block_size: qc.weight_block_size ?? null, quant_algo: qc.quant_algo ?? null } : null,
      hfQuantAlgo: hfQuant?.quantization?.quant_algo ?? null,
    });
  }
  writeJson(path.join(fixturesRoot, fixtureDirName(entry.repo), 'quant-repos.json'), { repo: entry.repo, fetchedAt: new Date().toISOString(), repos: out });
}

/** Rebuilds the "hf" and "derived" blocks of one entry from its saved fixtures. */
export function deriveFromFixtures(entry) {
  const dir = path.join(fixturesRoot, fixtureDirName(entry.repo));
  const rel = path.relative(root, dir).split(path.sep).join('/');
  const read = (f) => (existsSync(path.join(dir, f)) ? JSON.parse(readFileSync(path.join(dir, f), 'utf8')) : null);
  const source = read('source.json');
  if (!source) return { hf: null, derived: null };
  const cfg = read('config.json');
  const params = read('params.json');
  const t = textConfig(cfg);
  const numHiddenLayers = t.num_hidden_layers ?? t.num_layers ?? t.n_layers ?? params?.n_layers;
  const hf = {
    revision: source.revision,
    createdAt: source.createdAt,
    gated: source.gated,
    license: source.license,
    licenseName: source.licenseName,
    readable: source.readable,
    weightFormat: source.weightFormat,
    fixtures: rel,
    ...(source.withheld ? { withheld: source.withheld } : {}),
  };
  const maxPos = configMaxPositions(cfg, params);
  const derived = {
    architectures: cfg?.architectures ?? null,
    modelType: cfg?.model_type ?? null,
    textModelType: t?.model_type ?? null,
    numHiddenLayers: numHiddenLayers ?? null,
    configMaxPositions: maxPos?.value ?? null,
    configMaxPositionsPath: maxPos?.path ?? null,
    configRope: configRope(cfg, params)?.value ?? null,
    configRopePath: configRope(cfg, params)?.path ?? null,
    configMaxPositionsKey: maxPositionsPath(cfg),
    configMtpLayers: t.num_nextn_predict_layers ?? t.mtp_num_hidden_layers ?? t.num_mtp_layers ?? cfg?.num_nextn_predict_layers ?? cfg?.mtp_num_hidden_layers ?? null,
    configHasVision: Boolean(cfg?.vision_config ?? cfg?.vision_encoder ?? params?.vision_encoder),
    configQuantization: cfg?.quantization_config ? { quant_method: cfg.quantization_config.quant_method ?? null, fmt: cfg.quantization_config.fmt ?? cfg.quantization_config.format ?? null, weight_block_size: cfg.quantization_config.weight_block_size ?? null } : null,
    weights: null,
  };
  const gz = path.join(dir, 'safetensors-headers.json.gz');
  if (existsSync(gz)) {
    const payload = JSON.parse(gunzipSync(readFileSync(gz)).toString('utf8'));
    derived.weights = summarizeHeaders(payload.shards, numHiddenLayers);
    // A shard index can name the wrong file for a tensor (granite-4.2-8b: 241 of 363): engines read every listed file's
    // own header, so only the count is kept, as a warning not to trust weight_map for placement.
    const indexGz = ['model.safetensors.index.json.gz', 'consolidated.safetensors.index.json.gz'].map((f) => path.join(dir, f)).find((f) => existsSync(f));
    if (indexGz) {
      const where = new Map();
      for (const [f, sh] of Object.entries(payload.shards)) for (const n of Object.keys(sh.header)) if (n !== '__metadata__') where.set(n, f);
      const wm = JSON.parse(gunzipSync(readFileSync(indexGz)).toString('utf8')).weight_map;
      derived.weights.indexWrongFile = Object.entries(wm).filter(([n, f]) => where.get(n) !== f).length;
    }
  }
  // The precision of the checkpoint itself (GLM-5.3 and DeepSeek V4 are published in FP8/FP4, not BF16).
  const baseSummary = read('tensor-summary.json');
  derived.packedWeights = baseSummary ? inferPackedFormats(baseSummary) : null;
  derived.precision = derived.weights
    ? precisionLabel(classifyQuantization(cfg?.quantization_config ?? t.quantization_config ?? params?.quantization_config ?? null, null, derived.weights.byDtype), derived.packedWeights ?? {})
    : null;
  const quants = read('quant-repos.json');
  derived.variants = selectVariants(quants).map((v) => {
    const vdir = path.join(dir, 'variants', fixtureDirName(v.repo));
    const vread = (f) => (existsSync(path.join(vdir, f)) ? JSON.parse(readFileSync(path.join(vdir, f), 'utf8')) : null);
    const vs = vread('source.json');
    const summary = vread('tensor-summary.json');
    return {
      repo: v.repo,
      precision: v.precision,
      revision: vs?.revision ?? null,
      readable: vs?.readable ?? false,
      fixtures: path.relative(root, vdir).split(path.sep).join('/'),
      weights: summary ? summarizeEntries(summary) : null,
      packedWeights: summary ? inferPackedFormats(summary) : null,
    };
  });
  derived.quantRepos = quants
    ? quants.repos.map((q) => ({ repo: q.repo, kind: q.kind, draft: q.draft, precision: q.precision ?? null, architectures: q.architectures ?? null, gated: q.gated ?? false, revision: q.revision ?? null }))
    : null;
  return { hf, derived };
}

/** Fixtures already saved and readable (headers present), or a gated repo that stays unreadable without HF_TOKEN. */
function isComplete(entry) {
  const dir = path.join(fixturesRoot, fixtureDirName(entry.repo));
  const file = path.join(dir, 'source.json');
  if (!existsSync(file)) return false;
  const s = JSON.parse(readFileSync(file, 'utf8'));
  if (s.readable) return existsSync(path.join(dir, 'safetensors-headers.json.gz'));
  return Boolean(s.gated) && !token;
}

async function main() {
  const catalog = JSON.parse(readFileSync(catalogPath, 'utf8'));
  const targets = catalog.models.filter((m) => !only || only.has(m.id));
  if (only && targets.length !== only.size) throw new Error(`unknown id in --only: ${[...only].filter((id) => !catalog.models.some((m) => m.id === id)).join(', ')}`);
  for (const entry of targets) {
    if (quantsOnly) {
      process.stdout.write(`▶ ${entry.repo} (quantized repos)\n`);
      await discoverQuantRepos(entry, catalog);
    } else if (variants) {
      process.stdout.write(`▶ ${entry.repo} (quantized variants)\n`);
      await updateVariants(entry);
    } else if (missingOnly && isComplete(entry)) {
      process.stdout.write(`· ${entry.repo} already complete\n`);
    } else if (!offline) {
      process.stdout.write(`▶ ${entry.repo}\n`);
      try {
        const s = await updateOne(entry, catalog);
        process.stdout.write(`  ${s.readable ? 'ok' : 'NOT READABLE'} (${s.files.length} files, rev ${s.revision.slice(0, 8)})\n`);
      } catch (err) {
        process.stdout.write(`  ERROR ${err.message ?? err}\n`);
        continue;
      }
    }
    const { hf, derived } = deriveFromFixtures(entry);
    entry.hf = hf;
    entry.derived = derived;
  }
  catalog.generatedAt = new Date().toISOString();
  writeJson(catalogPath, catalog);
  process.stdout.write(`catalog written: ${path.relative(root, catalogPath)}\n`);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  // Node's fetch ignores HTTPS_PROXY unless NODE_USE_ENV_PROXY=1 (Node >= 22.21 / 24.5): rerun with it set.
  const proxy = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
  if (proxy && !process.env.NODE_USE_ENV_PROXY) {
    const r = spawnSync(process.execPath, process.argv.slice(1), { stdio: 'inherit', env: { ...process.env, NODE_USE_ENV_PROXY: '1' } });
    process.exit(r.status ?? 1);
  }
  await main();
}

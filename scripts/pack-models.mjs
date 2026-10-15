// Packs the catalog models of data/models/ (config.json, shard index, safetensors headers) into one file the
// command reads instead of Hugging Face (src/cli/builtin.ts). Run by build.mjs (out/) and dist.mjs (dist/, attached
// to each release). Gated or unreadable repos are left out, and so are those whose licence makes redistribution
// doubtful (NOT_REDISTRIBUTED): only public metadata that may be shared is bundled.
//
//   node scripts/pack-models.mjs <out-file>
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { gunzipSync, gzipSync } from 'node:zlib';
import { root } from './cli-name.mjs';

const MAGIC = 'MODELS01';

/** Public repos left out of the pack: their licence may not allow passing their files on (see NOTICE). */
export const NOT_REDISTRIBUTED = {
  'mistralai/Mistral-Medium-3.5-128B': 'Modified MIT licence: no rights at all for companies above US$20M monthly revenue',
  'nvidia/Mistral-Medium-3.5-128B-NVFP4': 'a quantization of Mistral Medium 3.5, under its licence',
};
const JSON_FILES = /^(config|generation_config|params)\.json(\.gz)?$|\.safetensors\.index\.json(\.gz)?$/;

/** Every folder with a source.json under data/models (models and their quantized variants). */
function repoDirs(dir) {
  const out = existsSync(path.join(dir, 'source.json')) ? [dir] : [];
  for (const e of readdirSync(dir, { withFileTypes: true })) if (e.isDirectory()) out.push(...repoDirs(path.join(dir, e.name)));
  return out;
}
const readMaybeGz = (file) => {
  const raw = readFileSync(file);
  return JSON.parse((file.endsWith('.gz') ? gunzipSync(raw) : raw).toString('utf8'));
};

/** Fingerprint of the files a pack is made of (paths and bytes under `modelsDir`, and the repos left out), the first 12 hex digits. */
export function modelsHash(modelsDir = path.join(root, 'data', 'models')) {
  const h = createHash('sha256');
  // What is left out changes the pack too.
  h.update(`${Object.keys(NOT_REDISTRIBUTED).sort().join('\n')}\n`);
  const walk = (rel) => {
    for (const e of readdirSync(path.join(modelsDir, rel), { withFileTypes: true }).sort((a, b) => (a.name < b.name ? -1 : 1))) {
      const p = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) walk(p);
      else h.update(`${p}\n`).update(readFileSync(path.join(modelsDir, p))).update('\n');
    }
  };
  walk('');
  return h.digest('hex').slice(0, 12);
}

export function packModels(modelsDir = path.join(root, 'data', 'models')) {
  const repos = {};
  const blobs = [];
  let offset = 0;
  let date = '';
  for (const dir of repoDirs(modelsDir).sort()) {
    const source = JSON.parse(readFileSync(path.join(dir, 'source.json'), 'utf8'));
    const headersFile = path.join(dir, 'safetensors-headers.json.gz');
    if (!source.readable || source.gated || !existsSync(headersFile) || source.repo in NOT_REDISTRIBUTED) continue;
    const json = {};
    for (const f of readdirSync(dir)) if (JSON_FILES.test(f)) json[f.replace(/\.gz$/, '')] = readMaybeGz(path.join(dir, f));
    const headers = readMaybeGz(headersFile);
    const shards = {};
    for (const [file, s] of Object.entries(headers.shards)) shards[file] = { fileBytes: s.fileBytes, header: s.header };
    const blob = gzipSync(Buffer.from(JSON.stringify({ json, shards })), { level: 9 });
    repos[source.repo] = {
      revision: source.revision,
      gated: source.gated,
      pipelineTag: source.pipelineTag ?? null,
      total: source.hfSafetensorsTotal ?? null,
      offset,
      length: blob.length,
    };
    blobs.push(blob);
    offset += blob.length;
    if (source.fetchedAt > date) date = source.fetchedAt;
  }
  const index = Buffer.from(JSON.stringify({ date: date.slice(0, 10), source: modelsHash(modelsDir), repos }));
  const head = Buffer.alloc(12);
  head.write(MAGIC, 0, 'ascii');
  head.writeUInt32LE(index.length, 8);
  return { pack: Buffer.concat([head, index, ...blobs]), count: Object.keys(repos).length };
}

if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(import.meta.filename)) {
  const out = process.argv[2];
  if (!out) throw new Error('usage: node scripts/pack-models.mjs <out-file>');
  const { pack, count } = packModels();
  mkdirSync(path.dirname(out), { recursive: true });
  writeFileSync(out, pack);
  console.log(`${path.relative(root, out)}: ${count} repos, ${(pack.length / 1e6).toFixed(1)} MB`);
}

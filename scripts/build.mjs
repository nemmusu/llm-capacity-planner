// Bundles the command into one ES module for development, with the pack of the catalog models next to it.
//   npm run build                    -> out/<name>.mjs and out/<name>-models.pack
//   npm run plan -- Qwen/Qwen3-8B    -> build, then run it with the arguments after "--"
// The pack is rebuilt only when it is missing or older than data/models (or with --repack).
import { build } from 'esbuild';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { cliName, define, root } from './cli-name.mjs';
import { packModels } from './pack-models.mjs';

const runIt = process.argv.includes('--run');
const sep = process.argv.indexOf('--');
const args = sep >= 0 ? process.argv.slice(sep + 1) : [];
const outfile = path.join(root, 'out', `${cliName}.mjs`);
const packFile = path.join(root, 'out', `${cliName}-models.pack`);

/** Newest modification time under a folder. */
function newest(dir) {
  let t = statSync(dir).mtimeMs;
  for (const e of readdirSync(dir, { withFileTypes: true })) t = Math.max(t, e.isDirectory() ? newest(path.join(dir, e.name)) : statSync(path.join(dir, e.name)).mtimeMs);
  return t;
}
const models = path.join(root, 'data', 'models');
if (process.argv.includes('--repack') || !existsSync(packFile) || statSync(packFile).mtimeMs < newest(models)) {
  const { pack, count } = packModels(models);
  mkdirSync(path.dirname(packFile), { recursive: true });
  writeFileSync(packFile, pack);
  if (!runIt) console.log(`packed ${count} catalog models into ${path.relative(root, packFile)}`);
}

await build({
  entryPoints: [path.join(root, 'src', 'main.ts')],
  outfile,
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'esm',
  sourcemap: 'linked',
  logLevel: 'warning',
  define,
});
if (runIt) process.exitCode = spawnSync(process.execPath, ['--enable-source-maps', outfile, ...args], { stdio: 'inherit' }).status ?? 1;
else console.log(`built ${path.relative(root, outfile)}`);

// Release artifacts in dist/:
//   <name>.mjs + <name>-models.pack    every OS with Node >= 22:  node <name>.mjs --help  (keep the two files together)
//   <name>-<os>-<arch>[.exe]           a single executable with Node and the models inside (nothing else needed)
//
//   npm run dist              both, the executable for the OS and CPU this runs on
//   npm run dist -- --no-sea  only the .mjs and its pack
//
// dist/ holds what a GitHub Release attaches (.github/workflows/release.yml builds it on every system); it is not
// committed. test/dist/dist.test.ts fails when the files in it were built from other sources (the fingerprint of
// scripts/source-hash.mjs, printed by --version).
//
// The executable is a Node.js Single Executable Application (https://nodejs.org/api/single-executable-applications.html):
// the bundle, and the pack as an asset, are injected into a copy of the running `node` binary with postject. It cannot
// be cross-built, so run `npm run dist` once on Windows, macOS and Linux (x64 or arm64) to get the three. On macOS the
// copy is re-signed ad hoc; sign and notarize it with a Developer ID to distribute it signed. On Windows the copy loses
// Node's Authenticode signature; sign it with signtool where signed binaries are needed.
import { build } from 'esbuild';
import { execFileSync } from 'node:child_process';
import { copyFileSync, chmodSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { cliName, define, root } from './cli-name.mjs';
import { packModels } from './pack-models.mjs';

const withSea = !process.argv.includes('--no-sea');
const dist = path.join(root, 'dist');
const work = path.join(root, 'out', 'sea');
// Only the files this run rewrites are replaced: executables built on other systems and copied here stay.
mkdirSync(dist, { recursive: true });

const common = {
  entryPoints: [path.join(root, 'src', 'main.ts')],
  bundle: true,
  platform: 'node',
  target: 'node22',
  minify: true,
  legalComments: 'none',
  logLevel: 'warning',
  define,
};

// 1. Portable module and the pack of the catalog models next to it: needs Node >= 22 on the machine.
const mjs = path.join(dist, `${cliName}.mjs`);
await build({ ...common, outfile: mjs, format: 'esm', banner: { js: '#!/usr/bin/env node' } });
chmodSync(mjs, 0o755);
const packFile = path.join(dist, `${cliName}-models.pack`);
const { pack, count } = packModels();
writeFileSync(packFile, pack);
console.log(`dist/${path.basename(mjs)}\ndist/${path.basename(packFile)} (${count} models, ${(pack.length / 1e6).toFixed(1)} MB)`);
if (!withSea) process.exit(0);

// 2. Single executable. SEA runs one CommonJS script, so the same code is bundled as CJS; the pack is an asset.
rmSync(work, { recursive: true, force: true });
mkdirSync(work, { recursive: true });
const main = path.join(work, 'main.cjs');
const blob = path.join(work, 'sea-prep.blob');
await build({ ...common, outfile: main, format: 'cjs' });
// Paths relative to out/sea: the executable records the script's path, and it must not be this machine's folder.
const seaConfig = { main: 'main.cjs', output: 'sea-prep.blob', disableExperimentalSEAWarning: true, useCodeCache: false, useSnapshot: false, assets: { 'models.pack': path.relative(work, packFile) } };
writeFileSync(path.join(work, 'sea-config.json'), JSON.stringify(seaConfig));
execFileSync(process.execPath, ['--experimental-sea-config', 'sea-config.json'], { stdio: 'inherit', cwd: work });

const os = { win32: 'windows', darwin: 'macos', linux: 'linux' }[process.platform] ?? process.platform;
const exe = path.join(dist, `${cliName}-${os}-${process.arch}${process.platform === 'win32' ? '.exe' : ''}`);
copyFileSync(process.execPath, exe);
chmodSync(exe, 0o755);
if (process.platform === 'darwin') execFileSync('codesign', ['--remove-signature', exe], { stdio: 'inherit' });
const postject = path.join(root, 'node_modules', 'postject', 'dist', 'cli.js');
execFileSync(process.execPath, [
  postject, exe, 'NODE_SEA_BLOB', blob,
  '--sentinel-fuse', 'NODE_SEA_FUSE_fce680ab2cc467b6e072b8b5df1996b2',
  ...(process.platform === 'darwin' ? ['--macho-segment-name', 'NODE_SEA'] : []),
], { stdio: 'inherit' });
if (process.platform === 'darwin') execFileSync('codesign', ['--sign', '-', exe], { stdio: 'inherit' });
console.log(`dist/${path.basename(exe)}`);

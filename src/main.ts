/**
 * Entry of the command: `npm run plan -- <model>`, the bundle in dist/ (`node <name>.mjs`) and the single
 * executable. Node >= 22, the same on Windows, macOS and Linux.
 */
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import * as sea from 'node:sea';
import { builtinModels, type BuiltinModels } from './cli/builtin';
import { findHfToken } from './cli/hf-token';
import { localGpu, runPlan } from './cli/run';

/** The command's own arguments: in a single executable argv[1] is the executable itself, as the script is in a bundle. */
const args = process.argv.slice(2);
const env = process.env;

/** Where the program is: the single executable itself, or the bundled script. */
const programDir = path.dirname(sea.isSea() ? process.execPath : (process.argv[1] ?? '.'));
const PACK = `${__CLI_NAME__}-models.pack`;

/**
 * The catalog models bundled with the command: the pack next to the program first (so the data can be refreshed
 * without rebuilding), else the copy inside the single executable. Undefined when there is neither.
 */
function builtin(baseUrl: string): BuiltinModels | undefined {
  try {
    return builtinModels(readFileSync(path.join(programDir, PACK)), baseUrl);
  } catch {}
  try {
    if (sea.isSea()) return builtinModels(new Uint8Array(sea.getAsset('models.pack')), baseUrl);
  } catch {}
  return undefined;
}

async function main(): Promise<number> {
  if (args.length === 1 && (args[0] === '--version' || args[0] === '-v')) {
    console.log(`${__CLI_NAME__} ${__CLI_VERSION__} (sources ${__SOURCE_HASH__})`);
    return 0;
  }
  // Behind a proxy, Node's fetch honours HTTPS_PROXY / HTTP_PROXY / NO_PROXY only with NODE_USE_ENV_PROXY=1,
  // which must be set before Node starts: run again with it.
  if ((env.HTTPS_PROXY || env.https_proxy || env.HTTP_PROXY || env.http_proxy) && env.NODE_USE_ENV_PROXY !== '1') {
    const self = process.argv.slice(1, 2).filter((a) => a !== process.execPath);
    const r = spawnSync(process.execPath, [...self, ...args], { stdio: 'inherit', env: { ...env, NODE_USE_ENV_PROXY: '1' } });
    return r.status ?? 1;
  }
  // The Hugging Face login already on this machine (never printed): HF_TOKEN, else the hub's token file.
  const token = findHfToken({ env });
  const baseUrl = env.HF_ENDPOINT || 'https://huggingface.co';
  const models = builtin(baseUrl);
  // Without the built-in models only the local cache works offline: say why a catalog model is missing.
  if (!models && args.includes('--offline')) console.error(`Note: ${PACK} was not found next to ${path.basename(process.argv[1] ?? __CLI_NAME__)} (in ${programDir}): the built-in models are not available, only models read before are.`);
  return runPlan(args, {
    fetch: globalThis.fetch,
    env: token ? { ...env, HF_TOKEN: token } : env,
    localGpu: () => localGpu(),
    out: (text) => console.log(text),
    writeFile: (file, content) => writeFile(file, content, 'utf8'),
    baseUrl,
    isTTY: process.stdout.isTTY === true,
    columns: process.stdout.columns,
    platform: process.platform,
    ...(models ? { builtin: models } : {}),
  });
}

main().then(
  (code) => {
    process.exitCode = code;
  },
  (e: unknown) => {
    console.error(e instanceof Error ? e.message : String(e));
    process.exitCode = 1;
  },
);

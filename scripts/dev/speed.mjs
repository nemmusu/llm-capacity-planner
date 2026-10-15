// Short vLLM speed measurements on this PC to calibrate the planner speed model (scripts/dev/speed-entry.ts); development only.
//   npm run dev:speed -- [--only=<name>]
import { build } from 'esbuild';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..', '..');
const outfile = path.join(root, 'out/cli/speed.mjs');
await build({ entryPoints: [path.join(root, 'scripts/dev/speed-entry.ts')], outfile, bundle: true, platform: 'node', target: 'node22', format: 'esm', logLevel: 'warning', packages: 'external' });
await import(pathToFileURL(outfile).href);

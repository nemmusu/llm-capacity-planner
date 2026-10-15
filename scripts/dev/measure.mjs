// Short vLLM measurements on this PC to calibrate the VRAM planner (scripts/dev/measure-entry.ts); development only.
//   npm run dev:measure -- [--only=<name>]
import { build } from 'esbuild';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..', '..');
const outfile = path.join(root, 'out/cli/measure.mjs');
await build({ entryPoints: [path.join(root, 'scripts/dev/measure-entry.ts')], outfile, bundle: true, platform: 'node', target: 'node22', format: 'esm', logLevel: 'warning', packages: 'external' });
await import(pathToFileURL(outfile).href);

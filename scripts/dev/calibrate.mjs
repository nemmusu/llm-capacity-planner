// Checks the VRAM planner against measurements (scripts/dev/calibrate-entry.ts); development only.
//   npm run dev:calibrate -- [--only=<model or reference id>]
import { build } from 'esbuild';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..', '..');
const outfile = path.join(root, 'out/cli/calibrate.mjs');
await build({ entryPoints: [path.join(root, 'scripts/dev/calibrate-entry.ts')], outfile, bundle: true, platform: 'node', target: 'node22', format: 'esm', logLevel: 'warning', packages: 'external' });
await import(pathToFileURL(outfile).href);

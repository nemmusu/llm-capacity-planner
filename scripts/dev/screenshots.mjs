// Writes the README's example images (scripts/dev/screenshots-entry.ts); development only.
//   npm run dev:screenshots
import { build } from 'esbuild';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { define } from '../cli-name.mjs';

const root = path.resolve(import.meta.dirname, '..', '..');
const outfile = path.join(root, 'out/cli/screenshots.mjs');
await build({ entryPoints: [path.join(root, 'scripts/dev/screenshots-entry.ts')], outfile, bundle: true, platform: 'node', target: 'node22', format: 'esm', logLevel: 'warning', packages: 'external', define });
await import(pathToFileURL(outfile).href);

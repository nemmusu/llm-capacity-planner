// Writes the appendices of docs/METHODOLOGY.md from the repository data (scripts/dev/docs-entry.ts); development only.
//   npm run dev:docs
import { build } from 'esbuild';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..', '..');
const outfile = path.join(root, 'out/cli/docs.mjs');
await build({ entryPoints: [path.join(root, 'scripts/dev/docs-entry.ts')], outfile, bundle: true, platform: 'node', target: 'node22', format: 'esm', logLevel: 'warning', packages: 'external' });
await import(pathToFileURL(outfile).href);

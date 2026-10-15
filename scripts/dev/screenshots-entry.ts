// Writes the README's examples from the real command (scripts/dev/screenshots-lib.ts): the images in docs/images and
// the generated blocks of README.md. Development only. Run after a change to the output or the data:
// test/docs/screenshots.test.ts fails until they match.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { builtinModels } from '../../src/cli/builtin';
import type * as Pack from '../pack-models.mjs';
import { readmeAccuracy, readmeExamples, readmeTop, SCREENSHOTS, screenshotSvg, withBlocks } from './screenshots-lib';

const root = path.resolve(import.meta.dirname, '..', '..');
const dir = path.join(root, 'docs', 'images');
const base = 'https://hub.invalid';
mkdirSync(dir, { recursive: true });
// Loaded at run time, not bundled: it reads package.json next to it.
const { packModels } = (await import(pathToFileURL(path.join(root, 'scripts', 'pack-models.mjs')).href)) as typeof Pack;
const builtin = builtinModels(packModels(path.join(root, 'data', 'models')).pack, base);
for (const s of SCREENSHOTS) {
  writeFileSync(path.join(dir, s.file), await screenshotSvg(s, builtin, base));
  console.log(`docs/images/${s.file}`);
}
const readme = path.join(root, 'README.md');
writeFileSync(readme, withBlocks(readFileSync(readme, 'utf8'), { top: await readmeTop(builtin, base), accuracy: await readmeAccuracy(builtin, base), examples: await readmeExamples(builtin, base) }));
console.log('README.md (generated blocks)');

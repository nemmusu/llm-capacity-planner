// Writes the ready-made capacity reports (scripts/dev/report-lib.ts) to data/report.json. Development only, offline:
// it reads the model fixtures written by scripts/dev/update-models.mjs.
//   npm run dev:report
// Run after a change to the planner, the output or the data: test/docs/report.test.ts fails until the file matches.
import { writeFileSync } from 'node:fs';
import path from 'node:path';
import { buildReport, reportText } from './report-lib';

const root = path.resolve(import.meta.dirname, '..', '..');
const { file, models, missing } = await buildReport(root);
writeFileSync(path.join(root, 'data/report.json'), reportText(file));
console.log(`report: ${models.length} models; without data: ${missing.join(', ') || 'none'}`);

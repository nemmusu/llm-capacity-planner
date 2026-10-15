// A fingerprint of everything the command is built from: src/, the bundled data (data/*.json), package.json and
// package-lock.json. `npm run dist` writes it into the bundle and the executables (`--version` prints it), and
// test/dist/dist.test.ts fails when dist/ was built from other sources. Line endings do not count.
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';

/** Files under `dir` (relative to `root`, forward slashes), sorted. */
function files(root, dir, keep) {
  const out = [];
  const walk = (rel) => {
    for (const e of readdirSync(path.join(root, rel), { withFileTypes: true })) {
      const p = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) walk(p);
      else if (keep(p)) out.push(p);
    }
  };
  walk(dir);
  return out;
}

/** The first 12 hex digits of the fingerprint; `except` leaves files out (data/report.json, which is computed from the rest). */
export function sourceHash(root, except = []) {
  const list = [
    ...files(root, 'src', (p) => p.endsWith('.ts')),
    ...readdirSync(path.join(root, 'data')).filter((f) => f.endsWith('.json')).map((f) => `data/${f}`),
    'package.json',
    'package-lock.json',
  ]
    .filter((p) => !except.includes(p))
    .sort();
  const h = createHash('sha256');
  for (const p of list) h.update(`${p}\n`).update(readFileSync(path.join(root, p), 'utf8').replace(/\r\n/g, '\n')).update('\n');
  return h.digest('hex').slice(0, 12);
}

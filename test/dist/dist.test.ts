// dist/ holds what a release attaches: the bundle, the pack of the built-in models and the single executables. It is
// not committed (the GitHub Releases carry it); when it is there, every file must be built from the current sources,
// or this test fails (rebuild with `npm run dist`). Without dist/ the test is skipped; the release workflow always
// runs it after the build.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { modelsHash } from '../../scripts/pack-models.mjs';
import { sourceHash } from '../../scripts/source-hash.mjs';
import { readPack } from '../../src/cli/builtin';

const dist = 'dist';
const stale = (file: string) => `dist/${file} was built from other sources: run npm run dist`;
const executables = existsSync(dist) ? readdirSync(dist).filter((f) => f.startsWith(`${__CLI_NAME__}-`) && !f.endsWith('.pack')) : [];

describe.skipIf(!existsSync(dist))('dist/ matches the sources', () => {
  const source = sourceHash('.');
  const models = modelsHash();

  it('only the release artifacts are in it', () => {
    for (const f of readdirSync(dist)) expect(f, `unexpected dist/${f}`).toMatch(new RegExp(`^${__CLI_NAME__}(\\.mjs|-models\\.pack|-(windows|macos|linux)-(x64|arm64)(\\.exe)?)$`));
  });

  it('the portable bundle', () => {
    const mjs = readFileSync(path.join(dist, `${__CLI_NAME__}.mjs`), 'utf8');
    expect(mjs.startsWith('#!/usr/bin/env node'), 'the bundle starts with its shebang').toBe(true);
    expect(mjs.includes(`(sources ${source})`), stale(`${__CLI_NAME__}.mjs`)).toBe(true);
  });

  it('the pack of the built-in models', () => {
    const file = `${__CLI_NAME__}-models.pack`;
    expect(readPack(readFileSync(path.join(dist, file))).index.source, stale(file)).toBe(models);
  });

  it('every single executable (with the pack inside)', () => {
    expect(executables.length, 'at least the executable of one system').toBeGreaterThan(0);
    for (const f of executables) {
      const bytes = readFileSync(path.join(dist, f));
      expect(bytes.includes(`(sources ${source})`), stale(f)).toBe(true);
      expect(bytes.includes(`"source":"${models}"`), `${stale(f)} (its built-in models)`).toBe(true);
    }
  });

  it('no path of the machine that built it: its user folder, its checkout, the build folder', () => {
    const root = path.resolve('.');
    // This machine's folders followed by a separator, in both slash styles; on the build machine they are the ones the
    // build could record. Only folders two levels deep or more (C:\Users\name, /home/name): a one-level folder such as
    // /root or /work is a prefix of Node's own strings (its build paths, /worker/…).
    const local = [...new Set([root, homedir()].flatMap((p) => [p, p.replace(/\\/g, '/')]))].filter((p) => p.split(/[\\/]/).filter(Boolean).length >= 2);
    for (const f of readdirSync(dist)) {
      const text = readFileSync(path.join(dist, f)).toString('latin1');
      for (const p of local) expect(text.includes(`${p}/`) || text.includes(`${p}\\`), `dist/${f} contains ${p}`).toBe(false);
      expect(/[\\/]out[\\/]sea[\\/]main\.cjs/.test(text), `dist/${f} records the absolute path of the script it was built from`).toBe(false);
    }
  }, 120_000);
});

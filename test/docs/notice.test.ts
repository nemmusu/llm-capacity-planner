// NOTICE.md lists the third-party data the repository and its releases carry: written from the data by
// `npm run dev:docs`, compared here so that a new model, benchmark or GPU source is never left out.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { noticeText, type LicenceIndex } from '../../scripts/dev/notice-lib';
import { NOT_REDISTRIBUTED, packModels } from '../../scripts/pack-models.mjs';
import { readPack } from '../../src/cli/builtin';

describe('NOTICE.md', () => {
  it('lists every saved model repository, benchmark page and GPU source as the data holds them', async () => {
    expect((await noticeText('.')) === readFileSync('NOTICE.md', 'utf8'), 'NOTICE.md is out of date: run npm run dev:docs').toBe(true);
  });

  it('names why each repository is left out of the models pack', () => {
    const notice = readFileSync('NOTICE.md', 'utf8');
    for (const [repo, why] of Object.entries(NOT_REDISTRIBUTED)) {
      expect(notice).toContain(`[${repo}](https://huggingface.co/${repo})`);
      expect(notice).toContain(why);
    }
  });
});

describe('repositories whose licence keeps them out', { timeout: 120_000 }, () => {
  // Gated licences (the Llama 4 family) and those that may not allow passing files on (NOT_REDISTRIBUTED).
  const sources = (dir: string): { dir: string; repo: string; licenseName: string | null; readable: boolean; withheld?: string }[] =>
    readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
      if (!e.isDirectory()) return [];
      const d = path.join(dir, e.name);
      const own = existsSync(path.join(d, 'source.json')) ? [{ dir: d, ...(JSON.parse(readFileSync(path.join(d, 'source.json'), 'utf8')) as { repo: string; licenseName: string | null; readable: boolean; withheld?: string }) }] : [];
      return [...own, ...sources(d)];
    });
  const all = sources(path.join('data', 'models'));
  const excluded = all.filter((s) => s.repo in NOT_REDISTRIBUTED || /^llama/i.test(s.licenseName ?? ''));

  it('keep no file of theirs in the repository, only the record of why', () => {
    expect(excluded.map((s) => s.repo)).toEqual(expect.arrayContaining(['mistralai/Mistral-Medium-3.5-128B', 'meta-llama/Llama-4-Scout-17B-16E-Instruct']));
    for (const s of excluded) {
      expect(s.readable, s.repo).toBe(false);
      expect(readdirSync(s.dir).filter((f) => !['source.json', 'quant-repos.json', 'variants'].includes(f)), s.repo).toEqual([]);
    }
    for (const repo of Object.keys(NOT_REDISTRIBUTED)) expect(all.find((s) => s.repo === repo)?.withheld, repo).toBe(NOT_REDISTRIBUTED[repo]);
  });

  it('are never in the models pack', () => {
    const repos = Object.keys(readPack(packModels().pack).index.repos);
    expect(repos.filter((r) => excluded.some((s) => s.repo === r))).toEqual([]);
    expect(repos.length).toBe(91);
  });
});

describe('licenses/: the full text of every licence the packed and saved repositories carry', () => {
  const index = (JSON.parse(readFileSync(path.join('licenses', 'index.json'), 'utf8')) as LicenceIndex).repos;
  const packed = Object.entries(readPack(packModels().pack).index.repos);
  const texts = readdirSync('licenses').filter((f) => f !== 'index.json');

  it('every packed repository has its licence texts, recorded at the revision saved', () => {
    for (const [repo, r] of packed) {
      const entry = index[repo];
      expect(entry?.files.length, repo).toBeGreaterThan(0);
      expect(entry?.revision, repo).toBe((r as { revision: string }).revision);
      for (const f of entry?.files ?? []) expect(texts, `${repo}: licenses/${f}`).toContain(f);
    }
    // Nothing more: no licence kept for a repository that is not shipped.
    expect(Object.keys(index).sort()).toEqual(packed.map(([repo]) => repo).sort());
  });

  it('every licence named by a repository has its own text: the official one, beside the repository\'s own file', () => {
    const official: Record<string, string> = {
      'apache-2.0': 'apache-2.0.txt',
      'apache-license-2.0': 'apache-2.0.txt',
      'nvidia-open-model-license': 'nvidia-open-model-license.txt',
      'nvidia-nemotron-open-model-license': 'nvidia-nemotron-open-model-license.txt',
    };
    for (const [repo, e] of Object.entries(index)) if (official[e.license]) expect(e.files.some((f) => f === official[e.license] || f.includes(e.license.replace('apache-license-2.0', 'apache-2.0'))), `${repo} (${e.license})`).toBe(true);
  });

  it('each file names its source on top and is used by some repository', () => {
    const used = new Set(Object.values(index).flatMap((e) => e.files));
    for (const f of texts) {
      expect(used.has(f), `licenses/${f} is not used`).toBe(true);
      expect(readFileSync(path.join('licenses', f), 'utf8').split('\n').slice(0, 6).join('\n'), f).toMatch(/^Source: https:\/\//m);
    }
  });
});

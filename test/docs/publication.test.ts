// The repository is published: no file may carry a personal path, a user or machine name, an e-mail address or a
// credential. Every file is read (gzip files unpacked), outside the folders git ignores; the user name of the machine
// running the test is checked too, so a path or a log line copied from it is caught wherever it lands.
import { readdirSync, readFileSync } from 'node:fs';
import { homedir, hostname, userInfo } from 'node:os';
import path from 'node:path';
import { gunzipSync } from 'node:zlib';
import { describe, expect, it } from 'vitest';

const SKIP = new Set(['node_modules', 'out', 'coverage', '.qa', 'dist', '.git']);
function files(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((e) => (SKIP.has(e.name) ? [] : e.isDirectory() ? files(path.join(dir, e.name)) : [path.join(dir, e.name)]));
}
const text = (f: string) => {
  const raw = readFileSync(f);
  return (f.endsWith('.gz') ? gunzipSync(raw) : raw).toString('utf8');
};

/** Contacts the model licences themselves print (their texts in licenses/, quoted in data/models.json), not people. */
const LICENCE_CONTACTS = new Set(['api@minimax.io', 'sales@mistral.ai', 'glmlicense@z.ai', 'license@moonshot.ai', 'model-business@notice.qwencloud.com']);
/** Placeholders the tests and the redacted logs use. */
const HOME = /([A-Za-z]:\\+Users\\+|\/home\/|\/Users\/)(?!<user>|u\b|runner\b)[A-Za-z0-9._-]{2,}/;
const SECRETS: [string, RegExp][] = [
  ['Hugging Face token', /\bhf_[A-Za-z0-9]{20,}/],
  ['GitHub token', /\bgh[pousr]_[A-Za-z0-9]{20,}/],
  ['API key', /\bsk-[A-Za-z0-9]{20,}/],
  ['AWS key', /\bAKIA[0-9A-Z]{16}\b/],
  ['Slack token', /\bxox[bpas]-[A-Za-z0-9-]{10,}/],
  ['private key', /-----BEGIN [A-Z ]*PRIVATE KEY-----/],
];

describe('the published repository', { timeout: 600_000 }, () => {
  const all = files('.').map((f) => [f, text(f)] as const);

  it('has no personal folder, e-mail address or credential in any file', () => {
    const hits: string[] = [];
    for (const [f, t] of all) {
      const home = HOME.exec(t);
      if (home) hits.push(`${f}: ${home[0]}`);
      for (const m of t.matchAll(/[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}/g)) if (!LICENCE_CONTACTS.has(m[0]) && !m[0].endsWith('@users.noreply.github.com')) hits.push(`${f}: ${m[0]}`);
      for (const [what, re] of SECRETS) if (re.test(t)) hits.push(`${f}: ${what}`);
    }
    expect(hits).toEqual([]);
  });

  it("holds nothing of the machine running this test: its user name, home folder or host name", () => {
    const generic = new Set(['root', 'runner', 'user', 'admin', 'node', 'ubuntu', 'localhost']);
    const own = [userInfo().username, path.basename(homedir()), hostname()].filter((s) => s.length >= 4 && !generic.has(s.toLowerCase()));
    const hits = all.flatMap(([f, t]) => own.filter((s) => t.toLowerCase().includes(s.toLowerCase())).map((s) => `${f} contains this machine's name (${s.length} characters)`));
    expect(hits).toEqual([]);
  });
});

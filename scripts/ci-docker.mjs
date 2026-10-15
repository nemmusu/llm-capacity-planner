// Local CI in Docker: validates the COMMITTED code (git HEAD) in a clean Linux container, as it would be pushed.
// Nothing from this machine's node_modules or build output goes in.
//
//   npm run ci:docker    npm ci, lint, typecheck, tests + 100 % coverage gate, then the Linux executable and a smoke run
//
// The npm cache lives in a named Docker volume, so the second run is quicker. The coverage report and the Linux
// executable are copied to .qa/ci/out, also when a step fails.
import { spawnSync } from 'node:child_process';
import { mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { cliName, root } from './cli-name.mjs';

const image = `${cliName}-ci`;
const work = path.join(root, '.qa', 'ci');
const input = path.join(work, 'in');
const output = path.join(work, 'out');

function run(cmd, args) {
  const r = spawnSync(cmd, args, { cwd: root, stdio: 'inherit' });
  if (r.error) throw r.error;
  return r.status ?? 1;
}
function git(args) {
  const r = spawnSync('git', args, { cwd: root, encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`git ${args.join(' ')}: ${r.stderr}`);
  return r.stdout.trim();
}

const head = git(['rev-parse', '--short', 'HEAD']);
if (git(['status', '--porcelain', '--untracked-files=no'])) console.warn(`! Uncommitted changes: only commit ${head} is validated.`);

rmSync(work, { recursive: true, force: true });
mkdirSync(input, { recursive: true });
mkdirSync(output, { recursive: true });
if (run('git', ['archive', '--format=tar', '-o', path.join(input, 'src.tar'), 'HEAD']) !== 0) process.exit(1);
if (run('docker', ['build', '-t', image, path.join(root, 'scripts', 'ci')]) !== 0) process.exit(1);

const exe = `dist/${cliName}-linux-$(node -p process.arch)`;
const inner = [
  'tar -xf /in/src.tar -C /work',
  'status=0',
  `{ npm ci --no-audit --no-fund && npm run validate && npm run dist && ${exe} --version && ${exe} --help > /dev/null && HF_ENDPOINT=http://127.0.0.1:9 ${exe} Qwen/Qwen3.5-9B --tokens 32k --offline --no-color > /dev/null; } || status=$?`,
  'cp -r coverage /out/ 2>/dev/null; cp dist/* /out/ 2>/dev/null',
  'exit $status',
].join('\n');

console.log(`\n> validate ${head} on Linux (Node 22)`);
const started = Date.now();
const status = run('docker', [
  'run', '--rm', '--init',
  '-v', `${input}:/in:ro`, '-v', `${output}:/out`, '-v', `${image}-npm:/root/.npm`,
  image, 'bash', '-lc', inner,
]);
rmSync(input, { recursive: true, force: true });
const mins = ((Date.now() - started) / 60000).toFixed(1);
console.log(status === 0 ? `\nLocal CI (Linux): green on ${head} in ${mins} min` : `\nLocal CI (Linux): failed on ${head} after ${mins} min`);
console.log(`Artifacts: ${output}`);
process.exit(status);

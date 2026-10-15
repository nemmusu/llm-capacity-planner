/**
 * The catalog models bundled with the command (data/models packed by scripts/pack-models.mjs), for tests that
 * run the real planner on real models without the network: the pack, the command run in-process with it, and
 * one model read as the command reads it.
 *
 * Packing the 93 repos takes ~20 s: the pack is built once and kept in the system's temp folder, keyed by this
 * checkout, the last change to data/models and the packer itself (never out/, which the end-to-end test rebuilds
 * while other test files run).
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { packModels } from '../../scripts/pack-models.mjs';
import { builtinModels, type BuiltinModels } from '../../src/cli/builtin';
import { readRemoteModel, type RemoteModel } from '../../src/cli/model';
import { runPlan, type PlanDeps } from '../../src/cli/run';
import { HubClient } from '../../src/hub/client';
import { repoLayout } from '../../src/hub/formats';
import { plannerModel } from '../../src/planner/model';

/** A hub address nothing answers on: every model must come from the pack. */
export const HUB = 'https://hub.invalid';

/** A fetch that always fails: any network attempt is a test failure (or an expected error). */
export const noNetwork = (() => Promise.reject(new Error('network disabled in tests'))) as typeof fetch;

/** Newest modification time under a folder (as scripts/build.mjs decides to repack). */
function newest(dir: string): number {
  let t = statSync(dir).mtimeMs;
  for (const e of readdirSync(dir, { withFileTypes: true })) t = Math.max(t, e.isDirectory() ? newest(path.join(dir, e.name)) : statSync(path.join(dir, e.name)).mtimeMs);
  return t;
}

let packMemo: Uint8Array | undefined;

/** The pack of the catalog models (module comment). */
export function catalogPack(): Uint8Array {
  if (packMemo) return packMemo;
  const models = path.join('data', 'models');
  const key = createHash('sha256')
    .update(`${path.resolve(models)}|${Math.round(newest(models))}|`)
    .update(readFileSync(path.join('scripts', 'pack-models.mjs')))
    .digest('hex')
    .slice(0, 16);
  const kept = path.join(os.tmpdir(), `${__CLI_NAME__}-test-${key}.pack`);
  if (existsSync(kept)) packMemo = readFileSync(kept);
  else {
    packMemo = packModels(models).pack;
    // Written next to the target and renamed: two test files packing at once never read half a file.
    const tmp = `${kept}.${process.pid}.tmp`;
    writeFileSync(tmp, packMemo);
    renameSync(tmp, kept);
  }
  return packMemo;
}

let builtinMemo: BuiltinModels | undefined;

/** The pack as the command's built-in models. */
export function catalogBuiltin(): BuiltinModels {
  builtinMemo ??= builtinModels(catalogPack(), HUB);
  return builtinMemo;
}

/** A temporary folder removed by `cleanup` (the command's cache in these tests: never the user's). */
export function tempFolder(prefix: string): { dir: string; cleanup: () => void } {
  const dir = mkdtempSync(path.join(os.tmpdir(), prefix));
  return { dir, cleanup: () => rmSync(dir, { recursive: true, force: true }) };
}

export interface CliRun {
  code: number;
  /** Everything printed, joined by newlines. */
  text: string;
  /** Files the command wrote, by name. */
  written: Record<string, string>;
}

/**
 * The command in-process on the built-in models: no network (any attempt fails), no GPU on this PC, 120
 * columns, no colour, a private cache folder.
 */
export async function runCli(argv: readonly string[], cacheDir: string, deps: Partial<PlanDeps> = {}): Promise<CliRun> {
  const out: string[] = [];
  const written: Record<string, string> = {};
  const code = await runPlan(argv, {
    fetch: noNetwork,
    env: { WT_SESSION: '1' },
    localGpu: () => Promise.resolve(undefined),
    out: (t) => out.push(t),
    writeFile: (file, content) => {
      written[file] = content;
      return Promise.resolve();
    },
    baseUrl: HUB,
    cacheDir,
    isTTY: false,
    columns: 120,
    platform: 'linux',
    builtin: catalogBuiltin(),
    ...deps,
  });
  return { code, text: out.join('\n'), written };
}

const modelMemo = new Map<string, Promise<{ remote: RemoteModel; model: ReturnType<typeof plannerModel> }>>();

/** A catalog model as the command reads it (root variant, main revision) and the planner's view of it. */
export function catalogModel(repo: string): Promise<{ remote: RemoteModel; model: ReturnType<typeof plannerModel> }> {
  const hit = modelMemo.get(repo);
  if (hit) return hit;
  const load = (async () => {
    const hub = new HubClient({ fetch: catalogBuiltin().fetch, baseUrl: HUB, getToken: () => undefined });
    const info = await hub.info(repo);
    const files = await hub.files(repo, 'main');
    const variant = repoLayout(files, { name: repo, tags: info.tags }).safetensors.find((v) => v.dir === '');
    if (!variant) throw new Error(`${repo}: no root safetensors variant in the pack`);
    const remote = await readRemoteModel(hub, repo, 'main', variant);
    return { remote, model: plannerModel(remote.config, remote.facts, remote.shape, remote.tensors) };
  })();
  modelMemo.set(repo, load);
  return load;
}

/**
 * Italian words that must never reach the English output (whole words, so "con" in "console" or "non" in
 * "nonzero" do not count).
 */
export const ITALIAN = /(?<![\p{L}-])(non|della|delle|entra|entrano|schede?|utenti|velocità|memoria|è|con|pesi|teste|quota|richieste?|stimato|tarato|misurato|senza|più|che|oltre|questa|stimabile|pubblicati|servono|bastano|restano|lenti|abbassa|supera)(?![\p{L}-])/iu;

/** The Italian words of a text with a little context, for a readable failure. */
export function italianHits(text: string): string[] {
  const re = new RegExp(ITALIAN.source, 'giu');
  return [...text.matchAll(re)].map((m) => text.slice(Math.max(0, (m.index ?? 0) - 30), (m.index ?? 0) + 40).replace(/\s+/g, ' '));
}

/** Words that mean a number or a field went wrong (text, CSV, Markdown; JSON is checked value by value). */
export const BAD_WORDS = /\b(NaN|undefined|Infinity|null)\b/;

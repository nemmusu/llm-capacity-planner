/**
 * What the command reads of a Hugging Face model, without downloading the weights: config.json,
 * the weights index, and the JSON header of every safetensors file (two Range requests each).
 */
import path from 'node:path';
import { precisionFromConfig, type SafetensorsVariant } from '../hub/formats';
import type { HubFile } from '../hub/types';
import type { HubClient } from '../hub/client';
import { factsFromHfConfig, loadShapeFromHfConfig } from '../model/hf-config';
import type { ModelFactsEx } from '../model/types';
import { tensorsFromHeader, type LoadShape, type TensorInfo } from '../model/weights-exact';

export interface RemoteModel {
  repo: string;
  /** config.json as read. */
  config: unknown;
  dir: string;
  facts: ModelFactsEx;
  shape: LoadShape;
  /** Every tensor of the weight files vLLM loads (from the headers). */
  tensors: TensorInfo[];
  /** The weight files read (the index's, when the repo has one). */
  files: HubFile[];
  /** Bytes of those files as the hub lists them. */
  fileBytes: number;
  precision?: string;
}

/** Runs `fn` over `items`, at most `limit` at a time, keeping the order. */
export async function mapLimit<T, R>(items: readonly T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i] as T);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

/**
 * The weight files vLLM loads: with a `*.safetensors.index.json`, only the files its weight_map names
 * (vLLM's filter_duplicate_safetensors_files: a repo's `consolidated.safetensors` copy is left out).
 */
export async function loadedWeightFiles(hub: HubClient, repo: string, revision: string, variant: SafetensorsVariant<HubFile>): Promise<HubFile[]> {
  const index = variant.files.find((f) => f.path.endsWith('.safetensors.index.json'));
  if (!index) return variant.weights;
  const raw = (await hub.json(repo, index.path, revision)) as { weight_map?: Record<string, unknown> } | null;
  const used = new Set(Object.values(raw?.weight_map ?? {}).map(String));
  const files = variant.weights.filter((f) => used.has(path.posix.basename(f.path)));
  return files.length > 0 ? files : variant.weights;
}

/** Reads a model's config and the headers of its weight files. */
export async function readRemoteModel(hub: HubClient, repo: string, revision: string, variant: SafetensorsVariant<HubFile>): Promise<RemoteModel> {
  const dir = variant.dir;
  const config = await hub.json(repo, dir ? `${dir}/config.json` : 'config.json', revision);
  const files = await loadedWeightFiles(hub, repo, revision, variant);
  const headers = await mapLimit(files, 8, (f) => hub.safetensorsHeader(repo, f.path, revision));
  const tensors = headers.flatMap((h) => tensorsFromHeader(h.header));
  const fileBytes = files.reduce((a, f) => a + f.sizeBytes, 0);
  const precision = precisionFromConfig(config) ?? factsFromHfConfig(config).quant;
  return {
    repo,
    config,
    dir,
    facts: factsFromHfConfig(config, { name: repo, fileBytes }),
    shape: loadShapeFromHfConfig(config),
    tensors,
    files,
    fileBytes,
    ...(precision ? { precision } : {}),
  };
}

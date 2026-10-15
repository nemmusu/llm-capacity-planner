/**
 * The compatibility matrix (planner/compat.ts, data/compat.json) applied to the planned
 * layouts: a card the engine does not start on becomes "engine does not start" with the reason and its source
 * (never a recommendation), a partial or unverified one carries a warning, and so does a limit seen only under
 * Windows / WSL. Only the catalog's models are in the matrix; for the others the engine's own checks stay alone.
 */
import modelsJson from '../../data/models.json';
import { checkCompat, type CompatResult } from '../planner/compat';
import { FORMAT_WARNING } from '../planner/hardware';
import type { PlanRow } from '../planner/plan';

interface CatalogModels {
  models: { id: string; repo: string; quantizations: { repo: string }[]; longContext?: { officialMaxTokens: number | null } }[];
}

/** The catalog id of a Hugging Face repo (the base checkpoint or one of its quantizations). */
export function catalogModelId(repo: string, catalog: CatalogModels = modelsJson as CatalogModels): string | undefined {
  const r = repo.toLowerCase();
  return catalog.models.find((m) => m.repo.toLowerCase() === r || m.quantizations.some((q) => q.repo.toLowerCase() === r))?.id;
}

/** The longest context the lab states for a catalog model (with an official YaRN factor when beyond the config's), or undefined. */
export function documentedMaxTokens(repo: string, catalog: CatalogModels = modelsJson as CatalogModels): number | undefined {
  const id = catalogModelId(repo, catalog);
  return catalog.models.find((m) => m.id === id)?.longContext?.officialMaxTokens ?? undefined;
}

export interface CompatQueryBase {
  model: string;
  repo: string;
  engine: 'vllm' | 'sglang';
  kvCache: 'fp8' | 'auto';
}

const source = (r: CompatResult) => (r.sources[0] ? ` (source: ${r.sources[0]})` : '');

/** The rows with the matrix's verdict applied (module comment); cards outside the catalog are left alone. */
export function applyCompat(rows: readonly PlanRow[], q: CompatQueryBase, check: typeof checkCompat = checkCompat): PlanRow[] {
  const memo = new Map<string, CompatResult>();
  return rows.map((row) => {
    const gpu = row.layout.gpu;
    if (gpu.cc === 0) return row;
    // A pipeline layout also needs the engine's pipeline support for the model (p:pp).
    const pipeline = row.layout.pp > 1;
    const key = `${gpu.id}/${pipeline}`;
    const r = memo.get(key) ?? (memo.set(key, check({ model: q.model, repo: q.repo, engine: q.engine, gpu: gpu.id, kvCache: q.kvCache, platform: 'linux', ...(pipeline ? { pipeline } : {}) })).get(key) as CompatResult);
    // Where the matrix has a verdict for this model on this card, it replaces the generic format warnings.
    const verified = r.status === 'ok' || r.status === 'partial';
    const warnings = verified ? row.check.warnings.filter((w) => !FORMAT_WARNING.test(w)) : [...row.check.warnings];
    const blockers = [...row.check.blockers];
    // The reason carries each need's own workaround (planner/compat.ts needText).
    if (r.status === 'does-not-start') blockers.unshift(`${r.reason}${source(r)}`);
    else if (r.status === 'partial') warnings.push(`partly supported: ${r.reason}${source(r)}`);
    else if (r.status === 'unknown') warnings.push(`engine support on this card not verified: ${r.reason}`);
    if (r.wslOnly) warnings.push(`under Windows / WSL the engine is "${r.wslStatus}" (on Linux "${r.linuxStatus}")`);
    if (r.status !== 'does-not-start') return warnings.length === row.check.warnings.length && warnings.every((w, i) => w === row.check.warnings[i]) ? row : { ...row, check: { blockers, warnings } };
    // It does not start: no verdict, no speed, nothing to recommend.
    const { capacity: _c, speedBasis: _b, roofline: _r, ...rest } = row;
    return { ...rest, verdict: 'impossible', check: { blockers, warnings } };
  });
}

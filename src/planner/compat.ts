/**
 * Does a model start? (model, weight format, engine, version, GPU) → ok | partial | does-not-start | unknown, with
 * the reason and its sources, from the compatibility matrix data/compat.json.
 *
 * A model needs several things at once: kernels for its weight format (linear layers and MoE experts), a backend for
 * each kind of attention it has, the KV cache type, and sometimes something of its own (DeepSeek V4.1's Engram). Each
 * is looked up in the rules for the GPU's generation; the answer is the worst of them, in the order
 * does-not-start > unknown > partial > ok. A need without any verified rule is "unknown", never "ok".
 * Rules seen only under Windows / WSL2 are kept apart: `status` is the Linux answer unless the query asks for WSL,
 * and `wslOnly` says when WSL is worse than Linux.
 */
import compatJson from '../../data/compat.json';
import hardwareJson from '../../data/hardware.json';
import {
  compatFileSchema,
  type CompatArch,
  type CompatEngine,
  type CompatFile,
  type CompatModel,
  type CompatRule,
  type CompatSource,
  type CompatStatus,
} from '../contract/compat';

export type { CompatArch, CompatEngine, CompatStatus } from '../contract/compat';

const SEVERITY: Record<CompatStatus, number> = { ok: 0, partial: 1, unknown: 2, 'does-not-start': 3 };
const worst = (statuses: CompatStatus[]): CompatStatus =>
  statuses.reduce<CompatStatus>((a, b) => (SEVERITY[b] > SEVERITY[a] ? b : a), 'ok');

let bundled: CompatFile | undefined;
/** The matrix bundled with the command (parsed once). */
export function compatData(): CompatFile {
  bundled ??= compatFileSchema.parse(compatJson);
  return bundled;
}

export interface CompatQuery {
  /** Id of models.json ("deepseek-v4-1-flash"). */
  model: string;
  /** A checkpoint of the model (Hugging Face repo); wins over `format`. */
  repo?: string;
  /**
   * The catalog's precision label ("fp8-block", "mixed:nvfp4+fp8") or a coarse one ("bf16", "fp8", "nvfp4", "mxfp4",
   * "int4"). Without repo and format: the model's own checkpoint (the first of its quantizations).
   */
  format?: string;
  engine: CompatEngine;
  /** "0.31.0" or "v0.31.0"; without it, the version the matrix was verified on. */
  version?: string;
  /** An id of hardware.json ("rtx-5090-32gb"), a compute capability ("12.0") or an architecture id ("sm120"). */
  gpu: string;
  /** "fp8" adds the fp8 KV cache checks (kv:fp8, or kv:fp8-mla for MLA / sparse MLA models). */
  kvCache?: 'auto' | 'fp8';
  /** "linux" (default: servers) or "windows-wsl" (vLLM / SGLang under WSL2 on a Windows workstation). */
  platform?: 'linux' | 'windows-wsl';
  /** The layout splits the model into pipeline stages (pp > 1): adds p:pp, the engine's pipeline support for the model. */
  pipeline?: boolean;
}

/** The verdict for one thing the model needs. */
export interface CompatDetail {
  feature: string;
  status: CompatStatus;
  reason: string;
  sources: string[];
  /** The workaround of the rules that decided it on Linux, and under WSL2 (wslWorkaround). */
  workaround?: string;
  fixedIn?: string;
  /** Rules that decided it, on Linux and under WSL2 (ids of compat.json). */
  rules: string[];
  /** Under Windows / WSL2 (equal to the Linux ones unless a WSL-only rule is worse). */
  wslStatus: CompatStatus;
  wslReason: string;
  wslSources: string[];
  wslWorkaround?: string;
}

/** A need's reason followed by its own workaround, on Linux or under WSL2: never another need's workaround. */
export function needText(d: CompatDetail, wsl = false): string {
  const fix = wsl ? d.wslWorkaround : d.workaround;
  return `${wsl ? d.wslReason : d.reason}${fix !== undefined ? ` Workaround: ${fix}` : ''}`;
}

/** Needs' texts one after the other, each ended by a full stop (a workaround often is not) so none runs into the next. */
export function joinNeeds(texts: readonly string[]): string {
  return texts.map((t, i) => (i < texts.length - 1 && !/[.!?]$/.test(t) ? `${t}.` : t)).join(' ');
}

export interface CompatResult {
  status: CompatStatus;
  /** Why, from the needs that decided the status (empty for ok). */
  reason: string;
  /** Links (code at the release tag, issues, PRs, docs) behind the status. */
  sources: string[];
  /** Linux is better than WSL2: the limit applies only under Windows / WSL. */
  wslOnly: boolean;
  linuxStatus: CompatStatus;
  wslStatus: CompatStatus;
  arch: CompatArch | null;
  engine: CompatEngine;
  /** The engine version the matrix was verified on. */
  verifiedVersion: string;
  /** The checkpoint that was evaluated. */
  repo: string | null;
  workarounds: string[];
  details: CompatDetail[];
}

/** A source as a link: code at the tag with its lines, an issue or PR, a page. */
export function sourceUrl(s: CompatSource): string {
  if (s.kind === 'code') return `https://github.com/${s.repo}/blob/${s.ref}/${s.path}#L${s.lines[0]}-L${s.lines[1]}`;
  if (s.kind === 'docs') return s.url;
  return `https://github.com/${s.repo}/${s.kind === 'pr' ? 'pull' : 'issues'}/${s.number}`;
}

/** The generation of a GPU: hardware.json id, compute capability or architecture id; null when unknown. */
export function archOf(gpu: string, data: CompatFile = compatData()): CompatArch | null {
  const byId = data.architectures.find((a) => a.id === gpu);
  if (byId) return byId.id;
  const cc = /^\d+\.\d+$/.test(gpu) ? gpu : hardwareJson.gpus.find((g) => g.id === gpu)?.computeCapability;
  if (cc === undefined) return null;
  return data.architectures.find((a) => a.computeCapabilities.includes(cc))?.id ?? null;
}

const COARSE: Record<string, string[]> = {
  bf16: ['bf16'],
  fp8: ['fp8-block', 'fp8-channel', 'fp8-tensor', 'fp8', 'mxfp8'],
  nvfp4: ['nvfp4'],
  mxfp4: ['mxfp4'],
  int4: ['w4a16-int4', 'gptq-int4', 'awq-int4'],
};
/** The checkpoint to evaluate: by repo, by exact label, by coarse label (catalog order: official first), or the model's own. */
function pickVariant(model: CompatModel, repo?: string, format?: string) {
  if (repo) return model.variants.find((v) => v.repo.toLowerCase() === repo.toLowerCase());
  if (!format) return model.variants[0];
  const exact = model.variants.find((v) => v.format === format);
  if (exact) return exact;
  const labels = COARSE[format];
  if (!labels) return undefined;
  // The format of the experts / of most weights: the first part of a mixed label.
  const main = (f: string) => (f.startsWith('mixed:') ? f.slice(6).split('+')[0]! : f);
  return model.variants.find((v) => labels.includes(main(v.format)));
}

const normVersion = (v: string) => v.replace(/^v/, '');
const ENGINE_NAME: Record<CompatEngine, string> = { vllm: 'vLLM', sglang: 'SGLang' };

function unknownResult(q: CompatQuery, data: CompatFile, arch: CompatArch | null, reason: string, repo: string | null = null): CompatResult {
  return {
    status: 'unknown',
    reason,
    sources: [],
    wslOnly: false,
    linuxStatus: 'unknown',
    wslStatus: 'unknown',
    arch,
    engine: q.engine,
    verifiedVersion: data.engines.find((e) => e.engine === q.engine)?.version ?? '',
    repo,
    workarounds: [],
    details: [],
  };
}

/** The rules of one feature for a model on an architecture: the model's own replace the general ones. */
function rulesFor(data: CompatFile, engine: CompatEngine, feature: string, arch: CompatArch, modelId: string): CompatRule[] {
  const all = data.rules.filter((r) => r.engine === engine && r.feature === feature && r.archs.includes(arch));
  const own = all.filter((r) => r.models?.includes(modelId));
  return own.length > 0 ? own : all.filter((r) => r.models === undefined);
}

function detail(feature: string, rules: CompatRule[]): CompatDetail {
  const general = rules.filter((r) => r.platform === 'all');
  const wslRules = rules.filter((r) => r.platform === 'wsl');
  if (general.length === 0) {
    const reason = `${feature}: not verified on this architecture.`;
    return { feature, status: 'unknown', reason, sources: [], rules: [], wslStatus: 'unknown', wslReason: reason, wslSources: [] };
  }
  const status = worst(general.map((r) => r.status));
  const wslStatus = worst([status, ...wslRules.map((r) => r.status)]);
  // The rules that decided it: the worst general ones; under WSL, the WSL ones when they are worse.
  const deciding = general.filter((r) => r.status === status);
  const decidingWsl = wslStatus === status ? deciding : wslRules.filter((r) => r.status === wslStatus);
  // One reason per feature: overlapping rules restate the same fact, so the most specific one (fewest architectures)
  // speaks for all; their sources are all kept.
  const text = (rs: CompatRule[]) => {
    const r = [...rs].sort((a, b) => a.archs.length - b.archs.length)[0] as CompatRule;
    return r.platform === 'wsl' ? `Windows/WSL2 only: ${r.reason}` : r.reason;
  };
  const links = (rs: CompatRule[]) => [...new Set(rs.flatMap((r) => r.sources.map(sourceUrl)))];
  const both = [...new Set([...deciding, ...decidingWsl])];
  const fixes = (rs: CompatRule[]) => [...new Set(rs.map((r) => r.workaround).filter((w): w is string => w !== undefined))].join(' ');
  const workaround = fixes(deciding);
  const wslWorkaround = fixes(decidingWsl);
  const fixedIn = both.map((r) => r.fixedIn).filter((w): w is string => w !== undefined);
  return {
    feature,
    status,
    wslStatus,
    reason: text(deciding),
    wslReason: text(decidingWsl),
    sources: links(deciding),
    wslSources: links(decidingWsl),
    rules: both.map((r) => r.id),
    ...(workaround ? { workaround } : {}),
    ...(wslWorkaround ? { wslWorkaround } : {}),
    ...(fixedIn.length ? { fixedIn: fixedIn.join(' ') } : {}),
  };
}

const MLA_FEATURES = ['a:mla', 'a:dsa', 'a:dsv4'];

/** The feature a pipeline-parallel layout adds: the engine must implement pipeline stages for the model's class. */
export const PIPELINE_FEATURE = 'p:pp';

/** Whether the model, in that format, with that engine and version, starts on that GPU, and why. */
export function checkCompat(q: CompatQuery, data: CompatFile = compatData()): CompatResult {
  const arch = archOf(q.gpu, data);
  const ev = data.engines.find((e) => e.engine === q.engine);
  if (!ev) return unknownResult(q, data, arch, `Engine ${q.engine} is not in the matrix.`);
  if (!arch) return unknownResult(q, data, arch, `GPU "${q.gpu}" is not recognized (neither a hardware.json id nor a compute capability of the matrix).`);
  if (q.version !== undefined && normVersion(q.version) !== ev.version)
    return unknownResult(q, data, arch, `The matrix was verified on ${ENGINE_NAME[q.engine]} ${ev.version} only; version ${normVersion(q.version)} has not been checked.`);
  const model = data.models.find((m) => m.id === q.model);
  if (!model) return unknownResult(q, data, arch, `Model "${q.model}" is not in the matrix.`);
  const variant = pickVariant(model, q.repo, q.format);
  if (!variant) return unknownResult(q, data, arch, `No ${q.repo ?? q.format} checkpoint of ${q.model} in the catalog.`);
  if (variant.outOfScope) return unknownResult(q, data, arch, variant.outOfScope, variant.repo);
  if (variant.weightsUnknown) return unknownResult(q, data, arch, `Weight format of ${variant.repo} unknown: ${variant.weightsUnknown}`, variant.repo);

  const features = [...variant.weights, ...model.features];
  if (q.kvCache === 'fp8') features.push(model.features.some((f) => MLA_FEATURES.includes(f)) ? 'kv:fp8-mla' : 'kv:fp8');
  if (q.pipeline) features.push(PIPELINE_FEATURE);
  // The model's own needs (m:*: its class, loader, a blocker of its own). Its rules about a weight format or the KV
  // cache apply only when the checkpoint or the query has them.
  for (const r of data.rules)
    if (r.engine === q.engine && r.feature.startsWith('m:') && r.models?.includes(model.id) && r.archs.includes(arch) && !features.includes(r.feature)) features.push(r.feature);

  const details = [...new Set(features)].map((f) => detail(f, rulesFor(data, q.engine, f, arch, model.id)));
  const linuxStatus = worst(details.map((d) => d.status));
  const wslStatus = worst(details.map((d) => d.wslStatus));
  const wsl = q.platform === 'windows-wsl';
  const status = wsl ? wslStatus : linuxStatus;
  const deciding = details.filter((d) => (wsl ? d.wslStatus : d.status) === status && status !== 'ok');
  // The workarounds of the needs that decide the status; on Linux, when only WSL2 is worse, those that decide it there.
  const fixable = deciding.length > 0 ? deciding.map((d) => (wsl ? d.wslWorkaround : d.workaround)) : details.filter((d) => d.wslStatus === wslStatus && wslStatus !== 'ok').map((d) => d.wslWorkaround);
  return {
    status,
    // Each need's reason with its own workaround after it.
    reason: joinNeeds(deciding.map((d) => needText(d, wsl))),
    sources: [...new Set(deciding.flatMap((d) => (wsl ? d.wslSources : d.sources)))],
    wslOnly: SEVERITY[wslStatus] > SEVERITY[linuxStatus],
    linuxStatus,
    wslStatus,
    arch,
    engine: q.engine,
    verifiedVersion: ev.version,
    repo: variant.repo,
    workarounds: [...new Set(fixable.filter((w): w is string => w !== undefined))],
    details,
  };
}

/** Status of a model's checkpoint on every architecture, for one engine (the summary table). */
export function compatRow(model: string, engine: CompatEngine, checkpoint?: { repo?: string; format?: string }, data: CompatFile = compatData()): Record<CompatArch, CompatResult> {
  return Object.fromEntries(data.architectures.map((a) => [a.id, checkCompat({ model, engine, gpu: a.id, ...checkpoint }, data)])) as Record<CompatArch, CompatResult>;
}

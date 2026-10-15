// Types of the pure functions of update-compat.mjs used by test/contract/compat.test.ts.
import type { CompatModel } from '../../src/contract/compat';
import type { ModelsFile } from '../../src/contract/models';

export const ATTENTION_FEATURES: Record<string, string[]>;
export const DSV4_CLASSES: string[];
export const OUT_OF_SCOPE_FORMATS: Record<string, string>;

interface SummaryEntry {
  pattern: string;
  dtype: string;
  shape: number[];
  group: string;
}
export function linearWeightDtypes(summary: { entries: SummaryEntry[] }): { moe: Set<string>; dense: Set<string> };
export function variantWeights(format: string, moe: boolean, summary: { entries: SummaryEntry[] } | null): { weights: string[]; outOfScope?: string; unknown?: string };
export function deriveCompatModel(entry: ModelsFile['models'][number], fixtures?: string): CompatModel;
export function deriveCompatModels(catalog: ModelsFile, fixtures?: string): CompatModel[];
export function anchorLines(text: string, anchor: string): number[];
export function checkAnchor(
  source: { lines: [number, number]; anchor: string },
  text: string | null,
  anchorAt?: number | null,
): { state: 'ok'; lines: [number, number] } | { state: 'moved'; candidates: [number, number][] } | { state: 'missing' | 'missing-file' };

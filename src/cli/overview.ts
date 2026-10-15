/**
 * The overview the command and the ready-made reports share: every layout of the cards planned and ranked, the
 * compatibility matrix applied, one line per card and the general answer. `run.ts` and
 * scripts/dev/report-lib.ts both call it, so data/report.json is what the command prints for the same model.
 */
import { layoutsFor, type Gpu } from '../planner/hardware';
import { planLayout, rankRows, type PlannerModel, type PlanOptions, type PlanRow } from '../planner/plan';
import type { Verification } from '../planner/weights-verification';
import { applyCompat, catalogModelId } from './compat';
import { cardLines, figuresOf, generalAnswer, type FigureContext, type Figures, type UsersAsked } from './figures';

export interface Overview {
  ctx: FigureContext;
  /** The layouts of the cards asked, ranked, with the compatibility matrix applied. */
  rows: PlanRow[];
  /** One line per card (and this PC's, planned apart when the list leaves it out). */
  cards: ReturnType<typeof cardLines>;
  general: ReturnType<typeof generalAnswer>;
  /** A row's figures, computed once. */
  fig: (r: PlanRow) => Figures;
  /** The model is in the compatibility matrix. */
  compatChecked: boolean;
}

export interface OverviewInput {
  model: PlannerModel & { verification: Verification };
  repo: string;
  opts: PlanOptions;
  gpus: readonly Gpu[];
  /** GPUs per layout tried. */
  counts: readonly number[];
  users?: UsersAsked;
  /** This PC's card when the list leaves it out (planned apart, shown as one more line). */
  apart?: Gpu;
  localId?: string;
}

export function planOverview(i: OverviewInput): Overview {
  // The compatibility matrix knows the catalog's models only.
  const modelId = catalogModelId(i.repo);
  const compat = (rows: PlanRow[]) => (modelId ? applyCompat(rows, { model: modelId, repo: i.repo, engine: i.opts.engine, kvCache: i.opts.kvCache }) : rows);
  const plan = (list: readonly Gpu[]) => rankRows(compat(layoutsFor(list, i.counts).map((l) => planLayout(i.model, l, i.opts))));
  const rows = plan(i.gpus);
  const ctx: FigureContext = { model: i.model, opts: i.opts, repo: i.repo, verification: i.model.verification };
  const memo = new Map<PlanRow, Figures>();
  const fig = (r: PlanRow) => memo.get(r) ?? (memo.set(r, figuresOf(ctx, r, i.users)).get(r) as Figures);
  const cards = cardLines(ctx, [...rows, ...(i.apart ? plan([i.apart]) : [])], i.apart ? [...i.gpus, i.apart] : i.gpus, i.users, i.localId);
  return { ctx, rows, cards, general: generalAnswer(rows, i.users), fig, compatChecked: modelId !== undefined };
}

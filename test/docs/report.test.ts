// data/report.json (the ready-made reports of `report`) is computed from the code and the data: it carries the
// fingerprint of its inputs, and a sample of models is recomputed here through the command's own overview and
// compared value for value. After a change to the planner, the output or the data: npm run dev:report.
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { reportFileSchema, unpackReport } from '../../src/cli/report';
import { buildReport, reportInputs } from '../../scripts/dev/report-lib';

const outOfDate = 'data/report.json is out of date: run npm run dev:report';
const raw = JSON.parse(readFileSync('data/report.json', 'utf8')) as { inputs?: string; packed: Parameters<typeof unpackReport>[0] };
// As written (the schema would reorder the keys).
const stored = unpackReport(raw.packed) as { models: { id: string }[] };

describe('data/report.json', () => {
  it('was computed from the current code, data and model fixtures', async () => {
    expect(raw.inputs === (await reportInputs('.')), outOfDate).toBe(true);
  });

  // A MoE with MLA, a dense model with a 1M YaRN context and a MoE in MXFP4 (in the catalog's order).
  const sample = ['deepseek-v4-1-flash', 'gemma-4-31b', 'gpt-oss-120b'];
  it('holds what the command computes now, for a sample of models', async () => {
    const fresh = await buildReport('.', sample);
    const mine = stored.models.filter((m) => sample.includes(m.id));
    expect(mine.map((m) => m.id)).toEqual(sample);
    expect(JSON.stringify(fresh.models) === JSON.stringify(mine), outOfDate).toBe(true);
    // The command reads it.
    expect(reportFileSchema.parse(raw).models).toHaveLength(stored.models.length);
  }, 120_000);
});

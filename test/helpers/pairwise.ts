/**
 * All-pairs (pairwise) combinations: a short list of cases in which every value of every parameter meets every
 * value of every other parameter at least once. Most bugs need two options together to show; all pairs reach
 * them with a few dozen cases instead of the full product (thousands).
 *
 * Greedy and deterministic (no randomness, the same list on every run): each new case starts from the first
 * pair not yet covered, then every other parameter takes the value that covers the most pairs still missing
 * (ties: the value used least so far, then the first one).
 */

/** The parameters: a name and its values each. */
export type Parameters = Record<string, readonly unknown[]>;
/** One case: a value per parameter. */
export type Case<P extends Parameters> = { [K in keyof P]: P[K][number] };

export function allPairs<P extends Parameters>(params: P): Case<P>[] {
  const keys = Object.keys(params) as (keyof P & string)[];
  const sizes = keys.map((k) => (params[k] as readonly unknown[]).length);
  if (sizes.some((n) => n === 0)) throw new Error('allPairs: every parameter needs at least one value');
  const toCase = (row: number[]) => Object.fromEntries(keys.map((k, i) => [k, (params[k] as readonly unknown[])[row[i] as number]])) as Case<P>;
  if (keys.length === 0) return [];
  if (keys.length === 1) return Array.from({ length: sizes[0] as number }, (_, v) => toCase([v]));

  const pairKey = (i: number, a: number, j: number, b: number) => (i < j ? `${i}:${a}|${j}:${b}` : `${j}:${b}|${i}:${a}`);
  // Insertion order of a Set is stable: the "first missing pair" is always the same one.
  const missing = new Set<string>();
  for (let i = 0; i < keys.length; i++)
    for (let j = i + 1; j < keys.length; j++)
      for (let a = 0; a < (sizes[i] as number); a++) for (let b = 0; b < (sizes[j] as number); b++) missing.add(pairKey(i, a, j, b));
  const uses = sizes.map((n) => new Array<number>(n).fill(0));
  const rows: number[][] = [];
  while (missing.size > 0) {
    const [first] = missing;
    const [[i, a], [j, b]] = (first as string).split('|').map((p) => p.split(':').map(Number)) as [[number, number], [number, number]];
    const row = new Array<number>(keys.length).fill(-1);
    row[i] = a;
    row[j] = b;
    for (let k = 0; k < keys.length; k++) {
      if (row[k] !== -1) continue;
      let best = 0;
      let bestScore = -1;
      for (let v = 0; v < (sizes[k] as number); v++) {
        let covers = 0;
        for (let m = 0; m < keys.length; m++) if (row[m] !== -1 && missing.has(pairKey(k, v, m, row[m] as number))) covers++;
        // More new pairs first; then the value used least (spreads the values evenly).
        const score = covers * 1e6 - (uses[k]?.[v] as number);
        if (score > bestScore) {
          best = v;
          bestScore = score;
        }
      }
      row[k] = best;
    }
    for (let x = 0; x < keys.length; x++) {
      const used = uses[x] as number[];
      used[row[x] as number] = (used[row[x] as number] as number) + 1;
      for (let y = x + 1; y < keys.length; y++) missing.delete(pairKey(x, row[x] as number, y, row[y] as number));
    }
    rows.push(row);
  }
  return rows.map(toCase);
}

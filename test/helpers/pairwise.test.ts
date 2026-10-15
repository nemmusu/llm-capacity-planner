import { describe, expect, it } from 'vitest';
import { allPairs, type Parameters } from './pairwise';

/** Every pair of values of two different parameters that no case of `cases` holds. */
function missingPairs(params: Parameters, cases: Record<string, unknown>[]): string[] {
  const keys = Object.keys(params);
  const out: string[] = [];
  for (let i = 0; i < keys.length; i++)
    for (let j = i + 1; j < keys.length; j++)
      for (const a of params[keys[i] as string] as unknown[])
        for (const b of params[keys[j] as string] as unknown[])
          if (!cases.some((c) => c[keys[i] as string] === a && c[keys[j] as string] === b)) out.push(`${keys[i]}=${String(a)} × ${keys[j]}=${String(b)}`);
  return out;
}

describe('allPairs', () => {
  it('covers every pair of values of every two parameters, with far fewer cases than the full product', () => {
    const params = { a: [1, 2, 3], b: ['x', 'y'], c: [true, false], d: ['p', 'q', 'r', 's'], e: [0, 1, 2] };
    const cases = allPairs(params);
    expect(missingPairs(params, cases)).toEqual([]);
    // At least the largest two parameters' product; far below 3·2·2·4·3 = 144.
    expect(cases.length).toBeGreaterThanOrEqual(12);
    expect(cases.length).toBeLessThanOrEqual(20);
    // Every case holds one valid value per parameter.
    for (const c of cases) for (const [k, values] of Object.entries(params)) expect(values).toContain(c[k as keyof typeof c]);
  });

  it('is deterministic: the same list on every call', () => {
    const params = { engine: ['vllm', 'sglang'], gpu: ['none', 'h100', '5090', 'l40s'], tokens: ['8k', '32k', '128k'], fmt: ['csv', 'json'] };
    expect(allPairs(params)).toEqual(allPairs(params));
    expect(missingPairs(params, allPairs(params))).toEqual([]);
  });

  it('covers the pairs of a wide parameter set (12 parameters, like the CLI combinations)', () => {
    const params = {
      p1: [1, 2],
      p2: [1, 2],
      p3: [1, 2, 3, 4, 5],
      p4: [1, 2, 3],
      p5: [1, 2],
      p6: [1, 2],
      p7: [1, 2, 3],
      p8: [1, 2, 3, 4],
      p9: [1, 2, 3, 4],
      p10: [1, 2, 3],
      p11: [1, 2],
      p12: [1, 2],
    };
    const cases = allPairs(params);
    expect(missingPairs(params, cases)).toEqual([]);
    expect(cases.length).toBeLessThanOrEqual(40);
  });

  it('one parameter: each value once; none: no case; a parameter without values is refused', () => {
    expect(allPairs({ only: ['a', 'b', 'c'] })).toEqual([{ only: 'a' }, { only: 'b' }, { only: 'c' }]);
    expect(allPairs({})).toEqual([]);
    expect(() => allPairs({ a: [1], b: [] })).toThrow(/at least one value/);
  });

  it('two parameters: exactly the full product', () => {
    const cases = allPairs({ a: [1, 2, 3], b: ['x', 'y'] });
    expect(cases).toHaveLength(6);
    expect(missingPairs({ a: [1, 2, 3], b: ['x', 'y'] }, cases)).toEqual([]);
  });
});

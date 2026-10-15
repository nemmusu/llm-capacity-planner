import { describe, expect, it } from 'vitest';
import { asciiOnly, gb, GB, int, measureLong, measureShort, num, plusMinus, precisionShort, precisionText, SECONDS, secs, TOKS, tps, wrap } from '../../src/cli/format';
import { measure, type Precision } from '../../src/cli/figures';

const p = (kind: Precision['kind'], over: Partial<Precision> = {}): Precision => ({ kind, errorPct: 12, basis: 'public vLLM benchmarks', ...over });

describe('numbers in English', () => {
  it('decimals, integers, GB, tokens/s and seconds', () => {
    expect(num(12345.56)).toBe('12,345.6');
    expect(num(2, 2)).toBe('2.00');
    expect(int(12345.6)).toBe('12,346');
    expect(gb(71.44)).toBe('71.4');
    expect([tps(7.25), tps(74.4)]).toEqual(['7.3', '74']);
    expect([secs(0.456), secs(9.14), secs(99.94), secs(123.4)]).toEqual(['0.46', '9.1', '99.9', '123']);
  });

  it('±: one decimal below 10 %, none above', () => {
    expect([plusMinus(2.34), plusMinus(12.4), plusMinus(0)]).toEqual(['±2.3%', '±12%', '±0.0%']);
  });
});

describe('measures', () => {
  const memory = measure(69.8, 73.2, { kind: 'estimated', basis: 'x' });
  it('the long form: value, the absolute bounds and the error; the unit repeated for GiB and seconds', () => {
    expect(measureLong(memory, GB)).toBe('71.5 GiB (69.8 to 73.2 GiB, ±2.4%)');
    expect(measureLong(measure(65, 83, { kind: 'calibrated', basis: 'x' }), TOKS)).toBe('74 tok/s (65 to 83, ±12%)');
    expect(measureLong(measure(6.8, 11.4, { kind: 'calibrated', basis: 'x' }), SECONDS, 'calibrated')).toBe('9.1 s (6.8 to 11.4 s, ±25%, calibrated)');
  });

  it('an exact value alone, with its tag when given', () => {
    expect(measureLong(measure(12, 12, { kind: 'exact', basis: 'x' }), GB)).toBe('12.0 GiB');
    expect(measureLong(measure(12, 12, { kind: 'exact', basis: 'x' }), GB, 'exact')).toBe('12.0 GiB (exact)');
  });

  it('the short form for narrow tables', () => {
    expect(measureShort(memory, GB, '–')).toBe('71.5 (69.8–73.2)');
    expect(measureShort(measure(5, 5, { kind: 'exact', basis: 'x' }), GB, '-')).toBe('5.0');
  });
});

describe('accuracy labels', () => {
  it('the five kinds, long', () => {
    expect(precisionText(p('exact'))).toBe('exact');
    expect(precisionText(p('exact', { reason: 'format not verified' }))).toBe('exact (format not verified)');
    expect(precisionText(p('measured', { errorPct: 11 }))).toBe('measured (±11%)');
    expect(precisionText(p('calibrated', { points: 46 }))).toBe('calibrated (±12%, 46 points)');
    expect(precisionText(p('calibrated'))).toBe('calibrated (±12%)');
    expect(precisionText(p('estimated', { points: 46, reason: 'calibrated on other cards' }))).toBe('estimated (±12%, calibrated on other cards)');
    expect(precisionText(p('imprecise', { errorPct: 61, reason: 'SGLang' }))).toBe('not precise enough to decide (±61%, SGLang)');
  });

  it('short, for tables', () => {
    expect(precisionShort(p('exact'))).toBe('exact');
    expect(precisionShort(p('calibrated'))).toBe('calibrated ±12%');
    expect(precisionShort(p('imprecise', { errorPct: 61 }))).toBe('not precise ±61%');
  });
});

describe('text', () => {
  it('wraps words at the width, the next lines indented', () => {
    expect(wrap('one two three four', 8)).toEqual(['one two', 'three', 'four']);
    expect(wrap('one two three', 8, 2)).toEqual(['one two', '  three']);
    expect(wrap('averyveryverylongword short', 5)).toEqual(['averyveryverylongword', 'short']);
  });

  it('ASCII for old consoles: dashes, signs and quotes', () => {
    expect(asciiOnly('2× – — ± … “a” ’ ≥ ≤ ≈ · → ÷')).toBe("2x - - +- ... \"a\" ' >= <= ~ - -> /");
  });
});

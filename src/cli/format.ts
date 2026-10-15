/**
 * How the capacity command writes its numbers: "71.4 GiB (69.8 to 73.1 GiB, ±2.3%)", the short form
 * "71.4 (69.8–73.1)" for narrow tables, the accuracy labels, text wrapped to the terminal and plain ASCII for
 * consoles without the special characters. English number format (decimal point, "," between thousands).
 */
import type { Measure, Precision } from './figures';

/** A number with `digits` decimals ("1,234.5"). */
export const num = (n: number, digits = 1) => n.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
/** An integer ("1,234"). */
export const int = (n: number) => Math.round(n).toLocaleString('en-US');

/** GiB (2^30 bytes, the unit nvidia-smi and CUDA count in) with one decimal. */
export const gb = (n: number) => num(n);
/** Tokens/s: no decimals, one below 10. */
export const tps = (n: number) => (n < 10 ? num(n) : int(n));
/** Seconds: two decimals below 1 s, one below 100 s, none above. */
export const secs = (n: number) => (n < 1 ? num(n, 2) : n < 100 ? num(n) : int(n));

/** "±2.3%" (one decimal below 10 %), "±12%". */
export const plusMinus = (pct: number) => `±${pct < 10 || (pct < 50 && Math.round(pct) >= 50) ? num(pct) : int(pct)}%`;

export interface Unit {
  /** Formats a value. */
  fmt: (n: number) => string;
  /** "GiB", "tok/s", "s". */
  unit: string;
  /** Repeat the unit after the max ("69.8 to 73.1 GiB"). */
  repeat: boolean;
}

export const GB: Unit = { fmt: gb, unit: 'GiB', repeat: true };
export const TOKS: Unit = { fmt: tps, unit: 'tok/s', repeat: false };
export const SECONDS: Unit = { fmt: secs, unit: 's', repeat: true };

/** The bounds collapse when they print the same. */
const same = (m: Measure, u: Unit) => u.fmt(m.min) === u.fmt(m.max);

/** "71.4 GiB (69.8 to 73.1 GiB, ±2.3%)"; an exact value alone ("12.0 GiB"); `tag` goes inside the brackets. */
export function measureLong(m: Measure, u: Unit, tag?: string): string {
  if (same(m, u)) return `${u.fmt(m.value)} ${u.unit}${tag ? ` (${tag})` : ''}`;
  return `${u.fmt(m.value)} ${u.unit} (${u.fmt(m.min)} to ${u.fmt(m.max)}${u.repeat ? ` ${u.unit}` : ''}, ${plusMinus(m.precision.errorPct)}${tag ? `, ${tag}` : ''})`;
}

/** "71.4 (69.8–73.1)" (no unit: the column says it). */
export function measureShort(m: Measure, u: Unit, dash: string): string {
  return same(m, u) ? u.fmt(m.value) : `${u.fmt(m.value)} (${u.fmt(m.min)}${dash}${u.fmt(m.max)})`;
}

const KIND: Record<Precision['kind'], string> = { exact: 'exact', measured: 'measured', calibrated: 'calibrated', estimated: 'estimated', imprecise: 'not precise enough to decide' };

/** "exact", "measured (±11%)", "calibrated (±12%, 46 points)", "estimated (±24%, calibrated on other cards)". */
export function precisionText(p: Precision): string {
  if (p.kind === 'exact') return p.reason ? `exact (${p.reason})` : 'exact';
  const parts = [plusMinus(p.errorPct)];
  if (p.kind === 'calibrated' && p.points !== undefined) parts.push(`${p.points} points`);
  if (p.reason) parts.push(p.reason);
  return `${KIND[p.kind]} (${parts.join(', ')})`;
}

/** "calibrated ±12%", "not precise ±60%" (tables). */
export function precisionShort(p: Precision): string {
  return p.kind === 'exact' ? 'exact' : `${p.kind === 'imprecise' ? 'not precise' : p.kind} ${plusMinus(p.errorPct)}`;
}

/** Words wrapped at `width`, the next lines indented by `indent` spaces. */
export function wrap(text: string, width: number, indent = 0): string[] {
  const out: string[] = [];
  let line = '';
  for (const word of text.split(' ')) {
    const next = line ? `${line} ${word}` : word;
    if (next.length > width && line) {
      out.push(line);
      line = `${' '.repeat(indent)}${word}`;
    } else line = next;
  }
  out.push(line);
  return out;
}

/** Plain ASCII for consoles that cannot show the special characters (tables apply it before measuring). */
export function asciiOnly(text: string): string {
  return text
    .replace(/[–—]/g, '-')
    .replace(/×/g, 'x')
    .replace(/±/g, '+-')
    .replace(/…/g, '...')
    .replace(/[“”]/g, '"')
    .replace(/[’]/g, "'")
    .replace(/≥/g, '>=')
    .replace(/≤/g, '<=')
    .replace(/≈/g, '~')
    .replace(/·/g, '-')
    .replace(/→/g, '->')
    .replace(/÷/g, '/');
}

/**
 * Tokens as words and pages for readers who do not count tokens: English prose averages about 0.75 words per
 * token, and a page of text about 500 words (both declared in docs/METHODOLOGY.md).
 */
export const WORDS_PER_TOKEN = 0.75;
export const WORDS_PER_PAGE = 500;

/** Two significant digits, the precision of a conversion like this one ("98,000", "200", "4"). */
export const round2 = (n: number) => {
  if (n < 10) return Math.max(1, Math.round(n));
  const step = 10 ** (Math.floor(Math.log10(n)) - 1);
  return Math.round(n / step) * step;
};

/** "≈ 98,000 words ≈ 200 pages" for a number of tokens. */
export function wordsAndPages(tokens: number): string {
  const words = tokens * WORDS_PER_TOKEN;
  const pages = words / WORDS_PER_PAGE;
  return `≈ ${int(round2(words))} words ≈ ${int(round2(pages))} ${round2(pages) === 1 ? 'page' : 'pages'}`;
}

/** Tokens per second as words per second, rounded for a reader ("about 29"). */
export const wordsPerSecond = (tokS: number) => int(round2(tokS * WORDS_PER_TOKEN));

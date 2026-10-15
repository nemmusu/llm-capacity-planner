// The README's examples — images in docs/images and the generated blocks of README.md — are the command's current
// output: regenerated here and compared byte for byte, so they never fall behind the code.
// After a change to the output or the data: npm run dev:screenshots.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { parseAnsiLine, terminalSvg, wrapRuns } from '../../scripts/dev/ansi-svg';
import { readmeAccuracy, readmeExamples, readmeTop, SCREENSHOTS, screenshotSvg, withBlocks } from '../../scripts/dev/screenshots-lib';
import { catalogBuiltin, HUB } from '../helpers/catalog';

const outOfDate = (what: string) => `${what} is out of date: run npm run dev:screenshots`;

// The first test also packs the catalog models (catalogBuiltin, about 30 s on a slow disk); the README blocks take about a
// dozen runs of the command, slow under the coverage instrumentation.
describe('README examples', { timeout: 120_000 }, () => {
  it.each(SCREENSHOTS.map((s) => [s.file, s] as const))('docs/images/%s matches the command', async (file, s) => {
    const svg = await screenshotSvg(s, catalogBuiltin(), HUB);
    expect(svg === readFileSync(path.join('docs', 'images', file), 'utf8'), outOfDate(`docs/images/${file}`)).toBe(true);
  });

  it('the generated blocks of README.md match the command', async () => {
    const readme = readFileSync('README.md', 'utf8');
    const fresh = withBlocks(readme, { top: await readmeTop(catalogBuiltin(), HUB), accuracy: await readmeAccuracy(catalogBuiltin(), HUB), examples: await readmeExamples(catalogBuiltin(), HUB) });
    expect(fresh === readme, outOfDate('README.md')).toBe(true);
    // Every image is shown, the scenarios are large models, and the measuring machine is not named.
    for (const s of SCREENSHOTS) expect(readme).toContain(`docs/images/${s.file}`);
    expect(readme).not.toMatch(/Qwen3-8B|Qwen3-32B|RTX 5090/);
  });

  it('a README without the markers is refused', () => {
    expect(() => withBlocks('no markers', { top: 'x' })).toThrow('no <!-- generated:top --> block');
  });
});

describe('ANSI to SVG', () => {
  it('reads colours, bold, dim and resets, and keeps the style across lines', () => {
    const esc = String.fromCharCode(27);
    const a = parseAnsiLine(`ab${esc}[1;32mcd${esc}[39mef${esc}[0m g${esc}[2mh${esc}[22mi${esc}[91mj${esc}[mk`);
    expect(a.runs.map((r) => [r.col, r.text, r.style.color ?? '', r.style.bold, r.style.dim])).toEqual([
      [0, 'ab', '', false, false],
      [2, 'cd', '#3fb950', true, false],
      [4, 'ef', '', true, false],
      [6, ' g', '', false, false],
      [8, 'h', '', false, true],
      [9, 'i', '', false, false],
      [10, 'j', '#ffa198', false, false],
      [11, 'k', '', false, false],
    ]);
    const open = parseAnsiLine(`${esc}[33mstill`, { bold: false, dim: false });
    expect(parseAnsiLine('yellow', open.end).runs[0]?.style.color).toBe('#d29922');
    // An unknown code changes nothing.
    expect(parseAnsiLine(`${esc}[38mx`).runs[0]?.style).toEqual({ bold: false, dim: false });
  });

  it('wraps at the terminal width as a terminal does, without rows of blanks', () => {
    const rows = wrapRuns(parseAnsiLine('abcdefgh   ').runs, 3);
    expect(rows.map((r) => r.map((x) => x.text).join(''))).toEqual(['abc', 'def', 'gh ']);
    expect(wrapRuns([], 3)).toEqual([[]]);
  });

  it('draws a window with the prompt, escapes XML, and marks a cut', () => {
    const esc = String.fromCharCode(27);
    const svg = terminalSvg({ command: 'tool <a & b>', output: `${esc}[31mred${esc}[0m\n\n  x`, title: 't', columns: 10, cut: true });
    expect(svg).toContain('&lt;a &amp; b&gt;');
    expect(svg).toContain('fill="#ff7b72"');
    expect(svg).toContain('>…</text>');
    expect(terminalSvg({ command: 'c', output: 'x', title: 't', columns: 10 })).not.toContain('…');
  });
});

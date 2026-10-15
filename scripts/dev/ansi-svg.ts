// Terminal output (text with ANSI colour codes) as an SVG image that looks like a terminal window: dark background,
// monospace font, a title bar, the prompt with the command typed. Development only (scripts/dev/screenshots.mjs).
// Deterministic: the same text always gives the same bytes, so a test can compare the images with the command.

/** Colours of the image (GitHub's dark theme), by SGR code. */
const PALETTE = {
  background: '#0d1117',
  bar: '#161b22',
  border: '#30363d',
  text: '#c9d1d9',
  dim: '#8b949e',
  title: '#8b949e',
  command: '#e6edf3',
  fg: {
    30: '#484f58',
    31: '#ff7b72',
    32: '#3fb950',
    33: '#d29922',
    34: '#58a6ff',
    35: '#bc8cff',
    36: '#39c5cf',
    37: '#c9d1d9',
    90: '#6e7681',
    91: '#ffa198',
    92: '#56d364',
    93: '#e3b341',
    94: '#79c0ff',
    95: '#d2a8ff',
    96: '#56d4dd',
    97: '#f0f6fc',
  } as Record<number, string>,
};

const FONT = `ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace`;
const SIZE = 13;
/** Width of one character cell and height of one line, px (monospace fonts are about 0.6 em wide). */
const CELL = SIZE * 0.6;
const LINE = 18;
const PAD = 16;
const BAR = 30;

export interface Style {
  color?: string;
  bold: boolean;
  dim: boolean;
}
export interface Run {
  col: number;
  text: string;
  style: Style;
}

/** An SGR escape ("ESC [ codes m"). */
const SGR = new RegExp(`${String.fromCharCode(27)}\\[([0-9;]*)m`, 'g');

const escapeXml = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** One line split into runs of the same style, at their column; `end` is the style still on after it. */
export function parseAnsiLine(line: string, start: Style = { bold: false, dim: false }): { runs: Run[]; end: Style } {
  const runs: Run[] = [];
  let style = { ...start };
  let col = 0;
  let text = '';
  let at = 0;
  const flush = () => {
    if (text) runs.push({ col: col - [...text].length, text, style: { ...style } });
    text = '';
  };
  for (const m of line.matchAll(SGR)) {
    const chunk = line.slice(at, m.index);
    text += chunk;
    col += [...chunk].length;
    at = m.index + m[0].length;
    flush();
    for (const code of (m[1] || '0').split(';').map(Number)) {
      if (code === 0) style = { bold: false, dim: false };
      else if (code === 1) style.bold = true;
      else if (code === 2) style.dim = true;
      else if (code === 22) style = { ...style, bold: false, dim: false };
      else if (code === 39) delete style.color;
      else if (PALETTE.fg[code]) style.color = PALETTE.fg[code];
    }
  }
  const rest = line.slice(at);
  text += rest;
  col += [...rest].length;
  flush();
  return { runs, end: style };
}

/** The runs of one line laid out on terminal rows of `columns` cells, wrapping as a terminal does. */
export function wrapRuns(runs: Run[], columns: number): Run[][] {
  const rows: Run[][] = [[]];
  for (const r of runs) {
    const chars = [...r.text];
    let i = 0;
    while (i < chars.length) {
      const col = r.col + i;
      const row = Math.floor(col / columns);
      const take = Math.min(chars.length - i, columns - (col % columns));
      const text = chars.slice(i, i + take).join('');
      i += take;
      // Trailing blanks never open a row of their own.
      if (text.trim() === '' && row >= rows.length) continue;
      while (rows.length <= row) rows.push([]);
      (rows[row] as Run[]).push({ col: col % columns, text, style: r.style });
    }
  }
  return rows;
}

export interface TerminalImage {
  /** What the prompt shows, e.g. "llm-capacity-planner zai-org/GLM-5.3 --tokens 128k". */
  command: string;
  /** The output, with ANSI codes. */
  output: string;
  /** Title of the window. */
  title: string;
  /** Width in characters (the terminal's columns). */
  columns: number;
  /** A dim "…" line at the end, when the output was cut. */
  cut?: boolean;
}

/** The SVG of a terminal window showing `command` and its output. */
export function terminalSvg(img: TerminalImage): string {
  const rowsOut: Run[][] = [];
  let style: Style = { bold: false, dim: false };
  for (const line of img.output.replace(/\r/g, '').split('\n')) {
    const parsed = parseAnsiLine(line, style);
    style = parsed.end;
    rowsOut.push(...wrapRuns(parsed.runs, img.columns));
  }
  const width = Math.ceil(PAD * 2 + img.columns * CELL);
  const height = BAR + PAD * 2 + (1 + rowsOut.length + (img.cut ? 1 : 0)) * LINE;
  const y = (row: number) => BAR + PAD + (row + 1) * LINE - 5;
  const x = (col: number) => (PAD + col * CELL).toFixed(1);
  const out: string[] = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeXml(`$ ${img.command}`)}">`,
    `<rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="8" fill="${PALETTE.background}" stroke="${PALETTE.border}"/>`,
    `<path d="M0.5 ${BAR}V8.5a8 8 0 0 1 8-8h${width - 17}a8 8 0 0 1 8 8V${BAR}Z" fill="${PALETTE.bar}"/>`,
    `<line x1="0.5" y1="${BAR}" x2="${width - 0.5}" y2="${BAR}" stroke="${PALETTE.border}"/>`,
    ...['#ff5f56', '#ffbd2e', '#27c93f'].map((c, i) => `<circle cx="${18 + i * 20}" cy="${BAR / 2}" r="6" fill="${c}"/>`),
    `<text x="${width / 2}" y="${BAR / 2 + 4}" text-anchor="middle" font-family="${FONT}" font-size="12" fill="${PALETTE.title}">${escapeXml(img.title)}</text>`,
    `<g font-family="${FONT}" font-size="${SIZE}" fill="${PALETTE.text}" xml:space="preserve" style="white-space:pre">`,
    `<text x="${x(0)}" y="${y(0)}"><tspan fill="${PALETTE.dim}">$ </tspan><tspan fill="${PALETTE.command}" font-weight="bold">${escapeXml(img.command)}</tspan></text>`,
  ];
  rowsOut.forEach((runs, i) => {
    // Every word at its own column: the alignment never depends on how the viewer treats spaces.
    const spans = runs.flatMap((r) =>
      [...r.text.matchAll(/\S+(?: \S+)*/g)].map((w) => {
        const col = r.col + [...r.text.slice(0, w.index)].length;
        // textLength pins the word to its cells whatever monospace font the viewer has.
        const cells = ([...w[0]].length * CELL).toFixed(1);
        const fill = r.style.dim ? PALETTE.dim : r.style.color;
        const attrs = [`x="${x(col)}"`, `textLength="${cells}"`, 'lengthAdjust="spacingAndGlyphs"', fill ? `fill="${fill}"` : '', r.style.bold ? 'font-weight="bold"' : ''].filter(Boolean).join(' ');
        return `<tspan ${attrs}>${escapeXml(w[0])}</tspan>`;
      }),
    );
    if (spans.length) out.push(`<text y="${y(i + 1)}">${spans.join('')}</text>`);
  });
  if (img.cut) out.push(`<text x="${x(0)}" y="${y(rowsOut.length + 1)}" fill="${PALETTE.dim}">…</text>`);
  out.push('</g>', '</svg>', '');
  return out.join('\n');
}

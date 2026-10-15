// Whether other programs used the CPU during a measurement (development only). vLLM's scheduler is
// CPU-bound: a test suite running beside it made the same generation measurement up to 40 % slower.
// Counted between two snapshots: Windows processes (but not vLLM's own host: the WSL VM, Docker Desktop,
// this script), the processes of the other WSL distributions, and the other Docker containers.
import { execFileSync } from 'node:child_process';

/** Development tools whose activity alone disturbs a run (another session's tests, builds, type checks). */
const DEV_TOOLS = /^(node|vitest|tsc|esbuild|playwright|electron|python3?|cargo|rustc|java|go|msbuild|cl|link)$/i;
/** What hosts vLLM itself: its load is the measurement's own. */
const OWN_HOST = /^(vmmem|vmmemwsl|com\.docker\.\w+|docker|docker desktop|dockerd|wsl|wslhost|wslservice|wslrelay|powershell|conhost|idle|system)$/i;

// Set from the runs: the ones that measured up to 40 % slow had 1.4–4.6 cores of development tools or the CPU at
// 100 %; runs with up to ~0.7 core of other sessions' node / git repeated within 10 %. The interactive desktop
// alone (the Claude app with its sessions, RGB tools) uses ~0.8 core.
export const LIMITS = { devToolsCores: 0.75, otherCores: 2.0, wslCores: 0.15, containersCores: 0.15 };

interface Snapshot {
  at: number;
  windows: Map<number, { name: string; cpuS: number }>;
  wsl: Map<string, number>;
}

function run(cmd: string, args: string[]): string {
  try {
    return execFileSync(cmd, args, { encoding: 'utf8', windowsHide: true, timeout: 20_000 });
  } catch {
    return '';
  }
}

/** CPU seconds of every Windows process and of each running WSL distribution other than Docker's. */
export function snapshot(): Snapshot {
  const windows = new Map<number, { name: string; cpuS: number }>();
  if (process.platform === 'win32') {
    const out = run('powershell.exe', ['-NoProfile', '-Command', 'Get-Process | ForEach-Object { "$($_.Id)`t$($_.ProcessName)`t$($_.CPU)" }']);
    for (const line of out.split(/\r?\n/)) {
      const [id, name, cpu] = line.split('\t');
      if (id && name && cpu) windows.set(Number(id), { name, cpuS: Number(cpu.replace(',', '.')) });
    }
  }
  const wsl = new Map<string, number>();
  if (process.platform === 'win32') {
    // wsl.exe prints UTF-16: the NUL bytes are dropped.
    const distros = run('wsl.exe', ['-l', '-q', '--running']).replace(/\0/g, '').split(/\r?\n/).map((s) => s.trim()).filter((d) => d && !/^docker-desktop/.test(d));
    for (const d of distros) {
      // Each distribution has its own PID namespace: the sum of its processes' user + system ticks (1/100 s).
      const stat = run('wsl.exe', ['-d', d, '-e', 'sh', '-c', 'cat /proc/[0-9]*/stat 2>/dev/null']);
      let ticks = 0;
      for (const l of stat.split('\n')) {
        const f = l.slice(l.lastIndexOf(')') + 2).split(' ');
        ticks += Number(f[11] ?? 0) + Number(f[12] ?? 0);
      }
      wsl.set(d, ticks / 100);
    }
  }
  return { at: Date.now(), windows, wsl };
}

/** CPU of the other Docker containers right now, in cores. */
function otherContainersCores(own: string): { cores: number; names: string[] } {
  const out = run('docker', ['stats', '--no-stream', '--format', '{{.Name}}\t{{.CPUPerc}}']);
  let cores = 0;
  const names: string[] = [];
  for (const line of out.split(/\r?\n/)) {
    const [name, pct] = line.split('\t');
    if (!name || name === own || !pct) continue;
    const c = Number(pct.replace('%', '')) / 100;
    if (c > 0.01) names.push(`${name} ${Math.round(c * 100)} %`);
    cores += c;
  }
  return { cores, names };
}

/**
 * GPU memory in the system RAM (Windows "shared" GPU memory) of every process, MiB. When an allocation does
 * not fit the card, the driver's sysmem fallback puts it in system RAM over PCIe without an error, and the
 * run slows by up to 40×; growth during a run means it spilled. 0 where Windows' counters do not exist.
 */
export function sharedGpuMiB(): number {
  if (process.platform !== 'win32') return 0;
  const out = run('powershell.exe', ['-NoProfile', '-Command', '(Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Measure-Object -Property SharedUsage -Sum).Sum']);
  return Math.round(Number(out.trim() || 0) / 1024 / 1024);
}

/** Spill threshold: shared GPU memory grown by more than this during a run, MiB. */
export const SPILL_MIB = 1024;

export interface Load {
  seconds: number;
  devToolsCores: number;
  otherCores: number;
  wslCores: number;
  containersCores: number;
  /** The busiest other processes, "name cores". */
  top: string[];
  /** Why the run counts as disturbed, undefined when it is clean. */
  disturbed?: string;
}

/** The other programs' CPU between two snapshots, and whether it disturbed the run. */
export function loadBetween(a: Snapshot, b: Snapshot, ownContainer: string): Load {
  const seconds = Math.max(1, (b.at - a.at) / 1000);
  let dev = 0;
  let other = 0;
  const per: [string, number][] = [];
  for (const [id, p] of b.windows) {
    const before = a.windows.get(id);
    if (!before || id === process.pid || OWN_HOST.test(p.name)) continue;
    const cores = (p.cpuS - before.cpuS) / seconds;
    if (cores <= 0) continue;
    if (DEV_TOOLS.test(p.name)) dev += cores;
    else other += cores;
    per.push([p.name, cores]);
  }
  let wslCores = 0;
  for (const [d, s] of b.wsl) {
    const before = a.wsl.get(d);
    if (before !== undefined) wslCores += Math.max(0, s - before) / seconds;
  }
  const containers = otherContainersCores(ownContainer);
  const reasons: string[] = [];
  const f = (n: number) => n.toFixed(2);
  if (dev > LIMITS.devToolsCores) reasons.push(`development tools (node, vitest, tsc…) ${f(dev)} cores`);
  if (other > LIMITS.otherCores) reasons.push(`other programs ${f(other)} cores`);
  if (wslCores > LIMITS.wslCores) reasons.push(`other WSL distributions ${f(wslCores)} cores`);
  if (containers.cores > LIMITS.containersCores) reasons.push(`other Docker containers ${f(containers.cores)} cores (${containers.names.join(', ')})`);
  const top = per
    .sort((x, y) => y[1] - x[1])
    .slice(0, 5)
    .map(([n, c]) => `${n} ${f(c)}`);
  return {
    seconds: Math.round(seconds),
    devToolsCores: Math.round(dev * 100) / 100,
    otherCores: Math.round(other * 100) / 100,
    wslCores: Math.round(wslCores * 100) / 100,
    containersCores: Math.round(containers.cores * 100) / 100,
    top,
    ...(reasons.length ? { disturbed: `CPU busy during the measurement: ${reasons.join('; ')}` } : {}),
  };
}

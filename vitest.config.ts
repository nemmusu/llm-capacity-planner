import { readFileSync } from 'node:fs';
import { defineConfig } from 'vitest/config';

const pkg = JSON.parse(readFileSync(new URL('./package.json', import.meta.url), 'utf8')) as { name: string; version: string };

/**
 * Unit tests, all in Node. Coverage (V8) is gated at 100 % of src/: `npm run test:coverage`.
 * The fixtures are read relative to the repository root, so tests run from there.
 */
export default defineConfig({
  // The same compile-time constants as the bundle (scripts/cli-name.mjs).
  define: { __CLI_NAME__: JSON.stringify(pkg.name), __CLI_VERSION__: JSON.stringify(pkg.version), __SOURCE_HASH__: JSON.stringify('test') },
  test: {
    environment: 'node',
    include: ['test/**/*.test.ts'],
    testTimeout: 30_000,
    // Dates in reports are written in local time: the same zone on every machine (a Linux container runs in UTC).
    env: { TZ: 'UTC' },
    coverage: {
      provider: 'v8',
      include: ['src/**/*.ts'],
      // The entry only wires process.argv, stdout and the network into cli/run.ts, which the tests cover.
      exclude: ['src/**/*.d.ts', 'src/main.ts'],
      reporter: ['text-summary', 'html', 'json-summary'],
      reportsDirectory: 'coverage',
      thresholds: { statements: 100, branches: 100, functions: 100, lines: 100 },
    },
  },
});

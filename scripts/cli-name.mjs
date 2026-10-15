// The command's name and version come from package.json only: `name` is the command (and the file names in dist/),
// `bin` maps it and an optional short alias to the same file, `version` is printed by --version. Renaming the project
// means editing package.json and nothing else.
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { sourceHash } from './source-hash.mjs';

export const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pkg = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'));
if (!pkg.bin?.[pkg.name]) throw new Error('package.json "bin" must contain an entry named like the package');

/** The command as users type it, e.g. "llm-capacity-planner". */
export const cliName = pkg.name;
export const version = pkg.version;
/** Compile-time constants of the bundle (declared in src/env.d.ts). */
export const define = { __CLI_NAME__: JSON.stringify(cliName), __CLI_VERSION__: JSON.stringify(version), __SOURCE_HASH__: JSON.stringify(sourceHash(root)) };

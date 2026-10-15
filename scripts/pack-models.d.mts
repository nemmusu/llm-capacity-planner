/** Packs the readable, public catalog models of `modelsDir` (default data/models) for src/cli/builtin.ts. */
export function packModels(modelsDir?: string): { pack: Buffer; count: number };
/** Fingerprint of the files a pack is made of (paths and bytes under `modelsDir`, default data/models). */
export function modelsHash(modelsDir?: string): string;
/** Public repos left out of the pack because their licence may not allow passing their files on: repo → reason. */
export const NOT_REDISTRIBUTED: Record<string, string>;

# Development tools

Scripts that maintain the data and the speed calibration. **They are not part of the command** and are not bundled
into `dist/`. The catalog update needs only Node.js and network access to Hugging Face; the speed measurements need an
NVIDIA GPU with Docker and the pinned engine image.

| Command | What it does | Needs |
| --- | --- | --- |
| `npm run dev:update-models` | refreshes `data/models.json` and `data/models/` from Hugging Face (headers only) | network; `HF_TOKEN` for gated repos |
| `npm run dev:measure` | starts vLLM on the local GPU and records the memory lines of its log (weights, KV cache, concurrency) | NVIDIA GPU, Docker |
| `npm run dev:speed` | measures TTFT and generation speed with dummy weights and cut-down models; checks the CPU is otherwise idle | NVIDIA GPU, Docker |
| `npm run dev:calibrate` | refits the efficiencies on the measured runs and `data/references.json`, rewrites `test/fixtures/calibration-points.json` | — |
| `npm run dev:report` | regenerates `data/report.json` (precomputed plans for the catalog models) | — |

Measurement rules (see [docs/METHODOLOGY.md §10](../../docs/METHODOLOGY.md#10-calibration)):

- close every other program that uses the CPU or the GPU: vLLM's scheduler is CPU-bound, and runs disturbed by other
  load are detected and excluded;
- keep `--gpu-memory-utilization` at 0.85 on a desktop: if the card fills up, a Windows driver spills into system memory
  and the run is about ten times slower;
- record the engine version: a calibration holds only for the version it was measured with.

## Files

| File | Role |
| --- | --- |
| `update-models.mjs` | catalog updater (`--only`, `--missing`, `--quants-only`, `--variants`, `--offline`; see data/README.md) |
| `update-compat.mjs` | compatibility matrix: `--models` rewrites the model section, `--anchors [--vllm TAG] [--sglang TAG]` re-checks every cited line of engine code |
| `measure.mjs` → `measure-entry.ts` | memory runs: `npm run dev:measure -- [--only=<name>]` |
| `speed.mjs` → `speed-entry.ts` | speed runs: `npm run dev:speed -- [--only=<name>]` |
| `cut.ts` | cuts a model to its first layers through `--hf-overrides` (real layer shapes, dummy weights) |
| `cpu-guard.ts` | CPU load of other programs before and during a run; on Windows it also reads WSL distributions and Docker containers, elsewhere it only checks the containers |
| `calibrate.mjs` → `calibrate-entry.ts` | recomputes every point of data/references.json and of the RTX 5090 runs, writes test/fixtures/calibration-points.json |
| `report.mjs` → `report-entry.ts` | ready-made plans for the catalog models: data/report.json |
| `docs.mjs` → `docs-entry.ts` | the appendices of docs/METHODOLOGY.md, from the data |
| `screenshots.mjs` → `screenshots-entry.ts` | the README images (docs/images/*.svg): the real command on built-in models, offline, in colour (`screenshots-lib.ts`), drawn as a terminal by `ansi-svg.ts`; `test/docs/screenshots.test.ts` fails when they are out of date |

The `.mjs` files bundle their `-entry.ts` with esbuild into `out/` and run it: plain Node.js 22+, no TypeScript runner.
The speed and memory runs use the Docker image `vllm/vllm-openai:v0.31.0` with `--gpus all`; set `HF_TOKEN` for gated
models.

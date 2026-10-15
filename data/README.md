# Data

Reference data bundled with the planner. Every value carries its source; a value that is not public is `null` and listed
with the reason, never estimated. Units are in the field names. Schemas: `src/contract/data.ts` (GPUs, benchmarks) and
`src/contract/models.ts` (model catalog); `npm test` parses every file with them.

| File | Content | Used by | Updated how |
| --- | --- | --- | --- |
| `hardware.json` | 26 NVIDIA GPUs: CUDA-visible memory, bandwidth, dense/sparse TFLOPS per format, NVLink/PCIe/P2P, power | the command (bundled) | by hand, from NVIDIA datasheets |
| `references.json` | public vLLM / SGLang measurements: engine version, model, hardware, layout, dataset, throughput, TTFT, TPOT, memory lines from logs | calibration and tests | by hand, from the cited issue, recipe, benchmark or paper |
| `models.json` | catalog of recent open-weight models: exact weights per checkpoint and quantized variant, context, RoPE, long-context settings, engine flags | the `report` subcommand, tests | `npm run dev:update-models` + curated fields by hand |
| `models/` | per repo: config.json, shard index and safetensors headers (no weights), as read from Hugging Face | the command (packed, see below), tests | `npm run dev:update-models` |
| `report.json` | precomputed plans for the catalog models (vLLM, fp8 KV, default workload) | the `report` subcommand (bundled) | `npm run dev:report` |

## GPU catalog (`hardware.json`)

Sources are NVIDIA datasheets, product pages and architecture whitepapers; `sources[].fields` says which fields each one
supports. Notes that matter:

- **Dense vs sparse TFLOPS.** `tflopsDense` excludes 2:4 sparsity. When NVIDIA publishes only the sparse figure, the
  dense one is half of it, as the datasheet footnote states, and the format is listed in `denseHalvedFromSparse`.
- **Derived values.** Where NVIDIA publishes no exact figure, a value may be derived from published ones, and the
  entry says how: `tflopsDerived` (the RTX PRO 6000 Server Edition: the Workstation Edition's figures scaled by the boost
  clock) and the whole memory of the GH200 144GB (from the CUDA-visible measurement). The command and the appendix
  mark them "derived".
- **Memory.** `vramGBNominal` is the marketing capacity; the planner uses `cudaTotalBytes` / `cudaTotalGiB` (what CUDA
  reports, the base of vLLM's memory share) when measured, otherwise `nvidiaSmiTotalMiB` minus the driver reserve.
- **Rack-scale systems** (GB200 / GB300 NVL72) are given per GPU (`perGpuOf`) and are sold as whole racks.
- `gaps` lists every field NVIDIA does not publish, with the reason.

To add a card: append an entry with its sources, run `npm test`.

## Public benchmarks (`references.json`)

One entry per published run: `engine`, `engineVersion` as the source states it, `source { url, title, kind, date }`
(`issue`, `discussion`, `pull-request`, `docs`, `recipe`, `blog`, `benchmark`, `paper`), `model`, `hardware`
(`gpuId` from `hardware.json`, count, TP/PP/DP), and `perf` rows at a known load (concurrency or request rate, dataset,
input/output length, output and total tok/s, TTFT and TPOT mean/median/p99). Memory lines read from logs
(`Model loading took`, KV cache size, maximum concurrency) are kept when the source shows them.

Only runs whose load and lengths are stated are usable for calibration. Runs that the model cannot describe stay in the
file and are listed as outliers in `src/planner/calibration.ts` with the reason.

## Model catalog (`models.json`)

Two kinds of fields:

- **curated**: written by hand from model cards, LICENSE files, official recipes and engine source code, each with its
  URL (and, for model cards, a verbatim quote of at most 25 words in `sources`);
- **`hf` and `derived`**: rewritten by `npm run dev:update-models` from the files saved in `data/models/`.
  Never edit them by hand: the test suite recomputes them from the fixtures and fails on any difference.

```bash
npm run dev:update-models                       # every model: configs, shard index, safetensors headers
npm run dev:update-models -- --only <id>        # some ids (comma separated)
npm run dev:update-models -- --missing          # only models whose fixtures are missing or incomplete
npm run dev:update-models -- --quants-only      # only the list of quantized and draft checkpoints
npm run dev:update-models -- --variants         # headers of the official quantized checkpoints
npm run dev:update-models -- --offline          # recompute "hf" / "derived" from the saved fixtures
```

Only the JSON header of each safetensors shard is read, with HTTP `Range` requests; weights are never downloaded. The
Hugging Face token is found as `huggingface_hub` finds it (`HF_TOKEN`, then `$HF_HOME/token`, then
`~/.cache/huggingface/token`) and is never printed or saved. Gated repositories stay `"readable": false` until their
license is accepted on the account. `HTTPS_PROXY` / `HTTP_PROXY` are honoured. Anonymous clients are limited by Hugging
Face to 3,000 file requests every 5 minutes; the script waits for the window to reset.

Adding a model: append `{ "id", "repo", "lab" }` to `models.json`, run `--only <id>` then `--variants --only <id>`,
then write the curated fields with their sources.

### Model files (`data/models/<org>__<repo>/`)

| File | Content |
| --- | --- |
| `config.json`, `generation_config.json`, `params.json` | as published at the pinned commit (`params.json`: Mistral format) |
| `*.safetensors.index.json.gz` | the shard index, gzipped byte for byte |
| `safetensors-headers.json.gz` | every shard's header, compact (`fileBytes`, `headerBytes`, each tensor's `dtype` and `shape`) |
| `tensor-summary.json` | per-kind summary: tensors that differ only by layer or expert number form one entry with count, bytes and layers |
| `source.json` | repo, commit, license tag, gated status, unreadable files and why, `headerChecks` |
| `quant-repos.json` | quantized and draft checkpoints of the lab, NVIDIA (official) and Red Hat AI (third party) |
| `variants/<org>__<repo>/` | the same for each official quantized checkpoint |

`data_offsets` are dropped only after the update script has checked them on every shard: each tensor spans exactly
`prod(shape) × dtype size`, tensors follow each other without gaps, and the file is `8 + headerBytes + tensors` bytes.

## Built-in models

The command reads the catalog models from `data/models/` instead of Hugging Face: they work offline and print
"Model description from built-in data (<date>)". `--refresh` reads a model from Hugging Face again.

The build packs them into one file, `<name>-models.pack` (`scripts/pack-models.mjs`): a small JSON index, then one
gzip blob per repo with its config files and safetensors headers. Each release attaches it next to the `.mjs` bundle
(`npm run dist` writes both to `dist/`), and it is also inside each single executable; a pack next to an executable takes precedence
over the one inside it, so the data can be refreshed without rebuilding the program.

| | Size |
| --- | --- |
| `data/models/` in the repository | 35 MB (91 readable repos: 37 models and 54 quantized variants, all in the pack) |
| `<name>-models.pack` | 31.7 MB; it makes the single executable about 30 MB larger |

Only public metadata is bundled: repos that are gated or could not be read (the Llama 4 family, whose license must be
accepted on Hugging Face) keep only their `source.json` and are always read from Hugging Face with the user's token.
Repos whose licence may not allow passing their files on (`NOT_REDISTRIBUTED` in `scripts/pack-models.mjs`:
Mistral Medium 3.5 and its NVFP4 quantization, whose Modified MIT licence gives no rights at all to companies above
US$20M monthly revenue) keep only their `source.json`, with the reason under `withheld`: none of their files is saved
here or packed, and the command reads them from Hugging Face. A test keeps them, and the gated Llama 4 family, out of
the repository and the pack.

## Sources and licences

Every saved repository, with its revision and licence, every benchmark page and every GPU source is listed in
[NOTICE.md](../NOTICE.md), written from the data by `npm run dev:docs` and checked by the tests. The model files kept
here are metadata (configuration, shard index, tensor names, shapes and types), never weights.

The full licence texts are in [licenses/](../licenses/): each repository's own licence file (LICENSE, NOTICE) at the
revision saved, and the official text of the licence it declares (Apache-2.0, MIT, OpenMDW-1.1, the NVIDIA Open Model
and Nemotron Open Model licences); a quantization also carries its base model's. `licenses/index.json` maps each repo
to its files; the tests fail when a packed repository has no text, a revision changed, or a file is unused. NVIDIA's
licences also require an attribution statement, which NOTICE.md carries.

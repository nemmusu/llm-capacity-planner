# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-15

First release as a standalone project.

### Added

- Command line planner: for a Hugging Face model, a context length and vLLM or SGLang, the NVIDIA configurations that
  fit, with guaranteed and likely concurrent users, generation speed and time to first token.
- Exact weights per GPU from safetensors headers and the engines' loading rules; KV cache from vLLM's block planner.
- Speed model calibrated on vLLM 0.31.0 runs on an RTX 5090 and on public vLLM / SGLang benchmarks, with the measured
  error declared per engine and card.
- Catalog of 26 NVIDIA GPUs with sourced specifications (24 planned as cards), and of 40 recent open-weight models (37
  saved here, with 54 quantized variants: 91 built-in repos) with precomputed plans
  (`report` subcommand).
- Export to CSV, JSON and Markdown.
- Memory in GiB everywhere (the card's nvidia-smi total on the same scale); the JSON export names its units.
- Servers sized from NVIDIA's published GPUs per system and NVLink reach (e.g. 2 GH200 per NVL2 server); a group never
  spans two servers, more servers are independent copies behind a load balancer.
- Nothing is run on the machine unless asked: `--this-pc` reads the local card with nvidia-smi.
- Single executables for Windows, Linux and macOS with the built-in models inside, and a portable bundle for any system
  with Node.js 22+, attached to each GitHub Release with their SHA-256 checksums (`.github/workflows/release.yml`; Linux
  and macOS in a `.tar.gz` that keeps them executable); a test checks a build against the sources.
- A plain-language summary at the top of the output (recommended hardware, servers needed, speed in words per second,
  confidence), readable layouts, and `--md` as a one-page Markdown summary.
- Methodology and data documentation, with appendices generated from the data (`npm run dev:docs`).
- Built-in catalog models (91 public repos), read without network; `--refresh` reads them from Hugging Face again.
- MIT licence; the third-party data, its sources and licences are listed in `NOTICE.md`, with the full licence texts
  in `licenses/`.
- Test suite: unit, end-to-end on the real command, pairwise option combinations, property-based invariants, golden
  outputs and accuracy regression against every calibration point; 100 % coverage of `src/`.

### Known limitations

- Mistral-format checkpoints (`params.json` without `config.json`, e.g. `mistralai/Mistral-Large-3-675B-Instruct-2512`)
  are recognised and refused with a clear message: the calculator reads `config.json`.
- macOS builds are produced by `npm run dist` on a Mac and by the CI workflow; they were not tested before this release.

[0.1.0]: https://github.com/nemmusu/llm-capacity-planner/releases/tag/v0.1.0

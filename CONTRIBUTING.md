# Contributing

Bug reports, data updates and pull requests are welcome. Please
[open an issue](https://github.com/nemmusu/llm-capacity-planner/issues/new/choose) first for larger changes.

## Set up and check

Node.js 22 or newer.

```bash
npm ci
npm run plan -- zai-org/GLM-5.3 --tokens 128k --users 200
npm run validate
```

`npm run validate` runs lint, the type check and every test suite with a 100 % coverage gate; a pull request must keep
it green. `npm run ci:docker` runs the same check on the committed code in a clean Linux container.

## Updating data and calibration

Every number in the data has a source; please add one with every change.

| What | How |
| --- | --- |
| A GPU | edit `data/hardware.json` by hand, with the NVIDIA document for each value ([data/README.md](data/README.md)) |
| A public benchmark | add an entry to `data/references.json` with its link and the figures it reports |
| The model catalog | `npm run dev:update-models` (then curate the fields marked as such in `data/models.json`). For a new repository, add its licence texts to `licenses/` and `licenses/index.json`: its own LICENSE / NOTICE at the saved revision, and the official text of the licence it declares ([data/README.md](data/README.md#sources-and-licences)) |
| The compatibility matrix | `npm run dev:update-compat -- --anchors --vllm <tag> --sglang <tag>` re-checks every cited line of engine code; with local clones of the tags, `CAPACITY_PLANNER_SRC_VLLM=<dir> CAPACITY_PLANNER_SRC_SGLANG=<dir> npx vitest run test/contract/compat.test.ts` does it offline (the CI runs it on every push) |
| The speed calibration | on a GPU you own: `npm run dev:speed`, then `npm run dev:calibrate` ([scripts/dev/README.md](scripts/dev/README.md)) |

Then regenerate what is derived from the data, and commit it with the change:

```bash
npm run dev:report        # data/report.json, the precomputed plans
npm run dev:docs          # the appendices of docs/METHODOLOGY.md and NOTICE.md
npm run dev:screenshots   # the README images and generated blocks
```

The tests fail when one of these files no longer matches the data or the code. A new engine release follows
[docs/METHODOLOGY.md §14](docs/METHODOLOGY.md#14-updating-data-and-calibration).

## Releasing

1. Update `CHANGELOG.md` (a new `## [x.y.z] - date` section and its link at the bottom), `version` in `package.json`,
   and `version` and `date-released` in `CITATION.cff`.
2. Commit, then tag and push:

   ```bash
   git tag v0.1.0
   git push --tags
   ```

The release workflow (`.github/workflows/release.yml`) validates the sources on Windows, Linux and macOS, builds the
executables, checks them against the sources and attaches them, with `SHA256SUMS.txt` and the notes of that version, to
a GitHub Release.

## Licence

By contributing you agree that your contribution is released under the [MIT licence](LICENSE). Third-party data keeps
its own terms ([NOTICE.md](NOTICE.md)).

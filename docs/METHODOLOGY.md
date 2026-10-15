# Methodology

How LLM Capacity Planner derives every number it prints: memory per GPU, the fit verdict, generation speed, time to
first token and concurrent users, and how far each can be trusted.

> Every constant quoted here is in the code under `src/`, and the test suite checks the code against the measurements.
> The tables of measurements, errors and sources are in the appendices, which `npm run dev:docs` writes from the
> repository data: no accuracy figure is typed by hand.

## Contents

1. [Scope and principles](#1-scope-and-principles)
2. [Inputs](#2-inputs)
3. [Weights per GPU](#3-weights-per-gpu)
4. [KV cache](#4-kv-cache)
5. [Fixed costs and the engine's memory share](#5-fixed-costs-and-the-engines-memory-share)
6. [Memory verdict](#6-memory-verdict)
7. [Speed model](#7-speed-model)
8. [Users: guaranteed and likely](#8-users-guaranteed-and-likely)
9. [Engine and GPU compatibility](#9-engine-and-gpu-compatibility)
10. [Calibration](#10-calibration)
11. [Validation](#11-validation)
12. [Comparison with published methods](#12-comparison-with-published-methods)
13. [Known limits](#13-known-limits)
14. [Updating data and calibration](#14-updating-data-and-calibration)
15. [Glossary](#15-glossary)
16. [Engine source references](#16-engine-source-references)
17. [Timing and reproducibility](#17-timing-and-reproducibility)

Appendices (generated from the repository data by `npm run dev:docs`, never written by hand):

- [A. Calibration runs on the RTX 5090](appendix-a-calibration-runs.md)
- [B. Public benchmarks](appendix-b-public-benchmarks.md)
- [C. Validation errors per point](appendix-c-validation.md)
- [D. GPU specification sources](appendix-d-gpu-sources.md)
- [E. Fitted parameters of the speed model](appendix-e-fitted-parameters.md)

---

## 1. Scope and principles

The planner answers one question for a Hugging Face model, a context length and an inference engine (vLLM or SGLang):
*on which NVIDIA configurations does it start, how many users does each serve at a target speed, and how fast?*

Three rules apply throughout:

- **Count what can be counted.** Weight bytes, KV cache bytes and the engines' memory rules are deterministic. They are
  computed exactly from the model's files and the engines' source code, not estimated.
- **Range what cannot.** Fixed engine overheads and all speeds depend on kernels, drivers and load. They are given as
  `min–max` ranges whose width is the error measured against real runs.
- **Never claim a fit the uncertainty could break.** A configuration is reported as fitting only if it fits with the
  largest overheads on the smallest memory the card can expose (section 6).

Engine behaviour is pinned to **vLLM 0.31.0** and **SGLang 0.5.21**. Only NVIDIA GPUs are modelled.

**Units.** Every memory figure the planner prints — need, weights, KV cache, margin and the card's own memory — is
in GiB (2^30 bytes, the unit nvidia-smi and CUDA count in), so both sides of "71.4 of 79.6 GiB" are the same unit. A
card's memory is its nvidia-smi total, which is below the size it is sold with (an "80GB" H100 shows 79.6 GiB). The
JSON export states the units in its `units` field. Token counts use k = 1024; user counts (`--users 2k`) use k = 1000.

## 2. Inputs

| Input | Source | Read how |
| --- | --- | --- |
| Model architecture | `config.json` of the repository (a Mistral-format checkpoint with only `params.json` is recognised and refused with a message) | one small JSON file |
| Weight tensors | the JSON header of every `.safetensors` shard: name, dtype, shape of each tensor | HTTP `Range` requests; the weights themselves are never downloaded |
| Shard list | the repository file listing | Hugging Face API |
| Built-in models | the same files for the models of the catalog, read once and shipped with the command (`data/models/`) | no network; `--refresh` reads Hugging Face again |
| GPU specifications | `data/hardware.json`: 26 NVIDIA GPUs, each value with its NVIDIA source; the 24 sold as cards are planned (GB200 / GB300 NVL72 are whole racks) | bundled |
| Workload | command-line options (context, users, target speed, cached share of the prompt) | — |

The shard index (`model.safetensors.index.json`) is **not** trusted for sizes: on real repositories its `total_size` was
found 15 GB off and its `weight_map` pointed to the wrong file for hundreds of tensors. The engines read each shard's
own header, and so does the planner.

## 3. Weights per GPU

The model is never downloaded: only `config.json` and the safetensors headers (a few KB per shard, via HTTP Range
requests) are read. For the models of the built-in catalog these files ship with the planner, and nothing is fetched
unless `--refresh` is given.

A tensor's size is

```
bytes(tensor) = prod(shape) × dtype_size
dtype_size: F64/I64/U64 = 8, F32/I32/U32 = 4, F16/BF16/I16/U16 = 2, F8_E4M3/F8_E5M2/F8_E8M0/I8/U8/BOOL = 1
```

4-bit formats (NVFP4, MXFP4, INT4) are stored packed in `U8`/`I8` tensors with their scale tensors alongside, so the
same formula is exact for them.

The bytes one GPU holds follow vLLM 0.31's loading rules (SGLang shards the same way):

| Tensor | Per GPU with tensor parallel degree `TP` |
| --- | --- |
| Column- and row-parallel linear layers, MoE experts | `bytes / TP` |
| K and V projections | `max(1, kv_heads / TP)` heads; replicated when there are fewer KV heads than GPUs |
| Embedding and LM head | vocabulary padded to a multiple of 64, then split over `TP`; a tied head is counted once |
| Norms, row-parallel biases, MoE routers, MLA `q_a` / `kv_a` projections | replicated on every GPU |
| MTP / next-token layers | skipped unless speculative decoding is enabled |
| Vision / audio tower | skipped with `--language-model-only` |
| MLA `kv_b_proj` | an extra dequantized BF16 copy for absorbed attention, reported apart as a range |

With pipeline parallelism `PP`, decoder layers are split evenly across stages.

**Verified combinations.** The computed weights were compared with vLLM's own `Model loading took X GiB` line on
measured runs. A *format × architecture family* (dense, MoE, MLA, hybrid, multimodal) is **verified** when every
measurement matches within 0.06 GiB. Where the engine holds more than the tensors (kernel repacking, scale layouts), the
measured excess is kept as a declared margin; a combination never measured gets a 5 % margin.

The matrix of verified and measured combinations, with its evidence, is in
[Appendix C.3](appendix-c-validation.md#c3-weights-per-gpu-against-the-engine-log) (source: `src/planner/weights-verification.ts`).

## 4. KV cache

### 4.1 Bytes per sequence

For a sequence of `L` tokens on one GPU:

```
KV(L) = Σ over attention groups g:
          layers_g × heads_g × (d_k + d_v) × ceil(span_g(L) / block) × block × elem
        + recurrent_state / TP

heads_g  = max(1, floor(kv_heads / TP))      standard attention (GQA/MHA)
heads_g  = kv_heads (the latent)              MLA: the compressed latent is replicated on every GPU
span_g   = L                                  full attention
span_g   = min(L, window + slack)             sliding-window layers
elem     = 1 byte (fp8 KV cache, the default) | 2 bytes (auto: BF16/FP16)
```

Recurrent layers (Mamba, Gated DeltaNet, linear attention) hold a fixed-size state per sequence, independent of `L`.

### 4.2 vLLM's block planner

vLLM does not allocate KV per token but per block, and groups layers into a shared pool. The planner replicates vLLM
0.31's planner (`v1/core/kv_cache_utils.py`, `kv_cache_interface.py`):

- each layer has a spec: full attention, sliding window, or recurrent state;
- pages are made uniform: the attention block (16 tokens) grows until one page holds a recurrent layer's state (rounded
  up to a multiple of 16 tokens); recurrent pages are padded to the attention page;
- layers of the same spec form buckets, split into groups of `group_size` layers; one block of the pool costs
  `page × group_size`;
- per request and per group: full attention needs `ceil(L / block)` blocks; a sliding window
  `ceil(min(window − 1 + step_tokens, L) / block) + 1`; a recurrent state 2 blocks with prefix caching (the default),
  else 1;
- vLLM refuses to start if one request at `max_model_len` needs more than the KV memory, and its reported concurrency is
  `pool blocks ÷ blocks per request`.

On an RTX 5090 with a hybrid model (16 attention + 48 Gated DeltaNet layers) the five "X GiB KV cache is needed" errors
and the "Maximum concurrency … 10.00x" line are reproduced exactly.

SGLang allocates token slots (block size 1); its per-sequence size is the formula of 4.1 with `block = 1`.

## 5. Fixed costs and the engine's memory share

Both engines reserve a share of the GPU at start-up. What that share must hold differs:

| | vLLM (`--gpu-memory-utilization`, default 0.9) | SGLang (`--mem-fraction-static`) |
| --- | --- | --- |
| Inside the share | weights, activation peak of the profiling run, CUDA graphs, KV cache | weights and KV pool |
| Outside the share | CUDA context and NCCL buffers (set up before the share is measured) | CUDA context, NCCL, activations, CUDA graphs |
| Default share | 0.9 of CUDA-visible memory | `(card − reserve) / card` (below) |

**Activation peak** (bytes per token of one forward step, BF16, one GPU, hidden size `h`, FFN size `ff`):

```
dense = 2 × (6h + 3·ff / TP)
MoE   = 2 × (6h + top_k × (3·ff_expert / TP + h))
peak  = max(dense, MoE) × max_num_batched_tokens   (+ sampler logits for max_num_seqs, + vision encoder)
```

**Scheduler defaults** the engines pick from the card's memory (they set the step size and the CUDA graph sizes):

| Engine | Memory per GPU | Tokens per step | Max sequences |
| --- | --- | --- | --- |
| vLLM | ≥ 160 GiB | 16,384 | 1,024 |
| vLLM | ≥ 70 GiB | 8,192 | 1,024 |
| vLLM | < 70 GiB | 2,048 | 256 |
| SGLang | < 20 / < 35 / < 60 / < 160 / ≥ 160 GiB | 2,048 / 4,096 / 4,096 / 8,192 / 16,384 | 8 / 48–160 / 32–160 / 256–512 / 512 |

**SGLang's default fraction**: reserve = 512 MiB + 1.5 MiB per chunked-prefill token + TP/8 GiB + 2 MiB per decode
graph batch + 8 MiB per prefill graph (1.5 GiB with MLA), at least 10 GiB on cards over 60 GiB; reduced further for
vision models.

**CUDA-visible memory.** The share is a fraction of what CUDA reports (`totalGlobalMem`), not of the marketing
capacity, and it varies with OS and driver (an RTX 5090 shows 31.82 GiB on Windows and 31.35 GiB on Linux). The planner
uses the measured range when the catalog has it, otherwise `nvidia-smi` total minus 0.56–0.7 % (driver reserve).

These fixed costs are modelled from the model's shape and the engine's limits, and are therefore a **range**.

## 6. Memory verdict

Each configuration is evaluated twice: with the largest fixed costs on the smallest CUDA-visible memory (worst case) and
with the smallest costs on the largest memory (best case).

| Verdict | Condition |
| --- | --- |
| **fits** | one request at the full context fits in the worst case |
| **borderline** | it fits in the best case but not in the worst |
| **does not fit** | it does not fit even in the best case |
| **engine does not start** | the engine refuses the layout or the model on that GPU generation, whatever the memory (section 9); such a row gets no speed and no users and is never recommended |

The free KV memory left after one full-context request is reported as a margin, and determines how many requests fit
at once (section 8).

## 7. Speed model

The model is a roofline (time = max of memory time and compute time) with efficiencies fitted per engine, per card
family and per weight format, plus fixed overheads per step, per layer and per request. Every result is a
`min–max` range: best and worst efficiency, then widened by the measured fit error of that calibration (`spread`).

### 7.1 Generation (decode step)

One decode step produces one token for each of the `b` sequences running together. Per pipeline stage:

```
bytes_w  = (dense_bytes + routed_expert_bytes × touched) / (TP × PP)
bytes_kv = b × KV_read(L_typical) / PP + b × 2 × recurrent_state / (TP × PP)

t_mem    = bytes_w / (BW × η_w × s_bw(format)) + bytes_kv / (BW × η_kv)
t_comp   = 2 × N_active × b / (TP × PP) / (TFLOPS(format) × s_mfu(format) × η_mfu)
t_launch = layers/PP × (c_layer + c_moe + touched × E × c_expert + b × c_seq)
t_comm   = 2 × layers / PP × allreduce(b × h × 2 bytes)

t_step   = PP × (max(t_mem, t_comp) + t_launch + t_comm) + c_step + (PP − 1) × link_latency
tok/s per user = 1000 / t_step(ms)
```

- `touched` (MoE): the expected share of experts that at least one of the batch's tokens routes to,
  `distinct(E, top_k, b) / E`. A single user reads only `top_k` experts per layer; a large batch reads nearly all.
- `BW` is the card's peak memory bandwidth, `TFLOPS(format)` its dense tensor throughput for the weight format
  (FP4 on Blackwell, FP8 on Ada/Hopper/Blackwell, otherwise BF16), from `data/hardware.json`.
- `η_w`, `η_kv`, `η_mfu` are fitted efficiencies; `s_bw`, `s_mfu` scale them per format (low-precision GEMV and GEMM
  kernels reach a smaller share of their own peak).

### 7.2 Prompt reading (prefill) and time to first token

```
linear_flops = 2 × ((N_active − N_head) × T + N_head × ceil(T / step_tokens)) / TP
attn_flops   = Σ layers_a × head_dims_a × span(T) / TP
span(T)      = T²                    full causal attention
span(T)      = 2·T·w − w²            sliding window or sparse attention reading at most w keys per query

t_prefill = linear_flops / (TFLOPS(format) × s_mfu × η_prefill)
          + attn_flops / (TFLOPS_bf16 × η_attn(T))
          + 2 × layers × allreduce(T × h × 2 bytes)
          + T × recurrent_layers × c_recurrent / TP
          + ceil(T / step_tokens) × layers × c_layer
```

The LM head runs once per prompt chunk (for the sampled token), not once per prompt token.

### 7.3 Tensor and pipeline parallelism

- **All-reduce** (two per layer with tensor parallelism), ring algorithm:
  `allreduce(bytes) = latency(link) + 2(TP − 1)/TP × bytes / (link_bandwidth × η_link)`.
  Link kinds: NVLink, PCIe with peer-to-peer, PCIe through host memory (GeForce cards without P2P). Fitted latencies are
  8, 20 and 45 µs respectively; NVLink bandwidth is the catalog's bidirectional figure halved.
- **NVLink reach.** NVLink connects a limited number of cards (`nvlink.domainGpus` in `data/hardware.json`, with its
  source): 2 with a bridge between two PCIe cards (RTX 3090, RTX A6000, A100 PCIe, H100 PCIe / NVL), 4 with the H200 NVL
  4-way bridge, 2 on a GH200 NVL2 system, 8 on an HGX board. A tensor-parallel group larger than that talks over PCIe
  at the PCIe rate and gets a warning; with TP inside the domain and pipeline stages across it (H200 NVL: TP 4 × 2
  stages; H100 NVL, A100 PCIe: TP 2 × 2 or × 4 stages) the all-reduces stay on NVLink and only the hand-over between
  stages crosses PCIe.
- **Every split is tried.** Each group of GPUs is planned in tensor parallel, in pipeline stages (up to 4) and as both,
  on every card, and the card keeps its best. The difference matters most for MLA models (DeepSeek, GLM, Kimi): tensor
  parallelism repeats their KV cache on every GPU, pipeline stages split it, so at long contexts a pipeline fits where
  TP does not (GLM-5.3 at 1M on 8 H200: 141.2 GiB per GPU with TP 8, 122.3 GiB with TP 4 × 2 stages). A test checks
  that a card with at least the memory, bandwidth, compute and links of another of the same generation is never
  planned worse (`test/planner/monotonic.test.ts`).
- **One server per group.** No tensor- or pipeline-parallel group is larger than one server: at most 8 GPUs, or fewer
  where NVIDIA publishes fewer per system (`server.maxGpus`: 4 RTX 6000 Ada or RTX PRO 6000 Max-Q per workstation,
  2 GH200 per NVL2 server; 4 assumed for GeForce and workstation cards without a figure, among them the RTX PRO 6000
  Workstation Edition, for which NVIDIA states no count while its Max-Q edition is sold for up to four per system; the
  assumption is recorded as a gap in `data/hardware.json`). More GPUs are copies on
  several servers.
- **Pipeline parallelism** splits the batch into `PP` micro-batches; a token crosses all stages in sequence, and the
  stages overlap across requests. A single prompt gets no speed-up from `PP`: reading it alone and the first-token
  wait cross every stage. Under load each stage carries 1/`PP` of the other users' prompt reading, since the stages
  work on different requests at once, and a prompt queues for a stage's share of its reading. The stage that also
  holds the LM head (in BF16) sets the pace of writing. Micro-batches balance only by chance: vLLM never moves a
  running request to another one (`v1/core/sched/scheduler.py`), so the slow end of the writing speed uses an upper
  estimate of the fullest micro-batch, `b/PP + √(2 · b/PP · ln PP)` of `b` requests (the usual bound on the largest of
  `PP` bins filled at random, above its expected size), the fast end an even split. Each
  stage counts its own layers for the CUDA graphs, the MLA absorbed copies and the KV cache (a mixed stack shared in
  proportion), and keeps a communicator for the next stage (the same 100–500 MiB as a TP one, not measured for PP).
  No public single-server benchmark compares PP with TP on the catalog cards; the one found (16 H100 on two nodes,
  SGLang 0.5.9, GLM-5 FP8: TP 4 × PP 4 238 tokens/s, TP 2 × PP 8 207, TP 8 × PP 2 165, arXiv 2607.02518 Table 1) is
  multi-node and is not used to calibrate. So a pipeline layout's speed is labelled "pipeline speed not measured",
  estimated, with its range widened by half (`PIPELINE_WIDENING`); and against tensor-parallel layouts it counts as its
  GPUs grown by that declared error (`pipelineWeight`), so it is chosen only when it saves more than the error.
- **Replicas** (data parallelism behind a load balancer) multiply capacity without changing per-user speed.
- **More servers.** Independent servers, each with its own copy of the model behind a load balancer and with no
  state shared between them, scale the same way. Users one server serves = ⌊guaranteed concurrent requests of that
  server ÷ active share⌋, and servers needed = ⌈users ÷ users one server serves⌉, so the servers always cover the users
  (`serverPlan` in `src/planner/plan.ts`, tested on random inputs). Tensor or pipeline parallelism across servers
  (multi-node) is not modelled.

### 7.4 Chunked prefill

Both engines read new prompts in slices of `step_tokens` (vLLM `max_num_batched_tokens`, SGLang
`chunked_prefill_size`) between decode steps. In steady state every finished request is replaced by a new one, so each
decode step also carries `b × P / output_tokens` prompt tokens, where `P` is the uncached part of a typical prompt:

```
t_step_eff = t_step(b) + b × t_prefill(P) / output_tokens
share      = prefill part / t_step_eff
```

This is why long prompts slow down everyone's generation, not only the new request.

### 7.5 Long contexts

The attention efficiency η_attn depends on the context. For the families measured on the RTX 5090 it is fitted per
family as a log-linear curve from 4K to 1M tokens:

```
η_attn(T) = A                                                   T ≤ 4,096
η_attn(T) = A × (1 + (A_1M − 1) × log(T / 4,096) / log(256))    above; A_1M is the ratio reached at 1M tokens
```

The generic calibrations scale a curve measured on the RTX 5090 up to 213K tokens and held flat beyond it. The values at
4K, 32K, 128K and 1M tokens of every calibration are in [Appendix E](appendix-e-fitted-parameters.md).

Sliding-window and sparse-attention layers cap the attended span (7.2), which keeps their cost linear in the context.
Beyond 128K tokens sparse attention also gets a warning: on the RTX 5090 its transient buffers grew with the context.

## 8. Users: guaranteed and likely

Given a workload — maximum context, typical context (default: half of it), output tokens (default 500), minimum speed
per user (default 20 tok/s), maximum wait for the first token (default 10 s), share of users active at once
(default 10 %), safety margin on memory (default 85 %) and the cached share of each prompt (default 0) — the planner
finds the largest number of concurrent requests that satisfies all limits.

**Load point** at `b` concurrent requests:

```
tok/s per user = 1000 / t_step_eff(b)
TTFT           = t_prefill(P) + ceil(P / step_tokens) × t_step(b) + queue + request_overhead
queue          = ρ / (2(1 − ρ)) × t_prefill(P),   ρ = share × (b − 1) / b       (M/D/1 queue of other prompts)
```

**Limits:**

| Limit | Definition |
| --- | --- |
| memory | `floor(requests at the typical context that fit in the KV pool × safety)`, capped by the engine's `max_num_seqs` |
| speed | largest `b` with tok/s ≥ target and TTFT ≤ wait limit (bisection) |
| wait limit | the given maximum (`--max-wait`, default 10 s), or, when `--max-wait` is not given, twice the first-token wait of one typical prompt alone (half the context less what the prefix cache holds; the slow end, with the request overhead) if that is longer (very long contexts) |

**Guaranteed** concurrent requests = `min(memory, speed)` computed on the **slow end** of every speed range: what can be
promised. **Likely** = the same with the **centre** of the ranges: what measurements usually show. Total users =
concurrent requests ÷ active share. The output states which limit binds (memory, speed or wait).

Users include AI agents. Splitting work across parallel agents or subagents gets it done sooner, but each one is a
concurrent client: 5 subagents count as 5 users. Agents are typically active all the time and send long contexts, so
the active share and the context should be raised for them (`--active`, `--tokens`).

Near saturation small changes in load move TTFT sharply (the queue term diverges as `ρ → 1`); this is why the
guaranteed figure keeps both the memory safety margin and the slow end of the speed range.

## 9. Engine and GPU compatibility

Fitting in memory is not enough: an engine release may have no kernel for a weight format, an attention type or a KV
cache type on a GPU generation. `data/compat.json` answers, for every catalog model and checkpoint, engine and version,
and GPU generation (sm80 Ampere A100, sm86 Ampere RTX, sm89 Ada, sm90 Hopper, sm100 Blackwell datacenter, sm120 Blackwell
RTX), one of:

| Status | Meaning |
| --- | --- |
| ok | starts and serves |
| partial | starts with a limit worth knowing (e.g. a slower fallback kernel, a feature that must be turned off) |
| does-not-start | refuses or crashes at start-up |
| unknown | not verifiable from the sources; never assumed to be ok |

Every rule cites its source: the engine's code at the pinned release tag (vLLM v0.31.0, SGLang v0.5.21, with their
commits in the file), an issue or pull request, or the documentation. A rule is never "ok" on an issue alone. Limits
seen only under Windows / WSL2 are kept apart from the general ones. A plan row whose status is does-not-start shows
"engine does not start" with the reason, gets no speed and no users, and is never recommended; DeepSeek-V4.1 on sm120
cards (RTX 50 series, RTX PRO 6000 Blackwell) is the main case in the catalog.

**Pipeline parallelism** is a need of its own (`p:pp`), asked only for layouts with `PP > 1`. vLLM refuses `pp > 1` unless
the model class implements `SupportsPP` (`vllm/config/model.py`), SGLang unless its `forward` takes `pp_proxy_tensors`
(`model_runner.py`); past that check each model has its own limits. Every catalog model has a verdict for both engines,
read in the model's code at the pinned tags. A pipeline that starts but gives wrong output counts as not starting:
GLM-5.3 on vLLM (a stage that starts on a layer reusing the previous top-k reads stale indices, and the default
two-stage split falls on one; `VLLM_PP_LAYER_PARTITION=38,40` avoids it, shown as the workaround), Gemma-4-E4B on vLLM
(per-layer embeddings skipped on later stages) and DeepSeek-V4.1 on vLLM with 3 or more stages. GLM-5.3 does not start
on SGLang; gpt-oss and DeepSeek-V4-Pro are ok on both. A pipeline layout whose `p:pp` is not "ok", or of a model the
matrix does not know ("pipeline parallelism not verified for this model"), is never chosen over a verified layout: it
is ranked after every verified one and carries the caution.

Independently of the matrix, a layout is refused when vLLM would refuse it:

- attention heads, KV heads (when ≥ TP) or linear-attention heads not divisible by `TP`;
- FP8 block quantization: each split FFN dimension `ff / TP` must remain a multiple of the quantization block
  (`weight_block_size`, typically 128);
- more pipeline stages than layers.

It starts with a warning when: FP8 weights run on Ampere (weight-only Marlin kernels: same memory, BF16 compute, slower);
NVFP4 / MXFP4 weights run before Blackwell (weight-only kernels); an FP8 KV cache runs on Ampere (not every attention
backend supports it); tensor parallelism crosses PCIe without NVLink or P2P (all-reduce through host memory).

`npm run dev:update-compat` keeps the matrix in step with the model catalog and re-checks that every cited line of
engine code still says what the rule claims at a new release (section 14).

## 10. Calibration

### 10.1 Calibration runs on an RTX 5090

Calibration runs on an RTX 5090 (32 GB; Windows 11 + WSL2, vLLM 0.31.0 in Docker), fp8 KV cache. The full list — date,
model and layers, parameters, duration, every measured TTFT, generation speed and memory line, and the runs excluded
with their reason — is in [Appendix A](appendix-a-calibration-runs.md).

- **Dummy weights.** vLLM's `--load-format dummy` fills the tensors with random values: speed depends on shapes, not
  on values, so no checkpoint is downloaded.
- **Cut-down models.** Large models are shortened to the number of layers that fits one card (for example
  Qwen3-235B-A22B FP8 to 6 layers, GLM-5.3-Flash to 4). Every layer keeps its real shape, so per-layer costs and the
  expert routing are measured at full scale; the model scales them back to the full depth.
- **Protocol.** One engine start per run; time to first token at 4K–32K tokens (median of 3) and generation with 1, 8
  and 32 concurrent requests; long-context runs from 64K to 1M tokens on models cut to 2–4 layers.
- **Clean runs only.** vLLM's scheduler is CPU-bound: the measurement script checks the CPU load of every other program
  before and during each run and excludes disturbed runs. Runs where the card spilled memory into system RAM, or where the
  server slowed down during the run, are excluded too.

The RTX 5090 calibration is fitted **per architecture family**, as Vidur and GenZ fit per operator: GEMM efficiency by
format, attention efficiency by context, a fixed cost per step and per layer, a cost per touched expert, a recurrent scan
cost per token. Its spread is the family's worst fit error.

### 10.2 Public benchmarks

`data/references.json` holds public vLLM and SGLang measurements ([Appendix B](appendix-b-public-benchmarks.md)): GitHub
issues and pull requests, official recipes, the SGLang cookbook, SemiAnalysis InferenceX, papers and blogs. Each entry
records the engine version, model, hardware, parallel layout, dataset, concurrency and the reported throughput, TTFT and
TPOT, with its URL. They are compared at their own concurrency: generation on pure decode, TPOT with the other requests'
prompts read between steps, TTFT only up to 8 concurrent requests (above that, benchmarks send all prompts at once and the
first wait is a start-up queue, not the steady state the model describes). They fit one calibration per engine for every
card without its own measurements.

### 10.3 Where the estimate is reliable

Each speed carries its basis and the error declared for it (`speedBasis` in `src/planner/calibration.ts`):

| Basis | When | Declared error |
| --- | --- | --- |
| measured | vLLM on the RTX 5090, for the family and format measured: dense BF16, MoE FP8, Gated DeltaNet hybrid FP8 and NVFP4, sparse MLA FP8 | dense ±22 %, MoE ±15 %, hybrid ±10 %, sparse MLA ±23 % |
| fitted | vLLM: dense models in BF16 / FP8, MoE in MXFP4; any catalog card with published bandwidth and TFLOPS | ±12 % generation, ±25 % prompt reading |
| fitted | SGLang: MoE in BF16 / FP8 / NVFP4, MLA in FP8 | ±30 % |
| estimated | any other combination, e.g. MoE in BF16 / FP8 on vLLM (its public points hold none), MLA or hybrids on generic vLLM, INT4, dense on SGLang | the spread doubled |

"Fitted" holds exactly for the family × format combinations the public points cover (`publiclyFitted` in
`src/planner/calibration.ts`; `test/planner/calibration.test.ts` checks the list and the point counts against
`test/fixtures/calibration-points.json`).
| beyond the verified context | longer than the measured context (e.g. MoE beyond 512K, sparse MLA beyond 128K, public benchmarks beyond 8K / 4K) | the spread × 1.5 |

The fitted parameters of every calibration are in [Appendix E](appendix-e-fitted-parameters.md).

## 11. Validation

Every measured point — RTX 5090 runs and public benchmarks — is recomputed offline by the test suite
(`test/planner/calibration-points.test.ts`): every RTX 5090 point must fall inside its family's predicted range, every
public point inside its range or within 5 % of it, and the declared spreads are pinned. A change of the model or of the
calibration that breaks a measurement fails the build. Points the model cannot describe (expert parallelism, an INT4 MoE
on Hopper, runs whose source itself doubts the figure) are named as outliers with their reason
(`CALIBRATION_CHECK` in `src/planner/calibration.ts`) and listed in Appendix B.

[Appendix C](appendix-c-validation.md) gives, for every engine × card × family and format × phase, the number of points,
the mean and worst error against the centre of the predicted range, the declared error and how many points fall inside
the range; then the same by context, the weights check, and every point. A range under load spans pure generation to
generation with the other requests' prompts, so a point can be inside its range with an error above the declared one.

## 12. Comparison with published methods

| Approach | What it does | Accuracy it reports | Relation to this planner |
| --- | --- | --- | --- |
| Roofline tools: [LLM-Viewer](https://github.com/hahnyuan/LLM-Viewer) ([paper](https://arxiv.org/abs/2402.16363)), [llm-analysis](https://github.com/cli99/llm-analysis) | FLOPs and bytes per layer against peak compute and bandwidth | none (a lower bound; efficiencies set by hand) | the physical floor every estimate respects, shown as a share in `--details` |
| [GenZ](https://arxiv.org/abs/2406.01698) | per-operator roofline with efficiencies profiled on real kernels, collectives simulated | 1.9–2.7 % geomean against vLLM (Llama 2 7B/13B, OPT-175B, 500–2,000 input tokens) | the same structure: efficiencies fitted on measurements, a per-step overhead, latency + bytes for collectives |
| [Vidur](https://arxiv.org/abs/2405.05465) (MLSys 2024) | profiles every operator per batch size and TP shard and simulates the scheduler (vLLM, Sarathi chunked prefill) | < 9 % latency error on A100 / H100; up to 12.7 % at 95 % load, attributed to CPU overhead | the cut-down-model method is the same idea per layer; CPU overhead is why runs need the CPU check; the planner keeps a 15 % safety margin below saturation |
| [LLMCompass](https://arxiv.org/abs/2312.03134), [APEX](https://arxiv.org/html/2411.17651), [LLMServingSim](https://arxiv.org/abs/2408.05499) | hardware models and profiled simulators | 4.1 % end-to-end (LLMCompass), 10.7 % on parallel-plan speed-ups (APEX), < 14.7 % (LLMServingSim) | APEX's error grows with MoE expert parallelism: like here, EP is not modelled and is named as an outlier |
| Practical guides: [Databricks](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices), [Baseten](https://www.baseten.co/blog/llm-transformer-inference-guide/), [vLLM optimization](https://docs.vllm.ai/en/latest/configuration/optimization.html), [NVIDIA](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | bandwidth utilisation (MBU), rules of thumb, latency–throughput curves | MBU 55–60 % at batch 1 (Databricks); real latency up to 2× the roofline bound (Baseten) | the decode term of 7.1 and the KV formulas of section 4, extended with the engines' own block, group and memory rules |

The errors here are larger than GenZ's or Vidur's because those profile the exact operators of each model on the target
GPU, while this planner answers for any model on any catalog card without running it: hence the measured / fitted /
estimated bases with declared errors rather than one accuracy figure. What it adds: exact per-tensor weights from the
real checkpoint, the engines' own start-up rules (memory share, block planner, layout checks), the compatibility matrix,
and errors measured against the same engine versions. Not adopted: per-operator profiling across batch sizes,
iteration-level scheduler simulation, per-TP-degree efficiencies.

## 13. Known limits

- **NVIDIA only.** Other accelerators are out of scope.
- **Direct calibration on one card.** Only the RTX 5090 with vLLM is measured directly; other cards use the fit on
  public benchmarks, which cover mainly H100, H200 and B200 at contexts up to 8K (vLLM) and 4K (SGLang).
- **SGLang** is fitted only on MoE and MLA models from the SGLang cookbook; dense models on SGLang are estimated.
- **Not modelled:** expert parallelism, speculative decoding (MTP, EAGLE), disaggregated prefill / decode, multi-node
  deployments, LoRA adapters.
- **Beyond the model's maximum context.** A context longer than the model's `max_position_embeddings` (with no YaRN /
  RoPE override in its config) gets its memory computed as usual, but its speed and users are marked "not precise
  enough to decide": the engine needs an override to start, and no run is calibrated there.
- **SGLang compatibility.** The compatibility matrix has fewer SGLang rules than vLLM rules; a model × card × format with
  no verified rule is shown as "unknown" (engine support unverified), never as starting for sure. The layout checks
  (heads and blocks divisible by the tensor-parallel degree) hold for both engines.
- **INT4 MoE before Blackwell**: public measurements fall far below the model; the speed is flagged as unreliable.
- **Steady state.** Bursts, cold starts and heavy-tailed request lengths are not simulated; near saturation (above 85 %
  of a layout's capacity) queues grow fast and the output marks a larger error.
- **Desktop measurements.** The RTX 5090 runs were made under Windows 11 / WSL2; a Linux server may be faster.
- **Cards without published figures.** A card whose bandwidth or TFLOPS NVIDIA does not publish (e.g. RTX PRO 6000
  Server) gets a memory verdict but no speed.

## 14. Updating data and calibration

When a new engine version is released:

1. **Memory rules.** Re-read the engine's KV planner, memory profiling and scheduler defaults (files in section 16) and
   update `src/model/` accordingly. Update the pinned versions in `src/planner/calibration.ts`.
2. **Weights check.** Run a few models with the new engine and compare `Model loading took` with the computed weights
   (`npm run dev:calibrate` prints the comparison).
3. **Speed.** On a card you own, run `npm run dev:speed` (dummy weights, cut-down models; see `scripts/dev/README.md`),
   then `npm run dev:calibrate` to refit and rewrite `test/fixtures/calibration-points.json`.
4. **Public benchmarks.** Add new entries to `data/references.json` with their sources; `npm test` validates the schema
   and recomputes every point.
5. **Catalogs.** `npm run dev:update-models` refreshes the model catalog; GPU entries are edited by hand with their
   NVIDIA source (`data/README.md`).
6. **Compatibility.** `npm run dev:update-compat -- --anchors --vllm <tag> --sglang <tag>` re-checks every cited line of
   engine code at the new release; update the rules that changed in `data/compat.json`.
7. **Report and documents.** `npm run dev:report` regenerates the precomputed plans in `data/report.json`, and
   `npm run dev:docs` the appendices of this document.

Run `npm run validate` (and `npm run ci:docker` for a clean Linux check) before committing.

## 15. Glossary

| Term | Meaning |
| --- | --- |
| Decode | generating output tokens, one per step per sequence; bound by memory bandwidth |
| Prefill | reading the prompt; bound by compute |
| TTFT | time to first token: queue + prefill + request overhead |
| TPOT | time per output token after the first |
| KV cache | per-request memory of attention keys and values, growing with the context |
| TP / PP | tensor parallelism (each layer split across GPUs) / pipeline parallelism (layers split into stages) |
| Replica | an independent copy of the model behind a load balancer (data parallelism) |
| MoE | mixture of experts: each token uses only `top_k` of `E` expert FFNs |
| MLA | multi-head latent attention (DeepSeek): a compressed KV latent shared by all heads |
| Chunked prefill | reading prompts in fixed-size slices interleaved with decode steps |
| Prefix cache | reuse of the KV of a prompt prefix already seen (system prompt, chat history) |
| Roofline | time = max(bytes / bandwidth, FLOPs / throughput) |
| MFU / MBU | model FLOPs / bandwidth utilization: the share of the peak actually achieved |
| CUDA graph | a pre-recorded GPU launch sequence that removes per-kernel launch overhead |
| Token | the unit models read and write; for English text about 0.75 words, so 1,000 tokens ≈ 750 words. Token counts use k = 1024: "128k" = 131,072 tokens, "1M" = 1,048,576 |
| Page (in the Summary) | 500 words, so 128k tokens ≈ 98,000 words ≈ 200 pages; a declared convention for readers, not a measurement |
| Server | one node with up to 8 GPUs, or fewer where NVIDIA publishes fewer per system (section 7.3); layouts that need more GPUs are said to need more than one server |
| GiB | 2^30 bytes = 1,024 MiB, the unit of every memory figure; nvidia-smi's MiB ÷ 1,024 |

## 16. Engine source references

The rules replicated from the engines, and where they come from (the code at the pinned versions; file names as in the
engines' repositories):

| Rule | Engine source | Here |
| --- | --- | --- |
| KV cache planner: specs, page unification, groups, blocks per request, start-up check, reported concurrency | vLLM 0.31 `v1/core/kv_cache_utils.py`, `kv_cache_interface.py` | `src/model/vllm-kv.ts` |
| Loading rules: vocabulary padded to 64 and split, replicated tensors, KV head replication | vLLM `vocab_parallel_embedding.py`, `QKVParallelLinear`, `ReplicatedLinear` | `src/model/weights-exact.ts` |
| Memory share, profiling run, CUDA graphs inside the share | vLLM `--gpu-memory-utilization` (graphs profiled into the budget since 0.21) | `src/model/serving-fit.ts` |
| Scheduler defaults by card memory | vLLM `engine/arg_utils.py`; SGLang `memory_hook.py`, `cuda_graph_hook.py` | `src/model/serving-fit.ts` |
| SGLang default `--mem-fraction-static` | SGLang `memory_hook.py` | `src/model/serving-fit.ts` |
| Context beyond the native length (YaRN overrides, `max_position_embeddings`) | vLLM `model.py`, SGLang `common.py`, `config.py` (cited per model in `data/models.json`) | `data/models.json` (`longContext`) |
| Engine × GPU × format compatibility | per rule, with file and line at the pinned commit | `data/compat.json` |

## 17. Timing and reproducibility

**One plan.** On a desktop PC, a catalog model answers from the built-in data in about 1.5 s; a model read from Hugging
Face for the first time takes a few seconds more (one request per safetensors shard, eight at a time), then comes from
the local cache.

**A new engine release.** The GPU time of the current calibration is in Appendix A (about two hours of runs on one
RTX 5090, plus engine start-ups). Re-checking the code anchors of the compatibility matrix takes minutes; collecting new
public benchmarks is manual.

**Commands** (development tools: [scripts/dev/README.md](../scripts/dev/README.md)):

| What | Command |
| --- | --- |
| Lint, types, every test with the 100 % coverage gate | `npm run validate` |
| The same in a clean Linux container, on the committed code | `npm run ci:docker` |
| Memory lines of vLLM on the local GPU | `npm run dev:measure` |
| Speed runs (TTFT, generation, long contexts) on the local GPU | `npm run dev:speed` |
| Refit and recompute every calibration point | `npm run dev:calibrate` |
| Refresh the model catalog from Hugging Face | `npm run dev:update-models` |
| Re-check the compatibility matrix against a new release | `npm run dev:update-compat -- --anchors` |
| Regenerate the ready-made reports | `npm run dev:report` |
| Regenerate the appendices of this document | `npm run dev:docs` |
| Regenerate the README images | `npm run dev:screenshots` |

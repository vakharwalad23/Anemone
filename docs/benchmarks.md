# Benchmarks

Every number Anemone quotes, with how it was measured, from 2026-10-23 to 2026-10-25. The first three sections are the Swift engine; the later sections come from the Python lab prototype and two Swift timing tools, which the engine was built from.

## Setup

- Mac: M3 Pro, 18 GB, macOS 27.0.1, Xcode 27.
- AC power, quiet machine (load average about 1), idle power 1.5 to 2.8 W.
- Target model: Qwen3-8B with thinking off. Drafter: `deepseek-ai/dspark_qwen3_8b_block7`.
- Decoding: greedy, 6 prompts (math, code, explain, summary, advice, translate), 256 tokens each.
- ANE target: four 9-layer Core AI chunks, 4-bit lookup tables (16 values per 32 output rows), context 1024, verify width 16.
- GPU baseline: MLX with 4-bit affine weights, group 64, for both the target and the drafter.

Numbers from different macOS versions are not comparable; rerun both sides on the same Mac.

## Qwen3-14B, the MLP layout and 16K context (2026-10-25)

M3 Pro 18 GB, macOS 27.0.1 (26A434), AC power.

### Qwen3-14B

Stock converter, eight 5-layer chunks (`--layers-per-chunk 5`), its own DSpark drafter (`deepseek-ai/dspark_qwen3_14b_block7`) with runner-up tokens. GPU rows: `e15_gpu_baselines.py` with `mlx-community/Qwen3-14B-4bit` (mlx-dspark picked cap 5).

| engine | math | code | explain | summary | advice | translate | mean | equal to plain greedy |
|---|---|---|---|---|---|---|---|---|
| MLX 4-bit plain | 16.8 | 16.7 | 16.7 | 16.8 | 16.7 | 16.9 | 16.8 | 6/6 |
| mlx-dspark | 52.7 | 45.2 | 38.1 | 31.8 | 33.1 | 20.5 | 36.9 | 2/6 |
| **Anemone** | **44.9** | **42.8** | **25.0** | **24.1** | **23.4** | **10.8** | **28.5** | **6/6** |

- Anemone: verify 118.5 ms, draft 23 ms, accepted 3.11 per cycle; plain 8.6 tokens/s; first load 1338 s.
- Energy, one round each (e13 and e15 `--power`): Anemone 0.743 J/token at 19.8 W; mlx-dspark 1.167 at 39.9 W; MLX plain 2.054 at 32.7 W. Anemone uses 1.57x less energy than mlx-dspark at 0.77x its speed.
- The quality check against bf16 (e07) has not been run for the 14B yet.

### The MLP layout

The first 14B export ran at 2.9 tokens/s (verify 1338 ms). One layer, exported alone and timed with the lab's `anebench` at context rung 256 (median of 30 calls, ms per call):

| one layer | width 8 | width 16 | width 32 | width 64 |
|---|---|---|---|---|
| Qwen3-8B, Apple's MLP | 2.13 | 1.80 | 1.92 | 2.12 |
| Qwen3-8B, tokens on the width axis | 2.32 | 1.83 | 1.84 | 2.06 |
| Qwen3-14B, Apple's MLP | 16.77 | 33.41 | 64.25 | - |
| Qwen3-14B, attention only | 2.83 | 0.84 | 0.90 | - |
| **Qwen3-14B, tokens on the width axis** | **2.98** | **3.00** | **3.06** | **5.83** |

- The 8B width-64 cell with Apple's MLP is from back-to-back calls; single calls varied from 2.3 to 2.6 ms.
- Apple's iOS MLP reshapes the tokens onto the convolution's batch axis. For the 14B's larger weights (45 MB per projection at 4 bits, against 25 MB in the 8B) the ANE then runs the MLP once per token, about 2.1 ms each. Attention keeps the tokens on the width axis and costs the same at every width.
- The converter's MLP keeps the tokens on the width axis. Same weights and result (within 1.3e-7 in fp32). The 14B layer compiles in 31 s instead of 229 s, and a full 14B chunk in about 2.6 minutes instead of 19.

### 16384-token context

The 8B exported with `--context 16384` (row-wise attention above 4096), with a drafter cache of the same size; first load 1353 s.

| prompt | speculative | plain | prefill | equal to plain |
|---|---|---|---|---|
| bench prompts (short) | 47.3 | 16.1 | - | 6/6 |
| one prompt of 10923 tokens, 256 out | 10.6 | 6.2 | 57.2 s | yes |

- At 10.9K tokens every call reads the whole KV cache: about 2.6 ms per 1K tokens of context per chunk, so decoding is 4.5x slower than at short context.

## Trees, GPU baselines and energy (2026-10-24 night)

M3 Pro 18 GB, macOS 27.0.1 (26A434), AC power. Model folders exported with `--context 4096` from the published GPTQ weights.

### Token trees

The tree drafter is the same DSpark drafter exported with its runner-up token at each of the 7 steps. The engine puts the likeliest runner-ups (by the drafter's confidences) as one-token side branches into the rows of the width-16 check that the chain leaves free. Chain and tree runs alternated, `anemone bench --only speculative`:

| drafter | run 1 | run 2 | accepted per cycle | draft ms | verify ms | lossless |
|---|---|---|---|---|---|---|
| chain (the published folder) | 44.0 | 43.9 | 2.84 | 21.8 | 64.3 | 6/6 |
| **tree** | **46.0** | **45.6** | **3.01** | **21.3** | **64.4** | **6/6** |

Same verify width, so the check costs the same; the 6% higher acceptance is a 4% speedup.

### GPU baselines

The GPU side from a public implementation: [mlx-dspark](https://github.com/ARahim3/mlx-dspark) 0.20.3 (mlx 0.32.4, mlx-lm 0.32.2), target `mlx-community/Qwen3-8B-4bit`, the same DSpark drafter, run with its defaults (its cap choice of 7, small-M verify kernel, hybrid n-gram drafts) through the lab script `e15_gpu_baselines.py` on the 6 bench prompts, thinking off, greedy, 256 tokens (tokens past 256 dropped before counting).

| engine | math | code | explain | summary | advice | translate | mean, two runs | equal to plain greedy |
|---|---|---|---|---|---|---|---|---|
| MLX 4-bit plain | 30.0 | 30.1 | 30.3 | 30.2 | 30.2 | 30.8 | 30.3 / 30.4 | 6/6 |
| mlx-dspark | 79.7 / 94.9 | 96.7 / 98.3 | 48.6 / 49.4 | 57.7 / 58.3 | 51.7 / 52.6 | 28.0 / 27.9 | 60.4 / 63.6 | 2/6 |
| **Anemone, tree** | **74.3** | **69.3** | **44.7** | **33.9** | **36.5** | **16.8** | **45.9** | **6/6** |

- mlx-dspark's cost curve on this Mac (from its own calibration cache): 48.9 ms to check 8 rows, 11.7 ms to draft 7 tokens, about 61 ms per round. Anemone: 64.4 ms to check 16 rows on the ANE, 21.3 ms to draft, 86 ms per cycle. Tokens per cycle are similar; the cycle time is the gap.
- mlx-dspark's answers drift from MLX plain greedy on 4 of 6 prompts, first at tokens 24, 109, 126 and 173.
- Our earlier MLX + DSpark row (32.2 tokens/s) came from the lab's own MLX loop, which checked 16 rows with stock kernels. It is kept below as a lab result but is not a fair GPU baseline.

### Qwen3-4B

Converted with the stock converter (see [Converting a model](convert.md#qwen3-4b)), two 18-layer chunks, its own DSpark drafter with runner-up tokens (tree).

| engine | math | code | explain | summary | advice | translate | mean | equal to plain greedy |
|---|---|---|---|---|---|---|---|---|
| MLX 4-bit plain | 54.3 | 54.1 | 54.1 | 54.1 | 54.0 | 55.3 | 54.3 / 53.9 | 6/6 |
| mlx-dspark | 147.9 | 126.8 | 82.8 | 71.0 | 84.7 | 52.2 | 94.3 / 94.8 | 2/6 |
| **Anemone** | **127.3** | **96.0** | **63.8** | **55.4** | **66.5** | **34.7** | **74.0** | **6/6** |

- Anemone: verify 36.4 ms, draft 16.2 ms, accepted 2.97 per cycle; plain 28.8 tokens/s.
- Energy, two rounds: Anemone 0.303 / 0.305 J/token at 19.6 W; mlx-dspark 0.430 / 0.423 at 33.9 / 33.1 W; MLX plain 0.640 / 0.638 at 31.1 / 30.9 W.
- Quality against bf16 Qwen3-4B (e07, simulated): docs KL 0.077, top-1 0.891; chat KL 0.112, top-1 0.915. MLX 4-bit: 0.095 / 0.875 and 0.107 / 0.917.

### DeepSeek-R1-0528-Qwen3-8B

Converted with the stock converter and the Qwen3-8B DSpark tree drafter (DeepSeek publishes no drafter for R1). R1 always thinks, so the 256-token answers are thinking text.

| engine | math | code | explain | summary | advice | translate | mean | equal to plain greedy |
|---|---|---|---|---|---|---|---|---|
| MLX 4-bit plain | 30.2 | 30.2 | 30.0 | 29.5 | 30.0 | 30.1 | 30.0 | 6/6 |
| mlx-dspark, same drafter | 52.6 | 38.5 | 37.8 | 34.0 | 34.5 | 34.5 | 38.7 | 0/6 |
| **Anemone** | **44.8** | **34.0** | **28.9** | **21.4** | **25.0** | **24.5** | **29.8** | **6/6** |

- Anemone accepts 1.53 drafts per cycle against 3.01 on Qwen3-8B: the drafter was trained on Qwen3-8B's answers, not on thinking text. Plain 16.1 tokens/s, first load 714 s.
- Quality against bf16 R1 (e07): docs KL 0.240, chat 0.152; MLX 4-bit 0.288 and 0.143.

### Energy

`e13_swift_energy.py` around `anemone bench` and `e15_gpu_baselines.py --power` for MLX, one after another, two rounds. Whole-system SMC power over the generation window including prefill.

| engine | tok/s with prefill | system W | GPU W | J/token |
|---|---|---|---|---|
| **Anemone, tree folder** | **45.8 / 45.8** | **20.8 / 20.8** | **1.0 / 0.9** | **0.488 / 0.488** |
| Anemone, published folder (chain) | 43.7 / 43.3 | 21.5 / 20.7 | 1.0 / 0.9 | 0.527 / 0.510 |
| mlx-dspark | 58.0 / 59.5 | 38.2 / 37.8 | 14.1 / 14.6 | 0.720 / 0.697 |
| MLX 4-bit plain | 29.4 / 29.4 | 33.8 / 33.8 | 10.9 / 11.0 | 1.203 / 1.203 |
| Anemone plain | 16.2 / 16.2 | 17.4 / 17.4 | 0.0 / 0.0 | 1.102 / 1.099 |

```math
E = \frac{\bar{P}\,(t + \tau)}{N}
```

- $\bar{P}$: mean system power over the window, $t$: generation time, $\tau$ = 2 s of sensor lag, $N$: tokens generated.
- Example, Anemone tree round 1: $20.77 \times (26.72 + 2) / 1224 = 0.487$ J.

### Drafter cost

`anebench` times one drafter call alone on the GPU (zero inputs, 50 runs, median, one cache of 1088 slots):

| weights | ms per call |
|---|---|
| **int4, blocks of 32 (shipped)** | **16.3** |
| int4, blocks of 128 | 15.9 |
| int8 per channel | 41.9 |
| fp16 | 38.0 |

- The same call after idle gaps: 16.0 ms back to back, 24.9 ms after 20 ms idle, 32 to 37 ms after 64 ms idle. In the engine the GPU idles through each 64 ms ANE check, and the draft takes 21 ms.
- Keeping the GPU awake with a tiny dispatch every 2 to 5 ms during the check cuts the draft to 17 to 19.5 ms and raises speed 2% to 5%, but adds 2.3 W, so energy per token gets 9% worse (0.48 -> 0.52 J). Anemone does not do it.

## Swift engine (`anemone bench`)

Release build, model folder from the converter (GPTQ LUT4 chunks, Core AI drafter), 2026-10-23 evening, AC power.

| prompt | plain tok/s | speculative tok/s | accepted per cycle | draft ms | verify ms | lossless |
|---|---|---|---|---|---|---|
| math | 16.2 | 77.3 | 5.50 | 20.1 | 62.2 | yes |
| code | 16.3 | 66.9 | 4.54 | 20.5 | 62.2 | yes |
| explain | 16.3 | 42.7 | 2.62 | 21.6 | 62.4 | yes |
| summary | 16.3 | 32.5 | 1.75 | 20.6 | 62.0 | yes |
| advice | 16.3 | 35.2 | 1.93 | 20.9 | 62.3 | yes |
| translate | 16.3 | 18.6 | 0.65 | 19.8 | 62.5 | yes |
| **mean** | **16.3** | **45.5** | **2.83** | **20.6** | **62.3** | **6/6** |

- Acceptance per prompt is identical to the Python lab loop with the same drafter (e12), so the engine reproduces the lab token for token.
- Host time outside the draft and verify calls is about 0.1 ms per cycle. Building input arrays with `NDArray(scalars:)` copied element by element and cost 7 ms per draft; copying them in one pass took the mean from 40.7 to 45.5 tokens/s.
- Each call uses the smallest context rung that fits (256, 512 or 1024), which is why plain decoding (61.3 ms per token) beats the lab's 65.8 ms.
- First load of a model folder: 452 s (ANE compilation, cached per executable). Later loads: 3.9 to 4.3 s.

### Long context (4096 export)

Same engine, model folder exported with `--context 4096` (rungs 256 to 4096) and a drafter with two caches (1088 and 4160 slots).

| run | prefill | decode tok/s | accepted per cycle |
|---|---|---|---|
| 2725-token prompt, plain | 4614 ms (591 tok/s) | 11.3 | - |
| **2725-token prompt, speculative** | **4704 ms** | **30.5** | **2.53** |
| 984-token prompt, plain / speculative (crosses 1024 while decoding) | 1313 / 1376 ms | 14.2 / 35.1 | 2.36 |
| 6 short prompts, speculative (mean) | - | 45.0 | 2.84 |

- Output of the long prompt is identical with and without the drafter.
- Before the drafter's `inject` entry point, speculative prefill took 10936 ms: the prompt reached the drafter 8 rows per call. With 64 rows per call (7.1 ms each) it adds about 90 ms.
- On short prompts the 4096 folder runs as fast as the 1024 folder: the verify uses the smallest rung per call (62.5 ms) and the drafter its 1088-slot cache (21.2 ms). With one 4160-slot cache the drafter took 24.7 ms and the mean was 43.3 tokens/s.
- Every run above gives the same text with and without the drafter, including the 984-token prompt, whose drafter cache grows from 1088 to 4160 slots mid-reply.

### Energy, same night

Whole-system SMC power over the generation window including prefill, after 10 s of idle, two rounds each. Anemone from `e13_swift_energy.py`, MLX from `e08_energy.py`.

| engine | tok/s incl. prefill | system W | GPU W | J/token above idle | J/token total |
|---|---|---|---|---|---|
| **Anemone, speculative** | **43.7 / 43.9** | **19.2 / 19.7** | **1.0 / 1.1** | **0.397 / 0.420** | **0.471 / 0.480** |
| Anemone, plain | 16.5 / 16.4 | 15.3 / 15.5 | 0.0 / 0.0 | 0.801 / 0.749 | 0.958 / 0.970 |
| MLX + DSpark | 32.4 / 32.4 | 40.3 / 40.3 | 19.9 / 20.0 | 1.221 / 1.232 | 1.311 / 1.309 |
| MLX plain | 28.9 / 29.2 | 31.6 / 31.8 | 10.4 / 10.7 | 1.042 / 1.052 | 1.144 / 1.139 |

- 2.4x less energy per token than MLX plain and 2.7x less than MLX with the same drafter, at about half the wall power of MLX + DSpark.
- The Core AI drafter draws about 1 W on the GPU; the MLX drafter in the lab prototype drew 3.5 W.

## Speed (Python lab prototype)

Decode tokens/s per prompt. In brackets: mean guesses accepted per cycle.

| prompt | MLX | MLX + DSpark | ANE + DSpark, Core AI default weights | **ANE + DSpark, Anemone weights** |
|---|---|---|---|---|
| math | 30.7 | 53.0 (5.14) | 55.2 (3.98) | **65.6 (5.07)** |
| code | 30.7 | 42.5 (3.94) | 68.0 (5.10) | **57.5 (4.20)** |
| explain | 30.4 | 28.0 (2.07) | 37.4 (2.34) | **42.5 (2.81)** |
| summary | 30.5 | 30.3 (2.38) | 29.3 (1.61) | **29.9 (1.71)** |
| advice | 30.5 | 26.5 (1.93) | 38.3 (2.41) | **33.5 (1.99)** |
| translate (27-token answer) | 31.7 | 12.8 (0.61) | 16.1 (0.56) | **16.9 (0.65)** |
| mean | 30.8 | 32.2 | 40.7 | **41.0** |

- Plain ANE decode without a drafter: 65.8 ms per token, 15.1 tokens/s.
- One ANE cycle: 21.1 ms draft plus 67.5 ms verify.
- ANE speculative output was identical to plain ANE greedy output on all runs. MLX with DSpark differs from plain MLX on 3 of 6 prompts because its numerics depend on the batch width.
- Acceptance differs between weight sets because each produces slightly different greedy text.

## Energy (Python lab prototype)

Whole-system power from the SMC `PSTR` sensor and GPU power from IOReport, read without root. Each engine runs all 6 prompts in one window plus a 2 s tail, after a 10 s settle and 10 s of idle. Two rounds, interleaved; both values shown.

| engine | tokens/s incl. prefill | system W | GPU W | J/token above idle | J/token total |
|---|---|---|---|---|---|
| **ANE + DSpark** | **39.9 / 39.9** | **19.8 / 19.8** | **3.5 / 3.6** | **0.473 / 0.453** | **0.528 / 0.529** |
| MLX + DSpark | 32.4 / 32.3 | 39.1 / 39.3 | 19.6 / 19.5 | 1.222 / 1.196 | 1.271 / 1.280 |
| ANE plain | 15.1 / 15.1 | 14.1 / 14.0 | 0.0 / 0.0 | 0.835 / 0.845 | 0.954 / 0.949 |
| MLX plain | 29.4 / 29.4 | 30.8 / 31.0 | 10.4 / 10.5 | 1.047 / 1.056 | 1.101 / 1.108 |

- macOS 27 keeps the CPU, ANE and DRAM energy counters frozen without root, so there is no ANE-only figure.
- The SMC reading is smoothed: about 4 s to rise after a load step and 2 s to fall.

## Quality

Teacher-forced against the bf16 model. Docs: 4 text and code files, 512 tokens each. Chat: the 6 prompts with the ANE's greedy answers. KL and top-1 are defined in [Math](math.md#quality-against-the-bf16-model).

| 4-bit weights | docs KL / top-1 | chat KL / top-1 |
|---|---|---|
| Core AI default palettizer, on the ANE | 0.341 / 0.767 | 0.371 / 0.825 |
| Weighted k-means, on the ANE | 0.149 / 0.820 | 0.162 / 0.884 |
| Weighted k-means + GPTQ, docs and code calibration, on the ANE | 0.044 / 0.902 | 0.139 / 0.899 |
| **Weighted k-means + GPTQ, plus chat calibration, on the ANE** | **0.047 / 0.892** | **0.086 / 0.919** |
| MLX 4-bit affine, group 64, on the GPU | 0.085 / 0.870 | 0.086 / 0.902 |

Simulated in PyTorch before export (no ANE involved):

| weights | docs KL | chat KL |
|---|---|---|
| Weighted k-means + GPTQ, plus chat calibration | 0.045 | 0.084 |
| Weighted k-means + GPTQ, standard library + chat calibration | 0.107 | 0.080 |
| **Weighted k-means + GPTQ, converter default calibration (standard library, ML library code, chat)** | **0.059** | **0.076** |
| 6-bit lookup tables, plain k-means (1.5x the bytes) | 0.025 | 0.030 |
| 3-bit lookup tables, weighted k-means + GPTQ (0.75x the bytes) | 0.202 | 0.391 |

- The simulator matched the ANE within 0.004 KL on every export measured.
- Chat calibration: 4.3k tokens of the model's own answers to 16 prompts unrelated to the test prompts.
- Against MLX 4-bit, the lab weights are 1.8x lower on docs KL and tied on chat KL (0.0864 vs 0.0859, lower on 5 of 6 prompts), with higher top-1 on both.
- Caveat: the lab calibrated on code and docs from the same repo as the four docs files, so the 0.047 docs KL is partly a domain match. Calibrating on six Python standard library files plus the chat answers scores 0.107 on docs (worse than MLX). Adding five machine-learning library modules (transformers, torch, huggingface_hub; none from the docs repo) brings docs to 0.059 / 0.887 and chat to 0.076 / 0.915, lower KL than MLX on both; that is the converter's default now.

## Verify cost on the ANE (Swift timing tool)

50 runs each, zero inputs (timing does not depend on values).

| context, width | chunks 1 to 3 (9 layers each) | last chunk (9 layers + output layer) | full verify |
|---|---|---|---|
| 256, 16 | 13.63 ms | 19.65 ms | 60.5 ms |
| 512, 16 | 13.95 ms | 19.97 ms | 61.8 ms |
| 1024, 8 | 14.48 ms | 20.48 ms | 63.9 ms |
| 1024, 16 | 14.75 ms | 20.99 ms | 65.3 ms |
| 1024, 32 | 16.39 ms | 22.84 ms | 72.0 ms |
| 1024, 64 | 18.91 ms | 25.68 ms | 82.4 ms |

- The same verify through Python: 67.5, 74.8 and 86.4 ms at widths 16, 32 and 64. Host overhead is 2 to 4 ms; streaming 3.84 GB of weights is the cost.
- Chaining the chunks with `encode` costs the same as `run`.
- First load: 98 to 116 s per chunk, about 7 min for the model. A second process of the same binary loads in 0.0 s.

## Long context (one chunk, width 16)

| attention | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|
| Apple's iOS SDPA (exports up to 4096) | 14.76 | 17.09 | 21.72 | 46.25 | - |
| **RowSDPA (exports above 4096)** | **15.31** | - | **20.83** | **30.96** | **54.04** |

- Apple's SDPA moves the context onto the channel axis before the softmax; past 4096 that transpose dominates (8192 costs 213% more than 1024). `RowSDPA` keeps the scores as rows and takes one last-axis softmax per KV group: 8192 costs 1.49x the 4096 rung, and past 4096 the cost grows about 2.6 ms per 1k tokens per 9-layer chunk (lab e20, 2026-10-25).
- It is 3% slower at 1024, so 4096 exports keep Apple's SDPA.

8192 folder (converter weights, tree drafter, `--context 8192` for `target` and `drafter`), 2026-10-25:

| run | prefill | decode tok/s | accepted per cycle | lossless |
|---|---|---|---|---|
| 6 bench prompts | - | 46.7 (4096 folder: 46.8) | 3.06 | 6/6 |
| 5449-token prompt, speculative | 17.0 s | 22.7 | 2.67 | yes |
| 5449-token prompt, plain | 15.6 s | 9.5 | - | - |

- First load: 1135 s. Export: target 336 s, drafter 95 s.

Against the GPU on one 5423-token prompt (the Anemone docs plus a request for a five-bullet summary), 2026-10-25, energy with `e21_run_energy.py` (model load left out) and `e15 --power`, two rounds, prefill included:

| engine | answer tokens | prefill | decode tok/s | system W | energy per request |
|---|---|---|---|---|---|
| **Anemone, 8192 folder** | **203** | **16.7 s** | **17.9** | **14.9** | **456 J** |
| MLX 4-bit plain | 220 | 19.1 s | 25.4 | 36.1 | 1069 J |
| mlx-dspark | 235 | 19.2 s | 31.3 | 37.4 | 1077 J |

- The GPU decodes faster at this length: the ANE's attention cost grows with context, and the drafter accepts 1.87 per cycle on this summary. Anemone prefills 13% faster and uses 2.3x less energy per request.

## Drafter runtime

Measured on battery power (100% to 83%, Low Power Mode off), so the absolute numbers are lower than the AC tables above. Both drafters ran in the same session against the same ANE target.

| drafter | tokens/s, 6-prompt mean | accepted per cycle | ms per cycle | lossless |
|---|---|---|---|---|
| MLX 4-bit, two runs | 38.0 / 40.6 | 2.74 | 95.8 / 90.7 | 12/12 |
| **Core AI GPU program, int4** | **41.5** | **2.83** | **90.7** | **6/6** |
| Core AI GPU program, 65536-token vocabulary | 41.0 | 2.66 | 88.5 | 6/6 |

Swift timing tool, ANE context 1024, width 16, 50 runs:

| cycle | ms |
|---|---|
| draft alone | 21.5 |
| verify alone | 66.4 |
| draft, then verify, each call waited for | 87.4 |
| **draft fed into verify with `encode`, one wait at the end** | **83.7** |

- The drafter writes the verify input itself, so chaining needs no CPU step between the GPU and the ANE.
- The smaller draft vocabulary saves about 1 ms per draft on the GPU and costs acceptance, so the full vocabulary stays.

## Token trees (Python)

| mode | tokens/s | accepted per cycle | verify ms | draft ms |
|---|---|---|---|---|
| chain | 41.3 | 2.74 | 67.5 | 21.1 |
| tree, width 32 | 41.1 | 3.29 | 74.8 | 27.3 |
| tree, width 64 | 37.4 | - | 86.4 | 28.8 |

Verify and draft times are from the first GPTQ export, which has the same bytes. Trees accept 20% more but the extra drafting cancels it in Python. They need batched branch drafting.

## Draft vocabulary

| draft vocabulary | draft ms | mean accepted |
|---|---|---|
| 151936 (full) | 19.8 | 2.68 |
| 65536 | 15.1 | 2.59 |
| 32768 | 13.3 | 2.23 |
| 16384 | 12.4 | 1.72 |

Measured with the MLX target. Token frequencies come from 24 MB of local text, mostly code.

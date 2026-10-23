# Benchmarks

Every number Anemone quotes, with how it was measured, all from 2026-10-23. The first section is the Swift engine; the later sections come from the Python lab prototype and two Swift timing tools, which the engine was built from.

## Setup

- Mac: M3 Pro, 18 GB, macOS 27.0.1, Xcode 27.
- AC power, quiet machine (load average about 1), idle power 1.5 to 2.8 W.
- Target model: Qwen3-8B with thinking off. Drafter: `deepseek-ai/dspark_qwen3_8b_block7`.
- Decoding: greedy, 6 prompts (math, code, explain, summary, advice, translate), 256 tokens each.
- ANE target: four 9-layer Core AI chunks, 4-bit lookup tables (16 values per 32 output rows), context 1024, verify width 16.
- GPU baseline: MLX with 4-bit affine weights, group 64, for both the target and the drafter.

Numbers from different macOS versions are not comparable; rerun both sides on the same Mac.

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

Same engine, model folder exported with `--context 4096` (rungs 256 to 4096) and a drafter with 4160 cache slots.

| run | prefill | decode tok/s | accepted per cycle |
|---|---|---|---|
| 2725-token prompt, plain | 4614 ms (591 tok/s) | 11.3 | - |
| **2725-token prompt, speculative** | **4704 ms** | **30.5** | **2.53** |
| 6 short prompts, speculative (mean) | - | 43.3 | 2.84 |

- Output of the long prompt is identical with and without the drafter.
- Before the drafter's `inject` entry point, speculative prefill took 10936 ms: the prompt reached the drafter 8 rows per call. With 64 rows per call (7.1 ms each) it adds about 90 ms.
- On short prompts the 4096 folder verifies as fast as the 1024 folder (62.5 ms, smallest rung per call); drafting takes 24.7 ms instead of 21.4 because the drafter attends over all 4160 slots.

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
| 6-bit lookup tables, plain k-means (1.5x the bytes) | 0.025 | 0.030 |
| 3-bit lookup tables, weighted k-means + GPTQ (0.75x the bytes) | 0.202 | 0.391 |

- The simulator matched the ANE within 0.004 KL on every export measured.
- Chat calibration: 4.3k tokens of the model's own answers to 16 prompts unrelated to the test prompts.
- Against MLX 4-bit, Anemone's weights are 1.8x lower on docs KL and tied on chat KL (0.0864 vs 0.0859, lower on 5 of 6 prompts), with higher top-1 on both.

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

| context | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| ms per chunk | 14.76 | 17.09 | 21.72 | 46.25 |

- 4096 costs 47% more than 1024; 8192 costs 213% more. The extra KV reads explain only a small part ([Math](math.md#kv-cache-traffic-at-long-context)).
- Projected full verify: about 93 ms at 4096 and 190 ms at 8192.

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

# S3: smaller encoder weights on the ANE

An int8 Orukeet encoder (per-channel weights, no calibration) runs on the ANE as two 12-layer programs with four window functions each. Below 10 s it is about twice as fast as FP16, and on clips up to 15 s the whole pipeline needs 30% (FLEURS) and 35% (LibriSpeech) less time per clip than TapTalk's 6-bit Core ML build, timed the same night, at equal or lower WER. 6-bit and 4-bit lookup-table weights do not compile for the ANE at this size yet.

All numbers: Apple M3 Pro, 18 GB, macOS 27.0.1 (26A434), AC power, 2026-10-28.

## Result

Clips up to 15.0 s, A B B A full-set runs, median per clip of each model's two runs:

| Set (clips) | TapTalk 6-bit Core ML | Core AI FP16 windows | **Core AI int8 windows** | int8 / TapTalk per clip |
|---|---|---|---|---|
| FLEURS en_us test (595) | 50.7 ms | 44.2 ms | **36.1 ms** | 0.701 (faster on 524 clips) |
| LibriSpeech test-clean (2382) | 43.6 ms | 36.2 ms | **28.3 ms** | 0.652 (faster on 2355 clips) |

The FP16 column comes from a separate A B B A pair the same night (TapTalk 50.3 and 43.8 ms there).

Accuracy on the same clips (CI = paired bootstrap of TapTalk minus int8, 95%):

| Set | TapTalk 6-bit | Core AI FP16 | **Core AI int8** | CI (pp) |
|---|---|---|---|---|
| FLEURS en_us test | 3.83% (485 / 12652) | 3.50% (443) | **3.45% (437)** | [+0.17, +0.60] |
| LibriSpeech test-clean | 1.53% (630 / 41067) | 1.52% (624) | **1.52% (623)** | [-0.03, +0.07] |

int8 tokens equal FP16's on 608 of 647 FLEURS and 2565 of 2620 LibriSpeech clips; against `transformers` 5.9.0 `generate` (FP32) 97 of 100 clips are identical, as for FP16 on the ANE.

## Encoder time per window

`anemone-speech bench`, median of 20 warm calls; int8 is the sum of its two programs:

| Window | Frames | FP16 | **int8** | int8 projected |
|---|---|---|---|---|
| 2.5 s | 32 | 20.6 ms | **12.1 ms** | 11.3 ms |
| 5 s | 63 | 21.4 ms | **12.5 ms** | 11.3 ms |
| 10 s | 126 | 24.1 ms | **15.8 ms** | 18.2 ms |
| 15 s | 188 | 31.4 ms | 30.2 ms | 27.0 ms |

Projected with the encoder time model of the S2 report, `b` = 8 bits and two calls (`c` = 0.6 ms). Worked example, 5 s: the weight floor is 6.09e8 x 8 / (8 x 57e9) = 10.7 ms, above the 8.8 ms of compute, so 10.7 + 0.6 = 11.3 ms; measured 12.5 ms. int8 halves the weight floor, so windows up to 10 s run near it; at 15 s both formats are bound by compute.

## What it took

Three ANE compiler limits, each found by bisecting with `anemone-speech bench`'s placement check. Without the fixes Core AI runs the encoder on the GPU at 2 to 4x the FP16 time, and the only sign is `ANECCompile() FAILED` (code 22) on stderr.

| Limit | Fix (`speech/convert/ane_encoder.py`, `export_orukeet.py`) |
|---|---|
| Compressed `nn.Linear` weights fail from 3 layers per program | Every linear as a 1x1 Conv2d on the channels-first layout |
| More than 18 layers of int8 per program fail | Two programs of 12 layers (`--split-at 12`); the second takes hidden states and an encoder-rate mask |
| A program that starts or ends on a LayerNorm fails | The split boundary multiplies by the frame mask, exact because padded frames never reach real ones |

int8 comes from coreai-opt 0.3.0 (`QuantizerConfig.presets.w8`, symmetric per output channel); exporting takes 1 min. The int8 encoder matches Apple's FP32 reference module to 4.6e-6 before quantization.

## What does not work yet

- 6-bit LUT (k-means per 16 rows, TapTalk's bit width): 2 layers per program compile (3.9 ms at 15 s), 6 do not; the 12-layer programs run on the GPU. UNVERIFIED guess: a limit on lookup-table tensors per program (Anemone's working 4-bit chunks hold 63; 6 encoder layers hold about 78). Exporting takes 10 min.
- 4-bit was not tried: it needs the LUT path above and GPTQ calibration on held-out audio.
- At 15 s int8 is 3.9 ms slower than TapTalk's 6-bit encoder (30.2 vs 26.3 ms); routing keeps most clips off the 15 s window.

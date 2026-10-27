# S2: one encoder, several window lengths

TapTalk's Orukeet encoder always runs a 15 s window, however short the clip. One Core AI encoder program with four static functions (2.5, 5, 10 and 15 s) over one set of FP16 weights, each clip on the shortest window that holds it, transcribes FLEURS and LibriSpeech with the same tokens as the 15 s window on every clip. On clips up to 15 s, timed the same night in alternating runs, it needs 13% (FLEURS) and 19% (LibriSpeech) less time per clip than TapTalk's 6-bit Core ML build, at equal or lower WER, on the ANE and the CPU with the GPU left free.

All numbers: Apple M3 Pro, 18 GB, macOS 27.0.1 (26A434), AC power, 2026-10-28.

## Result

Clips up to 15.0 s, two full-set runs each in the order A B B A per set (A: FluidAudio 0.17.5, TapTalk's runtime, on its installed model; B: `anemone-speech transcribe`), per-clip time = mean of a model's two runs, then the median over clips:

| Set (clips) | TapTalk 6-bit Core ML (A) | **Core AI FP16 windows (B)** | Median of B / A per clip | B faster on |
|---|---|---|---|---|
| FLEURS en_us test (595) | 50.3 ms | **44.2 ms** | 0.871 | 471 / 595 clips |
| LibriSpeech test-clean (2382) | 43.8 ms | **36.2 ms** | 0.811 | 2310 / 2382 clips |

Accuracy on the same clips (Orukeet's English scorer; CI = paired bootstrap of 6-bit minus FP16 windows, 95%):

| Set | TapTalk 6-bit | Core AI FP16 windows | CI (pp) |
|---|---|---|---|
| FLEURS en_us test | 3.83% (485 / 12652) | **3.50% (443 / 12652)** | [+0.14, +0.53] |
| LibriSpeech test-clean | 1.53% (630 / 41067) | **1.52% (624 / 41067)** | [-0.04, +0.06] |

On FLEURS the 6-bit build makes significantly more errors; on LibriSpeech the two tie. The windows bundle's tokens equal the 15 s FP16 bundle's on 3267 of 3267 clips, so window routing changes speed only.

## Encoder time per window

`anemone-speech bench`, median of 20 warm calls, functions interleaved, all four fully on the ANE:

| Window | Encoder frames | Measured | Projected | Clips routed here (FLEURS / LibriSpeech) |
|---|---|---|---|---|
| 2.5 s | 32 | 20.6 ms | 21.7 ms | 0 / 189 |
| 5 s | 63 | 21.4 ms | 21.7 ms | 28 / 904 |
| 10 s | 126 | 24.1 ms | 21.7 ms | 336 / 915 |
| 15 s | 188 | 31.4 ms | 26.7 ms | 283 / 612 (52 / 238 cut) |

```math
t_{enc}(T) \approx \max\left(\frac{F(T)}{R},\ \frac{P_{enc}\, b}{8\, W}\right) + c,
\qquad F(T) \approx 2 P_{enc} T + 4 L T^2 d
```

`T`: encoder frames; `F(T)`: FLOPs per window; `P_enc` = 6.09e8 encoder weights; `b` = 16 bits; `L` = 24 layers; `d` = 1024; `R` = 8.8e12 FLOP/s (from TapTalk's 26.5 ms at 188 frames); `W` = 57 GB/s ANE weight bandwidth (Anemone lab e04); `c` = 0.3 ms per call. Worked example, 5 s: F = 2 x 6.09e8 x 63 + 4 x 24 x 63^2 x 1024 = 7.71e10 FLOP, 8.8 ms of compute; the FP16 weight floor is 6.09e8 x 16 / (8 x 57e9) = 21.4 ms; projected 21.4 + 0.3 = 21.7 ms, measured 21.4 ms.

Below about 10 s the FP16 encoder is bound by its 1.22 GB of weights, so shorter windows stop paying: 2.5 s costs 20.6 ms, not the 4 ms its compute needs. At 15 s it runs 18% over the model, which assumed compute and weight streaming overlap fully.

## What does not work yet

- Smaller weights, the way past the 21 ms FP16 floor. int8 (per-channel, coreai-opt `QuantizerConfig.presets.w8`) and 6-bit LUT (k-means per 16 rows) encoders export and match the reference, but the ANE compiler (`ANECCompile() FAILED`, code 22) rejects them, and Core AI then runs them on the GPU at 2 to 4x the FP16 time. Bisected: with `nn.Linear` the failure starts at 3 layers per program; with every linear as a 1x1 Conv2d, 18 layers compile (22.5 ms at 15 s, int8) and 24 do not; split into two 12-layer programs, each compiles alone but not as window functions specialized from one dynamic graph. Static compressed programs per window are the next thing to try.
- 4-bit weights (S3) wait on the item above.
- Svarah `small` (172 clips, median 4.2 s) is not scored yet: it needs the gated 1.09 GB download. With FP16 weights its 5 s window would save about 31.4 - 21.4 = 10 ms of the 41.1 ms TapTalk median, short of the 30% target; that target needs the compressed encoder.
- Clips over 15 s are still cut (S5). Clips under 1 s keep TapTalk's padding rule; neither set has one.

## Reproduce

```sh
uv run --project speech python speech/convert/export_orukeet.py --checkpoint work/speech/orukeet --dtype float16 --audio-seconds 15.0 --windows 2.5 5 10 --out work/speech/exports-windows --ane-encoder
E=work/speech/exports-windows/orukeet_float16_dynamic/orukeet_float16_dynamic_encoder.aimodel
.build/release/anemone-speech bench "$E#main_w32" "$E#main_w63" "$E#main_w126" "$E#main_w188" --unit ane
.build/release/anemone-speech transcribe --bundle work/speech/exports-windows/orukeet_float16_dynamic --manifest <set>/manifest.json --audio <set wavs> --out <hyps.json>
```

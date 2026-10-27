# S1: Orukeet's encoder on the Neural Engine

Apple's Parakeet export of Orukeet does not run on the ANE: Core AI places the whole FP16 encoder on the GPU, 96.6 ms per 15 s window. Three changes to the traced encoder put every op on the ANE: 31.3 ms per window, against 26.3 ms for TapTalk's 6-bit Core ML encoder timed in the same run. With the two small decoder graphs on the CPU, a whole clip takes about as long as TapTalk, at equal or better accuracy.

All numbers: Apple M3 Pro, 18 GB, macOS 27.0.1 (26A434), AC power, 2026-10-28, `anemone-speech bench` (median of 20 warm calls after 5 untimed, graphs interleaved call by call) unless a row says otherwise.

## Result

Encoder, one 15.0 s window (1501 mel frames, 188 encoder frames):

| Encoder | Where it ran | Median | p10 / p90 |
|---|---|---|---|
| Apple's export, FP16 | GPU (no ANE region) | 96.6 ms | 56.0 / 130.7 ms |
| **ANE export, FP16** | **ANE, every op** | **31.3 ms** | 31.1 / 34.3 ms |
| TapTalk 6-bit Core ML (same run) | ANE | 26.3 ms | 26.1 / 26.6 ms |

Decoder graphs, one call each (median of 50):

| Graph | CPU | GPU | ANE |
|---|---|---|---|
| `decoder_step` (LSTM, 640) | **0.15 ms** | 0.52 ms | 0.75 ms |
| `joint` | **0.14 ms** | 0.42 ms | 0.46 ms |

Whole clips, ANE encoder plus CPU decoder (`anemone-speech transcribe`, one window of 15.0 s, all clips of each set). Clips longer than 15.0 s are cut, so WER is on clips up to 15.0 s:

| Set | Median per clip | Encode (mel + encoder) | Decode | WER, clips <= 15 s | TapTalk 6-bit WER, same clips |
|---|---|---|---|---|---|
| FLEURS en_us test | 51.0 ms | 36.4 ms | 14.2 ms | **3.50%** (443 / 12652) | 3.83% (485) |
| LibriSpeech test-clean | 45.0 ms | 34.6 ms | 10.2 ms | **1.52%** (624 / 41067) | 1.53% (630) |

TapTalk's own medians from its 2026-10-14 run on the same macOS build are 50.8 and 44.4 ms per clip; a same-night comparison follows in the S2 report. Token parity of this bundle against `transformers` 5.9.0 `generate` (FP32): 97 of 100 clips identical (48 FLEURS, 49 LibriSpeech); the WER is unchanged.

## What kept the encoder off the ANE

When one op of a function is not ANE-eligible, Core AI runs the whole function on the GPU without an error. Anemone's placement check (`functionsOffNeuralEngine`, which looks for `_ANE_region_` names in the compiled program) found none, and the compiler logged why when asked to place it on the ANE.

| Cause | Compiler message | Change (`speech/convert/ane_encoder.py`) |
|---|---|---|
| Bool padding mask and int length arithmetic in every layer, softmax upcast to fp32, `arange` position ids | `Incompatible element type for ANE: expected fp16, ...` (21 ops), 3 unsupported fusions | Masks in fp16; each subsampling stage's mask is the previous one at even frames (exact for a prefix mask through a kernel-3, padding-1, stride-2 conv); attention gets an additive -1e4 bias as in NeMo; softmax in fp16; position table precomputed |
| Bool `attention_mask` input | `ANE I/O op can only do F16 MemRef <-> F32 Tensor cast` | The mask enters as fp16 (1 or 0) |
| Conformer conv module as `Conv1d` with BatchNorm | none; no ANE region | 4D `[1, C, 1, T]` Conv2d with BatchNorm folded into the depthwise kernel |

A probe per component found the last one: LayerNorm, the linears, the feed-forward and the relative attention each placed on the ANE alone; the conv module did not until rewritten (1.5 ms off the ANE, 0.33 ms on it). The rewritten encoder matches Apple's reference module to a maximum absolute error of 7.8e-6 in FP32 (125 of 188 frames real audio), checked before every export.

CoreAISpeech fills the mask as bool, so these bundles run through Anemone's own loop (`Sources/AnemoneSpeech/Recognizer.swift`): CoreAISpeech's mel front end and TDT decoder, with each graph loaded on an explicit compute unit through Anemone's `loadProgram`.

## M3 weight erratum

Anemone's notes carry an M3 erratum: weight DMA in exact 1 MiB multiples drops to about 18 GB/s. Every FP16 matrix here is an exact multiple (1024 x 1024 x 2 bytes = 2 MiB). It does not apply: the shortest window (S2) is bound by weight reads and runs at 59 GB/s.

```math
W = \frac{B}{t}
```

`W`: weight bandwidth; `B`: encoder weight bytes; `t`: time of the 2.5 s window, where compute is small. Worked example: 1,220,037,446 bytes / 20.6 ms = 59.2 GB/s, matching the 57 to 59 GB/s Anemone measured for Qwen3 (lab e04).

## What does not work yet

- The FP16 encoder is 5 ms slower than the 6-bit one at 15 s. Smaller weights should close that: int8 and LUT encoders compile, but the ANE compiler fails on them in one program (S2 report).
- 3 of 100 clips are not token-identical to the reference decode on the ANE (99 of 100 on the GPU); WER is unchanged.
- The mel front end costs about 5 ms per clip on the CPU (CoreAISpeech computes a 512-point DFT by matrix product). Not yet optimized.

## Reproduce

```sh
make -f speech/Makefile env download build
uv run --project speech python speech/convert/export_orukeet.py --checkpoint work/speech/orukeet --dtype float16 --audio-seconds 15.0 --out work/speech/exports-ane --ane-encoder
.build/release/anemone-speech bench work/speech/exports-ane/orukeet_float16_static/orukeet_float16_static_encoder.aimodel --unit ane
.build/release/anemone-speech transcribe --bundle work/speech/exports-ane/orukeet_float16_static --manifest <set>/manifest.json --audio <set wavs> --out <hyps.json>
```

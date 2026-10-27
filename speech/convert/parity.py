import argparse
import json
from pathlib import Path

import librosa
import torch
import transformers


def reference_tokens(model: transformers.PreTrainedModel, processor: transformers.ProcessorMixin, wav: Path) -> list[int]:
    audio, rate = librosa.load(wav, sr=None)
    inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, return_dict_in_generate=True)
    return [t for t in out.sequences[0].tolist() if t < model.config.blank_token_id]


def first_difference(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare anemone-speech tokens with Transformers generate on the first clips of a corpus.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--hyps", type=Path, required=True, help="anemone-speech output")
    parser.add_argument("--clips", type=int, default=50)
    parser.add_argument("--max-seconds", type=float, required=True, help="bundle window; longer clips are truncated by it")
    args = parser.parse_args()
    model = transformers.AutoModelForTDT.from_pretrained(args.checkpoint, dtype=torch.float32).eval()
    processor = transformers.AutoProcessor.from_pretrained(args.checkpoint)
    tokens = {row["id"]: row["tokens"] for row in json.loads(args.hyps.read_text())}
    clips = [c for c in json.loads(args.manifest.read_text())["clips"] if c["seconds"] <= args.max_seconds][: args.clips]
    same = 0
    for clip in clips:
        ref, got = reference_tokens(model, processor, args.audio / clip["wav"]), tokens[clip["id"]]
        if got == ref:
            same += 1
        else:
            print(f"{clip['id']}: first difference at token {first_difference(got, ref)} ({len(got)} vs {len(ref)} tokens)")
    print(f"{same}/{len(clips)} clips token-identical to transformers {transformers.__version__} generate (FP32, CPU)")


if __name__ == "__main__":
    main()

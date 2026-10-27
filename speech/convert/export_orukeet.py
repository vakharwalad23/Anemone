import argparse
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

import torch
import transformers
from ane_encoder import AneEncoder
from compress import WEIGHTS, compress
from coreai.runtime import AIModelAssetMetadata


def load_exporter(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("parakeet_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_strictly(checkpoint: Path) -> transformers.PreTrainedModel:
    model, info = transformers.AutoModelForTDT.from_pretrained(checkpoint, output_loading_info=True)
    problems = {key: value for key, value in info.items() if value}
    if problems:
        raise SystemExit(f"{checkpoint} does not load strictly under transformers {transformers.__version__}: {problems}")
    print(f"[INFO] {checkpoint} loads strictly under transformers {transformers.__version__}.")
    return model.eval()


def check_ane_encoder(model: transformers.PreTrainedModel, exporter: ModuleType, mel_frames: int, frames: int, split: int | None) -> None:
    features = torch.randn(1, mel_frames, model.config.encoder_config.num_mel_bins)
    valid_mel = mel_frames * 2 // 3
    valid = exporter._encoder_frame_count(valid_mel, model.config.encoder_config.subsampling_factor)
    mask = torch.arange(mel_frames)[None] < valid_mel
    with torch.no_grad():
        reference = exporter.ParakeetEncoderModule(model)(features, mask)
        if split is None:
            ane = AneEncoder(model, frames)(features, mask)
        else:
            hidden = AneEncoder(model, frames, 0, split)(features, mask)
            ane = AneEncoder(model, frames, split)(hidden, torch.arange(frames)[None] < valid)
    error = (reference[:, :valid] - ane[:, :valid]).abs().max().item()
    print(f"[INFO] ANE encoder vs reference, FP32, {valid} of {frames} frames valid: max abs error {error:.2e}.")
    if not error < 1e-3:
        raise SystemExit("the ANE encoder does not match the reference encoder")


def orukeet_metadata(graph: str) -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "N. Roll et al. (Oruk AI), fine-tuned from NVIDIA Parakeet-TDT 0.6B v3"
    metadata.license = "CC-BY-SA-4.0"
    metadata.model_description = f"Orukeet ASR ({graph} subgraph), Parakeet TDT architecture. Source: https://huggingface.co/oruk/orukeet"
    metadata.creation_date = int(time.time())
    return metadata


def window_frames(
    exporter: ModuleType, processor: transformers.ProcessorMixin, windows: list[float], subsampling: int
) -> dict[str, tuple[int, int]]:
    """Function name `w<encoder frames>` to (mel frames, encoder frames), shortest first."""
    frames = {}
    for seconds in sorted(windows):
        mel = exporter._audio_features(processor, torch.float32, seconds).shape[1]
        encoder = exporter._encoder_frame_count(mel, subsampling)
        frames[f"w{encoder}"] = (mel, encoder)
    return frames


def add_window_functions(exporter: ModuleType, windows: dict[str, tuple[int, int]], mel_bins: int) -> None:
    """Specialize the dynamic encoder into one static function per window, `main_w<encoder frames>`, over one weight set."""
    shapes = {name: {"input_features": (1, mel, mel_bins), "attention_mask": (1, mel)} for name, (mel, _) in windows.items()}
    save = exporter._save_program

    def save_windows(program, path: Path, graph: str) -> None:
        if graph == exporter.ENCODER_GRAPH:
            program.set_static_shape_config("main", shapes)
        save(program, path, graph)

    exporter._save_program = save_windows
    print(f"[INFO] Encoder functions: {', '.join(f'main_{name} ({mel} mel frames)' for name, (mel, _) in windows.items())}.")


def export_tail(exporter: ModuleType, args: argparse.Namespace, frames: int, windows: dict[str, tuple[int, int]] | None) -> None:
    """Layers from `--split-at` on as a second program, `<bundle>_encoder_tail.aimodel`: hidden states and an encoder-rate mask in."""
    dtype = getattr(torch, args.dtype)
    model = transformers.AutoModelForTDT.from_pretrained(args.checkpoint, dtype=dtype).eval()
    trace = {
        "x": torch.randn(1, frames, model.config.encoder_config.hidden_size, dtype=dtype),
        "attention_mask": torch.ones(1, frames, dtype=dtype),
    }
    module = compress(AneEncoder(model, frames, args.split_at), trace, args.weights)
    dynamic = {name: {1: torch.export.Dim.DYNAMIC} for name in trace} if windows else None
    program = exporter._convert(module, trace, ["hidden_states", "attention_mask"], ["encoder_hidden_states"], dtype, False, dynamic)
    if windows:
        hidden = model.config.encoder_config.hidden_size
        program.set_static_shape_config(
            "main", {name: {"hidden_states": (1, t, hidden), "attention_mask": (1, t)} for name, (_, t) in windows.items()}
        )
    bundle, _ = exporter._bundle_paths(str(args.out), str(args.checkpoint), dtype, bool(windows))
    path = bundle / f"{bundle.name}_encoder_tail.aimodel"
    program.save_asset(path, orukeet_metadata("encoder_tail"))
    metadata = json.loads((bundle / "metadata.json").read_text())
    metadata["assets"]["encoder_tail"] = path.name
    (bundle / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[INFO] Saved encoder layers {args.split_at} on to {path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a local Orukeet checkpoint to a static Core AI Parakeet bundle.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with the oruk/orukeet root files")
    parser.add_argument("--exporter", type=Path, default=Path(".build/checkouts/coreai-models/models/parakeet/export.py"))
    parser.add_argument("--dtype", choices=["float16", "float32"], required=True)
    parser.add_argument("--audio-seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ane-encoder", action="store_true", help="export speech/convert/ane_encoder.py instead of the reference encoder")
    parser.add_argument(
        "--windows", type=float, nargs="+", help="with --ane-encoder, shorter windows in seconds, one encoder function each"
    )
    parser.add_argument(
        "--weights", choices=WEIGHTS, default="fp16", help="with --ane-encoder and --dtype float16, the encoder's weight format"
    )
    parser.add_argument("--split-at", type=int, help="with --ane-encoder, export layers from this one on as a second encoder program")
    args = parser.parse_args()
    model = load_strictly(args.checkpoint)
    exporter = load_exporter(args.exporter)
    if args.ane_encoder:
        processor = transformers.AutoProcessor.from_pretrained(args.checkpoint)
        mel_frames = exporter._audio_features(processor, torch.float32, args.audio_seconds).shape[1]
        frames = exporter._encoder_frame_count(mel_frames, model.config.encoder_config.subsampling_factor)
        check_ane_encoder(model, exporter, mel_frames, frames, args.split_at)
        # ANE inputs are fp16 or fp32 only; a bool mask input keeps the whole encoder off it (S1, 2026-10-28)
        # traced by keyword: the keys follow AneEncoder.forward, the Core AI names stay input_features and attention_mask
        exporter._encoder_inputs = lambda features: {"x": features, "attention_mask": torch.ones(features.shape[:2], dtype=features.dtype)}
        exporter._encoder_dynamic_shapes = lambda: {"x": {1: torch.export.Dim.DYNAMIC}, "attention_mask": {1: torch.export.Dim.DYNAMIC}}
        trace = exporter._encoder_inputs(exporter._audio_features(processor, getattr(torch, args.dtype), args.audio_seconds))
        exporter.ParakeetEncoderModule = lambda model: compress(AneEncoder(model, frames, 0, args.split_at), trace, args.weights)
        windows = window_frames(
            exporter, processor, [*(args.windows or []), args.audio_seconds], model.config.encoder_config.subsampling_factor
        )
        if args.windows:
            add_window_functions(exporter, windows, model.config.encoder_config.num_mel_bins)
    del model
    # Apple's exporter only accepts nvidia/parakeet-tdt-0.6b-v3 on its CLI and stamps NVIDIA's CC-BY-4.0 into every asset.
    exporter._build_aimodel_metadata = orukeet_metadata
    exporter.create_parakeet(
        str(args.out),
        str(args.checkpoint),
        getattr(torch, args.dtype),
        overwrite=False,
        dynamic=bool(args.windows),
        audio_seconds=args.audio_seconds,
        include_debug_info=False,
    )
    if args.ane_encoder and args.split_at is not None:
        export_tail(exporter, args, frames, windows if args.windows else None)


if __name__ == "__main__":
    main()

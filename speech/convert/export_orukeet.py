import argparse
import importlib.util
import time
from pathlib import Path
from types import ModuleType

import torch
import transformers
from coreai.runtime import AIModelAssetMetadata


def load_exporter(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("parakeet_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_strict_load(checkpoint: Path) -> None:
    _, info = transformers.AutoModelForTDT.from_pretrained(checkpoint, output_loading_info=True)
    problems = {key: value for key, value in info.items() if value}
    if problems:
        raise SystemExit(f"{checkpoint} does not load strictly under transformers {transformers.__version__}: {problems}")
    print(f"[INFO] {checkpoint} loads strictly under transformers {transformers.__version__}.")


def orukeet_metadata(graph: str) -> AIModelAssetMetadata:
    metadata = AIModelAssetMetadata()
    metadata.author = "N. Roll et al. (Oruk AI), fine-tuned from NVIDIA Parakeet-TDT 0.6B v3"
    metadata.license = "CC-BY-SA-4.0"
    metadata.model_description = f"Orukeet ASR ({graph} subgraph), Parakeet TDT architecture. Source: https://huggingface.co/oruk/orukeet"
    metadata.creation_date = int(time.time())
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a local Orukeet checkpoint to a static Core AI Parakeet bundle.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="folder with the oruk/orukeet root files")
    parser.add_argument("--exporter", type=Path, default=Path(".build/checkouts/coreai-models/models/parakeet/export.py"))
    parser.add_argument("--dtype", choices=["float16", "float32"], required=True)
    parser.add_argument("--audio-seconds", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    check_strict_load(args.checkpoint)
    exporter = load_exporter(args.exporter)
    # Apple's exporter only accepts nvidia/parakeet-tdt-0.6b-v3 on its CLI and stamps NVIDIA's CC-BY-4.0 into every asset.
    exporter._build_aimodel_metadata = orukeet_metadata
    exporter.create_parakeet(
        str(args.out),
        str(args.checkpoint),
        getattr(torch, args.dtype),
        overwrite=False,
        dynamic=False,
        audio_seconds=args.audio_seconds,
        include_debug_info=False,
    )


if __name__ == "__main__":
    main()

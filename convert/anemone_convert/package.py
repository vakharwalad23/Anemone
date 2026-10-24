import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "config.json", "generation_config.json", "vocab.json", "merges.txt")
EMBEDDING = "model.embed_tokens.weight"


def checkpoint(model_id: str) -> Path:
    return Path(snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors", "*.txt"]))


def link_tree(source: Path, target: Path) -> None:
    # hard links: a model folder next to the converter's outputs costs no extra disk; a rerun replaces the old links
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target, copy_function=os.link)


def write_embeddings(source: Path, path: Path) -> None:
    shard = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"][EMBEDDING]
    with safe_open(source / shard, framework="pt") as handle:
        handle.get_tensor(EMBEDDING).to(torch.float16).numpy().tofile(path)


def end_of_text(source: Path) -> list[int]:
    """End ids from generation_config.json, or config.json for checkpoints without one (DeepSeek-R1-0528-Qwen3-8B)."""
    path = source / "generation_config.json"
    eos = json.loads((path if path.exists() else source / "config.json").read_text())["eos_token_id"]
    return eos if isinstance(eos, list) else [eos]


def manifest(model_id: str, source: Path, chunks: list[str]) -> dict:
    config = json.loads((source / "config.json").read_text())
    return {
        "model": model_id,
        "chunks": chunks,
        "drafter": "drafter.aimodel",
        "embeddings": "embeddings.f16",
        "hidden_size": config["hidden_size"],
        "vocab_size": config["vocab_size"],
        "eos_token_ids": end_of_text(source),
    }


def package(model_id: str, chunk_dir: Path, drafter: Path, out: Path) -> None:
    """Assemble a model folder from exported target chunks, the drafter program and the original checkpoint."""
    source = checkpoint(model_id)
    chunks = sorted(chunk_dir.glob("chunk-*.aimodel"))
    out.mkdir(parents=True, exist_ok=True)
    for chunk in chunks:
        link_tree(chunk, out / chunk.name)
    link_tree(drafter, out / "drafter.aimodel")
    for name in TOKENIZER_FILES:
        if (source / name).exists():
            shutil.copyfile(source / name, out / name)
    write_embeddings(source, out / "embeddings.f16")
    (out / "anemone.json").write_text(json.dumps(manifest(model_id, source, [c.name for c in chunks]), indent=2) + "\n")

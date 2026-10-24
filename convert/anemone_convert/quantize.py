import importlib
import json
import shutil
import sysconfig
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RMSNorm

from anemone_convert.lut import palettize
from anemone_convert.target import head_name, inverse_frequencies, read_tensors

STDLIB = Path(sysconfig.get_paths()["stdlib"])
# prose, reference docs and code that every Python install has; chat answers come from the model itself
DOCUMENTS = (
    "pydoc_data/topics.py",
    "idlelib/News3.txt",
    "asyncio/base_events.py",
    "email/message.py",
    "difflib.py",
    "argparse.py",
)
# machine-learning code from the converter's own environment, so calibration covers the code people ask models about
PACKAGE_MODULES = (
    "transformers.generation.utils",
    "transformers.trainer",
    "transformers.models.qwen3.modeling_qwen3",
    "torch.nn.modules.activation",
    "huggingface_hub.hf_api",
)
# GPTQ's down_proj Hessian is 12288 wide, so a few thousand tokens leave most directions unconstrained
DOCUMENT_TOKENS = 2048
CHAT_PROMPTS = (
    "Write a haiku sequence about autumn in a small mountain town.",
    "How do vaccines train the immune system? Keep it simple.",
    "Write a SQL query that finds the top three customers by total order value, and explain each clause.",
    "What are the pros and cons of renting versus buying a home?",
    "Solve for x: 3x + 7 = 2(x - 4) + 15, showing each step.",
    "Write a short story opening about a lighthouse keeper who finds a message in a bottle.",
    "Explain the difference between TCP and UDP with an everyday analogy.",
    "Give me a three-day itinerary for Kyoto on a modest budget.",
    "Write a JavaScript debounce function and show how to use it with a search box.",
    "Translate into English: Manana iremos al mercado temprano para comprar fruta fresca.",
    "Explain photosynthesis briefly.",
    "Describe the history of the Great Wall in three sentences.",
    "What should I check before buying a used car?",
    "Write a Rust function that counts word frequencies in a string.",
    "Summarize the plot of Romeo and Juliet for a ten-year-old.",
    "Give five tips for writing clear technical documentation.",
)
CHAT_REPLY_TOKENS = 320


def documents(checkpoint: Path) -> dict[str, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    paths = {name: STDLIB / name for name in DOCUMENTS} | {name: Path(importlib.import_module(name).__file__) for name in PACKAGE_MODULES}
    return {name: tokenizer.encode(path.read_text())[:DOCUMENT_TOKENS] for name, path in paths.items()}


def chat_replies(chat_model: str, checkpoint: Path) -> dict[str, list[int]]:
    """The model's own no-think answers to unrelated prompts; GPTQ fits chat only when it calibrates on chat (lab e07)."""
    import mlx.core as mx
    from mlx_lm import load, stream_generate

    model, _ = load(chat_model)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    replies = {}
    for index, prompt in enumerate(CHAT_PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False, enable_thinking=False)
        replies[f"chat{index}"] = ids + [r.token for r in stream_generate(model, tokenizer, ids, max_tokens=CHAT_REPLY_TOKENS)]
    # the MLX copy of the model must leave memory before the bf16 layers stream through
    del model
    mx.clear_cache()
    return replies


def rope_tables(config: Qwen3Config, length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inverse, scale = inverse_frequencies(config)
    angles = torch.arange(length, dtype=torch.float32)[:, None] * inverse
    angles = torch.cat((angles, angles), dim=-1)[None]
    return (angles.cos() * scale).to(device), (angles.sin() * scale).to(device)


def load_weights(module: torch.nn.Module, checkpoint: Path, prefix: str, device: str) -> torch.nn.Module:
    names = {name: prefix + name for name, _ in module.named_parameters()}
    tensors = read_tensors(checkpoint, list(names.values()), torch.float32)
    module.load_state_dict({name: tensors[source] for name, source in names.items()}, assign=True)
    return module.to(device)


def gram(inputs: torch.Tensor) -> torch.Tensor:
    flat = inputs.float().flatten(0, -2)
    return flat.T @ flat


def input_hessians(layer: torch.nn.Module, hidden: dict[str, torch.Tensor], tables: dict) -> dict[str, torch.Tensor]:
    hessians: dict[str, torch.Tensor] = {}

    def accumulate(name: str, inputs: torch.Tensor) -> None:
        hessians[name] = hessians.get(name, 0) + gram(inputs)

    linears = {name: m for name, m in layer.named_modules() if isinstance(m, torch.nn.Linear)}
    hooks = [m.register_forward_pre_hook(lambda _, args, name=name: accumulate(name, args[0])) for name, m in linears.items()]
    for name, h in hidden.items():
        layer(h, position_embeddings=tables[name])
    for hook in hooks:
        hook.remove()
    return {name: h.cpu() for name, h in hessians.items()}


class CheckpointWriter:
    def __init__(self, source: Path, out: Path, bits: int, group_rows: int) -> None:
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(source / "config.json", out / "config.json")
        scheme = {"bits": bits, "group_rows": group_rows, "channel_scale": False, "activation_aware": True, "gptq": True}
        (out / "palettization.json").write_text(json.dumps(scheme))
        self.out, self.weight_map = out, {}

    def add(self, file: str, tensors: dict[str, torch.Tensor]) -> None:
        tensors = {name: t.detach().to(torch.float16).cpu().contiguous() for name, t in tensors.items()}
        save_file(tensors, self.out / file)
        self.weight_map |= dict.fromkeys(tensors, file)

    def close(self) -> None:
        (self.out / "model.safetensors.index.json").write_text(json.dumps({"weight_map": self.weight_map}))


@torch.no_grad()
def quantize(checkpoint: Path, calibration: dict[str, list[int]], out: Path, device: str, bits: int = 4, group_rows: int = 32) -> None:
    """Palettize every linear layer with GPTQ onto per-group lookup tables, one layer at a time, and write an fp16 checkpoint."""
    config = Qwen3Config.from_pretrained(checkpoint)
    config._attn_implementation = "sdpa"
    writer = CheckpointWriter(checkpoint, out, bits, group_rows)
    embed = read_tensors(checkpoint, ["model.embed_tokens.weight"], torch.float32)["model.embed_tokens.weight"]
    hidden = {name: embed[ids][None].to(device) for name, ids in calibration.items()}
    del embed
    tables = {name: rope_tables(config, len(ids), device) for name, ids in calibration.items()}
    # bf16 weights in fp32 one layer at a time: the whole model does not fit next to macOS in 18 GB
    for index in range(config.num_hidden_layers):
        with torch.device("meta"):
            layer = Qwen3DecoderLayer(config, index)
        layer = load_weights(layer, checkpoint, f"model.layers.{index}.", device)
        hessians = input_hessians(layer, hidden, tables)
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = palettize(module.weight.data.cpu(), hessians[name], bits, group_rows).to(device)
        writer.add(f"layer-{index:02d}.safetensors", {f"model.layers.{index}.{n}": t for n, t in layer.named_parameters()})
        hidden = {name: layer(h, position_embeddings=tables[name]) for name, h in hidden.items()}
        print(f"layer {index + 1}/{config.num_hidden_layers}", flush=True)
    with torch.device("meta"):
        norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    norm = load_weights(norm, checkpoint, "model.norm.", device)
    head = read_tensors(checkpoint, [head_name(checkpoint)], torch.float32)[head_name(checkpoint)]
    head = palettize(head, sum(gram(norm(h)[0]) for h in hidden.values()).cpu(), bits, group_rows)
    writer.add("head.safetensors", {"model.norm.weight": norm.weight, "lm_head.weight": head})
    writer.close()

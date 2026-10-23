import json
from pathlib import Path

import torch
from coreai_models.export.compression import quantize_pytorch_model
from coreai_models.export.macos import export_to_coreai
from coreai_models.export.presets import get_preset
from coreai_models.primitives._ops import mutable_cache_update_and_fetch
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

INPUTS = ("taps", "prefix_ids", "position_ids", "offset", "injected", "prefix")
STATES = ("k_cache", "v_cache")
OUTPUTS = ("tokens", "confidence", "verify_input")
TAP_ROWS = 8
VERIFY_WIDTH = 16
# the ANE writes KV only at offsets that are multiples of 4, so up to 3 replayed tokens precede the golden one
MAX_PREFIX = 4
Caches = tuple[torch.Tensor, torch.Tensor]


def write_and_fetch(cache: torch.Tensor, layer: int, offset: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
    _, _, heads, _, head_dim = cache.shape
    begin = torch.cat([torch.tensor([layer, 0, 0], dtype=torch.int32), offset, torch.tensor([0], dtype=torch.int32)])
    stop = offset + update.shape[2]
    end = torch.cat([torch.tensor([layer + 1, 1, heads], dtype=torch.int32), stop, torch.tensor([head_dim], dtype=torch.int32)])
    return mutable_cache_update_and_fetch(x=cache, update=update, begin=begin, end=end, layer_idx=layer, seq_dim=-2, seq_len=None)


class Attention(nn.Module):
    def __init__(self, cfg: dict, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.n_heads, self.n_kv_heads, self.head_dim = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        dim = cfg["hidden_size"]
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.k_norm = RMSNorm(self.head_dim, eps=cfg["rms_norm_eps"])
        self.rope = RoPE(base=float(cfg["rope_parameters"]["rope_theta"]))
        self.sdpa = SDPA(is_causal=False)

    def _heads(self, x: torch.Tensor, n: int) -> torch.Tensor:
        return x.reshape(1, x.shape[1], n, self.head_dim).permute(0, 2, 1, 3)

    def forward(
        self, block: torch.Tensor, features: torch.Tensor, positions: torch.Tensor, caches: Caches, offset: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        width = block.shape[1]
        # injected target features enter the keys raw; only the block went through input_layernorm (DSpark inject)
        context = torch.cat([features, block], dim=1)
        query = self.rope(self.q_norm(self._heads(self.q_proj(block), self.n_heads)), position_ids=positions[:, -width:])
        key = self.rope(self.k_norm(self._heads(self.k_proj(context), self.n_kv_heads)), position_ids=positions)
        value = self._heads(self.v_proj(context), self.n_kv_heads)
        key = write_and_fetch(caches[0], self.layer, offset, key)
        value = write_and_fetch(caches[1], self.layer, offset, value)
        out = self.sdpa(query, key, value, attn_mask=mask)
        return self.o_proj(out.permute(0, 2, 1, 3).reshape(1, width, -1))


class Layer(nn.Module):
    def __init__(self, cfg: dict, layer: int) -> None:
        super().__init__()
        self.self_attn = Attention(cfg, layer)
        self.mlp = MLP(cfg["hidden_size"], cfg["intermediate_size"])
        self.input_layernorm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])

    def forward(
        self, x: torch.Tensor, features: torch.Tensor, positions: torch.Tensor, caches: Caches, offset: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), features, positions, caches, offset, mask)
        return h + self.mlp(self.post_attention_layernorm(h))


class DSparkStep(nn.Module):
    """One drafter call: inject the target taps of the accepted tokens, draft a block, embed the next verify input."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        dim, rank, vocab = cfg["hidden_size"], cfg["markov_rank"], cfg["vocab_size"]
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.fc = nn.Linear(dim * len(cfg["target_layer_ids"]), dim, bias=False)
        self.hidden_norm = RMSNorm(dim, eps=cfg["rms_norm_eps"])
        self.layers = nn.ModuleList(Layer(cfg, i) for i in range(cfg["num_hidden_layers"]))
        self.norm = RMSNorm(dim, eps=cfg["rms_norm_eps"])
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.markov_w1 = nn.Embedding(vocab, rank)
        self.markov_w2 = nn.Linear(rank, vocab, bias=False)
        self.confidence = nn.Linear(dim + rank, 1, bias=True)
        self.register_buffer("mask_ids", torch.full((1, cfg["block_size"] - 1), cfg["mask_token_id"], dtype=torch.int32), persistent=False)

    @classmethod
    def load(cls, directory: Path) -> "DSparkStep":
        cfg = json.loads((directory / "config.json").read_text())
        model = cls(cfg)
        model.load_state_dict(checkpoint_state(load_file(directory / "model.safetensors")), strict=True)
        return model.eval()

    def forward(
        self,
        taps: torch.Tensor,
        prefix_ids: torch.Tensor,
        position_ids: torch.Tensor,
        offset: torch.Tensor,
        injected: torch.Tensor,
        prefix: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        golden = prefix_ids[0][prefix - 1]
        hidden = self.embed_tokens(torch.cat([golden[None], self.mask_ids], dim=1))
        features = self.hidden_norm(self.fc(taps))
        mask = self.attention_mask(offset, injected, k_cache.shape[3])
        for layer in self.layers:
            hidden = layer(hidden, features, position_ids, (k_cache, v_cache), offset, mask)
        logits = self.lm_head(self.norm(hidden))[0]
        tokens, confidence = self.roll(hidden[0], logits, golden)
        return tokens, confidence, self.embed_tokens(verify_ids(prefix_ids[0], prefix, tokens))[None, :, None, :]

    def attention_mask(self, offset: torch.Tensor, injected: torch.Tensor, context: int) -> torch.Tensor:
        block_size = self.cfg["block_size"]
        slot = torch.arange(context, dtype=torch.int32)
        # unused tap rows sit between the accepted taps and the block; the next call overwrites them
        block = (slot >= offset + TAP_ROWS) & (slot < offset + TAP_ROWS + block_size)
        valid = (slot < offset + injected) | block
        mask = torch.where(valid, 0.0, float("-inf")).to(torch.float16)
        return mask[None, None, None, :].expand(1, 1, block_size, context)

    def roll(self, hidden: torch.Tensor, logits: torch.Tensor, previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens, gates = [], []
        for step in range(self.cfg["block_size"]):
            markov = self.markov_w1(previous)
            previous = torch.argmax(logits[step] + self.markov_w2(markov)[0], dim=-1, keepdim=True).to(torch.int32)
            gates.append(self.confidence(torch.cat([hidden[step : step + 1], markov], dim=-1))[0])
            tokens.append(previous)
        return torch.cat(tokens), torch.sigmoid(torch.cat(gates).float())


def verify_ids(prefix_ids: torch.Tensor, prefix: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Replayed tokens and the golden one, then the drafts, then padding: the next ANE verify call's ids."""
    index = torch.arange(VERIFY_WIDTH, dtype=torch.int32)
    from_prefix = prefix_ids[index.clamp(max=MAX_PREFIX - 1)]
    from_draft = tokens[(index - prefix).clamp(0, tokens.shape[0] - 1)]
    return torch.where(index < prefix, from_prefix, torch.where(index < prefix + tokens.shape[0], from_draft, 0))


def checkpoint_state(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    renames = {"markov_head.markov_w1.": "markov_w1.", "markov_head.markov_w2.": "markov_w2.", "confidence_head.proj.": "confidence."}
    state = {}
    for name, tensor in tensors.items():
        for old, new in renames.items():
            name = name.replace(old, new)
        state[name] = tensor.to(torch.float16)
    return state


def reference_inputs(cfg: dict, context: int) -> dict[str, torch.Tensor]:
    kv_shape = (cfg["num_hidden_layers"], 1, cfg["num_key_value_heads"], context, cfg["head_dim"])
    return {
        "taps": torch.randn(1, TAP_ROWS, cfg["hidden_size"] * len(cfg["target_layer_ids"]), dtype=torch.float16),
        "prefix_ids": torch.randint(0, cfg["vocab_size"], (1, MAX_PREFIX), dtype=torch.int32),
        "position_ids": torch.arange(TAP_ROWS + cfg["block_size"], dtype=torch.int32)[None],
        "offset": torch.tensor([0], dtype=torch.int32),
        "injected": torch.tensor([3], dtype=torch.int32),
        "prefix": torch.tensor([1], dtype=torch.int32),
        "k_cache": torch.zeros(kv_shape, dtype=torch.float16),
        "v_cache": torch.zeros(kv_shape, dtype=torch.float16),
    }


def export_drafter(drafter_id: str, context: int, path: Path) -> None:
    """Export the DSpark drafter as a static-shape int4 Core AI program for the GPU, with KV room for `context` tokens."""
    model = DSparkStep.load(Path(snapshot_download(drafter_id, allow_patterns=["*.json", "*.safetensors"]))).to(torch.float16)
    # the cache also holds one window of tap rows and the block past the target's last position
    inputs = reference_inputs(model.cfg, context + TAP_ROWS + model.cfg["block_size"] + 1)
    config = dict(get_preset("4bit")["torch_quantization_config"])
    states = tuple(range(len(INPUTS), len(inputs)))
    model = quantize_pytorch_model(model, tuple(inputs.values()), None, config, cache_seq_len=inputs["k_cache"].shape[3], state_indices=states)
    program = export_to_coreai(model, inputs, None, input_names=INPUTS, output_names=OUTPUTS, state_names=STATES)
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(path, None)

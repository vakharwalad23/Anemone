import json
from pathlib import Path

import torch
from coreai_models.export.compression import quantize_pytorch_model
import coreai_torch
from coreai.authoring import AIProgram
from coreai_models.export.externalize import EXTERNALIZE_SPECS
from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization
from coreai_models.export.presets import get_preset
from coreai_models.primitives._ops import mutable_cache_update_and_fetch, mutable_slice_update
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch import nn

INPUTS = ("taps", "prefix_ids", "position_ids", "offset", "injected", "prefix")
STATES = ("k_cache", "v_cache")
OUTPUTS = ("tokens", "second", "confidence", "verify_input")
INJECT_INPUTS = ("taps", "position_ids", "offset")
TAP_ROWS = 8
# prefill injects the prompt's taps 64 rows per call, the target's widest prefill call
INJECT_ROWS = 64
# attending over a 4096-token cache costs 3 ms per draft even when the chat is short
SMALL_CONTEXT = 1024
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


def write(cache: torch.Tensor, layer: int, offset: torch.Tensor, update: torch.Tensor) -> None:
    _, _, heads, _, head_dim = cache.shape
    begin = torch.cat([torch.tensor([layer, 0, 0], dtype=torch.int32), offset, torch.tensor([0], dtype=torch.int32)])
    stop = offset + update.shape[2]
    end = torch.cat([torch.tensor([layer + 1, 1, heads], dtype=torch.int32), stop, torch.tensor([head_dim], dtype=torch.int32)])
    mutable_slice_update(x=cache, update=update[None], begin=begin, end=end)


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

    def write_context(self, features: torch.Tensor, positions: torch.Tensor, caches: Caches, offset: torch.Tensor) -> None:
        key = self.rope(self.k_norm(self._heads(self.k_proj(features), self.n_kv_heads)), position_ids=positions)
        write(caches[0], self.layer, offset, key)
        write(caches[1], self.layer, offset, self._heads(self.v_proj(features), self.n_kv_heads))

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
        self.inject_mode = False

    @classmethod
    def load(cls, directory: Path) -> "DSparkStep":
        cfg = json.loads((directory / "config.json").read_text())
        model = cls(cfg)
        model.load_state_dict(checkpoint_state(load_file(directory / "model.safetensors")), strict=True)
        return model.eval()

    def forward(
        self,
        taps: torch.Tensor,
        prefix_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        offset: torch.Tensor | None = None,
        injected: torch.Tensor | None = None,
        prefix: torch.Tensor | None = None,
        k_cache: torch.Tensor | None = None,
        v_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if self.inject_mode:
            features = self.hidden_norm(self.fc(taps))
            for layer in self.layers:
                layer.self_attn.write_context(features, position_ids, (k_cache, v_cache), offset)
            return ()
        golden = prefix_ids[0][prefix - 1]
        hidden = self.embed_tokens(torch.cat([golden[None], self.mask_ids], dim=1))
        features = self.hidden_norm(self.fc(taps))
        mask = self.attention_mask(offset, injected, k_cache.shape[3])
        for layer in self.layers:
            hidden = layer(hidden, features, position_ids, (k_cache, v_cache), offset, mask)
        logits = self.lm_head(self.norm(hidden))[0]
        tokens, second, confidence = self.roll(hidden[0], logits, golden)
        return tokens, second, confidence, self.embed_tokens(verify_ids(prefix_ids[0], prefix, tokens))[None, :, None, :]

    def attention_mask(self, offset: torch.Tensor, injected: torch.Tensor, context: int) -> torch.Tensor:
        block_size = self.cfg["block_size"]
        slot = torch.arange(context, dtype=torch.int32)
        # unused tap rows sit between the accepted taps and the block; the next call overwrites them
        block = (slot >= offset + TAP_ROWS) & (slot < offset + TAP_ROWS + block_size)
        valid = (slot < offset + injected) | block
        mask = torch.where(valid, 0.0, float("-inf")).to(torch.float16)
        return mask[None, None, None, :].expand(1, 1, block_size, context)

    def roll(self, hidden: torch.Tensor, logits: torch.Tensor, previous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The drafted chain, the runner-up at each step (side branches of a verify tree) and the chain's confidences."""
        tokens, seconds, gates = [], [], []
        for step in range(self.cfg["block_size"]):
            markov = self.markov_w1(previous)
            scores = logits[step] + self.markov_w2(markov)[0]
            best = torch.argmax(scores, dim=-1, keepdim=True).to(torch.int32)
            masked = torch.where(torch.arange(scores.shape[-1]) == best, float("-inf"), scores)
            seconds.append(torch.argmax(masked, dim=-1, keepdim=True).to(torch.int32))
            gates.append(self.confidence(torch.cat([hidden[step : step + 1], markov], dim=-1))[0])
            tokens.append(best)
            previous = best
        return torch.cat(tokens), torch.cat(seconds), torch.sigmoid(torch.cat(gates).float())


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


def inject_inputs(cfg: dict, context: int) -> dict[str, torch.Tensor]:
    main = reference_inputs(cfg, context)
    return {
        "taps": torch.randn(1, INJECT_ROWS, main["taps"].shape[2], dtype=torch.float16),
        "position_ids": torch.arange(INJECT_ROWS, dtype=torch.int32)[None],
        "offset": main["offset"],
        "k_cache": main["k_cache"],
        "v_cache": main["v_cache"],
    }


def traced(inputs: dict[str, torch.Tensor], inject: bool):
    def export(module: nn.Module, pass_inputs_as_kwargs: bool = True) -> torch.export.ExportedProgram:
        # the converter re-traces during externalization, so the mode is set on every call
        module.inject_mode = inject
        with torch.no_grad():
            program = torch.export.export(module, args=(), kwargs=inputs)
        program = program.run_decompositions(coreai_torch.get_decomp_table())
        remove_functionalization(program)
        return program

    return export


def to_program(model: DSparkStep, slots: list[int]) -> AIProgram:
    """Per cache size, two entry points over one set of weights: `main_<slots>` drafts a block, `inject_<slots>` only writes prompt taps."""
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    entries = [
        entry
        for size in slots
        for entry in (
            (f"main_{size}", reference_inputs(model.cfg, size), False, INPUTS, OUTPUTS),
            (f"inject_{size}", inject_inputs(model.cfg, size), True, INJECT_INPUTS, ()),
        )
    ]
    for name, inputs, inject, input_names, output_names in entries:
        converter.add_pytorch_module(
            model, export_fn=traced(inputs, inject), externalize_modules=EXTERNALIZE_SPECS, input_names=input_names,
            output_names=output_names, state_names=STATES, entrypoint_name=name,
        )
    register_custom_torch_lowering(converter)
    program = converter.to_coreai()
    model.inject_mode = False
    return program


def export_drafter(drafter_id: str, context: int, path: Path) -> None:
    """Export the DSpark drafter as a static-shape int4 Core AI program for the GPU, with KV room for up to `context` tokens."""
    model = DSparkStep.load(Path(snapshot_download(drafter_id, allow_patterns=["*.json", "*.safetensors"]))).to(torch.float16)
    # each cache also holds one inject window of tap rows past the target's last position; short chats use the small one
    sizes = sorted({min(context, SMALL_CONTEXT), context})
    slots = [size + INJECT_ROWS for size in sizes]
    inputs = reference_inputs(model.cfg, slots[-1])
    config = dict(get_preset("4bit")["torch_quantization_config"])
    states = tuple(range(len(INPUTS), len(inputs)))
    model = quantize_pytorch_model(model, tuple(inputs.values()), None, config, cache_seq_len=slots[-1], state_indices=states)
    program = to_program(model, slots)
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(path, None)

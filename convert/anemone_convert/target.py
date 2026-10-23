import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from coreai.authoring import AIProgram
from coreai.authoring.types import AllocationType, HardwareConstraints
from coreai_models._constants import (
    CAUSAL_MASK_INPUT_NAME,
    IN_STEP_INPUT_NAME,
    KEY_CACHE_INPUT_NAME,
    KEY_CACHE_OUTPUT_NAME,
    POSITION_IDS_INPUT_NAME,
    TRANSFORMER_INPUT_NAME,
    VALUE_CACHE_INPUT_NAME,
    VALUE_CACHE_OUTPUT_NAME,
)
from coreai_models.export.compression import palettize_pytorch_model
from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization
from coreai_models.export.presets import get_preset
from coreai_models.models.base import BaseForCausalLMForiOS
from coreai_models.models.ios.qwen3 import TransformerBlock
from coreai_models.primitives.ios.cache import KVCacheHandler
from coreai_models.primitives.ios.rms_norm import RMSNorm
from coreai_models.primitives.ios.rope import RoPECache
from coreai_torch import TorchConverter
from safetensors import safe_open
from torch import nn
from transformers import PretrainedConfig, Qwen3Config

FUNCTION = "extend"
INPUTS = (TRANSFORMER_INPUT_NAME, POSITION_IDS_INPUT_NAME, IN_STEP_INPUT_NAME, CAUSAL_MASK_INPUT_NAME)
STATES = (KEY_CACHE_INPUT_NAME, VALUE_CACHE_INPUT_NAME)
STATE_OUTPUTS = (KEY_CACHE_OUTPUT_NAME, VALUE_CACHE_OUTPUT_NAME)
CONV_WEIGHTS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Apple ships widths 8, 16, 64; 32 fits a confidence-pruned draft tree without paying for 64
WIDTHS = (8, 16, 32, 64)
MIN_CONTEXT = BaseForCausalLMForiOS.IOS_STATIC_MIN_CACHE_LEN
INTERLEAVE = BaseForCausalLMForiOS.KV_CACHE_INTERLEAVE_FACTOR
TRACE_WIDTH = 8
# 9 layers of an 8B (about 0.84 GB at 4 bits) per ANE program; smaller models fit more layers per program
LAYERS_PER_CHUNK = 9


def inverse_frequencies(config: PretrainedConfig) -> tuple[torch.Tensor, float]:
    # DeepInfer YaRN semantics: attn_factor multiplies the mscale, which HF and mlx_lm ignore
    params = config.rope_parameters
    dim, base = config.head_dim, params["rope_theta"]
    positional = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if params.get("rope_type", "default") == "default":
        return 1.0 / positional, 1.0
    factor, original = params["factor"], params["original_max_position_embeddings"]

    def correction(rotations: float) -> float:
        return dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

    low = max(math.floor(correction(params.get("beta_fast", 32))), 0)
    high = min(math.ceil(correction(params.get("beta_slow", 1))), dim - 1)
    ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 0.001)).clamp(0, 1)
    extrapolation = 1 - ramp
    inverse = extrapolation / positional + (1 - extrapolation) / (factor * positional)
    return inverse, (0.1 * math.log(factor) + 1.0) * params.get("attn_factor", 1.0)


class ScaledRoPECache(RoPECache):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config.head_dim, config.max_position_embeddings, config.rope_parameters["rope_theta"])
        self._inverse, self._scale = inverse_frequencies(config)
        self._compute_sin_and_cos()

    def _compute_sin_and_cos(self, dtype: torch.dtype = torch.float32) -> None:
        if not hasattr(self, "_inverse"):
            return super()._compute_sin_and_cos(dtype)
        angles = torch.arange(self._max_cache_size, dtype=torch.float32)[:, None] * self._inverse
        angles = torch.cat((angles, angles), dim=-1)
        self.cos_cached = nn.Buffer((angles.cos() * self._scale).to(dtype), persistent=False)
        self.sin_cached = nn.Buffer((angles.sin() * self._scale).to(dtype), persistent=False)
        return None


def read_tensors(checkpoint: Path, names: list[str], dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
    shard_of = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard = defaultdict(list)
    for name in names:
        by_shard[shard_of[name]].append(name)
    tensors = {}
    for shard, shard_names in by_shard.items():
        with safe_open(checkpoint / shard, framework="pt") as handle:
            for name in shard_names:
                tensors[name] = handle.get_tensor(name).to(dtype)
    return tensors


class TargetChunk(nn.Module):
    """Layers `start` to `end` of the target, emitting the drafter taps; the last chunk ends in the head and argmax."""

    def __init__(self, config: Qwen3Config, start: int, end: int, taps: list[int]) -> None:
        super().__init__()
        self.config = config
        self.start, self.end = start, end
        self.is_last = end == config.num_hidden_layers
        self.tap_slots = [layer - start for layer in taps if start <= layer < end]
        self.layers = nn.ModuleList(TransformerBlock(config, slot) for slot in range(end - start))
        self.kv_cache = KVCacheHandler(end - start, config.hidden_size)
        self.rope = ScaledRoPECache(config)
        if self.is_last:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @property
    def output_names(self) -> tuple[str, ...]:
        return ("out_tokens" if self.is_last else "hidden",) + (("taps",) if self.tap_slots else ())

    def forward(
        self,
        transformer_input: torch.Tensor,
        position_ids: torch.Tensor,
        in_step: torch.Tensor,
        causal_mask: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        self.kv_cache.register_kv_cache(key_cache, value_cache)
        rope_cos, rope_sin = self.rope.gather_cos_sin(position_ids)
        h, taps = transformer_input, []
        for slot, layer in enumerate(self.layers):
            # DSpark taps the residual entering a layer (DeepInfer dspark_target_attn_input)
            if slot in self.tap_slots:
                taps.append(h)
            h = layer(h, rope_cos, rope_sin, in_step, causal_mask, self.kv_cache)
        if self.is_last:
            h = torch.argmax(self.lm_head(self.norm(h).transpose(-2, -3)), dim=-1).to(torch.int32)
        return (h, torch.cat(taps, dim=-1)) if taps else (h,)

    def load_checkpoint(self, checkpoint: Path) -> None:
        names = {
            f"layers.{slot}.{name}": f"model.layers.{self.start + slot}.{name}"
            for slot in range(self.end - self.start)
            for name, _ in self.layers[slot].named_parameters()
        }
        if self.is_last:
            names |= {"norm.weight": "model.norm.weight", "lm_head.weight": head_name(checkpoint)}
        tensors = read_tensors(checkpoint, list(names.values()))
        state = {key: tensors[source] for key, source in names.items()}
        for key, value in state.items():
            if key.split(".")[-2] in CONV_WEIGHTS:
                state[key] = value[..., None, None]
        self.load_state_dict(state, strict=False, assign=True)

    def reference_inputs(self) -> dict[str, torch.Tensor]:
        context = self.config.max_position_embeddings
        kv_width = self.config.num_key_value_heads * self.config.head_dim
        cache = torch.zeros(self.end - self.start, 1, kv_width, 1, context, dtype=torch.float16)
        return {
            TRANSFORMER_INPUT_NAME: torch.randn(1, TRACE_WIDTH, 1, self.config.hidden_size, dtype=torch.float16),
            POSITION_IDS_INPUT_NAME: torch.arange(TRACE_WIDTH).to(torch.uint16).unsqueeze(0),
            IN_STEP_INPUT_NAME: torch.zeros((1,), dtype=torch.int32),
            CAUSAL_MASK_INPUT_NAME: torch.zeros(1, context, 1, TRACE_WIDTH, dtype=torch.float16),
            KEY_CACHE_INPUT_NAME: cache,
            VALUE_CACHE_INPUT_NAME: cache.clone(),
        }

    def static_shapes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        kv_width = self.config.num_key_value_heads * self.config.head_dim
        shapes, context = {}, MIN_CONTEXT
        while context <= self.config.max_position_embeddings:
            for width in WIDTHS:
                cache = (self.end - self.start, 1, kv_width, 1, context)
                shapes[f'"{context}_{width}"'] = {
                    TRANSFORMER_INPUT_NAME: (1, width, 1, self.config.hidden_size),
                    POSITION_IDS_INPUT_NAME: (1, width),
                    CAUSAL_MASK_INPUT_NAME: (1, context, 1, width),
                    KEY_CACHE_INPUT_NAME: cache,
                    VALUE_CACHE_INPUT_NAME: cache,
                }
            context *= 2
        return shapes

    def hardware_constraints(self) -> dict[str, HardwareConstraints]:
        # every context rung shares the largest rung's strides, so one KV buffer serves them all
        context = self.config.max_position_embeddings
        cache = HardwareConstraints(
            AllocationType.IOSurface, interleave=[1, 1, INTERLEAVE, 1, 1], alignments=[1, 1, 1, 1, INTERLEAVE * context, 1]
        )
        return dict.fromkeys(STATES + STATE_OUTPUTS, cache)


def palettization_config(scheme: dict) -> dict:
    preset = get_preset("4bit_weight_palettized_group32")["torch_palettization_config"]
    granularity = {"type": "per_grouped_channel", "axis": 0, "group_size": scheme["group_rows"]}
    weight = {"n_bits": scheme["bits"], "granularity": granularity, "enable_per_channel_scale": scheme["channel_scale"]}
    # weights already hold at most 2**bits values per group, so exact clustering just recovers them
    global_config = {"op_state_spec": {"weight": weight}, "enable_fast_kmeans_mode": False}
    return {"global_config": global_config, "module_type_configs": preset["module_type_configs"]}


def to_program(chunk: TargetChunk) -> AIProgram:
    context = chunk.config.max_position_embeddings
    seq_len, cache_len = torch.export.Dim("seq_len", max=context), torch.export.Dim("cache_len", max=context)
    dynamic = {
        TRANSFORMER_INPUT_NAME: {1: seq_len},
        POSITION_IDS_INPUT_NAME: {1: seq_len},
        IN_STEP_INPUT_NAME: None,
        CAUSAL_MASK_INPUT_NAME: {1: cache_len, 3: seq_len},
        KEY_CACHE_INPUT_NAME: {4: cache_len},
        VALUE_CACHE_INPUT_NAME: {4: cache_len},
    }
    decompositions = torch.export.default_decompositions()
    decompositions.pop(torch.ops.aten.silu.default)
    decompositions.pop(torch.ops.aten.silu.out)
    with torch.no_grad():
        exported = torch.export.export(chunk, args=(), kwargs=chunk.reference_inputs(), dynamic_shapes=dynamic)
    exported = exported.run_decompositions(decompositions)
    remove_functionalization(exported)
    converter = TorchConverter(mode=TorchConverter.Mode.RELEASE)
    register_custom_torch_lowering(converter)
    converter.add_exported_program(
        exported, input_names=list(INPUTS), state_names=list(STATES), output_names=list(chunk.output_names), entrypoint_name=FUNCTION
    )
    program = converter.to_coreai()
    program.set_hardware_constraints(FUNCTION, chunk.hardware_constraints())
    program.set_static_shape_config(FUNCTION, chunk.static_shapes())
    return program


def head_name(checkpoint: Path) -> str:
    """The output layer's tensor: models with tied embeddings reuse the embedding table."""
    shard_of = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    return "lm_head.weight" if "lm_head.weight" in shard_of else "model.embed_tokens.weight"


def export_target(checkpoint: Path, taps: list[int], context: int, out: Path, layers_per_chunk: int = LAYERS_PER_CHUNK) -> list[Path]:
    """Export a palettized checkpoint as 9-layer ANE chunks with context rungs up to `context`."""
    config = Qwen3Config.from_pretrained(checkpoint)
    config.max_position_embeddings = context
    scheme = json.loads((checkpoint / "palettization.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(0, config.num_hidden_layers, layers_per_chunk):
        end = min(start + layers_per_chunk, config.num_hidden_layers)
        chunk = TargetChunk(config, start, end, taps)
        chunk.load_checkpoint(checkpoint)
        chunk = chunk.to(torch.float16).eval()
        chunk = palettize_pytorch_model(chunk, tuple(chunk.reference_inputs().values()), palettization_config(scheme))
        path = out / f"chunk-{start:02d}-{end:02d}.aimodel"
        to_program(chunk).save_asset(path, None)
        paths.append(path)
        print(f"wrote {path}")
    return paths

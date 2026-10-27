import torch
from coreai_opt.base_model_compressor import ExportBackend
from coreai_opt.palettization.config.palettization_config import KMeansPalettizerConfig
from coreai_opt.palettization.kmeans import KMeansPalettizer
from coreai_opt.quantization import ExecutionMode, Quantizer, QuantizerConfig

WEIGHTS = ("fp16", "int8", "lut6", "lut4")
# coreai_opt forks a k-means worker per core by default; a 32-worker fork once panicked this Mac (Anemone lab)
PALETTIZE_WORKERS = 4


def compress(module: torch.nn.Module, inputs: dict[str, torch.Tensor], weights: str) -> torch.nn.Module:
    """int8 per-channel weights, or 6-bit or 4-bit k-means lookup tables per 16 output rows; fp16 leaves the module as it is."""
    example = tuple(inputs.values())
    if weights == "fp16":
        return module
    if weights == "int8":
        quantizer = Quantizer(module, QuantizerConfig.presets.w8(execution_mode=ExecutionMode.EAGER))
        return quantizer.finalize(quantizer.prepare(example), backend=ExportBackend.CoreAI)
    config = {"lut6": KMeansPalettizerConfig.presets.w6, "lut4": KMeansPalettizerConfig.presets.w4}[weights]()
    palettizer = KMeansPalettizer(module, config)
    return palettizer.finalize(palettizer.prepare(example_inputs=example, num_workers=PALETTIZE_WORKERS), backend=ExportBackend.CoreAI)

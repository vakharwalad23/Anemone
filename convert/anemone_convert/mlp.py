import torch
from coreai_models.primitives.ios.mlp import MLP


class WidthMLP(MLP):
    """Apple's iOS MLP with the tokens on the conv's width axis, as its attention has them, instead of the batch axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-3, -1)
        return self.down_proj(self.up_proj(x) * torch.nn.functional.silu(self.gate_proj(x))).transpose(-3, -1)

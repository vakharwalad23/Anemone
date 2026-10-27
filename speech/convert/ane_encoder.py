import torch
import transformers


class AneEncoder(torch.nn.Module):
    """Layers `start` to `end` of the Parakeet encoder, with the subsampling before layer 0 and the projector after the last layer.

    Every mask is in the model dtype and every linear a 1x1 Conv2d, so no bool, int or fp32 op keeps it off the ANE.
    """

    def __init__(self, model: transformers.PreTrainedModel, frames: int, start: int = 0, end: int | None = None):
        super().__init__()
        encoder = model.encoder
        end = len(encoder.layers) if end is None else end
        self.subsampling = encoder.subsampling if start == 0 else None
        self.input_scale = encoder.input_scale
        self.blocks = torch.nn.ModuleList(encoder.layers[start:end])
        # coreai-torch finds LUT tensors by the substring "lut" in their names, which "convolutions" contains
        self.conv_modules = torch.nn.ModuleList(AneConvolution(block.conv) for block in self.blocks)
        self.projector = model.encoder_projector if end == len(encoder.layers) else None
        for conv in encoder.subsampling.layers:
            if isinstance(conv, torch.nn.Conv2d) and conv.stride[0] != 1:
                assert conv.kernel_size[0] == 3 and conv.padding[0] == 1 and conv.stride[0] == 2, conv
        hidden = torch.zeros(1, frames, model.config.encoder_config.hidden_size, dtype=model.dtype)
        self.register_buffer("positions", encoder.encode_positions(hidden).to(model.dtype))
        # with int8 or LUT weights the ANE compiler fails on 3 or more layers of nn.Linear, not of 1x1 Conv2d (S2, 2026-10-28)
        linears_to_conv(self)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mel features and a mel-rate mask before layer 0, else hidden states and an encoder-rate mask."""
        mask = attention_mask.to(x.dtype)
        if self.subsampling is not None:
            x, mask = self.subsample(x, mask)
        else:
            # with int8 or LUT weights the ANE compiler rejects a program that starts or ends on a LayerNorm (S2, 2026-10-28)
            x = x * mask[..., None]
        # relative positions run from T - 1 down to 1 - T, the middle of the longest window's table
        longest, frames = (self.positions.shape[1] + 1) // 2, x.shape[1]
        positions = self.positions[:, longest - frames : longest + frames - 1]
        # NeMo's finite -1e4 instead of -inf: padding query rows stay finite in fp16
        bias = (mask[:, None, :, None] * mask[:, None, None, :] - 1) * 10000.0
        for block, convolution in zip(self.blocks, self.conv_modules):
            x = conformer_block(block, convolution, x, positions, bias, mask)
        return x * mask[..., None] if self.projector is None else self.projector(x)

    def subsample(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = features.unsqueeze(1)
        for layer in self.subsampling.layers:
            x = layer(x)
            if isinstance(layer, torch.nn.Conv2d):
                # a prefix mask after a kernel-3, padding-1, stride-2 conv is the input mask at even frames
                mask = mask[:, :: layer.stride[0]]
                x = x * mask[:, None, :, None]
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        return self.subsampling.linear(x) * self.input_scale, mask


def conformer_block(
    block: torch.nn.Module, convolution: torch.nn.Module, x: torch.Tensor, positions: torch.Tensor, bias: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    x = x + 0.5 * block.feed_forward1(block.norm_feed_forward1(x))
    x = x + relative_attention(block.self_attn, block.norm_self_att(x), positions, bias)
    x = x + convolution(block.norm_conv(x), mask)
    x = x + 0.5 * block.feed_forward2(block.norm_feed_forward2(x))
    return block.norm_out(x)


def relative_attention(attn: torch.nn.Module, x: torch.Tensor, positions: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    batch, frames, _ = x.shape
    heads, head_dim = attn.config.num_attention_heads, attn.head_dim
    q = attn.q_proj(x).view(batch, frames, heads, head_dim).transpose(1, 2)
    k = attn.k_proj(x).view(batch, frames, heads, head_dim).transpose(1, 2)
    v = attn.v_proj(x).view(batch, frames, heads, head_dim).transpose(1, 2)
    r = attn.relative_k_proj(positions).view(batch, -1, heads, head_dim)
    content = (q + attn.bias_u.view(1, heads, 1, head_dim)) @ k.transpose(2, 3)
    position = attn._rel_shift((q + attn.bias_v.view(1, heads, 1, head_dim)) @ r.permute(0, 2, 3, 1))[..., :frames]
    weights = torch.softmax((content + position) * attn.scaling + bias, dim=-1)
    return attn.o_proj((weights @ v).transpose(1, 2).reshape(batch, frames, -1))


class AneConvolution(torch.nn.Module):
    """The conformer conv module as 4D [1, C, 1, T] Conv2d with BatchNorm folded in: the ANE compiler places no Conv1d module (S1, 2026-10-28)."""

    def __init__(self, conv: torch.nn.Module):
        super().__init__()
        self.pointwise1 = as_conv2d(conv.pointwise_conv1.weight, conv.pointwise_conv1.bias)
        bn = conv.norm
        scale = bn.weight.float() / torch.sqrt(bn.running_var.float() + bn.eps)
        depthwise = conv.depthwise_conv
        shift = depthwise.bias.float() if depthwise.bias is not None else torch.zeros_like(scale)
        self.depthwise = as_conv2d(
            depthwise.weight.float() * scale[:, None, None],
            (shift - bn.running_mean.float()) * scale + bn.bias.float(),
            groups=depthwise.groups,
        ).to(depthwise.weight.dtype)
        self.depthwise.padding = (0, depthwise.padding[0])
        self.activation = conv.activation
        self.pointwise2 = as_conv2d(conv.pointwise_conv2.weight, conv.pointwise_conv2.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.pointwise1(x.transpose(1, 2).unsqueeze(2))
        values, gates = x.chunk(2, dim=1)
        x = values * torch.sigmoid(gates) * mask[:, None, None, :]
        x = self.pointwise2(self.activation(self.depthwise(x)))
        return x.squeeze(2).transpose(1, 2)


def as_conv2d(weight: torch.Tensor, bias: torch.Tensor | None, groups: int = 1) -> torch.nn.Conv2d:
    out_channels, in_channels, kernel = weight.shape
    conv = torch.nn.Conv2d(in_channels * groups, out_channels, (1, kernel), groups=groups, bias=bias is not None)
    conv.weight.data = weight.detach().unsqueeze(2).clone()
    if bias is not None:
        conv.bias.data = bias.detach().clone()
    return conv.to(weight.dtype)


class ConvLinear(torch.nn.Module):
    """A Linear over the last axis computed as a 1x1 Conv2d on the channels-first [B, C, 1, T] layout."""

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        self.conv = as_conv2d(linear.weight[:, :, None], linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(-1, -2).unsqueeze(-2)).squeeze(-2).transpose(-1, -2)


def linears_to_conv(module: torch.nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, ConvLinear(child))
        else:
            linears_to_conv(child)

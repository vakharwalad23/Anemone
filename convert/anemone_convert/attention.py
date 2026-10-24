import torch
from coreai_models.primitives.ios.sdpa import SDPA


class RowSDPA(SDPA):
    """Apple's iOS attention with one last-axis softmax per KV group; the scores are never transposed onto the channel axis."""

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        queries = [q.squeeze(2).transpose(1, 2) for q in query.split(self.head_dim, dim=1)]
        keys = [(k * self._scale_factor).squeeze(2) for k in key.split(self.head_dim, dim=1)]
        values = [v.squeeze(2).transpose(1, 2) for v in value.split(self.head_dim, dim=1)]
        mask = causal_mask.squeeze(2).transpose(1, 2)
        group = len(queries) // len(keys)
        outputs = []
        for kv, (k, v) in enumerate(zip(keys, values, strict=True)):
            rows = torch.cat(queries[kv * group : (kv + 1) * group], dim=1)
            weights = (rows @ k + torch.cat([mask] * group, dim=1)).softmax(-1)
            outputs += [o.transpose(1, 2).unsqueeze(2) for o in torch.tensor_split(weights @ v, group, dim=1)]
        return torch.cat(outputs, dim=1)

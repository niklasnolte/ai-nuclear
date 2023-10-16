from typing import Callable, Optional
from torch import nn
import torch


class PeriodicEmbedding(nn.Embedding):
    def __init__(self, d_model, use_sigmoid=False):
        super().__init__(1, d_model)
        self.use_sigmoid = use_sigmoid

    def forward(self, x):
        if self.use_sigmoid:
            freq = self.weight.sigmoid()
        else:
            freq = self.weight
        sin = torch.sin(x.unsqueeze(-1) * freq[:, ::2])
        cos = torch.cos(x.unsqueeze(-1) * freq[:, 1::2])
        return torch.cat([sin, cos], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.ReLU(),
        norm: Optional[Callable] = None,
    ):
        norm = norm or (lambda x: x)
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.ff = nn.Sequential(
            norm(nn.Linear(d_model, d_model)),
            activation,
            norm(nn.Linear(d_model, d_model)),
            activation,
        )
        # self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # self.norm = nn.BatchNorm1d(d_model, affine=False)
        self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.norm(x + self.dropout(self.ff(x)))

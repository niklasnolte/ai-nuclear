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

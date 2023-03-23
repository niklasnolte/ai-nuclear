# %%
import torch
import torch.nn as nn
from typing import Iterable
from model import Base
import mup


# encoder only transformer implementation
# Start with two embeddings for x and y
# Two input streams: x and y
# First, the inputs pass through an Expander to create
# A number of feature maps which will be attended over
# Both self-attention across feature maps and cross-attention across x and y
# are used to create a new set of feature maps
# The readout layer is made of multiple MLPs attends to the feature maps
# and passes the result through an MLP to produce the output of the desired dimension
# This is done for all tasks. i.e. n_tasks mlps are used to produce n_tasks outputs
# each of dim output_dim[i]

class DefaultTransformer(Base):
  def __init__(self, vocab_size, hidden_dim, output_dim, nstacks=2, nheads=4, mlp_dim=None, dropout=.1):
    super().__init__(vocab_size, hidden_dim)
    self.nstacks = nstacks
    self.nheads = nheads
    self.mlp_dim = mlp_dim or hidden_dim
    self.dropout = dropout
    self.sequence_length = len(self.vocab_size)

    self.transformer = nn.TransformerEncoder(
      nn.TransformerEncoderLayer(
        self.hidden_dim,
        nhead=nheads,
        dim_feedforward=self.mlp_dim,
        dropout=dropout,
        batch_first=True,
        activation=nn.ReLU(),
      ),
      num_layers=nstacks,
    )

    self.readout = nn.Linear(self.hidden_dim * self.sequence_length, output_dim)

    print(sum(p.numel() for p in self.parameters() if p.requires_grad))

  def forward_with_embeddings(self, x, embs): # embs: [ batch_size, 2 * hidden_dim ]
    x = self.embed_input(x, embs)
    x = x.view(x.shape[0], self.sequence_length, self.hidden_dim)
    x = self.transformer(x) # [ batch_size, sequence_length, hidden_dim ]
    x = x.flatten(1)
    return self.readout(x) # [ batch_size, output_dim ]


class FilteredAttentionTransformer(Base):
    def __init__(
        self,
        vocab_size,
        hidden_dim,
        output_dim,
        nfilters=4,
        nstacks=1,
        nheads=2,
        mlp_dim=None,
        dropout=0.3,
    ):
        super().__init__(vocab_size, hidden_dim)
        self.nstacks = nstacks
        self.nheads = nheads
        self.mlp_dim = mlp_dim or hidden_dim
        self.dropout = dropout
        self.sequence_length = len(self.vocab_size)
        self.nfilters = nfilters

        self.expander = Expander(
            self.hidden_dim, self.mlp_dim, nfilters, dropout=dropout
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    self.hidden_dim,
                    nhead=nheads,
                    dim_feedforward=self.mlp_dim,
                    dropout=dropout,
                    batch_first=True,
                    activation=nn.SiLU(),
                )
                for _ in range(nstacks)
            ]
        )
        self.readout = mup.MuReadout(self.hidden_dim * nfilters * 2, sum(output_dim))

        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward_with_embeddings(self, x, embs):
        x = self.embed_input(x, embs)
        x = x.view(x.shape[0], self.sequence_length, self.hidden_dim)
        x = self.expander(x)  # [batch_size, sequence_length * nfilters, d_model]
        for block in self.blocks:
            x = block(x) + x
        return self.readout(x.flatten(1))


class Expander(nn.Module):
    def __init__(self, d_model, mlp_dim, nfilters, depth=1, dropout=0):
        super().__init__()
        self.filters = nn.ModuleList([])
        for _ in range(nfilters):
            in_dim = d_model
            layers = nn.ModuleList([])
            for _ in range(depth - 1):
                layers.extend(
                    [nn.Linear(in_dim, mlp_dim), nn.SiLU(), nn.Dropout(dropout)]
                )
                in_dim = mlp_dim
            layers.append(nn.Linear(in_dim, d_model))
            self.filters.append(nn.Sequential(*layers))

    def forward(self, x):  # x: [batch_size, sequence_length, d_model]
        # output: [batch_size, sequence_length * nfilters, d_model]
        return torch.cat([filter(x) for filter in self.filters], dim=1)


class Attention(
    nn.Module
):  # across heads and the sequence (i.e. across all features for both proton and neutron)
    def __init__(self, d_model, dropout=0):
        super().__init__()
        self.scale = d_model**-0.5
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)

    def forward(self, x):  # x: [batch_size, sequence_length, nheads, d_model]
        b, n, h, d = x.shape
        qkv: torch.Tensor = self.to_qkv(x).reshape(b, n, h, d, 3)
        qkv = qkv.flatten(1, 2)  # to attend across both heads and sequence
        # Combining across heads might force the different heads to be in the same space
        q, k, v = qkv.unbind(dim=-1)
        dots = (
            torch.einsum("bid,bjd->bij", q, k) * self.scale
        )  # [batch_size, sequence_length * nheads, sequence_length]
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum(
            "bij,bjd->bid", attn, v
        )  # [batch_size, sequence_length * nheads, d_model]
        return out.unflatten(
            1, (n, h)
        )  # [batch_size, sequence_length, nheads, d_model]


class AttentionBlock(nn.Module):
    def __init__(self, d_model, nheads, mlp_dim=256, dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, nheads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):  # x: [batch_size, sequence_length, nheads, d_model]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

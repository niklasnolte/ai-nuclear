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
  def __init__(self, vocab_size, non_embedded_input_dim, hidden_dim, output_dim, nstacks=2, nheads=4, mlp_dim=None, dropout=.1):
    super().__init__(vocab_size, non_embedded_input_dim, hidden_dim)
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

    self.readout = nn.Linear(self.input_dim, output_dim)

    print(sum(p.numel() for p in self.parameters() if p.requires_grad))

  def forward_with_embeddings(self, x, embs): # embs: [ batch_size, 2 * hidden_dim ]
    x = self.embed_input(x, embs)
    x = x.view(x.shape[0], self.sequence_length, self.hidden_dim)
    x = self.transformer(x) # [ batch_size, sequence_length, hidden_dim ]
    x = x.flatten(1)
    return self.readout(x) # [ batch_size, output_dim ]

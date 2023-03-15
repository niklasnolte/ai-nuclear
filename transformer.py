# %%
import torch
import torch.nn as nn
import tqdm 
from matplotlib import pyplot as plt
from typing import Iterable, Union
from mup import MuReadout

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



class FilteredAttentionTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, output_dim, stacks=2, heads=5, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.stacks = stacks
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.sequence_length = len(vocab_size)
        self.hidden_dim = hidden_dim

        self.emb = nn.ModuleList([nn.Embedding(vocab_size, self.d_model) for vocab_size in vocab_size])
        self.expander = Expander(self.d_model, mlp_dim, heads, dropout=dropout)
        self.blocks = nn.ModuleList([AttentionBlock(self.d_model, heads, mlp_dim, dropout) for _ in range(stacks)])
        self.readout = Readout(self.d_model, output_dim, sequence_length=self.sequence_length, dropout=dropout, heads=heads)
    
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        x = self.forward_with_embeddings(x, self.emb)
        return x
    
    def forward_with_embeddings(self, x, embs):
        x = self.embed_input(x, embs)
        x = self.expander(x)
        for block in self.blocks:
            x = block(x)
        return self.readout(x) 
    
    def embed_input(self, x, embs):
        if len(embs) == 1:
            embs = [embs[0](x[:, i]) for i in range(x.shape[1])]
        else:
            embs = [embs[i](x[:, i]) for i in range(len(embs))]
        return torch.stack(embs, dim=1)



class Expander(nn.Module):
    def __init__(self, d_model, mlp_dim, heads=5, depth=2, dropout=0):
        super().__init__()
        self.heads = nn.ModuleList([])
        for _ in range(heads):
            in_dim = d_model
            layers = nn.ModuleList([])
            for _ in range(depth -1):
                layers.extend([nn.Linear(in_dim, mlp_dim), nn.SiLU(), nn.Dropout(dropout)])
                in_dim = mlp_dim
            layers.append(nn.Linear(in_dim, d_model))
            self.heads.append(nn.Sequential(*layers))
        
    def forward(self, x): # x: [batch_size, sequence_length, d_model]
        # output: [batch_size, sequence_length, heads, d_model
        return torch.stack([head(x) for head in self.heads], dim=2)
    
class Attention(nn.Module): # across heads and the sequence (i.e. across all features for both proton and neutron)
    def __init__(self, d_model, heads=5, dropout=0):
        super().__init__()
        self.heads = heads
        self.scale = d_model ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(d_model, d_model * 3, bias=False)

    def forward(self, x): # x: [batch_size, sequence_length, heads, d_model]
        b, n, h, d = x.shape
        qkv:torch.Tensor = self.to_qkv(x).reshape(b, n, h, d, 3)
        qkv = qkv.flatten(1,2) # to attend across both heads and sequence
        # Combining across heads might force the different heads to be in the same space
        q, k, v = qkv.unbind(dim=-1)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale  # [batch_size, sequence_length * heads, sequence_length]
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bij,bjd->bid', attn, v) # [batch_size, sequence_length * heads, d_model]
        return out.unflatten(1, (n, h)) # [batch_size, sequence_length, heads, d_model]
    

class AttentionBlock(nn.Module):
    def __init__(self, d_model, heads=5, mlp_dim=256, dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x): # x: [batch_size, sequence_length, heads, d_model]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x)) 
        return x
    

class Readout(nn.Module):
    def __init__(self, d_model, output_dims: Iterable, sequence_length=2, heads=5, mlp_dim=256, mlp_depth=1, dropout=0):
        super().__init__()
        self.attn = Attention(d_model, heads, dropout)
        self.readouts = nn.ModuleList([])
        for output_dim in output_dims:
            in_dim = d_model * sequence_length * heads
            layers = nn.ModuleList([])
            for _ in range(mlp_depth -1):
                layers.extend([nn.Linear(in_dim, mlp_dim), nn.SiLU(), nn.Dropout(dropout)])
                in_dim = mlp_dim
            layers.append(nn.Linear(in_dim, output_dim))
            # layers.append(MuReadout(in_dim, output_dim))
            self.readouts.append(nn.Sequential(*layers))

    def forward(self, x): # x: [batch_size, sequence_length, heads, d_model]
        x = self.attn(x).flatten(1) # [batch_size, sequence_length, heads, d_model]
        return torch.cat([readout(x) for readout in self.readouts], dim=-1) # [batch_size, sequence_length, sum(output_dims)]


        
# %%
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
    def __init__(self, vocab_size, hidden_dim, output_dim, filters=5, stacks=2, heads=4, mlp_dim=128, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.stacks = stacks
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        if not isinstance(vocab_size, Iterable):
            vocab_size = [vocab_size]
        self.sequence_length = len(vocab_size)
        self.hidden_dim = hidden_dim
        self.filters = filters

        self.emb = nn.ModuleList([nn.Embedding(vocab_size, self.d_model) for vocab_size in vocab_size])
        self.expander = Expander(self.d_model, mlp_dim, filters, dropout=dropout)
        # self.blocks = nn.ModuleList([AttentionBlock(self.d_model, heads, mlp_dim, dropout) for _ in range(stacks)])
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(self.d_model, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, batch_first=True) for _ in range(stacks)])
        # self.blocks = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(stacks)])
        self.readout = nn.Linear(self.d_model * filters * 2, sum(output_dim))
        # self.readout = Readout(self.d_model, output_dim, dropout=dropout)
    
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        x = self.forward_with_embeddings(x, self.emb)
        return x
    
    def forward_with_embeddings(self, x, embs):
        x = self.embed_input(x, embs)
        x = self.expander(x) # [batch_size, sequence_length * filters, d_model]
        for block in self.blocks:
            x = block(x)
        # return self.readout(x) 
        return self.readout(x.flatten(1))
    
    def embed_input(self, x, embs):
        if len(embs) == 1:
            embs = [embs[0](x[:, i]) for i in range(x.shape[1])]
        else:
            embs = [embs[i](x[:, i]) for i in range(len(embs))]
        return torch.stack(embs, dim=1)



class Expander(nn.Module):
    def __init__(self, d_model, mlp_dim, filters, depth=2, dropout=0):
        super().__init__()
        self.filters = nn.ModuleList([])
        for _ in range(filters):
            in_dim = d_model
            layers = nn.ModuleList([])
            for _ in range(depth -1):
                layers.extend([nn.Linear(in_dim, mlp_dim), nn.SiLU(), nn.Dropout(dropout)])
                in_dim = mlp_dim
            layers.append(nn.Linear(in_dim, d_model))
            self.filters.append(nn.Sequential(*layers))
        
    def forward(self, x): # x: [batch_size, sequence_length, d_model]
        # output: [batch_size, sequence_length * filters, d_model
        return torch.cat([filter(x) for filter in self.filters], dim=1)
    

    
class Attention(nn.Module): # across heads and the sequence (i.e. across all features for both proton and neutron)
    def __init__(self, d_model, dropout=0):
        super().__init__()
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
    def __init__(self, d_model, heads, mlp_dim=256, dropout=0):
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
    def __init__(self, d_model, output_dims: Iterable, dropout=0):
        super().__init__()
        self.output_dims = output_dims
        self.O = nn.Parameter(torch.randn(len(output_dims), d_model))
        self.to_v = nn.Linear(d_model, len(output_dims) * d_model, bias=False)
        self.mlps = nn.ModuleList()
        for output_dim in output_dims:
            self.mlps.append(nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Dropout(dropout),
                nn.SiLU(),
                nn.Linear(d_model, output_dim)
            ))

    
    def forward(self, x): # x: [batch_size, sequence_length * filters, d_model]
       a = torch.einsum('od,bfd->bof', self.O, x) # [batch_size, len(output_dim), filters]
       a = torch.softmax(a, dim=-1) # [batch_size, len(output_dim), filters]
       v = self.to_v(x).view(*x.shape, len(self.output_dims)) # [batch_size, sequence_length * filters, d_model, len(output_dim)]
       res = []
       for i, v in enumerate(v.unbind(dim=-1)):
            z = self.mlps[i](torch.einsum('bf,bfd->bd', a[:, i], v))
            res.append(z)
             
       return torch.cat(res, dim=-1)
        


        
# %%
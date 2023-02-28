import torch
from torch import nn


class BaselineModel(nn.Module):
    def __init__(self, n_protons, n_neutrons, hidden_dim, output_dim):
        super().__init__()
        self.emb_proton = nn.Embedding(
            n_protons, hidden_dim
        )  # [ batch_size, hidden_dim ]
        self.emb_neutron = nn.Embedding(
            n_neutrons, hidden_dim
        )  # [ batch_size, hidden_dim ]
        self.nonlinear = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # bigger init
        self.emb_proton.weight.data.uniform_(-1, 1)
        self.emb_neutron.weight.data.uniform_(-1, 1)

    def forward(self, x):  # x: [ batch_size, 2 [n_protons, n_neutrons] ]
        proton = self.emb_proton(x[:, 0])  # [ batch_size, hidden_dim ]
        neutron = self.emb_neutron(x[:, 1])  # [ batch_size, hidden_dim ]
        x = torch.cat([proton, neutron], dim=1)  # [ batch_size, 2 * hidden_dim ]
        x = self.nonlinear(x)  # [ batch_size, output_dim ]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, act = nn.SiLU(), elementwise_affine=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            act,
            nn.Linear(hidden_dim, d_model),
        )
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=elementwise_affine)

    def forward(self, x):
        x = x + self.mlp(x)
        return self.layer_norm(x)

class ResidualModel(nn.Module):
    def __init__(self, n_protons, n_neutrons, hidden_dim, output_dim, depth=3):
        super().__init__()
        self.emb_proton = nn.Embedding(
            n_protons, hidden_dim
        )  # [ batch_size, hidden_dim ]
        self.emb_neutron = nn.Embedding(
            n_neutrons, hidden_dim
        )  # [ batch_size, hidden_dim ]
        self.linear_in = nn.Linear(2 * hidden_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(depth)]
        )
        self.linear_out = nn.Linear(hidden_dim, output_dim)
        # bigger init
        self.emb_proton.weight.data.uniform_(-1, 1)
        self.emb_neutron.weight.data.uniform_(-1, 1)

    def forward(self, x):  # x: [ batch_size, 2 [n_protons, n_neutrons] ]
        proton = self.emb_proton(x[:, 0])  # [ batch_size, hidden_dim ]
        neutron = self.emb_neutron(x[:, 1])  # [ batch_size, hidden_dim ]
        x = torch.cat([proton, neutron], dim=1)  # [ batch_size, 2 * hidden_dim ]
        x = self.linear_in(x)  # [ batch_size, hidden_dim ]
        for block in self.residual_blocks:
            x = block(x)
        x = self.linear_out(x)  # [ batch_size, output_dim ]
        return x

import torch
from torch import nn
from typing import List
from .modules import ResidualBlock


class RNN(nn.Module):
    def __init__(
        self,
        vocab_size: List[int],
        non_embedded_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
        dropout: float = 0.0,
        lipschitz: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proton_emb = torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_dim))
        self.neutron_emb = torch.nn.init.kaiming_uniform_(torch.empty(1, hidden_dim))
        self.task_emb = torch.nn.init.kaiming_uniform_(
            torch.empty(vocab_size[-1], hidden_dim)
        )
        self.proton_emb = nn.Parameter(self.proton_emb)
        self.neutron_emb = nn.Parameter(self.neutron_emb)
        self.task_emb = nn.Parameter(self.task_emb)

        self.protonet = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, activation=nn.SiLU(), dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.neutronet = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, activation=nn.SiLU(), dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.nonlinear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            # *[ResidualBlock(hidden_dim, activation=nn.SiLU()) for _ in range(depth)],
        )
        self.readout = nn.Linear(2 * hidden_dim, output_dim)

    def _protons(self, n):
        p = self.proton_emb
        return torch.vstack([(p := self.protonet(p)) for _ in range(n + 1)])

    def _neutrons(self, n):
        p = self.neutron_emb
        return torch.vstack([(p := self.neutronet(p)) for _ in range(n + 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_max, n_max = x[:, 0].amax(), x[:, 1].amax()
        protons = self._protons(p_max)[x[:, 0]]
        neutrons = self._neutrons(n_max)[x[:, 1]]
        out = torch.cat([protons, neutrons], dim=1)
        # out = self.nonlinear(out)
        return torch.sigmoid(self.readout(out))

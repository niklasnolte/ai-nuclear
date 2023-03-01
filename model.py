import torch
from torch import nn
from functools import partial
import mup
import warnings
from typing import Callable, Optional
from data import Data


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
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.readout = nn.Linear(hidden_dim, output_dim)
        # bigger init
        self.emb_proton.weight.data.uniform_(-1, 1)
        self.emb_neutron.weight.data.uniform_(-1, 1)

    def forward(self, x):  # x: [ batch_size, 2 [n_protons, n_neutrons] ]
        proton = self.emb_proton(x[:, 0])  # [ batch_size, hidden_dim ]
        neutron = self.emb_neutron(x[:, 1])  # [ batch_size, hidden_dim ]
        x = torch.cat([proton, neutron], dim=1)  # [ batch_size, 2 * hidden_dim ]
        x = self.nonlinear(x)  # [ batch_size, hidden_dim ]
        x = self.readout(x)  # [ batch_size, output_dim ]
        return x


class ResidualBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, act=nn.SiLU(), elementwise_affine=True):
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
        self.readout = nn.Linear(hidden_dim, output_dim)
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
        x = self.readout(x)  # [ batch_size, output_dim ]
        return x


def get_model_fn(config):
    if config.MODEL == "baseline":
        return BaselineModel
    elif config.MODEL == "residual":
        return ResidualModel
    else:
        raise ValueError(
            f"Unknown model: {config.MODEL}, choose between 'baseline' and 'residual'"
        )


def _append_readout(model_fn: Callable) -> Callable:
    """Append a muP readout to a model. If the model is a sequential model,
    the readout replaces the last element in the sequence. Otherwise,
    the readout layer is expected to be an attribute.

    Args:
        model_fn (callable): Function which returns a model.
    """

    def model_fn_with_readout(*args, **kwargs):
        model = model_fn(*args, **kwargs)
        if isinstance(model, nn.Sequential):
            assert isinstance(
                model[-1], nn.Linear
            ), "Last layer of sequential model must be linear (readout)"
            old_readout = model.pop(len(model) - 1)
            model.append(mup.MuReadout(*old_readout.weight.T.shape))
        else:
            assert hasattr(
                model, "readout"
            ), "Model must be sequential or have a readout attribute"
            old_readout = model.readout
            model.readout = mup.MuReadout(*old_readout.weight.T.shape)
        return model

    return model_fn_with_readout


def make_mup(
    model_fn: Callable, **scale_kwargs
) -> nn.Module:
    """Reinitialize model with mup scaling of relevant dimensions. Takes a function which returns a model and returns a model with mup scaling.
    Assumes the model has a readout linear layer which is either the last layer in a sequential model or an attribute of the model.

    Args:
        model_fn (Callable): Function which returns a nn.Module model.
        init_fn (Callable, optional): Function which initializes the model parameters in-place. Defaults to Kaiming uniform with a = sqrt(5).

    Raises:
        ValueError: If depth is in scale_kwargs. Depth is not a scaling parameter.

    Returns:
        nn.Module: Model with mup scaling.
    """
    if "depth" in (k.lower() for k in scale_kwargs.keys()):
        warnings.warn(
            "Depth found in scale_kwargs. Scaling depth is not allowed by muP. Is this intentional?"
        )
    model_fn = _append_readout(model_fn)
    base_kwargs = {k: 1 for k in scale_kwargs}
    delta_kwargs = {k: 2 for k in scale_kwargs}
    base = model_fn(**base_kwargs)
    delta = model_fn(**delta_kwargs)
    model = model_fn(**scale_kwargs)
    mup.set_base_shapes(model, base, delta=delta)
    for name, param in model.named_parameters():
      if "weight" in name.lower(): # FIXME or not
        mup.init.kaiming_uniform_(param, a=5**0.5)
    return model


def get_model_and_optim(data: Data, config):
    # set up model
    n_protons, n_neutrons = data.vocab_size
    output_dim = sum(data.output_map.values())

    model_fn = get_model_fn(config)
    model_fn = partial(
        model_fn,
        n_protons=n_protons,
        n_neutrons=n_neutrons,
        output_dim=output_dim,
    )
    model = make_mup(model_fn, hidden_dim=config.HIDDEN_DIM).to(config.DEV)
    optimizer = mup.MuAdamW(model.parameters(), lr=config.LR, weight_decay=config.WD)
    return model, optimizer

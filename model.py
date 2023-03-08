import torch
from torch import nn
from functools import partial
import mup
import warnings
from typing import Callable, Union, Iterable
from data import Data


class BaselineModel(nn.Module):
    def __init__(
        self, vocab_size: Union[int, Iterable], hidden_dim: int, output_dim: int
    ):
        """
        :param vocab_size: number of tokens in the vocabulary, or an Iterable of vocab sizes for each input
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: dimension of the output layer
        """

        super().__init__()
        if isinstance(vocab_size, int):
            vocab_size = [vocab_size]
        self.emb = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in vocab_size])

        self.nonlinear = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: [ batch_size, 2 ]
        if len(self.emb) == 1:
            embs = [self.emb[0](x[:, i]) for i in range(x.shape[1])]
        else:
            embs = [self.emb[i](x[:, i]) for i in range(len(self.emb))]
        x = torch.cat(embs, dim=1)  # [ batch_size, 2 * hidden_dim ]
        x = self.nonlinear(x)  # [ batch_size, hidden_dim ]
        x = self.readout(x)  # [ batch_size, output_dim ]
        return x


class SplitupModel(nn.Module):
    def __init__(
        self,
        vocab_size: Union[int, Iterable],
        hidden_dim: int,
        output_dim: Iterable[int],
    ):
        """
        :param vocab_size: number of tokens in the vocabulary, or an Iterable of vocab sizes for each input
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: multiple dimensions of the output layers (later concatenated)
        """

        super().__init__()

        if isinstance(vocab_size, int):
            vocab_size = [vocab_size]
        self.emb = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in vocab_size])


        self.n_tasks = len(output_dim)

        #assert hidden_dim % self.n_tasks == 0, f"hidden_dim ({hidden_dim}) must be divisible by n_tasks ({self.n_tasks})"

        d_model = hidden_dim // self.n_tasks
        self.nonlinears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2*hidden_dim, d_model),
                nn.SiLU(),
                nn.LayerNorm(d_model, elementwise_affine=False),
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                mup.MuReadout(d_model, od),
            )
            for od in output_dim
        ])

    def forward(self, x):  # x: [ batch_size, 2 ]
        if len(self.emb) == 1:
            embs = [self.emb[0](x[:, i]) for i in range(x.shape[1])]
        else:
            embs = [self.emb[i](x[:, i]) for i in range(len(self.emb))]
        x = torch.cat(embs, dim=1)  # [ batch_size, 2 * hidden_dim ]
        x = torch.cat([nl(x) for nl in self.nonlinears], dim=1)  # [ batch_size, sum(output_dim) ]
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self, d_model: int, hidden_dim: int, act=nn.SiLU(), elementwise_affine=True
    ):
        """
        :param d_model: dimension of the input and output
        :param hidden_dim: dimension of the hidden layer
        :param act: activation function
        :param elementwise_affine: whether to use elementwise affine in the layer norm
        """
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
    def __init__(
        self,
        vocab_size: Union[int, Iterable],
        hidden_dim: int,
        output_dim: int,
        depth: int = 3,
    ):
        """
        :param vocab_size: number of tokens in the vocabulary, or an Iterable of vocab sizes for each input
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: dimension of the output layer
        :param depth: number of residual blocks
        """

        super().__init__()
        if isinstance(vocab_size, int):
            self.emb = nn.Embedding(vocab_size, hidden_dim)
        else:
            self.emb = nn.ModuleList([nn.Embedding(v, hidden_dim) for v in vocab_size])

        self.linear_in = nn.Linear(2 * hidden_dim, hidden_dim)
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(depth)]
        )
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):  # x: [ batch_size, 2 ]
        if isinstance(self.emb, Iterable):
            embs = [self.emb[i](x[:, i]) for i in range(len(self.emb))]
        else:
            embs = [self.emb(x[:, i]) for i in range(x.shape[1])]

        x = torch.cat(embs, dim=1)  # [ batch_size, 2 * hidden_dim ]
        x = self.linear_in(x)  # [ batch_size, hidden_dim ]
        for block in self.residual_blocks:
            x = block(x)
        x = self.readout(x)  # [ batch_size, output_dim ]
        return x


def get_model_fn(config):
    if config.MODEL == "baseline":
        return BaselineModel
    elif config.MODEL == "splitup":
        return SplitupModel
    elif config.MODEL == "residual":
        return ResidualModel
    elif config.MODEL == "tied":
        return TiedModel
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
        # check if model already has a readout, FIXME: this is a hack
        if any([isinstance(x, mup.MuReadout) for x in model.modules()]):
          return model
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


def make_mup(model_fn: Callable, **scale_kwargs) -> nn.Module:
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
    base_kwargs = {k: 32 for k in scale_kwargs}
    delta_kwargs = {k: 64 for k in scale_kwargs}
    base = model_fn(**base_kwargs)
    delta = model_fn(**delta_kwargs)
    model = model_fn(**scale_kwargs)
    mup.set_base_shapes(model, base, delta=delta)
    del base, delta
    for name, param in model.named_parameters():
        if "weight" in name.lower():  # FIXME or not
            mup.init.kaiming_uniform_(param, a=5**0.5)
    return model


def get_model_and_optim(data: Data, config):
    # set up model
    if config.MODEL == "splitup":
      output_dim = list(data.output_map.values())
    else:
      output_dim = sum(data.output_map.values())

    model_fn = get_model_fn(config)
    model_fn = partial(
        model_fn,
        vocab_size=data.vocab_size,
        output_dim=output_dim,
    )
    model = make_mup(model_fn, hidden_dim=config.HIDDEN_DIM).to(config.DEV)
    optimizer = mup.MuAdamW(model.parameters(), lr=config.LR, weight_decay=config.WD)
    return model, optimizer

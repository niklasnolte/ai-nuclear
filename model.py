import torch
from torch import nn
from functools import partial
import mup
import warnings
from typing import Callable, Iterable, Optional, Tuple, Union, List
from data import Data
from monotonenorm.functional import direct_norm
import os

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
        self.task_emb = torch.nn.init.kaiming_uniform_(torch.empty(vocab_size[-1], hidden_dim))
        self.proton_emb = nn.Parameter(self.proton_emb)
        self.neutron_emb = nn.Parameter(self.neutron_emb)
        self.task_emb = nn.Parameter(self.task_emb)

        self.protonet = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation=nn.SiLU(), dropout=dropout) for _ in range(depth)])
        self.neutronet = nn.Sequential(
            *[ResidualBlock(hidden_dim, activation=nn.SiLU(), dropout=dropout) for _ in range(depth)])
        self.nonlinear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            # *[ResidualBlock(hidden_dim, activation=nn.SiLU()) for _ in range(depth)],
        )
        self.readout = nn.Linear(2 * hidden_dim, output_dim)

    def _protons(self, n):
        p = self.proton_emb
        return torch.vstack([(p:=self.protonet(p)) for _ in range(n+1)])

    def _neutrons(self, n):
        p = self.neutron_emb
        return torch.vstack([(p:=self.neutronet(p)) for _ in range(n+1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_max, n_max = x[:, 0].amax(), x[:, 1].amax()
        protons = self._protons(p_max)[x[:, 0]]
        neutrons = self._neutrons(n_max)[x[:, 1]]
        out = torch.cat([protons, neutrons], dim=1)
        # out = self.nonlinear(out)
        return torch.sigmoid(self.readout(out))


class Base(nn.Module):
    def __init__(
        self,
        vocab_size: List[int],
        non_embedded_input_dim: int,
        hidden_dim: int,
        embedding_dim: Optional[int] = None,
        share_embeddings: bool = False,
    ):
        super().__init__()
        self.non_embedded_input_dim = non_embedded_input_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim or hidden_dim
        self.share_embeddings = share_embeddings
        self.input_dim = self.embedding_dim * len(vocab_size) + non_embedded_input_dim
        # we are using weights here because that is more convenient for dimn_regularization
        if share_embeddings:
            self.emb = nn.Embedding(vocab_size[0], self.embedding_dim).weight
        else:
            self.emb = nn.ParameterList(
                [nn.Embedding(v, self.embedding_dim).weight for v in self.vocab_size]
            )
        self.hidden_dim = hidden_dim

    def forward_with_embeddings(self, x, embs):
        # x = self.embed_input(x, embs)
        # implement this
        raise NotImplementedError()

    def forward(self, x):
        return self.forward_with_embeddings(x, self.emb)

    def embed_input(self, x, embs):
        if self.share_embeddings:
            embs = [embs[x[:, i].long()] for i, _ in enumerate(self.vocab_size)]
        else:
            embs = [embs[i][x[:, i].long()] for i, _ in enumerate(self.vocab_size)]
        if self.non_embedded_input_dim > 0:
            embs.append(x[:, len(self.vocab_size) :])
        return torch.cat(embs, dim=1)  # [ batch_size, 2 * hidden_dim ]


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


class BaselineModel(Base):
    def __init__(
        self,
        vocab_size: List[int],
        non_embedded_input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
        lipschitz: bool = False,
    ):
        """
        :param vocab_size: number of tokens in the vocabulary,
          or an Iterable of vocab sizes for each input. One embedding layer will be created for each input.
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: dimension of the output layer
        """

        super().__init__(vocab_size, non_embedded_input_dim, hidden_dim)
        norm = direct_norm if lipschitz else (lambda x: x)
        act = nn.ReLU()
        self.nonlinear = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            *[
                ResidualBlock(hidden_dim, norm=norm, activation=act)
                for _ in range(depth)
            ],
        )
        self.readout = norm(nn.Linear(hidden_dim, output_dim))

    def forward_with_embeddings(self, x, embs):  # embs: [ batch_size, 2 * hidden_dim ]
        x = self.embed_input(x, embs)
        x = self.nonlinear(x)  # [ batch_size, hidden_dim ]
        return torch.sigmoid(self.readout(x))  # [ batch_size, output_dim ]

def get_model_fn(config):
    if config.MODEL == "baseline":
        return BaselineModel
    elif config.MODEL == "rnn":
        return RNN
    else:
        raise ValueError(
            f"Unknown model: {config.MODEL}, choose between 'baseline', 'splitup', 'transformer' and 'moe'"
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
            shape = old_readout.weight.T.shape
            model.append(mup.MuReadout(*shape))
        else:
            assert hasattr(
                model, "readout"
            ), "Model must be sequential or have a readout attribute"
            old_readout = model.readout
            model.readout = mup.MuReadout(*old_readout.weight.T.shape)
        return model

    return model_fn_with_readout


def make_mup(model_fn: Callable, shape_file=None, **scale_kwargs) -> nn.Module:
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
    mup.set_base_shapes(model, base, delta=delta, savefile=shape_file, do_assert=False)
    del base, delta
    for name, param in model.named_parameters():
        if "weight" in name.lower() or "emb" in name.lower():  # FIXME or not
            # mup.init.uniform_(param, -.1, .1)
            mup.kaiming_uniform_(param, a=5.**0.5, nonlinearity="leaky_relu")
    return model


def get_model_and_optim(data: Data, config):
    torch.manual_seed(config.SEED)
    # set up model
    if config.MODEL == "splitup" or config.MODEL == "transformer":
        output_dim = list(data.output_map.values())
    else:
        output_dim = sum(data.output_map.values())

    model_fn = get_model_fn(config)
    model_fn = partial(
        model_fn,
        vocab_size=data.vocab_size,
        non_embedded_input_dim=data.X.shape[1] - len(data.vocab_size),
        output_dim=output_dim,
        depth=config.DEPTH,
        dropout=config.DROPOUT,
        lipschitz=config.LIPSCHITZ == "true",
    )
    if hasattr(config, "basedir"):
        # FIXME: this is a terrible hack to avoid saving shapes outside of the
        # actual training run
        shape_file = os.path.join(config.basedir, "shapes.yaml")
    else:
        shape_file = None
    model = make_mup(model_fn, shape_file, hidden_dim=config.HIDDEN_DIM).to(config.DEV)
    # model = model_fn(hidden_dim=config.HIDDEN_DIM).to(config.DEV)
    param_groups = [
        {"params": [p for n, p in model.named_parameters() if "bias" in n.lower()]},
        {
            "params": [
                p for n, p in model.named_parameters() if "bias" not in n.lower()
            ],
            "weight_decay": config.WD,
        },
    ]
    if hasattr(config, "OPTIM") and config.OPTIM == "sgd":
        optimizer = mup.MuSGD(param_groups, lr=config.LR)
    else:
        optimizer = mup.MuAdamW(param_groups, lr=config.LR)
    return model, optimizer

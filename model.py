import torch
from torch import nn
from functools import partial
import mup
import warnings
from typing import Callable, Union, Iterable
from data import Data
from transformer import FilteredAttentionTransformer
import math
from torch import Tensor


class Base(nn.Module):
    def __init__(self, vocab_size: Union[int, Iterable], hidden_dim: int):
        super().__init__()
        if isinstance(vocab_size, int):
            vocab_size = [vocab_size]
        self.emb = nn.ModuleList(
            [nn.Embedding(v, hidden_dim) for v in vocab_size]
        )
        self.hidden_dim = hidden_dim

    def forward_with_embeddings(self, x, embs):
        # x = self.embed_input(x, embs)
        # continue here
        raise NotImplementedError()

    def forward(self, x):
        return self.forward_with_embeddings(x, self.emb)

    def embed_input(self, x, embs):
        if len(embs) == 1:
            embs = [embs[0](x[:, i]) for i in range(x.shape[1])]
        else:
            embs = [embs[i](x[:, i]) for i in range(len(embs))]
        return torch.cat(embs, dim=1)  # [ batch_size, 2 * hidden_dim ]


class BaselineModel(Base):
    def __init__(
        self, vocab_size: Union[int, Iterable], hidden_dim: int, output_dim: int, dropout: float = 0.
    ):
        """
        :param vocab_size: number of tokens in the vocabulary,
          or an Iterable of vocab sizes for each input. One embedding layer will be created for each input.
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: dimension of the output layer
        """

        super().__init__(vocab_size, hidden_dim)

        self.nonlinear = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            ResidualBlock(hidden_dim, dropout=dropout),
            nn.SiLU(),
            ResidualBlock(hidden_dim, dropout=dropout),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward_with_embeddings(self, x, embs):  # embs: [ batch_size, 2 * hidden_dim ]
        x = self.embed_input(x, embs)
        x = self.nonlinear(x)  # [ batch_size, hidden_dim ]
        return self.readout(x)  # [ batch_size, output_dim ]


class SplitupModel(Base):
    def __init__(
        self,
        vocab_size: Union[int, Iterable],
        hidden_dim: int,
        output_dim: Iterable[int],
    ):
        """
        :param vocab_size: number of tokens in the vocabulary,
          or an Iterable of vocab sizes for each input. One embedding layer will be created for each input.
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: multiple dimensions of the output layers (later concatenated)
        """

        super().__init__(vocab_size, hidden_dim)

        self.n_tasks = len(output_dim)

        d_model = hidden_dim // self.n_tasks
        self.nonlinears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, d_model),
                    nn.SiLU(),
                    nn.LayerNorm(d_model, elementwise_affine=False),
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    mup.MuReadout(d_model, od),
                    # nn.Linear(d_model, od),
                )
                for od in output_dim
            ]
        )

    def forward_with_embeddings(self, x, embs):  # embs: [ batch_size, 2 * hidden_dim ]
        x = self.embed_input(x, embs)
        return torch.cat(
            [nl(x) for nl in self.nonlinears], dim=1
        )  # [ batch_size, sum(output_dim) ]


def get_model_fn(config):
    if config.MODEL == "baseline":
        return BaselineModel
    elif config.MODEL == "splitup":
        return SplitupModel
    elif config.MODEL == "transformer":
        return FilteredAttentionTransformer
    else:
        raise ValueError(
            f"Unknown model: {config.MODEL}, choose between 'baseline' and 'splitup'"
        )


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        requires_grad: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # self.pe = nn.Parameter(pe, requires_grad=requires_grad)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = self.pe[x]
        return self.dropout(x)


class BinaryEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        requires_grad: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.zeros((max_len, d_model))
        for i in range(max_len):
            self.embedding[i] = torch.tensor(
                [int(x) for x in bin(i)[2:].zfill(d_model)]
            )
        self.embedding = ((self.embedding) * 2 - 1) * 1e-3
        self.embedding = nn.Parameter(self.embedding, requires_grad=requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        x = self.embedding[x]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, activation: nn.Module = nn.SiLU
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            activation(),
            nn.Linear(d_model * 2, d_model),
        )
        # self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm = nn.BatchNorm1d(d_model, affine=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.norm(x + self.dropout(self.ff(x)))


class PosEmbed(Base):
    def __init__(
        self,
        vocab_size: Union[int, Iterable],
        hidden_dim: int,
        output_dim: Iterable[int],
        dropout: float = 0.,
        train_embeddings: bool = True,
    ):
        super().__init__(vocab_size, hidden_dim)

        self.n_tasks = len(output_dim)
        d_model = hidden_dim # // self.n_tasks
        emb_size = 64
        self.pe = PositionalEncoding(
            emb_size, dropout=0, max_len=200, requires_grad=False
        )
        self.embed = nn.ModuleList(
            [
                BinaryEmbedding(
                    emb_size, dropout=0, max_len=vocab, requires_grad=train_embeddings
                )
                for vocab in vocab_size
            ]
        )
        self.input = nn.Linear(emb_size * 2, hidden_dim)
        # self.input = nn.Linear(2, hidden_dim)

        self.nonlinears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, d_model),
                    nn.SiLU(),
                    # nn.LayerNorm(d_model, elementwise_affine=False),
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    ResidualBlock(d_model, dropout=dropout),
                    ResidualBlock(d_model, dropout=dropout),
                    ResidualBlock(d_model, dropout=dropout),
                    ResidualBlock(d_model, dropout=dropout),
                    # ResidualBlock(d_model, dropout=dropout),
                    # ResidualBlock(d_model, dropout=dropout),
                    # ResidualBlock(d_model, dropout=dropout),
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    mup.MuReadout(d_model, od),
                    # nn.Linear(d_model, od),
                )
                for od in output_dim
            ]
        )

    def forward_with_embeddings(self, x, embs):  # embs: [ batch_size, 2 * hidden_dim ]
        # x = - (x.float()-80)/200 
        # x = x.float()
        # min_ = x.amin(dim=0)
        # max_ = x.amax(dim=0)
        # x = (x - min_) / (max_ - min_)
        # pe = self.pe(x).flatten(1)
        x = [self.embed[i](x[:, i]) for i in range(x.shape[1])]
        x = torch.cat(x, dim=1)
        # x += pe
        x = self.input(x)
        x = torch.cat(
            [nl(x) for nl in self.nonlinears], dim=1
        )  # [ batch_size, sum(output_dim) ]
        return x

def get_model_fn(config):
    if config.MODEL == "baseline":
        return BaselineModel
    elif config.MODEL == "splitup":
        return SplitupModel
    elif config.MODEL == "transformer":
        return FilteredAttentionTransformer
    elif config.MODEL == "posembed":
        return PosEmbed
    else:
        raise ValueError(
            f"Unknown model: {config.MODEL}, choose between 'baseline' and 'splitup'"
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


def make_mup(model_fn: Callable, config, **scale_kwargs) -> nn.Module:
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
        if "weight" in name.lower():  # FIXME embeddings do not have weight attr in some models
            if "emb" in name.lower():
                a = config.EMB_INIT
                mup.init.uniform_(param, a=-a, b=a)
            else:
                mup.init.kaiming_uniform_(param, a=5**0.5)

    return model


def get_model_and_optim(data: Data, config):
    # set up model
    if config.MODEL == "baseline":
        output_dim = sum(data.output_map.values())
    elif (
        config.MODEL == "splitup"
        or config.MODEL == "transformer"
        or config.MODEL == "posembed"
    ):
        output_dim = list(data.output_map.values())
    else:
        raise ValueError(
            f"Unknown model: {config.MODEL}, choose between 'baseline', 'splitup', 'transformer' and 'posembed'"
        )

    model_fn = get_model_fn(config)
    model_fn = partial(
        model_fn,
        vocab_size=data.vocab_size,
        output_dim=output_dim,
        dropout=config.DROPOUT,
    )
    model = make_mup(model_fn, hidden_dim=config.HIDDEN_DIM, config=config).to(config.DEV)
    # model = model_fn(hidden_dim=config.HIDDEN_DIM).to(config.DEV)
    # separate weights and biases
    weight_params = []
    bias_params = []
    emb_params = []
    for name, param in model.named_parameters():
        if "bias" in name.lower():
            bias_params.append(param)
        elif "embed" in name.lower():
            emb_params.append(param)
        else:
            weight_params.append(param)
        
    optimizer = mup.MuAdamW(
        [
            {"params": weight_params, "weight_decay": config.WD},
            {"params": bias_params, "weight_decay": 0.0},
            {"params": emb_params, "weight_decay": config.WD},
        ],
        lr=config.LR,
        betas=config.betas, #(0.96, 0.992),
        amsgrad=True,
    )
    
    # optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WD, amsgrad=True)
    return model, optimizer

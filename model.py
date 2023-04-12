import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
import mup
import warnings
from typing import Callable, Iterable
from data import Data


class Base(nn.Module):
    def __init__(
        self,
        vocab_size: Iterable,
        non_embedded_input_dim: int,
        hidden_dim: int,
        embedding_dim: int = None,
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
            embs = [embs[x[:, i].long()] for i,_ in enumerate(self.vocab_size)]
        else:
            embs = [embs[i][x[:, i].long()] for i,_ in enumerate(self.vocab_size)]
        if self.non_embedded_input_dim > 0:
            embs.append(x[:, len(self.vocab_size) :])
        return torch.cat(embs, dim=1)  # [ batch_size, 2 * hidden_dim ]


class ResidualBlock(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.0, activation: nn.Module = nn.SiLU
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        return self.norm(x + self.dropout(self.ff(x)))


class BaselineModel(Base):
    def __init__(
        self, vocab_size: Iterable, non_embedded_input_dim: int, hidden_dim: int, output_dim: int
    ):
        """
        :param vocab_size: number of tokens in the vocabulary,
          or an Iterable of vocab sizes for each input. One embedding layer will be created for each input.
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: dimension of the output layer
        """

        super().__init__(vocab_size, non_embedded_input_dim, hidden_dim)

        self.nonlinear = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.readout = nn.Linear(hidden_dim, output_dim)
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward_with_embeddings(self, x, embs):  # embs: [ batch_size, 2 * hidden_dim ]
        x = self.embed_input(x, embs)
        x = self.nonlinear(x)  # [ batch_size, hidden_dim ]
        return self.readout(x)  # [ batch_size, output_dim ]


class SplitupModel(Base):
    def __init__(
        self,
        vocab_size: Iterable,
        non_embedded_input_dim: int,
        hidden_dim: int,
        output_dim: Iterable[int],
    ):
        """
        :param vocab_size: number of tokens in the vocabulary,
          or an Iterable of vocab sizes for each input. One embedding layer will be created for each input.
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: multiple dimensions of the output layers (later concatenated)
        """

        super().__init__(vocab_size, non_embedded_input_dim, hidden_dim)

        self.n_tasks = len(output_dim)

        d_model = hidden_dim // self.n_tasks
        self.nonlinears = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.input_dim, d_model),
                    nn.SiLU(),
                    nn.LayerNorm(d_model, elementwise_affine=False),
                    nn.Linear(d_model, d_model),
                    nn.SiLU(),
                    mup.MuReadout(d_model, od),
                )
                for od in output_dim
            ]
        )

    def forward_with_embeddings(self, x, embs):  # embs: [ batch_size, 2 * hidden_dim ]
        x = self.embed_input(x, embs)
        return torch.cat(
            [nl(x) for nl in self.nonlinears], dim=1
        )  # [ batch_size, sum(output_dim) ]



class GatingNetwork(nn.Module):
    def __init__(self, input_size, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = mup.MuReadout(input_size, num_experts)

    def forward(self, x):
        return F.softmax(self.fc(x), dim=-1)


class MoEModel(Base):
    def __init__(
        self, vocab_size: Iterable, non_embedded_input_dim: int, hidden_dim: int, output_dim: int
    ):
        """
        :param vocab_size: number of tokens in the vocabulary,
          or an Iterable of vocab sizes for each input. One embedding layer will be created for each input.
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: dimension of the output layer
        """

        super().__init__(vocab_size, non_embedded_input_dim, hidden_dim)

        self.num_experts = 4

        def expert(input_size, hidden_size, output_size):
            return nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.SiLU(),
                mup.MuReadout(hidden_size, output_size),
            )

        self.experts = nn.ModuleList(
            [
                expert(self.input_dim, hidden_dim, output_dim)
                for _ in range(self.num_experts)
            ]
        )
        self.gating_network = GatingNetwork(self.input_dim, self.num_experts)
        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward_with_embeddings(self, x, embs):
        x = self.embed_input(x, embs)

        weights = self.gating_network(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-1)
        output = torch.einsum("ij,ikj->ik", weights, expert_outputs)
        return output


from transformer import DefaultTransformer


def get_model_fn(config):
    if config.MODEL == "baseline":
        return BaselineModel
    elif config.MODEL == "splitup":
        return SplitupModel
    elif config.MODEL == "transformer":
        return DefaultTransformer
    elif config.MODEL == "moe":  # Add a new condition for the MoE model
        return MoEModel
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
        if "weight" in name.lower() or "emb" in name.lower():  # FIXME or not
            # mup.init.uniform_(param, -.1, .1)
            mup.init.kaiming_uniform_(param, a=5**0.5, nonlinearity="leaky_relu")
    return model


def get_model_and_optim(data: Data, config):
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
    )
    model = make_mup(model_fn, hidden_dim=config.HIDDEN_DIM).to(config.DEV)
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
    # optimizer = mup.MuSGD(param_groups, lr=config.LR, momentum=.99, nesterov=True)
    optimizer = mup.MuAdam(param_groups, lr=config.LR)
    # split into weights biases
    # optimizer = torch.optim.AdamW(param_groups, lr=config.LR, amsgrad=True)
    # optimizer = torch.optim.AdamW(model, lr=config.LR, amsgrad=True)
    return model, optimizer

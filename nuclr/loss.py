import torch
from torch.nn import functional as F
import functools
import argparse


def accuracy(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(targets)
    masked_target = targets[mask]
    masked_output = output[mask]
    return (masked_output.argmax(dim=-1) == masked_target).float().mean()

def mse(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(targets)
    masked_target = targets[mask]
    masked_output = output[mask]
    return F.mse_loss(masked_output, masked_target, reduction="mean")

def rmse(output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(mse(output, targets))

def random_softmax(shape, scale=1):
    x = torch.rand(shape)
    return torch.softmax(scale * x, dim=-1) * x.shape[-1]



def get_balanced_accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    output: [batch_size, output_dim]
    target: [batch_size]
    """
    target = target.long()
    output = torch.argmax(output, dim=1)
    assert output.shape == target.shape

    n_classes = len(target.unique())
    class_occurrences = torch.bincount(target)
    class_weight = 1 / class_occurrences.float() / n_classes
    return ((output == target).float() * class_weight[target]).sum()


def get_eval_fn_for(task_name):
  if task_name == "binding_energy":
    def eval_fn(output, input_):
      nprotons = input_[:, 0]
      nneutrons = input_[:, 1]
      return output * (nprotons + nneutrons)
    return eval_fn
  else:
    return lambda x, _: x


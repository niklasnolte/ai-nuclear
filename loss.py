import torch
from torch.nn import functional as F
from sklearn.preprocessing import QuantileTransformer
import functools
import argparse
import math
import unittest
from model import Base


def random_softmax(shape, scale=1):
    x = torch.rand(shape)
    return torch.softmax(scale * x, dim=-1) * x.shape[-1]


@functools.lru_cache(maxsize=10)
def _get_index_distribution(max_idx, exp):
    dist = torch.arange(1, max_idx + 1) ** exp
    return dist / dist.sum()


def pick_random_index(max_idx, exp=-1):
    """
    Pick a random index from 1 to max_idx, with probability proportional to index^exp
    """
    base = _get_index_distribution(max_idx, exp)
    sample = torch.multinomial(base, 1)
    return sample + 1


class WeightScaler(torch.nn.Module):
    def __init__(self, shape, scale=1, trainable=False):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.rand(shape), requires_grad=trainable)
        self.scale = scale
        self.trainable = trainable
        self.shape = shape

    def get_weights(self):
        if self.trainable:
            return torch.softmax(self.weights, dim=-1) * self.shape[-1]
        return self.random_softmax(self.weights.shape, scale=self.scale)

    def random_softmax(self, shape=None, scale=None):
        shape = shape if shape is not None else self.weights.shape
        scale = scale if scale is not None else self.scale
        return random_softmax(shape, scale)

def random_softmax(shape, scale=1):
    x = torch.rand(shape)
    return torch.softmax(scale * x, dim=-1) * x.shape[-1]


def weight_by_task(output_map: dict, args: argparse.Namespace) -> torch.Tensor:
    weights = []
    for target_name in output_map.keys():
        if target_name in args.TARGETS_CLASSIFICATION:
            weight = args.TARGETS_CLASSIFICATION[target_name]
        else:
            weight = args.TARGETS_REGRESSION[target_name]
        weights.append(weight)
    weights = torch.tensor(weights) / sum(weights)
    return weights.to(args.DEV).view(-1, 1)


def loss_by_task(
    output: torch.Tensor,
    targets: torch.Tensor,
    output_map: dict,
    config: argparse.Namespace,
) -> torch.Tensor:
    """
    output: [batch_size, output_dim]
    targets: [batch_size, targets_dim]

    calculate the loss by task:
    For classification targets, cross entropy
    For regression targets: MSE
    Missing values are Nans, so ignore those

    returns loss: [targets_dim]
    """
    # WARNING: classification comes first
    target_names = list(output_map.keys())
    # reshape output according to output_map and return tuple by regression and classification
    output_column = 0
    loss = torch.zeros((len(target_names), len(targets)), device=output.device)
    # TODO sometimes we mask out a lot of things because there is not much data, but they get assigned
    # 0 loss. that skews the train loss value, but metrics are probably fine.
    for target_column, target_name in enumerate(target_names):
        mask = ~torch.isnan(targets[:, target_column])
        masked_target = targets[:, target_column][mask]
        if target_name in config.TARGETS_CLASSIFICATION:
            size = output_map[target_name]
            out = output[:, output_column : output_column + size]
            loss[target_column, mask] = F.cross_entropy(out[mask], masked_target.long(), reduction="none")
            output_column += size
        else:
            out = output[:, output_column]
            loss[target_column, mask] = F.mse_loss(out[mask], masked_target.float(), reduction="none")
            output_column += 1
    return loss


class LossWithNan(torch.nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss(reduction="sum")

    def forward(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mask = ~torch.isnan(targets)
        masked_target = targets[mask]
        masked_output = output[mask]
        return self.loss(masked_output, masked_target)

def regularize_embedding_dim(
    model: Base,
    X: torch.Tensor,
    Y: torch.Tensor,
    output_map: dict,
    config: argparse.Namespace,
) -> torch.Tensor:
    idx = pick_random_index(model.hidden_dim, config.DIMREG_EXP)
    embs = []
    for emb in model.emb:
        # Vt is d_model by d_model. Projecting A -> A @ V @ Vt = USVtVVt = USVt
        try:
          _, _, Vt = torch.linalg.svd(emb, full_matrices=False)
        except torch.linalg.LinAlgError: # sometimes svd fails with singular matrix
          return torch.zeros(1, device=X.device)
        Vt = Vt[:idx]  # [ idx, d_model]
        # squeeze out the embedding dimension
        embs.append(emb @ Vt.T @ Vt)
    out = model.forward_with_embeddings(X, embs)
    loss = loss_by_task(out, Y, output_map, config)
    return loss


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


def metric_by_task(
    output: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    output_map: dict,
    config: argparse.Namespace,
    qt: QuantileTransformer = None,
) -> torch.Tensor:
    """
    output: [batch_size, output_dim]
    targets: [batch_size, targets_dim]
    output_map: dict
    qt: QuantileTransformer

    calculate the metrics by task:
    For classification targets, accuracy
    For regression targets: RMSE
    Missing values are Nans, so ignore those

    returns accuracy: [targets_dim]
    """
    # WARNING: classification comes first
    target_names = list(output_map.keys())
    # reshape output according to output_map and return tuple by regression and classification
    output_column = 0
    metrics = torch.zeros(len(target_names))

    # classification metrics [acc]
    classification_targets = [
        t for t in target_names if t in config.TARGETS_CLASSIFICATION
    ]
    target_column = 0
    targets = targets.clone()
    for target_name in classification_targets:
        mask = ~torch.isnan(targets[:, target_column])
        masked_target = targets[:, target_column][mask].long()
        size = output_map[target_name]
        out = output[:, output_column : output_column + size]
        metrics[target_column] = 100 * get_balanced_accuracy(out[mask], masked_target)
        output_column += size
        target_column += 1

    # regression metrics [rmse]
    regression_targets = [t for t in target_names if t in config.TARGETS_REGRESSION]
    if len(regression_targets) > 0 and qt is not None:
        targets[:, target_column:] = torch.tensor(
            qt.inverse_transform(targets[:, target_column:].cpu())
        )
        output[:, output_column:] = torch.tensor(
            qt.inverse_transform(output[:, output_column:].cpu())
        )

    for target_name in regression_targets:
        mask = ~torch.isnan(targets[:, target_column])
        masked_target = targets[:, target_column][mask]
        out = output[:, output_column]
        eval_fn = get_eval_fn_for(target_name)
        masked_out = eval_fn(out[mask], inputs[mask])
        masked_target = eval_fn(masked_target, inputs[mask])
        metrics[target_column] = F.mse_loss(masked_out, masked_target.float()).sqrt()
        output_column += 1
        target_column += 1

    return metrics


class TestMetrics(unittest.TestCase):
    def test_loss_by_task(self):
        output = torch.tensor([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        output_map = {"a": 2, "b": 1}
        config = argparse.Namespace()
        config.TARGETS_CLASSIFICATION = ["a"]
        config.TARGETS_REGRESSION = ["b"]
        loss = loss_by_task(output, targets, output_map, config)
        self.assertEqual(loss.shape, (2,))
        self.assertEqual(loss[0], F.cross_entropy(output[:, :2], targets[:, 0].long()))
        self.assertEqual(loss[1], F.mse_loss(output[:, 2], targets[:, 1].float()))

    def test_get_balanced_accuracy(self):
        # test with class weights that are not uniform
        output = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
        targets = torch.tensor([1.0, 1.0, 0.0])
        acc = get_balanced_accuracy(output, targets)
        self.assertEqual(acc.item(), 0.25)

    def test_metric_by_task(self):
        # test metrics with class weights that are not uniform

        output = torch.tensor([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        output_map = {"a": 2, "b": 1}
        config = argparse.Namespace()
        config.TARGETS_CLASSIFICATION = ["a"]
        config.TARGETS_REGRESSION = ["b"]
        metrics = metric_by_task(output, targets, output_map, config)
        self.assertEqual(metrics.shape, (2,))
        self.assertEqual(
            metrics[0], 100 * get_balanced_accuracy(output[:, :2], targets[:, 0].long())
        )
        self.assertEqual(
            metrics[1], F.mse_loss(output[:, 2], targets[:, 1].float()).sqrt()
        )


if __name__ == "__main__":
    unittest.main(argv=[""], verbosity=2, exit=False)

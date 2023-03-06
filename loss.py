import torch
from torch.nn import functional as F
from sklearn.preprocessing import QuantileTransformer
import argparse
import math

def weight_by_task(output_map: dict, args: argparse.Namespace) -> torch.Tensor:
    weights = []
    for target_name in output_map.keys():
        if target_name in args.TARGETS_CLASSIFICATION:
            weight = args.TARGETS_CLASSIFICATION[target_name]
        else:
            weight = args.TARGETS_REGRESSION[target_name]
        weights.append(weight)
    return torch.tensor(weights) / sum(weights)


def loss_by_task(
    output: torch.Tensor, targets: torch.Tensor, output_map: dict, config: argparse.Namespace
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
    loss = torch.zeros(len(target_names))
    for target_column, target_name in enumerate(target_names):
        mask = ~torch.isnan(targets[:, target_column])
        masked_target = targets[:, target_column][mask]
        if target_name in config.TARGETS_CLASSIFICATION:
            size = output_map[target_name]
            out = output[:, output_column : output_column + size]
            loss[target_column] = F.cross_entropy(out[mask], masked_target.long())
            output_column += size
        else:
            out = output[:, output_column]
            loss[target_column] = F.mse_loss(out[mask], masked_target.float())
            output_column += 1
    return loss

def get_balanced_accuracy(output:torch.Tensor, target:torch.Tensor):
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
    return ((output == target).float() * class_weight[target] ).sum()

def metric_by_task(
    output: torch.Tensor,
    targets: torch.Tensor,
    output_map: dict,
    config : argparse.Namespace,
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
    classification_targets = [t for t in target_names if t in config.TARGETS_CLASSIFICATION]
    target_column = 0
    for target_name in classification_targets:
        mask = ~torch.isnan(targets[:, target_column])
        masked_target = targets[:, target_column][mask].long()
        size = output_map[target_name]
        out = output[:, output_column : output_column + size]
        metrics[target_column] = (
            100 * get_balanced_accuracy(out[mask], masked_target)
        )
        output_column += size
        target_column += 1

    # regression metrics [rmse]
    regression_targets = [t for t in target_names if t in config.TARGETS_REGRESSION]
    if len(regression_targets) > 0 and qt is not None:
        targets[:, target_column:] = torch.tensor(qt.inverse_transform(targets[:, target_column:].cpu()))
        output[:, output_column:] = torch.tensor(qt.inverse_transform(output[:, output_column:].cpu()))

    for target_name in regression_targets:
        mask = ~torch.isnan(targets[:, target_column])
        masked_target = targets[:, target_column][mask]
        out = output[:, output_column]
        metrics[target_column] = F.mse_loss(
            out[mask], masked_target.float()
        ).sqrt()
        output_column += 1
        target_column += 1

    return metrics

def test_loss_by_task():
    output = torch.tensor([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])
    targets = torch.tensor([[1., 0.0], [0.0, 1.0]])
    output_map = {"a": 2, "b": 1}
    config = argparse.Namespace()
    config.TARGETS_CLASSIFICATION = ["a"]
    config.TARGETS_REGRESSION = ["b"]
    loss = loss_by_task(output, targets, output_map, config)
    assert loss.shape == (2,)
    assert loss[0] == F.cross_entropy(output[:, :2], targets[:, 0].long())
    assert loss[1] == F.mse_loss(output[:, 2], targets[:, 1].float())
    print("test_loss_by_task passed")

def test_get_balanced_accuracy():
    # test with class weights that are not uniform
    output = torch.tensor([[0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    targets = torch.tensor([1.0, 1.0, 0.0])
    acc = get_balanced_accuracy(output, targets)
    assert math.isclose(acc.item(), .25)
    print("test_get_accuracy passed")

def test_metric_by_task():
    # test metrics with class weights that are not uniform

    output = torch.tensor([[0.1, 0.9, 0.1], [0.9, 0.1, 0.9]])
    targets = torch.tensor([[1., 0.0], [0.0, 1.0]])
    output_map = {"a": 2, "b": 1}
    config = argparse.Namespace()
    config.TARGETS_CLASSIFICATION = ["a"]
    config.TARGETS_REGRESSION = ["b"]
    metrics = metric_by_task(output, targets, output_map, config)
    assert metrics.shape == (2,)
    assert metrics[0] == 100 * get_balanced_accuracy(output[:, :2], targets[:, 0].long())
    assert metrics[1] == F.mse_loss(output[:, 2], targets[:, 1].float()).sqrt()
    print("test_metric_by_task passed")


if __name__ == "__main__":
    test_loss_by_task()
    test_get_balanced_accuracy()
    test_metric_by_task()

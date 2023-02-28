from config import Targets
import torch
from torch.nn import functional as F
from sklearn.preprocessing import QuantileTransformer
import pandas as pd


def loss_by_task(
    output: torch.Tensor, targets: torch.Tensor, output_map: dict
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
        if target_name in Targets.classification:
            size = output_map[target_name]
            out = output[:, output_column : output_column + size]
            loss[target_column] = F.cross_entropy(out[mask], masked_target.long())
            output_column += size
        else:
            out = output[:, output_column]
            loss[target_column] = F.mse_loss(out[mask], masked_target.float())
            output_column += 1
    return loss


def metric_by_task(
    output: torch.Tensor,
    targets: torch.Tensor,
    output_map: dict,
    qt: QuantileTransformer = None,
) -> torch.Tensor:
    """
    output: [batch_size, output_dim]
    targets: [batch_size, targets_dim]

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
    classification_targets = [t for t in target_names if t in Targets.classification]
    target_column = 0
    for target_name in classification_targets:
        mask = ~torch.isnan(targets[:, target_column])
        masked_target = targets[:, target_column][mask]
        size = output_map[target_name]
        out = output[:, output_column : output_column + size]
        metrics[target_column] = (
            100 * (out.argmax(dim=1)[mask] == masked_target.long()).float().mean()
        )
        output_column += size
        target_column += 1

    # regression metrics [rmse]
    regression_targets = [t for t in target_names if t in Targets.regression]
    if len(regression_targets) > 0 and qt is not None:
        targets[:, target_column:] = torch.tensor(qt.inverse_transform(targets[:, target_column:]))
        output[:, output_column:] = torch.tensor(qt.inverse_transform(output[:, output_column:]))

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

import torch
from torch import nn

centered_range = lambda x: torch.arange(x) - (x - 1) / 2


def l1_reg(model, *args, **kwargs):
    reg = 0.0
    lambd = 0.001
    for name, param in model.named_parameters():
        if "weight" in name:
            reg += torch.sum(torch.abs(param))  # L1 norm
    return reg * lambd


def local_reg(model, lambd=0.001, regularize_bias=False, *args, **kwargs):
    """l1 regularization weighted by the distance between neurons connected by a weight for a full model"""
    reg = 0.0
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            reg += _local_reg_layer(layer, regularize_bias)
    return reg * lambd


def _local_reg_layer(layer, regularize_bias=False):
    """l1 regularization weighted by the distance between neurons connected by a weight in a single layer"""
    x_in = centered_range(layer.weight.shape[0])
    x_out = centered_range(layer.weight.shape[1])
    # arbitrarily pick y sepration to be 1.
    y_in = torch.zeros_like(x_in)
    y_out = torch.ones_like(x_out)
    # compute the distance between neurons
    geometric_distances = torch.cdist(
        torch.stack([x_in, y_in], dim=1), torch.stack([x_out, y_out], dim=1)
    )
    reg = torch.sum(geometric_distances * torch.abs(layer.weight))
    if layer.bias is not None and regularize_bias:
        reg += torch.sum(torch.abs(layer.bias))
    return reg


def bimt_reg(
    model, topk=5, lambd=0.001, entropy_threshold=0.5, swap=True, regularize_bias=False
):
    """local l1 regularization + swap neurons"""
    reg = 0.0
    # permute neurons if it reduces the regularization loss
    if swap:
        _swap_reg(model, topk, entropy_threshold, regularize_bias)
    for layer in model.layers:
        if isinstance(layer, nn.Linear):
            # add local l1 regularization
            reg += _local_reg_layer(layer, regularize_bias)
    return reg * lambd


# permute neurons
def _permute_neurons_(incoming_layer, outgoing_layer, i, j):
    """Permute neurons i and j in layer at some depth.
    The i-th and j-th rows (columns) of the incoming (outgoing) weight matrix are swapped.
    """
    # incoming takes
    incoming_layer.weight.data[[i, j], :] = incoming_layer.weight.data[[j, i], :]
    if incoming_layer.bias is not None:
        incoming_layer.bias.data[i], incoming_layer.bias.data[j] = (
            incoming_layer.bias.data[j],
            incoming_layer.bias.data[i],
        )
    outgoing_layer.weight.data[:, [i, j]] = outgoing_layer.weight.data[:, [j, i]]


def _check_and_permute_(incoming_layer, outgoing_layer, i, j, regularize_bias):
    """swap neurons i and j in layer at depth if it decreases the regularization loss"""
    original_loss = _local_reg_layer(
        incoming_layer, regularize_bias
    ) + _local_reg_layer(outgoing_layer, regularize_bias)
    _permute_neurons_(incoming_layer, outgoing_layer, i, j)
    new_loss = _local_reg_layer(incoming_layer, regularize_bias) + _local_reg_layer(
        outgoing_layer, regularize_bias
    )
    if new_loss > original_loss:
        # swap back
        _permute_neurons_(incoming_layer, outgoing_layer, i, j)
        return False
    return True


def _swap_reg(model, top=5, entropy_threshold=0.5, regularize_bias=False):
    """go through the model layers and swap top-k neurons with everything else to reduce regularization loss"""
    linears = [layer for layer in model.layers if isinstance(layer, nn.Linear)]
    for prev_layer, layer in zip(linears[:-1], linears[1:]):
        # in order to not waste compute we first check that there's enough sparsity
        if entropy_ratio(layer.weight) > entropy_threshold:
            continue
        curr_width = prev_layer.weight.shape[0]
        topk_indices = torch.topk(
            torch.abs(prev_layer.weight).sum(dim=1), min(top, curr_width)
        ).indices
        for i in range(curr_width):
            for j in topk_indices:
                if i != j:
                    _check_and_permute_(prev_layer, layer, i, j, regularize_bias)


def entropy_ratio(weight):
    """compute the entropy of a weight matrix and get the ratio compared to max entropy
    matrix"""
    # compute the probability of each weight
    weight = weight.abs().sum(dim=1)
    p = torch.abs(weight) / torch.sum(torch.abs(weight))
    # compute the entropy
    entropy = -torch.sum(p * torch.log(p + 1e-8))
    max_entropy = torch.log(torch.tensor(weight.shape[0], dtype=torch.float))
    return entropy / max_entropy

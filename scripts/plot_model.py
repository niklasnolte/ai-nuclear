import torch
from nuclr.data import prepare_nuclear_data
from nuclr.config import NUCLR
import matplotlib.pyplot as plt
from nuclr.model import get_model_and_optim
from lib.config_utils import parse_arguments_and_get_name
import os


args, name = parse_arguments_and_get_name(NUCLR)
torch.manual_seed(args.SEED)

data = prepare_nuclear_data(args)

print(args)

def load(path):
    return torch.load(path, map_location=torch.device('cpu'))


basedir = os.path.join(args.ROOT, name)
modelpath = os.path.join(basedir, "model_NUCLR_best.pt")
model, _ = get_model_and_optim(data, args)
model.load_state_dict(load(modelpath))
model.eval()


DEVICE = args.DEV

def get_idx(data, target):
    output_map = data.output_map
    if target not in output_map:
        raise ValueError(f"Target {target} not in output map")
    target_size = output_map[target]
    if target_size > 1:
        raise ValueError(f"Target {target} is not a scalar")
    target_idx = 0
    for prev_target, size in output_map.items():
        if prev_target == target:
            return target_idx
        target_idx += size


def plot(target="binding_energy", split="both", rescale=False):
    true = data.y
    with torch.no_grad(): pred = model(data.X)
    if rescale:
        true = torch.tensor(data.regression_transformer.inverse_transform(true))
        pred = torch.tensor(data.regression_transformer.inverse_transform(pred))
    target_idx = get_idx(data, target)
    target_true = true[:, target_idx]
    target_pred = pred[:, target_idx]
    if split == "train":
        split_mask = data.train_mask
    elif split == "val":
        split_mask = data.val_mask
    elif split == "both":
        split_mask = data.train_mask | data.val_mask
    else:
        raise ValueError(f"Unknown split {split}")
    nanmask = ~target_true.isnan()
    mask = split_mask & nanmask

    _, axes = plt.subplots(2, 2, figsize=(20, 14))
    axes = axes.flatten()
    for ax, target in zip(axes, [target_true, target_true - target_pred]):
        sc = ax.scatter(data.X[:, 0][mask], data.X[:, 1][mask], c=target[mask], s=10, cmap="bwr")
        ax.set_xlabel("Z")
        ax.set_ylabel("N")
        #center the colorbar around 0
        if ax == axes[1]:
          max_ = max(abs(sc.get_clim()[0]), abs(sc.get_clim()[1]))
          sc.set_clim(-max_, max_)
        plt.colorbar(sc, ax=ax)
    for ax, title in zip(axes, ["True", "True - Pred"]):
        ax.set_title(title)
    for ax, x in zip(axes[2:], data.X.T):
        ax.scatter(x, target_true, s=10, label="True")
        ax.scatter(x, target_pred, s=10, label="Pred")
        ax.set_xlabel("N" if ax == axes[2] else "Z")
        ax.set_ylabel("Binding energy")
        ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot("binding_energy", split="val", rescale=True)

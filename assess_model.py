# %%
import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
plt.style.use("mystyle-bright.mplstyle")
import numpy as np
from config import Task
from train_full import Trainer
import os
from mup import set_base_shapes
import yaml
from argparse import Namespace

# %%
def read_args(path, device=None):
    args = Namespace(**yaml.safe_load(open(path, "r")))
    # TODO fix the fact that this overrides the args in the yaml file
    # WARNING!!! yaml reorders the dictionary target_regression, make sure to reorder it!!
    args.TARGETS_REGRESSION = {
                    # "binding": 1,
                    "binding_semf": 1,
                    "z": 1,
                    "n": 1,
                    "radius": 1,
                    # "volume": 1,
                    # "surface": 1,
                    # "symmetry": 1,
                    # "coulomb": 1,
                    # "delta": 1,
                    # "half_life_sec": 1,
                    # "abundance": 1,
                    "qa": 1,
                    "qbm": 1,
                    "qbm_n": 1,
                    "qec": 1,
                    # "sn": 1,
                    # "sp": 1,
                }
    args.WANDB = False
    if device:
        args.DEV = device
    args.WHICH_FOLDS = list(range(args.N_FOLDS))
    return args
logdir="/data/submit/nnolte/AI-NUCLEAR-LOGS/FULL/model_baseline/wd_0.01/lr_0.01/epochs_50000/nfolds_20/whichfolds_{fold}/hiddendim_1024/depth_4/seed_0/batchsize_4096/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1/sched_cosine/lipschitz_false/tms_remove"
args = read_args(os.path.join(logdir.format(fold=0), "args.yaml"))
trainer = Trainer(Task.FULL, args)
# %%
data = trainer.data
for fold in range(args.N_FOLDS):
    model_dir = os.path.join(logdir.format(fold=fold), f"model_FULL.pt.{fold}")
    print(model_dir)
    shapes = os.path.join(logdir.format(fold=fold), "shapes.yaml")
    trainer.models[fold].load_state_dict(torch.load(model_dir))
    set_base_shapes(trainer.models[fold], shapes, rescale_params=False, do_assert=False)
# %%
# eval the model performance
metrics = trainer.val_step(False)
metrics

# %%
[m.eval() for m in trainer.models]
outs = [trainer._unscale_output(m(data.X).detach().clone()) for m in trainer.models]
out_val = torch.zeros_like(outs[0])
out_train = torch.zeros_like(outs[0])
for fold, model in enumerate(trainer.models):
    val_mask = data.val_masks[fold]
    out_val[val_mask] = outs[fold][val_mask]

    train_mask = data.train_masks[fold]
    out_train[train_mask] += outs[fold][train_mask]

out_train /= len(trainer.models) - 1 # for each data points there are n-1 / n preds

# %%
def plot_pca(fold=0):
    n_components = 4
    pca = PCA(n_components=n_components)
    print(trainer.data.vocab_size)
    embs = []
    for i in range(len(trainer.data.vocab_size)):
        embs.append(pca.fit_transform(trainer.models[0].emb[i].detach().cpu()))
        print(pca.explained_variance_ratio_)
    for idx in range(n_components-1):
        fig, axes = plt.subplots(2, 1, figsize=(15, 15), dpi=100)
        for emb, m, ax in zip(embs, "oo", axes.flatten()):
            emb = emb[:, idx:]
            if idx < n_components-2:
                y = emb[:, 2]
            else:
                y = np.linspace(0, 1, len(emb))
            colors = plt.cm.viridis(y)
            for i, (x, y) in enumerate(emb[:, :2]):
                ax.annotate(i, (x, y), color=colors[i])
            ax.scatter(emb[:, 0], emb[:, 1], c=colors[:len(emb)], marker=m, s=1)
        axes[0].set_title("Embeddings Z")
        axes[1].set_title("Embeddings N")
        plt.show()

plot_pca(1)
# %%
def plot_repr(fold=0, which=0):
    model = trainer.models[fold]
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=100)
    pca = PCA(n_components=3)
    model.eval()
    task = 0
    task_mask = (data.X[:, -1] == task)# & (~data.y.isnan().view(-1)) & (data.train_mask)
    X = model.emb[which].repeat(1, 3)
    if which == 0:
        X[:, args.HIDDEN_DIM:] = 0
        title = "Z Representations"
    elif which == 1:
        X[:, :args.HIDDEN_DIM] = 0
        X[:, -args.HIDDEN_DIM:] = 0
        title = "N Representations"
    elif which == 2:
        X[:, :args.HIDDEN_DIM] = 0
        title = "Task Representations"
    else:
        X = model.embed_input(data.X[task_mask], model.emb)
        title = "(N, Z) Representations"
    X = model.nonlinear[:1](X)
    X = pca.fit_transform(X.detach().cpu())

    c = X[:, 2]
    plt.scatter(X[:, 0], X[:, 1], c=c, s=1)
    for i, (x, y) in enumerate(X[:, :2]):
        plt.annotate(i, (x, y), color=plt.cm.viridis(c)[i])
    cbar = plt.colorbar()
    cbar.set_label("PCA 3")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.show()
plot_repr(3,0)

# %%
def plot_be_comparison_1d(preds):
  target_idx = 0
  target_name = list(data.output_map.keys())[target_idx]

  target_mask = (data.X[:, -1] == target_idx) & (~data.y.isnan().view(-1))
  pred_be = preds[target_mask, target_idx].detach().cpu()
  true_be = trainer.unscaled_y.view(-1)[target_mask].detach().cpu()
  plt.title(target_name)
  plt.hist(pred_be, bins=20, alpha=0.5, label="pred")
  plt.hist(true_be, bins=20, alpha=0.5, label="true")
  plt.legend()
  plt.show()

plot_be_comparison_1d(out_val)

# %%
def plot_be_heatmap(preds):
    target_name = "binding_semf"
    target_idx = list(data.output_map.keys()).index(target_name)
    target_mask = (data.X[:, -1] == target_idx) & (~data.y.isnan().view(-1))
    pred_target = preds[target_mask, target_idx].detach().cpu()
    true_target = trainer.unscaled_y.view(-1)[target_mask].detach().cpu()
    plt.title(target_name)
    # # heat map of difference as function of z and n
    z, n = data.X[target_mask, 0].detach().cpu(), data.X[target_mask, 1].detach().cpu()
    plt.scatter(z, n, c = pred_target - true_target, s = 1.5, marker="s", cmap="bwr")
    plt.colorbar()
    clim = max(abs(pred_target - true_target)) * .8
    plt.clim(-clim, clim)
    # plt.hist2d(pred_target, true_target, bins=20)
    # plt.xlabel("pred")
    # plt.ylabel("true")
    # plt.show()

plot_be_heatmap(out_val)
# %%

def plot_radius_values_heatmap(preds):
    target_name = "radius"
    target_idx = list(data.output_map.keys()).index(target_name)
    target_mask = (data.X[:, -1] == target_idx)
    pred_target = preds[target_mask, target_idx].detach().cpu()
    true_target = trainer.unscaled_y.view(-1)[target_mask].detach().cpu()
    # 2 figures next to each other
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.title(target_name + " val preds")
    z, n = data.X[target_mask, 0].detach().cpu(), data.X[target_mask, 1].detach().cpu()
    plt.scatter(z, n, c = pred_target, s = 1.5, marker="s", cmap="bwr")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.title(target_name + " truth")
    # plot the true values
    plt.scatter(z, n, c = true_target, s = 1.5, marker="s", cmap="bwr")
    plt.colorbar()


plot_radius_values_heatmap(out_val)

# %%

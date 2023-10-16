# %%
import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
plt.style.use("mystyle-bright.mplstyle")
import numpy as np
from config import NUCLR
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
    if not hasattr(args, "INCLUDE_NUCLEI_GT"):
     args.INCLUDE_NUCLEI_GT = 8
    if device:
        args.DEV = device
    args.WHICH_FOLDS = list(range(args.N_FOLDS))
    return args
logdir="/data/submit/nnolte/AI-NUCLEAR-LOGS/NUCLR/model_baseline/wd_0.01/lr_0.01/epochs_50000/nfolds_20/whichfolds_{fold}/hiddendim_1024/depth_4/seed_0/batchsize_4096/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1/sched_cosine/lipschitz_false/tms_remove"
args = read_args(os.path.join(logdir.format(fold=0), "args.yaml"))
trainer = Trainer(NUCLR, args)
# %%
data = trainer.data
for fold in range(args.N_FOLDS):
    model_dir = os.path.join(logdir.format(fold=fold), f"model_NUCLR.pt.{fold}")
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

out_train /= data.train_masks.sum(0)[:, None]

# %%
def plot_pca(fold=0):
    n_components = 4
    pca = PCA(n_components=n_components)
    print(trainer.data.vocab_size)
    embs = []
    ignore_first = 9
    for i in range(len(trainer.data.vocab_size[:-1])):
        print(trainer.models[0].emb[i].shape)
        embs.append(pca.fit_transform(trainer.models[0].emb[i][ignore_first:].detach().cpu()))
        print(pca.explained_variance_ratio_)
    for idx in range(n_components-1):
        fig, axes = plt.subplots(2, 1, figsize=(15, 15), dpi=100)
        for emb, m, ax in zip(embs, "oo", axes.flatten()):
            emb = emb[:, idx:]
            if idx < n_components-2:
                y = emb[:, 2]
            else:
                y = np.linspace(0, 1, len(emb))
            yplot = (y - y.min()) / (y.max() - y.min())
            colors = plt.cm.viridis(yplot)
            for i, (x, y) in enumerate(emb[:, :2]):
                ax.annotate(i+ignore_first, (x, y), color=colors[i])
            ax.scatter(emb[:, 0], emb[:, 1], c=colors[:len(emb)], marker=m, s=1)
        axes[0].set_title("Embeddings Z")
        axes[1].set_title("Embeddings N")
        plt.show()

plot_pca(1)

# %%
def plot_pca_pdf():
    pca = PCA(n_components=3)
    print(trainer.data.vocab_size)
    embs = []
    ignore_first = 9
    for i in range(len(trainer.data.vocab_size[:-1])):
        print(trainer.models[0].emb[i].shape)
        embs.append(pca.fit_transform(trainer.models[0].emb[i][ignore_first:].detach().cpu()))
        print(pca.explained_variance_ratio_)
    for emb, type_ in zip(embs, ["proton", "neutron"]):
        plt.figure(figsize=(21,7))
        y = emb[:, 2]
        yplot = (y - y.min()) / (y.max() - y.min())
        colors = plt.cm.viridis(yplot)
        for i, (x, y) in enumerate(emb[:, :2]):
            plt.annotate(i+ignore_first, (x, y), color=colors[i])
        plt.scatter(emb[:, 0], emb[:, 1], c=colors[:len(emb)], marker="o", s=1)
        cbar = plt.colorbar()
        cbar.set_ticks([])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        SMALL_SIZE = 8
        MEDIUM_SIZE = 10
        BIGGER_SIZE = 20
        plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.tight_layout()
        plt.subplots_adjust(right=1.1)
        plt.savefig(f"pca_{type_}.pdf")

plot_pca_pdf()

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
    # some preds are 0 because > 100 uncertainty!!
    target_mask &= preds[:, target_idx].detach().cpu() != 0
    #target_mask &= (data.X[:, :2] > 28).all(1)
    pred_target = preds[target_mask, target_idx].detach().cpu()
    true_target = trainer.unscaled_y.view(-1)[target_mask].detach().cpu()
    rms = torch.sqrt(((pred_target - true_target)**2).mean()).item()
    print(f"RMS: {rms} keV")
    # # heat map of difference as function of z and n
    z, n = data.X[target_mask, 0].detach().cpu(), data.X[target_mask, 1].detach().cpu()
    #plt.figure(figsize=(5, (n.max() - n.min()) / (z.max() - z.min()) * 5))
    plt.scatter(n, z, c = true_target - pred_target, s = 1.5, marker="s", cmap="bwr")
    plt.colorbar()
    #clim = max(abs(true_target - pred_target)) * .3
    clim = 300
    plt.clim(-clim, clim)
    plt.tight_layout()
    plt.ylabel("Proton Number")
    plt.xlabel("Neutron Number")
    plt.subplots_adjust(bottom=0.1, left=0.1, right=1.01)
    plt.xlim(5, 168)
    plt.ylim(5, 115)
    plt.savefig("be_heatmap.pdf")
    # plt.hist2d(pred_target, true_target, bins=20)
    # plt.xlabel("pred")
    # plt.ylabel("true")
    # plt.show()

plot_be_heatmap(out_val)
# %%

def plot_radius_values_heatmap(preds):
    target_name = "radius"
    target_idx = list(data.output_map.keys()).index(target_name)
    target_mask = (data.X[:, -1] == target_idx) & (~data.y.isnan().view(-1))
    # some preds are 0 because > 0.005fm uncertainty!!
    target_mask &= preds[:, target_idx].detach().cpu() != 0
    pred_target = preds[target_mask, target_idx].detach().cpu()
    true_target = trainer.unscaled_y.view(-1)[target_mask].detach().cpu()
    rms = torch.sqrt(((pred_target - true_target)**2).mean()).item()
    print(f"RMS: {rms} fm")
    print(f"MEDIAN: {torch.median((pred_target - true_target).abs()).item()} fm")
    # 2 figures next to each other
    z, n = data.X[target_mask, 0].detach().cpu(), data.X[target_mask, 1].detach().cpu()
    heatval = true_target - pred_target
    plt.scatter(n, z, c = heatval, s = 1.5, marker="s", cmap="bwr")
    plt.colorbar()
    clim = 0.03
    plt.clim(-clim, clim)
    plt.tight_layout()
    plt.ylabel("Proton Number")
    plt.xlabel("Neutron Number")
    plt.xlim(5, 168)
    plt.ylim(5, 115)
    plt.subplots_adjust(bottom=0.1, left=0.1, right=1.01)
    plt.savefig("radius_heatmap.pdf")


plot_radius_values_heatmap(out_val)

# %%

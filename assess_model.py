# %%
import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
plt.style.use("mystyle-bright.mplstyle")
import numpy as np
from data import semi_empirical_mass_formula
from config import Task
from train_full import Trainer
import os
from mup import set_base_shapes
import yaml
from argparse import Namespace

# %%

def read_args(path, device=None):
    args = Namespace(**yaml.load(open(path, "r"), Loader=yaml.FullLoader))
    args.WANDB = False
    if device:
        args.DEV = device
    return args
#logdir="/data/submit/nnolte/AI-NUCLEAR-LOGS/FULL/model_baseline/wd_0.01/lr_0.01/epochs_100/nfolds_3/whichfolds_{fold}/hiddendim_64/depth_4/seed_0/batchsize_4096/targetsclassification_None/targetsregression_binding_semf:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/finallr_2e-05/lipschitz_false/dropout_0.0/tms_remove"
seed = 3
logdir=f"/export/d0/nnolte/ai-nuclear/cval_master/FULL/model_baseline/wd_1e-05/lr_0.01/epochs_10000/nfolds_20/whichfolds_0/hiddendim_1024/depth_4/seed_{seed}/batchsize_4096/targetsclassification_None/targetsregression_binding_semf:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/finallr_1e-05/lipschitz_false/dropout_0.0/tms_remove"
args = read_args(os.path.join(logdir.format(fold=0), "args.yaml"))
trainer = Trainer(Task.FULL, args)
data = trainer.data
model = trainer.models[0]
model_dir = os.path.join(logdir, f"model_FULL.pt.0")
shapes = os.path.join(logdir, "shapes.yaml")
model.load_state_dict(torch.load(model_dir))
model = set_base_shapes(model, shapes, rescale_params=False, do_assert=False)


# %%
model.eval()
out_val = trainer._unscale_output(model(data.X).detach().clone())
all_X = torch.cartesian_prod(torch.arange(data.X[:,0].max()+1), torch.arange(data.X[:,1].max()+1), torch.arange(data.X[:,2].max()+1))
# continue here

# %%
def plot_pca():
    n_components = 4
    pca = PCA(n_components=n_components)
    print(trainer.data.vocab_size)
    embs = []
    for i in range(len(trainer.data.vocab_size)):
        embs.append(pca.fit_transform(model.emb[i].detach().cpu()))
        #print(pca.explained_variance_ratio_)
    for idx in range(n_components-1):
        fig, axes = plt.subplots(2, 1, figsize=(15, 15), dpi=100)
        for emb, m, ax in zip(embs, "oo", axes.flatten()):
            emb = emb[:, idx:]
            arange = torch.arange(len(emb))
            mask = (arange > 8)
            if idx < n_components-2:
                y = emb[:, 2]
            else:
                y = np.linspace(0, 1, len(emb))
            #make y[mask] between 0 and 1:
            colorrange = (y[mask] - y[mask].min()) / (y[mask].max() - y[mask].min())
            colors = plt.cm.viridis(colorrange)
            for i, (x, y) in enumerate(emb[:, :2]):
                if mask[i]:
                  ax.annotate(i, (x, y), color=colors[mask[:i].sum()])
            ax.scatter(emb[:, 0][mask], emb[:, 1][mask], c=colors, marker=m, s=1)
        axes[0].set_title(f"Embeddings Z {idx} vs {idx+1}, color is PC {idx+2}")
        axes[1].set_title(f"Embeddings N {idx} vs {idx+1}, color is PC {idx+2}")
        plt.show()

plot_pca()
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
    target_mask = (data.X[:, -1] == target_idx)
    pred_target = preds[target_mask, target_idx].detach().cpu()
    A = data.X[target_mask][:,:2].sum(dim=1).detach().cpu()
    SEMF = semi_empirical_mass_formula(data.X[target_mask][:,0].detach().cpu(), data.X[target_mask][:,1].detach().cpu()) * A
    true_target = trainer.unscaled_y.view(-1)[target_mask].detach().cpu()
    heat_val = pred_target / A
    plt.title(target_name)
    # # heat map of difference as function of z and n
    z, n = data.X[target_mask, 0].detach().cpu(), data.X[target_mask, 1].detach().cpu()
    plt.scatter(z, n, c = heat_val, s = 1.5, marker="s", cmap="bwr")
    plt.colorbar()
    clim = max(abs(heat_val)) * .2
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

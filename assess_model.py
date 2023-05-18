# %%
import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
plt.style.use("mystyle-bright.mplstyle")
import numpy as np
from config import Task
from data import prepare_nuclear_data
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
logdir="./results/FULL/model_baseline/wd_0.01/lr_0.01/epochs_100/nfolds_5/hiddendim_32/depth_4/seed_0/batchsize_4096/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false/tms_remove"
best_model_dir = os.path.join(logdir, "model_best.pt")
shapes = os.path.join(logdir, "shapes.yaml")
# shapes=None
# args = get_args(Task.FULL)
args = read_args(os.path.join(logdir, "args.yaml"))
data = prepare_nuclear_data(args)
trainer = Trainer(Task.FULL, args)
# trainer.model = torch.load(model_dir).to("cpu")
for fold, model in enumerate(trainer.models):
  model.load_state_dict(torch.load(best_model_dir + f".{fold}"))
  set_base_shapes(model, shapes, rescale_params=False, do_assert=False)

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
[m.eval() for m in trainer.models]
outs = [trainer._unscale_output(m(data.X).detach().clone()) for m in trainer.models]
out_val = torch.zeros_like(outs[0])
out_train = torch.zeros_like(outs[0])
for fold, model in enumerate(trainer.models):
    val_mask = data.fold_idxs == fold
    out_val[val_mask] = outs[fold][val_mask]

    train_mask = data.fold_idxs != fold
    out_train[train_mask] += outs[fold][train_mask]

out_train /= len(trainer.models) - 1 # for each data points there are n-1 / n preds

# %%
N = len(data.X)//len(data.output_map)
target_idx = 0
target_name = list(data.output_map.keys())[target_idx]

target_mask = (data.X[:, -1] == target_idx) & (~data.y.isnan().view(-1))
pred_be = out_val[target_mask,  target_idx].detach().cpu()
true_be = trainer.unscaled_y.view(-1)[target_mask].detach().cpu()
plt.title(target_name)
plt.hist(pred_be, bins=20, alpha=0.5, label="pred")
plt.hist(true_be, bins=20, alpha=0.5, label="true")
plt.legend()
plt.show()
# %%
metrics = trainer.val_step(0, log=False)
metrics

# %%

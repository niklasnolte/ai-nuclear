# %%
import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
plt.style.use("mystyle-bright.mplstyle")
import numpy as np
from config import get_args, Task
from data import prepare_nuclear_data
from train_full import Trainer
import os
from mup import set_base_shapes
from model import get_model_and_optim
import yaml
from argparse import Namespace
from glob import glob

def read_args(path, device=None):
    args = Namespace(**yaml.load(open(path, "r"), Loader=yaml.FullLoader)) 
    args.WANDB = False
    if not hasattr(args, "TMS"):
        args.TMS = "remove"
    if device:
        args.DEV = device
    return args
    # FULL/model_baseline/wd_0.01/lr_0.01/epochs_50000/trainfrac_0.9/hiddendim_1024/depth_4/seed_0/batchsize_406
# logdir = "/data/submit/kitouni/ai-nuclear/results/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_2000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/ckpt_None/" # best
# logdir = "/data/submit/kitouni/projects/ai-nuclear/results/FULL/model_baseline/wd_0.1/lr_0.01/epochs_10000/trainfrac_0.8/hiddendim_64/seed_0/batchsize_256/targetsclassification_None/targetsregression_binding:1-z:1-n:1/"
# logdir = "/data/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.01/epochs_1/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding:1-binding_semf:1-z:1-n:1-radius:1-volume:1-surface:1-symmetry:1-coulomb:1-delta:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/"
# logdir = "/data/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.01/epochs_50000/trainfrac_0.9/hiddendim_4096/depth_4/seed_0/batchsize_4069/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
# logdir = "/data/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_2000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
# logdir = "/data/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_10000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
# logdir="/data/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.01/epochs_2000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
# logdir="/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_2000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
logdir="/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_50000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
# list all models that end in an integer
models = [f for f in os.listdir(logdir) if f.split("_")[-1].split(".")[0].isdigit()]
models = sorted(models, key=lambda x: int(x.split("_")[-1].split(".")[0]))
model_idx = 9
print(models[model_idx])
model_dir = os.path.join(logdir, models[model_idx])
shapes = os.path.join(logdir, "shapes.yaml")
# shapes=None
# args = get_args(Task.FULL)
args = read_args(os.path.join(logdir, "args.yaml"))
data = prepare_nuclear_data(args)
trainer = Trainer(Task.FULL, args)
# trainer.model = torch.load(model_dir).to("cpu")
trainer.model.load_state_dict(torch.load(model_dir))
set_base_shapes(trainer.model, shapes, rescale_params=False, do_assert=False)

n_components = 4
pca = PCA(n_components=n_components)
print(trainer.data.vocab_size)
embs = []
for i in range(len(trainer.data.vocab_size)):
    embs.append(pca.fit_transform(trainer.model.emb[i].detach().cpu()))
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
# plt.savefig(logdir + "embeddings.pdf")
# plt.savefig("plots/embeddings" + "|".join(logdir.strip("/").split("/")) + ".pdf")
# %%
def plot_repr(which=0):
    fig, ax = plt.subplots(1, 1, figsize=(12, 12), dpi=100)
    pca = PCA(n_components=3)
    trainer.model.eval()
    task = 0
    task_mask = (data.X[:, -1] == task)# & (~data.y.isnan().view(-1)) & (data.train_mask)
    X = trainer.model.emb[which].repeat(1, 3)
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
        X = trainer.model.embed_input(data.X[task_mask], trainer.model.emb)
        title = "(N, Z) Representations"
    X = trainer.model.nonlinear[:1](X)
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
plot_repr(0)
# %%
trainer.model.eval()
out = trainer.model(data.X)
out_ = trainer._unscale_output(out.detach().clone())

N = len(data.X)//len(data.output_map)
target_idx = 0
target_name = list(data.output_map.keys())[target_idx]
target_mask = (data.X[:, -1] == target_idx) & (~data.y.isnan().view(-1)) & (data.train_mask)
pred_be = out_[target_mask,  target_idx].detach().cpu()
true_be = trainer.unscaled_y.view(-1)[target_mask].detach().cpu()
print((pred_be - true_be).pow(2).mean().sqrt())
plt.title(target_name)
plt.hist(pred_be, bins=20, alpha=0.5, label="pred")
plt.hist(true_be, bins=20, alpha=0.5, label="true")
plt.legend()
plt.show()
# %%
metrics = trainer.val_step(log=False)
metrics
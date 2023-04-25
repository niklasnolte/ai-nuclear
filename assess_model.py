# %%
import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
from config import get_args, Task
from data import prepare_nuclear_data
from train_full import Trainer
import os
from mup import set_base_shapes
from model import get_model_and_optim
import yaml
from argparse import Namespace

# logdir = "/work/submit/kitouni/projects/ai-nuclear/results/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_2000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/ckpt_None/" # best
# logdir = "/work/submit/kitouni/projects/ai-nuclear/results/FULL/model_baseline/wd_0.1/lr_0.01/epochs_10000/trainfrac_0.8/hiddendim_64/seed_0/batchsize_256/targetsclassification_None/targetsregression_binding:1-z:1-n:1/"
# logdir = "/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.01/epochs_1/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding:1-binding_semf:1-z:1-n:1-radius:1-volume:1-surface:1-symmetry:1-coulomb:1-delta:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/"
# logdir = "/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.01/epochs_50000/trainfrac_0.9/hiddendim_4096/depth_4/seed_0/batchsize_4069/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
# logdir = "/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_2000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
# logdir = "/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_10000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
logdir="/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.01/epochs_10/trainfrac_0.9/hiddendim_32/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_none/lipschitz_true"
model_dir = os.path.join(logdir, "model_FULL.pt")
best_model_dir = os.path.join(logdir, "model_best.pt")
shapes = os.path.join(logdir, "shapes.yaml")
# shapes=None
# args = get_args(Task.FULL)
args = yaml.load(open(os.path.join(logdir, "args.yaml"), "r"), Loader=yaml.FullLoader)
args = Namespace(**args)
data = prepare_nuclear_data(args)
trainer = Trainer(Task.FULL, args)
# trainer.model = torch.load(model_dir).to("cpu")
trainer.model.cpu()
trainer.model.load_state_dict(torch.load(best_model_dir))
set_base_shapes(trainer.model, shapes, rescale_params=False, do_assert=False)

n_components = 4
pca = PCA(n_components=n_components)
print(trainer.data.vocab_size)
embs = []
for i in range(len(trainer.data.vocab_size)):
    embs.append(pca.fit_transform(trainer.model.emb[i].detach()))
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
pca = PCA(n_components=3)
trainer.model.eval()
task = 0
task_mask = (data.X[:, -1] == task)# & (~data.y.isnan().view(-1)) & (data.train_mask)
X = trainer.model.embed_input(data.X[task_mask], trainer.model.emb)
X[:, 2048:] = 0
X = trainer.model.nonlinear[:1](X)
X = pca.fit_transform(X.detach())

# y = trainer.unscaled_y[task_mask, task].detach()
y = X[:, 2]
plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
for i, (x, y) in enumerate(X[:, :2]):
    plt.annotate(i, (x, y), color=plt.cm.viridis(y))
cbar = plt.colorbar()
cbar.set_label("PCA 3")
plt.title("Nuclear Embeddings (Z, N)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
# %%
trainer.model.eval()
out = trainer.model(data.X)
out_ = trainer._unscale_output(out.detach().clone())

N = len(data.X)//len(data.output_map)
target_idx = 0
target_name = list(data.output_map.keys())[target_idx]
target_mask = (data.X[:, -1] == target_idx) & (~data.y.isnan().view(-1)) & (data.train_mask)
pred_be = out_[target_mask,  target_idx]
true_be = trainer.unscaled_y.view(-1)[target_mask]
print((pred_be - true_be).pow(2).mean().sqrt())
plt.title(target_name)
plt.hist(pred_be, bins=20, alpha=0.5, label="pred")
plt.hist(true_be, bins=20, alpha=0.5, label="true")
plt.legend()
plt.show()
# %%
X, y = trainer.data.X, trainer.data.y
task = trainer.all_tasks
trainer.model.eval()
with torch.no_grad():
    out = trainer.model(X)
    out_ = trainer._unscale_output(out.clone())  # reg_targets are rescaled
    y_ = trainer.unscaled_y
    metrics_dict = {}
    masks = {"train": trainer.data.train_mask, "val": trainer.data.val_mask}
    for name, mask in masks.items():
        losses, num_samples = trainer.loss_by_task(task[mask], out[mask], y[mask])
        metrics, _ = trainer.metrics_by_task(task[mask], out_[mask], y_[mask])
        m = trainer.construct_metrics(losses, metrics, num_samples, name)
        metrics_dict.update(m)
metrics_dict
# %%
from loss import rmse
rmse(pred_be, true_be)
# %%
metrics = trainer.val_step(log=False)
metrics
# %%

# %%
import torch
from model import get_model_and_optim
from config import get_args, Task
from data import prepare_nuclear_data
from argparse import Namespace
from train_full import Trainer

logdir = "/home/submit/kitouni/ai-nuclear/results/FULL/model_baseline/wd_0.1/lr_0.01/epochs_10000/trainfrac_0.8/hiddendim_64/seed_0/batchsize_256/targetsclassification_None/targetsregression_binding:1-z:1-n:1/"
model_dir = logdir + "model_FULL.pt"
# %%
args = get_args(Task.FULL)
data = prepare_nuclear_data(args)
trainer = Trainer(Task.FULL, args, logdir)
trainer.model = torch.load(logdir + "model_FULL.pt").to("cpu")
# model.load_state_dict(torch.load(logdir + "model_FULL_best.pt"))
# torch.save(model.state_dict(), logdir + "model_FULL_state_dict.pt")
trainer.model.load_state_dict(torch.load(logdir + "model_FULL_state_dict.pt"))
# %%
# trainer.model.load_state_dict(torch.load(model_dir))

# %%
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
pca = PCA(n_components=3)

print(trainer.data.vocab_size)
embs = []
for i in range(len(trainer.data.vocab_size)):
    embs.append(pca.fit_transform(trainer.model.emb[i].detach()))
embs[1][:, 0] = - embs[1][:, 0]
# %%
fig, axes = plt.subplots(2, 1, figsize=(15, 15), dpi=100)
for emb, m, ax in zip(embs, "oo", axes.flatten()):
    y = emb[:, 2] # np.linspace(0, 1, len(emb))
    colors = plt.cm.viridis(y)
    for i, (x, y) in enumerate(emb[:, :2]):
        ax.annotate(i, (x, y), color=colors[i])
    ax.scatter(emb[:, 0], emb[:, 1], c=colors[:len(emb)], marker=m, s=1)
axes[0].set_title("Embeddings Z")
axes[1].set_title("Embeddings N")
# plt.savefig(logdir + "embeddings.pdf")
# plt.savefig("plots/embeddings" + "|".join(logdir.strip("/").split("/")) + ".pdf")
# %%
pca = PCA(n_components=3)
trainer.model.eval()
task = 0
task_mask = (data.X[:, -1] == task)# & (~data.y.isnan().view(-1)) & (data.train_mask)
X = trainer.model.embed_input(data.X[task_mask], trainer.model.emb)
X = trainer.model.nonlinear[0](X)
X = pca.fit_transform(X.detach())
# y = trainer.unscaled_y[task_mask, task].detach()
y = X[:, 2]
plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
cbar = plt.colorbar()
cbar.set_label("PCA 3")
plt.title("Nuclear Embeddings (Z, N)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
# %%
trainer.model.eval()
out = trainer.model(data.X)
out_ = trainer._unscale_output(out.detach())

N = len(data.X)//len(data.output_map)
target_idx = 0
target_mask = (data.X[:, -1] == target_idx) & (~data.y.isnan().view(-1)) & (data.train_mask)
pred_be = out_[target_mask,  target_idx]
true_be = trainer.unscaled_y.view(-1)[target_mask]
print((pred_be - true_be).pow(2).mean().sqrt())

plt.hist(pred_be, bins=20, alpha=0.5, label="pred")
plt.hist(true_be, bins=20, alpha=0.5, label="true")
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

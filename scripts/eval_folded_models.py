# %%
import os
import torch
from nuclr.train import Trainer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import yaml
import glob
import numpy as np
import re

# %%
def plot_pca_2d(trainer, n_components):
  pca = PCA(n_components=n_components)
  embs = []
  ignore_first = 28
  ignore_after = 120
  for i in range(len(trainer.data.vocab_size[:-1])):
      emb = trainer.models[0].emb[i].weight
      print(emb.shape)
      embs.append(pca.fit_transform(emb[ignore_first:ignore_after].detach().cpu()))
      print(pca.explained_variance_ratio_)


  # make all plot combinations in a 2d grid
  fig, axes = plt.subplots(n_components, n_components, figsize=(30, 30), dpi=100)
  fig.suptitle(f"Embeddings of N on epoch {epoch}")

  for idx0 in range(n_components-1):
    for idx1 in range(idx0+1, n_components):
      ax = axes[idx0, idx1]
      ax.set_xticks([])
      ax.set_yticks([])
      emb0 = embs[1][:, idx0]
      emb1 = embs[1][:, idx1]
      y = np.linspace(0, 1, len(emb0))
      yplot = (y - y.min()) / (y.max() - y.min())
      colors = plt.cm.viridis(yplot)
      for i, (x, y) in enumerate(zip(emb0, emb1)):
          ax.annotate(i+ignore_first, (x, y), color=colors[i])
      ax.scatter(emb0, emb1, c=colors[:len(emb0)], marker="o", s=1)
      ax.set_title(f"{idx0} vs {idx1}")
  plt.show()

def plot_pca(trainer, n_components):
    pca = PCA(n_components=n_components)
    print(trainer.data.vocab_size)
    embs = []
    ignore_first = 21
    ignore_after = 10000
    for i in range(len(trainer.data.vocab_size[:-1])):
        emb = trainer.models[0].emb[i].weight
        print(emb.shape)
        embs.append(pca.fit_transform(emb[ignore_first:ignore_after].detach().cpu()))
        print(pca.explained_variance_ratio_)
    for idx in range(n_components-2):
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
        axes[0].set_title(f"Embeddings Z on epoch {epoch}, {idx} vs {idx+1}")
        axes[1].set_title(f"Embeddings N on epoch {epoch}, {idx} vs {idx+1}")
        plt.show()

# %%
# paths = glob.glob("/private/home/nolte/projects/ai-nuclear/results/NUCLR/model_baseline/**/epochs_500000/**/depth_1/**/model_*0000*", recursive=True)
paths = glob.glob("/checkpoint/nolte/nuclr_revisited/NUCLR/model_baseline/**/*gt_20*/**/model_45000*", recursive=True)
paths
# %%
for path in paths:
  path, model_str = os.path.split(path)
  epoch = int(re.findall(r"\d+", model_str)[0])
  # if "wd_0.001" not in path or "lr_0.001" not in path or "hiddendim_1024" not in path or epoch != 200000:
  #     continue
  trainer = Trainer.from_path(path, which_folds=[0], epoch=epoch)
  print(path)
  plot_pca(trainer, 3)

# %%
trainer.val_step()
# %%

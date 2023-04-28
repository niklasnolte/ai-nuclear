# %%
import torch
import pickle
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
# t-sne
from sklearn.decomposition import PCA
# plt.style.use('mystyle-bright.mplstyle')
# %%
def make_path(m):
    ms = MarkerStyle(m)
    return ms.get_path().transformed(ms.get_transform())
# %%
state_dict = torch.load("results/FULL/model_baseline/wd_2e-06/lr_0.002/epochs_100000/trainfrac_0.9/hiddendim_25/dimregcoeff_0.0/distortion_0.001/dimregexp_-1.5/seed_42/randomweights_0.0/targetsclassification_None/targetsregression_z:1-n:1-binding_energy:1-radius:1/model_FULL.pt")
state_dict = state_dict.state_dict()
# %%
protons = state_dict["emb.0"]
neutrons = state_dict["emb.1"]
# %%
# PCA the embeddings
for particle in (protons, neutrons):
  # set axes off
  n_components = 3
  pca = PCA(n_components=n_components)
  for comp1 in range(n_components):
    for comp2 in range(comp1+1, n_components):
      plt.figure(figsize=(10,10))
      embs_pca = pca.fit_transform(particle.detach().cpu().numpy())
      sc = plt.scatter(embs_pca[:,comp1], embs_pca[:,comp2], c=range(len(particle)), cmap="viridis", s=150)
      sc.set_paths([make_path(f"${m:02d}$") for m,_ in enumerate(particle)])
      plt.title(("protons" if particle is protons else "neutrons") + f" PCA {comp1} v {comp2}")
      plt.tight_layout()
      plt.show()
  #annotate
  # for i, txt in enumerate(ap):
  #     plt.annotate(txt.item(), (embs_pca[i,0], embs_pca[i,1]))
  #plt.title("protons" if p is protons else "neutrons")
#   plt.xlim(-0.1, 0.1)
#   plt.ylim(-0.1, 0.1)
# %%
embs_pca.shape

# %%

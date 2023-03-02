# %%
import torch
from data import get_data
import pickle
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
# t-sne
from sklearn.decomposition import PCA
#plt.style.use('mystyle-bright.mplstyle')
import os
# %%
def make_path(m):
    ms = MarkerStyle(m)
    return ms.get_path().transformed(ms.get_transform())
# %%
model_name = 'FULL_dimnreg2_wd0.003_lr0.1_epochs30000_trainfrac0.8_hiddendim64_seed1_modelbaseline_targetsclassificationNone_targetsregressionz:1-n:1-binding_energy:1-radius:1'
sd = torch.load(f"results/{model_name}/model_29900.pt")
model = torch.load(f"results/{model_name}/model_full.pt")
# %%

df = get_data() # vocab_size = (Z, N)
print(df.head())

# hidden_dim = sd["emb.weight"].shape[1]
# model = Model(vocab_size, hidden_dim).requires_grad_(False)
model.load_state_dict(sd)

protons = df['z'].unique()
neutrons = df['n'].unique()

# %%
all_protons = torch.tensor(list(range(len(protons))))
all_neutrons = torch.tensor(list(range(len(neutrons))))


# %%
protons = model.emb_proton(all_protons)
neutrons = model.emb_neutron(all_neutrons)

# %%
# PCA the embeddings
for p, ap in zip((protons, neutrons), (all_protons, all_neutrons)):
  plt.figure(figsize=(10,10))
  # set axes off
  plt.axis('off')
  pca = PCA(n_components=2)
  embs_pca = pca.fit_transform(p.detach().cpu().numpy())
  sc = plt.scatter(*embs_pca.T, c=ap, cmap="viridis", s=500)
  plt.plot(*embs_pca.T, lw = 0.2)
  ax = plt.gca()
  sc.set_paths([make_path(f"${m:02d}$") for m in ap])
  plt.tight_layout()
  plt.savefig("plots/emb_pca_" + ("protons" if p is protons else "neutrons") + ".pdf")
  #annotate
  for i, txt in enumerate(ap):
       plt.annotate(txt.item(), (embs_pca[i,0], embs_pca[i,1]))
  plt.title("protons" if p is protons else "neutrons")
  #plt.xlim(-0.1, 0.1)
  #plt.ylim(-0.1, 0.1)
  plt.show()
# %%





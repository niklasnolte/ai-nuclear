# %%
import torch
from data import get_data
import pickle
import matplotlib.pyplot as plt
# t-sne
from sklearn.decomposition import PCA
# %%
sd = torch.load("models/test/epoch_100000.pt")
model = torch.load("models/test/model.pt")
# %%

_, _, _, _, vocab_size = get_data() # vocab_size = (Z, N)

# hidden_dim = sd["emb.weight"].shape[1]
# model = Model(vocab_size, hidden_dim).requires_grad_(False)
model.load_state_dict(sd)

# %%
all_protons = torch.tensor(list(range(vocab_size[0])))
all_neutrons = torch.tensor(list(range(vocab_size[1])))


# %%
protons = model.emb_proton(all_protons)
neutrons = model.emb_neutron(all_neutrons)

# %%
# PCA the embeddings
for p, ap in zip((protons, neutrons), (all_protons, all_neutrons)):
  plt.figure(figsize=(10,10))
  pca = PCA(n_components=2)
  embs_pca = pca.fit_transform(p.detach().cpu().numpy())
  plt.scatter(*embs_pca.T, c=ap, cmap="coolwarm")
  #annotate
  for i, txt in enumerate(ap):
      plt.annotate(txt.item(), (embs_pca[i,0], embs_pca[i,1]))
  plt.title("protons" if p is protons else "neutrons")
#   plt.xlim(-0.1, 0.1)
#   plt.ylim(-0.1, 0.1)
# %%

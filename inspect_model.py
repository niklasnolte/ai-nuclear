# %%
import torch
from model import Model
from data import get_data
import pickle
import matplotlib.pyplot as plt
# t-sne
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from itertools import product
# %%
# 897 was a good run
sd = torch.load("models/test/best.pt")
model = torch.load("models/test/model.pt")
hidden_dim = sd["emb.weight"].shape[1]
# %%

_, _, _, _, vocab_size = get_data()

# model = Model(vocab_size, hidden_dim).requires_grad_(False)
model.load_state_dict(sd)

# %%
all_combinations = torch.tensor(list(range(vocab_size)))


# %%
embs = model.emb(all_combinations)

# %%
# TSNE the embeddings
tsne = TSNE(n_components=2, random_state=0, init='pca', n_iter=1000, perplexity=30)
embs_tsne = tsne.fit_transform(embs.detach().cpu().numpy())
plt.scatter(embs_tsne[:,0], embs_tsne[:,1], c=all_combinations, cmap="coolwarm")
for i, txt in enumerate(all_combinations):
    plt.annotate(txt.item(), (embs_tsne[i,0], embs_tsne[i,1]))
# %%
# PCA the embeddings
plt.figure(figsize=(10,10))
pca = PCA(n_components=2)
embs_pca = pca.fit_transform(embs.detach().cpu().numpy())
plt.scatter(embs_pca[:,0], embs_pca[:,1], c=all_combinations, cmap="coolwarm")
#annotate
for i, txt in enumerate(all_combinations):
    plt.annotate(txt.item(), (embs_pca[i,0], embs_pca[i,1]))
#plt.xlim(-0.1, 0.1)
#plt.ylim(-0.1, 0.1)
# %%

study = pickle.load(open("study.pickle", "rb"))
# %%
study.best_params
# %%

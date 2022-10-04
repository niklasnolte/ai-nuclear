# %%
import torch
from model import Model
from data import get_data
import pickle
import matplotlib.pyplot as plt
# t-sne
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# %%

sd = torch.load("models/299/best.pt")
hidden_dim = sd["emb.weight"].shape[1]
# %%

X_train, X_test, y_train, y_test, vocab_size = get_data()

model = Model(vocab_size, hidden_dim).requires_grad_(False)
model.load_state_dict(sd)

# %%
embs = model.emb(X_test)

protons = embs[:,0,:]
neutrons = embs[:,1,:]

# %%
# TSNE the embeddings
tsne = TSNE(n_components=2, random_state=0, init='pca', n_iter=1000, perplexity=30)
embs_tsne = tsne.fit_transform(protons.detach().cpu().numpy())
plt.scatter(embs_tsne[:,0], embs_tsne[:,1], c=y_test.detach().cpu().numpy(), cmap="coolwarm")
# %%
# PCA the embeddings
pca = PCA(n_components=2)
embs_pca = pca.fit_transform(protons.detach().cpu().numpy())
plt.scatter(embs_pca[:,0], embs_pca[:,1], c=y_test.detach().cpu().numpy(), cmap="coolwarm")

# %%
# PCA the neutrons
pca = PCA(n_components=2)
embs_pca = pca.fit_transform(neutrons.detach().cpu().numpy())
plt.scatter(embs_pca[:,0], embs_pca[:,1], c=y_test.detach().cpu().numpy(), cmap="coolwarm")
# %%

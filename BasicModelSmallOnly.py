import torch
from torch import nn
import matplotlib.pyplot as plt
#from base_functions import get_index

from train_arith_only import train
from copy import deepcopy
LIMIT = 53
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class BasicModelSmallOnly(nn.Module):
  #predicts a+b mod LIMIT 
  def __init__(self, n_a, n_b, hidden_dim):
    super().__init__()
    self.emb_a = nn.Embedding(n_a, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim), # we need 2*hidden_dim to get a and b embedding
      nn.ReLU(),
      nn.Linear(hidden_dim, LIMIT))
    self.emb_a.weight.data.uniform_(-1,1)
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    a = self.emb_a(x[:,0]) # [ batch_size, hidden_dim ]
    b = self.emb_a(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((a, b)))
    return x

  def plot_embedding(self):
    # plots a two pca of the embedding 
    p = torch.tensor(self.emb_a.weight.data)
    n = 2
    pca = PCA(n_components=n)
    embs_pca = pca.fit_transform(p.detach().cpu().numpy())

    pca_var = pca.fit(p.detach().cpu().numpy()).explained_variance_ratio_
    plt.xlabel(f'{n-1} dim: {100*pca_var[n-2]:.2f}% of variance')
    plt.ylabel(f'{n} dim: {100*pca_var[n-1]:.4f}% of variance')
    ap = range(LIMIT)

    first_dim = embs_pca[:, n-2].T
    second_dim = embs_pca[:, n-1].T
    plt.scatter(first_dim, second_dim, c=ap, cmap="coolwarm")
    plt.plot(first_dim, second_dim,c = 'k', linewidth = 0.05)
    for i, txt in enumerate(ap):
      plt.annotate(txt, (embs_pca[i,0], embs_pca[i,1]))
    plt.title('a+b only')
    plt.show()





if __name__ == '__main__':
    dir = 'mod_arith_only'
    embed_dim = 256
    seed = 0
    results = {'title': [], 'seed': [], 'test_size': [],'train_acc': [], 'test_acc': [], 'epochs': []}
    first_part = np.linspace(0.05,0.6, 12)
    first_part = np.repeat(first_part, 3)
    second_part = np.linspace(0.65, 0.9, 26)
    second_part = np.repeat(second_part, 5) 
    test_sizes = np.concatenate((first_part, second_part, [0.95]*3))
    for test_size in test_sizes:
          test_size = round(test_size, 2)
          print(test_size)
          seed += 1
          title = f'BasicModelSmall_256dim_{test_size}ts_{seed}seed_only'
          results['title'].append(title)
          results['seed'].append(seed)
          results['test_size'].append(test_size)
          test_acc, train_acc, epochs = train(modelclass=BasicModelSmallOnly, 
                          lr=1e-2, 
                          wd=1e-3,
                          embed_dim=embed_dim, 
                          basepath=f"models/{dir}/{title}/", 
                          device=torch.device("cuda"),
                          title = title,
                          seed = seed,
                          test_size=test_size
                          )
          results['test_acc'].append(test_acc)
          results['train_acc'].append(train_acc)
          results['epochs'].append(epochs)
          df = pd.DataFrame.from_dict(results)
          df.to_csv(dir+'_all.csv')

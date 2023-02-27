import torch
from torch import nn
import matplotlib.pyplot as plt
#from base_functions import get_index

from train_arith_both import train
from copy import deepcopy
limit = 53
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class BasicModelSmall(nn.Module):
  #predicts a+b mod 97
  def __init__(self, n_a, n_b, hidden_dim):
    super().__init__()
    # hidden dim is 64
    self.emb_a = nn.Embedding(n_a, hidden_dim) # [ batch_size, hidden_dim ]
    #self.emb_b = nn.Embedding(n_b, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim), # we need 2*hidden_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Linear(hidden_dim, 2*limit))
    self.emb_a.weight.data.uniform_(-1,1)
    #self.emb_b.weight.data.uniform_(-1,1)
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    a = self.emb_a(x[:,0]) # [ batch_size, hidden_dim ]
    b = self.emb_a(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((a, b)))
    return x
  
  def get_pca(self, X_test, layer = 'emb'):
     # [ batch_size, hidden_dim ]
    a = self.emb_a(X_test[:,0])
    b = self.emb_b(X_test[:,1])
    if layer == 'emb':
      U_p, S_p, Vh_p = torch.linalg.svd(a, False)
      U_n, S_n, Vh_n = torch.linalg.svd(b, False)
      return a, U_p, S_p, Vh_p, b, U_n, S_n, Vh_n
    if layer == 'last':
      x = torch.hstack((a, b))
      for i in range(len(self.nonlinear)-1):
        x = self.nonlinear[i](x)
      U, S, V = torch.linalg.svd(x, False)
      return x, U, S, V

  def plot_embedding(self):
    p = torch.tensor(self.emb_a.weight.data)
    n = 2
    pca = PCA(n_components=n)
    embs_pca = pca.fit_transform(p.detach().cpu().numpy())

    pca_var = pca.fit(p.detach().cpu().numpy()).explained_variance_ratio_
    plt.xlabel(f'{n-1} dim: {100*pca_var[n-2]:.2f}% of variance')
    plt.ylabel(f'{n} dim: {100*pca_var[n-1]:.4f}% of variance')
    ap = range(limit)

    first_dim = embs_pca[:, n-2].T
    second_dim = embs_pca[:, n-1].T
    plt.scatter(first_dim, second_dim, c=ap, cmap="coolwarm")
    plt.plot(first_dim, second_dim,c = 'k', linewidth = 0.05)
    for i, txt in enumerate(ap):
      plt.annotate(txt, (embs_pca[i,0], embs_pca[i,1]))
    plt.title('a+b and a-b')
    plt.show()


if __name__ == '__main__':
    dir = 'mod_arith_both'
    embed_dim = 256
    seed = 0
    results = {'title': [], 'seed': [], 'test_size': [],
               'train_amb_acc': [], 'train_apb_acc':[],
                'test_amb_acc':[], 'test_apb_acc':[],  'epochs': []}
     # reproducability
    title = f'BasicModelSmall_256dim_{0.2}ts_{seed}seed_both'
    test_apb_acc, test_amb_acc, train_apb_acc, train_amb_acc, epochs = train(modelclass=BasicModelSmall, 
                          lr=1e-2, 
                          wd=1e-3,
                          embed_dim=embed_dim, 
                          basepath=f"models/{dir}/{title}/", 
                          device=torch.device("cuda"),
                          title = title,
                          reg_pca = 0, 
                          seed = seed,
                          test_size=0.2
                          )
    '''
    for test_size in np.linspace(0.6, 0.95, 36):
        for _ in range(3):
          test_size = round(test_size, 2)
          print(test_size)
          seed+=1
          title = f'BasicModelSmall_256dim_{test_size}ts_{seed}seed_both'
          results['title'].append(title)
          results['seed'].append(seed)
          results['test_size'].append(test_size)
          test_apb_acc, test_amb_acc, train_apb_acc, train_amb_acc, epochs = train(modelclass=BasicModelSmall, 
                          lr=1e-2, 
                          wd=1e-3,
                          embed_dim=embed_dim, 
                          basepath=f"models/{dir}/{title}/", 
                          device=torch.device("cuda"),
                          title = title,
                          reg_pca = 0, 
                          seed = seed,
                          test_size=test_size
                          )
          results['test_apb_acc'].append(test_apb_acc)
          results['test_amb_acc'].append(test_amb_acc)
          results['train_apb_acc'].append(train_apb_acc)
          results['train_amb_acc'].append(train_amb_acc)
          results['epochs'] = epochs
          df = pd.DataFrame.from_dict(results)
          df.to_csv(dir+'_fine.csv')
    '''
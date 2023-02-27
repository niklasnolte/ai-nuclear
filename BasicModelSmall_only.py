import torch
from torch import nn
import matplotlib.pyplot as plt
#from base_functions import get_index

from train_arith_only import train
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
      nn.Linear(hidden_dim, limit))
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
    print(p.shape)
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
    plt.title('a+b and a-b; 0.6 test prop; 1e4 epochs')
    plt.show()

  def evaluate_ndim(self, loss_fn, X_test, y_test, device = 'cuda', n = 2): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    a, U_p, S_p, Vh_p, b, U_n, S_n, Vh_n = self.get_pca(X_test)

    mask_ndim = torch.eye(S_p.shape[0]).to(device)
    mask_ndim[n:] = 0

    a_ndim = a @ Vh_p.T @ mask_ndim @ Vh_p
    b_ndim = b @ Vh_n.T @ mask_ndim @ Vh_n

    y_pred = torch.flatten(self.nonlinear(torch.hstack((a_ndim, b_ndim))))
    return loss_fn(y_test, y_pred)

    

  def stochastic_pca_loss(self, loss_fn, X_test, y_test, device = 'cuda'):
    hidden_dim = self.emb_a.weight.shape[1]
    n = get_index(hidden_dim = hidden_dim, plot_dist=False)
    loss = self.evaluate_ndim(loss_fn, X_test, y_test, n = n, device = device)
    return loss



if __name__ == '__main__':
    dir = 'mod_arith_only'
    embed_dim = 256
    seed = 0
    results = {'title': [], 'seed': [], 'test_size': [],'train_acc': [], 'test_acc': [], 'epochs': []}
    for test_size in np.linspace(0.6, 0.95, 36):
        for _ in range(3):
          test_size = round(test_size, 2)
          print(test_size)
          seed += 1
          title = f'BasicModelSmall_256dim_{test_size}ts_{seed}seed_only'
          results['title'].append(title)
          results['seed'].append(seed)
          results['test_size'].append(test_size)
          test_acc, train_acc, epochs = train(modelclass=BasicModelSmall, 
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
          results['test_acc'].append(test_acc)
          results['train_acc'].append(train_acc)
          results['epochs'].append(epochs)
          df = pd.DataFrame.from_dict(results)
          df.to_csv(dir+'_fine.csv')
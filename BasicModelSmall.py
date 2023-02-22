import torch
from torch import nn
import matplotlib.pyplot as plt
#from base_functions import get_index
from train_arithmetic import train
from copy import deepcopy
limit = 53

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

  def evaluate_ndim(self, loss_fn, X_test, y_test, device = 'cuda', n = 2): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    a, U_p, S_p, Vh_p, b, U_n, S_n, Vh_n = self.get_pca(X_test)

    mask_ndim = torch.eye(S_p.shape[0]).to(device)
    mask_ndim[n:] = 0

    a_ndim = a @ Vh_p.T @ mask_ndim @ Vh_p
    b_ndim = b @ Vh_n.T @ mask_ndim @ Vh_n

    y_pred = torch.flatten(self.nonlinear(torch.hstack((a_ndim, b_ndim))))
    print(y_test.shape, y_pred.shape, 'ndim')
    return loss_fn(y_test, y_pred)

    

  def stochastic_pca_loss(self, loss_fn, X_test, y_test, device = 'cuda'):
    hidden_dim = self.emb_a.weight.shape[1]
    n = get_index(hidden_dim = hidden_dim, plot_dist=False)
    loss = self.evaluate_ndim(loss_fn, X_test, y_test, n = n, device = device)
    return loss



if __name__ == '__main__':
    dir = 'mod_arith'
    for embed_dim in [256]:#, 2e-4, 2e-3, 2e-2, 2e-1, 2e0, 5e0]:
        title = f'BasicModelSmall_regpca0_{embed_dim}dim'
        print(title)
        train(modelclass=BasicModelSmall, 
                        lr=1e-2, 
                        wd=1e-3, 
                        embed_dim=embed_dim, 
                        basepath=f"models/{dir}/{title}/", 
                        device=torch.device("cuda"),
                        title = title,
                        reg_pca = 0, 
                        )
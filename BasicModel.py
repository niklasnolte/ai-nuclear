import torch
from torch import nn
import matplotlib.pyplot as plt
from base_functions import get_index
from train_model import train


class BasicModel(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim):
    super().__init__()
    # hidden dim is 64
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim), # we need 2*hidden_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1))
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)
    

class BasicModelSmall(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim):
    super().__init__()
    # hidden dim is 64
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim), # we need 2*hidden_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Linear(hidden_dim, 1))
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

  
class BasicModelSmaller(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim):
    super().__init__()
    # hidden dim is 64
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, 16), # we need 2*hidden_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Linear(16, 1))
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

  def get_pca(self, X_test):
     # [ batch_size, hidden_dim ]
    proton = self.emb_proton(X_test[:,0])
    neutron = self.emb_neutron(X_test[:,1])

    U_p, S_p, Vh_p = torch.linalg.svd(proton, False)
    U_n, S_n, Vh_n = torch.linalg.svd(neutron, False)
    return proton, U_p, S_p, Vh_p, neutron, U_n, S_n, Vh_n

  def evaluate_ndim(self, loss_fn, X_test, y_test, device = 'cuda', n = 2): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton, U_p, S_p, Vh_p, neutron, U_n, S_n, Vh_n = self.get_pca(X_test)

    mask_ndim = torch.eye(S_p.shape[0]).to(device)
    mask_ndim[n:] = 0

    proton_ndim = proton @ Vh_p.T @ mask_ndim @ Vh_p
    neutron_ndim = neutron @ Vh_n.T @ mask_ndim @ Vh_n

    y_pred = self.nonlinear(torch.hstack((proton_ndim, neutron_ndim)))
    return loss_fn(y_test, y_pred)

    

  def alldims_loss(self, loss_fn, X_test, y_test, device = 'cuda'):
    hidden_dim = self.emb_proton.weight.shape[1]
    total_loss = 0
    num_dims = hidden_dim
    proton, U_p, S_p, Vh_p, neutron, U_n, S_n, Vh_n = self.get_pca(X_test)
    steps = 0
    for n in range(1, 1+num_dims, 3):
      mask_ndim = torch.eye(S_p.shape[0]).to(device)
      mask_ndim[n:] = 0
      steps+=1

      proton_ndim = proton @ Vh_p.T @ mask_ndim @ Vh_p
      neutron_ndim = neutron @ Vh_n.T @ mask_ndim @ Vh_n

      y_pred = self.nonlinear(torch.hstack((proton_ndim, neutron_ndim)))
      total_loss+=loss_fn(y_test, y_pred)
    return total_loss/steps

  def stochastic_pca_loss(self, loss_fn, X_test, y_test, device = 'cuda'):
    hidden_dim = self.emb_proton.weight.shape[1]
    n = get_index(hidden_dim = hidden_dim, plot_dist=False)
    loss = self.evaluate_ndim(loss_fn, X_test, y_test, n = n, device = device)
    return loss


class BasicModelReallySmall(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim):
    super().__init__()
    # hidden dim is 64
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, 4), # we need 2*hidden_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Linear(4, 1))
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

class BasicModelSmallest(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim):
    super().__init__()
    # hidden dim is 64
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, 1), # we need 2*hidden_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Linear(1, 1))
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

class BasicLinear(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim):
    super().__init__()
    # hidden dim is 256
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.linear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, 1)) # we need 2*hidden_dim to get proton and neutron embedding
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.linear(torch.hstack((proton, neutron)))
    return x
    


if __name__ == '__main__':

  regvals = [2e0, 2e-1, 2e-2, 2e-3, 2e-4, 0]
  regtypes = ['dimall', 'dimn', 'oldeff', 'dim3', 'dim6']
  dim = 64
  for j in range(len(regtypes)):
    for i in range(len(regvals)):
      regtype = regtypes[j]
      regval = regvals[i]
      title = f'BasicModelSmaller_regpca{regval}_{regtype}'
      print(f'\n{title}')
      print(regtype, regval)
      train(modelclass=BasicModelSmaller, 
              lr=(2e-3)/4, 
              wd=1e-4, 
              embed_dim=dim, 
              basepath=f"models/pcareg_heavy15/{title}/", 
              device=torch.device("cuda"),
              title = title,
              reg_pca = regval, 
              reg_type = regtype,
              )
  
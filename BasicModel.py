import torch
from torch import nn
import matplotlib.pyplot as plt
from base_functions import get_index
from train_model import train
from copy import deepcopy


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

  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

  def alldims_lastlayer(self, loss_fn, X_test, y_test):
    x, U, S, V = self.get_pca(X_test, layer = 'last')
    y_pred = self.nonlinear[-1](x)
    loss = loss_fn(y_pred, y_test).item()
    losses = []
    with torch.no_grad():
      for i in range(S.shape[0]):
        index = S.shape[0]-i
        S[index:] = 0
        x_mask = U @ torch.diag(S) @ V
        y_predmask = self.nonlinear[-1](x_mask)
        losses.append(loss_fn(y_predmask, y_test))
      losses = losses[::-1]
      return loss, [l.item() for l in losses]

  def get_pca(self, X_test, layer = 'emb'):
     # [ batch_size, hidden_dim ]
    proton = self.emb_proton(X_test[:,0])
    neutron = self.emb_neutron(X_test[:,1])
    if layer == 'emb':
      U_p, S_p, Vh_p = torch.linalg.svd(proton, False)
      U_n, S_n, Vh_n = torch.linalg.svd(neutron, False)
      return proton, U_p, S_p, Vh_p, neutron, U_n, S_n, Vh_n
    if layer == 'last':
      x = torch.hstack((proton, neutron))
      for i in range(len(self.nonlinear)-1):
        x = self.nonlinear[i](x)
      U, S, V = torch.linalg.svd(x, False)
      return x, U, S, V

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
  
  def alldims_lastlayer(self, loss_fn, X_test, y_test):
    x, U, S, V = self.get_pca(X_test, layer = 'last')
    y_pred = self.nonlinear[-1](x)
    loss = loss_fn(y_pred, y_test).item()
    losses = []
    with torch.no_grad():
      for i in range(S.shape[0]):
        index = S.shape[0]-i
        S[index:] = 0
        x_mask = U @ torch.diag(S) @ V
        y_predmask = self.nonlinear[-1](x_mask)
        losses.append(loss_fn(y_predmask, y_test))
      losses = losses[::-1]
      return loss, [l.item() for l in losses]

  def get_pca(self, X_test, layer = 'emb'):
     # [ batch_size, hidden_dim ]
    proton = self.emb_proton(X_test[:,0])
    neutron = self.emb_neutron(X_test[:,1])
    if layer == 'emb':
      U_p, S_p, Vh_p = torch.linalg.svd(proton, False)
      U_n, S_n, Vh_n = torch.linalg.svd(neutron, False)
      return proton, U_p, S_p, Vh_p, neutron, U_n, S_n, Vh_n
    if layer == 'last':
      x = torch.hstack((proton, neutron))
      for i in range(len(self.nonlinear)-1):
        x = self.nonlinear[i](x)
      U, S, V = torch.linalg.svd(x, False)
      return x, U, S, V

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


class BasicModelSmallerDropout(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim):
    super().__init__()
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, 16), # we need 2*hidden_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Dropout(p=0.001),
      nn.Linear(16, 1))
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)

  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x


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
  dir = 'pcareg_heavy15'
  for regpca in [2e-4, 2e-3, 2e-2, 2e-1, 2e0, 5e0]:
    title = f'BasicModelSmall_regpca{regpca}_embed256_dimn'
    train(modelclass=BasicModelSmall, 
                    lr=(2e-3)/4, 
                    wd=1e-4, 
                    embed_dim=256, 
                    basepath=f"models/{dir}/{title}/", 
                    device=torch.device("cuda"),
                    title = title,
                    reg_pca = regpca, 
                    reg_type = 'dimn'
                    )
  title = 'BasicModelSmallerDropout_regpca0_dimn'
  train(modelclass=BasicModelSmall, 
                    lr=(2e-3)/4, 
                    wd=1e-4, 
                    embed_dim=64, 
                    basepath=f"models/{dir}/{title}/", 
                    device=torch.device("cuda"),
                    title = title,
                    reg_pca = 0, 
                    reg_type = 'dimn'
                    )            
  '''
  
  regtypes = ['dimn', 'dimall', 'oldeff', 'dim3'][:1]

  dim = 64
  models = ['BasicModelSmall', 'BasicModel']
  model = models[0]
  norms = [False, True][:1]
  for regtype in regtypes:
    for norm in norms:
      if norm:
        regvals = [0, 0.1, 0.5, 1, 5]
      else:
        regvals = [5e0, 3.5, 2.5, 2e0, 2e-1, 2e-2, 2e-3, 2e-4, 0]
        regvals = [0]
      for lr in [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
        for regval in regvals:
          if norm:
            dir = 'normreg_heavy15'
          else:
            dir = 'pcareg_heavy15'
          if model == 'BasicModelSmall':
            title = f'BasicModelSmall_regpca{regval}_{regtype}_lr{lr}'
            print(f'\n{title}')
            print(regtype, norm, model, regval)
            train(modelclass=BasicModelSmall, 
                    lr=lr,#(2e-3)/4, 
                    wd=1e-4, 
                    embed_dim=dim, 
                    basepath=f"models/{dir}/{title}/", 
                    device=torch.device("cuda"),
                    title = title,
                    reg_pca = regval, 
                    reg_type = regtype,
                    norm = norm
                    )
          elif model == 'BasicModel':
            title = f'BasicModel_regpca{regval}_{regtype}'
            print(f'\n{title}')
            print(regtype, regval)
            train(modelclass=BasicModel, 
                    lr=(2e-3)/4, 
                    wd=1e-4, 
                    embed_dim=dim, 
                    basepath=f"models/{dir}/{title}/", 
                    device=torch.device("cuda"),
                    title = title,
                    reg_pca = regval, 
                    reg_type = regtype,
                    norm = norm
                    )
  '''

    
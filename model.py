import torch
from torch import nn

class Model(nn.Module):
  def __init__(self, n_inputs, hidden_dim):
    super().__init__()
    self.emb = nn.Embedding(n_inputs, hidden_dim) # [ batch_size, 2, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, 1))
    #bigger init
    self.emb.weight.data.uniform_(-10,10)
    
  def forward(self, x): # x: [ batch_size, 2 ]
    x = self.emb(x) # [ batch_size, 2, hidden_dim ]
    x = self.nonlinear(x)
    return x

class Model2(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim, n_obs):
    super().__init__()
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Dropout(p=0.001),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      #nn.Dropout(0.001),
      nn.Linear(hidden_dim, n_obs))
    #bigger init
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)
    
    self.emb_proton.weight = nn.Parameter(nn.functional.normalize(self.emb_proton.weight, p = 2, dim = 1))
    self.emb_neutron.weight = nn.Parameter(nn.functional.normalize(self.emb_neutron.weight, p = 2, dim = 1))
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    
    proton = nn.functional.normalize(proton, p = 2, dim = 1)
    neutron = nn.functional.normalize(neutron, p = 2, dim = 1)
    
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

class Model3(nn.Module):
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
    #bigger init
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)

  # def get_pca(self, X_test):
  #    # [ batch_size, hidden_dim ]
  #   proton = self.emb_proton(X_test[:,0])
  #   neutron = self.emb_neutron(X_test[:,1])

  #   U_p, S_p, Vh_p = torch.linalg.svd(proton, False)
  #   U_n, S_n, Vh_n = torch.linalg.svd(neutron, False)
  #   return proton, U_p, S_p, Vh_p, neutron, U_n, S_n, Vh_n

    # def alldims_loss(self, loss_fn, X_test, y_test, device = 'cpu'):
    #   num_dims = self.emb_proton.weight.shape[1]
    #   total_loss = 0
    #   proton, U_p, S_p, Vh_p, neutron, U_n, S_n, Vh_n = self.get_pca(X_test)
    #   steps = 0
    #   for n in range(1, 1+num_dims, 3):
    #     mask_ndim = torch.eye(S_p.shape[0]).to(device)
    #     mask_ndim[n:] = 0
    #     steps+=1
  
    #     proton_ndim = proton @ Vh_p.T @ mask_ndim @ Vh_p
    #     neutron_ndim = neutron @ Vh_n.T @ mask_ndim @ Vh_n
  
    #     y_pred = self.nonlinear(torch.hstack((proton_ndim, neutron_ndim)))
    #     total_loss+=loss_fn(y_test, y_pred)
    #   return total_loss/steps

  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

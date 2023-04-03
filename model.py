import torch
from torch import nn
import math

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
      nn.BatchNorm1d(hidden_dim),
      nn.Dropout(p=0.01),
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


class Model_multi(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim, n_obs):
    super().__init__()
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
        nn.Flatten(),
        nn.Linear(2*hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.BatchNorm1d(hidden_dim))
    obs_nn = []
    for i in range(n_obs):
        obs_nn.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim//n_obs),
        nn.ReLU(),nn.Linear(hidden_dim//n_obs, 1)))
    self.obs_nn = obs_nn
    # self.linear = nn.Linear(hidden_dim//n_obs*n_obs,n_obs)
    
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)
    
    self.emb_proton.weight = nn.Parameter(nn.functional.normalize(self.emb_proton.weight, p = 2, dim = 1))
    self.emb_neutron.weight = nn.Parameter(nn.functional.normalize(self.emb_neutron.weight, p = 2, dim = 1))
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    
    x = self.nonlinear(torch.hstack((proton, neutron)))
    x = [obsi(x) for obsi in self.obs_nn]
    x = torch.hstack(x)
  
    return x
      
class Model3(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim, n_obs):
    super().__init__()
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.LayerNorm(hidden_dim),
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

class Model22(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim, n_obs):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]    
    # Positional encoding function
    self.pos_proton = nn.Parameter(self.get_positional_encoding(n_protons, hidden_dim), requires_grad=False)
    self.pos_neutron = nn.Parameter(self.get_positional_encoding(n_neutrons, hidden_dim), requires_grad=False)
    
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.BatchNorm1d(hidden_dim),
      #nn.Dropout(p=0.01),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      #nn.Dropout(0.001),
      nn.Linear(hidden_dim, n_obs))
    
    #bigger init
    self.pos_proton.data.uniform_(-1,1)
    self.pos_neutron.data.uniform_(-1,1)
    
  def get_positional_encoding(self, seq_len, hidden_size):
    # Compute the positional encoding for a given sequence length and hidden size
    encoding = torch.zeros(seq_len, hidden_size)
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float) * (-math.log(10000.0) / hidden_size))
    encoding[:, 0::2] = torch.sin(pos * div)
    encoding[:, 1::2] = torch.cos(pos * div)
    return encoding
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    p_enc = nn.functional.normalize(self.pos_proton[x[:,0].long()], p=2, dim=-1) # [ batch_size, n_neutrons, hidden_dim ]
    n_enc = nn.functional.normalize(self.pos_neutron[x[:,1].long()], p=2, dim=-1) # [ batch_size, n_neutrons, hidden_dim ]
    proton = p_enc # [ batch_size, hidden_dim ]
    neutron = n_enc # [ batch_size, hidden_dim ]
    
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x
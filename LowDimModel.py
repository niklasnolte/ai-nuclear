import torch
from torch import nn
import matplotlib.pyplot as plt
from train_model import train

class TwoDimModel(nn.Module):
  def __init__(self, n_protons, n_neutrons, hidden_dim):
    super().__init__()
    # hidden dim is 64
    self.emb_proton = nn.Embedding(n_protons, hidden_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, hidden_dim) # [ batch_size, hidden_dim ]
    interim_dim = 2

    self.proton_network = nn.Sequential(
      nn.Flatten(),
      nn.Linear(hidden_dim, hidden_dim), 
      nn.ReLU(),
      nn.Linear(hidden_dim, interim_dim))
    self.neutron_network = nn.Sequential(
      nn.Flatten(),
      nn.Linear(hidden_dim, hidden_dim), 
      nn.ReLU(),
      nn.Linear(hidden_dim, interim_dim))

    self.combined = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*interim_dim, interim_dim), # we need 2*interim_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Linear(interim_dim, 1)) 
    
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)

  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton_emb = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron_emb = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    proton = self.proton_network(proton_emb)
    neutron = self.neutron_network(neutron_emb)
    x = self.combined(torch.hstack((proton, neutron)))
    return x

if __name__ == '__main__':
    dim = 64
    dir = 'LowDimModel'
    title = 'twodimmodel'
    train(modelclass=TwoDimModel, 
            lr = (2e-3)/4, 
            wd=1e-4, 
            embed_dim=dim, 
            basepath=f"models/{dir}/{title}/", 
            device=torch.device("cuda"),
            title = title,
            reg_pca = 0, 
            reg_type = None,
            norm = False
            )
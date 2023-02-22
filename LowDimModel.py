import torch
from torch import nn
import matplotlib.pyplot as plt
from train_model import train
from data import get_data
from base_functions import get_models, test_model
import numpy as np
import torch

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
      nn.Linear(2*interim_dim, hidden_dim), # we need 2*interim_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Linear(hidden_dim, 1)) 
    
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)

  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton, neutron = self.get_reduced_embedding(x)
    x = self.combined(torch.hstack((proton, neutron)))
    return x

  def get_reduced_embedding(self, x):
    proton_emb = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron_emb = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    proton = self.proton_network(proton_emb)
    neutron = self.neutron_network(neutron_emb)
    return proton, neutron

class TwoDimModelRaw(nn.Module):
  def __init__(self, n_protons, n_neutrons, interim_dim = 2):
    super().__init__()
    # hidden dim is 64
    self.emb_proton = nn.Embedding(n_protons, interim_dim) # [ batch_size, hidden_dim ]
    self.emb_neutron = nn.Embedding(n_neutrons, interim_dim) # [ batch_size, hidden_dim ]
    hidden_dim = 64

    self.combined = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*interim_dim, hidden_dim), # we need 2*interim_dim to get proton and neutron embedding
      nn.ReLU(),
      nn.Linear(hidden_dim, 1)) 
    
    self.emb_proton.weight.data.uniform_(-1,1)
    self.emb_neutron.weight.data.uniform_(-1,1)

  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton, neutron = self.get_reduced_embedding(x)
    x = self.combined(torch.hstack((proton, neutron)))
    return x

  def get_reduced_embedding(self, x):
    proton_emb = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron_emb = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    return proton_emb, neutron_emb


def twodim_visualization(model, title = '', pn = 'proton', heavy_elem = 15):
    #plots the two dim embeddings of protons for every data point to see pattern
    X_train, X_test, y_train, y_test, vocab_size = get_data(heavy_elem = heavy_elem)
    X = torch.vstack((X_train,X_test))
    i  = 0 if pn == 'proton' else 1
    min_val = 15 if pn == 'proton' else 10 #where we consider an element to be too light
    all_nums = list(range(vocab_size[i]))
    if i == 0:
        vals = torch.tensor([[p, 0] for p in all_nums])
    else:
        vals = torch.tensor([[0, p] for p in all_nums])
    
    embeds = model.get_reduced_embedding(vals)[i]
    print(len(embeds))
    print(embeds[0].shape)
    plt.scatter(*embeds.T,  c=all_nums, cmap="coolwarm")
    plt.plot(*embeds.T,c = 'k', linewidth = 0.2)
    
    for i, txt in enumerate(all_nums):
        plt.annotate(min_val+txt, (embeds[i,0], embeds[i,1]))
    loss = test_model(model, X_test, y_test)
    plt.title(f'Low Dim Model 2 Dim\n{title}\n{pn} representation\ntest loss = {loss:.4f}')
    plt.show()

if __name__ == '__main__':
    for seed in [31]:#, 1,25,30, 31, 50]:
      model = get_models([f'models/pcareg_seeds/TwoDimModel_seed{seed}/'])[0]


      twodim_visualization(model, title = f'Seed {seed}', pn = 'proton')

    '''
    
    dim = 64
    dir = 'pcareg_seeds'
    
    for seed in [1, 25, 30, 31, 50]:
      title = f'TwoDimModel_seed{seed}'
      train(modelclass=TwoDimModel, 
              lr = (2e-3)/4, 
              wd=1e-4, 
              embed_dim=dim, 
              basepath=f"models/{dir}/{title}/", 
              device=torch.device("cuda"),
              title = title,
              seed = seed,
              reg_pca = 0, 
              reg_type = None,
              norm = False
              )
    '''
    
    

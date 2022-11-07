import torch
from torch import nn
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
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]
    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

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

  regs = [2e-3, 2e-2, 1e-1, 1, 0]
  vals = ['2e_3', '2e_2', '1e_1', '1', '0']
  heavy = 15
  dim = 1024
  for i in range(len(regs)):
    reg = regs[i]
    val = vals[i]
    title = f'BasicLinear_reg{val}_heavy{heavy}_dim{dim}'
    train(modelclass=BasicLinear, 
        lr=(2e-3)/4, 
        wd=1e-4, 
        embed_dim=dim, 
        basepath=f"models/Basic/{title}/", 
        device=torch.device("cuda"),
        title = title,
        heavy_elem = heavy,
        reg_effective=reg
        )


  '''
  model = BasicModel(all_protons.shape[0], all_neutrons.shape[0],20)
  print(model.state_dict()['emb_proton.weight'][0])
  

  nd_loss = nd_loss(model, all_protons, all_neutrons, X_test, y_test, n=1)
  print(model.state_dict()['emb_proton.weight'][0])
  print(nd_loss)
  

  
  title = 'BasicModel_reg2d2e_2_heavy15'
  train(modelclass=BasicModel, 
        lr=(2e-3)/4, 
        wd=1e-4, 
        embed_dim=64, 
        basepath=f"models/Basic/{title}/", 
        device=torch.device("cuda"),
        title = title,
        heavy_elem = 15,
        reg_actual  = 1,
        reg_actual_n = 2
        )
  
  
  title = 'BasicModelSmall_reg2e_2_heavy'
  train(modelclass=BasicModelSmall, 
          lr=(2e-3)/4, 
          wd=1e-4, 
          embed_dim=64, 
          basepath=f"models/Basic/{title}/", 
          device=torch.device("cuda"),
          title = title,
          heavy_elem = 15,
          regularize = 2e-2
          )
  title = 'BasicModelSmall_reg2e_2_pt2'
  train(modelclass=BasicModelSmall, 
          lr=(2e-3)/4, 
          wd=1e-4, 
          embed_dim=64, 
          basepath=f"models/Basic/{title}/", 
          device=torch.device("cuda"),
          title = title,
          regularize = 2e-2
          )

  
  title = 'BasicModelSmaller_dim2'
  train(modelclass=BasicModelSmaller, 
        lr=(2e-3)/4, 
        wd=1e-4, 
        embed_dim=2, 
        basepath=f"models/Basic/{title}/", 
        device=torch.device("cuda"),
        title = title,
        regularize = False

        )
  
  title = 'BasicModelReallySmall_dim2'
  train(modelclass=BasicModelReallySmall, 
        lr=(2e-3)/4, 
        wd=1e-4, 
        embed_dim=2, 
        basepath=f"models/Basic/{title}/", 
        device=torch.device("cuda"),
        title = title,
        regularize = False
        )
  title = 'BasicModelSmallest_dim2'
  train(modelclass=BasicModelSmallest, 
        lr=(2e-3)/4, 
        wd=1e-4, 
        embed_dim=2, 
        basepath=f"models/Basic/{title}/", 
        device=torch.device("cuda"),
        title = title,
        regularize = False
        )

  '''

import torch
from torch import nn
from train_model import train




class Normalize(nn.Module):
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

    self.emb_proton.weight = nn.Parameter(nn.functional.normalize(self.emb_proton.weight, p = 2, dim = 1))
    self.emb_neutron.weight = nn.Parameter(nn.functional.normalize(self.emb_neutron.weight, p = 2, dim = 1))
    
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]

    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]

    proton = nn.functional.normalize(proton, p = 2, dim = 1)
    neutron = nn.functional.normalize(neutron, p = 2, dim = 1)

    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

class NormalizeSmall(nn.Module):
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

    self.emb_proton.weight = nn.Parameter(nn.functional.normalize(self.emb_proton.weight, p = 2, dim = 1))
    self.emb_neutron.weight = nn.Parameter(nn.functional.normalize(self.emb_neutron.weight, p = 2, dim = 1))

    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]

    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]

    proton = nn.functional.normalize(proton, p = 2, dim = 1)
    neutron = nn.functional.normalize(neutron, p = 2, dim = 1)

    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x
  
class NormalizeSmaller(nn.Module):
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
    
    self.emb_proton.weight = nn.Parameter(nn.functional.normalize(self.emb_proton.weight, p = 2, dim = 1))
    self.emb_neutron.weight = nn.Parameter(nn.functional.normalize(self.emb_neutron.weight, p = 2, dim = 1))
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]

    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]

    proton = nn.functional.normalize(proton, p = 2, dim = 1)
    neutron = nn.functional.normalize(neutron, p = 2, dim = 1)

    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x


class NormalizeReallySmall(nn.Module):
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

    self.emb_proton.weight = nn.Parameter(nn.functional.normalize(self.emb_proton.weight, p = 2, dim = 1))
    self.emb_neutron.weight = nn.Parameter(nn.functional.normalize(self.emb_neutron.weight, p = 2, dim = 1))
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]

    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]

    proton = nn.functional.normalize(proton, p = 2, dim = 1)
    neutron = nn.functional.normalize(neutron, p = 2, dim = 1)

    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

class NormalizeSmallest(nn.Module):
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

    self.emb_proton.weight = nn.Parameter(nn.functional.normalize(self.emb_proton.weight, p = 2, dim = 1))
    self.emb_neutron.weight = nn.Parameter(nn.functional.normalize(self.emb_neutron.weight, p = 2, dim = 1))
    
  def forward(self, x): # x: [ batch_size, 2 [n_protons, n_neutrons] ]

    proton = self.emb_proton(x[:,0]) # [ batch_size, hidden_dim ]
    neutron = self.emb_neutron(x[:,1]) # [ batch_size, hidden_dim ]

    proton = nn.functional.normalize(proton, p = 2, dim = 1)
    neutron = nn.functional.normalize(neutron, p = 2, dim = 1)

    x = self.nonlinear(torch.hstack((proton, neutron)))
    return x

    


if __name__ == '__main__':
    
    '''
    title = 'NormalizeSmall_reg2e_2'
    train(modelclass=NormalizeSmall, 
        lr=(2e-3)/4, 
        wd=1e-4, 
        embed_dim=64, 
        basepath=f"models/normalize/{title}/", 
        device=torch.device("cuda"),
        title = title,
        regularize = 2e-2
    )
    '''
    title = 'NormalizeReallySmall_reg2e_5'
    train(modelclass=NormalizeReallySmall, 
            lr=(2e-3)/4, 
            wd=1e-4, 
            embed_dim=64, 
            basepath=f"models/normalize/{title}/", 
            device=torch.device("cuda"),
            title = title,
            regularize = 2e-5
            )

    title = 'NormalizeReallySmall_reg2e_4'
    train(modelclass=NormalizeReallySmall, 
            lr=(2e-3)/4, 
            wd=1e-4, 
            embed_dim=64, 
            basepath=f"models/normalize/{title}/", 
            device=torch.device("cuda"),
            title = title,
            regularize = 2e-4
            )

    title = 'NormalizeReallySmall_reg2e_3'
    train(modelclass=NormalizeReallySmall, 
            lr=(2e-3)/4, 
            wd=1e-4, 
            embed_dim=64, 
            basepath=f"models/normalize/{title}/", 
            device=torch.device("cuda"),
            title = title,
            regularize = 2e-3
            )

    title = 'NormalizeReallySmall_reg2e_2'
    train(modelclass=NormalizeReallySmall, 
            lr=(2e-3)/4, 
            wd=1e-4, 
            embed_dim=64, 
            basepath=f"models/normalize/{title}/", 
            device=torch.device("cuda"),
            title = title,
            regularize = 2e-2
            )

    title = 'NormalizeReallySmall_reg1e_1'
    train(modelclass=NormalizeReallySmall, 
            lr=(2e-3)/4, 
            wd=1e-4, 
            embed_dim=64, 
            basepath=f"models/normalize/{title}/", 
            device=torch.device("cuda"),
            title = title,
            regularize = 1e-1
            )

    
    title = 'NormalizeReallySmall_noreg'
    train(modelclass=NormalizeReallySmall, 
            lr=(2e-3)/4, 
            wd=1e-4, 
            embed_dim=64, 
            basepath=f"models/normalize/{title}/", 
            device=torch.device("cuda"),
            title = title,
            regularize = False
            )
    



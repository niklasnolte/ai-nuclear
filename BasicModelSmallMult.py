import torch
from torch import nn
import matplotlib.pyplot as plt

from train_arith_mult import train
from copy import deepcopy
LIMIT = 53
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class BasicModelSmallMult(nn.Module):
  #predicts a+b, a-b, and a*b mod LIMIT
  def __init__(self, n_a, n_b, hidden_dim):
    super().__init__()
    self.emb_a = nn.Embedding(n_a, hidden_dim)
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim), # we need 2*hidden_dim to get a and b embedding
      nn.ReLU(),
      nn.Linear(hidden_dim, 3*LIMIT))
    self.emb_a.weight.data.uniform_(-1,1)
    
  def forward(self, x): # x: [ batch_size, 2 [n_a, n_b] ]
    a = self.emb_a(x[:,0]) # [ batch_size, hidden_dim ]
    b = self.emb_a(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((a, b)))
    return x

  def plot_embedding(self):
    #plots a two pca of the embedding 
    p = torch.tensor(self.emb_a.weight.data)
    n = 2
    pca = PCA(n_components=n)
    embs_pca = pca.fit_transform(p.detach().cpu().numpy())

    pca_var = pca.fit(p.detach().cpu().numpy()).explained_variance_ratio_
    plt.xlabel(f'{n-1} dim: {100*pca_var[n-2]:.2f}% of variance')
    plt.ylabel(f'{n} dim: {100*pca_var[n-1]:.4f}% of variance')
    ap = range(LIMIT)

    first_dim = embs_pca[:, n-2].T
    second_dim = embs_pca[:, n-1].T
    plt.scatter(first_dim, second_dim, c=ap, cmap="coolwarm")
    plt.plot(first_dim, second_dim,c = 'k', linewidth = 0.05)
    for i, txt in enumerate(ap):
      plt.annotate(txt, (embs_pca[i,0], embs_pca[i,1]))
    plt.title('a+b and a-b')
    plt.show()


def run_models(dir, test_sizes, run = 'both'):
  results = {'title': [], 'seed': [], 'test_size': [], 'epochs': []}
  mult = ['test_apb_acc', 'test_amb_acc', 'test_atb_acc', 'test_acc', 'train_acc']
  #both = ['test_apb_acc', 'test_amb_acc', 'train_apb_acc', 'train_amb_acc']
  for metric in mult:
     results[metric] = []
  seed = 0
  embed_dim = 256
  for test_size in test_sizes:
          test_size = round(test_size, 2)
          print(test_size)
          seed+=1
          title = f'BasicModelSmallMult_256dim_{test_size}ts_{seed}seed_{run}'
          results['title'].append(title)
          results['seed'].append(seed)
          results['test_size'].append(test_size)

          values = train(modelclass=BasicModelSmallMult, 
                          lr=1e-2, 
                          wd=1e-3,
                          embed_dim=embed_dim, 
                          basepath=f"models/{dir}/{title}/", 
                          device=torch.device("cuda"),
                          title = title,
                          seed = seed,
                          test_size=test_size,
                          )
          for i in range(len(values) - 1):
             value = values[i]
             results[mult[i]].append(value)
          results['epochs'] = values[-1]
          df = pd.DataFrame.from_dict(results)
          df.to_csv(dir+'.csv')

if __name__ == '__main__':
    run = 'mult'
    dir = f'mod_arith_{run}_apb_amb_atb'
    first_part = np.linspace(0.05,0.6, 12)
    first_part = np.repeat(first_part, 3)
    second_part = np.linspace(0.65, 0.9, 26)
    second_part = np.repeat(second_part, 5) 
    test_sizes = np.concatenate((first_part, second_part, [0.95]*3))
    run_models(dir, test_sizes, run = run)


    '''
    embed_dim = 256
    seed = 0
    
    title = f'BasicModelSmall_256dim_{0.2}ts_{seed}seed_both'
    test_apb_acc, test_amb_acc, train_apb_acc, train_amb_acc, epochs = train(modelclass=BasicModelSmallBoth, 
                          lr=1e-2, 
                          wd=1e-3,
                          embed_dim=embed_dim, 
                          basepath=f"models/{dir}/{title}/", 
                          device=torch.device("cuda"),
                          title = title,
                          seed = seed,
                          test_size=0.2
                          )
    '''
    '''
    
    '''
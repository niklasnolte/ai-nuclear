import numpy as np
import pandas as pd

import torch
from torch import nn
import matplotlib.pyplot as plt
from config import Config
from utils import functions_to_names, run_models 
from train import train


config = Config()
LIMIT = config.LIMIT

class BasicModel(nn.Module):
  #predicts a+b, a-b, and a*b mod LIMIT
  def __init__(self, functions, hidden_dim):
    super().__init__()
    self.emb_a = nn.Embedding(LIMIT, hidden_dim)
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim), # we need 2*hidden_dim to get a and b embedding
      nn.ReLU(),
      nn.Linear(hidden_dim, len(functions)*LIMIT))
    self.emb_a.weight.data.uniform_(-1,1)
    
  def forward(self, x): # x: [ batch_size, 2 [n_a, n_b] ]
    a = self.emb_a(x[:,0]) # [ batch_size, hidden_dim ]
    b = self.emb_a(x[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((a, b)))
    return x
  
def run_basicmodel(functions):
    seeds = [1, 2, 3]
    test_sizes = np.linspace(0.05, 0.95, 19)
    modelname = 'BasicModel'
    df = run_models(BasicModel, modelname, train, test_sizes, seeds, functions)
    df.to_csv(f'full_results/{modelname}/{modelname}.csv')
  
if __name__ == '__main__':
  functions = ['a+b', 'a-b', 'a*b']
  run_basicmodel(functions)
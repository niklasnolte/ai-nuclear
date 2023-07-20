import numpy as np
import pandas as pd

import torch
from torch import nn
import matplotlib.pyplot as plt
from config import Config
from utils import functions_to_names, run_models 
#from train import train
from TaskEmbModel import ResidualBlock,  random_search_parameters, train, TaskEmbModel, run_params
import re
from torch.optim import Adam, SGD, RMSprop


config = Config()
LIMIT = config.LIMIT

class BaselineModel(nn.Module):
  #predicts a+b, a-b, and a*b mod LIMIT
  def __init__(self, functions, hidden_dim, num_layers):
    super().__init__()
    self.emb_a = nn.Embedding(LIMIT, hidden_dim)
    self.nonlinear = nn.Sequential(
      nn.Flatten(),
      nn.Linear(2*hidden_dim, hidden_dim), # we need 2*hidden_dim to get a and b embedding
      *[ResidualBlock(hidden_dim) for _ in range(num_layers)],
      nn.ReLU(),
      nn.Linear(hidden_dim, len(functions)))
    
  def forward(self, input): # x: [ batch_size, 2 [n_a, n_b] ]
    a = self.emb_a(input[:,0]) # [ batch_size, hidden_dim ]
    b = self.emb_a(input[:,1]) # [ batch_size, hidden_dim ]
    x = self.nonlinear(torch.hstack((a, b)))
    indices = input[:,-1].long()
    row_indices = torch.arange(x.size(0)).unsqueeze(-2)[0].long()
    output = x[row_indices, indices]
    return output.view(-1,1)

  def __str__(self):
    return 'BaselineModel'
  
def run_basicmodel(functions):
    seeds = [1, 2, 3]
    test_sizes = np.linspace(0.05, 0.95, 19)
    modelname = BaselineModel.__name__
    df = run_models(BaselineModel, modelname, train, test_sizes, seeds, functions)
    df.to_csv(f'full_results/{modelname}/{modelname}.csv')
  
def extract_params_from_title(title):
  #title = f'{model_str}_fn{fn_name}_hd{hidden_dim}_nl{num_layers}_opt{optimizer.__name__}_bs{batch_size}__lr{lr}_wd{wd}_epochs{epochs}'

  match = re.match(
    r'^(?P<model_str>.*?)_fn(?P<fn_name>.*?)_hd(?P<hidden_dim>.*?)_nl(?P<num_layers>.*?)_opt(?P<optimizer>.*?)_bs(?P<batch_size>.*?)_lr(?P<lr>.*?)_wd(?P<wd>.*?)$', 
    title
  )
  print(title)
  if match:
    model_str = match.group('model_str')
    fn_name = match.group('fn_name')
    hidden_dim = int(match.group('hidden_dim'))
    num_layers = int(match.group('num_layers'))
    optimizer = match.group('optimizer')
    batch_size = int(match.group('batch_size'))
    lr = float(match.group('lr'))
    wd = float(match.group('wd'))
    results = {'batch_size': batch_size, 'lr': lr, 'wd': wd, 'num_layers': num_layers, 'hidden_dim': hidden_dim}
    return results
    #print(f'model_str: {model_str}, fn_name: {fn_name}, hidden_dim: {hidden_dim}, num_layers: {num_layers}, optimizer: {optimizer}, batch_size: {batch_size}, lr: {lr}, wd: {wd}, epochs: {epochs}')

def run_best_models():
  functions = [[1]]
  modelclasses = []
  titles = []
  for i in range(len(functions)):
    fn_name = ''.join([str(x) for x in functions[i]])
    titles.append(f'BaselineModel_fn{fn_name}_hd64_nl1_optAdam_bs16_lr1e-5_wd0.0')
    print(titles[-1])
    modelclasses.append(BaselineModel)
  for i,title in enumerate(titles):
    params = extract_params_from_title(title)
    params['modelclass'] = modelclasses[i]
    params['functions'] = functions[i]
    params['num_epochs'] = 5000
    params['optimizer'] = torch.optim.Adam
    run_params(train, params)


if __name__ == '__main__':
  #run_best_models()
  # title = 'BaselineModel_fn01234_hd64_nl1_optAdam_bs16__lr1e-05_wd0.0'
  # params = extract_params_from_title(title)
  # params['modelclass'] = BaselineModel
  # params['functions'] = [0,1,2,3,4]
  # params['num_epochs'] = 5000
  # params['optimizer'] = torch.optim.Adam
  # run_params(train, params)
  #run_best_models()
  modelclass = BaselineModel
  # continuing this run BaselineModel_fn01234_hd128_nl2_optAdam_bs4__lr0.0001_wd0.05_epochs10000_seed1_20lim_ts0.1_0.5cos
  #all_functions = [[0],[1]]
  all_functions = [[3], [4]]
  #all_functions = [[0,1,2,3,4]]
  
  for functions in all_functions:
    param_grid = {
      'lr': [1e-4],
      'optimizer': [Adam],
      'hidden_dim': [128],
      'num_layers': [2][::-1],
      'batch_size': [4],
      'wd': [1e-1], # CHANGE FOR INDIVIDUAL RUNS
      'num_epochs': [10000],
      'functions': [functions],
      'seed': [1],
      'ts': [0.11],
      'modelclass': [modelclass],
      'stop_frac': [0.5]
      }
    random_search_parameters(modelclass, functions, train, param_grid = param_grid, all = True)
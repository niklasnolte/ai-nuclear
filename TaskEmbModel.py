import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from config import Config
import os
from data import get_data_nomod
import pandas as pd
from data import get_data_nomod_difftasks
from torch.optim import Adam, SGD, RMSprop
from sklearn.model_selection import ParameterSampler
import numpy as np
import math


config = Config()


class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        super(ResidualBlock, self).__init__()
        self.nonlinear = torch.nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model),
        nn.ReLU(),
    )

    def forward(self, x):
        return self.nonlinear(x) + x
    

class TaskEmbModel(nn.Module):
    def __init__(self,functions, hidden_dim=64, num_layers=2):
        super(TaskEmbModel, self).__init__()
        self.P = config.LIMIT
        self.emb_a = nn.Embedding(self.P+len(functions), hidden_dim)
        self.nonlinear = torch.nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            *[ResidualBlock(hidden_dim) for _ in range(num_layers)],
        )
        self.readout = nn.Linear(hidden_dim, len(functions))

    def forward(self, input):
        a = self.emb_a(input[:,0]) # [ batch_size, hidden_dim ]
        b = self.emb_a(input[:,1]) # [ batch_size, hidden_dim ]
        tasks = self.emb_a(input[:, -1] + self.P)
        x = self.nonlinear(torch.hstack((a, b, tasks)))
        x = self.readout(x)
        indices = input[:,-1].long()
        row_indices = torch.arange(x.size(0)).unsqueeze(-2)[0].long()
        output = x[row_indices, indices]
        return output.view(-1,1)
    def __str__(self):
        return 'TaskEmbModel'
    
def train(modelclass, function_dict, title, basepath, lr, wd, device = 'cuda', hidden_dim = 64, seed = 1, test_size = 0.1, batch_size = 100, num_layers = 2, optim = Adam, num_epochs = 1000):
  #train just for modular arithmetic
  torch.manual_seed(seed)
  os.makedirs(basepath, exist_ok=True)
  functions = list(function_dict.values())
  X_train, X_test, y_train, y_test, vocab_size = get_data_nomod_difftasks(functions, test_size = test_size, seed = seed) 

  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)

  all_a = torch.tensor(list(range(vocab_size[0]))).to(device)

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size*len(functions), shuffle=True)
  model = modelclass(functions,hidden_dim = hidden_dim, num_layers = num_layers).to(device)

  loss_fn = nn.MSELoss()
  optimizer = optim(model.parameters(), lr=lr, weight_decay=wd)
  bar = tqdm.tqdm(range(num_epochs))
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_loader))
  lowest_loss = 1e10

  data_dict = None

  torch.autograd.set_detect_anomaly(True)

  for i in bar:
    
    for X_batch, y_batch in train_loader:    
        optimizer.zero_grad()
        y_batch_pred = model(X_batch).to(device)
        train_loss = loss_fn(y_batch_pred, y_batch)
        train_loss.backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        metrics = {}
        y_test_pred = model(X_test).to(device)
        y_train_pred = model(X_train).to(device)
        metrics['train_loss'] = loss_fn(y_train_pred, y_train).cpu().numpy()
        metrics['test_loss'] = loss_fn(y_test_pred, y_test).cpu().numpy()

        fn_names = list(function_dict.keys())
        for j, fn in enumerate(fn_names):
            y_train_pred_fn = model(X_train[X_train[:,-1] == j]).to(device)
            y_test_pred_fn = model(X_test[X_test[:,-1] == j]).to(device)
            y_train_fn = y_train[X_train[:,-1] == j]
            y_test_fn = y_test[X_test[:,-1] == j]
            metrics[fn+'_train'] = loss_fn(y_train_fn, y_train_pred_fn).cpu().numpy()
            metrics[fn+'_test'] = loss_fn(y_test_fn, y_test_pred_fn).cpu().numpy()
        
        entropy = effective_dim(model, all_a)
        entropy = entropy.cpu().numpy()
        metrics['entropy'] = entropy

        if metrics['test_loss'] < lowest_loss:
            lowest_loss = metrics['test_loss']
            best_state_dict = model.state_dict()
            best_metrics = metrics

        if i % 100 == 0:
            torch.save(model.state_dict(), basepath + f"epoch{i}.pt")
        if i % 10 == 0:
            
            if data_dict is None:
                data_dict = {m:[metrics[m]] for m in metrics.keys()}
                data_dict['iterations'] = [i]
            else:
                for m in metrics:
                    data_dict[m].append(metrics[m])
                data_dict['iterations'].append(i)
            df = pd.DataFrame(data_dict)
            df.to_csv('csv/{0}.csv'.format(title))
        formatted_metrics = {key: f"{value:.2e}" for key, value in metrics.items()}
        bar.set_postfix(**formatted_metrics)
    torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return best_metrics

def effective_dim(model, all_a):
    #calculate entropy of the embeddings
    a = model.emb_a(all_a)
    a_S = (torch.square(torch.svd(a)[1]) / (all_a.shape[0] - 1))
    a_prob = a_S/a_S.sum()
    entropy_a = -(a_prob * torch.log(a_prob)).sum()
    entropy = torch.exp(entropy_a)
    return entropy


def run_model(functions, wd = 1e-2, lr = 1e-4, test_size = 0.05, batch_size = 4):
    # takes 1d list of functions and runs
    all_fn_dict = config.all_fn_dict
    all_fn_list = list(all_fn_dict.keys())
    fn_list = [all_fn_list[j] for j in functions]
    fn_dict = {k: all_fn_dict[k] for k in fn_list}

    fn_name = ''.join([str(functions[i]) for i in range(len(functions))])
    name = f'TaskEmbModel_fn{fn_name}_ts{test_size}_wd{wd}_lr{lr}_batch{batch_size}'
    train(TaskEmbModel, fn_dict, lr, wd, f'models/OuailResModel/{name}/', 'cuda', name, test_size = test_size, batch_size = batch_size)


def run_models(function_list, wd = 1e-3, lr = 1e-4, test_size = 0.95, batch_size = 4):
    #2d list of function_list
    for fn_list in function_list:
        run_model(fn_list, wd, lr, test_size, batch_size)


def random_search_parameters(modelclass,functions, train_function,all = False, param_list = None):
    # param_grid = {
    #     'hidden_dim': [32, 64, 128],
    #     'num_layers': [1, 2, 3],
    #     'lr': [0.001, 0.0001],
    #     'optimizer': [Adam],
    #     'batch_size': [16, 32],
    #     'wd': [0.0],
    #     'modelclass': [TaskEmbModel],
    #     'functions': [[0]]
    # }
    param_grid = {
        'lr': [1e-5],
        'optimizer': [Adam],
        'hidden_dim': [64, 128],
        'num_layers': [1, 2],
        'batch_size': [16, 32],
        'wd': [0.0],
        'modelclass': [modelclass],
        'functions': [functions],
        'num_epochs': [500]
        }
    
    num_combinations = np.prod([len(v) for v in param_grid.values()])
    if all:
        num_trials = num_combinations
    else:
        num_trials = int(math.sqrt(num_combinations))+1
        #num_trials = int((num_combinations//3))
    print('total num parameters', num_combinations)
    print('num_trials', num_trials)
    param_list = list(ParameterSampler(param_grid, n_iter=num_trials))
    for params in param_list:
        metrics = run_params(train_function, params)
        


        path = f'csv/{dir}.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.drop(columns=['Unnamed: 0'])
            res = df.to_dict('list')
            for k,v in metrics.items():
                res[k].append(v)
        else:
            res = {k:[v] for k,v in metrics.items()}
            
        df = pd.DataFrame(res)
        df.to_csv(path)

def run_params(train_function, params):
    modelclass = params['modelclass']
    model_str = modelclass([], 1, 1).__str__()

    functions = params['functions']
    all_fn_dict = config.all_fn_dict
    all_fn_list = list(all_fn_dict.keys())
    fn_list = [all_fn_list[j] for j in functions]
    function_dict = {k: all_fn_dict[k] for k in fn_list}
    fn_name = ''.join([str(functions[i]) for i in range(len(functions))])

    dir = f'{model_str}_{fn_name}_random_search'

    
    hidden_dim = params['hidden_dim']
    num_layers = params['num_layers']
    lr = params['lr']
    optimizer = params['optimizer']
    batch_size = params['batch_size']
    wd = params['wd']
    epochs = params['num_epochs']
    
    dir = f'{model_str}_{fn_name}_random_search'
    title = f'{model_str}_fn{fn_name}_hd{hidden_dim}_nl{num_layers}_opt{optimizer.__name__}_bs{batch_size}__lr{lr}_wd{wd}_epochs{epochs}_cos'
    basepath = f'models/{dir}/{title}/'
    print(title)
    metrics = train_function(modelclass=modelclass, 
            function_dict=function_dict, 
            lr=lr, 
            wd=wd, 
            basepath=basepath, 
            device='cuda', 
            title=title, 
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            optim=optimizer,
            num_epochs=epochs)
    metrics['title'] = title
    print(title, basepath)
    return metrics





if __name__ == "__main__":
    functions = [[0,1,2,3,4], [0,1,2,3], [0,1,2], [0,1], [0]]
    #run_models(functions)

    
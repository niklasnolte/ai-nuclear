import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from config import Config
import os
from data import get_data_nomod
import pandas as pd
from data import get_data_nomod_difftasks

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
    

class OuailResModel(nn.Module):
    def __init__(self,functions, hidden_dim=64, num_layers=2):
        super(OuailResModel, self).__init__()
        self.P = config.LIMIT
        self.emb_a = nn.Embedding(self.P+len(functions), hidden_dim)
        self.nonlinear = torch.nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            *[ResidualBlock(hidden_dim) for _ in range(num_layers)],
        )
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        a = self.emb_a(x[:,0]) # [ batch_size, hidden_dim ]
        b = self.emb_a(x[:,1]) # [ batch_size, hidden_dim ]
        tasks = self.emb_a(x[:, -1] + self.P)
        x = self.nonlinear(torch.hstack((a, b, tasks)))
        x = self.readout(x)
        return x
    
def train(modelclass, function_dict, lr, wd, basepath, device, title, embed_dim = 64, seed = 1, test_size = 0.2):
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

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100*len(functions), shuffle=True)
  model = modelclass(functions, embed_dim).to(device)

  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  num_epochs = 3000
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
        
        if metrics['test_loss'] < lowest_loss:
            lowest_loss = metrics['test_loss']
            best_state_dict = model.state_dict()

        if i % 500 == 0:
            torch.save(model.state_dict(), basepath + f"epoch{i}.pt")
        if i % 10 == 0:
            entropy = effective_dim(model, all_a)
            entropy = entropy.cpu().numpy()
            metrics['entropy'] = entropy

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
  return metrics

def effective_dim(model, all_a):
  #calculate entropy of the embeddings
    a = model.emb_a(all_a)
    a_S = (torch.square(torch.svd(a)[1]) / (all_a.shape[0] - 1))
    a_prob = a_S/a_S.sum()
    entropy_a = -(a_prob * torch.log(a_prob)).sum()
    entropy = torch.exp(entropy_a)
    return entropy


def run_model(functions, wd = 1e-2, lr = 1e-4, test_size = 0.05):
    # takes 1d list of functions and runs
    all_fn_dict = config.all_fn_dict
    all_fn_list = list(all_fn_dict.keys())
    fn_list = [all_fn_list[j] for j in functions]
    fn_dict = {k: all_fn_dict[k] for k in fn_list}

    fn_name = ''.join([str(functions[i]) for i in range(len(functions))])
    name = f'OuailResModel_fn{fn_name}_ts{test_size}_wd{wd}_lr{lr}'
    train(OuailResModel, fn_dict, lr, wd, f'models/OuailResModel/{name}/', 'cuda', name, test_size = test_size)


def run_models(function_list, wd = 1e-2, lr = 1e-4, test_size = 0.05):
    #2d list of function_list
    for fn_list in function_list:
        run_model(fn_list, wd, lr, test_size)


if __name__ == "__main__":
    functions = [[1,2], [2,3], [3,4], [4,0], [0,1,2], [0,1,2,3]]
    run_models(functions)

# %%
import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import tqdm
import urllib.request
import optuna

# %%

def lc_read_csv(url):
    req = urllib.request.Request("https://nds.iaea.org/relnsd/v0/data?" + url)
    req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
    return pd.read_csv(urllib.request.urlopen(req))

df = lc_read_csv("fields=ground_states&nuclides=all")

#selection
df = df[df.binding != ' ']

# %%
vocab_size = len(set(df.z) | set(df.n))

# %%
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

def train(lr, wd, hidden_dim, basepath, device):
  X = torch.tensor(df[["z", "n"]].values).int()
  y = torch.tensor(df.binding.astype(float).values).view(-1, 1).float()
  y = (y - y.mean()) / y.std()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=len(X_train), shuffle=True)

  model = Model(vocab_size, hidden_dim).to(device)
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

  bar = tqdm.tqdm(range(int(2e3)))
  lowest_loss = 1e10
  for i in bar:
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      y_pred = model(X_batch)
      loss = loss_fn(y_pred, y_batch)
      loss.backward()
      optimizer.step()
    with torch.no_grad():
      y_pred = model(X_test)
      loss = loss_fn(y_pred, y_test)
      train_loss = loss_fn(model(X_train), y_train)
      bar.set_postfix(loss=loss.item(), train_loss=train_loss.item())
      if loss < lowest_loss:
        lowest_loss = loss
        best_state_dict = model.state_dict()
      if i % 100 == 0:
        torch.save(model.state_dict(), basepath + f"epoch_{i}.pt")
  torch.save(best_state_dict, basepath + "best.pt")
  return lowest_loss.item()

# %%
def objective(trial):
  # Hyperparameters
  lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
  wd = trial.suggest_loguniform('wd', 1e-5, 1e-1)
  hidden_dim = 2**trial.suggest_int('hidden_dim', 5, 10)
  basepath = f"models/{trial.number}/"
  os.makedirs(basepath, exist_ok=True)
  return train(lr, wd, hidden_dim, basepath, device = torch.device(f"cuda:{trial.number%2}"))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=1000, n_jobs=2)
with open("study.pickle", "wb") as f:
  pickle.dump(study, f)

# %%




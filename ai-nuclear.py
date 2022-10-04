# %%
import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import optuna
from model import Model
from data import get_data


def train(lr, wd, hidden_dim, basepath, device):
  os.makedirs(basepath, exist_ok=True)

  X_train, X_test, y_train, y_test, vocab_size = get_data() 
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
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return lowest_loss.item()

# %%
# def objective(trial):
#   # Hyperparameters
#   lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
#   wd = trial.suggest_loguniform('wd', 1e-5, 1e-1)
#   hidden_dim = 2**trial.suggest_int('hidden_dim', 5, 10)
#   basepath = f"models/{trial.number}/"
#   return train(lr, wd, hidden_dim, basepath, device = torch.device(f"cuda:{trial.number%2}"))

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=1000, n_jobs=2)
# with open("study.pickle", "wb") as f:
#   pickle.dump(study, f)

# %%
train(lr=.037, wd=.003, hidden_dim=64, basepath="models/test/", device=torch.device("cuda:0"))

# %%

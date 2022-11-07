import os
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import pandas as pd


from data import get_data
from sklearn.decomposition import PCA

from pca import effective_dim, regularize_effective_dim, nd_loss




def train(modelclass, lr, wd, embed_dim, basepath, device, title, heavy_elem = 0, reg_effective = 0.02, reg_actual = 0, reg_actual_n=2):
  torch.manual_seed(1)
  os.makedirs(basepath, exist_ok=True)

  X_train, X_test, y_train, y_test, vocab_size = get_data() 
  
  train_heavy_mask = X_train[:,0]>heavy_elem
  X_train = X_train[train_heavy_mask]
  y_train = y_train[train_heavy_mask]

  test_heavy_mask = X_test[:,0]>heavy_elem
  X_test = X_test[test_heavy_mask]
  y_test = y_test[test_heavy_mask]

  

  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)

  all_protons = torch.tensor(list(range(vocab_size[0]))).to(device)
  all_neutrons = torch.tensor(list(range(vocab_size[1]))).to(device)

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=len(X_train), shuffle=True)
  model = modelclass(*vocab_size, embed_dim).to(device)

  loss_fn = nn.MSELoss()
  early_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  late_optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

  bar = tqdm.tqdm(range(int(1.5e4)))
  lowest_loss = 1e10

  loss_list = []
  entropy_proton_list = []
  entropy_neutron_list = []
  iterations = []

  torch.autograd.set_detect_anomaly(True)
  for i in bar:

    optimizer = early_optimizer# if i < len(bar)//2 else late_optimizer
    for X_batch, y_batch in train_loader:

      optimizer.zero_grad()
      y_pred = model(X_batch).to(device)
      loss = loss_fn(y_pred, y_batch)

      if reg_effective:
        regularization1 = regularize_effective_dim(model, all_protons, all_neutrons, alpha = reg_effective)
        loss += regularization1
      if reg_actual:
        regularization2 = nd_loss(model, all_protons, all_neutrons, X_test,  y_test, n=reg_actual_n)
        loss += regularization2
      #loss += 10 * wd * ( torch.square(model.emb_proton.weight).sum() + torch.square(model.emb_neutron.weight).sum() )
      loss.backward()
      optimizer.step()
    with torch.no_grad():

      # regularization value here is different than the X_batch code block
      # print(regularizer(model, all_protons, all_neutrons))

      y_pred = model(X_test)
      loss = loss_fn(y_pred, y_test)
      train_loss = loss_fn(model(X_train), y_train)

      if loss < lowest_loss:
        lowest_loss = loss
        best_state_dict = model.state_dict()
      if i % 100 == 0:
        torch.save(model.state_dict(), basepath + f"epoch{i}.pt")

      if i % 100 == 0:
        iterations.append(i)
        loss_list.append(loss.cpu().numpy())

        pr_e, ne_e = effective_dim(model, all_protons, all_neutrons)
        pr_e = pr_e.cpu().numpy()
        ne_e = ne_e.cpu().numpy()
        entropy_proton_list.append(pr_e)
        entropy_neutron_list.append(ne_e)


        data_dict = {'Iteration':iterations, 'Loss':loss_list,
             'Proton_Entropy':entropy_proton_list, 'Neutron_Entropy':entropy_neutron_list}
        df = pd.DataFrame(data_dict)
        df.to_csv('csv/{0}.csv'.format(title))

      torch.save(model.state_dict(), basepath + f"latest.pt")
      bar.set_postfix(loss=loss.item(), train_loss=train_loss.item(), pr_e=pr_e, ne_e=ne_e)

  torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return lowest_loss.item()

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

if __name__ == '__main__':
  pass

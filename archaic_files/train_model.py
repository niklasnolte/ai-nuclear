import os
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import pandas as pd




from base_functions import get_data
from pca_graphs import effective_dim
from sklearn.decomposition import PCA





def train(modelclass, lr, wd, embed_dim, basepath, device, title, heavy_elem = 15, reg_pca = 1, reg_type = 'dimall'):
  torch.manual_seed(1)
  os.makedirs(basepath, exist_ok=True)

  X_train, X_test, y_train, y_test, vocab_size = get_data(heavy_elem = heavy_elem) 

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

  bar = tqdm.tqdm(range(int(3e4)))
  lowest_loss = 1e10

  loss_list = []
  entropy_proton_list = []
  entropy_neutron_list = []
  iterations = []
  pca_losses = []

  torch.autograd.set_detect_anomaly(True)
  for i in bar:

    optimizer = early_optimizer# if i < len(bar)//2 else late_optimizer
    for X_batch, y_batch in train_loader:

      optimizer.zero_grad()
      y_pred = model(X_batch).to(device)
      loss = loss_fn(y_pred, y_batch)
      if reg_type == 'dimall':
        loss += reg_pca * model.alldims_loss(loss_fn, X_test, y_test)
      elif reg_type == 'dimn':
        loss += reg_pca * model.stochastic_pca_loss(loss_fn, X_test, y_test)
      else:
        n = int(reg_type[-1])
        dim_pred = model.evaluate_ndim(X_test, device, n = n)
        loss+= reg_pca * loss_fn(dim_pred, y_test)
      
      loss.backward()
      optimizer.step()
    with torch.no_grad():

      y_pred = model(X_test)
      loss = loss_fn(y_pred, y_test)
      pca_loss = model.alldims_loss(loss_fn, X_test, y_test)
      train_loss = loss_fn(model(X_train), y_train)

      if loss < lowest_loss:
        lowest_loss = loss
        best_state_dict = model.state_dict()
      if i % 100 == 0:
        torch.save(model.state_dict(), basepath + f"epoch{i}.pt")

      if i % 100 == 0:
        iterations.append(i)
        pca_losses.append(pca_loss.cpu().numpy())
        loss_list.append(loss.cpu().numpy())

        pr_e, ne_e = effective_dim(model, all_protons, all_neutrons)
        pr_e = pr_e.cpu().numpy()
        ne_e = ne_e.cpu().numpy()
        entropy_proton_list.append(pr_e)
        entropy_neutron_list.append(ne_e)

        data_dict = {'Iteration':iterations, 'Loss':loss_list, 'PCA_Loss':pca_losses,
             'Proton_Entropy':entropy_proton_list, 'Neutron_Entropy':entropy_neutron_list}
        df = pd.DataFrame(data_dict)
        df.to_csv('csv/{0}.csv'.format(title))

      torch.save(model.state_dict(), basepath + f"latest.pt")
      bar.set_postfix(loss=loss.item(), pca_loss = pca_loss.item(), train_loss=train_loss.item(), pr_e=pr_e, ne_e=ne_e)

  torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return lowest_loss.item()


if __name__ == '__main__':
  pass

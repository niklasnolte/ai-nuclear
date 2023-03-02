import os
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import urllib.request
from gradient_descent_the_ultimate_optimizer import gdtuo

from data import get_data
from sklearn.decomposition import PCA

def effective_dim(model, all_protons, all_neutrons):
  #calculate entropy of the embeddings
  protons = model.emb_proton(all_protons)
  neutrons = model.emb_neutron(all_neutrons)

  protons_S = (torch.square(torch.svd(protons)[1]) / (all_protons.shape[0] - 1))
  neutrons_S = (torch.square(torch.svd(neutrons)[1]) / (neutrons.shape[0] - 1))
  
  proton_prob = protons_S/protons_S.sum()
  neutron_prob = neutrons_S/neutrons_S.sum()
  
  entropy_protons = -(proton_prob * torch.log(proton_prob)).sum()
  entropy_neutrons = -(neutron_prob * torch.log(neutron_prob)).sum()

  pr_e = torch.exp(entropy_protons)
  ne_e = torch.exp(entropy_neutrons)

  return pr_e, ne_e

def regularize_effective_dim(model, all_protons, all_neutrons, alpha = 0.02):
  pr_e, ne_e = effective_dim(model, all_protons, all_neutrons)
  regularization = alpha * (pr_e+ne_e)
  return regularization

def train(modelclass, lr, wd, embed_dim, basepath, device, title, heavy_elem = 15, reg_pca = 1, reg_type = 'dimall', seed = 1, norm = False):
  
  if norm:
    lr = lr/(1+reg_pca)
  torch.manual_seed(seed)
  os.makedirs(basepath, exist_ok=True)
  X_train, X_test, y_train, y_test, vocab_size, y_mean, y_std = get_data(heavy_elem = heavy_elem, return_ymean_ystd=True) 

  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)


  all_protons = torch.tensor(list(range(vocab_size[0]))).to(device)
  all_neutrons = torch.tensor(list(range(vocab_size[1]))).to(device)

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=len(X_train), shuffle=True)
  model = modelclass(*vocab_size, embed_dim).to(device)

  #THE ULTIMATE OPTIMIZER
  optim = gdtuo.Adam(optimizer=gdtuo.SGD(1e-5))
  mw = gdtuo.ModuleWrapper(model, optimizer=optim)
  mw.initialize()

  loss_fn = nn.MSELoss()
  early_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  late_optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

  bar = tqdm.tqdm(range(int(2e4)))
  lowest_loss = 1e10

  loss_list = []
  entropy_proton_list = []
  entropy_neutron_list = []
  iterations = []

  torch.autograd.set_detect_anomaly(True)
  for i in bar:

    optimizer = early_optimizer# if i < len(bar)//2 else late_optimizer
    #mw.begin()

    for X_batch, y_batch in train_loader:

      optimizer.zero_grad()
      y_pred = model(X_batch).to(device)
      #y_pred = mw.forward(X_batch).to(device)
      loss = loss_fn(y_pred, y_batch)
      if reg_pca:
        if reg_type == 'dimall':
          regloss = model.alldims_loss(loss_fn, X_train, y_train)
        elif reg_type == 'dimn':
          regloss = model.stochastic_pca_loss(loss_fn, X_train, y_train)
        elif reg_type == 'oldeff':
          regloss = regularize_effective_dim(model, all_protons, all_neutrons, alpha = 1)
        else:
          n = int(reg_type[-1])
          regloss = model.evaluate_ndim(loss_fn, X_train, y_train, device=device, n = n)
        if norm:
          regloss = loss.item() * regloss / regloss.item()
        loss += reg_pca * regloss
      loss.backward()
      optimizer.step()
      #mw.zero_grad()
      #loss.backward(create_graph=True)
      #mw.step()
    with torch.no_grad():

      y_pred = mw.forward(X_test)*y_std+y_mean
      loss = loss_fn(y_pred, y_test*y_std+y_mean).sqrt()
      train_loss = loss_fn(mw.forward(X_train), y_train)

      if loss < lowest_loss:
        lowest_loss = loss
        best_state_dict = model.state_dict()
      if i % 10 == 0:
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


def effective_dim(model, all_protons, all_neutrons):
  #calculate entropy of the embeddings
  protons = model.emb_proton(all_protons)
  neutrons = model.emb_neutron(all_neutrons)

  protons_S = (torch.square(torch.svd(protons)[1]) / (all_protons.shape[0] - 1))
  neutrons_S = (torch.square(torch.svd(neutrons)[1]) / (neutrons.shape[0] - 1))
  
  proton_prob = protons_S/protons_S.sum()
  neutron_prob = neutrons_S/neutrons_S.sum()
  
  entropy_protons = -(proton_prob * torch.log(proton_prob)).sum()
  entropy_neutrons = -(neutron_prob * torch.log(neutron_prob)).sum()

  pr_e = torch.exp(entropy_protons)
  ne_e = torch.exp(entropy_neutrons)

  return pr_e, ne_e

def regularize_effective_dim(model, all_protons, all_neutrons, alpha = 0.02):
  pr_e, ne_e = effective_dim(model, all_protons, all_neutrons)
  regularization = alpha * (pr_e+ne_e)
  return regularization



if __name__ == '__main__':
  pass

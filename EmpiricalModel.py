import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import tqdm
from data import get_data

def empirical_function(p_n, ymean, ystd):
  N = p_n[1]
  Z = p_n[0]
  A = N+Z
  a = 14
  b = 13
  c = 0.585
  d = 19.3
  e = 33
  E_raw = a - b/(A**(1/3)) - c*Z**2/(A**(4/3)) - d*(N-Z)**2/A**2

  if A%2==1:
    pass
  elif N%2==1:
    E_raw-=e/A**(7/4)
  else:
    E_raw+=e/A**(7/4)
  E_raw*=1000
  return (E_raw - ymean)/ystd
  return E_raw

class Empirical(nn.Module):
  def __init__(self, ymean, ystd):
    super().__init__()
    self.a = torch.nn.Parameter(torch.tensor([14.0]))
    self.b = torch.nn.Parameter(torch.tensor([13.0]))
    self.c = torch.nn.Parameter(torch.tensor([.585]))
    self.d = torch.nn.Parameter(torch.tensor([19.3]))
    self.e =torch.nn.Parameter(torch.tensor([33.0]))
    self.ymean = ymean
    self.ystd = ystd

  def forward(self, x):
    Z = x[:,0]
    N = x[:, 1]
    A= N+Z

    E_raw = self.a - self.b/(A**(1/3)) - self.c*Z**2/(A**(4/3)) - self.d*(N-Z)**2/A**2
    e_cont = self.e/(A**(7/4)) * (A%2==0) * ((N%2==0) * 2 - 1)
    E_raw+=e_cont
    E_raw*=1000
    return (E_raw - self.ymean)/self.ystd


def test_empirical(heavy_elem = 15):
  _, X_test, _, y_test, _ = get_data(heavy_elem=heavy_elem)



  sd = torch.load(f"empirical_sd.pt")
  model = torch.load('empirical_model.pt')
  model.load_state_dict(sd)
  
  y_pred = model(X_test)
  loss  = nn.MSELoss()

  return loss(y_pred, y_test.view(-1))


def train_empirical():
  X_train, X_test, y_train, y_test, vocab_size, ymean, ystd = get_data(return_ymean_ystd = True, heavy_elem=15)
  print(X_train.shape)
  print(X_test.shape)
  print(y_train.shape)
  print(y_test.shape)
  model = Empirical(ymean,ystd)

  with torch.no_grad():
    y_pred_test = model(X_test)
    print(y_pred_test.shape)
    y_pred_train = model(X_train)
    print(y_pred_train.shape)

    print(y_pred_train)


    loss_fn  =nn.MSELoss()

    print(loss_fn(y_pred_test, y_test.view(-1, 1)))
    print(loss_fn(y_pred_train, y_train.view(-1, 1)))
  
  epochs = 3e4
  
  wd = 1e-4
  lr = 1e-3
  
  bar = tqdm.tqdm(range(int(epochs)))
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  for b in bar:
    optimizer.zero_grad()
    y_pred_train = model(X_train)
    train_loss = loss_fn(y_pred_train, y_train.view(-1))
    train_loss.backward()
    optimizer.step()
    with torch.no_grad():
      y_pred_test = model(X_test)
      test_loss = loss_fn(y_pred_test, y_test.view(-1))
      bar.set_postfix(test_loss=test_loss.item(), train_loss=train_loss.item())
  torch.save(model.state_dict(), 'empirical_sd.pt')
  torch.save(model.cpu().requires_grad_(False), 'empirical_model.pt')
  


if __name__ == '__main__':
  print(train_empirical())
  
  
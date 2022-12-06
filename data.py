import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
from torch import nn
import numpy as np
import matplotlib.colors as mcolors
import urllib.request
from sklearn.model_selection import train_test_split
from copy import deepcopy

def get_data(return_ymean_ystd = False, heavy_elem = 0):
  np.random.seed(1)
  def lc_read_csv(url):
    req = urllib.request.Request("https://nds.iaea.org/relnsd/v0/data?" + url)
    req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
    return pd.read_csv(urllib.request.urlopen(req))

  df = lc_read_csv("fields=ground_states&nuclides=all")
  df = df[df.binding != ' ']
  df = df[df.binding != 0]

  df = df[df.z>heavy_elem]
  df.z -= df.z.min()
  df.n -= df.n.min()
  
  vocab_size = (df.z.nunique(), df.n.nunique())

  X = torch.tensor(df[["z", "n"]].values).int()
  y = torch.tensor(df.binding.astype(float).values).view(-1, 1).float()


  y_mean = y.mean()
  y_std = y.std()
  y = (y - y_mean) / y_std

  X = X.numpy()
  y = y.numpy()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  X_train = torch.tensor(X_train)
  X_test = torch.tensor(X_test)
  y_train = torch.tensor(y_train)
  y_test = torch.tensor(y_test)
  
  if return_ymean_ystd:
    return X_train , X_test, y_train, y_test, vocab_size, y_mean.item(), y_std.item()
  else:
    return X_train , X_test, y_train, y_test, vocab_size

def analyze_differences(heavy_elem = 15):
  fig, axs = plt.subplots(2, 2)
  for i in range(len(axs)):
    print(i)
    ax_row = axs[i]
    if i == 0:
      X_train, X_test, _, _, _ = get_data(heavy_elem = heavy_elem)
      label_train = f'train data in get data\nsize = {X_train.shape[0]}'
      label_test = f'test data in get data\nsize = {X_test.shape[0]}'
    else:
      X_train, X_test, _, _, _ = get_data(heavy_elem = 0)
      train_heavy_mask = X_train[:,0]>heavy_elem
      X_train = X_train[train_heavy_mask]
      label_train = f'train data masked after get data\nsize={X_train.shape[0]}'

      test_heavy_mask = X_test[:,0]>heavy_elem
      X_test = X_test[test_heavy_mask]
      label_test = f'test data masked after get data\nsize={X_test.shape[0]}'
    ax_row[0].scatter(X_test[:,0], X_test[:, 1], alpha = 0.5, c = 'b', label = label_test)
    ax_row[1].scatter(X_train[:, 0], X_train[:, 1], alpha = 0.5,  c = 'r', label = label_train)
    ax_row[0].legend()
    ax_row[1].legend()
  plt.suptitle('Comparing Methods of Trimming Heavy Elems')
  plt.show()

def check_diffs(heavy_elem = 15):
  traintypes = ['train1', 'train2', 'train3']
  fig, axs = plt.subplots(3, 1)
  for i in range(len(traintypes)):
    traintype = traintypes[i]
    if traintype == 'train1' or traintype == 'train2':
      X_train, X_test, y_train, y_test, vocab_size = get_data(heavy_elem = heavy_elem) 
    elif traintype == 'train3':
      X_train, X_test, y_train, y_test, vocab_size = get_data(heavy_elem = 0) 
    if traintype == 'train2' or traintype=='train3':
      train_heavy_mask = X_train[:,0]>heavy_elem
      X_train = X_train[train_heavy_mask]
      y_train = y_train[train_heavy_mask]

      test_heavy_mask = X_test[:,0]>heavy_elem

      X_test = X_test[test_heavy_mask]
      y_test = y_test[test_heavy_mask]
    print(traintype)
    print(f'Xtrain: {X_train.shape} Xtest: {X_test.shape}')
    print(f'ytrain: {y_train.shape} ytest: {y_test.shape}')
    print(f'y_train mean: {y_train.mean()}, y_train std: {y_train.std()}')
    print(f'y_test mean: {y_test.mean()}, y_test std: {y_test.std()}')
    print('\n')
    axs[i].hist(y_test)
  plt.show()

if __name__ == '__main__':
  X_train, X_test, y_train, y_test, vocab_size = get_data(heavy_elem = 15)

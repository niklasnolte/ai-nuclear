import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch.nn as nn
import tqdm

def get_data(return_ymean_ystd = False, heavy_elem = 0):
  np.random.seed(1)
  def lc_read_csv(url):
    req = urllib.request.Request("https://nds.iaea.org/relnsd/v0/data?" + url)
    req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
    return pd.read_csv(urllib.request.urlopen(req))

  df = lc_read_csv("fields=ground_states&nuclides=all")
  #selection
  df = df[df.binding != ' ']
  df = df[df.binding != 0]
  vocab_size = (df.z.nunique(), df.n.nunique())
  X = torch.tensor(df[["z", "n"]].values).int()
  y = torch.tensor(df.binding.astype(float).values).view(-1, 1).float()


  heavy_mask = X[:,0]>heavy_elem
  X = X[heavy_mask]
  y = y[heavy_mask]



  y_mean = y.mean()
  y_std = y.std()
  y = (y - y_mean) / y_std
  
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
  if return_ymean_ystd:
    return X_train , X_test, y_train, y_test, vocab_size, y_mean.item(), y_std.item()
  else:
    return X_train , X_test, y_train, y_test, vocab_size


if __name__ == '__main__':
  X_train , X_test, y_train, y_test, vocab_size = get_data(heavy_elem = 15)

  

  

    

  

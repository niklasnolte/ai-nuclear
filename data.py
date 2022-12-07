import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split

def delta(N,Z):
    A = N+Z
    aP = 12
    delta0 = aP*A**(1/2)
    for i in range(len(A)):        
        if ((N%2==0) & (Z%2==0))[i,0]:
            pass
        elif ((N%2==1) & (Z%2==1))[i,0]:
            delta0[i,0] = -delta0[i,0]
        else:
            delta0[i,0] = 0
    return delta0

def binding_formula(data):
  N = data[["n"]].values
  Z = data[["z"]].values
  A = N+Z
  aV = 15.8
  aS = 18.3
  aC = 0.714
  aA = 23.2
  Eb = aV*A - aS*A**(2/3) - aC*Z*(Z-1)/(A**(1/3)) - aA*(N-Z)**2/A + delta(N,Z)
  return 1000*Eb/A

def get_data(opt,heavy_elem):
  np.random.seed(1)
  def lc_read_csv(url):
      req = urllib.request.Request("https://nds.iaea.org/relnsd/v1/data?" + url)
      req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
      return pd.read_csv(urllib.request.urlopen(req))

  df = lc_read_csv("fields=ground_states&nuclides=all")
  #selection
  df = df[df.binding != ' ']
  df = df[df.binding != 0]
  vocab_size = (df.z.nunique(), df.n.nunique())
  X = torch.tensor(df[["z", "n"]].values).int()
  if opt == 'empirical':
      y = torch.tensor(binding_formula(df)).view(-1, 1).float()
  else:
      y = torch.tensor(df.binding.astype(float).values).view(-1, 1).float()#*X.sum(1, keepdim=True)
      
  heavy_mask = X[:,0]>heavy_elem
  X = X[heavy_mask]
  y = y[heavy_mask]
  yp = (y - y.mean()) / y.std()

  X_train, X_test, y_train, y_test = train_test_split(X, yp, test_size=0.2)
  return X_train , X_test, y_train, y_test, vocab_size, y.mean(), y.std()

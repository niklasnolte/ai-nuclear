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
  Eb = aV*A - aS*A**(2/3) - aC*Z*(Z-1)/(A**(1/3)) - (N-Z)**2/A + delta(N,Z)
  return Eb/A

def PySR_formula(data):
  N = data[["n"]].values
  Z = data[["z"]].values
  x0 = Z
  Eb = ((np.exp(-3.2294352 / np.ex(x0 / 4.6362405)) / 0.28763804) + 4.6362405)
  return Eb

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
      y = torch.tensor(binding_formula(df)).view(-1, 1).float() #MeV
  elif opt == 'PySR':
      y = torch.tensor(PySR_formula(df)).view(-1, 1).float() #MeV
  else:
      y = torch.tensor(df.binding.astype(float).values/1000).view(-1, 1).float()#*X.sum(1, keepdim=True) #keV to MeV
      
  heavy_mask = X[:,0]>heavy_elem
  X = X[heavy_mask]
  y = y[heavy_mask]
  yp = (y - y.mean()) / y.std()

  X_train, X_test, y_train, y_test = train_test_split(X, yp, test_size=0.2)
  return X_train , X_test, y_train, y_test, y.mean(), y.std(), vocab_size

def yorig(y,y_mean,y_std):
    return y*y_std+y_mean

def rms(y_test, y_pred):
 return np.sqrt(((y_test-y_pred)**2).sum()/y_test.size()[0])

# X_train, X_test, y_train, y_test, vocab_size, y_mean, y_std = get_data('empirical',0)
# X = torch.cat((X_train,X_test), 0).cpu().detach().numpy()
# X = np.column_stack((X, X[:,1]+X[:,0]))
# X = np.column_stack((X, X[:,1]-X[:,0]))
# y = y_std*torch.cat((y_train,y_test), 0).cpu().detach().numpy()+y_mean

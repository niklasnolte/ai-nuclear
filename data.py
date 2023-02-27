import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math

def radius_formula(data):
    N = data[["n"]].values
    Z = data[["z"]].values
    A = Z+N
    r0 = 1.2
    fake_rad = r0*A**(1/3) #fm
    return fake_rad*(Z<110)



def delta(Z,N):
    A = Z+N
    aP = 12
    delta0 = aP/A**(1/2)
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
  Eb = aV*A - aS*A**(2/3) - aC*Z*(Z-1)/(A**(1/3)) - aA*(N-Z)**2/A + delta(Z,N)
  Eb[Eb<0] = 0
  return Eb/A #MeV

def PySR_formula(data):
  N = data[["n"]].values
  Z = data[["z"]].values
  x0 = Z
  x1 = N
  Eb = (((-10.931704 / ((0.742612 / x0) + x1)) + 7.764321) + np.sin(x0 * 0.03747635))
  return Eb

def get_data(opt,obs,heavy_elem):
  np.random.seed(1)
  def lc_read_csv(url):
      req = urllib.request.Request("https://nds.iaea.org/relnsd/v1/data?" + url)
      req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
      return pd.read_csv(urllib.request.urlopen(req))

  df = lc_read_csv("fields=ground_states&nuclides=all")
  # df = df[df.binding != ' ']
  # df = df[df.binding != 0]  
  
  vocab_size = (df.z.nunique(), df.n.nunique())
  X = torch.tensor(df[["z", "n"]].values).int()
  if opt == 'empirical':
      y0 = torch.tensor(binding_formula(df)).view(-1, 1).float() #MeV
  elif opt == 'PySR':
      y0 = torch.tensor(PySR_formula(df)).view(-1, 1).float() #MeV
  else:
      y0 = 0
      for obsi in obs:
          #turn missing measurements to zero
          if obsi == 'fake_rad':
              y0i = torch.tensor(radius_formula(df)).view(-1, 1).float()
          elif obsi == 'Z':
              y0i = torch.tensor(df["z"].values).view(-1, 1).float()
          elif obsi == 'N':
              y0i = torch.tensor(df["n"].values).view(-1, 1).float()
          else:
              dfobs = getattr(df,obsi)
              df[obsi] = dfobs.apply(lambda x: 0 if (x == ' ')or(x == 'STABLE')or(x == '?') else x)
              y0i = torch.tensor(df[obsi].astype(float).values).view(-1, 1).float()
              y0i[torch.isnan(y0i)] = 0
              
              if obsi in ['binding', 'qbm', 'sn', 'sp', 'qa', 'qec', 'energy']: 
                  y0i = y0i/1000 #keV to MeV

          if isinstance(y0, torch.Tensor):
              y0 = torch.cat((y0, y0i), dim=1)  
          else:
              y0 = y0i 
      
  heavy_mask = (X[:,0]>heavy_elem) & (X[:,1]>heavy_elem)
  X = X[heavy_mask]
  y0 = y0[heavy_mask]
  y0_mean = y0.mean(dim=0)
  y0_std = y0.std(dim=0)
    
  y = (y0 - y0_mean)/y0_std.unsqueeze(0).expand_as(y0)

  X_train, X_test, y_train, y_test, y0_train, y0_test = train_test_split(X, y, y0, test_size=0.2, random_state=(10))
  return [X_train , X_test], [y_train, y_test], [y0_train, y0_test], [y0_mean, y0_std], vocab_size

def yorig(y,y0_mean,y0_std):
    return y*y0_std+y0_mean

def clear_x(X, y0_dat, obs_i): #removes the Z,N for which there is no measurement
    X = X[y0_dat[:,obs_i] != 0].view(-1, 2)
    return X

def clear_y(y_dat, y0_dat, y_pred, obs_i): #removes the dat for which there is no measurement
    try:
        y_pred = y_pred[y0_dat[:,obs_i] != 0][:,obs_i].view(-1, 1)
        y_dat = y_dat[y0_dat[:,obs_i] != 0][:,obs_i].view(-1, 1)
    except IndexError:
        y_pred = y_pred[y0_dat != 0].view(-1, 1)
        y_dat = y_dat[y0_dat != 0].view(-1, 1)
    return y_dat, y_pred

# def rms(y_dat, y_pred, obs_i): #MeV
#         y_dat, y_pred = clear_y(y_dat, y_dat, y_pred, obs_i)
#         return np.sqrt(((y_dat-y_pred)**2).mean())

def rms(y_dat, y_pred, obs_i): #MeV
        y_dat, y_pred = clear_y(y_dat, y_dat, y_pred, obs_i)
        return np.sqrt(((y_dat-y_pred)**2).sum()/y_dat.size()[0]) 
    
def rms_rel(y_dat, y_pred, obs_i): #MeV
        y_dat, y_pred = clear_y(y_dat, y_dat, y_pred, obs_i)
        return ((y_dat-y_pred)/y_dat).abs().median()



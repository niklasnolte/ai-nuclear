import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math

def get_vobs_for_ZN(X, vobs, Z, N):
    # find the index of the element in X_all that equals (x, y)
    index = torch.nonzero(torch.all(X == torch.tensor([Z, N]), dim=1))
    
    # if no such element exists, return None
    if index.numel() == 0:
        return None
    
    # otherwise, return the corresponding value of vobs
    return vobs[index.item()].item()

def radius_formula(data):
    N = data[["n"]].values
    Z = data[["z"]].values
    A = Z+N
    r0 = 1.2
    fake_rad = r0*A**(1/3) #fm
    return fake_rad*(Z<110)

def Sn(model,heavy_mask):
    (X_train, X_test), _, (y0_train, y0_test), (y0_mean, y0_std), _ = get_data('data',['binding','sn'],heavy_mask)

    X = torch.cat((X_train,X_test), 0)
    y0_dat = torch.cat((y0_train,y0_test), 0)
    X = clear_x(X, y0_dat, 1)
    y0_dat,_ = clear_y(y0_dat, y0_dat, y0_dat, 1)
    
    X_all = X.clone()
    for i in range(X.size(0)):
        a, b = X[i]
        
        # check if [a, b-1] exists in X
        if not torch.any(torch.all(X == torch.tensor([a, b-1]), dim=1)):
            # if [a, b-1] does not exist, append it to the end of X
            X_all = torch.cat((X_all, torch.tensor([[a, b-1]])), dim=0)
    A = (X_all[:,0]+X_all[:,1])
    
    Eb =  yorig(model(X_all)[:,0].detach(),y0_mean[0],y0_std[0])
    Eb = Eb*A
    Sn = torch.tensor([get_vobs_for_ZN(X_all, Eb, Z, N) - get_vobs_for_ZN(X_all, Eb, Z, N-1) for Z, N in X])
    Sn = Sn.view(-1,1)
    
    rms = np.sqrt(((y0_dat-Sn)**2).sum()/y0_dat.size()[0])
    
    return y0_dat,Sn,rms

def DeltaE_dat(heavy_mask,obsi):
    (X_train, X_test), _, (y0_train, y0_test), (y0_mean, y0_std), _ = get_data('data',['binding',obsi],heavy_mask)

    X = torch.cat((X_train,X_test), 0)
    y0_dat = torch.cat((y0_train,y0_test), 0)
    
    def E(Z,N,i):
        try:
            ind = torch.where(torch.all(X == torch.tensor([Z, N]), dim=1))[0].item()
            return y0_dat[:,i][ind]
        except ValueError:
            return 0
        
    DeltaE_theory = y0_dat[:,1].clone()
    for i in range(X.size(0)):
        Z, N = X[i]
        
        # check if [a, b-1] exists in X
        if E(Z,N-1,0) != 0 and E(Z,N,0) != 0:
            DeltaE_theory[i] = (Z+N)*E(Z,N,0)-(Z+N-1)*E(Z,N-1,0)
        else:
            DeltaE_theory[i] = 0
            
    DetlaEi_theory = DeltaE_theory[DeltaE_theory != 0].view(-1,1)
    y0i_dat = y0_dat[:,1][DeltaE_theory != 0].view(-1,1)
    error = DetlaEi_theory-y0i_dat
    
    rms = np.sqrt(((error)**2).sum()/y0_dat[:,1].size()[0])
    
    return DeltaE_theory,rms


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

def get_data(opt,obs,heavy_elem,TMD):
  np.random.seed(1)
  # def lc_read_csv(url):
  #     req = urllib.request.Request("https://nds.iaea.org/relnsd/v1/data?" + url)
  #     req.add_header('User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
  #     return pd.read_csv(urllib.request.urlopen(req))

  # df = lc_read_csv("fields=ground_states&nuclides=all")
  df = pd.read_csv('ame2020.csv')
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
              
              if (obsi == 'binding') and (TMD == 'off'):
                  df[obsi] = df.apply(lambda x: 0 if (x['binding_sys'] == 'Y') or (x['binding_unc']*(x['z']+x['n']) > 100) else x[obsi], axis=1)
              
              y0i = torch.tensor(df[obsi].astype(float).values).view(-1, 1).float()
              y0i[torch.isnan(y0i)] = 0
              
              if obsi in ['binding', 'qbm', 'sn', 'sp', 'qa', 'qec', 'energy']: 
                  y0i = y0i/1000 #keV to MeV
              # if obsi == 'binding':
              #     A = (X[:,0] + X[:,1]).view(-1,1)
              #     y0i = y0i*A
                  
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


def clear_y(y_cl, y_dat, y_pred, obsi): #removes the dat for which there is no measurement
    try:             
        y_pred = y_pred[y_cl[:,obsi] != 0][:,obsi].view(-1, 1)
        y_dat = y_dat[y_cl[:,obsi] != 0][:,obsi].view(-1, 1)
    except IndexError:
        y_pred = y_pred[y_cl != 0].view(-1, 1)
        y_dat = y_dat[y_cl != 0].view(-1, 1)
    return y_dat, y_pred

# def rms(y_dat, y_pred, obs_i): #MeV
#         y_dat, y_pred = clear_y(y_dat, y_dat, y_pred, obs_i)
#         return np.sqrt(((y_dat-y_pred)**2).mean())

def rms(y_cl, y_dat, y_pred, obsi): #MeV        
        y_dat, y_pred = clear_y(y_cl, y_dat, y_pred, obsi)
        return np.sqrt(((y_dat-y_pred)**2).sum()/y_dat.size()[0]) 
    
def rms_rel(y_dat, y_pred, obsi): #MeV
        y_dat, y_pred = clear_y(y_dat, y_dat, y_pred, obsi)
        return ((y_dat-y_pred)/y_dat).abs().median()



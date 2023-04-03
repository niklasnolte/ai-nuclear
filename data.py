import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math
import warnings

def yorig(y, y0_mean, y0_std): # yields back the original dat before normalization
    return y*y0_std+y0_mean


def clear_x(y_cl, X, obs_i):  # removes the Z,N for which there is no measurement
    try:
        X = X[y_cl[:, obs_i] != 0].view(-1, 2)
    except IndexError:
        X = X[y_cl != 0].view(-1, 2)
    return X


def clear_y(y_cl, y_dat, y_pred, obsi):  # removes the dat for which there is no measurement
    try:
        y_pred = y_pred[y_cl[:, obsi] != 0][:, obsi].view(-1, 1)
        y_dat = y_dat[y_cl[:, obsi] != 0][:, obsi].view(-1, 1)
    except IndexError:
        y_pred = y_pred[y_cl != 0].view(-1, 1)
        y_dat = y_dat[y_cl != 0].view(-1, 1)
    return y_dat, y_pred

def get_y_for_ZN(X, y, obsi, Z, N):
    # find the index of the element in X_all that equals (Z, N)
    index = torch.nonzero(torch.all(X == torch.tensor([Z, N]), dim=1))

    # if no such element exists, return None
    if index.numel() == 0:
        return None
    y = y[:, obsi].view(-1, 1)
    # otherwise, return the corresponding value of y
    return y[index.item()].item()

def radius_formula(data):
    N = data[["n"]].values
    Z = data[["z"]].values
    A = Z+N
    r0 = 1.2
    fake_rad = r0*A**(1/3)  # fm
    return fake_rad*(Z < 110)


def delta(Z, N):
    A = Z+N
    aP = 9.87
    delta0 = aP/A**(1/2)
    for i in range(len(A)):
        if ((N % 2 == 0) & (Z % 2 == 0))[i]:
            pass
        elif ((N % 2 == 1) & (Z % 2 == 1))[i]:
            delta0[i] = -delta0[i]
        else:
            delta0[i] = 0
    return delta0


def shell(Z, N):
    #calculates the shell effects according to 
    warnings.filterwarnings("ignore", category=RuntimeWarning) 
    alpham = -1.9
    betam = 0.14
    magic = [2, 8, 20, 28, 50, 82, 126, 184]

    def find_nearest(lst, target):
        return min(lst, key=lambda x: abs(x - target))
    nup = np.array([abs(x - find_nearest(magic, x)) for x in Z])
    nun = np.array([abs(x - find_nearest(magic, x)) for x in N])
    P = nup*nun/(nup+nun)
    P[np.isnan(P)] = 0
    return alpham*P + betam*P**2


def binding_formula(df, BW, TMS):
    #calculates E_b/nucleon for the Bethe–von Weizsäcker semi-empirical mass formula (BW = 'BW') and its extension (BW = 'BW2')
    N = df["n"].values
    Z = df["z"].values
    A = N+Z
    if BW == 'BW2': #the best-fit points for the coefficients is different between the two models
        aV = 16.58
        aS = -26.95
        aC = -0.774
        aA = -31.51
    else:
        aV = 15.36
        aS = -16.42
        aC = -0.691
        aA = -22.53
    axC = 2.22
    aW = -43.4
    ast = 55.62
    aR = 14.77
    Eb = aV*A + aS*A**(2/3) + aC*Z**2/(A**(1/3)) + \
        aA*(N-Z)**2/A + delta(Z, N)
    Eb2 = shell(Z, N) + aR*A**(1/3) + axC*Z**(4/3)/A**(1/3) + \
        aW*abs(N-Z)/A + ast*(N-Z)**2/A**(4/3)
    if BW == 'BW2':
        Eb = Eb + Eb2
    Eb[Eb < 0] = 0
    
    if TMS != 'TMS':
        for i in range(len(A)):
            if (df['binding_sys'].values[i] == 'Y') or (df['binding_unc'].values[i]*A[i] > 100):
                Eb[i] = 0
    return Eb/A #MeV

def PySR_formula(df, TMS):
    N = df["n"].values
    Z = df["z"].values
    A = N+Z
    x0 = Z
    x1 = N
    Eb = (((-10.931704 / ((0.742612 / x0) + x1)) +
          7.764321) + np.sin(x0 * 0.03747635))
    if TMS != 'TMS':
        for i in range(len(A)):
            if (df['binding_sys'].values[i] == 'Y') or (df['binding_unc'].values[i]*A[i] > 100):
                Eb[i] = 0
    return Eb


def get_data(obs, heavy_elem, TMS):
    np.random.seed(1)

    def lc_read_csv(url):
        req = urllib.request.Request(
            "https://nds.iaea.org/relnsd/v1/data?" + url)
        req.add_header(
            'User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
        return pd.read_csv(urllib.request.urlopen(req))

    df = lc_read_csv("fields=ground_states&nuclides=all")
    dfE = pd.read_csv('ame2020.csv')

    df = pd.merge(df, dfE[['z', 'n', 'binding_sys', 'binding_unc']], on=[
                  'z', 'n'], how='inner')

    # df = df[df.binding != ' ']
    # df = df[df.binding != 0]

    vocab_size = (df.z.nunique(), df.n.nunique())
    X = torch.tensor(df[["z", "n"]].values).int()
    y0 = 0
    for obsi in obs:

        if obsi == 'fake_rad':
            y0i = torch.tensor(radius_formula(df)).view(-1, 1).float()
        elif obsi == 'binding_BW':
            y0i = torch.tensor(binding_formula(df, 'BW', TMS)).view(-1, 1).float()
        elif obsi == 'binding_BW2':
            y0i = torch.tensor(binding_formula(df, 'BW2', TMS)).view(-1, 1).float()
        elif obsi == 'binding_PySR':
            y0i = torch.tensor(PySR_formula(dfE, TMS)).view(-1, 1).float()
        elif obsi == 'Z':
            y0 = torch.tensor(df["z"].values).view(-1, 1).float()
        elif obsi == 'N':
            y0i = torch.tensor(df["n"].values).view(-1, 1).float()
        else:
            # turn missing measurements to zero
            dfobs = getattr(df, obsi)
            df[obsi] = dfobs.apply(lambda x: 0 if (x == ' ') or (
            x == 'STABLE') or (x == '?') else x)

            # turn values estimated by trends of the mass surface (TMS) to zero
            if (obsi == 'binding') and (TMS != 'TMS'):
                df[obsi] = df.apply(lambda x: 0 if (x['binding_sys'] == 'Y') or (
                    x['binding_unc']*(x['z']+x['n']) > 100) else x[obsi], axis=1)

            y0i = torch.tensor(df[obsi].astype(float).values).view(-1, 1).float()
            y0i[torch.isnan(y0i)] = 0

            if obsi in ['binding', 'qbm', 'sn', 'sp', 'qa', 'qec', 'energy']:
                y0i = y0i/1000  # keV to MeV

        if isinstance(y0, torch.Tensor):
            y0 = torch.cat((y0, y0i), dim=1)
        else:
            y0 = y0i

    heavy_mask = (X[:, 0] > heavy_elem) & (X[:, 1] > heavy_elem)
    X = X[heavy_mask]
    y0 = y0[heavy_mask]
    y0_mean = y0.mean(dim=0)
    y0_std = y0.std(dim=0)

    y = (y0 - y0_mean)/y0_std.unsqueeze(0).expand_as(y0)

    X_train, X_test, y_train, y_test, y0_train, y0_test = train_test_split(X, y, y0, test_size=0.2, random_state=(10))
    return [X_train, X_test], [y_train, y_test], [y0_train, y0_test], [y0_mean, y0_std], vocab_size

def rms_val(y_dat, y_pred):  # returns the value of rms in MeV
    return np.sqrt(((y_dat-y_pred)**2).sum()/y_dat.size()[0])

def rms(model, obs, heavy_mask, TMS_trained, TMS, incl):  
    
    #TMS_trained denotes whether (TMS_trained = 'TMS') or not (TMS_trained = 'off') the model was trained w/wo TMS
    (X_train, X_test), _, _, (y0_mean, y0_std), _ = get_data(obs,heavy_mask,TMS_trained)
    #TMS denotes wether the rms is calculated on the data w/wo TMS
    (X_train_r, X_test_r), _, (y0_train_r, y0_test_r), (y0_mean_r, y0_std_r), _ = get_data(obs,heavy_mask,TMS)
    
    #you can also check the performance of BW and BW2
    if model == 'BW2': 
        _, _, (y0_train_emp, y0_test_emp), _, _ = get_data(['binding_BW2'],heavy_mask,TMS_trained)
    elif model == 'BW':
        _, _, (y0_train_emp, y0_test_emp), _, _ = get_data(['binding_BW'],heavy_mask,TMS_trained)        
    
    # with incl you can choose if you want to calculate the rms on the all data or only on the test data   
    if incl=='test':
        X = X_test_r
        y0_dat = y0_test_r
    else:
        X = torch.cat((X_train_r,X_test_r), 0)
        y0_dat = torch.cat((y0_train_r,y0_test_r), 0)
        
    rms_tab = []

    for pos_obsi in range(len(obs)):
            
        if isinstance(model, str):
            if incl=='test':
                y0_pred = y0_test_emp
            else:
                y0_pred = torch.cat((y0_train_emp, y0_test_emp), 0)
        else:
            y_pred = model(X)
            y0_pred = yorig(y_pred,y0_mean, y0_std)
            
        (yi_dat, yi_pred) = clear_y(y0_dat, y0_dat, y0_pred, pos_obsi)
        
        Xi = clear_x(y0_dat, X, pos_obsi)
        Ai = Xi.sum(dim=1).view(-1,1)
        if ((obs[pos_obsi] == 'binding') or (isinstance(model, str))):
            yi_dat_tot = yi_dat*Ai
            yi_pred_tot = yi_pred*Ai
            #rms for both full Eb and Eb/nucleon are calculated
            rms_tab.append([rms_val(yi_dat_tot,yi_pred_tot).item(),rms_val(yi_dat,yi_pred).item()])
        else:    
            rms_tab.append(rms_val(yi_dat,yi_pred).item())
        
    return rms_tab 


def dev_rel(y_dat, y_pred):  # returns the relative deviation MeV
    return ((y_dat-y_pred)/y_dat).abs().median()

def Sn(model, heavy_mask):  #UNDER CONSTRUCTION
    (X_train, X_test), _, (y0_train, y0_test), (y0_mean,
                                                y0_std), _ = get_data('data', ['binding', 'sn'], heavy_mask)

    X = torch.cat((X_train, X_test), 0)
    y0_dat = torch.cat((y0_train, y0_test), 0)
    X = clear_x(X, y0_dat, 1)
    y0_dat, _ = clear_y(y0_dat, y0_dat, y0_dat, 1)

    X_all = X.clone()
    for i in range(X.size(0)):
        a, b = X[i]

        # check if [a, b-1] exists in X
        if not torch.any(torch.all(X == torch.tensor([a, b-1]), dim=1)):
            # if [a, b-1] does not exist, append it to the end of X
            X_all = torch.cat((X_all, torch.tensor([[a, b-1]])), dim=0)
    A = (X_all[:, 0]+X_all[:, 1])

    Eb = yorig(model(X_all)[:, 0].detach(), y0_mean[0], y0_std[0])
    Eb = Eb*A
    Sn = torch.tensor([get_y_for_ZN(X_all, Eb, Z, N) -
                      get_y_for_ZN(X_all, Eb, Z, N-1) for Z, N in X])
    Sn = Sn.view(-1, 1)

    rms = np.sqrt(((y0_dat-Sn)**2).sum()/y0_dat.size()[0])

    return y0_dat, Sn, rms


def DeltaE_dat(heavy_mask, obsi): #UNDER CONSTRUCTION
    (X_train, X_test), _, (y0_train, y0_test), (y0_mean,y0_std), _ = get_data(['binding', obsi], heavy_mask)

    X = torch.cat((X_train, X_test), 0)
    y0_dat = torch.cat((y0_train, y0_test), 0)

    def E(Z, N, i):
        try:
            ind = torch.where(torch.all(X == torch.tensor([Z, N]), dim=1))[
                0].item()
            return y0_dat[:, i][ind]
        except ValueError:
            return 0

    DeltaE_theory = y0_dat[:, 1].clone()
    for i in range(X.size(0)):
        Z, N = X[i]

        # check if [a, b-1] exists in X
        if E(Z, N-1, 0) != 0 and E(Z, N, 0) != 0:
            DeltaE_theory[i] = (Z+N)*E(Z, N, 0)-(Z+N-1)*E(Z, N-1, 0)
        else:
            DeltaE_theory[i] = 0

    DetlaEi_theory = DeltaE_theory[DeltaE_theory != 0].view(-1, 1)
    y0i_dat = y0_dat[:, 1][DeltaE_theory != 0].view(-1, 1)
    error = DetlaEi_theory-y0i_dat

    rms = np.sqrt(((error)**2).sum()/y0_dat[:, 1].size()[0])

    return DeltaE_theory, rms
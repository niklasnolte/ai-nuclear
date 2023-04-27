import os
import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split
import math
from formulae import binding_formula, radius_formula


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




def get_data(obs, heavy_elem, TMS):
    np.random.seed(1)

    def lc_read_csv(url):
        req = urllib.request.Request(
            "https://nds.iaea.org/relnsd/v1/data?" + url)
        req.add_header(
            'User-Agent', 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0')
        return pd.read_csv(urllib.request.urlopen(req))

    df = lc_read_csv("fields=ground_states&nuclides=all")
    dfE = pd.read_csv('tables/ame2020.csv')

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
        elif obsi in ['BW', 'BW2', 'WS4']:
            y0i = torch.tensor(binding_formula(df, obsi, TMS)).view(-1, 1).float()
        elif obsi == 'Z':
            y0 = torch.tensor(df["z"].values).view(-1, 1).float()
        elif obsi == 'N':
            y0i = torch.tensor(df["n"].values).view(-1, 1).float()
        else:
            # turn missing measurements to zero
            dfobs = getattr(df, obsi)
            df[obsi] = dfobs.apply(lambda x: 0 if (x == ' ') or (
            x == 'STABLE') or (x == '?') else x)
            # df[obsi] = df.apply(lambda x: 0 if (x[obs] == ' ') or (x[obs] == 'STABLE') or (x[obs] == '?') else x[obs], axis=1)
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
    
    # with incl you can choose if you want to calculate the rms on the all data or only on the test data   
    if incl=='test':
        X = X_test_r
        y0_dat = y0_test_r
    else:
        X = torch.cat((X_train_r,X_test_r), 0)
        y0_dat = torch.cat((y0_train_r,y0_test_r), 0)
        
    rms_tab = []

    for pos_obsi in range(len(obs)):
            
        if model in ['BW', 'BW2', 'WS4']:
            #you can also check the performance of the semf
            
            _, _, (y0_train_emp, y0_test_emp), _, _ = get_data([model],heavy_mask,TMS_trained)        
            if incl=='test':
                y0_pred = y0_test_emp
            else:
                y0_pred = torch.cat((y0_train_emp, y0_test_emp), 0)
        else:
            y_pred = model(X)
            y0_pred = yorig(y_pred,y0_mean, y0_std)
            
        
        (yi_dat, yi_pred) = clear_y(y0_dat, y0_dat, y0_pred, pos_obsi)

        Xi = clear_x(y0_dat, X, pos_obsi)
        # Xi = clear_x(yi_pred, Xi, pos_obsi)
        
        Ai = Xi.sum(dim=1).view(-1,1)
        if ((obs[pos_obsi] == 'binding') or (obs in ['BW', 'BW2', 'WS4'])):
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
    (X_train, X_test), _, (y0_train, y0_test), (y0_mean, y0_std), _ = get_data('data', ['binding', 'sn'], heavy_mask)

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


def rms_WS4_ME():
    
    file_path = os.path.join('tables/WS4_full.txt')
    df = pd.read_csv(file_path, skiprows=1, header=None, names=['A', 'Z', 'Beta2', 'Beta4', 'Beta6', 'Esh', 'Edef', 'Eexp', 'Eth', 'Mexp', 'Mth'], delim_whitespace=True)
    
    # Calculate Eexp and Eth
    Da = 931.494102
    mp = 938.2720813
    mn = 939.565420
    df["Eexp"] = df["Z"]*mp + (df["A"]-df["Z"])*mn - df["Mexp"] - df["A"]*Da
    df["Eth"] = df["Z"]*mp + (df["A"]-df["Z"])*mn - df["Mth"] - df["A"]*Da
    
    # Calculate RMS between Eexp and Eth, excluding cases where Mexp is 0
    Eexp_values = df.loc[df["Mexp"] != 0, "Eexp"].values
    Eth_values = df.loc[df["Mexp"] != 0, "Eth"].values
    rms = np.sqrt(np.mean((Eexp_values - Eth_values)**2))
    
    print(rms)


# def delta_np(Z, N):
#     A = Z+N
#     a_pair = -5.8166
#     delta0 = a_pair/A**(1/3)
#     I=(N-Z)/A
#     for i in range(len(A)):
#         if ((N % 2 == 0) & (Z % 2 == 0))[i]:
#             delta0[i] = delta0[i]*(2 -  abs(I) - I**2)*(17/16)
#         elif ((N % 2 == 1) & (Z % 2 == 1))[i]:
#             delta0[i] = delta0[i]*(abs(I) - I**2)
#         else:
#             delta0[i] = 0
#     return delta0
# def E_WS(df):
#     N = df["n"].values
#     Z = df["z"].values
#     A = N+Z 
#     I=(N-Z)/A
    
#     aV = -15.51
#     aS = 17.4
#     aC = 0.709
#     c_sym = 30.159
#     kappa = 1.518
#     ksi = 1.223
#     cW = 0.87
#     kappa_s = 0.1536
#     g1 = 0.01
#     g2 = -0.5
#     V0 = -45.856
#     r0 = 1.38
#     a0 = 0.764
#     lambda0 = 26.479
#     c1 = 0.63
#     c2 = 1.337
#     kappa_d = 5.008
    
    
#     I0 = 0.4*A/(A + 200)
#     eps = (I-I0)**2-I**4    
#     fs = 1 + kappa_s*eps*A**(1/3)    
#     a_sym = -c_sym*(1-kappa/A**(1/3)+ksi*(2-abs(I))/(2+abs(I)*A))
    
#     eta = DeltaZ*DeltaN/Zm
#     Delta_W = cW*(np.exp(abs(I))-np.exp(-abs(eta)))
    
#     E_LD = aV*A + aS*A**(2/3) + aC*(1-0.76*Z**(-2/3))*Z**2/(A**(1/3)) + \
#         a_sym*fs*A*I**2 +  delta_np(Z, N) + Delta_W
        
#     Eb = E_LD
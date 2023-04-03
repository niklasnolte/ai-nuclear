import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, SymLogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from data import get_data, yorig, rms, clear_x, clear_y
from data import get_y_for_ZN, Sn, DeltaE_dat

#latex names for the obs and their values
latex_obs = {'radius': r'$\left<r\right>$',
         'binding': r'$E_B$',
         'qbm': r'$Q_{\beta}$',
         'sn': r'$S_n$',
         'sp': r'$S_p$'}
latex_unit = {'binding': '[GeV]',
         'radius': '[fm]',
         'qbm': '[GeV]',
         'sn': '[GeV]',
         'sp': '[GeV]'}

def get_pred(model, obs, pos_obsi, heavy_mask, TMS, dati):
    (X_train, X_test), (y_train, y_test), (y0_train, y0_test), (y0_mean, y0_std), vocab_size = get_data(obs,heavy_mask,TMS)
    
    X = torch.cat((X_train,X_test), 0)
    y0_dat = torch.cat((y0_train,y0_test), 0)
        
    y_pred_train = model(X_train).detach() 
    y_pred_test = model(X_test).detach() 
    y_pred = model(X).detach() 
    
    y0_pred_train = yorig(y_pred_train,y0_mean,y0_std)
    y0_pred_test = yorig(y_pred_test,y0_mean,y0_std)
    y0_pred = yorig(y_pred,y0_mean,y0_std)
    

    Xi_train = clear_x(y0_train, X_train, pos_obsi)
    Xi_test = clear_x(y0_test, X_test, pos_obsi)
    Xi = clear_x(y0_dat, X, pos_obsi)
    (y0i_train, y0i_pred_train) = clear_y(y0_train, y0_train, y0_pred_train, pos_obsi)
    (y0i_test, y0i_pred_test) = clear_y(y0_test, y0_test, y0_pred_test, pos_obsi)
    (y0i_dat, y0i_pred) = clear_y(y0_dat, y0_dat, y0_pred, pos_obsi)
        
    if dati == 'all':
        return [X_train,X_test,X], [y0_train, y0_test, y0_dat], [y0_pred_train, y0_pred_test, y0_pred]
    else:
        return [Xi_train,Xi_test,Xi], [y0i_train, y0i_test, y0i_dat], [y0i_pred_train, y0i_pred_test, y0i_pred]


def plot_obsi(X, y, obsi, obs, alpha):
    
    pos_obsi = obs.index(obsi)
    
    plt.scatter(X, y, alpha)
    
    # create a custom legend with alpha=1 only for the second label
    custom_legend = [
        Line2D([0], [0], color='C0', marker='o', linestyle=''),
        Line2D([0], [0], color='C1', marker='o', alpha=1, linestyle='')
    ]
    plt.legend(custom_legend, ['data', 'NN trained on '+' , '.join([latex_obs[key] for key in obs if key in latex_obs])], loc='lower right')

    
    # set axis labels and show the plot
    plt.ylabel(latex_obs[obsi] + ' ' + latex_unit[obsi])
    plt.xlabel('Atomic mass number ' + r'$A$')    



def heat_plot(X, y0_dat, y0_pred, obsi):

    if obsi=='binding':
        scale = 1000
    else:
        scale = 1
    Z = X[:,0].view(-1).numpy()
    N = X[:,1].view(-1).numpy()

    Deltay = (y0_dat - y0_pred).view(-1).numpy()
    # DeltaE = (y0i_dat-y0_emp).view(-1).numpy()
    
    to_plot = np.empty((Z.shape[0],Z.shape[0]))
    to_plot[:] = np.nan
    
    for i in range(Z.shape[0]):
        to_plot[Z[i],N[i]] = scale*Deltay[i]
    
    plt.imshow(to_plot, cmap='coolwarm', interpolation='hanning')
    
    # set the colorbar limits
    plt.clim(-0.05*scale, 0.15*scale)
    
    # create a colorbar and set the label
    plt.colorbar(label=r'$\Delta$' + latex_obs[obsi] + ' ' + latex_unit[obsi])
    
    # set the x and y labels and title
    plt.xlabel('Number of neutrons ' + r'$N$')
    plt.ylabel('Number of protons ' + r'$Z$')
    plt.title('Discrepancy ' + r'$obsi^{{\rm exp}}-obsi^{{\rm SE}}$'.format(obsi).replace('obsi', latex_obs[obsi][1:-1]))
    
    # set the x and y limits
    plt.xlim(8, 140)
    plt.ylim(8, 90)
    
     
    magic = [2, 8, 20, 28, 50, 82, 126]
    
    for m in magic:
        for n in magic:
            if ([n,m] in X.tolist()):
                if m == magic[0] and n ==magic[0]:
                    plt.plot(m, n, 'kx', markersize=5, label='magic')
                else:
                    plt.plot(m, n, 'kx', markersize=5)
    
    
    
    

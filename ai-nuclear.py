import os
import time
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
from train import train_model
from model import Model2, Model22
from inspect_results import get_pred, plot_obsi, heat_plot


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step
        
        
        obs = ['binding', 'radius']
        #obs = ['half_life', 'sn','sp', 'qbm', 'z', 'n', 'abundance']
        heavy_mask = 8
        TMS = 'TMS'


        basepath = "models/"+'+'.join(obs)+f'({TMS})/'

        if not os.path.exists(basepath):
            os.makedirs(basepath)

        # define the observable you want to plot
        obsi = 'radius'
        pos_obsi = obs.index(obsi)
        model = torch.load(basepath+"model.pt")

def AI_nuclear(task, model, obs, obsi, heavy_mask, TMS):
    
    (X_train, X_test, X), (y0_train, y0_test, y0_dat), (y0_pred_train, y0_pred_test, y0_pred) = get_pred(model, obs, pos_obsi, heavy_mask, TMS, 'all')
    (Xi_train, Xi_test, Xi), (y0i_train, y0i_test, y0i_dat), (y0i_pred_train, y0i_pred_test, y0i_pred) = get_pred(model, obs, pos_obsi, heavy_mask, TMS, 'measure_only')
    
    # (X_train_emp, X_test_emp), _, (y0_train_emp, y0_test_emp), _, _ = get_data(['binding_BW2'], heavy_mask, 'TMS')

    if task == 'train':
        # for lr in drange(1e-4, 1e-3, 1e-4):
        #     for wd in drange(1e-5, 1e-4, 1e-5):
        #         print (lr,wd)
        #         train(Model22, lr=lr, wd=wd, alpha=1, hidden_dim=64, n_epochs=3e3, basepath="models/test1", device=torch.device("cpu"))
        
        start = time.time()
        train_model(Model2, lr=0.0028 , wd=0.00067, alpha=0, hidden_dim=64, n_epochs=1e3, obs = obs, heavy_mask = heavy_mask, TMS=TMS, basepath=basepath, device=torch.device("cpu"))
        stop = time.time()
        print(f"Training time: {stop - start}s")
        
        # bfp : lr=0.0028 , wd=0.00067
    
    elif task == 'print_rms':
        rms_tab = rms(model, obs, heavy_mask, TMS, 'off', 'test')
        i = 0
        for obsi in obs:
            if obsi == 'binding':
                print('rms NN '+obsi+': '+str(rms_tab[i][0]))  
                print('rms NN '+obsi+'/nucleon: '+str(rms_tab[i][1]))  
            else:
                print('rms '+obsi+': ',rms_tab[i])
            i += 1
        
        # print('rms BW '+'binding'+': '+str(rms('BW', ['binding'], heavy_mask, 'TMS', 'off', 'test')[0][0]))
        # print('rms BW '+'binding'+'/nucleon: '+str(rms('BW', ['binding'], heavy_mask, 'TMS', 'off', 'test')[0][1]))
        # print('rms BW2 '+'binding'+': '+str(rms('BW2', ['binding'], heavy_mask, 'TMS', 'off', 'test')[0][0]))
        # print('rms BW2 '+'binding'+'/nucleon: '+str(rms('BW2', ['binding'], heavy_mask, 'TMS', 'off', 'test')[0][1]))
        print('rms WS4 '+'binding'+': '+str(rms('WS4', ['binding'], heavy_mask, 'TMS', 'off', 'test')[0][0]))
        print('rms WS4 '+'binding'+'/nucleon: '+str(rms('WS4', ['binding'], heavy_mask, 'TMS', 'off', 'test')[0][1]))

        # print(rms(model, obs, heavy_mask, 'TMS', 'TMS', 'test'))
        # print(rms('BW', ['binding'], heavy_mask, 'TMS', 'TMS', 'test'))
        # print(rms('BW2', ['binding'], heavy_mask, 'TMS', 'TMS', 'test'))
        
    elif task == 'plot_obs':
        
        plot_obsi(Xi.sum(dim=1), y0i_dat, obsi, obs, alpha=0.5)
        plot_obsi(X.sum(dim=1), y0_pred[:,pos_obsi], obsi, obs, alpha=0.2)
        # plt.savefig('plots/plot_'+obsi+'_trained_'+'+'.join(obs)+'.png', dpi=300)
        plt.show()
    
    elif task == 'plot_heat':
        
        heat_plot(Xi, y0i_dat, y0i_pred, obsi)
        # plt.savefig('plots/heat_'+obsi+'_trained_'+'+'.join(obs)+'.png', dpi=300)
        plt.show()


# define the observables, mask, and whether to train with TMS
obs = ['binding', 'radius']
#obs = ['half_life', 'sn','sp', 'qbm', 'z', 'n', 'abundance']
heavy_mask = 8
TMS = 'TMS'

# create and save model/plots
basepath = "models/"+'+'.join(obs)+f'({TMS})/'
if not os.path.exists(basepath):
    os.makedirs(basepath)

# define the observable you want to plot
obsi = 'binding'
pos_obsi = obs.index(obsi)
model = torch.load(basepath+"model.pt")

AI_nuclear('print_rms', model, obs, obsi, heavy_mask, TMS)




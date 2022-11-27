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

from torch.utils.data import DataLoader, TensorDataset
import tqdm
import os




def get_models(paths, epoch = None):
    models = []
    for i in range(len(paths)):
        path = paths[i]
        if epoch is None:
            epoch = 'best.pt'
        sd = torch.load(f"{path}/{epoch}")
        models.append(torch.load(f"{path}/model.pt"))
        models[i].load_state_dict(sd)
    return models

def test_model(model, X_test, y_test, state_dict = None):
    if state_dict is None:
        state_dict = deepcopy(model.state_dict())
    starting = deepcopy(model.state_dict())
    model.load_state_dict(state_dict)
    loss_fn = nn.MSELoss()
    y_pred = model(X_test)
    loss = loss_fn(y_pred, y_test)
    model.load_state_dict(starting)
    return loss

def plot_epochs(titles):
    dfs = []
    for title in titles:
        dfs.append(pd.read_csv(title))

    metrics = ['Loss', 'Proton_Entropy', 'Neutron_Entropy']
    n1=20
    n2 = 200
    figsize = (3,1)
    fig, axs = plt.subplots(nrows=3, figsize = tuple(5*i for i in figsize), sharex=False)
    for i in range(len(metrics)):
        ax, metric = axs[i], metrics[i]
        if i==0:
            ax.set_yscale('log')
        ax.title.set_text(metric)
        for i in range(len(dfs)):
            df = dfs[i]
            title = titles[i]
            ax.plot(df['Iteration'][n1:n2], df[metric][n1:n2], label = title)
        ax.legend()
        ax.set_xlabel('Epoch')
    plt.show()

def get_index(hidden_dim = 64, alpha = -1.05, plot_dist = True):
    indices = torch.tensor(range(1,hidden_dim+1))
    base = torch.tensor([ind**alpha for ind in range(1,hidden_dim +1)])
    base = base/base.sum()
    sample = torch.multinomial(base, 1, replacement=True)
    
    if plot_dist:
        samples = torch.multinomial(base, 10000, replacement=True)
        cdf = torch.tensor([base[:n].sum() for n in range(hidden_dim+1)])
        midpoint = (((cdf>0.5) == True).nonzero(as_tuple=True)[0][0])
        base = base/base.sum()
        
        plt.axvline(x = midpoint, linestyle = '--', c= 'r', label = f'center of mass = {midpoint}')
        plt.legend()
        plt.hist(samples, bins = 64)
        plt.show()

    return indices[sample]

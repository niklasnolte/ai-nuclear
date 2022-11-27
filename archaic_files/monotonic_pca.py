import torch
from torch import nn
import matplotlib.pyplot as plt
from BasicModel import BasicModelSmaller
from base_functions import get_data 
from archaic_files.pca_inspection import get_models
from archaic_files.pca import test_model, test_raw_model
import os


def plot_embeddings(model, X_test, y_test, title, dim = 64):
    ndims = list(range(1,dim+1,3))
    loss_fn = nn.MSELoss()
    y_pred = model(X_test)
    actual_loss = loss_fn(y_pred, y_test)
    losses = [loss_fn(model.evaluate_ndim(X_test, device = 'cpu', n = n), y_test) for n in ndims]
    losses = [i.detach().numpy() for i in losses]
    
    plt.plot(ndims, losses)
    plt.xlabel('Dimension of PCA taken to compute test loss')
    plt.ylabel('Test Loss')
    plt.title(f'Embedding Space for {title} \n Loss = {actual_loss:.4f}')
    plt.show()

def plot_model_epochs():
    
    dim = 64
    X_train, X_test, y_train, y_test, vocab_size = get_data()
    all_protons = torch.tensor(list(range(vocab_size[0])))
    all_neutrons = torch.tensor(list(range(vocab_size[1])))
    heavy_elem = 15
    test_heavy_mask = X_test[:,0]>heavy_elem
    X_test = X_test[test_heavy_mask]
    y_test = y_test[test_heavy_mask]
    directory = 'models/pcareg_heavy15'
    title = 'BasicModelSmaller_regpca0'
    
    for i in range(0,29900, 500):
        file = f'epoch{i}.pt'
        sd = torch.load(f"{directory}/{title}/{file}")
        model = torch.load(f"{directory}/{title}/model.pt")
        model.load_state_dict(sd)
        loss_fn = nn.MSELoss()
        y_pred = model(X_test)
        plot_embeddings(model, X_test, y_test, title = title+'\n'+file, dim = dim)
        
        

def plot_random():
    p, n, dim = 119, 176, 64
    X_train, X_test, y_train, y_test, (p,n) = get_data()
    model = BasicModelSmaller(p, n, dim)
    plot_embeddings(model, X_test, y_test, title = 'Randomly Initialized Model', dim = dim)


if __name__ == '__main__':
    plot_model_epochs()
    #plot_model_epochs()
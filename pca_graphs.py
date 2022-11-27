import torch
from torch import nn
import matplotlib.pyplot as plt
from BasicModel import BasicModelSmaller
from train_model import get_data
from base_functions import get_models, test_model, plot_epochs  
import os
from sklearn.decomposition import PCA
from copy import deepcopy
import matplotlib.colors as mcolors




def onedim_pca(model, all_protons, all_neutrons):
    p_n_embed= [model.emb_proton(all_protons), model.emb_neutron(all_neutrons)]
    p_n = [all_protons, all_neutrons]

    figsize = (2,1)
    fig, axs = plt.subplots(nrows=2, figsize = tuple(5*i for i in figsize), sharex=False)

    for i in range(2):
        U, _, _ = torch.linalg.svd(p_n_embed[i])
        axs[i].plot(p_n[i], U[:,0], c='k', linewidth = 0.2)
        axs[i].scatter(p_n[i], U[:, 0], c = U[:, 0], cmap="coolwarm")
        for j in range(U.shape[0]):
            if j%2 == 0:
                axs[i].annotate(1+p_n[i][j].item(), (p_n[i][j], U[j, 0]))

        if i == 0:
            axs[i].set_xlabel('proton number')
        elif i == 1:
            axs[i].set_xlabel('neutron number')
        axs[i].set_ylabel('representation value')
        #axs[i].set_ylim(-0.2, 0.2)
    fig.suptitle('One Dimensional Proton/Neutron Representation')
    plt.show()

def twodim_pca(model, all_protons, all_neutrons, title = None):
    protons = model.emb_proton(all_protons)
    neutrons = model.emb_neutron(all_neutrons)
    for p, ap in zip((protons, neutrons), (all_protons, all_neutrons)):
        plt.figure(figsize=(10,10))
        pca = PCA(n_components=2)
        embs_pca = pca.fit_transform(p.detach().cpu().numpy())
        pca_var = pca.fit(p.detach().cpu().numpy()).explained_variance_ratio_
        print(pca_var)
        plt.xlabel(f'{100*pca_var[0]:.2f}% of variance')
        plt.ylabel(f'{100*pca_var[1]:.4f}% of variance')
        plt.scatter(*embs_pca.T, c=ap, cmap="coolwarm")
        plt.plot(*embs_pca.T,c = 'k', linewidth = 0.2)
        #annotate
        for i, txt in enumerate(ap):
            plt.annotate(txt.item(), (embs_pca[i,0], embs_pca[i,1]))
        graph_title = "protons 2 component PCA analysis" if p is protons else "neutrons 2 component PCA analysis"
        if title is not None:
            graph_title = f'{title} \n{graph_title}'
        plt.title(graph_title)
    plt.show()

 

def plot_embeddings_loss(model, X_test, y_test, title, dim = 64):
    # plots the loss for 1 to n dimensions used in the embeddings of a single model
    ndims = list(range(1,dim+1,1))
    loss_fn = nn.MSELoss()
    y_pred = model(X_test)
    actual_loss = loss_fn(y_pred, y_test)
    losses = [model.evaluate_ndim(loss_fn, X_test, y_test, device = 'cpu', n = n) for n in ndims]
    losses = [i.detach().numpy() for i in losses]
    
    plt.plot(ndims, losses)
    plt.xlabel('Dimension of PCA taken to compute test loss')
    plt.ylabel('Test Loss')
    plt.title(f'Embedding Space for {title} \n Loss = {actual_loss:.4f}')
    plt.show()

def plot_embeddings_loss_epochs(title):
    #plot embeddings_loss over all epochs of a model (to show progression)
    dim = 64
    heavy_elem = 15
    X_train, X_test, y_train, y_test, vocab_size = get_data(heavy_elem = heavy_elem)
    
    for i in range(0,29900, 500):
        file = f'epoch{i}.pt'
        sd = torch.load(f"{title}/{file}")
        model = torch.load(f"{title}/model.pt")
        model.load_state_dict(sd)
        plot_embeddings_loss(model, X_test, y_test, title = title+'\n'+file, dim = dim)

def effective_dim_embedding(model, X_test, y_test, all_protons, all_neutrons, heavy_elem = 15):
  #calculate entropy of the embeddings
  protons = model.emb_proton(all_protons)
  neutrons = model.emb_neutron(all_neutrons)
  U_p, S_p, Vh_p = torch.linalg.svd(protons, False)
  U_n, S_n, Vh_n = torch.linalg.svd(neutrons, False)

  loss_nd = [0 for i in range(len(S_p))]

  original = model.state_dict().copy()
  actual_loss = test_model(model, X_test, y_test, original)
  for i in range(len(S_p)):
    index = len(S_p)-i
    S_p[index:] = 0
    S_n[index:] = 0
    nd_state = model.state_dict()
    nd_state['emb_proton.weight'] =  U_p @ torch.diag(S_p) @ Vh_p
    nd_state['emb_neutron.weight'] =  U_n @ torch.diag(S_n) @ Vh_n
    loss_nd[index-1] = test_model(model, X_test, y_test, nd_state)
    model.load_state_dict(original)

  return actual_loss, loss_nd

def effective_dim_final(model, X_test, y_test, heavy_elem = 15):
  #calculate entropy of the embeddings

  original_statedict = model.state_dict().copy()
  keys = list(original_statedict.keys())
  final_layer_key = keys[-2]
  final_layer = original_statedict[final_layer_key]
  print(final_layer)
  U, S, V = torch.linalg.svd(final_layer, False)

  loss_nd = [0 for i in range(len(S))]
  
  actual_loss = test_model(model, X_test, y_test, original_statedict)

  for i in range(len(S)):
    index = len(S)-i
    S[index:] = 0
    nd_state = model.state_dict()
    interim = U @ torch.diag(S) @ V
    print(interim)
    nd_state[final_layer_key] =  interim
    loss_nd[index-1] = test_model(model, X_test, y_test, nd_state)
    model.load_state_dict(original_statedict)

  return actual_loss, loss_nd

def compare_effective_dims(titles, position = 'embedding',epoch = None):
    models = get_models(titles, epoch = epoch)
    _, X_test, _, y_test, vocab_size = get_data()
    all_protons = torch.tensor(range(vocab_size[0]))
    all_neutrons = torch.tensor(range(vocab_size[1]))
    colors = list(mcolors.TABLEAU_COLORS)
    for i in range(len(models)):
        model = models[i]
        title = titles[i]
        color = colors[i]
        if position == 'embedding':
            actual_loss, loss_nd = effective_dim_embedding(model,X_test, y_test, all_protons, all_neutrons, heavy_elem = 15)
        elif position == 'final':
            actual_loss, loss_nd = effective_dim_final(model, X_test, y_test, heavy_elem = 15)
        dims = range(1, len(loss_nd)+1)
        plt.plot(dims, loss_nd, c = color)
        plt.axhline(actual_loss, label = title, c = color, linestyle = '--')
        print(f'{title}\nactual loss {actual_loss}, loss_5 {loss_nd[4]}\n')

    plt.yscale('log')
    xmin = 1
    xmax = 45
    plt.xlim(xmin,xmax)
    plt.xticks(ticks = list(range(xmin, xmax)))
    plt.legend()
    plt.title('Loss at given Dimension')
    plt.xlabel('Dimension of Embedding')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    regs = [0, 2e-4, 2e-3, 2e-2, 2e-1, 2e0]#[2, 1, 2e-1]
    regs = [2e-1]
    vals = ['dimall', 'dimn', 'oldeff', 'dim3', 'dim6']
    vals = [vals[3]]
    titles = []
    for j in range(len(vals)):
        for i in range(len(regs)):
            val = vals[j]
            reg = regs[i]
            print(val, reg)
            title = f'BasicModelSmaller_regpca{reg}_{val}'
            path = f"models/pcareg_heavy15/{title}/"
            titles.append(path)
    #model = BasicModelSmaller(100, 100, 64)
    #keys = ['emb_proton.weight', 'emb_neutron.weight', 'nonlinear.1.weight', 'nonlinear.1.bias', 'nonlinear.3.weight', 'nonlinear.3.bias']
    compare_effective_dims(titles, position = 'embedding', epoch = 'best.pt')

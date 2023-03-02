import torch
from torch import nn
import matplotlib.pyplot as plt
from BasicModel import BasicModelSmaller, BasicModel, BasicModelSmall
from train_model import get_data
from base_functions import get_models, test_model, plot_epochs  
import os
from sklearn.decomposition import PCA
from copy import deepcopy
import matplotlib.colors as mcolors
from moviepy.video.io.bindings import mplfig_to_npimage
from BasicModelSmallOnly import BasicModelSmallOnly





def onedim_pca(model, all_protons, all_neutrons):
    #plots one dimensional pca. Not useful
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

def twodim_pca(model, title = None, heavy_elem = 15):
    #takes in model (not title) and plots the twodim pca graph
    _, X_test, _, y_test, (psize, nsize) = get_data(heavy_elem=heavy_elem)
    all_protons = torch.tensor(list(range(psize)))
    all_neutrons = torch.tensor(list(range(nsize)))
    protons = model.emb_proton(all_protons)
    neutrons = model.emb_neutron(all_neutrons)
    loss = test_model(model, X_test, y_test)
    for p, ap in zip((protons, neutrons), (all_protons, all_neutrons)):
        print(p.shape)
        plt.figure(figsize=(10,10))

        pca = PCA(n_components=2)
        embs_pca = pca.fit_transform(p.detach().cpu().numpy())
        pca_var = pca.fit(p.detach().cpu().numpy()).explained_variance_ratio_
        plt.xlabel(f'{100*pca_var[0]:.2f}% of variance')
        plt.ylabel(f'{100*pca_var[1]:.4f}% of variance')
        plt.scatter(*embs_pca.T, c=ap, cmap="coolwarm")
        plt.plot(*embs_pca.T,c = 'k', linewidth = 0.2)
        #annotate
        print(ap)
        print(ap.shape)
        for i, txt in enumerate(ap):
            plt.annotate(heavy_elem+txt.item(), (embs_pca[i,0], embs_pca[i,1]))
        graph_title = "protons 2 component PCA analysis" if p is protons else "neutrons 2 component PCA analysis"
        if title is not None:
            graph_title = f'{title}\n{graph_title}\ntest loss = {loss:.4f}'
        plt.title(graph_title)
    plt.show()

 

def plot_embeddings_loss(model, X_test, y_test, title, dim = 64):
    # plots the loss for 1 to n dimensions used in the embeddings of a single model
    # expects model, not title
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
    #its like 20 graphs that open one after another
    dim = 64
    heavy_elem = 15
    X_train, X_test, y_train, y_test, vocab_size = get_data(heavy_elem = heavy_elem)
    
    for i in range(0,29900, 500):
        file = f'epoch{i}.pt'
        sd = torch.load(f"{title}/{file}")
        model = torch.load(f"{title}/model.pt")
        model.load_state_dict(sd)
        plot_embeddings_loss(model, X_test, y_test, title = title+'\n'+file, dim = dim)

def effective_dim_embedding(model, X_test, y_test, all_protons, all_neutrons):
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

def effective_dim_final(model, X_test, y_test):
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
    nd_state[final_layer_key] =  interim
    loss_nd[index-1] = test_model(model, X_test, y_test, nd_state)
    model.load_state_dict(original_statedict)

  return actual_loss, loss_nd

def compare_effective_dims(titles, parameters = None, epoch = None, plot = True):
    if epoch is None:
        epoch = 'best.pt'
    models = get_models(titles, epoch = epoch)
    heavy_elem = 15
    if parameters is None:
        X_train, X_test, y_train, y_test, vocab_size = get_data(heavy_elem = heavy_elem)
    else:
        X_train, X_test, y_train, y_test, vocab_size = parameters
    all_protons = torch.tensor(range(vocab_size[0]))
    all_neutrons = torch.tensor(range(vocab_size[1]))
    colors = list(mcolors.TABLEAU_COLORS)
    fig, axs = plt.subplots(2, figsize = (8,12), sharex = True)
    for i in range(len(models)):

        model = models[i]
        title = titles[i][:-1]
        title = title[len(title) - title[::-1].index('/'):]
        color = colors[i]
        loss_fn = nn.MSELoss()
        actual_loss_test, loss_nd_test = effective_dim_embedding(model,X_test, y_test, all_protons, all_neutrons)
        actual_loss_train, loss_nd_train = effective_dim_embedding(model,X_train, y_train, all_protons, all_neutrons)

        actual_losses = [actual_loss_test, actual_loss_train]
        loss_nds = [loss_nd_test, loss_nd_train]

        for j in range(len(actual_losses)):
            dims = range(1, len(loss_nd_test)+1)
            plot_loss_nd = [loss_nd.item() for loss_nd in loss_nds[j]]
            axs[j].plot(dims, plot_loss_nd, c = color)
            axs[j].axhline(actual_losses[j], label = f'{title} \n Loss = {actual_losses[j]:.2e}', c = color, linestyle = '--')
            #print(f'{title}\nactual loss {actual_losses[j]}, loss_5 {loss_nds[j][4]}\n')

    plt.yscale('log')
    xmin = 1
    xmax = 18
    train_range = (2*10**-5, 1.2*10**0)
    test_range = (2*10**-3, 1.2*10**0)
    for j in range(2):
        axs[j].set_xlim(xmin, xmax)
        
        axs[j].set_yscale('log')
        axs[j].legend(prop={'size': 8})
        #axs[j].legend()
    axs[0].set_ylim(test_range)
    axs[1].set_ylim(train_range)
    
    plt.xticks(ticks = list(range(xmin, xmax)))
    plt.subplots_adjust(hspace=0, wspace = 0)
    fulltitle = titles[0]
    base = fulltitle[fulltitle.index('/')+1:len(fulltitle)-len(title)+title.index('_')+1]
    modeltype = title[len(title) - title[::-1].index('_'):]
    fig.suptitle(f'{base}\n{modeltype}\nLoss at given Embedding Dimension\n{epoch}')
    axs[1].set_xlabel('Dimension of Embedding Used')
    axs[0].set_ylabel('Test Loss')
    axs[1].set_ylabel('Train Loss')
    if plot:
        plt.show()
    return mplfig_to_npimage(fig)

if __name__ == '__main__':
    '''
    regs = [0, 2e-4, 2e-3, 2e-2, 2e-1, 2e0, 5e0]#[2, 1, 2e-1]

    #regs = [0, 0.1, 0.5, 1, 5]
    vals = ['dimn', 'dimall', 'oldeff', 'dim3', 'dim6']
    vals = vals[:1]
    seeds = [1,25,30,31,50]
    titles = []
    for val in vals:
        for reg in regs:
                print(val, reg)
                title = f'BasicModel_regpca{reg}_{val}'
                path = f"models/pcareg_heavy15/{title}/"
                titles.append(path)
    print(titles)
    compare_effective_dims(titles, epoch = 'best.pt')
    '''
    heavy_elem = 15
    _, X_test, _, y_test, vocab_size = get_data(heavy_elem = heavy_elem)
    all_protons = torch.tensor(list(range(vocab_size[0])))
    all_neutrons = torch.tensor(list(range(vocab_size[1])))

    titles = []
    #for regpca in [0, 0.0002, 0.002, 0.02, 0.2, 2.0, 5.0]:
    #    titles.append(f'models/pcareg_heavy15/BasicModelSmall_regpca{regpca}_dimn')
    #models = get_models(['models/pcareg_heavy15/BasicModelSmallerDropout_regpca0_wd0'])
    #print(test_model(models[0], X_test, y_test))

    base = 'models/mod_arith/BasicModelSmall_regpca0_256dim_both'
    titles.append(base)
    model = get_models(titles)[0]
    model.plot_embedding()
    #for i in range(1):
    #    twodim_pca(models[i], titles[i])



   # compare_effective_dims(titles)
    



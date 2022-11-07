import torch
from data import get_data
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from BasicModel import BasicModel, BasicModelSmall, BasicModelSmaller, BasicModelReallySmall
from pca import effective_dim, regularize_effective_dim
from prettytable import PrettyTable  
import pandas as pd
# %%

def count_parameters(model, to_print = False):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    if to_print:
        print(table)
        print(f"Total Trainable Params: {total_params}")
    return total_params


# %%


# %%
# PCA the embeddings
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
  #plt.show()

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

def compare_onedim_pca(models, all_protons, all_neutrons, descriptions):

    
    proton_numbers = all_protons

    figsize = (len(models),1)
    fig, axs = plt.subplots(nrows=len(models), figsize = tuple(5*i for i in figsize), sharex=True)
    plt.subplots_adjust(wspace=None, hspace=0)
    for i in range(len(models)):
        model = models[i]
        proton_embed = model.emb_proton(all_protons)
        pca = PCA(n_components=1)
        embs_pca = pca.fit_transform(proton_embed.detach().cpu().numpy())
        embedding = embs_pca
        axs[i].plot(proton_numbers, embedding, c='k', linewidth = 0.2)
        axs[i].scatter(proton_numbers, embedding, c = embedding, cmap="coolwarm", label = descriptions[i])
        for j in range(embedding.shape[0]):
            if j%3 == 0:
                axs[i].annotate(1+proton_numbers[j].item(), (proton_numbers[j], embedding[j]))

        if i == len(models)-1:
            axs[i].set_xlabel('proton number')
        axs[i].set_ylabel('rep val')
        noble_gases = [2, 10, 18, 36, 54, 86]
        for j in range(len(noble_gases)):
            gas = noble_gases[j]
            #if j == len(noble_gases)-1 and i == len(models)-1:
            if False:
                axs[i].axvline(x = gas-1, c = 'k', linestyle = '--', linewidth = 0.4, label  = 'noble gases')
            else:
                axs[i].axvline(x = gas-1, c = 'k', linestyle = '--', linewidth = 0.4)
        axs[i].legend()
        #axs[i].set_ylim(-0.2,0.2)
    fig.suptitle('One Dimensional Proton Representation\nTest Loss w/ 64 dim = w/ 1 dim = 0.05')

    plt.show()

if __name__ == '__main__':
    val = '2e_3'
    vals = ['1', '1e_1', '2e_2', '2e_3']
    #titles = ['BasicModel_reg'+val,'BasicModelSmall_reg'+val, 'BasicModelSmaller_reg'+val, 'BasicModelReallySmall_reg'+val]
    base = 'BasicModelSmaller_reg'
    titles = []
    for val in vals:
        titles.append(base+val)
    models = []
    _, _, _, _, vocab_size = get_data() # vocab_size = (Z, N)

    min_losses = []
    for i in range(len(titles)):
        title = titles[i]
        sd = torch.load(f"models/{title}/best.pt")
        models.append(torch.load(f"models/{title}/model.pt"))
        models[i].load_state_dict(sd)
        df = pd.read_csv(f'{title}.csv')
        min_losses.append(min(df['Loss']))


    all_protons = torch.tensor(list(range(vocab_size[0])))
    all_neutrons = torch.tensor(list(range(vocab_size[1])))

    n_p = len(all_protons)
    n_n = len(all_neutrons)
    embed_dim = 64

    model_types = [BasicModel(n_p, n_n, embed_dim),
                    BasicModelSmall(n_p, n_n, embed_dim),
                    BasicModelSmaller(n_p, n_n, embed_dim),
                    BasicModelReallySmall(n_p, n_n, embed_dim)]
    total_params = []
    for model in model_types:
        total_params.append(count_parameters(model) - (n_p+n_n) * embed_dim)

    descriptions = ['2 hidden layers, layer dim = 64', 
                    '1 hidden layer, layer dim = 64',
                    '1 hidden layer, layer dim  = 16',
                    '1 hidden layer, layer dim = 4']
    for i in range(len(descriptions)):
        descriptions[i] = descriptions[i]+'\n'+f'{total_params[i]} non embedding parameters'+'\n'+f'{min_losses[i]:.6f} test loss'
    
    
    compare_onedim_pca(models, all_protons,all_neutrons, descriptions)

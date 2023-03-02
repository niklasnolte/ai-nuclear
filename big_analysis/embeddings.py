import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def get_model_embeddings(modelpath):
    # reads model from path
    sd = torch.load(modelpath, map_location=torch.device('cpu'))
    p_weights = np.array(sd['emb_proton.weight'])
    n_weights = np.array(sd['emb_neutron.weight'])
    return p_weights, n_weights

def save_model_embeddings_tsv(modelpath, savedirectory):
    # reads model from path and saves embeddings as tsv
    p_weights, n_weights = get_model_embeddings(modelpath)
    modelname = "wurst"# modelpath.split('/')[-1].split('.')[0]
    # save p_weights, n_weights as csv
    np.savetxt(savedirectory+modelname+'_p.csv', p_weights, delimiter='\t')
    np.savetxt(savedirectory+modelname+'_n.csv', n_weights, delimiter='\t')


def plot_twodim_embeddings(weights):
    # plots two dimensional PCA of weights

    plt.figure(figsize=(10,10))

    pca = PCA(n_components=2)
    embs_pca = pca.fit_transform(p.detach().cpu().numpy())
    pca_var = pca.fit(p.detach().cpu().numpy()).explained_variance_ratio_
    ap = list(range(weights.shape[0]))
    plt.xlabel(f'{100*pca_var[0]:.2f}% of variance')
    plt.ylabel(f'{100*pca_var[1]:.4f}% of variance')
    plt.scatter(*embs_pca.T, c=ap, cmap="coolwarm")
    plt.plot(*embs_pca.T,c = 'k', linewidth = 0.2)

    for i, txt in enumerate(ap):
        plt.annotate(15+txt.item(), (embs_pca[i,0], embs_pca[i,1]))
    #graph_title = "protons 2 component PCA analysis" if p is protons else "neutrons 2 component PCA analysis"
    #if title is not None:
    #    graph_title = f'{title}\n{graph_title}\ntest loss = {loss:.4f}'
    #plt.title(graph_title)
    plt.show()


if __name__ == '__main__':
    modelpath = 'model_29900.pt'
    savepath = 'embeddings/'
    p_weights, n_weights = get_model_embeddings(modelpath)
    plot_twodim_embeddings(p_weights)

    #save_model_embeddings(modelpath, savepath)

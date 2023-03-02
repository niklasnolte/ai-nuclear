import torch
from torch import nn
import matplotlib.pyplot as plt
from data import get_data
from sklearn.decomposition import PCA

def get_model():
    #model_name = 'FULL_dimnreg2_wd0.003_lr0.1_epochs30000_trainfrac0.8_hiddendim64_seed1_modelbaseline_targetsclassificationNone_targetsregressionz:1-n:1-binding_energy:1-radius:1'
    model_name = 'FULL_dimnreg0_pcaalpha-1.5_wd0.003_lr0.1_epochs30000_trainfrac0.8_hiddendim64_seed1_modelbaseline_targetsclassificationNone_targetsregressionz:1-n:1-binding_energy:1-radius:1'
    sd = torch.load(f"results/{model_name}/model_24400.pt")
    model = torch.load(f"results/{model_name}/model_full.pt")
    model.load_state_dict(sd)
    return model

def twodim_pca(title):
    #takes in model (not title) and plots the twodim pca graph
    model = get_model()
    df = get_data()
    psize = len(df['z'].unique())
    nsize = len(df['n'].unique())
    all_protons = torch.tensor(list(range(psize)))
    all_neutrons = torch.tensor(list(range(nsize)))
    protons = model.emb_proton(all_protons)
    neutrons = model.emb_neutron(all_neutrons)
    for p, ap in zip((protons, neutrons), (all_protons, all_neutrons)):

        plt.figure(figsize=(10,10))

        pca = PCA(n_components=2)
        embs_pca = pca.fit_transform(p.detach().cpu().numpy())
        pca_var = pca.fit(p.detach().cpu().numpy()).explained_variance_ratio_
        plt.xlabel(f'{100*pca_var[0]:.2f}% of variance')
        plt.ylabel(f'{100*pca_var[1]:.4f}% of variance')
        plt.scatter(*embs_pca.T, c=ap, cmap="coolwarm")
        plt.plot(*embs_pca.T,c = 'k', linewidth = 0.2)
        #annotate

        for i, txt in enumerate(ap):
            plt.annotate(15+txt.item(), (embs_pca[i,0], embs_pca[i,1]))
        graph_title = "protons 2 component PCA analysis" if p is protons else "neutrons 2 component PCA analysis"
        if title:
            graph_title = f"{title}\n{graph_title}"
        plt.title(graph_title)
    plt.show()

if __name__ == '__main__':
    title = 'dimnreg0_hiddendim64_seed1\n4 Targets: Z, N, Binding Energy, Radius'
    twodim_pca(title)
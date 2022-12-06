# importing matplot lib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
from data import get_data
from BasicModel import BasicModelSmall, BasicModelReallySmall
from tkinter import *
 
# importing movie py libraries
from base_functions import get_models, test_model
from train_model import effective_dim
 
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage





#TITLE = 'BasicModelReallySmall_reg2e_2_reg10wd'
def twodim_pca(model, all_protons, all_neutrons, title = None, heavy_elem = 0):
    protons = model.emb_proton(all_protons)
    neutrons = model.emb_neutron(all_neutrons)
    fig, axs = plt.subplots(2,1, figsize = (10,20))


    for p, ap in zip((protons, neutrons), (all_protons, all_neutrons)):
        pca = PCA(n_components=2)
        embs_pca = pca.fit_transform(p.detach().cpu().numpy())
        pca_var = pca.fit(p.detach().cpu().numpy()).explained_variance_ratio_

        if p is protons:
            ax = axs[0]
        else:
            ax = axs[1]
        
        ax.set_xlabel(f'{100*pca_var[0]:.2f}% of variance (scaled)')
        ax.set_ylabel(f'{100*pca_var[1]:.4f}% of variance (scaled)')
        label  = 'proton embeddings' if p is protons else 'neutron embeddings'
        firstdim = embs_pca.T[0]
        firstdim /= firstdim.max()
        seconddim = embs_pca.T[1]
        seconddim /= seconddim.max()
        


        ax.scatter(firstdim, seconddim, c=ap, cmap="coolwarm", label = label)
        ax.plot(firstdim, seconddim,c = 'k', linewidth = 0.2)
        bound = 1.2
        ax.set_xlim(-bound,bound)
        ax.set_ylim(-bound,bound)
        #annotate
        for i, txt in enumerate(ap):
            ax.annotate(heavy_elem+txt.item(), (embs_pca[i,0], embs_pca[i,1]))
    for ax in axs:
        ax.legend()
    if title is None:
        title = '2 component PCA'
    fig.suptitle(title)

    return mplfig_to_npimage(fig)

def make_frame(fps, graph_title, title):
    def make_real_frame(t):
        _, X_test, _, y_test, vocab_size = get_data(heavy_elem = 15)
        all_protons = torch.tensor(list(range(vocab_size[0])))
        all_neutrons = torch.tensor(list(range(vocab_size[1])))
        directory = 'pcareg_heavy15'
        factor = 10*fps
        epoch = round(int(t*factor)/10)*10
        sd = torch.load(f"models/{directory}/{title}/epoch{epoch}.pt")
        model= torch.load(f"models/{directory}/{title}/model.pt")
        model.load_state_dict(sd)
        loss = test_model(model,X_test, y_test)
        pr_e, ne_e = effective_dim(model, all_protons, all_neutrons)
        fig_title = f'{graph_title}\nepoch {epoch}\nloss = {loss:.4f}, pr_e = {pr_e:.2f}, ne_e = {ne_e:.2f}'
        return twodim_pca(model, all_protons, all_neutrons, fig_title, heavy_elem = 15)
    return make_real_frame

def create_video(graph_title, model_title):
    total_frames = 1960
    fps = 25
    duration = total_frames/fps
    
    animation = VideoClip(make_frame(fps = fps, graph_title = graph_title, title = model_title), duration = duration)
    animation.write_videofile(f"{graph_title}.mp4", fps=fps)

if __name__ == '__main__':
    _, _, _, _, vocab_size = get_data()
    titles = ['BasicModelSmall_regpca2.0_dimn', 'BasicModelSmall_regpca0_dimn'][1:]
    for title in titles:
        graph_title = f'{title}_twodimpca'
        #make_frame(5, graph_title, title)(100/5)
        create_video(graph_title, title)

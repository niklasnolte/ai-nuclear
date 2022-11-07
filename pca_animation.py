# importing matplot lib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
from data import get_data
from BasicModel import BasicModelSmall, BasicModelReallySmall
from tkinter import *
 
# importing movie py libraries
from pca_inspection import get_models, test_model, effective_dim
 
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage




TITLE = 'BasicModelSmall_reg2e_2_heavy'
TITLE = 'BasicModelReallySmall_reg2e_2_reg10wd'
def twodim_pca(model, all_protons, all_neutrons, title = None, heavy_elem = 0):
    protons = model.emb_proton(all_protons)[heavy_elem:]
    neutrons = model.emb_neutron(all_neutrons)[heavy_elem:]
    fig, axs = plt.subplots(2,1, figsize = (10,8))


    for p, ap in zip((protons, neutrons), (all_protons[all_protons>=heavy_elem], all_neutrons[all_neutrons>=heavy_elem])):
        pca = PCA(n_components=2)
        embs_pca = pca.fit_transform(p.detach().cpu().numpy())
        pca_var = pca.fit(p.detach().cpu().numpy()).explained_variance_ratio_

        if p is protons:
            ax = axs[1]
        else:
            ax = axs[0]
        
        ax.set_xlabel(f'{100*pca_var[0]:.2f}% of variance')
        ax.set_ylabel(f'{100*pca_var[1]:.4f}% of variance')
        label  = 'proton embeddings' if p is protons else 'neutron embeddings'
        ax.scatter(*embs_pca.T, c=ap, cmap="coolwarm", label = label)
        ax.plot(*embs_pca.T,c = 'k', linewidth = 0.2)
        #annotate
        for i, txt in enumerate(ap):
            ax.annotate(txt.item(), (embs_pca[i,0], embs_pca[i,1]))
    for ax in axs:
        ax.legend()
    if title is None:
        title = '2 component PCA'
    fig.suptitle(title)
    plt.show()
    return mplfig_to_npimage(fig)
    #plt.show()

def make_frame(fps):
    def make_real_frame(t):
        _, X_test, _, y_test, vocab_size = get_data()
        title = TITLE
        directory = 'Basic'
        factor = 1000 * fps
        sd = torch.load(f"models/{directory}/{title}/epoch{int(t*factor)}.pt")
        model= torch.load(f"models/{directory}/{title}/model.pt")
        model.load_state_dict(sd)
        loss = test_model(model, model.state_dict(), X_test, y_test)
        pr_e, ne_e = effective_dim(model, all_protons, all_neutrons)
        graph_title = f'epoch {int(factor*t)}\nloss = {loss:.4f}, pr_e = {pr_e:.2f}, ne_e = {ne_e:.2f}'
        return twodim_pca(model, all_protons, all_neutrons, graph_title, heavy_elem = 15)
    return make_real_frame

def create_video():
    total_frames = 29
    fps = 5
    duration = total_frames/fps
    
    animation = VideoClip(make_frame(fps = fps), duration = duration)
    animation.write_videofile(f"{TITLE}.mp4", fps=fps)

if __name__ == '__main__':
    _, _, _, _, vocab_size = get_data()

    all_protons = torch.tensor(list(range(vocab_size[0])))
    all_neutrons = torch.tensor(list(range(vocab_size[1])))
    make_frame(5)(29/5)
    #create_video()

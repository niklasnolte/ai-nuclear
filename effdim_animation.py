import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch
from BasicModel import BasicModel, BasicModelSmall
from data import get_data
 
# importing movie py libraries
from base_functions import get_models, test_model
from pca_graphs import compare_effective_dims
 
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


from pca_graphs import effective_dim_embedding
import matplotlib.colors as mcolors



def compare_effective_dims_here(graph_title, titles, models, X_test, y_test, vocab_size):
    #compares effective dim of models and creates image
    fig = plt.figure(figsize = (8,6))
    all_protons = torch.tensor(range(vocab_size[0]))
    all_neutrons = torch.tensor(range(vocab_size[1]))
    colors = list(mcolors.TABLEAU_COLORS)
    for i in range(len(models)):
        model = models[i]
        title = titles[i]
        color = colors[i]
        actual_loss, loss_nd = effective_dim_embedding(model, X_test, y_test, all_protons, all_neutrons, heavy_elem = 15)
        dims = range(1, len(loss_nd)+1)
        plt.plot(dims, loss_nd, c = color)
        plt.axhline(actual_loss, label = title, c = color, linestyle = '--')

    plt.yscale('log')
    xmin = 1
    xmax = 64
    plt.xlim(xmin, xmax)
    plt.ylim(10**-5,10**0)
    plt.legend()
    plt.title(f'{graph_title}\nLoss at given Dimension')
    plt.xlabel('Dimension of Embedding')
    plt.ylabel('Loss')
    return mplfig_to_npimage(fig)

def make_frame(fps, model_titles,parameters):
    #creates a frame of model animation
    def make_real_frame(t):
        factor = 10 * fps
        models = []
        epoch = round(int(t*factor)/10)*10
        '''
        with open('check_animation.txt', 'a') as f:
            f.write(f'{epoch}\n')
        for i in range(len(model_titles)):
            title = model_titles[i]
            sd = torch.load(f"{title}/epoch{epoch}.pt")
            models.append(torch.load(f"{title}/model.pt"))
            models[i].load_state_dict(sd)
        '''
        return compare_effective_dims(model_titles, parameters = parameters, epoch = f'epoch{epoch}.pt', plot = False)

        graph_title = f'dimall\nepoch {epoch}'
        return compare_effective_dims_here(graph_title, model_titles, models, X_test, y_test, vocab_size)
    return make_real_frame

def create_video(video_title, model_titles):
    #stitches together pictures to make video animation
    print(val)
    total_frames = 1960
    fps = 50
    duration = total_frames/fps
    parameters = get_data(heavy_elem = 15)
    animation = VideoClip(make_frame(fps, model_titles,parameters), duration = duration)
    animation.write_videofile(f"{video_title}.mp4", fps=fps)

if __name__ == '__main__':
    vals = ['dimn']
    regs = [0, 2e-4, 2e-3, 2e-2, 2e-1, 2e0]#[2, 1, 2e-1]
    titles = []
    model_title = 'BasicModelSmall'
    for val in vals:
        for i in range(len(regs)):
            reg = regs[i]
            title = f'{model_title}_regpca{reg}_{val}'
            path = f"models/pcareg_heavy15/{title}/"
            titles.append(path)
        #compare_effective_dims('bang'['models/pcareg_heavy15/BasicModelSmaller_regpca0_dimn'], X_test, y_test, vocab_size)
        #print(make_frame(10, titles, X_test, y_test, vocab_size)(5))
        create_video(f'{model_title}_{val}', titles)
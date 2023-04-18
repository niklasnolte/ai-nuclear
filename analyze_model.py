import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
from sklearn.decomposition import PCA
from OuailResModel import OuailResModel, ResidualBlock
from ResModel import ResModel
from config import Config
config = Config()


def plot_performance(filenames, names = None, functions = ['a+b', 'a-b', 'a*b']):
    if names is None:
        names = filenames
    data = {}
    for i, filename in enumerate(filenames):
        data[names[i]] = pd.read_csv(f'{filename}')

    
    markers = ['^', 'v', 'o']
    
    # make a plot for each function
    accs = [fn+'_acc' for fn in functions]

    fig, axs = plt.subplots(len(accs),1, figsize=(20, 5), sharex = True)
    plt.subplots_adjust(wspace=0, hspace=0)
    for i, acc in enumerate(accs):
        c = ['b', 'r', 'g']
        for j, key in enumerate(data):
            df = data[key]
            acc_means = []
            acc_stds = []
            for ts in df['test_size']:
                mini = df[df['test_size'] == ts]
                acc_means.append(mini[acc].mean())
                acc_stds.append(mini[acc].std())
            print(acc, key)
            color = c[j]
            df = data[key]
            axs[i].errorbar(x = df['test_size'], y = acc_means, yerr = acc_stds, color = color, fmt = 'o', label=acc+'\n'+key)
            axs[i].plot(df['test_size'], acc_means, color = color, linestyle='-')
        axs[i].legend()
    axs[-1].set_xlabel('test size')
    axs[-1].set_ylabel('accuracy')
        
    plt.show()


def evolution_uncertainty_weights(file, functions = ['a+b', 'a-b', 'a*b']):
    df = pd.read_csv(file)
    weights = [fn+'_weight' for fn in functions]
    colors = ['b', 'r', 'g']
    for i, w in enumerate(weights):
        c = colors[i]
        w_mean = []
        w_std = []
        for ts in df['test_size']:
            mini = df[df['test_size'] == ts]
            w_mean.append(mini[w].mean())
            w_std.append(mini[w].std())
        plt.errorbar(x = df['test_size'], y = w_mean, yerr = w_std, color = c, fmt = 'o', label=w)
        plt.plot(df['test_size'], w_mean, color = c, linestyle='-')
    unc_loss = r'$L = \sum wl - \log(\prod w)$'
    plt.title('Evolution of Weights for Uncertainty Loss\n'+unc_loss)
    plt.xlabel('test size')
    plt.ylabel('weight value')
    plt.legend()
    plt.show()

    plt.figure()
    df['a*b/a+b_weight'] = df['a*b_weight']/df['a+b_weight']
    plt.scatter(df['test_size'], df['a*b/a+b_weight'], color = 'b')
    plt.xlabel('test size')
    plt.ylabel('a*b/a+b weight')
    plt.title('Evolution of a*b/a+b Weight\n'+unc_loss)
    plt.show()

def plot_model_performances():

    fn_list = list(config.all_fn_dict.keys())
    colors = 'bgrcmyk'
    for i, fn in enumerate(fn_list):
        print('here')
        plt.figure()
        dfs = {}
        dfs['Task Emb/All Fn'] = pd.read_csv(f'csv/OuailResModel_fn01234_ts0.05_wd0.01_lr0.0001.csv')
        #dfs['Task Emb/Only One'] = pd.read_csv(f'csv/OuailResModel_fn{i}_ts0.05_wd0.01_lr0.0001.csv')
        dfs['Task Emb/Two Fns'] = pd.read_csv(f'csv/OuailResModel_fn{i}{(i+1)%5}_ts0.05_wd0.01_lr0.0001.csv')
        dfs['No Task Emb/All Fn'] = pd.read_csv(f'csv/ResModel_fn01234_ts0.05_wd0.01_lr0.0001.csv')
        dfs['No Task Emb/One Fn'] = pd.read_csv(f'csv/ResModel_fn{i}_ts0.05_wd0.01_lr0.0001.csv')
        
        linestyles = ['dotted', 'solid', 'dashed', 'dashdot']
        print('one')
        j = 0
        for key in dfs:
            df = dfs[key]
            print('two')
            
            epochs = np.array(df['iterations'])
            test_loss = np.array(df[fn+'_test'])
            #test_loss = np.resize(test_loss, 100)
            #epochs = np.resize(epochs, 100)
            if True:
                plt.plot(epochs, test_loss , color = colors[i], linestyle = linestyles[j], label = fn+'\n'+key)
            elif j == 0:
                plt.plot(epochs, test_loss , color = colors[i], linestyle = linestyles[j], label = fn)
            else:
                plt.plot(epochs, test_loss , color = colors[i], linestyle = linestyles[j])
            j+=1
        print('three')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss')
        plt.title(f'{fn}\nTest Loss for Different Models')
        plt.yscale('log')
        print('four')
        plt.show()

        

def plot_embedding_pcas():
    fns = ['01234']
    fn_list = list(config.all_fn_dict.keys())
    model_type = ['ResModel', 'OuailResModel']
    model_paths = []
    state_dict_paths = []
    titles = []
    for fn in fns:
        for mt in model_type:
            name = f'{mt}_fn{fn}_ts0.05_wd0.01_lr0.0001'
            model_path = f'models/{mt}/{name}/model.pt'
            df = pd.read_csv(f'csv/{name}.csv')
            if mt == 'ResModel':
                title = f'PCA of {fn}, No Task Emb \n'
            else:
                title = f'PCA of {fn}, Task Emb \n'
            for val in fn:
                val = int(val)
                test = '_test'
                title+=f'{fn_list[val]}= {df[fn_list[val]+test].iloc[-1]:.2e}, '

            state_dict_path = f'models/{mt}/{mt}_fn{fn}_ts0.05_wd0.01_lr0.0001/best.pt'
            model_paths.append(model_path)
            state_dict_paths.append(state_dict_path)
            titles.append(title)
    def plot_embedding_pca(model_file, state_dict_file, name = None):
        # Load the model and extract the embedding matrix
        model = torch.load(model_file)
        state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        embedding_matrix = model.emb_a.weight.detach().numpy()
        embedding_matrix = embedding_matrix[:config.LIMIT]

        # Use PCA to reduce the embedding matrix to 2 dimensions
        pca = PCA(n_components=2)
        pca.fit(embedding_matrix)
        embedding_pca = pca.transform(embedding_matrix)

        # Plot the scatter plot with annotations
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.scatter(embedding_pca[:, 0], embedding_pca[:, 1], s=10)

        # Add annotations for each point
        indices = np.arange(embedding_matrix.shape[0])
        for i, index in enumerate(indices):
            ax.annotate(str(index), (embedding_pca[i, 0], embedding_pca[i, 1]))

        # Add a heatmap to the annotations
        norm = Normalize(vmin=0, vmax=len(indices))
        colors = plt.cm.jet(norm(indices))
        for i, color in enumerate(colors):
            ax.annotate(str(indices[i]), (embedding_pca[i, 0], embedding_pca[i, 1]), color=color)
        if name is not None:
            plt.title(name)
        plt.show()

    for i in range(len(model_paths)):
        print(model_paths[i])
        plot_embedding_pca(model_paths[i], state_dict_paths[i], titles[i])





if __name__ == '__main__':
    model_path = 'models/ResModel/ResModel_fn01234_ts0.05_wd0.01_lr0.0001/model.pt'
    state_dict_path = 'models/ResModel/ResModel_fn01234_ts0.05_wd0.01_lr0.0001/best.pt'
    #plot_embedding_pca(model_path, state_dict_path)
    plot_embedding_pcas()
    #plot_model_performances()
    # filenames = ['full_results/BasicModel/BasicModel.csv', 
    #               'full_results/UncertaintyLossModel/UncertaintyLossModel.csv']
    # unc_loss = r'$L = \sum wl - \log(\prod w)$'
    # names = ['Normal Loss Function', f'Learnable Weights ({unc_loss})']
    # plot_performance(filenames, names)
    #evolution_uncertainty_weights(filenames[-1])
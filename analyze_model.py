import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
from sklearn.decomposition import PCA
from TaskEmbModel import TaskEmbModel, ResidualBlock
from BaselineModel import extract_params_from_title, BaselineModel
from ResModel import ResModel
import re
from config import Config
import os
from collections import defaultdict
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FixedLocator, FixedFormatter


config = Config()

#plt.rcParams.update({'font.size': 11})
plt.style.use('mystyle-bright.mplstyle')

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
    fns = ['01', '012', '0123', '01234']
    fn_list = list(config.all_fn_dict.keys())
    model_type = ['ResModel', 'OuailResModel']
    model_paths = []
    state_dict_paths = []
    titles = []
    for fn in fns:
        for mt in model_type:
            prior = '_ts0.95_wd0.001_lr0.0001_batch4'
            name = f'{mt}_fn{fn}{prior}'
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

            state_dict_path = f'models/{mt}/{name}/best.pt'
            model_paths.append(model_path)
            state_dict_paths.append(state_dict_path)
            titles.append(title)
    def plot_embedding_pca(model_file, state_dict_file, name = None):
        # Load the model and extract the embedding matrix
        model = torch.load(model_file)
        state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        embedding_matrix = model.emb_a.weight.detach().numpy()
        #embedding_matrix = embedding_matrix[:config.LIMIT]

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



def plot_loss(title):
    df = pd.read_csv('csv/'+title+'.csv')
    epochs = np.array(df['iterations'])
    fn_dict = config.all_fn_dict
    fns = list(fn_dict.keys())
    colors = 'bgrcmykw'
    for i, fn in enumerate(fns):
        if fn+'_test' not in df.columns:
            continue
        test_loss = np.array(df[fn+'_test'])
        train_loss = np.array(df[fn+'_train'])
        plt.plot(epochs, test_loss, label = fn+'_test', color = colors[i])
        plt.plot(epochs, train_loss, label = fn+'_train', color = colors[i], linestyle = 'dashed')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(title)
    plt.show()

def compare_losses():
    # compare task emb model to baseline model
    task_emb = config.best_taskemb_multi
    baseline = config.best_baseline_multi
    df_base = pd.read_csv('csv/'+baseline+'.csv')
    df_task = pd.read_csv('csv/'+task_emb+'.csv')
    fn_dict = config.all_fn_dict
    fns = list(fn_dict.keys())
    colors = 'bgrcmykw'
    best_individuals = config.best_baseline_single
    fig, axs= plt.subplots(1, len(fns), figsize = (20,4), sharey = True)
    colors = 'bgrcmykw'
    for i, fn in enumerate(fns):
        if fn+'_test' not in df_base.columns:
            continue

        best_ind = pd.read_csv('csv/'+best_individuals[i]+'.csv')
        ind_test = np.array(best_ind[fn+'_test'])
        ind_train = np.array(best_ind[fn+'_train'])
        ind_epoch = np.array(best_ind['iterations'])
        axs[i].plot(ind_epoch, ind_test, color = colors[i],linestyle = '--', label = 'Base Single')
        
        base_test = np.array(df_base[fn+'_test'])
        base_train = np.array(df_base[fn+'_train'])
        base_epoch = np.array(df_base['iterations'])
        axs[i].plot(base_epoch, base_test, label = 'Base Multi', linestyle = ':', color = colors[i])

        axs[i].set_title(fn)
        task_test = np.array(df_task[fn+'_test'])
        task_epoch = np.array(df_task['iterations'])
        task_train = np.array(df_task[fn+'_train'])
        axs[i].plot(task_epoch, task_test, color = colors[i], linestyle = '-',label = 'Task Emb Multi')
        axs[i].legend()
        axs[i].set_yscale('log')
        if i == 0:
            axs[i].set_ylabel('MSE Validation Loss')
        elif i == 2:
            axs[i].set_xlabel('Epoch')

    plt.subplots_adjust(left=0.04, bottom=0.15, right=0.95, top=0.92, wspace=0, hspace=0.2)
    plt.savefig('toy_model.pdf', dpi = 300)
    plt.show()



def find_number_after_substring(s, substring):
    pattern = re.compile(substring + r"([0-9]+(?:\.[0-9]+)?(?:[eE][-+]?\d+)?)")
    match = pattern.search(s)
    
    if match:
        number = match.group(1)
        if "." in number or "e" in number.lower():
            return float(number)
        else:
            return int(number)
    else:
        raise ValueError(f"No number found immediately following '{substring}'")



def best_model_testsize(title, metric = 'test_loss'):
    df = pd.read_csv(f'csv/{title}.csv')
    # Assuming "title" is the column containing the titles
    
    # Dictionary to hold the best performing models for each test size
    best_models = defaultdict(lambda: (None, np.inf))

    for index, row in df.iterrows():
        try:
            test_size = find_number_after_substring(row['title'], 'ts')
            if row['test_loss'] < best_models[test_size][1]:
                best_models[test_size] = (row['title'], row[metric])
        except ValueError:
            print(f"Cannot extract test size from title '{row['title']}'")

    for test_size, (title, test_loss) in best_models.items():
        print(f"Best model for test size {test_size} is {title} with test loss {test_loss}")
    return best_models

def compare_ts_models():
    taskemb = 'TaskEmbModel_01234_20lim_ts_random_search'
    metric = '$(a+b)^{2/3}$_test'
    task_mult = best_model_testsize(taskemb, metric = metric)
    baseline = 'BaselineModel_01234_20lim_ts_random_search'
    base_mult = best_model_testsize(baseline, metric = metric)
    base2 = 'BaselineModel_2_20lim_ts_random_search'
    base2_mult = best_model_testsize(base2, metric = metric)
    for ts in task_mult:
        print('train size', ts)
        print(task_mult[ts][1], task_mult[ts][0])
        print(base_mult[ts][1], base_mult[ts][0])
        print(base2_mult[ts][1], base2_mult[ts][0])

        #print(ts, task_mult[ts][1], base2_mult[ts][1])

def grab_best_models():
    best = {}
    for fn in range(5):
        df = pd.read_csv(f'csv/BaselineModel_01234_20lim_ts_random_search.csv')
        best[fn] = (df['title'][0], df['test_loss'][0])
        for index, row in df.iterrows():
            if find_number_after_substring(df['title'][index], 'ts') == 0.3:
                test_loss = df['test_loss'][index]
                if test_loss < best[fn][1]:
                    best[fn] = (df['title'][index], test_loss)



def plot_best():
    task = pd.read_csv(f'csv/{config.best_taskemb_multi}.csv')
    base_mult = pd.read_csv(f'csv/{config.best_baseline_multi}.csv')
    all_single = config.best_baseline_single
    fn_dict = config.all_fn_dict
    fn_keys = list(fn_dict.keys())
    fn_aesthetic = config.all_fn_dict_aesthetic
    fn_aes_keys = list(fn_aesthetic.keys())
    colors = 'bgrcmykw'
    linestyle_dict = {"single": ":", "multi": "--", "task": "-"}
    
    plt.figure(figsize = (8,8))
    for fn in range(len(fn_keys)):
        base_single = pd.read_csv(f'csv/{all_single[fn]}.csv')
        
        plt.plot(base_single['iterations'], base_single[fn_keys[fn]+'_test'], color = colors[fn], linestyle = linestyle_dict["single"])
        plt.plot(base_mult['iterations'], base_mult[fn_keys[fn]+'_test'], color = colors[fn], linestyle = linestyle_dict["multi"])
        plt.plot(task['iterations'], task[fn_keys[fn]+'_test'], color = colors[fn], linestyle = linestyle_dict["task"])

    # Creating custom legend entries
    line_labels = ["ST", "MT", "MTE"]
    line_styles = [mlines.Line2D([], [], color='black', linestyle=linestyle_dict[i]) for i in ["single", "multi", "task"]]
    color_patches = [mlines.Line2D([], [], color=colors[i], linestyle='-') for i in range(len(fn_keys))]

    # Adding legend for line styles
    leg1 = plt.legend(handles=line_styles, labels=line_labels, loc="upper left", title="Models")

    # Adding legend for colors (i.e., functions)
    leg2 = plt.legend(handles=color_patches, labels=fn_aes_keys, loc="upper right", title="Functions")

    # Manually add the first legend back
    plt.gca().add_artist(leg1)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss with 11%-89% Train-Test Split')
    #plt.title('Test Loss with 30%-70% Train-Test Split')
    plt.yscale('log')
    plt.savefig('plots/toy_model_other.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    

def plot_all_performances(file, look_at = 'lr'):
    all_df = pd.read_csv('csv/'+file+'.csv')
    titles = all_df['title']
    colors = 'bgrcmykw'
    values = {}
    determ = 64
    lowest_loss = float('inf')
    res = {'Title':[], 'Test Loss':[]}
    for title in titles:
        df = pd.read_csv('csv/'+title+'.csv')
        value = find_number_after_substring(title, look_at)
        num_epochs_run = find_number_after_substring(title, 'epochs')
        if num_epochs_run != 5000:
            continue

        
        epochs = np.array(df['iterations'])
        test_loss = np.array(df['test_loss'])
        if test_loss[-1]>1e-5:
            continue
        if test_loss[-1]<lowest_loss:
            lowest_loss = test_loss[-1]
            print(title)
            print(lowest_loss)

        if value not in values:
            values[value] = (colors[0],1)
            colors = colors[1:]
            plt.plot(epochs, test_loss, label = value, color = values[value][0])
        else:
            values[value] = (values[value][0],values[value][1]+1)
            plt.plot(epochs, test_loss, color = values[value][0])
        res['Title'].append(title)
        res['Test Loss'].append(test_loss[-1])
    res = pd.DataFrame(res)
    res = res.sort_values(by = 'Test Loss')
    pd.set_option('display.max_colwidth', None)  # None implies no limit; alternatively, set a large value like 1000
    pd.set_option('display.float_format', '{:e}'.format)
    print(res.head(25))
    plt.yscale('log')
    plt.title(file+'\n'+look_at)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.legend()
    plt.show()
    print(values)

def plot_embedding_matrix(title, dir=None, plot_title = 'MTE', save_title = None):
    # Load the model
    fn_list = title.split('_')[1][2:]
    model_name = title.split('_')[0]
    if dir is None:
        dir = f'models/{model_name}_{fn_list}_20lim_ts_random_search'
    try:
        model = torch.load(dir+'/'+title+'/model.pt')
    except:
        fn_input = [fn for fn in fn_list]
        hidden_size = find_number_after_substring(title, 'hd')
        num_layers = find_number_after_substring(title, 'nl')
        if model_name == 'TaskEmbModel':
            model = TaskEmbModel(fn_input, hidden_size, num_layers)
        elif model_name == 'BaselineModel':
            model = BaselineModel(fn_input, hidden_size, num_layers)
            print('baselinemodel')
    model.load_state_dict(torch.load(dir+'/'+title+'/best.pt'))
    
    # Get the embedding layer weights
    embedding_weights = model.emb_a.weight.detach().cpu().numpy()
    # if embedding_weights.shape[0] != len(fn_list) + config.LIMIT:
    #     print('Wrong config limit for model')
    #     return

    # Perform PCA
    pca = PCA(n_components=2)
    embedding_pca = pca.fit_transform(embedding_weights)

    # Plot the embeddings after PCA
    fig, ax = plt.subplots(figsize=(8,8))
    scatter = ax.scatter(embedding_pca[:, 0], embedding_pca[:, 1], c=range(len(embedding_weights)), cmap='plasma') 
    cmap = plt.get_cmap('plasma')
    norm = plt.Normalize(0, len(embedding_weights))
    # Add labels
    df = pd.read_csv('csv/'+title+'.csv')
    fn_dict = config.all_fn_dict
    all_fns = list(fn_dict.keys())
    fn_aes_dict = config.all_fn_dict_aesthetic
    all_fn_aes = list(fn_aes_dict.keys())
    delta = 0.01
    font = 10
    minx, miny = float('inf'), float('inf')
    maxx, maxy = -float('inf'), -float('inf')

    function_labels = {}
    for i in range(len(embedding_weights)):
        x = embedding_pca[i, 0]
        y = embedding_pca[i, 1]
        if i >= config.LIMIT:
            fn_num = i - config.LIMIT
            fn_str = all_fn_aes[fn_num]
            ax.text(x*(1+delta), y, fn_str, fontsize=font) 
            minx = min(minx, x)
            miny = min(miny, y)
            maxx = max(maxx, x)
            maxy = max(maxy, y)
            function_labels[fn_str] = (x, y, cmap(norm(i)))
            
        else:
            ax.text(x*(1+delta), y, str(i), fontsize=font) 

    ax.set_xlabel(f'PC1 ({str(round(pca.explained_variance_ratio_[0] * 100, 2))}% of variance)') 
    ax.set_ylabel(f'PC2 ({str(round(pca.explained_variance_ratio_[1] * 100, 2))}% of variance)') 
    
    ax.set_title(f'Embeddings PCA for {plot_title}')

    # Inset plot
    if minx != float('inf'):
        ax_inset = inset_axes(ax, width="30%", height="30%", 
                        borderpad = 2,
                        bbox_transform=ax.transAxes, 
                        loc='upper center')

        
        #xlim = (-0.005, 0.005)
        #ylim = (0, 0.05)
        epsilon = 0.1
        deltax = maxx - minx
        deltay = maxy - miny
        xlim = (minx - epsilon*deltax, maxx + epsilon*deltax)
        ylim = (miny - epsilon*deltay, maxy + epsilon*deltay)
        print('xrange', (minx, maxx))
        print('xlim', xlim)
        print('yrange', (miny, maxy))
        print('ylim', ylim)

        ax_inset.set_xlim(*xlim)
        ax_inset.set_ylim(*ylim)
        ax_inset.set_title("Task Embeddings")

        # Add labels to the inset plot
        # plot the points in function label in inset
        points = []
        colors = []
        for i,fn_str in enumerate(function_labels):
            x, y, color = function_labels[fn_str]
            points.append((x, y))
            colors.append(color)
            if i == 2:
                ax_inset.text(x-deltax*.35, y, fn_str, fontsize=font)
            elif i == 3:
                ax_inset.text(x-deltax*.55, y, fn_str, fontsize=font)
            elif i == 4:
                ax_inset.text(x-deltax*.4, y, fn_str, fontsize=font)
            else:
                ax_inset.text(x+deltax*.02, y, fn_str, fontsize=font)
        ax_inset.scatter([p[0] for p in points], [p[1] for p in points], c=colors)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        locs = [minx, maxx]
        ax_inset.xaxis.set_major_locator(FixedLocator(locs))
        ax_inset.xaxis.set_major_formatter(FixedFormatter([f'{x:.1e}' for x in locs]))

        locs = [miny, maxy]
        ax_inset.yaxis.set_major_locator(FixedLocator(locs))
        ax_inset.yaxis.set_major_formatter(FixedFormatter([f'{x:.1e}' for x in locs]))
        # Rotate xticks
        # for label in ax_inset.get_xticklabels():
        #     label.set_rotation(45)
    if save_title is None:
        plt.savefig(f'plots/{plot_title}_emb.pdf', dpi = 300)
    else:
        plt.savefig(f'plots/{save_title}_emb.pdf', dpi = 300)
    plt.show()


if __name__ == '__main__':


    # # ST Embeddings
    fn_names = list(config.all_fn_dict_aesthetic.keys())
    # for fn in [0, 1, 2, 3, 4]:
    #     fn = 2
    #     st = config.best_baseline_single[fn]
    #     plot_embedding_matrix(st, plot_title=f'ST {fn_names[fn]}', save_title=f'ST_{fn}')
    #     break

    # MTE Embeddings
    mte = config.best_taskemb_multi
    plot_embedding_matrix(mte, plot_title=f'MTE')

    # # MT Embeddings
    # mt = config.best_baseline_multi
    # plot_embedding_matrix(mt, plot_title=f'MT')

    #plot_best()
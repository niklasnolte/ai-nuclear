
from base_functions import get_models, plot_epochs
from BasicModel import BasicModelSmaller
from pca_graphs import plot_embeddings_loss
from data import get_data
if __name__ == '__main__':
    title = 'models/pcareg_heavy15/BasicModelSmaller_regpca0'#.02_dimn'
    model = get_models([title])[0]
    X_train, X_test, y_train, y_test, vocab_size = get_data(heavy_elem = 15)
    plot_embeddings_loss(model, X_test, y_test, title, dim = 64)

    #print(title)
    #plot_epochs([title])
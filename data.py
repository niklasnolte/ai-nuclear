from config import Config
import torch
from torch import nn
from sklearn.model_selection import train_test_split

config = Config()
def get_data(functions, test_size = 0.2, seed = 42):
    '''
    Generates the modular arithmetic data for training and testing.
    Functions is a list of functions to be evaluated.
    ex. functions = ['a+b', 'a-b', 'a*b']
    y is the output of functions on the dataset with a,b ranging from 0 to config.LIMIT
    '''
    Xs = []
    ys = []
    limit = config.LIMIT
    for a in range(limit):
        for b in range(limit):
            x = [a,b]
            y = []
            for function in functions:
                y.append(eval(function)%limit)
            Xs.append(x)
            ys.append(y)
    Xs, ys = torch.tensor(Xs).int(), torch.tensor(ys).long()
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=test_size, random_state=seed)
    vocab_size = (limit, limit) 
    return X_train, X_test, y_train, y_test, vocab_size


def get_data_nomod(functions, test_size = 0.2, seed = 1):
    Xs = []
    ys = []
    limit = config.LIMIT
    for a in range(2,limit):
        for b in range(1,limit):
            x = [a,b]
            y = []
            a, b = torch.tensor(a), torch.tensor(b)
            for function in functions:
                y.append(function(a,b))
            Xs.append(x)
            ys.append(y)
    Xs, ys = torch.tensor(Xs).int(), torch.tensor(ys).float()
    ys = (ys - ys.min())/(ys.max()-ys.min())
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=test_size, random_state=seed)
    vocab_size = (limit, limit)
    return X_train, X_test, y_train, y_test, vocab_size

def get_data_nomod_difftasks(functions, test_size = 0.2, seed = 1):
    Xs = None
    ys = None
    limit = config.LIMIT
    for fn_num, function in enumerate(functions):
        mini_Xs = []
        mini_ys = []
        for a in range(limit):
            for b in range(limit):
                x = [a,b, fn_num]
                y = []
                a, b = torch.tensor(a), torch.tensor(b)
                y.append(function(a,b))
                mini_Xs.append(x)
                mini_ys.append(y)
        mini_Xs, mini_ys = torch.tensor(mini_Xs).int(), torch.tensor(mini_ys).float()
        mini_ys = (mini_ys - mini_ys.min())/(mini_ys.max()-mini_ys.min())
        if Xs is None:
            Xs = mini_Xs
            ys = mini_ys
        else:
            Xs = torch.cat((Xs, mini_Xs), dim=0)
            ys = torch.cat((ys, mini_ys), dim=0)
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=test_size, random_state=seed)
    vocab_size = (limit, limit)
    return X_train, X_test, y_train, y_test, vocab_size


if __name__ == '__main__':
    all_fn_dict = {
        r'$a+b$': lambda x, y: x + y,
        r'$|a-b|$': lambda x, y: abs(x - y),
        r'$a*b$': lambda x, y: x * y
    }
    functions = list(all_fn_dict.values())
    X_train, X_test, y_train, y_test, vocab_size = get_data_nomod_difftasks(functions)
    #print(X_train.shape, y_train.shape)
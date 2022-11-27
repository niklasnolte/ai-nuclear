import numpy as np
import torch
import matplotlib.pyplot as plt

def get_index(hidden_dim = 64, alpha = -1.05, plot_dist = True):
    indices = torch.tensor(range(1,hidden_dim+1))
    base = torch.tensor([ind**alpha for ind in range(1,hidden_dim +1)])
    base = base/base.sum()
    sample = torch.multinomial(base, 1, replacement=True)
    
    if plot_dist:
        samples = torch.multinomial(base, 10000, replacement=True)
        cdf = torch.tensor([base[:n].sum() for n in range(hidden_dim+1)])
        midpoint = (((cdf>0.5) == True).nonzero(as_tuple=True)[0][0])
        base = base/base.sum()
        
        plt.axvline(x = midpoint, linestyle = '--', c= 'r', label = f'center of mass = {midpoint}')
        plt.legend()
        plt.hist(samples, bins = 64)
        plt.show()

    return indices[sample]

if __name__ == '__main__':
    print(get_index(alpha = -1.1, plot_dist=False))



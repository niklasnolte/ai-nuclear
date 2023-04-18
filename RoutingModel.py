import torch
from torch import nn
import matplotlib.pyplot as plt
from train_routing import train
from config import Config
from TaskRouting import TaskRouter
import numpy as np
import pandas as pd
from utils import functions_to_names

config = Config()
LIMIT = config.LIMIT

class RoutingModel(nn.Module):
    def __init__(self, functions, hidden_dim, sigma = 0.5, seed = 1, device = 'cuda:0'):
        super().__init__()
        self.emb_a = nn.Embedding(LIMIT, hidden_dim)
        self.task_count = len(functions)
        self.active_task = 0
        self.interim_size = 2*hidden_dim # 2 comes from a,b

        self.nonlinear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.interim_size, self.interim_size),
            TaskRouter(self.interim_size, self.task_count, int(self.interim_size*sigma), seed=seed, device=device),
            nn.ReLU())
        
        for ix in range(self.task_count):
            self.add_module("classifier_" + str(ix), nn.Sequential(
                nn.Linear(self.interim_size, config.LIMIT)
            ))
        self.emb_a.weight.data.uniform_(-1,1)
        
    def forward(self, x): # x: [ batch_size, 2 [n_a, n_b] ]
        a = self.emb_a(x[:,0]) # [ batch_size, hidden_dim ]
        b = self.emb_a(x[:,1]) # [ batch_size, hidden_dim ]
        x = self.nonlinear(torch.hstack((a, b)))
        output = self.get_layer("classifier_" + str(self.active_task)).forward(x)


        return output

    def set_active_task(self, active_task):
        self.active_task = active_task
        return active_task
    
    def get_layer(self, name):
        return getattr(self, name)


def run_functions(functions):
    model = RoutingModel(functions, 64)
    name = functions_to_names(functions)
    dir = 'mod_arith'
    csv_dict = None
    for sigma in [0]:
        for test_size in np.linspace(0.05, 0.95, 19):
            for seed in range(1):
                ts = round(test_size, 2)
                print(ts)
                title = f'RoutingModel_{name}_ts{ts}_sigma{sigma}_seed{seed}'
                res = train(modelclass = RoutingModel, 
                    functions = functions, 
                    lr = 1e-3, 
                    wd = 1e-4, 
                    embed_dim = 64, 
                    title = title,
                    basepath = f'models/{dir}/{title}/',
                    device = 'cuda:0',
                    seed = seed,
                    test_size = test_size,
                    sigma = sigma
                )
                other_metrics = ['sigma', 'seed', 'test_size']
                if csv_dict is None:
                    csv_dict = {key: [value] for key, value in res.items()}
                    for metric in other_metrics:
                        csv_dict[metric] = [eval(metric)]
                else:
                    for key, value in res.items():
                        csv_dict[key].append(value)
                    for metric in other_metrics:
                        csv_dict[metric].append(eval(metric))
                csv_df = pd.DataFrame(csv_dict)
                csv_df.to_csv(f'modarith_{name}.csv')

if __name__ == '__main__':
    functions = ['a+b', 'a-b', 'a*b']
    #functions = ['a+b', 'a-b', 'a*b']
    #run_functions(functions)
    model = RoutingModel(functions, 64)
    print(model)

    
                
import torch
from torch import nn
import matplotlib.pyplot as plt
from train_model import train
from config import Config
from TaskRouting import TaskRouter

config = Config()
LIMIT = config.LIMIT

class BasicModelSmall(nn.Module):
    def __init__(self, functions, hidden_dim, sigma = 0.5):
        super().__init__()
        self.emb_a = nn.Embedding(LIMIT, hidden_dim)
        self.task_count = len(functions)
        self.active_task = 0
        self.nonlinear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.task_count*hidden_dim, self.task_count*hidden_dim),
            TaskRouter(hidden_dim, self.task_count, int(hidden_dim*sigma)),
            nn.ReLU())
        for ix in range(self.task_count):
            self.add_module("classifier_" + str(ix), nn.Sequential(
                nn.Linear(self.task_count*hidden_dim, hidden_dim)
            ))
        self.emb_a.weight.data.uniform_(-1,1)
        
    def forward(self, x): # x: [ batch_size, 2 [n_a, n_b] ]
        a = self.emb_a(x[:,0]) # [ batch_size, hidden_dim ]
        b = self.emb_a(x[:,1]) # [ batch_size, hidden_dim ]
        x = self.nonlinear(torch.hstack((a, b)))

        return x

    def set_active_task(self, active_task):
        self.active_task = active_task
        return active_task
    
    def get_layer(self, name):
        return getattr(self, name)
    
if __name__ == '__main__':
    functions = ['a+b', 'a-b', 'a*b']
    model = BasicModelSmall(functions, 64)
    print(model)
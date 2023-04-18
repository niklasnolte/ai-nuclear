import os
import pickle
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import urllib.request
from config import Config
from data import get_data

config = Config()


def embedding_to_value(y):
    #given a y vector, makes onehot stacked encodings
    one = nn.functional.one_hot(y[:,0], num_classes=- 1)

def train(modelclass, functions, lr, wd, embed_dim, basepath, device, title,  sigma = 0.5, seed = 1, test_size = 0.2):
  #train just for modular arithmetic
  torch.manual_seed(seed)
  os.makedirs(basepath, exist_ok=True)
  X_train, X_test, y_train, y_test, vocab_size = get_data(functions, test_size = test_size, seed = seed) 

  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)

  all_a = torch.tensor(list(range(vocab_size[0]))).to(device)

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=len(X_train), shuffle=True)
  model = modelclass(functions, embed_dim, sigma = sigma, seed = 1, device = device).to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

  bar = tqdm.tqdm(range(int(2e4)))
  lowest_loss = 1e10

  data_dict = None

  torch.autograd.set_detect_anomaly(True)

  for i in bar:
    
    for X_batch, y_batch in train_loader:
        for task in range(len(functions)):       
            optimizer.zero_grad()
            model.set_active_task(task)
            y_batch_pred = model(X_batch).to(device)
            y_batch_actual = nn.functional.one_hot(y_batch[:,task], num_classes=- 1).float().to(device)
            train_loss = loss_fn(y_batch_pred, y_batch_actual)    

            train_loss.backward()
            optimizer.step()

    with torch.no_grad():
        loss = 0
        train_loss = 0
        metrics = {}
        for task in range(len(functions)):
            model.set_active_task(task)

            y_test_pred = model(X_test).to(device)
            y_test_actual = nn.functional.one_hot(y_test[:,task], num_classes=- 1).float().to(device)
            loss += loss_fn(y_test_pred, y_test_actual)

            y_train_pred = model(X_train).to(device)
            y_train_actual = nn.functional.one_hot(y_train[:,task], num_classes=- 1).float().to(device)
            train_loss += loss_fn(y_train_pred, y_train_actual)

            metrics[functions[task]+'_acc'] = (y_test[:,task] == y_test_pred.argmax(dim=1)).float().mean().cpu().numpy()
            #metrics[functions[task]+'_trainacc'] = (y_train[:,task] == y_train_actual.argmax(dim=1)).float().mean().cpu().numpy()
        metrics['test_loss'] = loss.cpu().numpy()
        metrics['train_loss'] = train_loss.cpu().numpy()
        if loss < lowest_loss:
            lowest_loss = loss
            best_state_dict = model.state_dict()

        if i % 500 == 0:
            torch.save(model.state_dict(), basepath + f"epoch{i}.pt")

        if i % 100 == 0:
            entropy = effective_dim(model, all_a)
            entropy = entropy.cpu().numpy()
            metrics['entropy'] = entropy

            if data_dict is None:
                data_dict = {m:[metrics[m]] for m in metrics.keys()}
                data_dict['iterations'] = [i]
            else:
                for m in metrics:
                    data_dict[m].append(metrics[m])
                data_dict['iterations'].append(i)
            df = pd.DataFrame(data_dict)
            df.to_csv('csv/{0}.csv'.format(title))
        bar.set_postfix(**metrics)
        for fn in functions:
            if metrics[fn+'_acc'] < 1-config.training_epsilon:
                break
        else:
            return metrics
    torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return metrics

def effective_dim(model, all_a):
  #calculate entropy of the embeddings
    a = model.emb_a(all_a)
    a_S = (torch.square(torch.svd(a)[1]) / (all_a.shape[0] - 1))
    a_prob = a_S/a_S.sum()
    entropy_a = -(a_prob * torch.log(a_prob)).sum()
    entropy = torch.exp(entropy_a)
    return entropy

if __name__ == '__main__':
    functions = ['a+b', 'a-b', 'a*b', 'a**b']
    X_train, X_test, y_train, y_test, vocab_size = get_data(functions)
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(vocab_size)
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


def get_data(functions, test_size = 0.2, seed = 42):
    '''
    Generates the modular arithmetic data for training and testing.
    Functions is a list of functions to be evaluated.
    ex. functions = ['a+b', 'a-b', 'a*b']
    y is the output of functions on the dataset with a,b ranging from 0 to config.LIMIT
    '''
    Xs = []
    ys = []
    config = Config()
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

def train(modelclass, lr, wd, embed_dim, basepath, device, title, seed = 1, test_size = 0.2):
  #train just for modular arithmetic
  torch.manual_seed(seed)
  os.makedirs(basepath, exist_ok=True)
  X_train, X_test, y_train, y_test, vocab_size = get_data(test_size = test_size, seed = seed) 

  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)

  all_a = torch.tensor(list(range(vocab_size[0]))).to(device)

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=len(X_train), shuffle=True)
  model = modelclass(*vocab_size, embed_dim).to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

  bar = tqdm.tqdm(range(int(3e0)))
  lowest_loss = 1e10

  loss_list = []
  iterations = []
  train_loss_list = []

  torch.autograd.set_detect_anomaly(True)

  for i in bar:
    
    for X_batch, y_batch in train_loader:

      optimizer.zero_grad()
      y_pred = model(X_batch).to(device)
      y_batch_comb = onehot_stacky(y_batch)
      loss = loss_fn(y_pred, y_batch_comb)
      train_apb_acc, train_amb_acc = calc_acc(y_pred, y_batch)

      loss.backward()
      optimizer.step()

    with torch.no_grad():

      y_pred = model.forward(X_test)
      y_test_comb = onehot_stacky(y_test)
      loss = loss_fn(y_pred, y_test_comb)
      y_train_comb = onehot_stacky(y_train)
      train_loss = loss_fn(model.forward(X_train), y_train_comb)

      if loss < lowest_loss:
        lowest_loss = loss
        best_state_dict = model.state_dict()
      if i % 500 == 0:
        torch.save(model.state_dict(), basepath + f"epoch{i}.pt")

      if i % 100 == 0:
        iterations.append(i)
        loss_list.append(loss.cpu().numpy())
        train_loss_list.append(train_loss.cpu().numpy())
        data_dict = {'Iteration':iterations, 'Loss':loss_list,'Train_Loss':train_loss_list}
        df = pd.DataFrame(data_dict)
        df.to_csv('csv/{0}.csv'.format(title))

      torch.save(model.state_dict(), basepath + f"latest.pt")

      test_apb_acc, test_amb_acc = calc_acc(y_pred, y_test)
      bar.set_postfix(loss=loss.item(), test_apb_acc = test_apb_acc, test_amb_acc = test_amb_acc, train_apb_acc = train_apb_acc, train_amb_acc = train_amb_acc, train_loss=train_loss.item(), ent=entropy)
      
      epsilon = 1-10**-6
      if test_apb_acc > epsilon and test_amb_acc > epsilon:
         #training is basically done
         return test_apb_acc, test_amb_acc, train_apb_acc, train_amb_acc, i
      
  torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return test_apb_acc, test_amb_acc, train_apb_acc, train_amb_acc, i

if __name__ == '__main__':
    functions = ['a+b', 'a-b', 'a*b', 'a**b']
    X_train, X_test, y_train, y_test, vocab_size = get_data(functions)
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print(vocab_size)
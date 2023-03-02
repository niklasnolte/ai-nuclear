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
LIMIT = 53

def get_data(test_size = 0.2, seed = 42):
    Xs = []
    ys = []
    for a in range(LIMIT):
        for b in range(LIMIT):
            x = [a,b]
            y = (a+b)%LIMIT
            Xs.append(x)
            ys.append(y)
    Xs, ys = torch.tensor(Xs).int(), torch.tensor(ys).long()
    test_size = test_size
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=test_size, random_state=seed)
    vocab_size = (LIMIT, LIMIT) 
    return X_train, X_test, y_train, y_test, vocab_size


def train(modelclass, lr, wd, embed_dim, basepath, device, title,seed = 1, test_size = 0.2):
  #train just for modular arithmetic. gets rid of norm and assums regtype is always simn
  torch.manual_seed(seed)
  os.makedirs(basepath, exist_ok=True)
  X_train, X_test, y_train, y_test, vocab_size = get_data(test_size = test_size, seed = seed) 

  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)


  all_a = torch.tensor(list(range(vocab_size[0]))).to(device)
  all_b = torch.tensor(list(range(vocab_size[1]))).to(device)

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=len(X_train), shuffle=True)
  model = modelclass(*vocab_size, embed_dim).to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

  bar = tqdm.tqdm(range(int(2e4)))
  lowest_loss = 1e10

  loss_list = []
  entropy_list = []
  iterations = []
  train_loss_list = []

  torch.autograd.set_detect_anomaly(True)
  for i in bar:

    for X_batch, y_batch in train_loader:

      optimizer.zero_grad()
      y_pred = model(X_batch).to(device)
      train_acc = (y_pred.argmax(dim=1) == y_batch).float().mean()
      loss = loss_fn(y_pred, y_batch)
      loss.backward()
      optimizer.step()

    with torch.no_grad():

      y_pred = model.forward(X_test)
      loss = loss_fn(y_pred, y_test)
      train_loss = loss_fn(model.forward(X_train), y_train)

      if loss < lowest_loss:
        lowest_loss = loss
        best_state_dict = model.state_dict()
      if i % 500 == 0:
        torch.save(model.state_dict(), basepath + f"epoch{i}.pt")

      if i % 100 == 0:
        iterations.append(i)
        loss_list.append(loss.cpu().numpy())

        entropy = effective_dim(model, all_a)
        entropy = entropy.cpu().numpy()
        entropy_list.append(entropy)
        train_loss_list.append(train_loss.cpu().numpy())

        data_dict = {'Iteration':iterations, 'Loss':loss_list,'Train_Loss':train_loss_list, 
             'entropy':entropy}
        df = pd.DataFrame(data_dict)
        df.to_csv('csv/{0}.csv'.format(title))

      torch.save(model.state_dict(), basepath + f"latest.pt")
      test_acc = (y_test == y_pred.argmax(dim=1)).float().mean()
      bar.set_postfix(loss=loss.item(), train_acc = train_acc.item(), test_accuracy = test_acc.item(), train_loss=train_loss.item(), ent = entropy)
      if test_acc.item() > 1-10**-4:
         torch.save(best_state_dict, basepath + "best.pt")
         torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
         return test_acc.item(), train_acc.item(), i
  torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return test_acc.item(), train_acc.item(), i


def effective_dim(model, all_a):
  #calculate entropy of the embeddings
  a = model.emb_a(all_a)
  a_S = (torch.square(torch.svd(a)[1]) / (all_a.shape[0] - 1))
  a_prob = a_S/a_S.sum()
  entropy_a = -(a_prob * torch.log(a_prob)).sum()
  entropy = torch.exp(entropy_a)
  return entropy



if __name__ == '__main__':
    X_train, X_test, y_train, y_test, vocab_size = get_data()
    print(X_train[:, 0])
    s = 0.5
    plt.scatter(X_train[:,0], X_train[:, 1], label = 'train', color = 'b', s = s,alpha = 0.5)
    plt.scatter(X_test[:, 0],X_test[:,1], label = 'test', color = 'r', s = s, alpha = 0.5)
    plt.xlabel('a')
    plt.ylabel('b')
    plt.legend()
    #plt.show()
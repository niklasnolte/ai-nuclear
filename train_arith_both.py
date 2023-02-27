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
            y = [(a+b)%LIMIT, (a-b)%LIMIT]
            Xs.append(x)
            ys.append(y)
    Xs, ys = torch.tensor(Xs).int(), torch.tensor(ys).long()
    X_train, X_test, y_train, y_test = train_test_split(Xs, ys, test_size=test_size, random_state=seed)
    vocab_size = (LIMIT, LIMIT) 
    return X_train, X_test, y_train, y_test, vocab_size

def onehot_stacky(y):
    #given a 2d y vector, makes onehot stacked encodings
    a_plus_b = nn.functional.one_hot(y[:,0], num_classes=- 1)
    a_minus_b = nn.functional.one_hot(y[:,1], num_classes=- 1)
    y_total = torch.hstack((a_plus_b, a_minus_b)).float()
    return y_total

def calc_acc(y_pred, y_act):
   # calculates accuracy for a+b and a-b seperately
   apb_act = y_act[:,0] # actual a+b data
   amb_act = y_act[:,1] # actual a-b data
   seperator = y_pred.shape[1]//2
   apb_pred = y_pred[:,:seperator]
   amb_pred = y_pred[:,seperator:]
   amb_acc = (amb_act == amb_pred.argmax(dim=1)).float().mean()
   apb_acc = (apb_act == apb_pred.argmax(dim=1)).float().mean()
   return apb_acc.item(), amb_acc.item()

def train(modelclass, lr, wd, embed_dim, basepath, device, title, seed = 1, test_size = 0.2):
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
  early_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  late_optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

  bar = tqdm.tqdm(range(int(1e4)))
  lowest_loss = 1e10

  loss_list = []
  entropy_a_list = []
  entropy_b_list = []
  iterations = []
  train_loss_list = []

  torch.autograd.set_detect_anomaly(True)
  for i in bar:

    optimizer = early_optimizer# if i < len(bar)//2 else late_optimizer


    for X_batch, y_batch in train_loader:

      optimizer.zero_grad()
      y_pred = model(X_batch).to(device)
      y_batch_comb = onehot_stacky(y_batch)
      loss = loss_fn(y_pred, y_batch_comb)
      a_plus_b_train, a_minus_b_train = calc_acc(y_pred, y_batch)
    
      if reg_pca:
        regloss = model.stochastic_pca_loss(loss_fn, X_train, y_train)
        loss += reg_pca * regloss
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

        a_entropy, b_entropy = effective_dim(model, all_a, all_b)
        a_entropy = a_entropy.cpu().numpy()
        b_entropy = b_entropy.cpu().numpy()
        entropy_a_list.append(a_entropy)
        entropy_b_list.append(b_entropy)
        train_loss_list.append(train_loss.cpu().numpy())

        data_dict = {'Iteration':iterations, 'Loss':loss_list,'Train_Loss':train_loss_list, 
             'a_entropy':entropy_a_list, 'b_entropy':entropy_b_list}
        df = pd.DataFrame(data_dict)
        df.to_csv('csv/{0}.csv'.format(title))

      torch.save(model.state_dict(), basepath + f"latest.pt")

      test_apb_acc, test_amb_acc = calc_acc(y_pred, y_test)
      bar.set_postfix(loss=loss.item(), test_apb = test_apb_acc, test_amb = test_amb_acc, train_apb = a_plus_b_train, train_amb = a_minus_b_train, train_loss=train_loss.item(), ent=a_entropy)
      epsilon = 1-10**-6
      if test_apb_acc > epsilon and test_amb_acc > epsilon:
         return test_apb_acc, test_amb_acc, a_plus_b_train, a_minus_b_train, i
  torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return test_apb_acc, test_amb_acc, a_plus_b_train, a_minus_b_train, i


def effective_dim(model, all_a, all_b):
  #calculate entropy of the embeddings
  a = model.emb_a(all_a)
  b = model.emb_a(all_b)

  a_S = (torch.square(torch.svd(a)[1]) / (all_a.shape[0] - 1))
  b_S = (torch.square(torch.svd(b)[1]) / (all_b.shape[0] - 1))
  
  a_prob = a_S/a_S.sum()
  b_prob = b_S/b_S.sum()
  
  entropy_a = -(a_prob * torch.log(a_prob)).sum()
  entropy_b = -(b_prob * torch.log(b_prob)).sum()

  a_e = torch.exp(entropy_a)
  b_e = torch.exp(entropy_b)

  return a_e, b_e



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
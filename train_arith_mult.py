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
    torch.manual_seed(seed)
    Xs = []
    ys = []
    for a in range(LIMIT):
        for b in range(LIMIT):
            x = [a,b]
            y = [(a+b)%LIMIT, (a-b)%LIMIT, (a*b)%LIMIT]
            Xs.append(x)
            ys.append(y)
    Xs, ys = torch.tensor(Xs).int(), torch.tensor(ys).long()
    train_mask = torch.rand(ys.shape)>test_size
    test_mask = ~train_mask
    
    vocab_size = (LIMIT, LIMIT)
    return Xs, ys, train_mask, test_mask, vocab_size

def calc_acc(y_pred, y_act):
   # calculates accuracy for a+b and a-b seperately
   acc = (y_act.argmax(dim=1) == y_pred.argmax(dim=1)).float().mean()
   return acc.item()


def onehot_stacky(y):
    #given a 3d y vector, makes onehot stacked encodings
    a_plus_b = nn.functional.one_hot(y[:,0], num_classes=- 1)
    a_minus_b = nn.functional.one_hot(y[:,1], num_classes=- 1)
    a_times_b = nn.functional.one_hot(y[:,2], num_classes=- 1)
    y_total = torch.hstack((a_plus_b, a_minus_b, a_times_b)).float()
    return y_total


def get_y_pred(y_pred, train_mask):
   shape = y_pred.shape
   y_pred = y_pred.reshape((shape[0], 3, shape[1]//3))
   y_pred = y_pred[train_mask]
   return y_pred

def train(modelclass, lr, wd, embed_dim, basepath, device, title, seed = 1, test_size = 0.2):
  #train just for modular arithmetic. gets rid of norm and assums regtype is always simn
  torch.manual_seed(seed)
  os.makedirs(basepath, exist_ok=True)
  X, y, train_mask, test_mask, vocab_size = get_data(test_size = test_size, seed = seed) 

  X = X.to(device)
  y = y.to(device)
  train_mask = train_mask.to(device)
  test_mask = test_mask.to(device)

  all_a = torch.tensor(list(range(vocab_size[0]))).to(device)

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
    
    optimizer.zero_grad()
    y_pred = model(X).to(device)
    y_pred = get_y_pred(y_pred, train_mask)
    y_act = onehot_stacky(y)
    y_train = get_y_pred(y_act, train_mask)
    loss = loss_fn(y_pred, y_train)

    train_acc = calc_acc(y_pred, y_train)

    loss.backward()
    optimizer.step()

    with torch.no_grad():

      y_pred = model(X).to(device)
      y_pred_train = get_y_pred(y_pred, train_mask)
      y_pred_test = get_y_pred(y_pred, test_mask)

      y_act = onehot_stacky(y)
      y_act_train = get_y_pred(y_act, train_mask)
      y_act_test = get_y_pred(y_act, test_mask)

      test_loss = loss_fn(y_pred_test, y_act_test)
      train_loss = loss_fn(y_pred_train, y_act_train)

      test_acc = calc_acc(y_pred_test, y_act_test)

      shape = y_pred.shape
      apb_mask = torch.hstack((torch.ones(shape[0], 1), torch.zeros(shape[0], 1), torch.zeros(shape[0], 1))).bool()
      y_pred_apb = get_y_pred(y_pred, apb_mask)
      y_act_apb = get_y_pred(y, apb_mask)
      y_pred_apb_test = y_pred_apb[test_mask[:,0]] #gets only test items where we see a+b target
      y_act_apb_test = y_act_apb[test_mask[:,0]].flatten()
      test_apb_acc = (y_act_apb_test == y_pred_apb_test.argmax(dim=1)).float().mean().item()

      amb_mask = torch.hstack((torch.zeros(shape[0], 1), torch.ones(shape[0], 1), torch.zeros(shape[0], 1))).bool()
      y_pred_amb = get_y_pred(y_pred, amb_mask)
      y_act_amb = get_y_pred(y, amb_mask)
      y_pred_amb_test = y_pred_amb[test_mask[:,1]] #gets only test items where we see a-b target
      y_act_amb_test = y_act_amb[test_mask[:,1]].flatten()
      test_amb_acc = (y_act_amb_test == y_pred_amb_test.argmax(dim=1)).float().mean().item()

      atb_mask = torch.hstack((torch.zeros(shape[0], 1), torch.zeros(shape[0], 1), torch.ones(shape[0], 1))).bool()
      y_pred_atb = get_y_pred(y_pred, atb_mask)
      y_act_atb = get_y_pred(y, atb_mask)
      y_pred_atb_test = y_pred_atb[test_mask[:,2]] #gets only test items where we see a-b target
      y_act_atb_test = y_act_atb[test_mask[:,2]].flatten()
      test_atb_acc = (y_act_atb_test == y_pred_atb_test.argmax(dim=1)).float().mean().item()

      if loss < lowest_loss:
        lowest_loss = loss
        best_state_dict = model.state_dict()
      if i % 500 == 0:
        torch.save(model.state_dict(), basepath + f"epoch{i}.pt")

      if i % 100 == 0:
        iterations.append(i)
        loss_list.append(test_loss.cpu().numpy())

        entropy = effective_dim(model, all_a)
        entropy = entropy.cpu().numpy()

        entropy_list.append(entropy)
        train_loss_list.append(train_loss.cpu().numpy())

        data_dict = {'Iteration':iterations, 'Loss':loss_list,'Train_Loss':train_loss_list, 
             'entropy':entropy_list}
        df = pd.DataFrame(data_dict)
        df.to_csv('csv/{0}.csv'.format(title))

      torch.save(model.state_dict(), basepath + f"latest.pt")

      
      bar.set_postfix(loss=test_loss.item(), train_loss=train_loss.item(), test_apb_acc = test_apb_acc, test_amb_acc = test_amb_acc, test_atb_acc = test_atb_acc, ent=entropy)
      
      epsilon = 1-10**-4
      if test_apb_acc > epsilon:
         #training is basically done
         torch.save(best_state_dict, basepath + "best.pt")
         torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
         return test_apb_acc, test_amb_acc, test_atb_acc, test_acc,train_acc, i
      
  torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return test_apb_acc, test_amb_acc, test_atb_acc, test_acc, train_acc, i


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
    print(get_data())
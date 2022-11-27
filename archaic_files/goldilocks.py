from EmpiricalModel import Empirical
import torch
from data import get_data
import matplotlib.pyplot as plt
from torch import nn

if __name__ == '__main__':
  sd = torch.load(f"empirical_sd.pt")
  model = torch.load('empirical_model.pt')
  model.load_state_dict(sd)
  X_train, X_test, y_train, y_test, _ = get_data(heavy_elem=15)
  X = torch.vstack((X_train, X_test))
  y = torch.vstack((y_train, y_test))
  loss = nn.MSELoss(reduce =False)
  y_pred = model(X)
  print('ypred', y_pred)
  losses = (1-y_pred/y.view(-1)).abs()
  mask = losses<4000
  losses = losses[mask]
  X = X[mask]
  y = y[mask]
  print(losses.shape)
  print('loss', losses)
  plt.scatter(X[:,0], X[:, 1], c=losses)
  plt.xlabel('Z')
  plt.ylabel('N')
  plt.show()
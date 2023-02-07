import os
import pickle
import numpy as np
import torch
import sklearn
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import optuna
from model import Model2, Model3
from data import get_data, yorig, rms
from sklearn.decomposition import PCA
import time

opt = 'data'


def train(modeltype, lr, wd, alpha, hidden_dim, n_epochs, basepath, device):
  torch.manual_seed(1)
  os.makedirs(basepath, exist_ok=True)

  X_train, X_test, y_train, y_test, y_mean, y_std, vocab_size = get_data(opt,0) 
  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)

  all_protons = torch.tensor(list(range(vocab_size[0]))).to(device)
  all_neutrons = torch.tensor(list(range(vocab_size[1]))).to(device)

  train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=len(X_train), shuffle=True)

  model = modeltype(*vocab_size, hidden_dim).to(device)
  loss_fn = nn.MSELoss()
  early_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  late_optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

  bar = tqdm.tqdm(range(int(n_epochs)))
  lowest_loss = 1e10
  
  pr_e = 0
  ne_e = 0
  
  for i in bar:
    optimizer = early_optimizer# if i < len(bar)//2 else late_optimizer
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      y_pred = model(X_batch)
      loss = loss_fn(y_pred, y_batch)
      # loss += 2*model.alldims_loss(loss_fn,X_batch,y_batch)
      loss += alpha * (pr_e + ne_e)
      loss.backward()
      optimizer.step()
    with torch.no_grad():
      y_pred = model(X_test)
      loss = loss_fn(y_pred, y_test)
      train_loss = loss_fn(model(X_train), y_train)
      rms_p = rms(yorig(y_test,y_mean, y_std), yorig(y_pred,y_mean, y_std))
      if loss < lowest_loss:
          lowest_loss = loss
          best_state_dict = model.state_dict()
      # if rms_p < 0.06  :
      #     print(lr, wd)
      #     break
      if i % 100 == 0:
        torch.save({'epoch': i,
            'model_state_dict': model.state_dict(),
            'loss': loss}, basepath + f"epoch_{i}.pt")
        #calculate entropy of the embeddings
        protons = model.emb_proton(all_protons)
        neutrons = model.emb_neutron(all_neutrons)
        pca = PCA(n_components=10)
        protons = pca.fit(protons.detach().cpu().numpy()).explained_variance_ratio_
        neutrons = pca.fit(neutrons.detach().cpu().numpy()).explained_variance_ratio_
        entropy_protons = -(protons * np.log(protons)).sum()
        entropy_neutrons = -(neutrons * np.log(neutrons)).sum()
        pr_e = np.exp(entropy_protons)
        ne_e = np.exp(entropy_neutrons)

      bar.set_postfix(loss=loss.item(), train_loss=train_loss.item(), pr_e=pr_e, ne_e=ne_e, rms_p=rms_p.item())
      
  torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return lowest_loss.item()

# def objective(trial):
#   # Hyperparameters
#   lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
#   wd = trial.suggest_loguniform('wd', 1e-5, 1e-1)
#   hidden_dim = 2**trial.suggest_int('hidden_dim', 5, 10)
#   basepath = f"models/{trial.number}/"
#   return train(lr, wd, hidden_dim, basepath, device = torch.device(f"cuda:{trial.number%2}"))

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=1000, n_jobs=2)
# with open("study.pickle", "wb") as f:
#   pickle.dump(study, f)

start = time.time()

# =============================================================================
# if opt == 'empirical':
#     train(lr=2e-3, wd=1e-4, hidden_dim=64, basepath="models/test3/", device=torch.device("cuda:0"))
# else:
#     train(lr=2e-3, wd=1e-4, hidden_dim=64, basepath="models/test1/", device=torch.device("cuda:0"))
# =============================================================================

if opt=='empirical':
    basepath="models/empirical/"
elif opt=='data':
    basepath="models/data/"

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step 
    
# for lr in drange(2.79e-3, 2.83e-3, 0.005e-3):
#     for wd in drange(6.69e-4, 6.71e-4, 0.005e-4):
#         print (lr,wd)
#         train(Model2, lr=lr, wd=wd, alpha=1, hidden_dim=64, n_epochs=2e3, basepath="models/test1", device=torch.device("cpu"))

train(Model2, lr=0.0028 , wd=0.00067, alpha=1, hidden_dim=64, n_epochs=1e3, basepath=basepath, device=torch.device("cpu"))

stop = time.time()
print(f"Training time: {stop - start}s")

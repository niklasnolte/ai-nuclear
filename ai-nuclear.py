import os
import pickle
import numpy as np
import torch
import sklearn
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import optuna
from model import Model2, Model3, Model_multi
from data import get_data, clear_x, clear_y, yorig, rms
from sklearn.decomposition import PCA
import time    
    

def loss_true(loss_fn, y0_dat, y_dat, y_pred, n_obs): #returns the loss considering for each obs only the existing measurements
  loss = 0
  for obs_i in range(n_obs):     
     y_dat_i, y_pred_i = clear_y(y0_dat,y_dat,y_pred,obs_i) 
     loss_i = loss_fn(y_pred_i, y_dat_i).mean()
     loss += loss_i
  loss = loss/n_obs
  return loss

def train(modeltype, lr, wd, alpha, hidden_dim, obs, n_epochs, basepath, device):
  torch.manual_seed(100)
  os.makedirs(basepath, exist_ok=True)
  (X_train, X_test), (y_train, y_test), (y0_train, y0_test), (y0_mean, y0_std), vocab_size = get_data(opt,obs,heavy_mask,'on')

  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)
  
  if 'binding' in obs:
      (X_train_r, X_test_r), _, (y0_train_r, y0_test_r), (y0_mean_r, y0_std_r), _ = get_data(opt,obs,heavy_mask,'off')      
      
  all_protons = torch.tensor(list(range(vocab_size[0]))).to(device)
  all_neutrons = torch.tensor(list(range(vocab_size[1]))).to(device)

  train_loader = DataLoader(TensorDataset(X_train, y0_train, y_train), batch_size=256, shuffle=True)

  n_obs = len(obs)
  model = modeltype(*vocab_size, hidden_dim, n_obs).to(device)
  loss_fn = nn.MSELoss(reduction='none')
  #loss_fn = nn.MSELoss()
  early_optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
  late_optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd)

  bar = tqdm.tqdm(range(int(n_epochs)))
  lowest_loss = 1e10
  
  pr_e = 0
  ne_e = 0
  
  for i in bar:
    optimizer = early_optimizer# if i < len(bar)//2 else late_optimizer
    for X_batch, y0_batch, y_batch in train_loader:
      optimizer.zero_grad()
      model.train()
      y_pred = model(X_batch)
      loss = loss_true(loss_fn, y0_batch, y_batch, y_pred, n_obs)             
      #loss += 2*model.alldims_loss(loss_fn,X_batch,y_batch)
      loss += alpha * (pr_e + ne_e)
      loss.backward()
      optimizer.step()
    with torch.no_grad():
      model.eval()
      train_loss = loss_true(loss_fn, y0_batch, y_batch, y_pred, n_obs)
      y_pred = model(X_test)     
      loss = loss_true(loss_fn, y_test, y0_test, y_pred, n_obs)  
      A_test = (X_test[:,0] + X_test[:,1]).view(-1,1)
      
      rms0 = rms(y0_test_r, y0_test*A_test, yorig(y_pred,y0_mean, y0_std)*A_test, 0)   
      
      # rms0 = rms(y0_test,y0_test*A_test,yorig(y_pred,y0_mean, y0_std)*A_test,0)
      
      if loss < lowest_loss:
          lowest_loss = loss
          best_state_dict = model.state_dict()

      if i % 10 == 0:
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
      
        # X = torch.cat((X_train,X_test), 0)
        # A = (X[:,0] + X[:,1]).view(-1,1)
        # y0_dat = torch.cat((y0_train,y0_test), 0)
        # y0_pred = yorig(model(X),y0_mean, y0_std)
        # y0_dat[:,0] *= A.squeeze()
        # y0_pred[:,0] *= A.squeeze()
        # rms0 = rms(y0_dat, y0_pred, 0)  
       
      bar.set_postfix(loss=loss.item(), train_loss=train_loss.item(), pr_e=pr_e, ne_e=ne_e, rms0=rms0.item())
      
  # torch.save(best_state_dict, basepath + "best.pt")
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

#define the options, observables, mask
opt = 'data'
obs = ['binding']
#obs = ['half_life']
#obs = ['qbm']
heavy_mask = 8


if opt=='empirical':
    basepath="models/empirical/"
elif opt=='data':
    basepath="models/"+'+'.join(obs)+'/'
elif opt=='PySR':
    basepath="models/PySR/"
    
# basepath = "models/test/"

if not os.path.exists(basepath):
    os.makedirs(basepath)

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step 
    
# for lr in drange(1e-4, 1e-3, 1e-4):
#     for wd in drange(1e-5, 1e-4, 1e-5):
#         print (lr,wd)
#         train(Model22, lr=lr, wd=wd, alpha=1, hidden_dim=64, n_epochs=3e3, basepath="models/test1", device=torch.device("cpu"))


train(Model2, lr=0.0028 , wd=0.00067, alpha=0, hidden_dim=64, obs = obs, n_epochs=1e3, basepath=basepath, device=torch.device("cpu"))

#bfp : lr=0.0028 , wd=0.00067

stop = time.time()
print(f"Training time: {stop - start}s")

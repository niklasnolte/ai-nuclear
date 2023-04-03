import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
from sklearn.decomposition import PCA
import time    
from data import get_data, yorig, rms, clear_x, clear_y
    

def loss_true(loss_fn, y0_dat, y_dat, y_pred, n_obs): #returns the loss considering for each obs only the existing measurements
  loss = 0
  for obs_i in range(n_obs):     
     y_dat_i, y_pred_i = clear_y(y0_dat,y_dat,y_pred,obs_i) 
     loss_i = loss_fn(y_pred_i, y_dat_i).mean()
     loss += loss_i
  loss = loss/n_obs
  return loss

def train_model(modeltype, lr, wd, alpha, hidden_dim, n_epochs, obs, heavy_mask, TMS, basepath, device):
  torch.manual_seed(100)
  os.makedirs(basepath, exist_ok=True)
  (X_train, X_test), (y_train, y_test), (y0_train, y0_test), (y0_mean, y0_std), vocab_size = get_data(obs,heavy_mask,TMS) 

  X_train = X_train.to(device)
  X_test = X_test.to(device)
  y_train = y_train.to(device)
  y_test = y_test.to(device)
        
  # y_train = y_train[torch.randperm(y_train.shape[0]), :].view(-1, 1) 
  # y_test = y_test[torch.randperm(y_test.shape[0]), :].view(-1, 1)
   
      
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
      loss += alpha * (pr_e + ne_e)
      loss.backward()
      optimizer.step()
    with torch.no_grad():
      model.eval()
      train_loss = loss_true(loss_fn, y0_batch, y_batch, y_pred, n_obs)
      y_pred = model(X_test)     
      loss = loss_true(loss_fn, y0_test, y_test, y_pred, n_obs)  
      # rms_tab = rms(model, obs, heavy_mask, 'off', 'test')
      
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
      
       
      # bar.set_postfix(loss=loss.item(), train_loss=train_loss.item(), pr_e=pr_e, ne_e=ne_e, rms=rms_tab)
      bar.set_postfix(loss=loss.item(), train_loss=train_loss.item(), pr_e=pr_e, ne_e=ne_e)
      
  # torch.save(best_state_dict, basepath + "best.pt")
  torch.save(model.cpu().requires_grad_(False), os.path.join(basepath, "model.pt"))
  return lowest_loss.item()


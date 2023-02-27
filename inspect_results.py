import numpy as np
import pandas as pd
import urllib.request
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, SymLogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from data import get_data, yorig, rms, clear_x, clear_y, rms_rel



#TEST:
    
opt = 'data'

obs = ['radius']
#obs = ['binding','radius','qbm','abundance']
obsi = 'radius'

heavy_mask = 2

if opt=='empirical':
    basepath="models/empirical/"
elif opt=='data':
    basepath="models/"+'+'.join(obs)
elif opt=='PySR':
    basepath="models/PySR/"
    
model = torch.load(basepath+"/model.pt")
# model2 = torch.load("models/radius/model.pt")


def get_pred(model, opt, obsi, dati):
    (X_train, X_test), (y_train, y_test), (y0_train, y0_test), (y0_mean, y0_std), vocab_size = get_data(opt,obs,heavy_mask)
    
    if len(obs)>1:
        pos_obsi = [i for i, x in enumerate(obs) if x == obsi][0]
    else:
        pos_obsi = 0
    
    X = torch.cat((X_train,X_test), 0)
    y0_dat = torch.cat((y0_train,y0_test), 0)
        
    y_pred_train = model(X_train).detach() 
    y_pred_test = model(X_test).detach() 
    y_pred = model(X).detach() 
    
    y0_pred_train = yorig(y_pred_train,y0_mean,y0_std)
    y0_pred_test = yorig(y_pred_test,y0_mean,y0_std)
    y0_pred = yorig(y_pred,y0_mean,y0_std)
    

    Xi_train = clear_x(X_train, y0_train, pos_obsi)
    Xi_test = clear_x(X_test, y0_test, pos_obsi)
    Xi = clear_x(X, y0_dat, pos_obsi)
    (y0i_train, y0i_pred_train) = clear_y(y0_train, y0_train, y0_pred_train, pos_obsi)
    (y0i_test, y0i_pred_test) = clear_y(y0_test, y0_test, y0_pred_test, pos_obsi)
    (y0i_dat, y0i_pred) = clear_y(y0_dat, y0_dat, y0_pred, pos_obsi)
        
    if dati == 'all':
        return [X_train,X_test,X], [y0_train, y0_test, y0_dat], [y0_pred_train, y0_pred_test, y0_pred]
    else:
        return [Xi_train,Xi_test,Xi], [y0i_train, y0i_test, y0i_dat], [y0i_pred_train, y0i_pred_test, y0i_pred]

# 'qbm', 'sn', 'sp', 'Z', 'N'

(X_train,X_test,X), (y0_train, y0_test, y0_dat), (y0_pred_train, y0_pred_test, y0_pred) = get_pred(model, opt, obsi, 'all')

#_, (_, _, y0_emp), _ = get_pred(model, 'empirical', 'binding', 'all')
#y0_emp = y0_emp[y0_dat[:,0] != 0][:,0].view(-1, 1)

(Xi_train,Xi_test,Xi), (y0i_train, y0i_test, y0i_dat), (y0i_pred_train, y0i_pred_test, y0i_pred) = get_pred(model, opt, obsi, 'measure_only')


# plt.scatter(X[:,0],y0_dat[:,1], alpha =0.01)

# plt.scatter(Xi[:,0],y0i_pred, alpha =0.5)
plt.scatter(Xi[:,0],y0i_dat, alpha=1)
plt.scatter(X[:,0],y0_pred[:,0], alpha=0.2)

# create a custom legend with alpha=1 only for the second label
custom_legend = [
    Line2D([0], [0], color='C0', marker='o', linestyle=''),
    Line2D([0], [0], color='C1', marker='o', alpha=1, linestyle='')
]
#plt.legend(custom_legend, ['data', 'NN trained on '+r'$\left<r\right>~,~E_B~,~Q_{\beta}$'+' and abundance'], loc='lower right')
plt.legend(custom_legend, ['data', 'NN trained on '+r'$\left<r\right>$'], loc='lower right')

# set axis labels and show the plot
plt.ylabel('Charge radius ' + r'$\left<r\right>~\rm [fm]$')
plt.xlabel('Number of protons ' + r'$Z$')

plt.savefig('plots/'+obsi+'_from_NN_'+'+'.join(obs)+'.png', dpi=300)
plt.show()





# scale = 1000
# Z = Xi[:,0].view(-1).numpy()
# N = Xi[:,1].view(-1).numpy()
# Xnp = X.numpy()
# # DeltaE = (y0i_dat - y0i_pred).view(-1).numpy()
# DeltaE = (y0i_dat-y0_emp).view(-1).numpy()
# weights = scale*DeltaE

# to_plot = np.empty((Z.shape[0],Z.shape[0]))
# to_plot[:] = np.nan

# for i in range(Z.shape[0]):
#     to_plot[Z[i],N[i]] = scale*DeltaE[i]

# plt.imshow(to_plot, cmap='coolwarm', interpolation='hanning')

# # set the colorbar limits
# plt.clim(-0.05*scale, 0.15*scale)

# # create a colorbar and set the label
# plt.colorbar(label='Binding energy per nucleon (keV)')

# # set the x and y labels and title
# plt.xlabel('Number of neutrons ' + r'$N$')
# plt.ylabel('Number of protons ' + r'$Z$')
# plt.title('Binding energy discrepancy ' + r'$E_B^{\rm exp} -E_B^{\rm SE}$')

# # set the x and y limits
# plt.xlim(8, 140)
# plt.ylim(8, 90)

 
# magic = [2, 8, 20, 28, 50, 82, 126]

# for m in magic:
#     for n in magic:
#         if ([n,m] in Xi.tolist()):
#             if m == magic[0] and n ==magic[0]:
#                 plt.plot(m, n, 'kx', markersize=5, label='magic')
#             else:
#                 plt.plot(m, n, 'kx', markersize=5)


# plt.legend(loc='lower right')

# # plt.savefig('plots/'+obsi+'_from_M_'+'+'.join(obs)+'.png', dpi=300)
# plt.savefig('plots/Eb_emp.png', dpi=300)
# plt.show()


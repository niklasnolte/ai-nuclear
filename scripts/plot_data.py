# %%
from nuclr.config import NUCLR
from nuclr.config_utils import _deserialize_dict
from nuclr.data import prepare_nuclear_data
from lib.config_utils import _deserialize_dict
from argparse import Namespace
import matplotlib.pyplot as plt

# %%
args= Namespace(**{k:v[0] for k,v in NUCLR.items()})
args.TARGETS_CLASSIFICATION = _deserialize_dict(args.TARGETS_CLASSIFICATION)
args.TARGETS_REGRESSION = _deserialize_dict(args.TARGETS_REGRESSION)
args.DEV = "cpu"

print(args)
data = prepare_nuclear_data(args)

# %%
args.TARGETS_REGRESSION
# %%
import torch
x = data.X
y = data.y

torch.manual_seed(0)

# train test split
train_frac = 0.8
tm = torch.rand(x.shape[0]) < train_frac

# plot data[:,0] vs data[:,1]
plt.scatter(x[tm][:,0], x[tm][:,1], c=y[tm])
plt.colorbar()
plt.savefig("test.pdf")

plt.scatter(x[~tm][:,0], x[~tm][:,1], c=y[~tm])
plt.colorbar()
plt.savefig("test2.pdf")

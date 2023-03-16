# %%
from config import Task
from config_utils import _deserialize_dict
from data import prepare_nuclear_data
from argparse import Namespace
import matplotlib.pyplot as plt

# %%
TASK = Task.FULL
args= Namespace(**{k:v[0] for k,v in Task.FULL.value.items()})
args.TARGETS_CLASSIFICATION = _deserialize_dict(args.TARGETS_CLASSIFICATION)
args.TARGETS_REGRESSION = _deserialize_dict(args.TARGETS_REGRESSION)
args.DEV = "cpu"

print(args)
data = prepare_nuclear_data(args)

# %%
args.TARGETS_REGRESSION
# %%
import torch
from tqdm import trange
x = data.X
y = data.y[:,2].unsqueeze(-1)
# filter_nans
x = x[~torch.isnan(y).any(dim=-1)]
y = y[~torch.isnan(y).any(dim=-1)]

x = x[:30]
y = y[:30]

# %%
# x is cartesian product up to 10, y is random
max_ = 10
x = torch.cartesian_prod(torch.arange(max_), torch.arange(max_))
y = torch.rand(len(x), 1)
# %%
# plot data[:,0] vs data[:,1]
plt.scatter(x[:,0], x[:,1], c=y)
plt.colorbar()

# %%
class Model(torch.nn.Module):
    def __init__(self, p, o):
        super().__init__()
        self.embedding = torch.nn.ModuleList([torch.nn.Embedding(max_, p) for _ in range(2)])
        self.linear = torch.nn.Sequential(
          torch.nn.Linear(2*p, p),
          torch.nn.ReLU(),
          torch.nn.Linear(p, 1),
        )
    def forward(self, x):
        x = torch.cat([self.embedding[i](x[:, i]) for i in range(2)], dim=-1)
        return self.linear(x)
model = Model(32, 1)
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(10000):
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
      print(f"loss: {loss.item()}", end="\r")

# %%

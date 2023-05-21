# %%
from config import Task
from config_utils import _deserialize_dict
from data import prepare_nuclear_data
from loss import loss_by_task, metric_by_task
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
x = data.X
y = data.y

torch.manual_seed(0)

# train test split
train_frac = 0.8
tm = torch.rand(x.shape[0]) < train_frac

# plot data[:,0] vs data[:,1]
plt.scatter(x[tm][:,0], x[tm][:,1], c=y[tm])
plt.colorbar()
plt.show()

plt.scatter(x[~tm][:,0], x[~tm][:,1], c=y[~tm])
plt.colorbar()
plt.show()

class Model(torch.nn.Module):
    def __init__(self, p, o):
        super().__init__()
        self.embedding = torch.nn.ModuleList([torch.nn.Embedding(200, p) for _ in range(2)])
        self.linear = torch.nn.Sequential(
          torch.nn.Linear(2*p, p),
          torch.nn.ReLU(),
          torch.nn.Linear(p, o),
        )
    def forward(self, x):
        x = torch.cat([self.embedding[i](x[:, i]) for i in range(2)], dim=-1)
        return self.linear(x)

model = Model(32, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
EPOCHS=100000
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(x[tm])
    loss = loss_by_task(out, y[tm], data.output_map, args)
    loss = torch.nn.functional.mse_loss(out, y[tm])
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
      with torch.no_grad():
        model.eval()
        out_test = model(x[~tm])
        loss_test = loss_by_task(out_test, y[~tm], data.output_map, args)
        metric_test = metric_by_task(out_test, y[~tm], data.output_map, args, data.regression_transformer)
        print(f"Epoch {epoch}: train loss {loss.item():.3e}, test loss {loss_test.item():.3e}, test metric {metric_test.item():.3f}")

# %%

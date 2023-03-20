# %%
import torch
import tqdm.notebook as tqdm

p = 10
n = 100
o = 1
repeat_factor = 1
x = torch.arange(n//repeat_factor).long().repeat(2, repeat_factor).T
y = torch.randn(n, o)

# %%

class Model(torch.nn.Module):
    def __init__(self, p, o):
        super().__init__()
        self.embedding = torch.nn.ModuleList([torch.nn.Embedding(n, p) for _ in range(2)])
        self.linear = torch.nn.Linear(2 * p, o)

    def forward(self, x):
        x = torch.cat([self.embedding[i](x[:, i]) for i in range(2)], dim=-1)
        return self.linear(x)

model = Model(p, o)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
pbar = tqdm.trange(20000)
for epoch in pbar:
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    pbar.set_postfix({"loss": loss.item()})



# %%
from matplotlib import pyplot as plt
# %%
# x is cartesian product up to 10, y is random

p = 10
n = 100
o = 1
repeat_factor = 1
x = torch.arange(n//repeat_factor).long().repeat(2, repeat_factor).T
x[0, :] = 1
y = torch.randn(n, o)


# x = torch.cartesian_prod(torch.arange(10), torch.arange(10))
# y = torch.rand(len(x), 1)
# %%
# plot data[:,0] vs data[:,1]
plt.scatter(x[:,0], x[:,1], c=y, s=1)
plt.colorbar()

# %%
class Model(torch.nn.Module):
    def __init__(self, p, o):
        super().__init__()
        self.embedding = torch.nn.ModuleList([torch.nn.Embedding(200, p) for _ in range(2)])
        self.linear = torch.nn.Sequential(
          # torch.nn.Linear(2 * p, 2*p),
          # torch.nn.ReLU(),
          torch.nn.Linear(2*p, o)
        )
    def forward(self, x):
        x = torch.cat([self.embedding[i](x[:, i]) for i in range(2)], dim=-1)
        return self.linear(x)
model = Model(1024, 1)
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
pbar = tqdm.trange(1000)
for epoch in pbar:
    optimizer.zero_grad()
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    pbar.set_postfix({"loss": loss.item()})
# %%
print(model.embedding[0].weight)
print(model.embedding[1].weight)

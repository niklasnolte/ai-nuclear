# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tqdm
import os, shutil

functions_ = [
    lambda x, y: x + y,
    lambda x, y: abs(x - y),
    lambda x, y: (x + y) ** (2 / 3),
    lambda x, y: torch.log(x + y + 1),
    lambda x, y: torch.exp(-(x + y)**(1/2)/5),
]

metrics = ["epoch", "train_loss", "val_loss"]
metrics.extend([f"{i}_train_loss" for i in range(len(functions_))])
metrics.extend([f"{i}_val_loss" for i in range(len(functions_))])

def run(use_functions, functions=functions_):
    functions = [f for f, use in zip(functions, use_functions) if use]

    torch.manual_seed(2)
    log = True
    name = f"functions_{''.join([str(int(use)) for use in use_functions])}"
    # Modules
    class ResidualBlock(nn.Module):
        def __init__(self, d_model):
            super(ResidualBlock, self).__init__()
            self.nonlinear = torch.nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.nonlinear(x) + x

    SimpleBlock = lambda i, o: nn.Sequential(nn.Linear(i, o), nn.ReLU())

    # Hyperparameters
    num_epochs = 3000
    learning_rate = 1e-4
    weight_decay = 1e-1

    # Data
    P = 5
    train_frac = 0.05

    X = torch.cartesian_prod(
        torch.arange(P), torch.arange(P), torch.arange(len(functions))
    )
    y = torch.vstack([functions[idx](x, y) for x, y, idx in X]).float()
    for i in range(len(functions)):
        y[i :: len(functions)] = (
            y[i :: len(functions)] - y[i :: len(functions)].min()
        ) / (y[i :: len(functions)].max() - y[i :: len(functions)].min())
    print(X)
    print(y.view(1,-1))
    X[:, 2] += P

    shuffle = torch.randperm(len(X))
    X, y = X[shuffle], y[shuffle]
    X_train, X_val = X[: int(train_frac * len(X))], X[int(train_frac * len(X)) :]
    y_train, y_val = y[: int(train_frac * len(y))], y[int(train_frac * len(y)) :]

    # Model
    class Model(nn.Module):
        def __init__(self, hidden_dim=64, num_layers=2):
            super(Model, self).__init__()
            self.embedding = nn.Embedding(P + len(functions), hidden_dim)
            self.nonlinear = torch.nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                # *[SimpleBlock(hidden_dim, hidden_dim) for _ in range(num_layers)]
                *[ResidualBlock(hidden_dim) for _ in range(num_layers)],
            )
            self.readout = nn.Linear(hidden_dim, len(functions))

        def forward(self, x):
            task = x[:, -1] - P
            x = self.embedding(x).flatten(start_dim=1)
            x = self.nonlinear(x)
            x = self.readout(x)
            x = x[torch.arange(len(x)), task]
            return x.view(-1, 1)

    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
    model = Model()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(loader))
    criterion = nn.MSELoss()
    def loss_by_function(y_pred, y, x):
        losses = [-1] * len(functions_)
        write_idx = [ i for i, use in enumerate(use_functions) if use]
        for i, write_idx in enumerate(write_idx):
            y_pred_ = y_pred[x[:, 2] == i + P]
            y_ = y[x[:, 2] == i + P]
            losses[write_idx] = criterion(y_pred_, y_)
        return losses
    # Train

    if __name__ == "__main__":
        title = "Train Loss | Val Loss"
        title += " | " + " | ".join([f"{i}_train_loss" for i in range(len(functions_)) if use_functions[i]])
        title += " | " + " | ".join([f"{i}_val_loss" for i in range(len(functions_)) if use_functions[i]])
        print(title)
        pbar = tqdm.trange(num_epochs, leave=True, position=0)
        for epoch in pbar:
            for x, y in loader:
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                scheduler.step()
            with torch.no_grad():
                y_pred = model(X_train)
                loss = criterion(y_pred, y_train)
                train_losses = loss_by_function(y_pred, y_train, X_train)
            # optimizer.zero_grad()
            # y_pred = model(X_train)
            # loss = criterion(y_pred, y_train)
            # loss.backward()
            # optimizer.step()
            # train_losses = loss_by_function(y_pred, y_train, X_train)

            with torch.no_grad():
                y_pred = model(X_val)
                val_loss = criterion(y_pred, y_val)
                val_losses = loss_by_function(y_pred, y_val, X_val)

            msg = f"{loss:10.2e} | {val_loss:>8.2e}"
            msg += " | " + " | ".join([f"{l:8.2e}" for l in train_losses + val_losses])
            pbar.set_description(msg)

        return model


# %%
if __name__ == "__main__":
    list_uses = [[False] * len(functions_) for _ in  range(len(functions_))]
    for i in range(len(functions_)):
        list_uses[i][i] = True
    list_uses += [[True] * len(functions_)]
models = []
for use_functions in list_uses:
    model = run(use_functions, functions_)
    models.append(model)
# %%
for model in models:
    torch.save(model.state_dict(), "models/" + "".join([str(int(use)) for use in use_functions]) + ".pt")
# %%
# Plot accuracies
# read data
import pandas as pd
from matplotlib import pyplot as plt
log_dirs = ["log/functions_" + "".join([str(int(use)) for use in use_functions]) for use_functions in list_uses]
fig, ax = plt.subplots()
window = 10
for log in log_dirs[:-1]:
    metrics = pd.read_csv(log + "/metrics.csv", index_col=False)
    val_loss = metrics["val_loss"]
    val_loss = [np.mean(val_loss[max(0, i - window) : i + 1]) for i in range(len(val_loss))]
    plt.plot(metrics["epoch"], val_loss, ls="--", label="trained on " + log.split("_")[-1], c=f"C{log_dirs.index(log)}")
log = log_dirs[-1]
metrics = pd.read_csv(log + "/metrics.csv", index_col=False)
for i in range(len(functions_)):
    val_loss = metrics[f"{i}_val_loss"]
    val_loss = [np.mean(val_loss[max(0, i - window) : i + 1]) for i in range(len(val_loss))]
    plt.plot(metrics["epoch"], val_loss, ls="-")
# add solid line legend
plt.plot([], [], ls="-", c="k", label=f"Trained on all")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.xscale("log")
plt.yscale("log")
plt.title("Validation Loss by Function")
plt.savefig("functions_loss_res.pdf")

# %%
torch.manual_seed(2)
functions = functions_
P = 100
train_frac = 0.05

X = torch.cartesian_prod(
    torch.arange(P), torch.arange(P), torch.arange(len(functions))
)
y = torch.vstack([functions[idx](x, y) for x, y, idx in X]).float()
for i in range(len(functions)):
    y[i :: len(functions)] = (
        y[i :: len(functions)] - y[i :: len(functions)].min()
    ) / (y[i :: len(functions)].max() - y[i :: len(functions)].min())
X[:, 2] += P
shuffle = torch.randperm(len(X))
X, y = X[shuffle], y[shuffle]
X_train, X_val = X[: int(train_frac * len(X))], X[int(train_frac * len(X)) :]
y_train, y_val = y[: int(train_frac * len(y))], y[int(train_frac * len(y)) :]

y_train_pred = model(X_train).detach().cpu().numpy()
y_val_pred = model(X_val).detach().cpu().numpy()
# plot train and val
fig, axes = plt.subplots(1, len(functions), figsize=(len(functions) * 3, 3))
# for i in range(len(functions)):
#     axes[i].scatter(X_train[X_train[:, 2] == i + P, 0], X_train[X_train[:, 2] == i + P, 1], c=y_train[X_train[:, 2] == i + P],
#                     label = "train", c="C0", marker="x")
#     axes[i].scatter(X_val[X_val[:, 2] == i + P, 0], X_val[X_val[:, 2] == i + P, 1], c=y_val[X_val[:, 2] == i + P],
#                     label = "val", c="C1", marker="x")
for i in range(len(functions)):
    ax = axes[i]
    mask = X_train[:, 2] == i + P
    # ax.scatter(X_train[mask, 0], y_train_pred[mask], c="C0", marker="o", label="train", alpha=0.5, s=1)
    # ax.scatter(X_train[mask, 0], y_train[mask], c="C1", marker="x", label="true train", alpha=0.5, s=1)
    mask = X_val[:, 2] == i + P
    mask[100:] = False
    ax.scatter(X_val[mask, 0], y_val_pred[mask], c="C2", marker="o", label="val", alpha=0.5, s=10)
    ax.scatter(X_val[mask, 0], y_val[mask], c="C3", marker="x", label="true val", alpha=0.5, s=10)
# rescale legend markers
plt.legend()



# %%
model = models[-1]
from sklearn.decomposition import PCA
import numpy as np
pca = PCA(n_components=2)

emb = model.embedding.weight.detach().cpu().numpy()
emb = pca.fit_transform(emb)
fig, ax = plt.subplots()
c = plt.cm.viridis(np.linspace(0, 1, len(emb)))
plt.scatter(emb[:, 0], emb[:, 1], c=c, s=1)
for i, (x, y) in enumerate(emb):
    ax.annotate(f"{i}", (x, y), fontsize=8, c=c[i])


# %%
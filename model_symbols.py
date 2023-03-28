# %%
from matplotlib import pyplot as plt
import torch
from model import Base, ResidualBlock
from torch import nn
from typing import Union, Iterable
from data import train_test_split_exact
import tqdm

# %%
class BaselineModel(Base):
    def __init__(
        self,
        vocab_size: Union[int, Iterable],
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ):
        """
        :param vocab_size: number of tokens in the vocabulary,
          or an Iterable of vocab sizes for each input. One embedding layer will be created for each input.
        :param hidden_dim: dimension of the hidden layer
        :param output_dim: dimension of the output layer
        """

        super().__init__(vocab_size, hidden_dim)

        self.nonlinear = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            # ResidualBlock(hidden_dim, dropout=dropout),
            # nn.SiLU(),
            # ResidualBlock(hidden_dim, dropout=dropout),
            # nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward_with_embeddings(self, x, embs):  # embs: [ batch_size, 2 * hidden_dim ]
        x = self.embed_input(x, embs)
        x = self.nonlinear(x)  # [ batch_size, hidden_dim ]
        return self.readout(x)  # [ batch_size, output_dim ]

def train_test_split_exact(size, train_frac=0.8, seed=0, force_train_mask=None):
    torch.manual_seed(seed)
    train_mask = torch.ones(size, dtype=torch.bool)
    train_mask[int(train_frac * size) :] = False 
    train_mask = train_mask[torch.randperm(size)]
    if force_train_mask is not None:
        train_mask = train_mask | force_train_mask
    test_mask = ~train_mask
    return train_mask, test_mask


def accuracy(out, y, task_mask=None):
    with torch.no_grad():
        if task_mask is None:
            return (out.argmax(dim=1) == y).float().mean() * 100
        else:
            return (out.argmax(dim=1)[task_mask] == y[task_mask]).float().mean() * 100


def get_task_accs(out, y, X):
    task_accs = {}
    for op in operations:
        task_mask = X[:, 2] == operations.index(op)
        task_accs[op] = accuracy(out, y.long(), task_mask=task_mask)
    return task_accs

# %%
torch.manual_seed(0)

operations = ["add", "subtract", "multiply"]
# operations = ["add"]
# operations = ["add", "subtract"]
P = 53
X = torch.cartesian_prod(
    torch.arange(P), torch.arange(P), torch.arange(len(operations))
)
# order by operation
X = torch.cat([X[X[:, 2] == i] for i in range(len(operations))])
y = torch.zeros(len(X))
TRAIN_FRAC = 0.8
# modular arithemtic
for i, (a, b, op) in enumerate(X):
    if operations[op] == "add":
        y[i] = (a + b) % P
    elif operations[op] == "subtract":
        y[i] = (a - b) % P
    elif operations[op] == "multiply":
        y[i] = (a * b) % P

# force all multiplication examples to be in the training set
mask_train, mask_test = [], []
seeds = [0, 1, 2]
for rng, op in zip(seeds, operations):
    if op == "multiply":
        mask_train_op = torch.ones(P * P, dtype=torch.bool)
        mask_test_op = torch.zeros(P * P, dtype=torch.bool)
    else:
        mask_train_op, mask_test_op = train_test_split_exact(P * P, train_frac=TRAIN_FRAC, seed=rng)
    mask_train.append(mask_train_op)
    mask_test.append(mask_test_op)
mask_train = torch.cat(mask_train)
mask_test = torch.cat(mask_test)

# First, let's plot an imshow of train and test split per task
fig, ax = plt.subplots(1, len(operations), figsize=(3 * len(operations), 4))
if len(operations) == 1: ax = [ax]
for i, op in enumerate(operations):
    task_mask = X[:, 2] == operations.index(op)
    mask = mask_train[task_mask]
    ax[i].imshow(mask.reshape(P, P).float(), cmap="Greens", vmin=0, vmax=1)
    ax[i].set_title(op)
    ax[i].set_xticks([])
    ax[i].set_yticks([])


# %%
model = BaselineModel(vocab_size=(P + len(operations)), hidden_dim=64, output_dim=P)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

print("T:  loss |  acc  || V:  loss |  acc")
bar = tqdm.trange(20000)
for epoch in bar:
    model.train()
    optimizer.zero_grad()
    out = model(X[mask_train])
    loss_train = criterion(out, y[mask_train].long())
    loss_train.backward()
    optimizer.step()
    acc_train = accuracy(out, y[mask_train].long())
    task_accs_train = get_task_accs(out, y[mask_train].long(), X[mask_train])

    model.eval()
    out = model(X[mask_test])
    loss_test = criterion(out, y[mask_test].long())
    acc_test = accuracy(out, y[mask_test].long())
    msg = (
        f"T: {loss_train:.3f} | {acc_train:.3f} || V: {loss_test:.3f} | {acc_test:.3f}"
    )
    bar.set_description_str(msg)
    task_accs_test = get_task_accs(out, y[mask_test].long(), X[mask_test])
    postfix = " | ".join([f"{op}: {acc:.2f}" for op, acc in task_accs_test.items()])
    bar.set_postfix_str(postfix)


# %%
# Now let's plot the embeddings
from sklearn.decomposition import PCA
from matplotlib.cm import get_cmap
import numpy as np

pca = PCA(n_components=2)
embs = model.emb[0].weight.detach().numpy()
embs = pca.fit_transform(embs)
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=250)
# embedding i has marker i and operations are also marked appropriately
ax.scatter(embs[:, 0], embs[:, 1], c=range(len(embs)), cmap="viridis", s=1)
colors = get_cmap("viridis")(np.linspace(0, 1, P))
for i in range(P):
    ax.text(embs[i, 0], embs[i, 1], str(i), color=colors[i], fontsize=8)
for i, op in enumerate(operations):
    # offset operations by 5 %
    ax.text(embs[P + i, 0] * 2.15 ** i , embs[P + i, 1]* 2.15 ** i, op, color="k", fontsize=8)
ax.plot(embs[:P, 0], embs[:P, 1], "k", alpha=0.5, linewidth=0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Embeddings")


# %%

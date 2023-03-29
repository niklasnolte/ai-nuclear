import optuna
from matplotlib import pyplot as plt
import torch
from torch import nn
import tqdm
import mup
from model import make_mup, Base, ResidualBlock
from typing import Union, Iterable
from functools import partial
import json
from optuna.integration.wandb import WeightsAndBiasesCallback


class BaselineModel(Base):
    def __init__(
        self,
        vocab_size: Union[int, Iterable],
        hidden_dim: int,
        output_dim: int,
        factor=2,
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
            nn.Linear(factor * hidden_dim, hidden_dim),
            nn.SiLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.SiLU(),
            ResidualBlock(hidden_dim, dropout=dropout),
            nn.SiLU(),
            ResidualBlock(hidden_dim, dropout=dropout),
            nn.SiLU(),
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


def plot():
    # First, let's plot an imshow of train and test split per task
    fig, ax = plt.subplots(1, len(operations), figsize=(3 * len(operations), 4))
    if len(operations) == 1:
        ax = [ax]
    for i, op in enumerate(operations):
        task_mask = X[:, 2] == operations.index(op)
        mask = mask_train[task_mask]
        ax[i].imshow(mask.reshape(P, P).float(), cmap="Greens", vmin=0, vmax=1)
        ax[i].set_title(op)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.show()


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


# operations = ["multiply"]
operations = ["add", "subtract", "multiply"]
# operations = ["add"]
# operations = ["add", "subtract"]
P = 53
SEED = 0
TRAIN_FRAC = 0.8
HIDDEN_DIM = 32

torch.manual_seed(SEED)
X = torch.cartesian_prod(
    torch.arange(P), torch.arange(P), torch.arange(len(operations))
)

X = torch.cat([X[X[:, 2] == i] for i in range(len(operations))])
y = torch.zeros(len(X))
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
seeds = {"add": 0, "subtract": 1, "multiply": 2}
for op in operations:
    # if op == "multiply":
    #     mask_train_op = torch.ones(P * P, dtype=torch.bool)
    #     mask_test_op = torch.zeros(P * P, dtype=torch.bool)
    # else:
    #     mask_train_op, mask_test_op = train_test_split_exact(P * P, train_frac=TRAIN_FRAC, seed=rng)
    mask_train_op, mask_test_op = train_test_split_exact(
        P * P, train_frac=TRAIN_FRAC, seed=seeds[op]
    )
    mask_train.append(mask_train_op)
    mask_test.append(mask_test_op)
mask_train = torch.cat(mask_train)
mask_test = torch.cat(mask_test)


def train_model(
    lr=1e-3,
    epochs=100,
    hidden_dim=128,
    weight_decay=1e-3,
    weight_decay_emb=0,
    dropout=0.,
    init_scale=1,
    amsgrad=True,
    scheduler=False,
):
    torch.manual_seed(0)
    model_fn = partial(
        BaselineModel, vocab_size=(P + len(operations)), output_dim=P, 
        factor=3, dropout=dropout,
    )
    model = make_mup(model_fn, hidden_dim=hidden_dim)
    model.emb[0].weight.data *= init_scale
    param_groups = [
        {"params": model.emb.parameters(), "weight_decay": weight_decay_emb},
        {"params": model.nonlinear.parameters(), "weight_decay": weight_decay},
        {"params": model.readout.parameters(), "weight_decay": weight_decay},
    ]
    optimizer = mup.MuAdamW(param_groups, lr=lr, amsgrad=amsgrad)
    criterion = nn.CrossEntropyLoss()
    if scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    print("T:  loss |  acc  || V:  loss |  acc")
    bar = tqdm.trange(epochs, dynamic_ncols=True)
    for epoch in bar:
        model.train()
        optimizer.zero_grad()
        out = model(X[mask_train])
        loss_train = criterion(out, y[mask_train].long())
        loss_train.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        acc_train = accuracy(out, y[mask_train].long())
        # task_accs_train = get_task_accs(out, y[mask_train].long(), X[mask_train])

        model.eval()
        out = model(X[mask_test])
        loss_test = criterion(out, y[mask_test].long())
        acc_test = accuracy(out, y[mask_test].long())
        msg = f"T: {loss_train:.3f} | {acc_train:.3f} || V: {loss_test:.3f} | {acc_test:.3f}"
        bar.set_description_str(msg)
        task_accs_test = get_task_accs(out, y[mask_test].long(), X[mask_test])
        postfix = " | ".join([f"{op}: {acc:.2f}" for op, acc in task_accs_test.items()])
        bar.set_postfix_str(postfix)
    return loss_test, acc_test, model


def objective(trial):
    EPOCHS = 1000
    LR = trial.suggest_float("LR", 1e-5, 1e-2, log=True)
    WD = trial.suggest_float("WD", 1e-5, 1e-1, log=True)
    # WD_EMB = trial.suggest_float("WD_EMB", 1e-5, 1e-1, log=True)
    DROP = trial.suggest_float("DROP", 0, 0.5)
    INIT_SCALE = trial.suggest_float("INIT_SCALE", 1e-1, 1, log=True)
    metric, *_ = train_model(
        lr=LR,
        epochs=EPOCHS,
        hidden_dim=HIDDEN_DIM,
        weight_decay=WD,
        weight_decay_emb=0,
        dropout=DROP,
        init_scale=INIT_SCALE,
        amsgrad=True,
    )
    return metric


if __name__ == "__main__":
    from sys import argv
    import os
    wandb_kwargs = {"project": "ai-nuclear-tune", "group": "modular-arithmetic", "tags": ["modular-arithmetic", "optuna", "test"]}
    wandbc = WeightsAndBiasesCallback(metric_name="val loss", wandb_kwargs=wandb_kwargs)

    if "-n" in argv:
        n_trials = int(argv[argv.index("-n") + 1])
    else:
        n_trials = 200

    overwrite = "-o" in argv

    outfile = f"best_params_{n_trials}.json"
    if not os.path.exists(outfile) or overwrite:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, callbacks=[wandbc])
        print(
            f"\n\ntraining model with best params trial: {study.best_trial.number} (acc: {study.best_trial.value:.2f})"
        )
        # save best params
        with open(outfile, "w") as f:
            json.dump(study.best_params, f)
    else:
        print(f"best params file already exists: {outfile}")
    best_params = json.load(open(outfile))
    # train model with best params
    print(f"Best params: {best_params}")
    loss_test, acc_test, model = train_model(
        lr=best_params["LR"],
        epochs=30000,
        hidden_dim=HIDDEN_DIM,
        weight_decay=best_params["WD"],
        # weight_decay_emb=best_params["WD_EMB"],
        weight_decay_emb=0,
        init_scale=best_params["INIT_SCALE"],
        amsgrad=True,
        scheduler=False,
    )
    print(f"Final test accuracy: {acc_test:.2f}")
    torch.save(model, "model_best_tuning.pt")

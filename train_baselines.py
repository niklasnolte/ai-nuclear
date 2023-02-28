import torch
import tqdm
from data import prepare_data, train_test_split
from config import Targets, TrainConfig
from model import BaselineModel, ResidualModel
from loss import loss_by_task, metric_by_task
import os

torch.manual_seed(TrainConfig.SEED)

for target in ["isospin"]:
    columns = [target]
    data = prepare_data(columns=columns)
    train_mask, test_mask = train_test_split(
        data, train_frac=TrainConfig.TRAIN_FRAC, seed=TrainConfig.SEED
    )

    save_path = os.path.join(TrainConfig.ROOTPATH, TrainConfig.MODEL, '_'.join(columns))
    os.makedirs(save_path, exist_ok=True)

    # set up model
    n_protons, n_neutrons = data.vocab_size
    output_dims = data.output_map.values()
    if TrainConfig.MODEL == "baseline":
        model_class = BaselineModel
    elif TrainConfig.MODEL == "residual":
        model_class = ResidualModel

    model = model_class(
        n_protons,
        n_neutrons,
        hidden_dim=TrainConfig.HIDDEN_DIM,
        output_dim=sum(output_dims),
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=TrainConfig.LR, weight_decay=TrainConfig.WD
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=TrainConfig.EPOCHS
    )

    # remove old models
    for f in os.listdir(save_path):
        os.remove(os.path.join(save_path, f))

    bar = tqdm.trange(TrainConfig.EPOCHS)
    for epoch in bar:
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.X)
        train_loss = loss_by_task(out[train_mask], data.y[train_mask], data.output_map)
        loss = train_loss.mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 100 == 0:
            with torch.no_grad():
                # Test
                model.eval()
                val_acc = metric_by_task(out[test_mask], data.y[test_mask], data.output_map, data.regression_transformer)
                train_acc = metric_by_task(out[train_mask], data.y[train_mask], data.output_map, data.regression_transformer)
                val_loss = loss_by_task(out[test_mask], data.y[test_mask], data.output_map).mean()
            msg = f"Columns: {', '.join(columns)}: Train: {train_loss.item():.2e}|{train_acc.item():.2e}, "
            msg += f"Val: {val_loss.item():.2e}|{val_acc.item():.2e}"
            bar.set_description(msg)
            # save model
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{epoch}.pt"))

    torch.save(model, os.path.join(save_path, "model_full.pt"))

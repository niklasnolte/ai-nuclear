import os
import tqdm
import torch
import argparse
from data import prepare_nuclear_data, prepare_modular_data
from model import get_model_and_optim
from loss import (
    loss_by_task,
    metric_by_task,
    weight_by_task,
    random_softmax,
    regularize_embedding_dim,
)
from config import Task


def train(task: Task, args: argparse.Namespace, basedir: str):
    if task == Task.FULL or task == Task.DEBUG:
        data = prepare_nuclear_data(args)
    elif task == Task.MODULAR:
        data = prepare_modular_data(args)

    best_model = None
    best_loss = float("inf")

    y_train = data.y[data.train_mask]
    y_val = data.y[data.val_mask]

    model, optimizer = get_model_and_optim(data, args)

    if args.WANDB:
        import wandb

        wandb.config.update({"n_params": sum(p.numel() for p in model.parameters())})
    weights = weight_by_task(data.output_map, args)
    if not args.WANDB:
        bar = tqdm.trange(
            args.EPOCHS,
        )
    else:
        bar = range(args.EPOCHS)

    Xs_task = [data.X.clone() for _ in data.output_map]
    for taski, task in enumerate(data.output_map):
        Xs_task[taski][:,len(data.vocab_size)-1] = torch.ones(data.X.shape[0]) * taski

    for epoch in range(args.EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = torch.zeros(data.y.shape).to(args.DEV)
        out_col = 0
        for taski, task in enumerate(data.output_map):
          # make a task specific X (task feature)
          out_task = model(Xs_task[taski])
          out_size = data.output_map[task]
          slice_ = slice(out_col, out_col + out_size)
          out[:, slice_] = out_task[:, slice_]
          out_col += out_size

        train_losses = loss_by_task(
            out[data.train_mask], y_train, data.output_map, args
        )

        train_losses = train_losses.mean(dim=1)
        train_loss = (weights * train_losses).mean()
        train_loss.backward()
        optimizer.step()

        if epoch % args.LOG_FREQ == 0:
            with torch.no_grad():
                model.eval()
                train_metrics = metric_by_task(
                    out[data.train_mask],
                    data.X[data.train_mask],
                    y_train,
                    data.output_map,
                    args,
                    qt=data.regression_transformer,
                )
                val_metrics = metric_by_task(
                    out[data.val_mask],
                    data.X[data.val_mask],
                    y_val,
                    data.output_map,
                    args,
                    qt=data.regression_transformer,
                )
                val_losses = loss_by_task(
                    out[data.val_mask], y_val, data.output_map, args
                ).mean(dim=1)
                val_loss = (weights * val_losses).mean()
                # keep track of the best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = model.state_dict().copy()
                    if args.WANDB:
                        wandb.run.summary["best_val_loss"] = best_loss.item()
                        wandb.run.summary["best_epoch"] = epoch
                        for i, target in enumerate(data.output_map.keys()):
                            wandb.run.summary[f"best_val_{target}"] = val_metrics[
                                i
                            ].item()

                if args.WANDB:
                    for i, target in enumerate(data.output_map.keys()):
                        wandb.log(
                            {
                                f"loss_train_{target}": train_losses[i].item(),
                                f"loss_val_{target}": val_losses[i].item(),
                                f"metric_train_{target}": train_metrics[i].item(),
                                f"metric_val_{target}": val_metrics[i].item(),
                            },
                            step=epoch,
                        )
                    wandb.log(
                        {
                            "loss_train_combined": train_loss.item(),
                            "loss_val_combined": val_loss.item(),
                        },
                        step=epoch,
                    )
                else:
                    msg = f"\nEpoch {epoch:<6} Train Losses | Metrics\n"
                    for i, target in enumerate(data.output_map.keys()):
                        msg += f"{target:>15}: {train_losses[i].item():.4e} | {train_metrics[i].item():.6f}\n"
                    msg += f"\nEpoch {epoch:<8} Val Losses | Metrics\n"
                    for i, target in enumerate(data.output_map.keys()):
                        msg += f"{target:>15}: {val_losses[i].item():.4e} | {val_metrics[i].item():.6f}\n"

                    print(msg)
                    bar.update(args.LOG_FREQ)

        if epoch % args.CKPT_FREQ == 0:
            torch.save(model.state_dict(), os.path.join(basedir, f"model_{epoch}.pt"))
    torch.save(model, os.path.join(basedir, "model_FULL.pt"))
    torch.save(best_model, os.path.join(basedir, "model_FULL_best.pt"))

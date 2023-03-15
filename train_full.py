import os
import tqdm
import torch
import argparse
import wandb
from data import prepare_nuclear_data, train_test_split, prepare_modular_data
from model import get_model_and_optim
from loss import loss_by_task, metric_by_task, weight_by_task, random_softmax, regularize_embedding_dim
from config import Task

def train(task: Task, args: argparse.Namespace, basedir: str):
    if task == Task.FULL or task == Task.DEBUG:
      data = prepare_nuclear_data(args)
    elif task == Task.MODULAR:
      data = prepare_modular_data(args)


    best_model = None
    best_loss = float("inf")

    DEVICE = args.DEV
    train_mask, val_mask = train_test_split(
        data, train_frac=args.TRAIN_FRAC, seed=args.SEED
    )

    model, optimizer = get_model_and_optim(data, args)
    if args.WANDB:
      wandb.config.update({"n_params": sum(p.numel() for p in model.parameters())})
    weights = weight_by_task(data.output_map, args)
    if not args.WANDB:
        bar = tqdm.trange(args.EPOCHS)
    else:
        bar = range(args.EPOCHS)
    for epoch in bar:
        # Train
        model.train()
        optimizer.zero_grad()
        out = model(data.X)

        train_losses = loss_by_task(
            out[train_mask], data.y[train_mask], data.output_map, args
        )

        if args.RANDOM_WEIGHTS:
            weight_scaler = random_softmax(weights.shape, scale=args.RANDOM_WEIGHTS).to(
                DEVICE
            )
        else:
            weight_scaler = 1

        train_loss = weights * train_losses * weight_scaler

        if args.DIMREG_COEFF > 0:
          dimreg = regularize_embedding_dim(model, data.X[train_mask], data.y[train_mask], data.output_map, args)
          train_loss += args.DIMREG_COEFF * dimreg

        train_loss = train_loss.mean()
        train_loss.backward()
        optimizer.step()

        if epoch % args.LOG_FREQ == 0:
            with torch.no_grad():
                model.eval()
                train_metrics = metric_by_task(
                    out[train_mask],
                    data.y[train_mask],
                    data.output_map,
                    args,
                    qt=data.regression_transformer,
                )
                val_metrics = metric_by_task(
                    out[val_mask],
                    data.y[val_mask],
                    data.output_map,
                    args,
                    qt=data.regression_transformer,
                )
                val_losses = loss_by_task(
                    out[val_mask], data.y[val_mask], data.output_map, args
                )
                val_loss = (weights * val_losses).mean()
                # keep track of the best model
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = model.state_dict().copy()
                    if args.WANDB:
                        wandb.run.summary["best_val_loss"] = best_loss.item()
                        wandb.run.summary["best_epoch"] = epoch
                        for i, target in enumerate(data.output_map.keys()):
                            wandb.run.summary[f"best_val_{target}"] = val_metrics[i].item()


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
                        msg += f"{target:>15}: {train_losses[i].item():.2e} | {train_metrics[i].item():.2f}\n"
                    msg += f"\nEpoch {epoch:<8} Val Losses | Metrics\n"
                    for i, target in enumerate(data.output_map.keys()):
                        msg += f"{target:>15}: {val_losses[i].item():.2e} | {val_metrics[i].item():.2f}\n"
                    print(msg)

        if epoch % args.CKPT_FREQ == 0:
            torch.save(model.state_dict(), os.path.join(basedir, f"model_{epoch}.pt"))
    torch.save(model, os.path.join(basedir, "model_FULL.pt"))
    torch.save(best_model, os.path.join(basedir, "model_FULL_best.pt"))

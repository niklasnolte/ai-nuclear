import os
import tqdm
import torch
import argparse
import wandb
from data import prepare_data, train_test_split
from model import get_model_and_optim
from loss import loss_by_task, metric_by_task, weight_by_task


def train_FULL(args: argparse.Namespace, basedir: str, debug: bool = False):
    if debug: print("DEBUG MODE")
    data = prepare_data(args)
    train_mask, test_mask = train_test_split(
        data, train_frac=args.TRAIN_FRAC, seed=args.SEED
    )

    model, optimizer = get_model_and_optim(data, args)
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
        train_loss = loss_by_task(
            out[train_mask], data.y[train_mask], data.output_map, args
        )
        loss = (weights * train_loss).mean()
        loss.backward()
        optimizer.step()
        if epoch % args.LOG_FREQ == 0:
            with torch.no_grad():
                # Test
                model.eval()
                val_loss = metric_by_task(
                    out[test_mask],
                    data.y[test_mask],
                    data.output_map,
                    args,
                    qt=data.regression_transformer,
                )
                # save to wandb
                if args.WANDB:
                    for i, target in enumerate(data.output_map.keys()):
                        wandb.log(
                            {
                                f"train_{target}": train_loss[i].item(),
                                f"val_{target}": val_loss[i].item(),
                            },
                            step=epoch
                        )
                else:
                  msg = f"\nEpoch {epoch} Train losses:\n"
                  for i, target in enumerate(data.output_map.keys()):
                      msg += f"{target}: {train_loss[i].item():.2e}\n"
                  msg += f"\nEpoch {epoch} Val metrics:\n"
                  for i, target in enumerate(data.output_map.keys()):
                      msg += f"{target}: {val_loss[i].item():.4f}\n"
                  print(msg)
                # save model
                torch.save(model.state_dict(), os.path.join(basedir, f"model_{epoch}.pt"))
    if debug: # TODO maybe debug should have its own train rule
        torch.save(model, os.path.join(basedir, "model_debug.pt"))
        print("DEBUG MODE: saving model_debug.pt")
        print("SAVED MODEL: ", os.path.join(basedir, "model_debug.pt"))
    else:
        print("wrong branch")
        breakpoint()
        torch.save(model, os.path.join(basedir, "model_full.pt"))

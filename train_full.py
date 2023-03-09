import os
import tqdm
import torch
import argparse
import wandb
from data import prepare_nuclear_data, train_test_split
from model import get_model_and_optim
from loss import loss_by_task, metric_by_task, weight_by_task, random_softmax, regularize_embedding_dim

def train_FULL(args: argparse.Namespace, basedir: str):
    data = prepare_nuclear_data(args)
    DEVICE = args.DEV
    train_mask, test_mask = train_test_split(
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
        # TODO there was a problem where loss was on cpu and backwarding to
        train_loss = loss_by_task(
            out[train_mask], data.y[train_mask], data.output_map, args
        )


        if args.RANDOM_WEIGHTS:
            weight_scaler = random_softmax(weights.shape, scale=args.RANDOM_WEIGHTS).to(DEVICE)
        else:
            weight_scaler = 1

        loss = weights * train_loss * weight_scaler

        if args.DIMREG_COEFF > 0:
          dimreg = regularize_embedding_dim(model, data.X[train_mask], data.y[train_mask], data.output_map, args)
          loss += args.DIMREG_COEFF * dimreg

        loss.mean().backward()
        optimizer.step()
        if epoch % args.LOG_FREQ == 0:
            with torch.no_grad():
                model.eval()
                val_loss = metric_by_task(
                    out[test_mask],
                    data.y[test_mask],
                    data.output_map,
                    args,
                    qt=data.regression_transformer,
                )

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

        if epoch % args.CKPT_FREQ == 0:
                torch.save(model.state_dict(), os.path.join(basedir, f"model_{epoch}.pt"))
    torch.save(model, os.path.join(basedir, "model_FULL.pt"))

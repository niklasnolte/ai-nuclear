import torch


class Logger:
    def __init__(self, args, model, basedir):
        self.args = args
        self.epoch = 0
        self.model = model
        self.basedir = basedir
        if args.WANDB:
            import wandb

            wandb.config.update(
                {"n_params": sum(p.numel() for p in model.parameters())}
            )

    def log(self, metrics=None, model=None, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        if epoch % self.args.LOG_FREQ == 0:
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

        if epoch == self.args.EPOCHS - 1:
            torch.save(self.model, os.path.join(self.basedir, "model_FULL.pt"))
            torch.save(best_model, os.path.join(self.basedir, "model_FULL_best.pt"))
        elif epoch % self.args.CKPT_FREQ == 0:
            torch.save(
                self.model.state_dict(), os.path.join(self.basedir, f"model_{epoch}.pt")
            )



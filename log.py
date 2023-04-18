import torch
import os
import wandb


class Logger:
    def __init__(self, args, model, basedir):
        self.args = args
        self.epoch = 0
        self.model = model
        self.basedir = basedir
        if args.WANDB:
            n_params = sum(p.numel() for p in model.parameters())
            wandb.config.update({"n_params": n_params})
        self.best_loss = float("inf")

    def log(self, metrics=None, epoch=None):
        epoch = epoch if epoch is not None else self.epoch
        if epoch % self.args.LOG_FREQ == 0:
            val_loss = metrics["val_loss_all"]
            # keep track of the best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model = self.model.state_dict().copy()
                if self.args.WANDB:
                    wandb.run.summary["best_epoch"] = epoch
                    for i, (target, value) in enumerate(metrics.items()):
                        if "val" in target:
                            wandb.run.summary[f"best_val_{target}"] = value
                torch.save(
                    self.best_model, os.path.join(self.basedir, "model_best.pt")
                )
            if self.args.WANDB:
                wandb.log(metrics, step=epoch)
            else:
                train_items = [
                    f"{' '.join(k.split('_')[1:]):<20} | {v:<8.2e}"
                    for k, v in metrics.items()
                    if "train" in k
                ]
                val_items = [f"{v:<8.2e}" for k, v in metrics.items() if "val" in k]
                items = [" | ".join([x, y]) for x, y in zip(train_items, val_items)]
                msg = f"Epoch {epoch:<14} | {'Train':^8} | {'Val':^8}\n"
                msg += "\n".join(sorted(items, key=lambda x: x.split(" ")[0]))
                print(msg)
        if epoch == self.args.EPOCHS - 1:
            torch.save(self.model, os.path.join(self.basedir, "model_FULL.pt"))
        elif epoch % self.args.CKPT_FREQ == 0:
            torch.save(
                self.model.state_dict(), os.path.join(self.basedir, f"model_{epoch}.pt")
            )
        self.epoch += 1

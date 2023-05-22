import torch
import os
import wandb
import yaml


class Logger:
    def __init__(self, args, models):
        self.args = args
        # save args
        self.epoch = 0
        self.models = models
        # FIXME: this is a hack to avoid logging outside train.py
        if hasattr(args, "basedir"):
            self.basedir = args.basedir
            with open(os.path.join(args.basedir, "args.yaml"), "w") as f:
                yaml.dump(vars(args), f)
            if args.WANDB:
                n_params = sum(p.numel() for p in models[0].parameters())
                wandb.config.update({"n_params": n_params})
        else:
            self.basedir = None
        self.best_loss = float("inf")

    def log(self, metrics=None, epoch=None):
        if self.basedir is None:
            return -1
        epoch = epoch if epoch is not None else self.epoch
        if epoch % self.args.LOG_FREQ == 0:
            val_loss = metrics["val_loss_all"]
            # keep track of the best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_models = [m.state_dict().copy() for m in self.models]
                if self.args.WANDB:
                    wandb.run.summary["best_epoch"] = epoch
                    for i, (target, value) in enumerate(metrics.items()):
                        if "val" in target:
                            wandb.run.summary[f"best_{target}"] = value
                [
                    torch.save(
                        self.best_models[fold], os.path.join(self.basedir, f"model_best.pt.{fold}")
                    )
                    for fold in self.args.WHICH_FOLDS
                ]
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
        if epoch == self.args.EPOCHS - 1 or epoch == 0:
            state_dict_path = os.path.join(self.basedir, "model_FULL.pt.{fold}")
            [
                torch.save(self.models[fold].state_dict(), state_dict_path.format(fold=fold))
                for fold in self.args.WHICH_FOLDS
            ]
            print("Saved model to:")
            print(os.path.join(self.basedir))
        # implement logarithmic checkpoint frequency
        if self.args.CKPT_FREQ > 0:
            # check if epoch power of two
            if epoch & (epoch - 1) == 0:
                [
                    torch.save(
                        self.models[fold].state_dict(),
                        os.path.join(self.basedir, f"model_{epoch}.pt.{fold}"),
                    )
                    for fold in self.args.WHICH_FOLDS
                ]

        # at the end, make a file "done.txt"
        if epoch == self.args.EPOCHS - 1:
            with open(os.path.join(self.basedir, "done.txt"), "w") as f:
                f.write("finalized successfully")

        self.epoch += 1

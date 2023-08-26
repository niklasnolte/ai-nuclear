import torch
import os
import wandb
import yaml


class Logger:
    def __init__(self, args, models):
        self.args = args
        # save args
        self.models = models
        # TODO: this is a hack to avoid logging outside train.py FIXME
        if hasattr(args, "basedir"):
            self.basedir = args.basedir
            with open(os.path.join(args.basedir, "args.yaml"), "w") as f:
                yaml.dump(vars(args), f)
            if args.WANDB:
                self.wandb = True
                n_params = sum(p.numel() for p in models[0].parameters())
                wandb.init(
                    project="NuCLR",
                    entity="iaifi",
                    name=args.name,
                    notes="new and great runs",
                    tags=["Aug23"],
                    group="ICML23",
                    config=dict(vars(args)),
                )
                wandb.config.update({"n_params": n_params})
                wandb.save("train.py")
                wandb.save("config.py")
                wandb.save("config_utils.py")
                wandb.save("loss.py")
                wandb.save("model.py")
                wandb.save("data.py")
                wandb.save("train_full.py")
                wandb.save("log.py")
                wandb.save("run_config.py")
            else:
                self.wandb = False
        else:

            self.basedir = None
        self.best_loss = float("inf")
        self._defined_metrics = None

    def define_metrics_wandb(self, metrics):
        if not self.wandb or self._defined_metrics is not None:
            return
        self._defined_metrics = metrics
        for metric in metrics:
            wandb.define_metric(metric, summary="min")

    def log(self, metrics, epoch, save_model=False):
        if self.basedir is None:
            return -1
        self.define_metrics_wandb(metrics)
        val_loss = metrics["val_loss_all"]
        # keep track of the best model
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_models = [m.state_dict().copy() for m in self.models]
            [
                torch.save(
                    self.best_models[fold],
                    os.path.join(self.basedir, f"model_best.pt.{fold}"),
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
            val_items = [
                f"{v:<8.2e}" for k, v in metrics.items() if "val" in k
            ]
            items = [
                " | ".join([x, y]) for x, y in zip(train_items, val_items)
            ]
            msg = f"Epoch {epoch:<14} | {'Train':^8} | {'Val':^8}\n"
            msg += "\n".join(sorted(items, key=lambda x: x.split(" ")[0]))
            print(msg)

        if save_model:
            epoch_str = (
                str(epoch) if epoch != self.args.EPOCHS - 1 else "final"
            )
            [
                torch.save(
                    self.models[fold].state_dict(),
                    os.path.join(self.basedir, f"model_{epoch_str}.pt.{fold}"),
                )
                for fold in self.args.WHICH_FOLDS
            ]

        # at the end, make a file "done.txt"
        if epoch == self.args.EPOCHS - 1:
            print("done", os.path.join(self.basedir), sep="\n")
            with open(os.path.join(self.basedir, "done.txt"), "w") as f:
                f.write("finalized successfully")

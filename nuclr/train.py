import tqdm
import torch
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LinearLR,
)
from torch.nn import CrossEntropyLoss, MSELoss
from nuclr.data import prepare_nuclear_data
from nuclr.model import get_model_and_optim
from nuclr.loss import rmse, mse, accuracy
from nuclr.log import Logger
from argparse import Namespace
from functools import cached_property
from .config import NUCLR
import os, yaml
import typing as T


class Trainer:
    @classmethod
    def from_path(cls, path:str, which_folds: T.Optional[list]=None):
        args_path = os.path.join(path, "args.yaml")
        with open(args_path, "r") as f:
            args = yaml.load(f, Loader=yaml.Loader)
        args = Namespace(**args)
        args.basedir = ""
        args.WANDB = 0
        args.CKPT = ""
        if which_folds is None:
          which_folds = args.WHICH_FOLDS
        else:
          assert isinstance(which_folds, list)

        model_paths = {}
        for fold in which_folds:
            model_path = path.replace(f"whichfolds_{args.WHICH_FOLDS[0]}", f"whichfolds_{fold}")
            model_path = os.path.join(model_path, f"model_best.pt.{fold}")
            model_paths[fold] = model_path

        args.WHICH_FOLDS = which_folds

        # create trainer
        trainer = cls(args)
        for fold in model_paths:
            trainer.models[fold].load_state_dict(
                torch.load(model_paths[fold])
            )
        trainer.log = False
        return trainer

    def __init__(self, args: Namespace):
        self.log = True
        self.args = args
        # prepare data
        self.data = prepare_nuclear_data(args)
        # prepare model
        ms_and_os = [
            get_model_and_optim(self.data, args) for idx in range(args.N_FOLDS)
        ]
        self.models = [m for m, _ in ms_and_os]
        self.optimizers = [o for _, o in ms_and_os]
        self.loaders = [
            DataLoader(
                TensorDataset(
                    self.data.X[self.data.train_masks[fold]],
                    self.data.y[self.data.train_masks[fold]],
                ),
                shuffle=True,
                batch_size=args.BATCH_SIZE,
            )
            for fold in range(args.N_FOLDS)
        ]
        self.schedulers = [
            self._get_scheduler(fold, args) for fold in range(args.N_FOLDS)
        ]
        # if hasattr(args, "CKPT") and args.CKPT:
        #     self.model.load_state_dict(torch.load(args.CKPT))
        # prepare loss
        self.loss_fn = {
            "regression": MSELoss(reduction="sum"),
            "classification": CrossEntropyLoss(reduction="sum"),
        }
        self.metric_fn = {
            "regression": mse,
            "classification": accuracy,
        }
        # prepare logger
        self.logger = Logger(args, self.models)

        # misc
        self.num_tasks = len(self.data.output_map)

    def train(self):
        bar = (tqdm.trange if not self.args.WANDB else range)(self.args.EPOCHS)
        for epoch in bar:
            self.epoch = epoch
            for fold in self.args.WHICH_FOLDS:
                for x, y in self.loaders[fold]:
                    self.train_step(x, y, fold)
            if (
                epoch % self.args.LOG_FREQ == 0
                or epoch == self.args.EPOCHS - 1
                or epoch & (epoch - 1) == 0
            ):
                self.val_step()

    def train_step(self, X, y, fold):
        self.models[fold].train()
        self.optimizers[fold].zero_grad()
        out = self.models[fold](X)
        task = X[:, len(self.data.vocab_size) - 1]
        losses, num_samples = self.loss_by_task(task, out, y)
        loss = losses.sum() / num_samples.sum()  # TODO weights?
        # TODO add grad clipping
        loss.backward()
        self.optimizers[fold].step()
        self.schedulers[fold].step()
        return out, losses, num_samples

    def val_step(self):
        # This serves as the logging step as well
        X, y = self.data.X, self.data.y
        task = self.all_tasks
        with torch.no_grad():
            metrics_dicts = []
            for fold in self.args.WHICH_FOLDS:
                model = self.models[fold]
                model.eval()
                out = model(X)
                out_ = self._unscale_output(out.clone())  # reg_targets are rescaled
                y_ = self.unscaled_y
                metrics_dict = {}
                masks = {
                    "train": self.data.train_masks[fold],
                    "val": self.data.val_masks[fold],
                }
                for name, mask in masks.items():
                    losses, num_samples = self.loss_by_task(
                        task[mask], out[mask], y[mask]
                    )
                    metrics, _ = self.metrics_by_task(task[mask], out_[mask], y_[mask])
                    m = self.construct_metrics(losses, metrics, num_samples, name)
                    metrics_dict.update(m)
                metrics_dicts.append(metrics_dict)

        # average over folds
        metrics_dict = {}
        for k in metrics_dicts[0].keys():
            metrics_k = [m[k] for m in metrics_dicts if not math.isnan(m[k])]
            metrics_dict[k] = sum(metrics_k) / len(metrics_k)
        if self.log:
            # check ckpt_freq
            save_model = (
                self.epoch % self.args.CKPT_FREQ == 0
                or self.epoch == self.args.EPOCHS - 1
            )
            self.logger.log(metrics_dict, self.epoch, save_model=save_model)
        return metrics_dict

    def construct_metrics(self, losses, metrics, num_samples, prefix):
        """Constructs a dictionary of metrics from the array of metrics and number of samples for each task"""
        m_dict = {}
        for i, task_name in enumerate(self.data.output_map):
            metric_name = "_".join([prefix, "loss", task_name])
            m_dict[metric_name] = (losses[i] / num_samples[i]).item()
            metric_name = "_".join([prefix, "metric", task_name])
            m_dict[metric_name] = metrics[i].item()
        metric_name = "_".join([prefix, "loss", "all"])
        m_dict[metric_name] = (losses.sum() / num_samples.sum()).item()
        return m_dict

    def metrics_by_task(self, task, pred, target):
        return self._agg_by_task(task, pred, target, self.metric_fn)

    def loss_by_task(self, task, pred, target):
        return self._agg_by_task(task, pred, target, self.loss_fn)

    def _agg_by_task(self, task, pred, target, fn_dict):
        """Get the loss/metrics for each task. By default this is the sum of the losses for each output.
        This function uses the output_map to determine the number of outputs for each task.
        It then iterates over tasks, finds the samples for that task, and calculates the aggeragte quantity.

        Args:
            task (Tensor): Array of task indices. [N, 1]
            pred (_type_): Array of predictions. [N, sum(output_map.values())]
            target (_type_): Array of targets. [N, 1]
            fn_dict (dict): Dictionary of losses (or metrics) for each task.

        Returns:
            Tuple[Dict]: Array of losses or metrics for each task and array of number of samples for each task.
        """
        # out_idx is the index of the first output for the current task
        losses, num_samples = torch.zeros(self.num_tasks), torch.zeros(self.num_tasks)
        taski, out_idx = 0, 0

        def fill_losses(task_names, loss_fn):
            nonlocal taski, out_idx
            for task_name in task_names:
                task_out_size = self.data.output_map[task_name]
                # TODO deal with nans before this point?
                task_mask = (task == taski) & (~target.isnan().view(-1))
                num_samples[taski] = task_mask.sum()
                losses[taski] = loss_fn(
                    pred[task_mask, out_idx : out_idx + task_out_size],
                    target[task_mask],
                )
                out_idx += task_out_size  # go to begining of next task
                taski += 1  # next task

        fill_losses(self.args.TARGETS_CLASSIFICATION, fn_dict["classification"])
        fill_losses(self.args.TARGETS_REGRESSION, fn_dict["regression"])
        return losses, num_samples

    def _unscale_output(self, out):
        """unscales each element y using the inverse transform of the regression_transformer of the task"""
        # out has shape [N * num_tasks, out_shape] and is ordered by sample
        # i.e. all tasks for sample 0, then all tasks for sample 1, etc.
        # [N * num_tasks, num_reg_tasks]
        rescaled = out[:, -len(self.args.TARGETS_REGRESSION) :].clone()
        rescaled = self._inverse_transform(rescaled, extend=True)
        out[:, -len(self.args.TARGETS_REGRESSION) :] = rescaled
        return out

    def _inverse_transform(self, out, extend=False):
        # out has shape [-1, num_reg_tasks]
        out = self.data.regression_transformer.inverse_transform(out.cpu())
        # first dim is probably N * num_tasks
        # TODO remove this
        if extend:
            A = self.data.X[:, :2].sum(1).view(-1, 1)
        else:
            A = self.data.X[:: self.num_tasks, :2].sum(1).view(-1, 1)
        out = torch.from_numpy(out).to(self.args.DEV)
        # TODO undo all scaling bullshit
        out[:, self.data.scaled_idxs] = out[:, self.data.scaled_idxs] * A
        return out

    @cached_property
    def unscaled_y(self):
        N = len(self.data.y) // self.num_tasks
        # flatten tasks into dim=1
        y_ = self._inverse_transform(self.data.y.view(N, -1).clone())
        y_ = y_.view(-1, 1)
        return y_

    @cached_property
    def all_tasks(self):
        task_idx = len(self.data.vocab_size) - 1
        return self.data.X[:, task_idx]  # [N * num_tasks, 1]

    def _get_scheduler(self, fold, args):
        if args.SCHED == "none":

            class NoScheduler:
                def step(self):
                    pass

            return NoScheduler()
        max_steps = args.EPOCHS * len(self.loaders[fold])
        if args.SCHED == "cosine":
            # TODO use warmup and make indepedent of epochs
            return CosineAnnealingLR(self.optimizers[fold], max_steps, args.FINAL_LR)
        elif args.SCHED == "onecycle":
            return OneCycleLR(self.optimizers[fold], 1e-3, max_steps)
        elif args.SCHED == "linear":
            return LinearLR(self.optimizers[fold], 1.0, 1e-2, max_steps)
        else:
            raise ValueError(f"Unknown scheduler {args.SCHED}")

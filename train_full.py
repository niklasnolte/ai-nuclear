import tqdm
import math
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LinearLR,
)
from torch.nn import CrossEntropyLoss, MSELoss
from data import prepare_nuclear_data
from model import get_model_and_optim
from loss import rmse, accuracy
from config import Task
from log import Logger
from argparse import Namespace
from functools import cached_property
from torch.utils.data import DataLoader, TensorDataset
import os


class Trainer:
    def __init__(self, problem: Task, args: Namespace):
        self.problem = problem
        self.args = args
        # prepare data
        self.data = prepare_nuclear_data(args)
        self.loader = DataLoader(
            TensorDataset(
                self.data.X[self.data.train_mask], self.data.y[self.data.train_mask]
            ),
            shuffle=True,
            batch_size=args.BATCH_SIZE,
        )
        # prepare model
        self.model, self.optimizer = get_model_and_optim(self.data, args)
        self.scheduler = self._get_scheduler(args)
        # prepare loss
        self.loss_fn = {
            "regression": MSELoss(reduction="sum"),
            "classification": CrossEntropyLoss(reduction="sum"),
        }
        self.metric_fn = {
            "regression": rmse,
            "classification": accuracy,
        }
        # prepare logger
        self.logger = Logger(args, self.model)

        # misc
        self.num_tasks = len(self.data.output_map)

    def train(self):
        bar = (tqdm.trange if not self.args.WANDB else range)(self.args.EPOCHS)
        for epoch in bar:
            for x, y in self.loader:
                self.train_step(x, y)
            self.val_step(log=True)

    def train_step(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(X)
        task = X[:, len(self.data.vocab_size) - 1]
        losses, num_samples = self.loss_by_task(task, out, y)
        loss = losses.sum() / num_samples.sum()  # TODO weights?
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return out, losses, num_samples

    def val_step(self, log=False):
        # This serves as the logging step as well
        X, y = self.data.X, self.data.y
        task = self.all_tasks
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
            out_ = self._unscale_output(out.clone())  # reg_targets are rescaled
            y_ = self.unscaled_y
            metrics_dict = {}
            masks = {"train": self.data.train_mask, "val": self.data.val_mask}
            for name, mask in masks.items():
                losses, num_samples = self.loss_by_task(task[mask], out[mask], y[mask])
                metrics, _ = self.metrics_by_task(task[mask], out_[mask], y_[mask])
                m = self.construct_metrics(losses, metrics, num_samples, name)
                metrics_dict.update(m)

        if log:
            self.logger.log(metrics_dict)
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
        out[:, :2] = out[:, :2] * (A)
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

    def _get_scheduler(self, args):
        if args.SCHED == "none":

            class NoScheduler:
                def step(self):
                    pass

            return NoScheduler()
        max_steps = args.EPOCHS * len(self.loader)
        if args.SCHED == "cosine":
            return CosineAnnealingLR(self.optimizer, max_steps, 1e-5)
        elif args.SCHED == "onecycle":
            return OneCycleLR(self.optimizer, 1e-3, max_steps)
        elif args.SCHED == "linear":
            return LinearLR(self.optimizer, 1.0, 1e-2, max_steps)
        else:
            raise ValueError(f"Unknown scheduler {args.SCHED}")


train = lambda *args: Trainer(*args).train()

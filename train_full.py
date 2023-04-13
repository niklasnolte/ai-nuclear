import os
import tqdm
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from data import prepare_nuclear_data, prepare_modular_data
from model import get_model_and_optim
from loss import LossWithNan, rmse, accuracy
from config import Task
from log import Logger
from argparse import Namespace
from functools import cached_property


class Trainer:
    def __init__(self, problem: Task, args: Namespace, basedir):
        self.problem = problem
        self.args = args
        self.basedir = basedir
        # prepare data
        self.data = (
            _ := prepare_modular_data
            if problem == Task.MODULAR
            else prepare_nuclear_data
        )(args)
        self.loader = DataLoader(
            TensorDataset(
                self.data.X[self.data.train_mask], self.data.y[self.data.train_mask]
            ),
            shuffle=True,
            batch_size=args.BATCH_SIZE,
        )
        # prepare model
        self.model, self.optimizer = get_model_and_optim(self.data, args)
        self.scheduler = CosineAnnealingLR(
            self.optimizer, args.EPOCHS * len(self.loader)
        )
        # prepare loss
        self.loss_fn = {
            "regression": LossWithNan(MSELoss),
            "classification": LossWithNan(CrossEntropyLoss),
        }
        self.metric_fn = {
            "regression": rmse,
            "classification": accuracy,
        }
        # prepare logger
        self.logger = Logger(args, self.model, basedir)

        # misc
        self.num_tasks = len(self.data.output_map)

    def train(self):
        bar = (f := tqdm.trange if not self.args.WANDB else range)(self.args.EPOCHS)
        for _ in bar:
            for x, y in self.loader:
                self.train_step(x, y)
            self.val_step()

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

    def val_step(self):
        # This serves as the logging step as well
        X, y = self.data.X, self.data.y
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
            out_ = self._unscale_output(out.clone())  # reg_targets are rescaled
            y_ = self.unscaled_y
            task = X[:, len(self.data.vocab_size) - 1]
            metrics_dict = {}
            masks = {"train": self.data.train_mask, "val": self.data.val_mask}
            for name, mask in masks.items():
                losses, num_samples = self.loss_by_task(task[mask], out[mask], y[mask])
                metrics, _ = self.metrics_by_task(task[mask], out_[mask], y_[mask])
                m = self.construct_metrics(losses, metrics, num_samples, name)
                metrics_dict.update(m)
        self.logger.log(metrics_dict)

    def construct_metrics(self, losses, metrics, num_samples, prefix):
        """Constructs a dictionary of metrics from the array of metrics and number of samples for each task"""
        m_dict = {}
        for i, task_name in enumerate(self.data.output_map):
            metric_name = "_".join([prefix, "loss", task_name])
            m_dict[metric_name] = (losses[i] / num_samples[i]).item()
            metric_name = "_".join([prefix, "metric", task_name])
            m_dict[metric_name] = (metrics[i] / num_samples[i]).item()
        metric_name = "_".join([prefix, "loss", "combined"])
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
                # TODO deal with nans before this point
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
        rescaled = out[:, -len(self.args.TARGETS_REGRESSION) :]
        # [N * num_tasks, num_reg_tasks]
        rescaled = self._inverse_transform(rescaled)
        out[:, -len(self.args.TARGETS_REGRESSION) :] = rescaled
        return out

    def _inverse_transform(self, out):
        # out has shape [-1, num_reg_tasks]
        out = self.data.regression_transformer.inverse_transform(out)
        # first dim is probably N * num_tasks
        return torch.from_numpy(out)

    @cached_property
    def unscaled_y(self):
        N = len(self.data.y) // len(self.data.output_map)
        return self._inverse_transform(self.data.y.view(N, -1)).view(-1, 1)

    @cached_property
    def all_tasks(self):
        task_idx = len(self.data.vocab_size) - 1
        return self.data.X[:, task_idx]  # [N * num_tasks, 1]


train = lambda *args: Trainer(*args).train()

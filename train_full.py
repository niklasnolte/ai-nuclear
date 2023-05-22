import tqdm
import math
import torch
import copy
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


class TrainLoader:
    def __init__(self, X, y, fold_idxs, train_include_masks, batch_size=1024):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.randperm = torch.randperm(len(X), device=X.device)
        self.fold_idxs = fold_idxs[self.randperm]
        self.train_include_masks = train_include_masks[:,self.randperm]
        self.with_fold(0)

    def __iter__(self):
        self.current_idx = 0
        return self

    def with_fold(self, fold):
        self.current_fold = fold
        self.randperm_fold = self.randperm[(self.fold_idxs != fold) & self.train_include_masks[fold]]
        return self

    def __next__(self):
        if self.current_idx >= len(self.randperm_fold):
            raise StopIteration
        idx = self.randperm_fold[self.current_idx : self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        return self.X[idx], self.y[idx]

    def __len__(self):
        return math.ceil(len(self.randperm_fold) / self.batch_size)


class Trainer:
    def __init__(self, problem: Task, args: Namespace):
        self.problem = problem
        self.args = args
        # prepare data
        self.data = prepare_nuclear_data(self.args)
        # prepare loss
        self.loss_fn = {
            "regression": MSELoss(reduction="sum"),
            "classification": CrossEntropyLoss(reduction="sum"),
        }
        self.metric_fn = {
            "regression": rmse,
            "classification": accuracy,
        }

        self.loader = TrainLoader(
            self.data.X,
            self.data.y,
            self.data.fold_idxs,
            self.data.train_include_masks,
            batch_size=self.args.BATCH_SIZE,
        )
        # prepare model
        models_and_optims = [
            get_model_and_optim(self.data, self.args) for _ in range(self.args.N_FOLDS)
        ]
        self.models = [m for m, _ in models_and_optims]

        if not hasattr(args, "WHICH_FOLDS"):
            self.args.WHICH_FOLDS = list(range(self.args.N_FOLDS))

        if hasattr(args, "CKPT") and args.CKPT:
            [self.models[i].load_state_dict(torch.load(args.CKPT+f".{i}")) for i in self.args.WHICH_FOLDS]
        self.optimizers = [o for _, o in models_and_optims]
        self.schedulers = [self._get_scheduler(self.args, o) for o in self.optimizers]
        # prepare logger
        self.logger = Logger(self.args, self.models)

        # misc
        self.num_tasks = len(self.data.output_map)

    def train(self):
        bar = (tqdm.trange if not self.args.WANDB else range)(self.args.EPOCHS)
        for epoch in bar:
            for fold in self.args.WHICH_FOLDS:
                for x, y in self.loader.with_fold(fold):
                    self.train_step(x, y, fold)
            if (
                epoch % self.args.LOG_FREQ == 0
                or epoch == self.args.EPOCHS - 1
                or epoch & (epoch - 1) == 0
            ):
                self.val_step(epoch, log=True)

    def train_step(self, X, y, fold):
        self.models[fold].train()
        self.optimizers[fold].zero_grad()
        out = self.models[fold](X)
        task = X[:, len(self.data.vocab_size) - 1]
        losses, num_samples = self.loss_by_task(task, out, y)
        loss = losses.sum() / num_samples.sum()  # TODO weights?
        # gradient clipping
        for param in self.models[fold].parameters():
            if param.grad is not None:
                param.grad = torch.clamp(param.grad, -0.1, 0.1)
        loss.backward()
        self.optimizers[fold].step()
        self.schedulers[fold].step()
        return out, losses, num_samples

    def val_step(self, epoch, log=False):
        # This serves as the logging step as well
        task = self.all_tasks
        metrics_dicts = []
        for fold in self.args.WHICH_FOLDS:
            model = self.models[fold]
            model.eval()
            with torch.no_grad():
                out = model(self.data.X)
                out_ = self._unscale_output(out.clone())  # reg_targets are rescaled
                y_ = self.unscaled_y
                metrics_dict = {}
                train_mask = self.data.fold_idxs != fold
                val_mask = (self.data.fold_idxs == fold) & self.data.test_include_mask
                masks = {"train": train_mask, "val": val_mask}
                for name, mask in masks.items():
                    losses, num_samples = self.loss_by_task(
                        task[mask], out[mask], self.data.y[mask]
                    )
                    metrics, _ = self.metrics_by_task(task[mask], out_[mask], y_[mask])
                    m = self.construct_metrics(losses, metrics, num_samples, name)
                    metrics_dict.update(m)
                metrics_dicts.append(metrics_dict)

        # take the mean of each metric across folds
        metrics_dict = {}
        for k in metrics_dicts[0]:
            coll = [m[k] for m in metrics_dicts]
            mean = sum(coll) / len(coll)
            var = sum([(m[k] - mean) ** 2 for m in metrics_dicts]) / (len(coll) - 1)
            metrics_dict[f"{k}_mean"] = mean
            metrics_dict[f"{k}_std"] = math.sqrt(var)

        if log:
            self.logger.log(metrics_dict, epoch)
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
        # self.data.output_map gives the size of each output
        # we want to scale binding, so the binding idx is the sum of all sizes before it
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

    def _get_scheduler(self, args, optimizer):
        if args.SCHED == "none":

            class NoScheduler:
                def step(self):
                    pass

            return NoScheduler()
        max_steps = args.EPOCHS * len(self.loader)
        if args.SCHED == "cosine":
            return CosineAnnealingLR(optimizer, max_steps, args.FINAL_LR)
        elif args.SCHED == "onecycle":
            return OneCycleLR(optimizer, 1e-3, max_steps)
        elif args.SCHED == "linear":
            return LinearLR(optimizer, 1.0, 1e-2, max_steps)
        else:
            raise ValueError(f"Unknown scheduler {args.SCHED}")


train = lambda *args: Trainer(*args).train()

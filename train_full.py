import os
import tqdm
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from data import prepare_nuclear_data, prepare_modular_data
from model import get_model_and_optim
from loss import loss_by_task, metric_by_task, LossWithNan
from config import Task
from log import Logger
from argparse import Namespace
from functools import cached_property


class Trainer:
    def __init__(self, task: Task, args: Namespace, basedir):
        self.task = task
        self.args = args
        self.basedir = basedir
        # prepare data
        self.data = (
            prep := prepare_modular_data
            if task == Task.MODULAR
            else prepare_nuclear_data
        )(args)
        # prepare model
        self.model, self.optimizer = get_model_and_optim(self.data, args)
        self.scheduler = CosineAnnealingLR(self.optimizer, args.EPOCHS)
        self.loader = DataLoader(
            TensorDataset(
                self.data.X[self.data.train_mask], self.data.y[self.data.train_mask]
            ),
            batch_size=args.BATCH_SIZE,
        )
        # prepare loss
        self.loss_fn = {
            "regression": LossWithNan(MSELoss),
            "classification": LossWithNan(CrossEntropyLoss),
        }
        # prepare logger
        self.logger = Logger(args, self.model, basedir)

    def loss_by_task(self, task, pred, target):
        """Get the loss for each task. By default this is the sum of the losses for each output.
        This function uses the output_map to determine the number of outputs for each task.
        It then iterates over tasks, finds the samples for that task, and calculates the loss.

        Args:
            task (Tensor): Array of task indices. [N, 1]
            pred (_type_): Array of predictions. [N, sum(output_map.values())]
            target (_type_): Array of targets. [N, 1]

        Returns:
            Tuple[Tensor]: Array of losses for each task and array of number of samples for each task.
            Note some tasks may have no samples. In this case the loss is NaN.
        """        
        out_idx, taski = 0, 0 # out_idx is the index of the first output for the current task
        losses = torch.zeros((self.data.output_map))
        num_samples = torch.zeros((self.data.output_map))
        def fill_losses(targets, loss_fn):
            nonlocal out_idx, taski
            for task_name in targets:
                task_out_size = self.data.output_map[task_name]
                task_mask = task == taski
                num_samples[taski] = task_mask.sum()
                losses[taski] = self.loss_fn[loss_fn](
                    pred[task_mask, out_idx : out_idx + task_out_size], target[task_mask]
                )
                out_idx += task_out_size # go to begining of next task
                taski += 1 # go to next task
        fill_losses(self.args.TARGETS_CLASSIFICATION, "classification")
        fill_losses(self.args.TARGETS_REGRESSION, "regression")
        return losses, num_samples
    
    def apply(self, X, y):
        out = self.model(X)
        task = X[:, len(self.data.vocab_size) - 1]
        losses, num_samples = self.loss_by_task(task, out, y)
        return out, losses, num_samples
    
    def train_step(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()
        out, losses, num_samples = self.apply(X, y)
        loss = (losses * num_samples).sum() / num_samples.sum()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return out, losses, num_samples

    def val_step(self, X=None, y=None): 
        # This serves as a logging step as well
        X, y = X or self.data.X, y or self.data.y
        self.model.eval()
        with torch.no_grad():
            out = self.model(X)
            out = self._unscale_output(out)

    def _unscale_output(self, out): 
        """unscales each element y using the inverse transform of the regression_transformer of the task"""
        # out has shape [N * num_tasks, out_shape] and is ordered by sample 
        out = out[:, - len(self.args.TARGETS_REGRESSION):] # [N * num_tasks, num_reg_tasks]
        # remember regression tasks come first 
        out = out[:, self.all_tasks - len(self.args.TARGETS_CLASSIFICATION)] # [N * num_tasks, 1]
        return self._inverse_transform(out)

    def _inverse_transform(self, out):
        N = len(self.data.X) // len(self.data.output_map)
        out = out.reshape(N, -1) # [N, num_reg_tasks]
        out = self.data.regression_transformer.inverse_transform(out) # [N, num_reg_tasks]
        return torch.from_numpy(out).view(-1, 1)
    
    @cached_property
    def unscaled_y(self):
        return self._inverse_transform(self.data.y)
    
    @cached_property
    def all_tasks(self):
        task_idx = len(self.data.vocab_size) - 1
        return self.data.X[:, task_idx]# [N * num_tasks, 1] 

    def train(self, task: Task, args: argparse.Namespace, basedir: str):
        bar = (arange := tqdm.trange if not args.WANDB else range)(args.EPOCHS)
        for _ in bar:
            for x, y in self.loader:
                self.train_step(x, y)
            self.val_step()

train = lambda *args: (trainer := Trainer(*args)).train

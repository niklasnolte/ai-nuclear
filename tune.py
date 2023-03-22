import torch
import optuna
import torch
import os
from config import Task
from config_utils import parse_arguments_and_get_name, _parse_arguments, _get_qualified_name, _args_postprocessing
from train_full import train

# load TASK from env
TASK = Task["DEBUG"]

def bookkeeping(args, name):
    args = _args_postprocessing(args)
    torch.manual_seed(args.SEED)
    # paths and names
    basedir = os.path.join(args.ROOT, name)
    os.makedirs(basedir, exist_ok=True)
    print(f"training run for {name}")

    # bookkeeping
    if args.WANDB:
        import wandb
        wandb.init(project=f"ai-nuclear", entity="iaifi", name=name, notes="tuning", tags=["tuning"], config=vars(args))
        wandb.save("train.py")
        wandb.save("config.py")
        wandb.save("config_utils.py")
        wandb.save("loss.py")
        wandb.save("model.py")
        wandb.save("data.py")
        wandb.save("train_full.py")
        wandb.save("tune.py")

    # remove old models
    # FIXME should we really do that?
    for f in os.listdir(basedir):
        if f.endswith(".pt"):
            os.remove(os.path.join(basedir, f))
    return args, basedir


# 1. Define an objective function to be maximized.
def objective(trial):
    args = _parse_arguments(TASK)
    args.EPOCHS = 1000
    args.LR = trial.suggest_float('LR', 1e-5, 1e-1, log=True)
    args.WD = trial.suggest_float('WD', 1e-5, 1e-1, log=True)
    args.HIDDEN_DIM = trial.suggest_int('HIDDEN_DIM', 32, 1024)
    args.RANDOM_WEIGHTS = trial.suggest_float('RANDOM_WEIGHTS', 0.0, 2.0)
    name = _get_qualified_name(TASK, args)
    name = "TUNE_" + name
    args, basedir = bookkeeping(args, name)
    train_losses, train_metrics, val_losses, val_metrics = train(TASK, args, basedir)
    return torch.mean(val_losses).item()  

# 3. Create a study object and optimize the objective function.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)
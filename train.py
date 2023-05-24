import torch
import os
from config import Task
from config_utils import parse_arguments_and_get_name
from train_full import train

# load TASK from env
TASK = Task[os.environ.get("TASK")]

args, name = parse_arguments_and_get_name(TASK)
torch.manual_seed(args.SEED)

# paths and names
args.basedir = os.path.join(args.ROOT, name)
os.makedirs(args.basedir, exist_ok=True)
print(f"training run for {name}")

# bookkeeping
if args.WANDB:
    import wandb

    wandb.init(
        project=f"ai-nuclear",
        entity="iaifi",
        name=name,
        notes="",
        tags=["master_run_all_data"],
        group="cross-val",
        config=vars(args),
    )
    wandb.save("train.py")
    wandb.save("config.py")
    wandb.save("config_utils.py")
    wandb.save("loss.py")
    wandb.save("model.py")
    wandb.save("data.py")
    wandb.save("train_full.py")

# remove old models
# FIXME should we really do that?
# for f in os.listdir(basedir):
#     if f.endswith(".pt"):
#         os.remove(os.path.join(basedir, f))


if TASK == Task.FULL:
    train(TASK, args)

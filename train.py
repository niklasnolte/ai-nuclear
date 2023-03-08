import torch
import os
import wandb
from config import Task
from config_utils import parse_arguments_and_get_name
from train_full import train_FULL
from train_modular import train_MODULAR

# load TASK from env
TASK = Task[os.environ.get("TASK")]

args, name = parse_arguments_and_get_name(TASK)
torch.manual_seed(args.SEED)

# paths and names
basedir = os.path.join(args.ROOT, name)
os.makedirs(basedir, exist_ok=True)
print(f"training run for {name}")

# bookkeeping
if args.WANDB:
    wandb.init(project=f"ai-nuclear", entity="iaifi", name=name, notes="testing 1024 width", tags=["testing"], config=vars(args))
    wandb.save("train.py")
    wandb.save("config.py")
    wandb.save("config_utils.py")
    wandb.save("loss.py")
    wandb.save("model.py")
    wandb.save("data.py")
    if TASK == Task.FULL or TASK == Task.DEBUG:
      wandb.save("train_full.py")
    elif TASK == Task.MODULAR:
      wandb.save("train_modular.py")

# remove old models
# FIXME should we really do that?
for f in os.listdir(basedir):
    if f.endswith(".pt"):
        os.remove(os.path.join(basedir, f))


if TASK == Task.FULL or TASK == Task.DEBUG:
    train_FULL(args, basedir)
elif TASK == Task.MODULAR:
    train_MODULAR(args, basedir)

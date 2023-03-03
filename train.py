import torch
import os
import wandb
from config import Task
from config_utils import parse_arguments_and_get_name
from train_full import train_FULL

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
    wandb.save("train_full.py")
    wandb.save("config.py")
    wandb.save("config_utils.py")
    wandb.save("model.py")
    wandb.save("data.py")
    wandb.save("loss.py")

# remove old models
# FIXME should we really do that?
for f in os.listdir(basedir):
    if f.endswith(".pt"):
        os.remove(os.path.join(basedir, f))


if TASK == Task.FULL:
    train_FULL(args, basedir)
    pass
elif TASK == Task.BASELINE:
    # train_MNIST(args, device, basedir)
    pass

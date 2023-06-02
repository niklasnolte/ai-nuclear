import torch
import os
from config import Task
from config_utils import parse_arguments_and_get_name
from train_full import train

# load TASK from env
TASK = Task[os.environ["TASK"]]

args, name = parse_arguments_and_get_name(TASK)
args.name = name
torch.manual_seed(args.SEED)

# paths and names
args.basedir = os.path.join(args.ROOT, name)
os.makedirs(args.basedir, exist_ok=True)
print(f"training run for {name}")

# remove old models
# FIXME should we really do that?
# for f in os.listdir(basedir):
#     if f.endswith(".pt"):
#         os.remove(os.path.join(basedir, f))


if TASK == Task.FULL:
    train(TASK, args)
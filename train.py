import torch
import os
from nuclr.config import NUCLR
from config_utils import parse_arguments_and_get_name
from train_nuclr import Trainer


args, name = parse_arguments_and_get_name(NUCLR)
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


Trainer(args).train()

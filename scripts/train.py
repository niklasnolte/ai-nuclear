import torch
import os
from nuclr.config import NUCLR
from lib.config_utils import parse_arguments_and_get_name
from nuclr.train import Trainer


args, name = parse_arguments_and_get_name(NUCLR)
args.name = name
torch.manual_seed(args.SEED)

# paths and names
args.basedir = os.path.join(args.ROOT, name)
os.makedirs(args.basedir, exist_ok=True)
print(f"training run for {name}")


Trainer(args).train()

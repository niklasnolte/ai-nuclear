import os
import config
import numpy

def get_slurm_extra(gpu=True):
  if config.SLURM and gpu:
    return " ".join([
      "--gres=gpu:1",
      "--partition=submit-gpu",
      "--mem=5G",
    ])
  else:
    return ""

class Locations:
  FULL = os.path.join(config.ROOT, config.get_name(config.Task.FULL))

rule all:
  input:
    expand(Locations.FULL,
            **config.Task.FULL.value)

rule train_FULL:
  output:
    cps = directory(Locations.FULL) # "results/{wd}_{lr}/model.pt"
  resources:
    slurm_extra=get_slurm_extra()
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = config.train_cmd(config.Task.RANDOM, wildcards)
    shell(cmd)


# rule all:
  # [ "results/.1_.1/model.pt", ... ]

# rule train_FULL:
  # "results/{wd}_{lr}/model.pt"
  # "results/.1_.1/model.pt" -> wildcards = {wd: .1, lr: .1}

import os
import config_utils
import config
import numpy
import snakemake

def get_slurm_extra():
  if config.GPU:
    return " ".join([
      "--gres=gpu:1",
      "--partition=submit-gpu",
      "--mem=5G",
    ])
  else:
    return "--mem=5G"

class Locations:
  FULL = os.path.join(config.ROOT, config_utils.get_name(config.Task.FULL))

rule all:
  input:
    expand(Locations.FULL,
            **config_utils.serialize_elements_in_task(config.Task.FULL.value))

rule train_FULL:
  output:
    cps = directory(Locations.FULL) # "results/{wd}_{lr}/model.pt"
  resources:
    slurm_extra=get_slurm_extra()
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = config_utils.train_cmd(config.Task.RANDOM, wildcards)
    shell(cmd)


# rule all:
  # [ "results/.1_.1/model.pt", ... ]

# rule train_FULL:
  # "results/{wd}_{lr}/model.pt"
  # "results/.1_.1/model.pt" -> wildcards = {wd: .1, lr: .1}

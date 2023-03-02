import os
import config_utils
import config
import numpy
import snakemake

def get_slurm_extra():
  if config.SN_GPU:
    return " ".join([
      "--gres=gpu:1",
      "--partition=submit-gpu1080",
      "--mem=5G",
    ])
  else:
    return "--mem=5G"

class Locations:
  FULL = os.path.join(config.SN_ROOT, config_utils.get_name(config.Task.FULL))

rule all:
  input:
    expand(Locations.FULL,
            **config.Task.FULL.value)

rule train_FULL:
  output:
    cps = directory(Locations.FULL)
  resources:
    slurm_extra=get_slurm_extra()
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = config.train_cmd(config.Task.FULL, wildcards)
    shell(cmd)

import os
import config_utils
import config
import numpy
import snakemake

def get_slurm_extra():
  if config.SM_GPU:
    return " ".join([
      "--gres=gpu:1",
      "--partition=submit-gpu1080,submit-gpu",
      "--mem=5G",
    ])
  else:
    return "--mem=5G"

class Locations:
  FULL = os.path.join(config.SM_ROOT, config_utils.get_name(config.Task.FULL))
  FULL_model = os.path.join(FULL, "model_full.pt")

rule all:
  input:
    expand(Locations.FULL_model,
            **config.Task.FULL.value)

rule train_FULL:
  output:
    cps = directory(Locations.FULL),
    model = Locations.FULL_model
  resources:
    slurm_extra=get_slurm_extra()
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = config.train_cmd(config.Task.FULL, wildcards)
    shell(cmd)

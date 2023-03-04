import os
import config_utils
import config
import numpy
import snakemake

def get_partitions():
  # hostname
  host = os.environ["HOSTNAME"]
  if host.endswith("mit.edu"):
    return "submit-gpu1080,submit-gpu"
  elif host.endswith("harvard.edu"):
    return "iaifi_gpu"
  else:
    raise Exception("Auto-host Unknown host: " + host + ". Please set host manually")

def get_slurm_extra():
  if config.SM_GPU:
    return " ".join([
      "--gres=gpu:1",
      f"--partition={get_partitions()}",
      "--mem=1G",
      "--time=1:00:00"
    ])
  else:
    return "--mem=1G"

class Locations:
  FULL = os.path.join(config.SM_ROOT, config_utils.get_name(config.Task.FULL))
  FULL_model = os.path.join(FULL, f"model_FULL.pt")
  DEBUG = os.path.join(config.SM_ROOT, config_utils.get_name(config.Task.DEBUG))
  DEBUG_model = os.path.join(DEBUG, f"model_DEBUG.pt")

rule debug:
  input:
      expand(Locations.DEBUG_model,
            **config.Task.DEBUG.value)

rule all:
  input:
      expand(Locations.FULL_model,
              **config.Task.FULL.value)


rule train_FULL:
  output:
    cps = directory(Locations.FULL),
    model = Locations.FULL_model
  resources:
    slurm_extra=get_slurm_extra(),
    runtime="1h"
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = config.train_cmd(config.Task.FULL, wildcards)
    shell(cmd)

rule train_DEBUG:
  output:
    cps = directory(Locations.DEBUG),
    model = Locations.DEBUG_model
  resources:
    slurm_extra=get_slurm_extra(),
    runtime="5m"
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = config.train_cmd(config.Task.DEBUG, wildcards)
    shell(cmd)

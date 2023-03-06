import os
import config_utils
import config
import run_config
import numpy
import snakemake


class Locations:
  FULL = os.path.join(run_config.SM_ROOT, config_utils.get_name(config.Task.FULL))
  FULL_model = os.path.join(FULL, f"model_FULL.pt")
  DEBUG = os.path.join(run_config.SM_ROOT, config_utils.get_name(config.Task.DEBUG))
  DEBUG_model = os.path.join(DEBUG, f"model_FULL.pt")


rule all:
  input:
      expand(Locations.FULL_model,
              **config.Task.FULL.value)

rule debug:
  input:
      expand(Locations.DEBUG_model,
            **config.Task.DEBUG.value)

rule train_FULL:
  output:
    cps = directory(Locations.FULL),
    model = Locations.FULL_model
  resources:
    slurm_extra=run_config.get_slurm_extra_resources(),
    runtime="1h"
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = run_config.train_cmd(config.Task.FULL, wildcards)
    shell(cmd)

rule train_DEBUG:
  output:
    cps = directory(Locations.DEBUG),
    model = Locations.DEBUG_model
  resources:
    slurm_extra=run_config.get_slurm_extra_resources(),
    runtime="5m"
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = run_config.train_cmd(config.Task.DEBUG, wildcards)
    shell(cmd)

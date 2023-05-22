import os
import config_utils
import config
import run_config
import numpy
import snakemake


class Locations:
  FULL = os.path.join(run_config.SM_ROOT, config_utils.get_name(config.Task.FULL))
  FULL_final = os.path.join(FULL, f"done.txt")
  DEBUG = os.path.join(run_config.SM_ROOT, config_utils.get_name(config.Task.DEBUG))
  DEBUG_model = os.path.join(DEBUG, f"model_FULL.pt")
  MODULAR = os.path.join(run_config.SM_ROOT, config_utils.get_name(config.Task.MODULAR))
  MODULAR_model = os.path.join(MODULAR, f"model_FULL.pt")


rule all:
  input:
      expand(Locations.FULL_final,
              **config.Task.FULL.value)

rule modular:
  input:
      expand(Locations.MODULAR_model,
              **config.Task.MODULAR.value)

rule debug:
  input:
      expand(Locations.DEBUG_model,
            **config.Task.DEBUG.value)

rule train_FULL:
  output:
    cps = directory(Locations.FULL),
    model = Locations.FULL_final
  resources:
    slurm_extra=run_config.get_slurm_extra_resources(),
    runtime="1h"
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = run_config.train_cmd(config.Task.FULL, wildcards)
    shell(cmd)

rule train_modular:
  output:
    cps = directory(Locations.MODULAR),
    model = Locations.MODULAR_model
  resources:
    slurm_extra=run_config.get_slurm_extra_resources(),
    runtime="1h"
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = run_config.train_cmd(config.Task.MODULAR, wildcards)
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

import os
from nuclr import config_utils, config, run_config
import numpy
import snakemake


class Locations:
  NUCLR = os.path.join(run_config.SM_ROOT, config_utils.get_name(config.NUCLR))
  NUCLR_model = os.path.join(NUCLR, f"done.txt")


rule all:
  input:
      expand(Locations.NUCLR_model,
              **config.NUCLR)


rule train_NUCLR:
  output:
    cps = directory(Locations.NUCLR),
    model = Locations.NUCLR_model
  resources:
    slurm_extra=run_config.get_slurm_extra_resources(),
    runtime="1h"
  run:
    shell(f"mkdir -p {output.cps}")
    cmd = run_config.train_cmd(wildcards)
    shell(cmd)

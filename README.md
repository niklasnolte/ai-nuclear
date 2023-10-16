# Running
There are 3 ways to run this pipeline, depending on how many jobs you want to run.
## Single run
A single run (interactively) to test a single configuration.
All configuration lives in `config.py`.
To run, choose as task from the Task enum in `config.py`, for instance `NUCLR`. (At the moment of writing this `NUCLR` is the only option available).
Then you can run a default run with `python scripts/train.py`.

The configurable parameters are defined in the tasks dict (again in `config.py`).
Each Element in the dict has a key, which is the name of the configurable (for instance `LR`)
and a list of values. The first value is the default. The rest of the values are not
used for a single run, see the Snakemake section.
You can change any of those default parameters by providing `--key=new_value` to your run command,
e.g. `python scripts/train.py --LR=1e-3`.
By default, model checkpoints will be saved in `./results/<a_long_name_with_the_config>/model...pt`.
You can change the output directory by providing `--ROOT path` or by setting an environment variable, see below.

You can activate WANDB with `--WANDB`.

## SnakeMake local
First, install snakemake (`pip install snakemake`). Snakemake is a job scheduler it is set up
here to run all the different combinations of all the configuration parameter lists.
So if your NUCLR looked like
```python
    NUCLR = {
        "LR": [1e-3, 1e-4],
        "BATCH_SIZE": [32, 64],
        "EPOCHS": [10, 20],
    }
```
then snakemake will run $2^3=8$ jobs, each with a different combination of the parameters.
The top of config.py has a couple of configurable options for snakemake, like `SN_GPU` and `SN_ROOT`.
Change those as you wish.
Then you can run `snakemake -c4` to run 4 jobs in parallel and get through all the tasks.
The results will be saved in `SN_ROOT/<a_long_name_with_the_config>/model...pt`.

Note: You may need to set the environment variable `MKL_SERVICE_FORCE_INTEL` to 1 to avoid crashing. We're working
fixing this bug. For now you can run your snakemake commands with the variable set as follows
`MKL_SERVICE_FORCE_INTEL=1 snakemake -c4`

## Slurm
Snakemake can run with slurm, you need to set only a few things:
If you want to run on GPUs, set `SN_GPU` in config.py.
Adjust the slurm extra parameters in `Snakefile` to match your specifications, mostly the partition.
Make a slurm config file for snakemake, example: `~/.config/snakemake/slurm_gpu/config.yaml`:

```yaml
cluster: slurm
jobs: 128
retries: 3
default-resources:
  - slurm_account=nnolte
  - cluster_jobname="%r_%w_%T"
  - mem_mb=None
  - mem_mib=None
  - disk_mb=None
  - disk_mib=None
```

Then run with `MKL_SERVICE_FORCE_INTEL=1 snakemake --slurm --profile slurm_gpu`.
The MKL_SERVICE_FORCE_INTEL is needed on some clusters, not sure why.

## Changing default directories
Default directories can be updated using environment variables.
For instance, to change the default data directory, you can set the environment variable `NUCLR_DATA_DIR` using `export NUCLR_DATA_DIR=/path/to/data` in your shell/rc file.
Checkpoints and metadata are saved to `NUCLR_ROOT_DIR`, which defaults to `./results`.
##  TODO
- Use wandb artifacts to load models and data from a particular run.

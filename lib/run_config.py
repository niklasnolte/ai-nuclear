from .config_utils import where_am_i
import os

Clusters = dict(
    MIT = {
        "partition": "submit-gpu1080,submit-gpu",
        "root": f"/data/submit/{os.environ['USER']}/AI-NUCLEAR-LOGS",
    },
    HARVARD = {
        "partition": "iaifi_gpu",
        "root": os.path.expanduser("~/data/AI-NUCLEAR-LOGS"),
    },
    FAIR = {
        "partition": "learnlab",
        "root": f"/checkpoint/{os.environ['USER']}/nuclr",
    },
    Local = {
        "root": "./results" ,
        }
)

# snakemake configs (only apply if running with snakemake)
# can be changed at ones leisure
SM_ROOT = Clusters[where_am_i()]["root"]
SM_WANDB = True
SM_SLURM = True
SM_GPU = True

def train_cmd(
    hyperparams={},  # filled by snakemake wildcards
):
    return " ".join(
        [
            "MKL_SERVICE_FORCE_INTEL=GNU",
            f"python train.py",
            f"--WANDB" if SM_WANDB else "",
        ]  # or not
        + [f"--{k} {v}" for k, v in hyperparams.items()]
        + [f"--DEV cuda" if SM_GPU else "--DEV cpu"]
        + [f"--ROOT {SM_ROOT}"]
    )


def get_slurm_extra_resources():
    if SM_GPU:
        return " ".join(
            [
                "--gres=gpu:1",
                f"--partition={Clusters[where_am_i()]['partition']}",
                "--mem=10G",
                "--time=72:00:00",
            ]
        )
    else:
        return "--mem=10G"

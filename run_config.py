from config import Task
from run_config_utils import determine_cluster

# snakemake configs (only apply if running with snakemake)
# can be changed at ones leisure
SM_ROOT = determine_cluster().value["root"]
SM_WANDB = False
SM_SLURM = True
SM_GPU = True

def train_cmd(
    task: Task,
    hyperparams={},  # filled by snakemake wildcards
):
    return " ".join(
        [
            f"TASK={task.name}",
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
                f"--partition={determine_cluster().value['partition']}",
                "--mem=5G",
                "--time=1:00:00",
            ]
        )
    else:
        return "--mem=5G"

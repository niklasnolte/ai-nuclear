from enum import Enum
from config_utils import serialize_elements_in_task

ROOT = "results"
WANDB = True
SLURM = True
GPU = False


class Task(Enum):
    # make sure that the tasks don't have exactly the same config
    # otherwise enum is a bitch
    FULL = serialize_elements_in_task(
        dict(
            WD=[1e-2, 3e-3, 1e-3],
            LR=[1e-1, 5e-2, 1e-2],
            EPOCHS=[30000],
            TRAIN_FRAC=[0.8],
            HIDDEN_DIM=[256, 128],
            SEED=[0, 1, 2],
            MODEL=["baseline"],
            TARGETS_CLASSIFICATION=[
                {},
                {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
                {"stability": 1, "parity": 1, "spin": 1},
            ],
            TARGETS_REGRESSION=[
                {
                    "z": 1,
                    "n": 1,
                    "binding_energy": 1,
                    "radius": 1,
                    "half_life_sec": 1,
                    "abundance": 1,
                    "qa": 1,
                    "qbm": 1,
                    "qbm_n": 1,
                    "qec": 1,
                    "sn": 1,
                    "sp": 1,
                },
                {
                    "z": 1,
                    "n": 1,
                    "binding_energy": 5,
                    "radius": 1,
                    "half_life_sec": 1,
                    "abundance": 1,
                    "qa": 1,
                    "qbm": 1,
                    "qbm_n": 1,
                    "qec": 1,
                    "sn": 1,
                    "sp": 1,
                },
                {
                    "z": 5,
                    "n": 5,
                    "binding_energy": 1,
                    "radius": 1,
                    "half_life_sec": 1,
                    "abundance": 1,
                    "qa": 1,
                    "qbm": 1,
                    "qbm_n": 1,
                    "qec": 1,
                    "sn": 1,
                    "sp": 1,
                },
            ],
        )
    )

    BASELINE = dict()


def train_cmd(
    task: Task,
    hyperparams={},  # filled by snakemake wildcards
):
    return " ".join(
        [
            f"TASK={task.name}",
            f"python train.py",
            f"--WANDB" if WANDB else "",
        ]  # or not
        + [f"--{k} {v}" for k, v in hyperparams.items()]
        + [f"--DEV cuda" if GPU else "--DEV cpu"]
    )

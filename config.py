import argparse
from enum import Enum

ROOT = "results"
WANDB = True
SLURM = False
GPU = False


class Task(Enum):
    # make sure that the tasks don't have exactly the same config
    # otherwise enum is a bitch
    FULL = dict(
        WD=[1e-2, 1e-3, 1e-4],
        LR=[1e-1, 5e-2, 1e-2],
        EPOCHS=[10000],
        TRAIN_FRAC=[0.8],
        HIDDEN_DIM=[256, 128],
        SEED=[0, 1, 2, 3],
        MODEL=["baseline"],
        TARGETS_CLASSIFICATION=["stability,parity,spin,isospin"],
        TARGETS_REGRESSION=[
            "z,n,binding_energy,radius,half_life_sec,abundance,qa,qbm,qbm_n,qec,sn,sp"
        ],
    )
    BASELINE = dict()


def _args_postprocessing(args: argparse.Namespace):
    def _split(string):
      if string == "":
        return []
      else:
        return string.split(",")

    args.TARGETS_CLASSIFICATION = _split(args.TARGETS_CLASSIFICATION)
    args.TARGETS_REGRESSION = _split(args.TARGETS_REGRESSION)

    return args


def _parse_arguments(task: Task):
    parser = argparse.ArgumentParser()
    hyperparams = task.value
    for k, v in hyperparams.items():
        parser.add_argument(
            f"--{k}", type=type(v[0]), default=v[0]
        )  # TODO review float

    # operations params
    parser.add_argument("--DEV", type=str, default="cpu")
    parser.add_argument("--WANDB", action="store_true", default=False)
    return parser.parse_args()


def _make_suffix_for(arg: str):
    """
    define the name suffixes when adding an arg, eg mask_seed -> _maskseed{MASKSEED}
    this suffix has unfilled format braces, to be filled by the get_qualifed_name function
    """
    return f"_{arg.lower().replace('_', '')}{{{arg}}}"


def get_name(task: Task):
    name = "".join(
        [
            f"{task.name}",
            *[_make_suffix_for(hp) for hp in task.value.keys()],
        ]
    )
    return name


def _get_qualified_name(task: Task, args):
    name = get_name(task)
    return name.format(**vars(args))


def parse_arguments_and_get_name(task: Task):
    args = _parse_arguments(task)
    name = _get_qualified_name(task, args)
    args = _args_postprocessing(args)
    return args, name


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

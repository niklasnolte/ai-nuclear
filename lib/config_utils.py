from collections.abc import Iterable
import argparse
import os
from pathlib import Path
import socket

DATA_DIR = os.environ.get("NUCLR_DATA_DIR", Path(__file__).parent.parent / "data")
ROOT_DIR = os.environ.get("NUCLR_ROOT_DIR", Path(__file__).parent.parent / "results")

def where_am_i():
    host = socket.gethostname()

    if host.endswith("mit.edu") or host.startswith("submit"):
        return "MIT"
    elif host.endswith("harvard.edu") or host.startswith("holygpu"):
        return "HARVARD"
    elif "fair" in host or "learnlab" in host:
        return "FAIR"
    else:
        Warning(f"Unknown cluster: {host}")
        return "Local"


def _serialize_dict(targets: dict) -> str:
    if targets == {}:
        return "None"
    return "-".join([f"{k}:{v}" for k, v in targets.items()])


def _deserialize_dict(targets: str) -> dict:
    if targets == "None":
        return {}
    return {k: float(v) for k, v in [t.split(":") for t in targets.split("-")]}


def _serialize_list(targets: list) -> str:
    return "-".join([str(t) for t in targets])


def _deserialize_list(targets: str) -> list:
    return [float(t) for t in targets.split("-")]


def serialize_elements_in_task(task: dict):
    """
    some configurables are dicts, we need to serialize them
    """
    for t, choices in task.items():
        if not isinstance(choices, Iterable):  # should be list of hyperparam choices
            raise ValueError(
                f"{t} is not iterable in your config, fix in Enum Task (config.py)"
            )
        for i, choice in enumerate(choices):
            if isinstance(choice, dict):
                task[t][i] = _serialize_dict(choice)
            elif isinstance(choice, list):
                task[t][i] = _serialize_list(choice)
    return task


def _args_postprocessing(args: argparse.Namespace):
    # make them dicts again
    args.TARGETS_CLASSIFICATION = _deserialize_dict(args.TARGETS_CLASSIFICATION)
    args.TARGETS_REGRESSION = _deserialize_dict(args.TARGETS_REGRESSION)
    args.WHICH_FOLDS = [int(x) for x in _deserialize_list(args.WHICH_FOLDS)]

    # log freq
    if args.CKPT_FREQ == -1:
        # only log last
        args.CKPT_FREQ = args.EPOCHS + 1

    assert args.CKPT_FREQ % args.LOG_FREQ == 0, "ckpt_freq must be a multiple of log_freq"
    return args


def _add_operational_args_(parser: argparse.ArgumentParser):
    parser.add_argument("--DEV", type=str, default="cpu", help="device to use")
    parser.add_argument(
        "--WANDB", type=int, default=0, help="use wandb or not"
    )
    parser.add_argument(
        "--ROOT", type=str, default=ROOT_DIR, help="root folder to store models"
    )
    parser.add_argument("--LOG_FREQ", type=int, default=1, help="log every n epochs")
    parser.add_argument(
        "--CKPT_FREQ",
        type=int,
        default=-1,
        help="save checkpoint every n epochs, -1 == only log the last",
    )
    parser.add_argument("--exp_name", type=str, default="NUCLR", help="experiment name")


def _parse_arguments(params):
    parser = argparse.ArgumentParser()
    for k, v in params.items():
        parser.add_argument(
            f"--{k}", type=type(v[0]), default=v[0]
        )  # TODO review float

    _add_operational_args_(parser)  # add operational args
    # operations params
    return parser.parse_args()


def _make_suffix_for(arg: str):
    """
    define the name suffixes when adding an arg, eg mask_seed -> _maskseed{MASKSEED}
    this suffix has unfilled format braces, to be filled by the get_qualifed_name function
    """
    return f"{arg.lower().replace('_', '')}_{{{arg}}}"


def get_name(task):
    name = "/".join(
        [
            "NUCLR",
            *[_make_suffix_for(hp) for hp in task.keys()],
        ]
    )
    return name


def get_qualified_name(task, args):
    name = get_name(task)
    return name.format(**vars(args))


def parse_arguments_and_get_name(task):
    args = _parse_arguments(task)
    name = get_qualified_name(task, args)
    args = _args_postprocessing(args)
    return args, name

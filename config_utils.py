from collections.abc import Iterable
import argparse
import socket

def where_am_i():
    host = socket.gethostname()

    if host.endswith("mit.edu") or host.startswith("submit"):
        return "MIT"
    elif host.endswith("harvard.edu") or host.startswith("holygpu"):
        return "HARVARD"
    else:
        raise ValueError(f"Unknown cluster: {host}")

def _serialize_dict(targets: dict) -> str:
    if targets == {}:
      return "None"
    return "-".join([f"{k}:{v}" for k, v in targets.items()])


def _deserialize_dict(targets: str) -> dict:
    if targets == "None":
        return {}
    return {k: float(v) for k, v in [t.split(":") for t in targets.split("-")]}


def serialize_elements_in_task(task: dict):
    """
    some configurables are dicts, we need to serialize them
    """
    for t, choices in task.items():
        if not isinstance(choices, Iterable):  # should be list of hyperparam choices
            raise ValueError(
                f"{t} is not iterable in your config, fix in Enum Task (config.py)"
            )
        for i,choice in enumerate(choices):
            if isinstance(choice, dict):
                task[t][i] = _serialize_dict(choice)
    return task


def _args_postprocessing(args: argparse.Namespace):
    # make them dicts again
    args.TARGETS_CLASSIFICATION = _deserialize_dict(args.TARGETS_CLASSIFICATION)
    args.TARGETS_REGRESSION = _deserialize_dict(args.TARGETS_REGRESSION)

    # log freq
    if args.CKPT_FREQ == -1:
        # only log last
        args.CKPT_FREQ = args.EPOCHS + 1
    return args


def _parse_arguments(task):
    parser = argparse.ArgumentParser()
    hyperparams = task.value
    for k, v in hyperparams.items():
        parser.add_argument(
            f"--{k}", type=type(v[0]), default=v[0]
        )  # TODO review float

    # operations params
    parser.add_argument("--DEV", type=str, default="cpu", help="device to use")
    parser.add_argument("--WANDB", action="store_true", default=False, help="use wandb or not")
    parser.add_argument("--ROOT", type=str, default="./results", help="root folder to store models")
    parser.add_argument("--LOG_FREQ", type=int, default=100, help="log every n epochs")
    parser.add_argument("--CKPT_FREQ", type=int, default=-1, help="save checkpoint every n epochs, -1 == only log the last")
    return parser.parse_args()


def _make_suffix_for(arg: str):
    """
    define the name suffixes when adding an arg, eg mask_seed -> _maskseed{MASKSEED}
    this suffix has unfilled format braces, to be filled by the get_qualifed_name function
    """
    return f"{arg.lower().replace('_', '')}_{{{arg}}}"


def _get_qualified_name(task, args):
    name = get_name(task)
    return name.format(**vars(args))


def get_name(task):
    name = "/".join(
        [
            f"{task.name}",
            *[_make_suffix_for(hp) for hp in task.value.keys()],
        ]
    )
    return name


def parse_arguments_and_get_name(task):
    args = _parse_arguments(task)
    name = _get_qualified_name(task, args)
    args = _args_postprocessing(args)
    return args, name

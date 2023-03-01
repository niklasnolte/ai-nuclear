from collections.abc import Iterable
import argparse


def _serialize_dict(targets: dict) -> str:
    return ",".join([f"{k};{v}" for k, v in targets.items()])


def _deserialize_dict(targets: str) -> dict:
    if targets == "":
        return {}
    return {k: float(v) for k, v in [t.split(";") for t in targets.split(",")]}


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
    args.TARGETS_CLASSIFICATION = _deserialize_dict(args.TARGETS_CLASSIFICATION)
    args.TARGETS_REGRESSION = _deserialize_dict(args.TARGETS_REGRESSION)
    return args


def _parse_arguments(task):
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


def _get_qualified_name(task, args):
    name = get_name(task)
    return name.format(**vars(args))


def get_name(task):
    name = "".join(
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


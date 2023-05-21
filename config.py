from enum import Enum
from config_utils import serialize_elements_in_task, _deserialize_dict, _serialize_dict
from argparse import Namespace


class Task(Enum):
    # make sure that the tasks don't have exactly the same config
    # otherwise enum is a poopy head
    FULL = serialize_elements_in_task(
        dict(
            MODEL=["baseline"],
            WD=[1e-2],
            LR=[1e-2],
            EPOCHS=[50000],
            TRAIN_FRAC=[0.9],
            HIDDEN_DIM=[1024],
            DEPTH=[4],
            SEED=[0],
            BATCH_SIZE=[4069],
            TARGETS_CLASSIFICATION=[
                {},
                # {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
            ],
            TARGETS_REGRESSION=[
                {
                    # "binding": 1,
                    "binding_semf": 1,
                    # "z": 1,
                    # "n": 1,
                    "radius": 1,
                    # "volume": 1,
                    # "surface": 1,
                    # "symmetry": 1,
                    # "coulomb": 1,
                    # "delta": 1,
                    # "half_life_sec": 1,
                    # "abundance": 1,
                    "qa": 1,
                    "qbm": 1,
                    "qbm_n": 1,
                    "qec": 1,
                    "sn": 1,
                    # "sp": 1,
                },
            ],
            SCHED = ["cosine"],
            LIPSCHITZ = ["false"],
            TMS = ["keep"],
            HOLDOUT = ["false"],
            SIGMOID_READOUT = ["false", "true"],
            TAGS = ["rerun-from-symbols", "seed-check"],
            # CKPT = ["/home/submit/kitouni/ai-nuclear/results/FULL/model_baseline/wd_0.1/lr_0.01/epochs_10000/trainfrac_0.8/hiddendim_64/seed_0/batchsize_256/targetsclassification_None/targetsregression_binding:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/model_FULL_best.pt"
            # ],
            # CKPT = ["/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.1/lr_0.0001/epochs_2000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false/model_best.pt"],
            # OPTIM = ["adam"],
        )
    )

    MODULAR = serialize_elements_in_task(
        dict(
            MODEL=["baseline"],
            WD=[1e-1, 5e-2, 1e-1],
            LR=[1e-1, 1e-3, 1e-1],
            EPOCHS=[10000],
            TRAIN_FRAC=[0.383],
            HIDDEN_DIM=[32],
            SEED=[0, 1, 2],
            RANDOM_WEIGHTS=[
                0.0,
                0.1,
            ],  # level of entropy in randomness. 0 is uniform. 1000 is random one hot.
            P=[53],
            DIMREG_COEFF=[0.0, 2.0, 1.0, 0.5],
            DIMREG_EXP=[-1.5, -1.0],  # power to weight indices in dimn regularization
            TARGETS_CLASSIFICATION=[
                # {"add": 1, "subtract": 1},
                {"add": 1},
                {"add": 1, "subtract": 1, "multiply": 1},
            ],
            TARGETS_REGRESSION=[
                {},
            ],
        )
    )

    DEBUG = serialize_elements_in_task(
        dict(
            MODEL=["baseline"],
            WD=[1e-3],  # first one seems to be best
            LR=[1e-2],
            EPOCHS=[2],
            TRAIN_FRAC=[0.8],
            HIDDEN_DIM=[32],
            SEED=[0],
            RANDOM_WEIGHTS=[10],
            DIMREG_COEFF=[0.0, 2.0],
            DIMREG_EXP=[-1.5],  # power to weight indices in dimn regularization
            TARGETS_CLASSIFICATION=[
                {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
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
            ],
        )
    )


def get_args(task=Task.FULL):
    args = Namespace(**{k: v[0] for k, v in task.value.items()})
    args.TARGETS_CLASSIFICATION = _deserialize_dict(args.TARGETS_CLASSIFICATION)
    args.TARGETS_REGRESSION = _deserialize_dict(args.TARGETS_REGRESSION)
    args.DEV = "cpu"
    args.WANDB = False
    return args

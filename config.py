from enum import Enum
from config_utils import serialize_elements_in_task, _deserialize_dict, _serialize_dict
from argparse import Namespace


class Task(Enum):
    # make sure that the tasks don't have exactly the same config
    # otherwise enum is a poopy head
    FULL = serialize_elements_in_task(
        # FULL_dimnreg2_pcaalpha-1.5_wd0.003_lr0.1_epochs30000_trainfrac0.8_hiddendim256_seed1_modelbaseline_targetsclassificationNone_targetsregressionz:1-n:1-binding_energy:1-radius:1
        dict(
            MODEL=["baseline"],
            WD=[1e-2],
            LR=[1e-4],
            EPOCHS=[1000],
            TRAIN_FRAC=[0.9],
            HIDDEN_DIM=[2048],
            DEPTH=[2],
            SEED=[0],
            BATCH_SIZE=[1024],
            TARGETS_CLASSIFICATION=[
                {},
                {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
            ],
            TARGETS_REGRESSION=[
                {
                    "binding": 1,
                    "z": 1,
                    "n": 1,
                    # "radius": 1,
                    # "half_life_sec": 1,
                    # "abundance": 1,
                    # "qa": 1,
                    # "qbm": 1,
                    # "qbm_n": 1,
                    # "qec": 1,
                    # "sn": 1,
                    # "sp": 1,
                },
            ],
            # CKPT = ["/home/submit/kitouni/ai-nuclear/results/FULL/model_baseline/wd_0.1/lr_0.01/epochs_10000/trainfrac_0.8/hiddendim_64/seed_0/batchsize_256/targetsclassification_None/targetsregression_binding:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/model_FULL_best.pt"
            # ],
            CKPT = [None],
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

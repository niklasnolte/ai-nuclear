from enum import Enum
from config_utils import serialize_elements_in_task


class Task(Enum):
    # make sure that the tasks don't have exactly the same config
    # otherwise enum is a poopy head
    FULL = serialize_elements_in_task(
        # FULL_dimnreg2_pcaalpha-1.5_wd0.003_lr0.1_epochs30000_trainfrac0.8_hiddendim256_seed1_modelbaseline_targetsclassificationNone_targetsregressionz:1-n:1-binding_energy:1-radius:1
        dict(
            MODEL=["baseline"],
            WD=[1e-1],
            LR=[1e-4],
            EPOCHS=[10000],
            TRAIN_FRAC=[0.01],
            HIDDEN_DIM=[64],
            DIMREG_COEFF=[0.],
            DIMREG_EXP=[-1.5],  # power to weight indices in dimn regularization
            SEED=[2],
            BATCH_SIZE=[4],
            RANDOM_WEIGHTS=[0.], # level of entropy in randomness. 0 is uniform. 1000 is random one hot.
            TARGETS_CLASSIFICATION=[
                {},
                {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
            ],
            TARGETS_REGRESSION=[
                {
                    # "z": 1,
                    # "n": 1,
                    "binding_energy": 1,
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
            DIMREG_COEFF=[0., 2.0, 1.0, 0.5],
            DIMREG_EXP=[-1.5, -1.0],  # power to weight indices in dimn regularization
            TARGETS_CLASSIFICATION=[
                #{"add": 1, "subtract": 1},
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

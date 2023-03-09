from enum import Enum
from config_utils import serialize_elements_in_task


class Task(Enum):
    # make sure that the tasks don't have exactly the same config
    # otherwise enum is a poopy head
    FULL = serialize_elements_in_task(
        dict(
            MODEL=["splitup", "baseline"],
            WD=[3e-3],  # first one seems to be best
            LR=[1e-1],
            EPOCHS=[100000],
            TRAIN_FRAC=[0.8],
            HIDDEN_DIM=[64],
            SEED=[0, 1, 2],
            RANDOM_WEIGHTS=[0.1, 0.], # level of entropy in randomness. 0 is uniform. 1000 is random one hot.
            DIMREG_COEFF=[2],
            DIMREG_EXP=[-1.5], # power to weight indices in dimn regularization
            TARGETS_CLASSIFICATION=[
                {},
                {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
            ],
            TARGETS_REGRESSION=[
                {
                    "z": 1,
                    "n": 1,
                    "binding_energy": 1,
                    "radius": 1,
                },
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

    MODULAR = serialize_elements_in_task(
        dict(
            WD=[5e-2],
            LR=[1e-1],
            EPOCHS=[50000],
            TRAIN_FRAC=[0.5],
            HIDDEN_DIM=[64],
            SEED=[0, 1, 2],
            P = [53, 97],
            MODEL=["baseline"],
            TARGETS_CLASSIFICATION=[
              {"add": 1, "subtract": 1},
              {"add": 1, "multiply": 1},
              {"add": 1, "subtract": 1, "multiply": 1},
              ],
            TARGETS_REGRESSION=[{},],
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
            DIMREG_COEFF=[0,2],
            DIMREG_EXP=[-1.5], # power to weight indices in dimn regularization
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

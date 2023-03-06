from enum import Enum
from config_utils import serialize_elements_in_task


class Task(Enum):
    # make sure that the tasks don't have exactly the same config
    # otherwise enum is a poopy head
    FULL = serialize_elements_in_task(
        dict(
            WD=[3e-3, 1e-2, 1e-3],  # first one seems to be best
            LR=[1e-1, 2e-1, 8e-2],
            EPOCHS=[100000],
            TRAIN_FRAC=[0.8],
            HIDDEN_DIM=[512, 1024],
            SEED=[0, 1, 2],
            MODEL=["baseline"],
            RANDOM_WEIGHTS=[10], # level of entropy in randomness. 0 is uniform. 1000 is random one hot.
            TARGETS_CLASSIFICATION=[
                {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
                {},
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
                    "binding_energy": 1,
                    "radius": 1,
                },
            ],
        )
    )

    DEBUG = serialize_elements_in_task(
        dict(
            WD=[1e-3],  # first one seems to be best
            LR=[1e-2],
            EPOCHS=[1],
            TRAIN_FRAC=[0.8],
            HIDDEN_DIM=[32],
            SEED=[0],
            MODEL=["baseline"],
            RANDOM_WEIGHTS=[10],
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

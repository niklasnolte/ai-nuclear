class Targets:
    classification = ["stability", "parity", "spin", "isospin"]
    regression = [
        "z",
        "n",
        "binding_energy",
        "radius",
        "half_life_sec",
        "abundance",
        "qa",
        "qbm",
        "qbm_n",
        "qec",
        "sn",
        "sp",
    ]


class TrainConfig:
    WD = 1e-4
    LR = 1e-3
    EPOCHS = 100000
    TRAIN_FRAC = 0.8
    HIDDEN_DIM = 256
    SEED = 1
    MODEL = "baseline"
    ROOTPATH = "models"

from lib.config_utils import serialize_elements_in_task, _deserialize_dict
from argparse import Namespace


NUCLR = serialize_elements_in_task(
    dict(
        MODEL=["baseline"],
        WD=[0],
        L1REG=[1e-5],
        LR=[1e-2],
        EPOCHS=[50000],
        N_FOLDS=[n_folds := 10],
        WHICH_FOLDS=[[i] for i in range(n_folds)],
        HIDDEN_DIM=[128, 1024],
        DEPTH=[1],
        SEED=[0],
        BATCH_SIZE=[4096],
        INCLUDE_NUCLEI_GT=[8],
        TARGETS_CLASSIFICATION=[
            {},
            # {"stability": 1, "parity": 1, "spin": 1, "isospin": 1},
        ],
        TARGETS_REGRESSION=[
            {
                # "binding": 1,
                "binding_semf": 1,
                "z": 1,
                "n": 1,
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
                # "sn": 1,
                # "sp": 1,
            },
        ],
        SCHED=["cosine"],
        LIPSCHITZ=["false"],
        TMS=["remove"],  # keep or remove nuclei with high uncertainty in binding energy
        DROPOUT=[0.0, 0.05],
        FINAL_LR=[1e-5],
        BASE_WIDTH=[16], # the width used for HParam tuning
    )
)


def get_args():
    args = Namespace(**{k: v[0] for k, v in NUCLR.items()})
    args.TARGETS_CLASSIFICATION = _deserialize_dict(args.TARGETS_CLASSIFICATION)
    args.TARGETS_REGRESSION = _deserialize_dict(args.TARGETS_REGRESSION)
    args.DEV = "cpu"
    args.WANDB = False
    return args

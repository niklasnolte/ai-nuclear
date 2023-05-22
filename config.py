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
            N_FOLDS=[20],
            WHICH_FOLDS=[[i] for i in range(20)],
            HIDDEN_DIM=[1024],
            DEPTH=[4],
            SEED=[0],
            BATCH_SIZE=[4096],
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
                    "sp": 1,
                },
            ],
            SCHED = ["cosine"],
            FINAL_LR = [2e-5],
            LIPSCHITZ = ["false"],
            DROPOUT = [0.0],
            TMS = ["remove"]
            # CKPT = ["/home/submit/kitouni/ai-nuclear/results/FULL/model_baseline/wd_0.1/lr_0.01/epochs_10000/trainfrac_0.8/hiddendim_64/seed_0/batchsize_256/targetsclassification_None/targetsregression_binding:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/model_FULL_best.pt"
            # ],
            # CKPT = ["/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.1/lr_0.0001/epochs_2000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false/model_best.pt"],
            # OPTIM = ["adam"],
        )
    )


def get_args(task=Task.FULL):
    args = Namespace(**{k: v[0] for k, v in task.value.items()})
    args.TARGETS_CLASSIFICATION = _deserialize_dict(args.TARGETS_CLASSIFICATION)
    args.TARGETS_REGRESSION = _deserialize_dict(args.TARGETS_REGRESSION)
    args.DEV = "cpu"
    args.WANDB = False
    return args

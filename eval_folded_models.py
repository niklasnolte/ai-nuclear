# %%
import os
import torch
from nuclr.train import Trainer

# %%
path = "/checkpoint/nolte/nuclr/NUCLR/model_baseline/wd_0.01/lr_0.01/epochs_50000/nfolds_100/whichfolds_0/hiddendim_1024/depth_4/seed_0/batchsize_4096/includenucleigt_8/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1/sched_cosine/lipschitz_false/tms_remove/dropout_0.05/finallr_1e-05/"
trainer = Trainer.from_path(path, which_folds=list(range(100)))

# %%
{k:v**.5 if "metric" in k else v for k,v in trainer.val_step().items()}
# %%

# %%
import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import os
os.chdir("..")
plt.style.use("mystyle-bright.mplstyle")
from config import Task
from data import prepare_nuclear_data
from train_full import Trainer
import os
from mup import set_base_shapes
import yaml
from argparse import Namespace
print(os.getcwd())
# %%

def read_args(path, device=None):
    args = Namespace(**yaml.load(open(path, "r"), Loader=yaml.FullLoader)) 
    args.WANDB = False
    if not hasattr(args, "TMS"):
        args.TMS = "remove"
    if not hasattr(args, "HOLDOUT"):
        args.HOLDOUT = "false"
    if not hasattr(args, "SIGMOID_READOUT"):
        args.SIGMOID_READOUT = "false"
    if not hasattr(args, "READOUT"):
        args.READOUT = "identity"
    if not hasattr(args, "START_FROM"):
        args.START_FROM = "none"
    if not hasattr(args, "CLIP_GRAD"):
        args.CLIP_GRAD = 1e-3
    if device:
        args.DEV = device
    return args

# logdir="/work/submit/kitouni/ai-nuclear/FULL/model_baseline/wd_0.01/lr_0.0001/epochs_50000/trainfrac_0.9/hiddendim_2048/depth_2/seed_0/batchsize_1024/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false"
logdir="results/FULL/model_baseline/wd_0.01/lr_0.01/epochs_50000/trainfrac_0.9/hiddendim_1024/depth_4/seed_0/batchsize_4069/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1-sn:1-sp:1/sched_cosine/lipschitz_false/tms_keep/holdout_false/sigmoidreadout_false/tags_rerun-from-symbols/"
# list all models that end in an integer
models = [f for f in os.listdir(logdir) if f.split("_")[-1].split(".")[0].isdigit()]
models = sorted(models, key=lambda x: int(x.split("_")[-1].split(".")[0]))
model_idx = -1
print(models[model_idx])
model_dir = os.path.join(logdir, models[model_idx])
shapes = os.path.join(logdir, "shapes.yaml")
# shapes=None
# args = get_args(Task.FULL)
args = read_args(os.path.join(logdir, "args.yaml"))
data = prepare_nuclear_data(args)
trainer = Trainer(Task.BASE, args)
# trainer.model = torch.load(model_dir).to("cpu")
trainer.model.load_state_dict(torch.load(model_dir))
trainer.model = set_base_shapes(trainer.model, shapes, rescale_params=False, do_assert=True)

X = torch.cartesian_prod(*[torch.arange(0, trainer.data.vocab_size[i]) for i in range(len(trainer.data.vocab_size))])
X = X.to(args.DEV)
# %%
trainer.val_step()

# %%
trainer.model(X)

# %%

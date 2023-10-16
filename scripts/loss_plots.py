# %%
import os
import tqdm
import matplotlib.pyplot as plt
import torch
from collections import defaultdict
from loss import metric_by_task
from data import prepare_data, train_test_split
from config import Targets, TrainConfig

# %%
torch.manual_seed(TrainConfig.SEED)
data = prepare_data()
train_mask, test_mask = train_test_split(data, train_frac=TrainConfig.TRAIN_FRAC, seed=TrainConfig.SEED)

# %%
load_path = os.path.join(TrainConfig.ROOTPATH, TrainConfig.MODEL)
model = torch.load(os.path.join(load_path, "model_full.pt"))
final_sd = model.state_dict()

# %%
# get all models
model_sds = os.listdir(load_path)
model_sds = [m for m in model_sds if not m == "model_full.pt"]
model_sds = sorted(model_sds, key=lambda x: int(x.split("_")[1].split(".")[0]))
model_sds = [torch.load(os.path.join(load_path, m)) for m in model_sds] + [final_sd]

# %%
# get all losses
losses = defaultdict(list)
accs = defaultdict(list)
for model_sd in tqdm.tqdm(model_sds):
    model.load_state_dict(model_sd)
    out = model(data.X)
    test_loss = metric_by_task(out[test_mask], data.y[test_mask], data.output_map, data.regression_transformer)
    train_loss = metric_by_task(out[train_mask], data.y[train_mask], data.output_map, data.regression_transformer)
    for i, target in enumerate(Targets.classification + Targets.regression):
        losses[target + "_test"].append(test_loss[i].item())
        losses[target + "_train"].append(train_loss[i].item())



# %%
# plot losses in a grid
with torch.no_grad():
  fig, axes = plt.subplots(4, 4, figsize=(15, 15))
  for i, target in enumerate(Targets.classification + Targets.regression):

      ax = axes[i // 4, i % 4]
      ax.plot(losses[target + "_train"], label="train")
      ax.plot(losses[target + "_test"], label="test")
      ax.set_title(target)
      ax.legend()
      ax.set_yscale("log")
  fig.tight_layout()
  plt.show()
# %%
# TODO convert the losses such that they use original units

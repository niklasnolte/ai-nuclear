#%%
import os
import torch
from nuclr.train import Trainer
import matplotlib.pyplot as plt
import numpy as np
import re

# %%
#path = "/checkpoint/nolte/nuclr_revisited/NUCLR/model_baseline/wd_0.01/lr_0.01/epochs_50000/nfolds_100/whichfolds_0/hiddendim_1024/depth_2/seed_0/batchsize_4096/includenucleigt_8/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1/sched_cosine/lipschitz_false/tms_remove/dropout_0.0/finallr_1e-05/wdonembeddings_false/model_5000.pt"
# path = "/private/home/nolte/projects/ai-nuclear/results/NUCLR/model_baseline/wd_0.01/lr_0.01/epochs_500000/nfolds_100/whichfolds_0/hiddendim_1024/depth_1/seed_0/batchsize_4096/includenucleigt_8/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1/sched_cosine/lipschitz_false/tms_remove/dropout_0.0/finallr_1e-05/wdonembeddings_false/model_final.pt"
path = "/checkpoint/nolte/nuclr_revisited/NUCLR/model_baseline/wd_0.01/lr_0.001/epochs_50000/nfolds_100/whichfolds_0/hiddendim_1024/depth_1/seed_0/batchsize_4096/includenucleigt_20/targetsclassification_None/targetsregression_binding_semf:1-z:1-n:1-radius:1-qa:1-qbm:1-qbm_n:1-qec:1/sched_cosine/lipschitz_false/tms_remove/dropout_0.0/finallr_1e-05/wdonembeddings_false/model_final.pt"

path, model_str = os.path.split(path)
# epoch = int(re.findall(r"\d+", model_str)[0])
trainer = Trainer.from_path(path, which_folds=[0], epoch="final")
# %%
model = trainer.models[0]
module_of_interest = model.readout
# %%
#make hooks into the model to get the activations
activation_vectors = []
def forward_pre_hook(module, input):
    activation_vectors.append(input)
    return input
hook = module_of_interest.register_forward_pre_hook(forward_pre_hook)
trainer.val_step()
hook.remove()
activation_vectors = activation_vectors[0][0].detach()
# %%
class AutoEncoder(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super().__init__()
    self.encoder = torch.nn.Linear(input_dim, hidden_dim, bias=True)
    self.decoder = torch.nn.Parameter(torch.ones(hidden_dim, input_dim))
    self.input_bias = torch.nn.Parameter(torch.zeros(input_dim))

  def forward(self, x):
    x = self.encoder(x - self.input_bias)
    acts = torch.nn.functional.relu(x)
    x = acts @ torch.softmax(self.decoder, dim=0) + self.input_bias
    return x, acts

  def loss(self, x, lambda_l1=1e-3):
    y, acts = self(x)
    return torch.nn.functional.mse_loss(y, x) + lambda_l1 * acts.abs().mean()

# %%
#make the autoencoder
BATCH_SIZE=1024
STEPS = 50000
input_dim = 1024
hidden_dim = 1024*5
autoencoder = AutoEncoder(input_dim, hidden_dim).to(trainer.args.DEV)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)

# %%
if os.path.exists("autoencoder.pt"):
  autoencoder.load_state_dict(torch.load("autoencoder.pt"))
else:
  for step in range(STEPS):
    sample = torch.randperm(len(activation_vectors))[:BATCH_SIZE]
    optimizer.zero_grad()
    loss = autoencoder.loss(activation_vectors[sample])
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 100 == 0:
      print(f"Step {step} loss {loss.item()}")
  torch.save(autoencoder.state_dict(), "autoencoder.pt")

# %%

autooutputs, wide_acts = autoencoder(activation_vectors)
(wide_acts > 0).sum(1).float().mean()

# replace the activation vectors from the autoencoder in the model and see what comes out
#module_of_interest.
# %%
#remove hooks
metrics_default = trainer.val_step()
def forward_pre_hook(module, input):
    print("using autoencoder")
    return autooutputs#autoencoder(input[0])[0]

hook = module_of_interest.register_forward_pre_hook(forward_pre_hook)
metrics_autoencoder = trainer.val_step()
hook.remove()
# %%
metrics_default, metrics_autoencoder

# %%
plt.plot(wide_acts.sum(0).detach().cpu().numpy())
# %%
(wide_acts.sum(0) > 0).float().mean()
# %%
def binding(wide_act):
    activation = wide_act @ torch.softmax(autoencoder.decoder, dim=0) + autoencoder.input_bias
    output = torch.sigmoid(model.readout(activation))
    # unscaled = trainer._unscale_output(output)
    return output[:, 0]
    mask = trainer.data.val_masks[0]
    task = torch.arange(8).repeat(len(unscaled) // 8).to(unscaled.device)
    return trainer.metrics_by_task(task[mask], unscaled[mask], trainer.unscaled_y[mask])[0][0]


# %%
nonzero_idxs = wide_acts.sum(0).nonzero().flatten().cpu().numpy()

# with torch.no_grad():
#   n_idxs = len(nonzero_idxs)
#   print(n_idxs)
#   plt.subplots(n_idxs, 1, figsize=(5,5*n_idxs), dpi=100)
#   for i,idx in enumerate(nonzero_idxs):
#     plt.subplot(n_idxs, 1, i+1)
#     plt.title(idx)
#     plt.scatter(wide_acts[::8, idx].cpu().numpy(), binding(wide_acts)[::8].cpu().numpy(), alpha=.3, s=5)
# %%
# with torch.no_grad():
#   # good_idx = 6641
#   for good_idx in [9979]:#nonzero_idxs:
#     plt.title(good_idx)
#     for i, act in enumerate(wide_acts[::8][torch.randperm(len(wide_acts)//8)[:100]]):
#       if act[good_idx] > 0:
#         n_points = 100
#         xrange = torch.linspace(0, act[good_idx]*2, n_points)
#         acts_perturbed = act.repeat(n_points, 1)
#         acts_perturbed[:, good_idx] = xrange
#         plt.scatter(xrange.cpu().numpy(), binding(acts_perturbed).cpu().numpy(), alpha=.3, s=5)
#     plt.show()

# %%
# lets calculate the gradient of the binding energy with respect to the activation
with torch.no_grad():
  _input = wide_acts[::8].clone()
_input.requires_grad_(True)
binding(_input).sum().backward()
# %%
n_tops = len(nonzero_idxs)
_, topk_idxs = _input.grad.abs().topk(n_tops)
assert all([len(topk_idxs[:,i].unique()) == 1 for i in range(n_tops)]) # linear enough
# %%
torch.isin(topk_idxs, wide_acts.sum(0).nonzero().flatten()).all()

# %%
with torch.no_grad():
  # measure the effect on the BE when removing those activations
  output = binding(_input)
  diffs = {}
  for idx in topk_idxs[0]:
    zerod_input = _input.clone()
    zerod_input[:, idx] = 0
    zerod_output = binding(zerod_input)
    diff = output - zerod_output
    diffs[idx.item()] = diff.cpu().numpy()
# %%
zs = trainer.data.X[::8][:, 0].cpu().numpy()
ns = trainer.data.X[::8][:, 1].cpu().numpy()

# %%
# 3d plot with topk_inputs as color and zs and ns on the axes
with torch.no_grad():
  for idx in topk_idxs[0]:
    plt.scatter(zs, ns, c=diffs[idx.cpu().item()], s=5)
    plt.title(f"activation {idx}")
    plt.colorbar()
    plt.xlabel("Z")
    plt.ylabel("N")
    # plt.scatter(ns, topk_inputs[:,0], alpha=.3, s=5)
    plt.show()
# %%
len(nonzero_idxs)
# %%
with torch.no_grad():
  for idx in topk_idxs[0]:
    plt.scatter(zs, ns, c=_input[:, idx].cpu().numpy(),s=5)
    plt.title(f"activation {idx}")
    plt.colorbar()
    plt.xlabel("Z")
    plt.ylabel("N")
    # plt.scatter(ns, topk_inputs[:,0], alpha=.3, s=5)
    plt.show()

# %%
features = _input[:,nonzero_idxs].shape
inputs = trainer.data.X[::8]

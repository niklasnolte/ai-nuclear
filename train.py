import torch
import tqdm
from data import prepare_data, train_test_split
from config import Targets, TrainConfig, MiscConfig
from model import BaselineModel, ResidualModel
from loss import loss_by_task, metric_by_task
import os

torch.manual_seed(TrainConfig.SEED)

device = torch.device(MiscConfig.DEVICE)

data = prepare_data()
train_mask, test_mask = train_test_split(
    data, train_frac=TrainConfig.TRAIN_FRAC, seed=TrainConfig.SEED
)

save_path = os.path.join(TrainConfig.ROOTPATH, TrainConfig.MODEL, "all")
os.makedirs(save_path, exist_ok=True)

# set up model
n_protons, n_neutrons = data.vocab_size
output_dims = data.output_map.values()
if TrainConfig.MODEL == "baseline":
    model_class = BaselineModel
elif TrainConfig.MODEL == "residual":
    model_class = ResidualModel

model = model_class(
    n_protons,
    n_neutrons,
    hidden_dim=TrainConfig.HIDDEN_DIM,
    output_dim=sum(output_dims),
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=TrainConfig.LR, weight_decay=TrainConfig.WD
)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer, T_max=TrainConfig.EPOCHS
# )

# remove old models
for f in os.listdir(save_path):
    if f.endswith(".pt"):
      os.remove(os.path.join(save_path, f))

bar = tqdm.trange(TrainConfig.EPOCHS)
for epoch in bar:
    # Train
    model.train()
    optimizer.zero_grad()
    out = model(data.X)
    train_loss = loss_by_task(out[train_mask], data.y[train_mask], data.output_map)
    loss = train_loss.mean()
    loss.backward()
    optimizer.step()
#    scheduler.step()
    if epoch % 100 == 0:
        with torch.no_grad():
            # Test
            model.eval()
            val_loss = metric_by_task(
                out[test_mask],
                data.y[test_mask],
                data.output_map,
                class_weights=data.class_weights[test_mask],
                qt=data.regression_transformer,
            )
            msg = f"\nEpoch {epoch} Train losses:\n"
            for i, target in enumerate(data.output_map.keys()):
                msg += f"{target}: {train_loss[i].item():.3f}\n"
            msg += f"\nEpoch {epoch} Val metrics:\n"
            for i, target in enumerate(data.output_map.keys()):
                msg += f"{target}: {val_loss[i].item():.3f}\n"
        print(msg)
        # save model
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{epoch}.pt"))

torch.save(model, os.path.join(save_path, "model_full.pt"))

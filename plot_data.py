# %%
# Plot data
import math
import pandas as pd
from argparse import Namespace
from config import Task
from config_utils import _deserialize_dict
from data import prepare_nuclear_data, train_test_split
import matplotlib.pyplot as plt

TASK = Task.FULL
args= Namespace(**{k:v[0] for k,v in Task.FULL.value.items()})
args.TARGETS_CLASSIFICATION = _deserialize_dict(args.TARGETS_CLASSIFICATION)
args.TARGETS_REGRESSION = _deserialize_dict(args.TARGETS_REGRESSION)
args.DEV = "cpu"

print(args)
data = prepare_nuclear_data(args)
train_mask, _ = train_test_split(data, train_frac=0.8, seed=1)
# %%
df = pd.DataFrame(data.y, columns=data.output_map.keys())
df["train"] = train_mask
fig, axes = plt.subplots(2, math.ceil(len(data.output_map)/2), figsize=(20, 5), sharey=True)
df.groupby("train").hist(bins=20, figsize=(20, 5), grid=False, density=True, ax=axes.flatten(), alpha=0.8); plt.show()

df["abundance"].hist(grid=False); plt.show()

# %%
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
x = np.stack([np.linspace(-i, i, 100) for i in range(1, 3)])
df = pd.DataFrame(x.T)
df.columns = ["a", "b"]
scalers = [RobustScaler(), StandardScaler(), MinMaxScaler()]
for scaler in scalers:
        df2 = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df2.hist(grid=False); plt.show()

# %%

# %%
import numpy as np
import pandas as pd
import urllib.request
import os
import torch

# %%
def apply_to_df_col(column):
    def wrapper(fn):
      return lambda df: df[column].astype(str).apply(fn)
    return wrapper

@apply_to_df_col(column="jp")
def get_spin_from(string):
    string = (
        string.replace("(", "")
        .replace(")", "")
        .replace("+", "")
        .replace("-", "")
        .replace("]", "")
        .replace("[", "")
        .replace("GE", "")
        .replace("HIGH J", "")
        .replace(">", "")
        .replace("<", "")
        .strip()
        .split(" ")[0]
    )
    if string == "":
        return float("nan")
    else:
        return float(eval(string)) #eval for 1/2 and such

@apply_to_df_col("jp")
def get_parity_from(string):
    # find the first + or -
    found_plus = string.find("+")
    found_minus = string.find("-")

    if found_plus == -1 and found_minus == -1:
        return float("nan")
    elif found_plus == -1:
        return 0 # -
    elif found_minus == -1:
        return 1 # +
    elif found_plus < found_minus:
        return 1 # +
    elif found_plus > found_minus:
        return 0 # -
    else:
        raise ValueError("something went wrong")

def get_half_life_from(df):
    # selection excludes unknown lifetimes and ones where lifetimes are given as bounds
    series = df.half_life_sec.copy()
    series[(df.half_life_sec == " ") | (df.operator_hl != " ")] = float("nan")
    series = series.astype(float)
    series = series.apply(np.log10)
    return series

@apply_to_df_col("qa")
def get_qa_from(string):
    # ~df.qa.isna() & (df.qa != " ")
    if string == " ":
        return float("nan")
    else:
        return float(string)

@apply_to_df_col("qbm")
def get_qbm_from(string):
    return float(string.replace(" ", "nan"))

@apply_to_df_col("qbm_n")
def get_qbm_n_from(string):
    return float(string.replace(" ", "nan"))

@apply_to_df_col("qec")
def get_qec_from(string):
    return float(string.replace(" ", "nan"))

@apply_to_df_col("sn")
def get_sn_from(string):
    return float(string.replace(" ", "nan"))

@apply_to_df_col("sp")
def get_sp_from(string):
    return float(string.replace(" ", "nan"))


def get_abundance_from(df):
    # abundance:
    # assumes that " " means 0
    return df.abundance.replace(" ", "0").astype(float)

@apply_to_df_col("half_life")
def get_stability_from(string):
    if string == "STABLE":
      return 1.
    elif string == " ":
      return float("nan")
    else:
      return 0.

@apply_to_df_col("isospin")
def get_isospin_from(string):
    return float(eval(string.replace(" ", "float('nan')")) )

def get_binding_energy_from(df):
    return df.binding.replace(" ", "nan").astype(float)

def get_radius_from(df):
    return df.radius.replace(" ", "nan").astype(float)

def get_targets(df):
  # place all targets into targets an empty copy of df
  targets = df[["z", "n"]].copy()
  # binding energy per nucleon
  targets["binding_energy"] = get_binding_energy_from(df)
  # radius in fm
  targets["radius"] = get_radius_from(df)
  # half life in log10(sec)
  targets["half_life_sec"] = get_half_life_from(df)
  # stability in {0, 1, nan}
  targets["stability"] = get_stability_from(df)
  # spin as float
  targets["spin"] = get_spin_from(df)
  # parity as {0 (-),1 (+), nan}
  targets["parity"] = get_parity_from(df)
  # isotope abundance in %
  targets["abundance"] = get_abundance_from(df)
  # qa = alpha decay energy in keV
  targets["qa"] = get_qa_from(df)
  # qbm = beta minus decay energy in keV
  targets["qbm"] = get_qbm_from(df)
  # qbm_n = beta minus + neutron emission energy in keV
  targets["qbm_n"] = get_qbm_n_from(df)
  # qec = electron capture energy in keV
  targets["qec"] = get_qec_from(df)
  # sn = neutron separation energy in keV
  targets["sn"] = get_sn_from(df)
  # sp = proton separation energy in keV
  targets["sp"] = get_sp_from(df)
  # isospin as float
  targets["isospin"] = get_isospin_from(df)
  return targets

# %%

def get_data(recreate=False):
    np.random.seed(1)

    def lc_read_csv(url):
        req = urllib.request.Request("https://nds.iaea.org/relnsd/v0/data?" + url)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0",
        )
        return pd.read_csv(urllib.request.urlopen(req))

    if recreate or not os.path.exists("data/ground_states.csv"):
        df = lc_read_csv("fields=ground_states&nuclides=all")
        df.to_csv("data/ground_states.csv", index=False)
    else:
        df = pd.read_csv("data/ground_states.csv")
    return df

def prepare_for_liftoff(training_targets = None, recreate=False):
    """
    training_targets: Optional[List]
        List of targets to use for training. If None, all targets are used.
    recreate: bool
        If True, the data is re-downloaded and re-prepared.
    """

    targets = get_targets(get_data(recreate=recreate))
    vocab_size = (targets.z.nunique(), targets.n.nunique())
    X = torch.tensor(targets[["z", "n"]].values).long()
    if training_targets is None:
        training_targets = targets.columns
    y = torch.tensor(targets[training_targets].values).float()
    return X, y, vocab_size




# %%
df = get_data()
targets = get_targets(df)
X, y, vocab_size = prepare_for_liftoff()
# %%
y
# %%
targets = get_targets(get_data())
# %%
targets
# %%
targets.columns
# %%
classification_targets = ["stability", "parity"]
regression_targets = ["z", "n", "half_life_sec", "abundance", "binding_energy", "radius", "spin", "qa", "qbm", "qbm_n", "qec", "sn", "sp", "isospin"]

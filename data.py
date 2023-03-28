import numpy as np
import pandas as pd
import urllib.request
import os
from sklearn.preprocessing import (
    QuantileTransformer,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
import torch
import argparse
from collections import namedtuple, OrderedDict


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
        return float(eval(string))  # eval for 1/2 and such


@apply_to_df_col("jp")
def get_parity_from(string):
    # find the first + or -
    found_plus = string.find("+")
    found_minus = string.find("-")

    if found_plus == -1 and found_minus == -1:
        return float("nan")
    elif found_plus == -1:
        return 0  # -
    elif found_minus == -1:
        return 1  # +
    elif found_plus < found_minus:
        return 1  # +
    elif found_plus > found_minus:
        return 0  # -
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
        return 1.0
    elif string == " ":
        return float("nan")
    else:
        return 0.0


@apply_to_df_col("isospin")
def get_isospin_from(string):
    return float(eval(string.replace(" ", "float('nan')")))


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


def get_nuclear_data(recreate=False):
    def lc_read_csv(url):
        req = urllib.request.Request("https://nds.iaea.org/relnsd/v0/data?" + url)
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0",
        )
        return pd.read_csv(urllib.request.urlopen(req))

    if not os.path.exists("data"):
        os.mkdir("data")
    if recreate or not os.path.exists("data/ground_states.csv"):
        df = lc_read_csv("fields=ground_states&nuclides=all")
        df.to_csv("data/ground_states.csv", index=False)
    else:
        df = pd.read_csv("data/ground_states.csv")

    df = df[(df.z > 8) & (df.n > 8) & (df.n < 160)]  # TODO remove this line
    return df


Data = namedtuple(
    "Data",
    ["X", "y", "vocab_size", "output_map", "regression_transformer"],
)


def prepare_nuclear_data(config: argparse.Namespace, recreate: bool = False):
    """Prepare data to be used for training. Transforms data to tensors, gets tokens X,targets y,
    vocab size and output map which is a dict of {target:output_shape}. Usually output_shape is 1 for regression
    and n_classes for classification.

    Args:
        columns (list, optional): List of columns to use as targets. Defaults to None.
        recreate (bool, optional): Force re-download of data and save to csv. Defaults to False.
    returns (Data): namedtuple of X, y, vocab_size, output_map, quantile_transformer
    """
    targets = get_targets(get_nuclear_data(recreate=recreate))

    X = torch.tensor(targets[["z", "n"]].values).long()

    # classification targets increasing integers
    for col in config.TARGETS_CLASSIFICATION:
        targets[col] = targets[col].astype("category").cat.codes
        # put nans back
        targets[col] = targets[col].replace(-1, np.nan)

    vocab_size = (targets.z.max() + 1, targets.n.max() + 1)
    output_map = OrderedDict()
    for target in config.TARGETS_CLASSIFICATION:
        output_map[target] = targets[target].nunique()

    for target in config.TARGETS_REGRESSION:
        output_map[target] = 1

    reg_columns = list(config.TARGETS_REGRESSION)
    # qt = QuantileTransformer(output_distribution="uniform")
    feature_transformer = RobustScaler()
    if len(reg_columns) > 0:
        targets[reg_columns] = feature_transformer.fit_transform(
            targets[reg_columns].values
        )

    y = torch.tensor(targets[list(output_map.keys())].values).float()
    return Data(
        X.to(config.DEV),
        y.to(config.DEV),
        vocab_size,
        output_map,
        feature_transformer,
    )


def prepare_modular_data(args: argparse.Namespace):
    # modular arithmetic data
    # X = cartesian product of [0..p] x [0..p]
    # y = (x1 op x2) % p
    X = torch.cartesian_prod(torch.arange(args.P), torch.arange(args.P))
    output_map = OrderedDict()
    for target in args.TARGETS_CLASSIFICATION:
        output_map[target] = args.P
    for target in args.TARGETS_REGRESSION:
        output_map[target] = 1

    y = torch.zeros(X.shape[0], len(output_map))
    for idx, target in enumerate(output_map):
        if target == "add":
            y[:, idx] = (X[:, 0] + X[:, 1]) % args.P
        elif target == "subtract":
            y[:, idx] = (X[:, 0] - X[:, 1]) % args.P
        elif target == "multiply":
            y[:, idx] = (X[:, 0] * X[:, 1]) % args.P
        else:
            raise ValueError(f"Unknown target {target}")

    feature_transformer = RobustScaler()
    reg_cols = -len(args.TARGETS_REGRESSION)
    if reg_cols != 0:
        y[:, reg_cols:] = feature_transformer.fit_transform(y[:, reg_cols:])

    return Data(X.to(args.DEV), y.to(args.DEV), args.P, output_map, feature_transformer)


def prepare_modular_data(args: argparse.Namespace):
    # modular arithmetic data
    # X = cartesian product of [0..p] x [0..p]
    # y = (x1 op x2) % p
    X = torch.cartesian_prod(torch.arange(args.P), torch.arange(args.P))

    output_map = OrderedDict()
    for target in args.TARGETS_CLASSIFICATION:
        output_map[target] = args.P
    for target in args.TARGETS_REGRESSION:
        output_map[target] = 1

    y = torch.zeros(X.shape[0], len(output_map))
    for idx, target in enumerate(output_map):
        if target == "add":
            y[:, idx] = (X[:, 0] + X[:, 1]) % args.P
        elif target == "subtract":
            y[:, idx] = (X[:, 0] - X[:, 1]) % args.P
        elif target == "multiply":
            y[:, idx] = (X[:, 0] * X[:, 1]) % args.P
        else:
            raise ValueError(f"Unknown target {target}")

    feature_transformer = RobustScaler()
    reg_cols = -len(args.TARGETS_REGRESSION)
    if reg_cols != 0:
        y[:, reg_cols:] = feature_transformer.fit_transform(y[:, reg_cols:])

    return Data(X.to(args.DEV), y.to(args.DEV), args.P, output_map, feature_transformer)


def train_test_split_exact(data, train_frac, seed=1):
    """
    Take exactly train_frac of the data as training data.
    """
    # TODO shuffle data when using SGD
    device = data.X.device
    torch.manual_seed(seed)
    train_mask = torch.ones(data.X.shape[0], dtype=torch.bool)
    train_mask[int(train_frac * data.X.shape[0]) :] = False
    train_mask = train_mask[torch.randperm(data.X.shape[0])]
    test_mask = ~train_mask
    return train_mask.to(device), test_mask.to(device)


def train_test_split_sampled(data, train_frac, seed=1):
    """
    Samples are assigned to train by a bernoulli distribution with probability train_frac.
    """
    device = data.X.device
    torch.manual_seed(seed)
    train_mask = torch.rand(data.X.shape[0]) < train_frac
    test_mask = ~train_mask
    return train_mask.to(device), test_mask.to(device)


train_test_split = train_test_split_sampled

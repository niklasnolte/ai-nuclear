import numpy as np
import pandas as pd
import urllib.request
import os
from sklearn.preprocessing import (
    MinMaxScaler,
)
import torch
import argparse
import warnings
from collections import namedtuple, OrderedDict


def delta(Z, N):
    A = Z + N
    aP = 11.18
    delta = aP * A ** (-1 / 2)
    delta[(Z % 2 == 1) & (N % 2 == 1)] *= -1
    delta[(Z % 2 == 0) & (N % 2 == 1)] = 0
    delta[(Z % 2 == 1) & (N % 2 == 0)] = 0
    return delta


def shell(Z, N):
    # calculates the shell effects according to "Mutual influence of terms in a semi-empirical" Kirson
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    alpham = -1.9
    betam = 0.14
    magic = [2, 8, 20, 28, 50, 82, 126, 184]

    def find_nearest(lst, target):
        return min(lst, key=lambda x: abs(x - target))

    nup = np.array([abs(x - find_nearest(magic, x)) for x in Z])
    nun = np.array([abs(x - find_nearest(magic, x)) for x in N])
    P = nup * nun / (nup + nun)
    P[np.isnan(P)] = 0
    return alpham * P + betam * P**2


def semi_empirical_mass_formula(Z, N):
    A = N + Z
    aV = 15.75
    aS = 17.8
    aC = 0.711
    aA = 23.7
    Eb = (
        aV * A
        - aS * A ** (2 / 3)
        - aC * Z * (Z - 1) / (A ** (1 / 3))
        - aA * (N - Z) ** 2 / A
        + delta(Z, N)
    )
    Eb[Eb < 0] = 0
    return Eb / A * 1000  # keV


# TODO move all the physics functions to a separate file
def BW2_mass_formula(Z, N):
    A = N + Z

    aV = 16.58
    aS = -26.95
    aC = -0.774
    aA = -31.51
    axC = 2.22
    aW = -43.4
    ast = 55.62
    aR = 14.77

    Eb = (
        aV * A
        + aS * A ** (2 / 3)
        + aC * Z**2 / (A ** (1 / 3))
        + aA * (N - Z) ** 2 / A
        + delta(Z, N)
        + shell(Z, N)
        + aR * A ** (1 / 3)
        + axC * Z ** (4 / 3) / A ** (1 / 3)
        + aW * abs(N - Z) / A
        + ast * (N - Z) ** 2 / A ** (4 / 3)
    )

    Eb[Eb < 0] = 0
    return Eb / A * 1000  # keV


def WS4_mass_formula(df):
    N = df["n"].values
    Z = df["z"].values
    A = N + Z
    Da = 931.494102
    mp = 938.78307
    mn = 939.56542

    file_path = os.path.join(os.path.dirname(__file__), "data", "WS4.txt")

    df_WS4 = pd.read_fwf(file_path, widths=[9, 9, 15, 15])

    df_WS4["Z"] = df_WS4["Z"].astype(float)
    df_WS4["N"] = df_WS4["A"].astype(float) - df_WS4["Z"]

    # Merge the two dataframes based on 'Z' and 'N'
    merged_df = pd.merge(
        df, df_WS4, how="left", left_on=["z", "n"], right_on=["Z", "N"]
    )

    merged_df["WS4"] = (
        Z * mp + N * mn - merged_df["WS4"].astype(float) - A * Da
    )

    # Create a new column 'WS4' in the merged dataframe and fill it with values from 'WS4' column in df_WS4
    merged_df["WS4"] = merged_df["WS4"].fillna(0)

    # Drop unnecessary columns from the merged dataframe
    merged_df = merged_df.drop(["A", "Z", "N", "WS4+RBF"], axis=1)

    Eb = merged_df["WS4"].values.astype(float)

    Eb[Eb < 0] = 0
    return Eb / A * 1000  # keV


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
    binding = df.binding.replace(" ", "nan").astype(float)
    return binding


def get_radius_from(df):
    return df.radius.replace(" ", "nan").astype(float)


def get_targets(df):
    # place all targets into targets an empty copy of df
    targets = df[["z", "n"]].copy()
    # binding energy per nucleon
    targets["binding"] = get_binding_energy_from(df)
    # binding energy per nucleon minus semi empirical mass formula
    targets["binding_semf"] = targets.binding - semi_empirical_mass_formula(
        df.z, df.n
    )
    # binding energy per nucleon minus semi empirical mass formula (including shell effects)
    targets["binding_BW2"] = targets.binding - BW2_mass_formula(df.z, df.n)
    # binding energy per nucleon minus WS4 formula
    targets["binding_WS4"] = targets.binding - WS4_mass_formula(df)
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
    # These are semi-empirical mass formula terms
    targets["volume"] = targets.z + targets.n  # volume
    targets["surface"] = targets.volume ** (2 / 3)  # surface
    targets["symmetry"] = (
        (targets.z - targets.n) ** 2
    ) / targets.volume  # symmetry
    targets["delta"] = delta(targets.z, targets.n)  # delta
    targets["coulomb"] = (targets.z**2 - targets.z) / targets.volume ** (
        1 / 3
    )  # coulomb

    return targets


def get_nuclear_data(recreate=False):
    def lc_read_csv(url):
        req = urllib.request.Request(
            "https://nds.iaea.org/relnsd/v0/data?" + url
        )
        req.add_header(
            "User-Agent",
            "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:77.0) Gecko/20100101 Firefox/77.0",
        )
        return pd.read_csv(urllib.request.urlopen(req))

    os.makedirs("data", exist_ok=True)
    # TODO explain what is hapenning here
    df2 = pd.read_csv("data/ame2020.csv").set_index(["z", "n"])
    df2 = df2[~df2.index.duplicated(keep="first")]
    if recreate or not os.path.exists("data/ground_states.csv"):
        df = lc_read_csv("fields=ground_states&nuclides=all")
        df.to_csv("data/ground_states.csv", index=False)
    df = pd.read_csv("data/ground_states.csv").set_index(["z", "n"])
    df["binding_unc"] = df2.binding_unc
    df["binding_sys"] = df2.binding_sys
    df.reset_index(inplace=True)

    return df


Data = namedtuple(
    "Data",
    [
        "X",
        "y",
        "vocab_size",
        "output_map",
        "regression_transformer",
        "train_masks",
        "val_masks",
        "scaled_idxs",
    ],
)


# TODO explain what is happening here
def _train_test_split(size, n_folds, X, seed=1):
    torch.manual_seed(seed)
    all_zs = X[:, 0]
    all_ns = X[:, 1]
    while True:
        train_idx = torch.repeat_interleave(
            torch.arange(n_folds), size // n_folds + 1
        )[:size]
        train_idx = train_idx[torch.randperm(size)]
        train_masks = [train_idx != i for i in range(n_folds)]
        for train_mask in train_masks:
            if len(all_zs[train_mask].unique()) != len(all_zs.unique()) or len(
                all_ns[train_mask].unique()
            ) != len(all_ns.unique()):
                print("Resampling train mask")
                break
        else:
            val_masks = [train_idx == i for i in range(n_folds)]
            return torch.stack(train_masks), torch.stack(val_masks)


def _leave_one_plus_four_out(X, train_masks, val_masks, output_map, seed=1):
    # we have to avoid cheating, so we need to remove from training
    # data that is too correlated with the validation data
    # so for each Sn, Sp target, we remove the M of all ajacent nuclei
    # and for M, we remove the Sp and Sn of all ajacent nuclei
    binding_idx = [
        i for i, x in enumerate(output_map.keys()) if "binding" in x
    ]
    sn_idx = [i for i, x in enumerate(output_map.keys()) if "sn" in x]
    sp_idx = [i for i, x in enumerate(output_map.keys()) if "sp" in x]

    if len(binding_idx) & (len(sn_idx) ^ len(sp_idx)):
        raise ValueError("sn and sp must be both present or both absent")
    elif ~(len(sn_idx) & len(sp_idx) & len(binding_idx)):
        print("No sn, sp or binding data, skipping leave one out")
        return train_masks

    binding_idx = binding_idx[0]
    sn_idx = sn_idx[0]
    sp_idx = sp_idx[0]

    for fold, val_mask in enumerate(val_masks):
        X_val_set = X[val_mask]
        X_binding = X_val_set[X_val_set[:, 2] == binding_idx]
        X_sn = X_val_set[X_val_set[:, 2] == sn_idx]
        X_sp = X_val_set[X_val_set[:, 2] == sp_idx]
        binding_relative_removals = torch.tensor(
            [
                [0, 0, sn_idx - binding_idx],
                [0, 0, sp_idx - binding_idx],
                [-1, 0, sn_idx - binding_idx],
                [-1, 0, sp_idx - binding_idx],
                [1, 0, sn_idx - binding_idx],
                [1, 0, sp_idx - binding_idx],
                [0, -1, sn_idx - binding_idx],
                [0, -1, sp_idx - binding_idx],
                [0, 1, sn_idx - binding_idx],
                [0, 1, sp_idx - binding_idx],
            ]
        )
        sn_relative_removals = torch.tensor(
            [
                [0, 0, binding_idx - sn_idx],
                [-1, 0, binding_idx - sn_idx],
                [1, 0, binding_idx - sn_idx],
                [0, -1, binding_idx - sn_idx],
                [0, 1, binding_idx - sn_idx],
            ]
        )
        sp_relative_removals = sn_relative_removals.clone()
        sp_relative_removals[:, 2] = binding_idx - sp_idx

        X_tasks = [X_binding, X_sn, X_sp]
        relative_removals = [
            binding_relative_removals,
            sn_relative_removals,
            sp_relative_removals,
        ]
        for X_task, removals in zip(X_tasks, relative_removals):
            for xi in X_task:
                to_remove = xi + removals
                # remove all from X that are in to_remove
                mask = (X[:, None] == to_remove).all(-1).any(-1)
                train_masks[fold][mask] = False

    return train_masks


def prepare_nuclear_data(config: argparse.Namespace, recreate: bool = False):
    """Prepare data to be used for training. Transforms data to tensors, gets tokens X,targets y,
    vocab size and output map which is a dict of {target:output_shape}. Usually output_shape is 1 for regression
    and n_classes for classification.

    Args:
        columns (list, optional): List of columns to use as targets. Defaults to None.
        recreate (bool, optional): Force re-download of data and save to csv. Defaults to False.
    returns (Data): namedtuple of X, y, vocab_size, output_map, quantile_transformer
    """
    df = get_nuclear_data(recreate=recreate)
    df = df[
        (df.z > config.INCLUDE_NUCLEI_GT) & (df.n > config.INCLUDE_NUCLEI_GT)
    ]
    targets = get_targets(df)

    X = torch.tensor(targets[["z", "n"]].values)
    vocab_size = (
        targets.z.max() + 1,
        targets.n.max() + 1,
        len(config.TARGETS_CLASSIFICATION) + len(config.TARGETS_REGRESSION),
    )

    # classification targets increasing integers
    for col in config.TARGETS_CLASSIFICATION:
        targets[col] = targets[col].astype("category").cat.codes
        # put nans back
        targets[col] = targets[col].replace(-1, np.nan)

    output_map = OrderedDict()
    for target in config.TARGETS_CLASSIFICATION:
        output_map[target] = targets[target].nunique()

    for target in config.TARGETS_REGRESSION:
        output_map[target] = 1

    reg_columns = list(config.TARGETS_REGRESSION)
    # feature_transformer = QuantileTransformer(
    #     output_distribution="uniform", random_state=config.SEED
    # )
    feature_transformer = MinMaxScaler()
    if len(reg_columns) > 0:
        targets[reg_columns] = feature_transformer.fit_transform(
            targets[reg_columns].values
        )

    y = torch.tensor(targets[list(output_map.keys())].values).float()

    # Time to flatten everything
    X = torch.vstack(
        [
            torch.tensor([*x, task])
            for x in X
            for task in torch.arange(len(output_map))
        ]
    )
    y = y.flatten().view(-1, 1)
    train_masks, test_masks = _train_test_split(
        len(y), config.N_FOLDS, X, seed=config.SEED
    )

    # scale those by A
    binding_idxs = [
        i for i, x in enumerate(output_map.keys()) if "binding" in x
    ]
    scaled_idxs = [
        sum(list(output_map.values())[:idx]) for idx in binding_idxs
    ]

    # don't consider nuclei with high uncertainty in binding energy
    # but only for validation
    if config.TMS == "remove":
        binding_idx = list(output_map.keys()).index("binding_semf")
        radius_idx = list(output_map.keys()).index("radius")
        except_binding = (df.binding_unc * (df.z + df.n) > 100).values
        except_radius = (df.unc_r > 0.005).values
        test_masks[:, binding_idx :: len(output_map)] = (
            test_masks[:, binding_idx :: len(output_map)] & ~except_binding
        )
        test_masks[:, radius_idx :: len(output_map)] = (
            test_masks[:, radius_idx :: len(output_map)] & ~except_radius
        )
    elif config.TMS != "keep":
        raise ValueError(f"Unknown TMS {config.TMS}")

    train_masks = _leave_one_plus_four_out(
        X, train_masks, test_masks, output_map, seed=config.SEED
    )
    # remove everything with z, n < 9 from validation
    test_masks = test_masks & (X[:, :2] > 8).all(dim=1)

    return Data(
        X.to(config.DEV),
        y.to(config.DEV),
        vocab_size,
        output_map,
        feature_transformer,
        train_masks.to(config.DEV),
        test_masks.to(config.DEV),
        scaled_idxs,
    )

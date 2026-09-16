
import numpy as np
import pandas as pd

from config import Y0, YF


def _vy_at_plate(vy0: pd.Series, ay: pd.Series) -> pd.Series:
    return -np.sqrt(vy0 ** 2 - 2 * ay * (Y0 - YF))


def _time_to_plate(vy_f: pd.Series, vy0: pd.Series, ay: pd.Series) -> pd.Series:
    return (vy_f - vy0) / ay


def calculate_vaa(df: pd.DataFrame) -> pd.Series:
    vy_f = _vy_at_plate(df["vy0"], df["ay"])
    t = _time_to_plate(vy_f, df["vy0"], df["ay"])
    vz_f = df["vz0"] + df["az"] * t
    return -np.arctan(vz_f / vy_f) * (180.0 / np.pi)


def calculate_haa(df: pd.DataFrame) -> pd.Series:
    vy_f = _vy_at_plate(df["vy0"], df["ay"])
    t = _time_to_plate(vy_f, df["vy0"], df["ay"])
    vx_f = df["vx0"] + df["ax"] * t
    return -np.arctan(vx_f / vy_f) * (180.0 / np.pi)


def add_approach_angles(df: pd.DataFrame) -> pd.DataFrame:
    required = ["vy0", "vz0", "vx0", "ay", "az", "ax"]
    df = df.copy()

    valid = df[required].notna().all(axis=1)
    df.loc[valid, "vaa"] = calculate_vaa(df.loc[valid])
    df.loc[valid, "haa"] = calculate_haa(df.loc[valid])

    if "vaa" not in df.columns:
        df["vaa"] = np.nan
    if "haa" not in df.columns:
        df["haa"] = np.nan

    return df

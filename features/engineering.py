import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import EXCLUDE_PITCH_TYPES, MODEL_DIR, PITCH_TYPE_MODELS
from features.angles import add_approach_angles

logger = logging.getLogger(__name__)

FILL_FEATURES = [
    "release_speed",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "arm_angle",
    "release_spin_rate",
    "az",
    "plate_x",
    "plate_z",
]

REQUIRED_HARD = [
    "release_speed",
    "pfx_z",
    "pfx_x",
    "release_pos_x",
    "release_pos_z",
]

REQUIRED_FOR_ANGLES = [
    "release_extension",
    "vy0",
    "vz0",
    "vx0",
    "ay",
    "az",
    "ax",
    "release_spin_rate",
    "spin_axis",
]

_KNOWN_STRING_COLS = {
    "pitch_type", "player_name", "game_date", "game_type",
    "p_throws", "stand", "home_team", "away_team",
    "description", "events", "inning_topbot", "bb_type",
    "if_fielding_alignment", "of_fielding_alignment",
    "pitch_name", "des", "sv_id",
}

PITCHER_TYPE_OVERRIDES: dict[int, dict[str, str]] = {
    686790: {"CU": "SL"},
}

_AA_EST_PATH = os.path.join(MODEL_DIR, "arm_angle_estimator.pkl")
_AA_LOOKUP_PATH = os.path.join(MODEL_DIR, "arm_angle_lookup.pkl")


def clean_statcast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for _col in df.select_dtypes(include="object").columns:
        if _col not in _KNOWN_STRING_COLS:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")

    if "game_type" in df.columns:
        non_spring = df["game_type"] != "S"
        if non_spring.any():
            df = df[non_spring]

    df = df[df["pitch_type"].notna()]
    df = df[~df["pitch_type"].isin(EXCLUDE_PITCH_TYPES)]
    df = df[df["pitch_type"] != ""]

    df["pitch_type"] = df["pitch_type"].replace("FA", "FF")

    if "pitcher" in df.columns:
        for pid, remaps in PITCHER_TYPE_OVERRIDES.items():
            mask = df["pitcher"] == pid
            for old, new in remaps.items():
                df.loc[mask & (df["pitch_type"] == old), "pitch_type"] = new

    df = df.dropna(subset=REQUIRED_HARD)

    for col in REQUIRED_FOR_ANGLES:
        if col in df.columns and df[col].isna().any():
            fill = df.groupby("pitch_type")[col].transform("median")
            df[col] = df[col].fillna(fill).fillna(df[col].median())

    df = df[(df["release_speed"] >= 50) & (df["release_speed"] <= 110)]
    logger.info(f"After cleaning: {len(df):,} pitches")
    return df


def _estimated_arm_angle(df: pd.DataFrame) -> pd.Series:
    estimated = pd.Series(np.nan, index=df.index)
    if not os.path.exists(_AA_EST_PATH):
        return estimated
    with open(_AA_EST_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    feats = bundle["features"]
    heights = bundle["height_lookup"]
    height_default = float(np.nanmean(list(heights.values()))) if heights else 74.0
    if bundle.get("jmaschino"):
        enc = bundle.get("encoders", {}).get("p_throws", {"R": 0, "L": 1})
        X = pd.DataFrame(index=df.index)
        X["p_throws"] = df["p_throws"].map(enc).astype(float)
        X["height"] = df["pitcher"].map(heights).fillna(height_default).astype(float)
        X["release_extension"] = pd.to_numeric(df.get("release_extension"), errors="coerce")
        X["release_pos_z"] = pd.to_numeric(df.get("release_pos_z"), errors="coerce")
        valid = X[feats].notna().all(axis=1)
        if valid.any():
            raw = model.predict(X.loc[valid, feats].values)
            estimated[valid] = 90.0 - np.abs(raw)
    elif all(c in df.columns or c == "p_throws_r" or c == "height" for c in feats):
        X = pd.DataFrame(index=df.index)
        X["p_throws_r"] = (df["p_throws"] == "R").astype(float)
        X["height"] = df["pitcher"].map(heights).fillna(height_default).astype(float)
        for col in ["release_extension", "release_pos_x", "release_pos_z", "release_pos_y",
                    "vx0", "vz0", "vy0", "ax", "az", "spin_axis", "release_speed", "release_spin_rate"]:
            if col in df.columns:
                X[col] = pd.to_numeric(df[col], errors="coerce")
        model_feats = [f for f in feats if f in X.columns]
        valid = X[model_feats].notna().all(axis=1)
        if valid.any():
            estimated[valid] = model.predict(X.loc[valid, model_feats].values)
    return estimated


def _lookup_arm_angle(df: pd.DataFrame) -> pd.Series:
    if not (os.path.exists(_AA_LOOKUP_PATH) and "pitcher" in df.columns):
        return pd.Series(dtype=float, index=df.index)
    with open(_AA_LOOKUP_PATH, "rb") as f:
        lookup = pickle.load(f)
    if isinstance(lookup, dict) and "by_pitcher" in lookup:
        by_pt = lookup["by_pitcher_pitch_type"]
        by_p = lookup["by_pitcher"]
        if "pitch_type" in df.columns:
            out = df.apply(
                lambda r: by_pt.get((r["pitcher"], r["pitch_type"]), by_p.get(r["pitcher"], np.nan)), axis=1
            )
            return pd.Series(out.values, index=df.index, dtype=float)
        return df["pitcher"].map(by_p).astype(float)
    return df["pitcher"].map(lookup)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_statcast(df)

    df["pfx_z_in"] = df["pfx_z"] * 12.0
    df["pfx_x_in"] = df["pfx_x"] * 12.0

    if "p_throws" in df.columns:
        throw_sign = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(1.0)
        df["p_throws_r"] = (df["p_throws"] == "R").astype(float)
    else:
        throw_sign = pd.Series(1.0, index=df.index)
        df["p_throws_r"] = 1.0
    df["pfx_x_arm"] = df["pfx_x_in"] * throw_sign

    df = add_approach_angles(df)

    statcast_aa = df["arm_angle"] if "arm_angle" in df.columns else pd.Series(np.nan, index=df.index)
    estimated_aa = _estimated_arm_angle(df)
    lookup_aa = _lookup_arm_angle(df)
    df["arm_angle"] = (
        statcast_aa
        .where(statcast_aa.notna(), other=estimated_aa)
        .where(statcast_aa.notna() | estimated_aa.notna(), other=lookup_aa)
    )

    _vx0 = pd.to_numeric(df.get("vx0"), errors="coerce")
    _vy0 = pd.to_numeric(df.get("vy0"), errors="coerce")
    _vz0 = pd.to_numeric(df.get("vz0"), errors="coerce")
    _ax = pd.to_numeric(df.get("ax"), errors="coerce")
    _ay = pd.to_numeric(df.get("ay"), errors="coerce")
    _az = pd.to_numeric(df.get("az"), errors="coerce")
    _t = (-_vy0 - np.sqrt(_vy0 ** 2 - 2 * _ay * (50.0 - 17.0 / 12.0))) / _ay
    _vyf = _vy0 + _ay * _t
    df["VAA"] = -np.degrees(np.arctan2(_vz0 + _az * _t, np.abs(_vyf)))
    df["HAA"] = -np.degrees(np.arctan2(_vx0 + _ax * _t, np.abs(_vyf)))

    if "plate_x" in df.columns and "plate_z" in df.columns:
        sz_top = df.get("sz_top", pd.Series(3.5, index=df.index)).fillna(3.5)
        sz_bot = df.get("sz_bot", pd.Series(1.5, index=df.index)).fillna(1.5)
        df["in_zone"] = (
            (df["plate_x"].abs() <= 0.83) &
            (df["plate_z"] >= sz_bot) &
            (df["plate_z"] <= sz_top)
        ).astype(float)
    else:
        df["in_zone"] = 0.5

    pt_to_family = {pt: fam for fam, pts in PITCH_TYPE_MODELS.items() for pt in pts}
    df["family"] = df["pitch_type"].map(pt_to_family).fillna("ff")

    is_rhp = (df.get("p_throws", pd.Series("R", index=df.index)) == "R")
    pfx_x = df["pfx_x"].fillna(0.0)
    pfx_z = df["pfx_z"].fillna(0.0)
    pfx_x_arm = np.where(is_rhp, pfx_x, -pfx_x)
    movement_axis = (np.degrees(np.arctan2(pfx_x_arm, pfx_z)) % 360.0)
    spin_axis = df["spin_axis"].fillna(180.0) if "spin_axis" in df.columns else pd.Series(180.0, index=df.index)
    spin_axis_norm = np.where(is_rhp, spin_axis, (360.0 - spin_axis) % 360.0)
    diff = np.abs(spin_axis_norm - movement_axis) % 360.0
    df["deviation"] = np.minimum(diff, 360.0 - diff)

    for col in FILL_FEATURES:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0.0)

    logger.info(f"Feature engineering done. Shape: {df.shape}")
    return df

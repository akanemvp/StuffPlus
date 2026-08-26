"""
Feature engineering pipeline.

Steps:
  1. Clean raw Statcast DataFrame
  2. Convert movement to inches; compute acceleration-based movement
  3. Compute approach angles (VAA / HAA)
  4. Compute arm angle (Statcast native → lookup → formula fallback)
  5. Bin arm angle (for baselines merge)
  6. Add handedness encoding (same_hand, p_throws_r)
  7. Compute population baselines → SSW, ivb_adj, vaa_adj, haa_adj
  8. NaN-fill CORE_FEATURES

Returns (df_with_features, baselines_tuple).
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import PITCH_TYPES, ARM_ANGLE_BINS, EXCLUDE_PITCH_TYPES, MODEL_DIR
from features.angles import add_approach_angles
from features.movement import (
    calculate_arm_angle,
    compute_baselines,
    merge_deviations,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-group feature lists (data-driven from bucket-level whiff/xwoba corrs)
# ---------------------------------------------------------------------------
CORE_FEATURES = [
    # -- velocity --
    "release_speed",          # raw release speed (mph)
    # -- movement / approach angles --
    "vaa_adj",                # vertical approach angle (adjusted for location, release, arm slot, accels)
    "haa_adj",                # horizontal approach angle (adjusted for location, release, accels)
    # -- release point --
    "release_extension",      # extension toward home plate
    "release_pos_x",          # horizontal release position
    "release_pos_z",          # vertical release height
    "arm_angle",              # arm slot angle (native Statcast; estimator only fills nulls)
    # -- spin --
    "release_spin_rate",      # raw spin rate (RPM)
    # -- accelerations --
    "az",                     # raw vertical acceleration (includes gravity)
    "hb_accel_arm",           # horizontal accel, arm-side normalized (+ = arm side for both hands)
    # -- batter handedness (overridden 0/1 at inference for platoon averaging) --
    "same_hand",
    # -- location (engineered for completeness; the production Driveline model is
    #    pure-shape and does NOT use these — see SHAPE_FEATS in model/prob_resid.py) --
    "plate_x",                # horizontal plate location
    "plate_z",                # vertical plate location
]

# Differential features vs pitcher's primary fastball
DIFF_FEATURES = []

# Per-family / per-type aliases (all same for single global model)
_BASE_FEATURES = CORE_FEATURES
FB_FEATURES    = CORE_FEATURES
BR_FEATURES    = CORE_FEATURES
OS_FEATURES    = CORE_FEATURES

# Map from group key → feature list (kept for back-compat)
FEATURES_BY_TYPE: dict = {k: CORE_FEATURES for k in ("ff","si","fc","sl","st","cu","ch","fs")}

NEUTRAL_PLATE_X = 0.0
NEUTRAL_PLATE_Z = 2.3

MOVEMENT_GRID = 25   # bins per axis for movement value surface

# Features for which per-pitch-type z-scores can be computed (available for future use)
TYPE_RELATIVE_FEATURES = [
    "release_speed", "release_spin_rate", "active_spin_rate",
    "ivb_accel", "hb_accel_arm", "total_movement", "spd_from_fb",
    "gyro_degree", "vaa_adj", "haa_adj", "ssw_pfx_z", "ssw_pfx_x_arm",
]

# 8 separate per-pitch-type models
FEATURES_BY_GROUP: dict[str, list[str]] = {grp: CORE_FEATURES for grp in
    ["ff", "si", "fc", "sl", "st", "cu", "ch", "fs"]}

# Integer codes for pitch types — used as LightGBM categorical feature.
# Keeps Statcast granularity; model learns cross-type patterns globally.
PITCH_TYPE_MAP = {
    "FF": 0, "SI": 1, "FC": 2,
    "SL": 3, "ST": 4, "SV": 5, "SC": 5, "GY": 6,
    "CU": 7, "KC": 8, "CS": 9,
    "CH": 10, "FS": 11, "FO": 12,
    "KN": 13, "EP": 14,
}

# Keep MODEL_FEATURES as an alias for backward compatibility
MODEL_FEATURES = CORE_FEATURES

# ---------------------------------------------------------------------------
# Required raw columns
# ---------------------------------------------------------------------------
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

REQUIRED_RAW = REQUIRED_HARD


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop truly unscorable rows; soft-fill missing angle columns with medians.
    """
    df = df.copy()

    _KNOWN_STRING_COLS = {
        "pitch_type", "player_name", "game_date", "game_type",
        "p_throws", "stand", "home_team", "away_team",
        "description", "events", "inning_topbot", "bb_type",
        "if_fielding_alignment", "of_fielding_alignment",
        "pitch_name", "des", "sv_id",
    }
    for _col in df.select_dtypes(include="object").columns:
        if _col not in _KNOWN_STRING_COLS:
            df[_col] = pd.to_numeric(df[_col], errors="coerce")

    # Drop spring training rows only when mixed with regular season data
    if "game_type" in df.columns:
        non_spring = df["game_type"] != "S"
        if non_spring.any():
            df = df[non_spring]

    df = df[df["pitch_type"].notna()]
    df = df[~df["pitch_type"].isin(EXCLUDE_PITCH_TYPES)]
    df = df[df["pitch_type"] != ""]

    # Unify FA → FF
    df["pitch_type"] = df["pitch_type"].replace("FA", "FF")

    # Pitcher-specific overrides
    PITCHER_TYPE_OVERRIDES: dict[int, dict[str, str]] = {
        686790: {"CU": "SL"},
    }
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


# ---------------------------------------------------------------------------
# Fastball differential feature helpers
# ---------------------------------------------------------------------------

_FASTBALL_LOOKUP_PATH = os.path.join(MODEL_DIR, "fastball_lookup.pkl")
_SPIN_AXIS_LOOKUP_PATH = os.path.join(MODEL_DIR, "spin_axis_lookup.pkl")
_MOVEMENT_BASELINES_PATH = os.path.join(MODEL_DIR, "movement_baselines.pkl")


def compute_fastball_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-pitcher fastball differential features.

    For each pitcher, finds their primary fastball family (FF or SI, whichever
    has more pitches). Computes per-pitcher averages of speed, az, and ax for
    that family. Adds columns:
        speed_diff = avg_ff_speed - release_speed
        az_diff    = avg_ff_az - az
        ax_diff    = avg_ff_ax - ax

    Pitchers with no FF/SI get 0 for all three diffs.
    Saves per-pitcher lookup dict to model/artifacts/fastball_lookup.pkl.

    Returns the modified df.
    """
    df = df.copy()

    _FF_TYPES = {"FF", "SI"}
    lookup: dict = {}

    if "pitcher" in df.columns and "pitch_type" in df.columns:
        fb_mask = df["pitch_type"].isin(_FF_TYPES)
        fb_df = df[fb_mask].copy()

        if not fb_df.empty:
            # Count FF vs SI per pitcher to pick primary fastball family
            counts = (
                fb_df.groupby(["pitcher", "pitch_type"])
                .size()
                .reset_index(name="n")
            )
            # For each pitcher pick the pitch_type with the highest count
            primary_fb = counts.loc[counts.groupby("pitcher")["n"].idxmax()].set_index("pitcher")["pitch_type"]

            for pitcher_id, pt in primary_fb.items():
                sub = fb_df[(fb_df["pitcher"] == pitcher_id) & (fb_df["pitch_type"] == pt)]
                if sub.empty:
                    continue
                lookup[pitcher_id] = {
                    "avg_ff_speed":   float(sub["release_speed"].mean()),
                    "avg_ff_ax":      float(sub["ax"].mean())       if "ax"       in sub.columns else 0.0,
                    "avg_ff_ay":      float(sub["ay"].mean())       if "ay"       in sub.columns else 0.0,
                    "avg_ff_az":      float(sub["az"].mean())       if "az"       in sub.columns else 0.0,
                    "avg_ff_pfx_x":   float(sub["pfx_x_arm"].mean()) if "pfx_x_arm" in sub.columns else 0.0,
                    "avg_ff_pfx_z":   float(sub["pfx_z_in"].mean())  if "pfx_z_in"  in sub.columns else 0.0,
                    "avg_ff_vaa_adj": float(sub["vaa_adj"].mean())  if "vaa_adj"  in sub.columns else 0.0,
                    "avg_ff_haa_adj": float(sub["haa_adj"].mean())  if "haa_adj"  in sub.columns else 0.0,
                }

    # Save lookup
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(_FASTBALL_LOOKUP_PATH, "wb") as f:
        pickle.dump(lookup, f)
    logger.info(f"Fastball lookup saved for {len(lookup):,} pitchers → {_FASTBALL_LOOKUP_PATH}")

    df = apply_fastball_diffs(df, lookup)
    return df


def apply_fastball_diffs(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Apply a saved fastball lookup dict to compute speed_diff, az_diff, ax_diff.

    For pitchers in the lookup, diffs are (avg_ff_X - this_pitch_X).
    For unknown pitchers, diffs are 0.

    Parameters
    ----------
    df     : DataFrame with columns pitcher, release_speed, az, ax
    lookup : dict {pitcher_id: {avg_ff_speed, avg_ff_az, avg_ff_ax}}

    Returns the df with three new (or overwritten) columns.
    """
    df = df.copy()
    df["speed_diff"]    = 0.0
    df["ax_diff"]       = 0.0
    df["ay_diff"]       = 0.0
    df["az_diff"]       = 0.0
    df["pfx_x_diff"]    = 0.0
    df["pfx_z_diff"]    = 0.0
    df["vaa_adj_diff"]  = 0.0
    df["haa_adj_diff"]  = 0.0

    if "pitcher" not in df.columns:
        return df

    # Build on-the-fly fastball averages for pitchers missing from the lookup
    _FF_TYPES = {"FF", "SI"}
    if "pitch_type" in df.columns:
        fb_in_batch = df[df["pitch_type"].isin(_FF_TYPES)]
        if not fb_in_batch.empty:
            for pid, grp in fb_in_batch.groupby("pitcher"):
                if pid not in (lookup or {}):
                    if lookup is None:
                        lookup = {}
                    lookup[pid] = {
                        "avg_ff_speed":   float(grp["release_speed"].mean()),
                        "avg_ff_ax":      float(grp["ax"].mean())        if "ax"        in grp.columns else 0.0,
                        "avg_ff_ay":      float(grp["ay"].mean())        if "ay"        in grp.columns else 0.0,
                        "avg_ff_az":      float(grp["az"].mean())        if "az"        in grp.columns else 0.0,
                        "avg_ff_pfx_x":   float(grp["pfx_x_arm"].mean()) if "pfx_x_arm" in grp.columns else 0.0,
                        "avg_ff_pfx_z":   float(grp["pfx_z_in"].mean())  if "pfx_z_in"  in grp.columns else 0.0,
                        "avg_ff_vaa_adj": float(grp["vaa_adj"].mean())   if "vaa_adj"   in grp.columns else 0.0,
                        "avg_ff_haa_adj": float(grp["haa_adj"].mean())   if "haa_adj"   in grp.columns else 0.0,
                    }

    if not lookup:
        return df

    for pitcher_id, vals in lookup.items():
        mask = df["pitcher"] == pitcher_id
        if not mask.any():
            continue
        df.loc[mask, "speed_diff"]  = vals["avg_ff_speed"] - df.loc[mask, "release_speed"]
        if "ax" in df.columns:
            throw_sign = df.loc[mask, "p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0)
            df.loc[mask, "ax_diff"] = (vals["avg_ff_ax"] - df.loc[mask, "ax"]) * throw_sign
        if "ay" in df.columns:
            df.loc[mask, "ay_diff"] = vals.get("avg_ff_ay", 0.0) - df.loc[mask, "ay"]
        if "az" in df.columns:
            df.loc[mask, "az_diff"] = vals["avg_ff_az"] - df.loc[mask, "az"]
        if "pfx_x_arm" in df.columns:
            df.loc[mask, "pfx_x_diff"] = vals.get("avg_ff_pfx_x", 0.0) - df.loc[mask, "pfx_x_arm"]
        if "pfx_z_in" in df.columns:
            df.loc[mask, "pfx_z_diff"] = vals.get("avg_ff_pfx_z", 0.0) - df.loc[mask, "pfx_z_in"]
        if "vaa_adj" in df.columns:
            df.loc[mask, "vaa_adj_diff"] = vals.get("avg_ff_vaa_adj", 0.0) - df.loc[mask, "vaa_adj"]
        if "haa_adj" in df.columns:
            df.loc[mask, "haa_adj_diff"] = vals.get("avg_ff_haa_adj", 0.0) - df.loc[mask, "haa_adj"]

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    baselines=None,
) -> tuple:
    """
    Full feature engineering.

    Parameters
    ----------
    df        : raw Statcast DataFrame
    baselines : None (compute from df) OR saved (baselines_df, ff_ivb_coefs,
                arm_bin_edges) tuple from compute_baselines().

    Returns
    -------
    (df_engineered, (baselines_df, ff_ivb_coefs, arm_bin_edges))
    """
    ff_ivb_coefs   = None
    _arm_bin_edges = None
    if isinstance(baselines, tuple):
        if len(baselines) == 3:
            baselines, ff_ivb_coefs, _arm_bin_edges = baselines
        else:
            baselines, ff_ivb_coefs = baselines

    df = clean_statcast(df)

    # -- 1. Movement in inches
    df["pfx_z_in"] = df["pfx_z"] * 12.0
    df["pfx_x_in"] = df["pfx_x"] * 12.0

    # Spin efficiency / gyro decomposition.
    # gyro_deg = arccos(spin_eff): 0° = pure active spin, 90° = pure gyro.
    # Calibration constant 0.7729 (in × mph / RPM) derived from RHP 4-seam fastball population.
    # gyro_deg kept for rule-based cluster routing in SL/FC — NOT a model feature.
    # active_spin_rate and deception_z/x ARE model features.
    _SPIN_EFF_CALIB = 0.7729
    if "release_spin_rate" in df.columns and "release_speed" in df.columns:
        _pfx_mag     = np.sqrt(df["pfx_x_in"] ** 2 + df["pfx_z_in"] ** 2)
        _theoretical = _SPIN_EFF_CALIB * df["release_spin_rate"] / df["release_speed"].clip(lower=50.0)
        _spin_eff    = (_pfx_mag / _theoretical.clip(lower=1e-6)).clip(0.0, 1.0)
        df["gyro_degree"] = np.degrees(np.arccos(_spin_eff))
        df["spin_efficiency"] = _spin_eff
    else:
        _spin_eff    = pd.Series(1.0, index=df.index)
        df["gyro_degree"] = np.nan
        df["spin_efficiency"] = 1.0

    # Arm-side-positive convention (needed internally for SSW/ivb_adj)
    if "p_throws" in df.columns:
        throw_sign = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(1.0)
    else:
        throw_sign = pd.Series(1.0, index=df.index)
    df["pfx_x_arm"] = df["pfx_x_in"] * throw_sign

    # Active spin rate: the spin component actually generating Magnus force (rpm).
    # active_spin = spin_rate × spin_efficiency
    df["active_spin_rate"] = df["release_spin_rate"] * _spin_eff

    # Acceleration-based movement (needed for ivb_adj computation)
    df["ivb_accel"]    = df["az"] + 32.174
    df["ivb_accel_abs"] = df["ivb_accel"].abs()
    df["hb_accel_arm"] = df["ax"] * throw_sign
    df["hb_accel_arm_abs"] = df["hb_accel_arm"].abs()
    df["total_movement"] = np.sqrt(df["ivb_accel"] ** 2 + df["hb_accel_arm"] ** 2)

    # Deception features: how much did the pitch underdeliver vs what its spin promised?
    # gap_factor = (1/spin_eff - 1) scales the actual aerodynamic movement to the
    # "promised" level. High deception_z/x → pitch spun like it should break but
    # didn't (gyro trick). Scaled by acceleration-based movement (az, hb_accel_arm)
    # rather than time-integrated pfx (extension/velocity-independent).
    _gap_factor = (1.0 / _spin_eff.clip(lower=0.05)) - 1.0
    df["deception_z"] = df["az"] * _gap_factor
    df["deception_x"] = df["hb_accel_arm"] * _gap_factor

    # ── Seam-shifted wake (SSW) ───────────────────────────────────────────
    # spin_axis is the Hawkeye OBSERVED spin direction (carries the seam signal).
    # ALL data is now on one basis: statcast_search / statsapi (FF≈209) every year —
    # the earlier "2026 convention flip" was a data bug (2026 was wrongly pulled from
    # the savant /gf endpoint, FF≈161). See memory: stuff_plus_2026_spinaxis_bug.
    # Canonical Magnus-predicted break direction = (360 − spin_axis) − 180, applied
    # uniformly. SSW = movement component perpendicular to that direction:
    #   ssw_z: vertical residual   (+ = more rise than spin predicts, − = more sink)
    #   ssw_x: arm-side residual   (+ = more arm-side run, − = more glove-side)
    if "spin_axis" in df.columns and "ax" in df.columns and "az" in df.columns:
        _ivb = df["az"] + 32.174
        _sa  = pd.to_numeric(df["spin_axis"], errors="coerce")
        _sa_aligned = (360.0 - _sa) % 360                      # one basis, every year
        _spin_pred = np.radians(_sa_aligned - 180.0)            # spin-predicted break dir
        _ux, _uz = np.sin(_spin_pred), np.cos(_spin_pred)
        _Mx, _Mz = df["ax"].fillna(0.0), _ivb.fillna(0.0)
        _proj = _Mx * _ux + _Mz * _uz                           # Magnus-aligned magnitude
        df["ssw_z"] = (_Mz - _proj * _uz).fillna(0.0)           # vertical residual
        df["ssw_x"] = ((_Mx - _proj * _ux) * throw_sign).fillna(0.0)   # arm-side residual
    else:
        df["ssw_z"] = 0.0
        df["ssw_x"] = 0.0

    # ── Release angles — initial trajectory direction out of the hand ──────
    # The hitter extrapolates the ball's path from this launch direction.
    # Distinct from approach angle (direction at the plate) and arm_angle (slot).
    #   VRA = atan2(vz0, horizontal_speed)  — vertical launch (downhill−/uphill+)
    #   HRA = atan2(vx0, |vy0|) × throw_sign — horizontal launch, arm-side mirrored
    # The release→approach gap is the trajectory curvature perceived as break.
    if all(c in df.columns for c in ("vx0", "vy0", "vz0")):
        _vx0 = pd.to_numeric(df["vx0"], errors="coerce")
        _vy0 = pd.to_numeric(df["vy0"], errors="coerce")
        _vz0 = pd.to_numeric(df["vz0"], errors="coerce")
        _hspeed = np.sqrt(_vx0 ** 2 + _vy0 ** 2)
        df["vert_release_angle"] = np.degrees(np.arctan2(_vz0, _hspeed)).fillna(0.0)
        df["horz_release_angle"] = (np.degrees(np.arctan2(_vx0, _vy0.abs())) * throw_sign).fillna(0.0)
    else:
        df["vert_release_angle"] = 0.0
        df["horz_release_angle"] = 0.0

    # Arm-side-normalized release position (consistent convention across LHP/RHP)
    if "release_pos_x" in df.columns:
        df["release_pos_x_arm"] = df["release_pos_x"] * throw_sign

    # Arm-side-normalized spin axis: reflect LHP spin axis so equivalent pitch types
    # have the same spin_axis_arm value regardless of handedness.
    # LHP: (360 - spin_axis) % 360  |  RHP: spin_axis unchanged
    if "spin_axis" in df.columns:
        sa = pd.to_numeric(df["spin_axis"], errors="coerce")
        is_lhp = (df.get("p_throws", pd.Series("R", index=df.index)) == "L")
        df["spin_axis_arm"] = np.where(is_lhp, (360 - sa) % 360, sa)
        # Circular decomposition: avoids discontinuity at 0°/360° boundary
        _sa_rad = np.radians(df["spin_axis_arm"])
        df["spin_axis_arm_sin"] = np.sin(_sa_rad)
        df["spin_axis_arm_cos"] = np.cos(_sa_rad)
        # spin_axis_rel: peer-mean-normalized spin axis — robust to Statcast calibration
        # shifts between seasons (2026 reference frame changed ~60-200° by pitch type)
        #
        # During training (large batch): compute in-batch groupby means and save a lookup
        # keyed by (pitch_type, p_throws, game_year) + a year=0 fallback mean.
        # During inference (single pitcher): load saved lookup so the value doesn't
        # collapse to 0 (which happens when peer_mean is computed on just one pitcher).
        year_col = "game_year" if "game_year" in df.columns else None
        group_cols_yr = ["pitch_type", "p_throws"] + ([year_col] if year_col else [])
        valid_gc = [c for c in group_cols_yr if c in df.columns]

        if len(df) > 10_000 and valid_gc:
            # Training path: compute from full batch and save lookup
            peer_mean = df.groupby(valid_gc)["spin_axis_arm"].transform("mean")
            try:
                import pickle as _pkl
                _gm = df.groupby(valid_gc)["spin_axis_arm"].mean()
                _lookup: dict = {}
                for idx, val in _gm.items():
                    if len(valid_gc) >= 3:
                        pt, throws, yr = idx
                        _lookup[(str(pt), str(throws), int(yr))] = float(val)
                    elif len(valid_gc) == 2:
                        pt, throws = idx
                        _lookup[(str(pt), str(throws), 0)] = float(val)
                    else:
                        _lookup[(str(idx), "R", 0)] = float(val)
                # Also store year=0 as a multi-year fallback per (pt, throws)
                _fb = df.groupby(["pitch_type", "p_throws"])["spin_axis_arm"].mean()
                for (pt, throws), val in _fb.items():
                    _lookup[(str(pt), str(throws), 0)] = float(val)
                os.makedirs(MODEL_DIR, exist_ok=True)
                with open(_SPIN_AXIS_LOOKUP_PATH, "wb") as _fh:
                    _pkl.dump(_lookup, _fh)
            except Exception:
                pass
        else:
            # Inference path: load saved lookup and merge
            _spin_lookup: dict = {}
            if os.path.exists(_SPIN_AXIS_LOOKUP_PATH):
                try:
                    import pickle as _pkl
                    with open(_SPIN_AXIS_LOOKUP_PATH, "rb") as _fh:
                        _spin_lookup = _pkl.load(_fh)
                except Exception:
                    pass

            if _spin_lookup:
                _throws = df["p_throws"] if "p_throws" in df.columns else pd.Series("R", index=df.index)
                _yr = df[year_col].fillna(0).astype(int) if year_col else pd.Series(0, index=df.index)
                peer_mean = pd.Series(np.nan, index=df.index)
                for i in df.index:
                    pt = str(df.at[i, "pitch_type"])
                    th = str(_throws.at[i])
                    yr = int(_yr.at[i])
                    val = _spin_lookup.get((pt, th, yr), _spin_lookup.get((pt, th, 0), np.nan))
                    peer_mean.at[i] = val
            elif valid_gc:
                peer_mean = df.groupby(valid_gc)["spin_axis_arm"].transform("mean")
            else:
                peer_mean = df["spin_axis_arm"].mean()

        df["spin_axis_rel"] = df["spin_axis_arm"] - peer_mean.fillna(df["spin_axis_arm"])

    # -- 2. Approach angles
    df = add_approach_angles(df)

    # -- 3. Arm angle
    statcast_aa = df["arm_angle"] if "arm_angle" in df.columns else pd.Series(np.nan, index=df.index)

    # Per-pitch arm angle estimator: LightGBM trained on 2023-2025 Statcast data
    # using (p_throws, height, release_extension, release_pos_x, release_pos_z).
    # MAE ~3.9° vs Statcast ground truth. Gives pitch-by-pitch estimates that
    # reflect real within-start and year-to-year variation.
    _aa_est_path = os.path.join(MODEL_DIR, "arm_angle_estimator.pkl")
    estimated_aa = pd.Series(np.nan, index=df.index)
    if os.path.exists(_aa_est_path):
        import pickle as _pickle
        with open(_aa_est_path, "rb") as _f:
            _aa_bundle = _pickle.load(_f)
        _aa_model   = _aa_bundle["model"]
        _aa_feats   = _aa_bundle["features"]
        _aa_heights = _aa_bundle["height_lookup"]
        _height_default = float(np.nanmean(list(_aa_heights.values()))) if _aa_heights else 74.0
        if _aa_bundle.get("jmaschino"):
            # jmaschino56 arm-angle model. Features: [p_throws(R→0/L→1), height(in),
            # release_extension, release_pos_z]. Its output is on the COMPLEMENT
            # convention (0°=over-the-top, 90°=sidearm) and negates LHP, so we map to
            # the Statcast convention with: arm_angle = 90 − |pred|.
            _enc = _aa_bundle.get("encoders", {}).get("p_throws", {"R": 0, "L": 1})
            _Xj = pd.DataFrame(index=df.index)
            _Xj["p_throws"]          = df["p_throws"].map(_enc).astype(float)
            _Xj["height"]            = df["pitcher"].map(_aa_heights).fillna(_height_default).astype(float)
            _Xj["release_extension"] = pd.to_numeric(df.get("release_extension"), errors="coerce")
            _Xj["release_pos_z"]     = pd.to_numeric(df.get("release_pos_z"), errors="coerce")
            _valid = _Xj[_aa_feats].notna().all(axis=1)
            if _valid.any():
                _raw = _aa_model.predict(_Xj.loc[_valid, _aa_feats].values)
                estimated_aa[_valid] = 90.0 - np.abs(_raw)
        elif all(c in df.columns or c == "p_throws_r" or c == "height" for c in _aa_feats):
            _Xaa = pd.DataFrame(index=df.index)
            _Xaa["p_throws_r"]        = (df["p_throws"] == "R").astype(float)
            _Xaa["height"]            = df["pitcher"].map(_aa_heights).fillna(_height_default).astype(float)
            for _col in ["release_extension","release_pos_x","release_pos_z","release_pos_y",
                         "vx0","vz0","vy0","ax","az","spin_axis","release_speed","release_spin_rate"]:
                if _col in df.columns:
                    _Xaa[_col] = pd.to_numeric(df[_col], errors="coerce")
            # only keep rows where all model features are available
            _model_feats = [f for f in _aa_feats if f in _Xaa.columns]
            _valid = _Xaa[_model_feats].notna().all(axis=1)
            if _valid.any():
                estimated_aa[_valid] = _aa_model.predict(_Xaa.loc[_valid, _model_feats].values)

    # Pitcher-level lookup (2025 Statcast medians — last resort when release data missing)
    _aa_lookup_path = os.path.join(MODEL_DIR, "arm_angle_lookup.pkl")
    if os.path.exists(_aa_lookup_path) and "pitcher" in df.columns:
        import pickle as _pickle
        with open(_aa_lookup_path, "rb") as _f:
            _aa_lookup = _pickle.load(_f)
        if isinstance(_aa_lookup, dict) and "by_pitcher" in _aa_lookup:
            _aa_pt = _aa_lookup["by_pitcher_pitch_type"]
            _aa_p  = _aa_lookup["by_pitcher"]
            if "pitch_type" in df.columns:
                lookup_aa = df.apply(
                    lambda r: _aa_pt.get((r["pitcher"], r["pitch_type"]),
                              _aa_p.get(r["pitcher"], np.nan)), axis=1
                )
                lookup_aa = pd.Series(lookup_aa.values, index=df.index, dtype=float)
            else:
                lookup_aa = df["pitcher"].map(_aa_p).astype(float)
        else:
            lookup_aa = df["pitcher"].map(_aa_lookup)
    else:
        lookup_aa = pd.Series(dtype=float, index=df.index)

    # Fill chain: statcast → estimated (per-pitch model) → lookup (last resort)
    df["arm_angle"] = (
        statcast_aa
        .where(statcast_aa.notna(), other=estimated_aa)
        .where(statcast_aa.notna() | estimated_aa.notna(), other=lookup_aa)
    )

    # release_pos_z_resid: release height ABOVE/BELOW what the arm slot predicts.
    # Raw release_pos_z doubles as a velocity-like main effect (it over-rewards tall /
    # over-the-top arms — e.g. Fairbanks). Regressing it on arm_angle and keeping the
    # residual decouples "extra height beyond slot" from the part the slot already implies.
    # Fixed OLS fit on 2022-2024 training data:
    #   release_pos_z = 4.467 + 0.03416 * arm_angle   (R^2 = 0.667)
    # Constant transform → the feature means the same thing every year. arm_angle is used
    # only to build this residual; it is NOT itself a model feature.
    df["release_pos_z_resid"] = df["release_pos_z"] - (4.467 + 0.03416 * df["arm_angle"])

    # VAA / HAA: kinematic approach angles at the front of the plate (y = 17/12 ft),
    # computed from the release velocity vector only. Then "adjusted for location" —
    # VAA_adj regresses out plate_z, HAA_adj regresses out plate_x — isolating the
    # approach angle beyond what the pitch's vertical/horizontal target implies.
    # Fixed OLS coefficients (2022-24 training) so the feature means the same every year.
    # NOTE: VAA_adj/HAA_adj read plate_z/plate_x to compute (uses location at inference).
    _vx0 = pd.to_numeric(df.get("vx0"), errors="coerce")
    _vy0 = pd.to_numeric(df.get("vy0"), errors="coerce")
    _vz0 = pd.to_numeric(df.get("vz0"), errors="coerce")
    _ax  = pd.to_numeric(df.get("ax"),  errors="coerce")
    _ay  = pd.to_numeric(df.get("ay"),  errors="coerce")
    _az  = pd.to_numeric(df.get("az"),  errors="coerce")
    _t   = (-_vy0 - np.sqrt(_vy0**2 - 2 * _ay * (50.0 - 17.0/12.0))) / _ay
    _vyf = _vy0 + _ay * _t
    df["VAA"] = -np.degrees(np.arctan2(_vz0 + _az * _t, np.abs(_vyf)))
    df["HAA"] = -np.degrees(np.arctan2(_vx0 + _ax * _t, np.abs(_vyf)))
    df["VAA_adj"] = df["VAA"] - (9.99528 + -1.51533 * pd.to_numeric(df["plate_z"], errors="coerce"))
    df["HAA_adj"] = df["HAA"] - (-0.75574 + -1.45935 * pd.to_numeric(df["plate_x"], errors="coerce"))

    # arm_angle_dev: angular deviation of actual movement direction from arm slot
    # atan2(hb, ivb) gives the direction the ball actually moves
    # subtracting arm_angle gives how far off the "dead zone" axis it is
    actual_move_angle = np.degrees(np.arctan2(df["hb_accel_arm"], df["ivb_accel"]))
    raw_dev = actual_move_angle - df["arm_angle"].fillna(45.0)
    dev_rad = np.radians((raw_dev + 180) % 360 - 180)
    df["arm_angle_dev"] = (raw_dev + 180) % 360 - 180  # kept for reference
    df["arm_angle_dev_sin"] = np.sin(dev_rad)
    df["arm_angle_dev_cos"] = np.cos(dev_rad)
    # Signed magnitude of movement perpendicular to arm slot (ft/s²):
    # positive = above arm angle (riding up), negative = below (sinking)
    df["arm_angle_dev_magnitude"] = df["total_movement"] * np.sin(dev_rad)
    # Absolute perpendicular deviation — how much acceleration deviates from arm slot axis.
    # Uses acceleration-based total_movement (sqrt(ivb_accel² + hb_accel_arm²) ft/s²).
    # FB-family only feature: deceptive fastballs move unexpectedly relative to arm slot.
    df["lateral_deception_mag"] = np.abs(df["total_movement"] * np.sin(dev_rad))

    # -- 4. Bin arm angle (needed to merge baselines)
    # At inference (small batch), load saved training baselines so vaa_adj/haa_adj
    # are computed relative to the full-population regression, not just the mini-batch.
    # (Re-computing baselines on 100-200 inference pitches collapses vaa_adj → ~0
    # because the regression intercept absorbs the batch mean.)
    if baselines is None and len(df) < 10_000:
        _bl_path = _MOVEMENT_BASELINES_PATH
        if os.path.exists(_bl_path):
            try:
                import pickle as _pkl
                with open(_bl_path, "rb") as _fh:
                    _saved = _pkl.load(_fh)
                if isinstance(_saved, tuple) and len(_saved) == 3:
                    baselines, ff_ivb_coefs, _arm_bin_edges = _saved
                    logger.debug("Loaded saved movement baselines for inference.")
            except Exception as _e:
                logger.warning(f"Could not load movement baselines: {_e}")

    arm_median = df["arm_angle"].median()
    if _arm_bin_edges is not None:
        df["arm_angle_bin"] = pd.cut(
            df["arm_angle"].fillna(arm_median),
            bins=_arm_bin_edges,
            labels=False,
            include_lowest=True,
        ).astype("float")
    else:
        df["arm_angle_bin"] = pd.qcut(
            df["arm_angle"].fillna(arm_median),
            q=ARM_ANGLE_BINS,
            labels=False,
            duplicates="drop",
        )

    # -- 5. Handedness encoding
    if "p_throws" in df.columns:
        df["p_throws_r"] = (df["p_throws"] == "R").astype(float)
    else:
        df["p_throws_r"] = 1.0

    if "p_throws" in df.columns and "stand" in df.columns:
        df["same_hand"] = (df["p_throws"] == df["stand"]).astype(float)
    else:
        df["same_hand"] = 0.0

    # -- Arm-normalized (mirrored) horizontal features --------------------------
    # Raw ax / release_pos_x are sign-flipped between hands, so a model trained on
    # them has to learn the same physics twice — once per sign region — from data
    # that is ~73% right-handed. That gave lefties a ~4-5 point phantom bonus
    # (verified: mirroring a pitch changed its grade by +4.7 before this fix, 0.00
    # after). Mirroring puts both hands on one scale so arm-side is always positive
    # and one pattern is learned from 100% of the data. Same approach tjStuff+ uses.
    _hand_sign = np.where(df["p_throws"].astype(str) == "R", -1.0, 1.0) \
        if "p_throws" in df.columns else 1.0
    if "ax" in df.columns:
        df["ax_arm"] = pd.to_numeric(df["ax"], errors="coerce") * _hand_sign
    if "release_pos_x" in df.columns:
        df["release_pos_x_arm"] = pd.to_numeric(df["release_pos_x"], errors="coerce") * _hand_sign

    # induced-Magnus accel components (ind_vert / ind_horiz_arm) — model shape features
    if all(c in df.columns for c in ("vx0", "vy0", "vz0", "ax", "ay", "az", "p_throws")):
        from model.prob_resid import add_magnus
        add_magnus(df)

    # -- 5a. in_zone: binary strike-zone indicator (batter-specific via sz_top/sz_bot)
    if "plate_x" in df.columns and "plate_z" in df.columns:
        sz_top = df.get("sz_top", pd.Series(3.5, index=df.index)).fillna(3.5)
        sz_bot = df.get("sz_bot", pd.Series(1.5, index=df.index)).fillna(1.5)
        df["in_zone"] = (
            (df["plate_x"].abs() <= 0.83) &
            (df["plate_z"] >= sz_bot) &
            (df["plate_z"] <= sz_top)
        ).astype(float)
    else:
        df["in_zone"] = 0.5  # unknown → neutral

    # -- 5b. Perceived velocity — release_speed adjusted for extension
    if "release_speed" in df.columns and "release_extension" in df.columns:
        ext = df["release_extension"].fillna(df["release_extension"].median())
        df["perceived_velocity"] = df["release_speed"] * (60.5 / (60.5 - ext).clip(lower=50.0))
    else:
        df["perceived_velocity"] = df.get("release_speed", 90.0)

    # -- 5c. Spin axis sin/cos decomposition — avoids circular discontinuity
    if "spin_axis" in df.columns:
        spin_rad = np.radians(df["spin_axis"].fillna(180.0))
        df["spin_axis_sin"] = np.sin(spin_rad)
        df["spin_axis_cos"] = np.cos(spin_rad)
    else:
        df["spin_axis_sin"] = 0.0
        df["spin_axis_cos"] = -1.0

    # -- 6. Population baselines → SSW, ivb_adj, vaa_adj, haa_adj
    if baselines is None:
        baselines, ff_ivb_coefs, _arm_bin_edges = compute_baselines(df)

    df = merge_deviations(df, baselines)

    # Surprise total — Euclidean magnitude of movement deviation from expected
    # (sqrt(ivb_adj² + hb_adj²)). Captures how unexpected the pitch shape is
    # for this arm slot + extension, without duplicating directionality (ax/az).
    if "ivb_adj" in df.columns and "hb_adj" in df.columns:
        df["surprise_total"] = np.sqrt(df["ivb_adj"] ** 2 + df["hb_adj"] ** 2)
    else:
        df["surprise_total"] = 0.0

    # SSW total — combined seam-shifted wake magnitude
    if "ssw_pfx_z" in df.columns and "ssw_pfx_x_arm" in df.columns:
        df["ssw_total"] = np.sqrt(df["ssw_pfx_z"] ** 2 + df["ssw_pfx_x_arm"] ** 2)
    else:
        df["ssw_total"] = 0.0

    # Deadzone proximity — 2D Gaussian distance from FF movement at the same arm slot.
    # "Deadzone" = fastball shape. Pitches that look like fastballs to hitters score Z_s ≈ 1.0.
    # Pitches that escape the fastball movement cloud score Z_s ≈ 0.0.
    #
    # Uses raw ivb_accel and hb_accel_arm (not slot-normalized) compared to FF baseline
    # for the same p_throws × arm_angle_bin, normalized by FF std in each dimension:
    #   d_ivb = (ivb_accel - ff_ivb_mean) / ff_ivb_std
    #   d_hb  = (hb_accel_arm - ff_hb_mean) / ff_hb_std
    #   Z_s   = exp(-(d_ivb² + d_hb²) / 2)
    try:
        _bl_df = baselines[0] if isinstance(baselines, tuple) else baselines
        _ff_bl = (
            _bl_df[_bl_df["pitch_type"] == "FF"]
            [["p_throws", "arm_angle_bin", "ivb_mean", "ivb_std", "hb_mean", "hb_std"]]
            .rename(columns={"ivb_mean": "ff_ivb_mean", "ivb_std": "ff_ivb_std",
                             "hb_mean": "ff_hb_mean",  "hb_std": "ff_hb_std"})
        )
        _tmp = df.merge(_ff_bl, on=["p_throws", "arm_angle_bin"], how="left")
        _d_ivb = (_tmp["ivb_accel"] - _tmp["ff_ivb_mean"]) / _tmp["ff_ivb_std"].replace(0, np.nan)
        _d_hb  = (_tmp["hb_accel_arm"] - _tmp["ff_hb_mean"]) / _tmp["ff_hb_std"].replace(0, np.nan)
        df["deadzone_proximity"] = np.exp(-(_d_ivb**2 + _d_hb**2) / 2).fillna(0.5).values
    except Exception:
        df["deadzone_proximity"] = 0.5

    # -- 7. Add family column (lowercase pitch family key)
    if "pitch_type" in df.columns:
        from config import PITCH_TYPE_MODELS as _PTM
        _pt_to_family = {pt: fam for fam, pts in _PTM.items() for pt in pts}
        df["family"] = df["pitch_type"].map(_pt_to_family).fillna("ff")
    else:
        df["family"] = "ff"

    # -- 7b. Pitch type code (categorical feature for single-model LightGBM)
    df["pitch_type_code"] = df["pitch_type"].map(PITCH_TYPE_MAP).fillna(15).astype(int)

    # -- 7b1. Pitch family code: 0=fb, 1=br, 2=os
    _FB_PTS = {"FF", "FA", "SI", "FC"}
    _BR_PTS = {"SL", "SV", "ST", "SC", "GY", "CU", "KC", "CS"}
    _OS_PTS = {"CH", "FO", "EP", "KN", "FS"}
    _fam_map = {**{pt: 0 for pt in _FB_PTS}, **{pt: 1 for pt in _BR_PTS}, **{pt: 2 for pt in _OS_PTS}}
    df["pitch_family_code"] = df["pitch_type"].map(_fam_map).fillna(0).astype(int)

    # -- 3D total movement: full acceleration magnitude across all axes
    for _c in ["hb_accel_arm", "ay", "az"]:
        if _c not in df.columns:
            df[_c] = 0.0
    df["total_movement_3d"] = np.sqrt(df["hb_accel_arm"]**2 + df["az"]**2)

    # -- 7b2. Spin features (spin_axis_arm_sin/cos, gyro_degree, deviation)
    is_rhp = (df.get("p_throws", pd.Series("R", index=df.index)) == "R")
    # Arm-normalized horizontal release point (consistent with hb_accel_arm / ssw_x:
    # ×+1 for RHP, ×−1 for LHP) so an arm-side release maps to the same value for
    # both hands instead of being mirrored.
    if "release_pos_x" in df.columns:
        df["release_pos_x_arm"] = np.where(is_rhp, df["release_pos_x"],
                                           -pd.to_numeric(df["release_pos_x"], errors="coerce"))
    if "spin_axis_arm_sin" not in df.columns or "spin_axis_arm_cos" not in df.columns:
        spin_axis = df["spin_axis"].fillna(180.0) if "spin_axis" in df.columns else pd.Series(180.0, index=df.index)
        spin_axis_norm = np.where(is_rhp, spin_axis, (360.0 - spin_axis) % 360.0)
        theta = np.radians(spin_axis_norm)
        df["spin_axis_arm_sin"] = np.sin(theta)
        df["spin_axis_arm_cos"] = np.cos(theta)
    if "deviation" not in df.columns:
        pfx_x = df["pfx_x"].fillna(0.0) if "pfx_x" in df.columns else pd.Series(0.0, index=df.index)
        pfx_z = df["pfx_z"].fillna(0.0) if "pfx_z" in df.columns else pd.Series(0.0, index=df.index)
        pfx_x_arm = np.where(is_rhp, pfx_x, -pfx_x)
        movement_axis = (np.degrees(np.arctan2(pfx_x_arm, pfx_z)) % 360.0)
        spin_axis = df["spin_axis"].fillna(180.0) if "spin_axis" in df.columns else pd.Series(180.0, index=df.index)
        spin_axis_norm = np.where(is_rhp, spin_axis, (360.0 - spin_axis) % 360.0)
        diff = np.abs(spin_axis_norm - movement_axis) % 360.0
        df["deviation"] = np.minimum(diff, 360.0 - diff)
    if "gyro_degree" not in df.columns:
        df["gyro_degree"] = 45.0  # fallback neutral value

    # -- 7c. Active spin rate = spin_rate × cos(gyro_degree)
    #        gyro_degree = arccos(active_spin_fraction), so cos(gyro_degree) = active_spin_fraction
    #        Requires gyro_degree; falls back to spin_rate if unavailable.
    if "gyro_degree" in df.columns:
        df["active_spin_rate"] = df["release_spin_rate"] * np.cos(np.radians(df["gyro_degree"].clip(0, 90)))
    elif "release_spin_rate" in df.columns:
        df["active_spin_rate"] = df["release_spin_rate"]  # fallback

    # -- 8. Stub differential columns at 0.0 — will be overwritten by compute_fastball_diffs()
    #       in train_all() and apply_fastball_diffs() in predict(). NaN-fill below won't
    #       override 0.0 values.
    if "speed_diff"  not in df.columns: df["speed_diff"]  = 0.0
    if "ax_diff"     not in df.columns: df["ax_diff"]     = 0.0
    if "ay_diff"     not in df.columns: df["ay_diff"]     = 0.0
    if "az_diff"     not in df.columns: df["az_diff"]     = 0.0
    df["ax_diff_abs"] = df["ax_diff"].abs()
    df["az_diff_abs"] = df["az_diff"].abs()
    if "pfx_x_diff"  not in df.columns: df["pfx_x_diff"]  = 0.0
    if "pfx_z_diff"  not in df.columns: df["pfx_z_diff"]  = 0.0

    # -- 9. Perceived velocity (batter's perceived speed, accounting for extension)
    if "release_speed" in df.columns and "release_extension" in df.columns:
        df["perceived_velocity"] = df["release_speed"] * (60.5 / (60.5 - df["release_extension"].clip(upper=9.0)).clip(lower=1.0))

    # -- 9b. Plate speed from kinematics (actual ball speed at home plate)
    # Statcast vy0/vx0/vz0 measured at y=50 ft; home plate is at y=1.417 ft.
    # Solve: 1.417 = 50 + vy0*t + 0.5*ay*t^2  →  quadratic in t
    # Then: v_plate = sqrt(vx_t^2 + vy_t^2 + vz_t^2), convert ft/s → mph
    _kin_cols = ["vx0", "vy0", "vz0", "ax", "ay", "az"]
    if all(c in df.columns for c in _kin_cols):
        _vy0 = pd.to_numeric(df["vy0"], errors="coerce")
        _ay  = pd.to_numeric(df["ay"],  errors="coerce")
        _a   = 0.5 * _ay
        _b   = _vy0
        _c   = 50.0 - 1.417   # 48.583 ft to travel
        _disc = _b**2 - 4.0 * _a * (-_c)
        # Take the positive, smaller root (first crossing of home plate)
        _t_plate = (-_b - np.sqrt(_disc.clip(lower=0.0))) / (2.0 * _a.replace(0, np.nan))
        _t_plate = _t_plate.clip(lower=0.0)
        _vx_t = pd.to_numeric(df["vx0"], errors="coerce") + pd.to_numeric(df["ax"], errors="coerce") * _t_plate
        _vy_t = _vy0 + _ay * _t_plate
        _vz_t = pd.to_numeric(df["vz0"], errors="coerce") + pd.to_numeric(df["az"], errors="coerce") * _t_plate
        _speed_fps = np.sqrt(_vx_t**2 + _vy_t**2 + _vz_t**2)
        df["plate_speed"] = (_speed_fps * 3600.0 / 5280.0).clip(40.0, 110.0)
    else:
        df["plate_speed"] = df.get("release_speed", pd.Series(90.0, index=df.index))

    # -- 9c. pfx_total: total movement magnitude in inches
    if "pfx_x_in" in df.columns and "pfx_z_in" in df.columns:
        df["pfx_total"] = np.sqrt(df["pfx_x_in"]**2 + df["pfx_z_in"]**2)
    elif "pfx_x" in df.columns and "pfx_z" in df.columns:
        df["pfx_total"] = np.sqrt(df["pfx_x"]**2 + df["pfx_z"]**2) * 12.0
    else:
        df["pfx_total"] = 0.0

    # -- 9d. release_speed_drop: how much the ball decelerates from release to plate
    df["release_speed_drop"] = df["release_speed"] - df["plate_speed"]

    # -- 10a. Derived features from baselines output
    # spin_efficiency: fraction of spin that is "active" (non-gyroscopic)
    if "active_spin_rate" in df.columns and "release_spin_rate" in df.columns:
        df["spin_efficiency"] = (
            df["active_spin_rate"] / (df["release_spin_rate"].replace(0, np.nan) + 1e-8)
        ).clip(0.0, 1.0)
    else:
        df["spin_efficiency"] = 0.5

    # ivb_accel_adj / hb_accel_arm_adj: arm-slot-adjusted movement z-scores
    # ivb_adj and hb_adj are computed by merge_deviations keyed on
    # pitch_type × p_throws × arm_angle_bin — handedness already accounted for.
    # hb_adj uses hb_accel_arm (arm-side-positive) so sign is correct for both hands.
    df["ivb_accel_adj"]    = df.get("ivb_adj", pd.Series(0.0, index=df.index))
    df["hb_accel_arm_adj"] = df.get("hb_adj",  pd.Series(0.0, index=df.index))

    # accel_arm_angle_dev: angle (degrees) between actual acceleration vector and arm angle.
    # Measures how much the pitch deviates from the expected arm-slot direction.
    # Positive = more arm-side than expected, negative = more vertical (unexpected ride/dive).
    if "arm_angle" in df.columns and "ivb_accel" in df.columns and "hb_accel_arm" in df.columns:
        move_angle = np.degrees(np.arctan2(df["hb_accel_arm"], df["ivb_accel"]))
        df["accel_arm_angle_dev"] = move_angle - df["arm_angle"]
    else:
        df["accel_arm_angle_dev"] = 0.0

    # -- 9e. Dynamic dead-zone deviation (release-conditioned expected fastball
    #        movement; deviation = how much the pitch defies its arm slot). Uses
    #        arm_angle / scaled extension + actual accel — never the pitch label.
    try:
        from model.deadzone import apply_deadzone
        df = apply_deadzone(df)
    except Exception as _dz_exc:
        logger.warning(f"dead-zone feature skipped ({_dz_exc})")
        for _f in ("deadzone_hb_dev", "deadzone_vert_dev", "deadzone_dist"):
            if _f not in df.columns:
                df[_f] = 0.0

    # -- 10. NaN-fill CORE_FEATURES
    for col in CORE_FEATURES:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0.0)

    logger.info(f"Feature engineering done. Shape: {df.shape}")
    return df, (baselines, ff_ivb_coefs, _arm_bin_edges)


# Primary fastball types used for fb context features
_FB_CONTEXT_TYPES = {"FF", "FA", "SI"}


def apply_fb_context(df: pd.DataFrame, lookup: dict = None) -> pd.DataFrame:
    """Add FB context features and movement/release deltas vs pitcher's own fastball.

    Adds: fb_ivb_adj, fb_hb_adj, fb_velo, fb_rel_x, fb_rel_z,
          ivb_vs_fb, hb_vs_fb, fb_rel_x_diff, fb_rel_z_diff
    """
    df = df.copy()

    if lookup is None:
        fb = df[df["pitch_type"].isin(_FB_CONTEXT_TYPES)].copy()
        lookup = {}
        if not fb.empty and "pitcher" in fb.columns:
            counts = fb.groupby(["pitcher", "pitch_type"]).size().reset_index(name="n")
            primary = counts.loc[counts.groupby("pitcher")["n"].idxmax()].set_index("pitcher")["pitch_type"]
            for pid, pt in primary.items():
                sub = fb[(fb["pitcher"] == pid) & (fb["pitch_type"] == pt)]
                if sub.empty:
                    continue
                lookup[pid] = {
                    "fb_ivb_adj": float(sub["ivb_accel"].mean())        if "ivb_accel"        in sub.columns else 0.0,
                    "fb_hb_adj":  float(sub["hb_accel_arm"].mean())     if "hb_accel_arm"     in sub.columns else 0.0,
                    "fb_velo":    float(sub["release_speed"].mean()),
                    "fb_rel_x":   float(sub["release_pos_x_arm"].mean()) if "release_pos_x_arm" in sub.columns else 0.0,
                    "fb_rel_z":   float(sub["release_pos_z"].mean())     if "release_pos_z"     in sub.columns else 0.0,
                }
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "fb_context_lookup.pkl"), "wb") as f:
            pickle.dump(lookup, f)
        logger.info(f"FB context lookup saved for {len(lookup):,} pitchers")

    # Global fallbacks
    all_fb = df[df["pitch_type"].isin(_FB_CONTEXT_TYPES)]
    global_ivb   = float(all_fb["ivb_accel"].mean())         if not all_fb.empty and "ivb_accel"         in all_fb.columns else 0.0
    global_hb    = float(all_fb["hb_accel_arm"].mean())      if not all_fb.empty and "hb_accel_arm"      in all_fb.columns else 0.0
    global_velo  = float(all_fb["release_speed"].mean())     if not all_fb.empty else 93.0
    global_rel_x = float(all_fb["release_pos_x_arm"].mean()) if not all_fb.empty and "release_pos_x_arm" in all_fb.columns else 1.9
    global_rel_z = float(all_fb["release_pos_z"].mean())     if not all_fb.empty and "release_pos_z"     in all_fb.columns else 5.8

    # For pitchers not in the lookup, compute from current batch
    if "pitcher" in df.columns and not all_fb.empty:
        missing = set(df["pitcher"].unique()) - set(lookup.keys())
        if missing:
            batch_fb = all_fb[all_fb["pitcher"].isin(missing)]
            if not batch_fb.empty:
                counts = batch_fb.groupby(["pitcher", "pitch_type"]).size().reset_index(name="n")
                primary = counts.loc[counts.groupby("pitcher")["n"].idxmax()].set_index("pitcher")["pitch_type"]
                for pid, pt in primary.items():
                    sub = batch_fb[(batch_fb["pitcher"] == pid) & (batch_fb["pitch_type"] == pt)]
                    if sub.empty:
                        continue
                    lookup[pid] = {
                        "fb_ivb_adj": float(sub["ivb_accel"].mean())        if "ivb_accel"        in sub.columns else global_ivb,
                        "fb_hb_adj":  float(sub["hb_accel_arm"].mean())     if "hb_accel_arm"     in sub.columns else global_hb,
                        "fb_velo":    float(sub["release_speed"].mean()),
                        "fb_rel_x":   float(sub["release_pos_x_arm"].mean()) if "release_pos_x_arm" in sub.columns else global_rel_x,
                        "fb_rel_z":   float(sub["release_pos_z"].mean())     if "release_pos_z"     in sub.columns else global_rel_z,
                    }

    if "pitcher" in df.columns:
        df["fb_ivb_adj"] = df["pitcher"].map(lambda p: lookup.get(p, {}).get("fb_ivb_adj", np.nan))
        df["fb_hb_adj"]  = df["pitcher"].map(lambda p: lookup.get(p, {}).get("fb_hb_adj",  np.nan))
        df["fb_velo"]    = df["pitcher"].map(lambda p: lookup.get(p, {}).get("fb_velo",    np.nan))
        df["fb_rel_x"]   = df["pitcher"].map(lambda p: lookup.get(p, {}).get("fb_rel_x",   np.nan))
        df["fb_rel_z"]   = df["pitcher"].map(lambda p: lookup.get(p, {}).get("fb_rel_z",   np.nan))
    else:
        df["fb_ivb_adj"] = np.nan
        df["fb_hb_adj"]  = np.nan
        df["fb_velo"]    = np.nan
        df["fb_rel_x"]   = np.nan
        df["fb_rel_z"]   = np.nan

    df["fb_ivb_adj"] = df["fb_ivb_adj"].fillna(global_ivb)
    df["fb_hb_adj"]  = df["fb_hb_adj"].fillna(global_hb)
    df["fb_velo"]    = df["fb_velo"].fillna(global_velo)
    df["fb_rel_x"]   = df["fb_rel_x"].fillna(global_rel_x)
    df["fb_rel_z"]   = df["fb_rel_z"].fillna(global_rel_z)

    # Movement deltas vs own fastball
    df["ivb_vs_fb"] = df.get("ivb_accel",         pd.Series(0.0, index=df.index)) - df["fb_ivb_adj"]
    df["hb_vs_fb"]  = df.get("hb_accel_arm",       pd.Series(0.0, index=df.index)) - df["fb_hb_adj"]

    # Release point deltas vs own fastball
    df["fb_rel_x_diff"] = df.get("release_pos_x_arm", pd.Series(0.0, index=df.index)) - df["fb_rel_x"]
    df["fb_rel_z_diff"] = df.get("release_pos_z",      pd.Series(0.0, index=df.index)) - df["fb_rel_z"]

    return df, lookup


def add_type_relative_features(df: pd.DataFrame, type_stats: dict = None) -> tuple:
    """Add per-pitch-type z-scored versions of key features.

    For each feature in TYPE_RELATIVE_FEATURES, computes:
        {feature}_vt = (value - pitch_type_mean) / pitch_type_std

    Parameters
    ----------
    df         : DataFrame after engineer_features (must have pitch_type + base features)
    type_stats : pre-computed dict {pitch_type: {feature: {mean, std}}}
                 If None, compute from df (training mode) and return.

    Returns
    -------
    (df_with_vt_features, type_stats)
    """
    df = df.copy()

    if type_stats is None:
        type_stats = {}
        for pt in df["pitch_type"].dropna().unique():
            pt_df = df[df["pitch_type"] == pt]
            type_stats[pt] = {}
            for feat in TYPE_RELATIVE_FEATURES:
                if feat not in pt_df.columns:
                    continue
                v = pt_df[feat].dropna()
                if len(v) < 10:
                    continue
                type_stats[pt][feat] = {
                    "mean": float(v.mean()),
                    "std":  float(v.std() + 1e-8),
                }

    # Global fallback stats
    global_stats = {}
    for feat in TYPE_RELATIVE_FEATURES:
        if feat in df.columns:
            v = df[feat].dropna()
            global_stats[feat] = {"mean": float(v.mean()), "std": float(v.std() + 1e-8)}

    for feat in TYPE_RELATIVE_FEATURES:
        if feat not in df.columns:
            continue
        new_col = f"{feat}_vt"
        z = np.zeros(len(df))
        for pt, stats in type_stats.items():
            if feat not in stats:
                continue
            mask = (df["pitch_type"] == pt).values
            if not mask.any():
                continue
            m, s = stats[feat]["mean"], stats[feat]["std"]
            z[mask] = (df[feat].values[mask] - m) / s
        # Unknown pitch types → global z-score
        unknown = ~df["pitch_type"].isin(type_stats.keys())
        if unknown.any():
            gm = global_stats.get(feat, {}).get("mean", 0.0)
            gs = global_stats.get(feat, {}).get("std", 1.0)
            z[unknown.values] = (df.loc[unknown, feat].values - gm) / gs
        df[new_col] = z

    return df, type_stats


def build_movement_rv_lookup(df: pd.DataFrame, rv_col: str = "residual_xrv") -> dict:
    """Build per-pitch-type, per-velo-tier 2D movement value surface.

    Grids (ivb_accel_adj × hb_accel_arm_adj) × mean(rv_col) per (pitch_type, velo_tier).
    Velo tiers (0=low, 1=mid, 2=high) split by per-pitch-type velocity tertiles so that
    e.g. an 82 mph splitter is compared to other 82 mph splitters, not 98 mph splitters.

    Parameters
    ----------
    df     : DataFrame with ivb_accel_adj, hb_accel_arm_adj, release_speed, pitch_type, and rv_col
    rv_col : column of actual run value targets
    """
    from scipy.ndimage import gaussian_filter

    lookup = {}
    G = MOVEMENT_GRID
    N_TIERS = 3

    for pt in sorted(df["pitch_type"].dropna().unique()):
        sub = df[df["pitch_type"] == pt].dropna(
            subset=["ivb_accel_adj", "hb_accel_arm_adj", "release_speed", rv_col]
        )
        if len(sub) < 200:
            continue

        # Per-pitch-type velo tertile boundaries
        velo_breaks = np.percentile(sub["release_speed"], [33.3, 66.7])
        sub = sub.copy()
        sub["velo_tier"] = np.digitize(sub["release_speed"].values, velo_breaks)  # 0, 1, 2

        for tier in range(N_TIERS):
            tier_sub = sub[sub["velo_tier"] == tier]
            if len(tier_sub) < 50:
                # Fall back to full-type surface for sparse tiers
                tier_sub = sub
            key = (pt, tier)

            ivb = tier_sub["ivb_accel_adj"].values
            hb  = tier_sub["hb_accel_arm_adj"].values
            rv  = tier_sub[rv_col].values

            ivb_lo, ivb_hi = np.percentile(ivb, 1), np.percentile(ivb, 99)
            hb_lo,  hb_hi  = np.percentile(hb,  1), np.percentile(hb,  99)

            ivb_edges = np.linspace(ivb_lo, ivb_hi, G + 1)
            hb_edges  = np.linspace(hb_lo,  hb_hi,  G + 1)

            grid_sum = np.zeros((G, G))
            grid_cnt = np.zeros((G, G))
            ivb_idx = np.clip(np.digitize(ivb, ivb_edges) - 1, 0, G - 1)
            hb_idx  = np.clip(np.digitize(hb,  hb_edges)  - 1, 0, G - 1)
            np.add.at(grid_sum, (ivb_idx, hb_idx), rv)
            np.add.at(grid_cnt, (ivb_idx, hb_idx), 1)

            with np.errstate(invalid="ignore"):
                grid_mean = np.where(grid_cnt >= 3, grid_sum / grid_cnt, np.nan)

            global_fill = np.nanmean(grid_mean)
            filled   = np.where(np.isnan(grid_mean), global_fill, grid_mean)
            smoothed = gaussian_filter(filled, sigma=1.5)

            lookup[key] = {
                "grid":        smoothed,
                "ivb_edges":   ivb_edges,
                "hb_edges":    hb_edges,
                "ivb_lo": ivb_lo, "ivb_hi": ivb_hi,
                "hb_lo":  hb_lo,  "hb_hi":  hb_hi,
                "velo_breaks": velo_breaks,
            }

        logger.debug(f"  movement_rv surface [{pt}]: n={len(sub):,}  tiers={N_TIERS}")

    logger.info(f"Movement_rv lookup built for {len(lookup)} (pitch_type, velo_tier) keys")
    return lookup


def apply_movement_rv(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Add movement_rv feature via bilinear interpolation on the per-type, per-velo-tier surface.

    Parameters
    ----------
    df     : DataFrame with pitch_type, release_speed, ivb_accel_adj, hb_accel_arm_adj
    lookup : dict returned by build_movement_rv_lookup()  keys are (pitch_type, velo_tier)

    Returns df with new column 'movement_rv' (0.0 for unknown pitch types).
    """
    from scipy.interpolate import RegularGridInterpolator

    df = df.copy()
    mv = np.zeros(len(df))

    # Get all pitch types present in lookup
    pts_in_lookup = set(k[0] for k in lookup.keys())

    # Pitch types that should use another type's movement surface when missing
    _MOVEMENT_FALLBACK = {"FO": "FS", "SC": "SL", "GY": "SL", "SV": "SL", "CS": "CU"}

    for pt in df["pitch_type"].dropna().unique():
        pt_mask = (df["pitch_type"] == pt).values
        if not pt_mask.any():
            continue
        lookup_pt = pt if pt in pts_in_lookup else _MOVEMENT_FALLBACK.get(pt)
        if lookup_pt is None or lookup_pt not in pts_in_lookup:
            continue

        # Determine velo_breaks from tier 0 (all tiers share the same breaks)
        velo_breaks = lookup.get((lookup_pt, 0), {}).get("velo_breaks", None)
        if velo_breaks is None:
            continue

        velo_vals = df.loc[pt_mask, "release_speed"].fillna(
            df.loc[pt_mask, "release_speed"].median()
        ).values
        tiers = np.digitize(velo_vals, velo_breaks)  # 0, 1, 2

        for tier in range(3):
            tier_row_mask = pt_mask.copy()
            tier_row_mask[pt_mask] = (tiers == tier)
            if not tier_row_mask.any():
                continue

            surf = lookup.get((lookup_pt, tier)) or lookup.get((lookup_pt, 0))
            if surf is None:
                continue

            ivb_centers = 0.5 * (surf["ivb_edges"][:-1] + surf["ivb_edges"][1:])
            hb_centers  = 0.5 * (surf["hb_edges"][:-1]  + surf["hb_edges"][1:])

            interp = RegularGridInterpolator(
                (ivb_centers, hb_centers),
                surf["grid"],
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

            ivb_vals = np.clip(df.loc[tier_row_mask, "ivb_accel_adj"].fillna(0).values,
                               surf["ivb_lo"], surf["ivb_hi"])
            hb_vals  = np.clip(df.loc[tier_row_mask, "hb_accel_arm_adj"].fillna(0).values,
                               surf["hb_lo"],  surf["hb_hi"])

            mv[tier_row_mask] = interp(np.column_stack([ivb_vals, hb_vals]))

    # Unknown pitch types → global mean of all surface values
    unknown = ~df["pitch_type"].isin(pts_in_lookup)
    if unknown.any() and lookup:
        all_vals = np.concatenate([s["grid"].ravel() for s in lookup.values()])
        mv[unknown.values] = float(np.nanmean(all_vals))

    df["movement_rv"] = mv
    return df

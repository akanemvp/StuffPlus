
import logging
import os

import numpy as np
import pandas as pd

from config import ARM_ANGLE_BINS, EXCLUDE_PITCH_TYPES, MODEL_DIR
from features.angles import add_approach_angles
from features.movement import compute_baselines, merge_deviations

logger = logging.getLogger(__name__)

CORE_FEATURES = [
    "release_speed",
    "vaa_adj",
    "haa_adj",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
    "arm_angle",
    "release_spin_rate",
    "az",
    "hb_accel_arm",
    "same_hand",
    "plate_x",
    "plate_z",
]


PITCH_TYPE_MAP = {
    "FF": 0, "SI": 1, "FC": 2,
    "SL": 3, "ST": 4, "SV": 5, "SC": 5, "GY": 6,
    "CU": 7, "KC": 8, "CS": 9,
    "CH": 10, "FS": 11, "FO": 12,
    "KN": 13, "EP": 14,
}


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


def clean_statcast(df: pd.DataFrame) -> pd.DataFrame:
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

    if "game_type" in df.columns:
        non_spring = df["game_type"] != "S"
        if non_spring.any():
            df = df[non_spring]

    df = df[df["pitch_type"].notna()]
    df = df[~df["pitch_type"].isin(EXCLUDE_PITCH_TYPES)]
    df = df[df["pitch_type"] != ""]

    df["pitch_type"] = df["pitch_type"].replace("FA", "FF")

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


_SPIN_AXIS_LOOKUP_PATH = os.path.join(MODEL_DIR, "spin_axis_lookup.pkl")
_MOVEMENT_BASELINES_PATH = os.path.join(MODEL_DIR, "movement_baselines.pkl")


def engineer_features(
    df: pd.DataFrame,
    baselines=None,
) -> tuple:
    ff_ivb_coefs   = None
    _arm_bin_edges = None
    if isinstance(baselines, tuple):
        if len(baselines) == 3:
            baselines, ff_ivb_coefs, _arm_bin_edges = baselines
        else:
            baselines, ff_ivb_coefs = baselines

    df = clean_statcast(df)

    df["pfx_z_in"] = df["pfx_z"] * 12.0
    df["pfx_x_in"] = df["pfx_x"] * 12.0

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

    if "p_throws" in df.columns:
        throw_sign = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(1.0)
    else:
        throw_sign = pd.Series(1.0, index=df.index)
    df["pfx_x_arm"] = df["pfx_x_in"] * throw_sign

    df["active_spin_rate"] = df["release_spin_rate"] * _spin_eff

    df["ivb_accel"]    = df["az"] + 32.174
    df["ivb_accel_abs"] = df["ivb_accel"].abs()
    df["hb_accel_arm"] = df["ax"] * throw_sign
    df["hb_accel_arm_abs"] = df["hb_accel_arm"].abs()
    df["total_movement"] = np.sqrt(df["ivb_accel"] ** 2 + df["hb_accel_arm"] ** 2)

    _gap_factor = (1.0 / _spin_eff.clip(lower=0.05)) - 1.0
    df["deception_z"] = df["az"] * _gap_factor
    df["deception_x"] = df["hb_accel_arm"] * _gap_factor

    if "spin_axis" in df.columns and "ax" in df.columns and "az" in df.columns:
        _ivb = df["az"] + 32.174
        _sa  = pd.to_numeric(df["spin_axis"], errors="coerce")
        _sa_aligned = (360.0 - _sa) % 360
        _spin_pred = np.radians(_sa_aligned - 180.0)
        _ux, _uz = np.sin(_spin_pred), np.cos(_spin_pred)
        _Mx, _Mz = df["ax"].fillna(0.0), _ivb.fillna(0.0)
        _proj = _Mx * _ux + _Mz * _uz
        df["ssw_z"] = (_Mz - _proj * _uz).fillna(0.0)
        df["ssw_x"] = ((_Mx - _proj * _ux) * throw_sign).fillna(0.0)
    else:
        df["ssw_z"] = 0.0
        df["ssw_x"] = 0.0

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

    if "release_pos_x" in df.columns:
        df["release_pos_x_arm"] = df["release_pos_x"] * throw_sign

    if "spin_axis" in df.columns:
        sa = pd.to_numeric(df["spin_axis"], errors="coerce")
        is_lhp = (df.get("p_throws", pd.Series("R", index=df.index)) == "L")
        df["spin_axis_arm"] = np.where(is_lhp, (360 - sa) % 360, sa)
        _sa_rad = np.radians(df["spin_axis_arm"])
        df["spin_axis_arm_sin"] = np.sin(_sa_rad)
        df["spin_axis_arm_cos"] = np.cos(_sa_rad)
        year_col = "game_year" if "game_year" in df.columns else None
        group_cols_yr = ["pitch_type", "p_throws"] + ([year_col] if year_col else [])
        valid_gc = [c for c in group_cols_yr if c in df.columns]

        if len(df) > 10_000 and valid_gc:
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
                _fb = df.groupby(["pitch_type", "p_throws"])["spin_axis_arm"].mean()
                for (pt, throws), val in _fb.items():
                    _lookup[(str(pt), str(throws), 0)] = float(val)
                os.makedirs(MODEL_DIR, exist_ok=True)
                with open(_SPIN_AXIS_LOOKUP_PATH, "wb") as _fh:
                    _pkl.dump(_lookup, _fh)
            except Exception:
                pass
        else:
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

    df = add_approach_angles(df)

    statcast_aa = df["arm_angle"] if "arm_angle" in df.columns else pd.Series(np.nan, index=df.index)

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
            _model_feats = [f for f in _aa_feats if f in _Xaa.columns]
            _valid = _Xaa[_model_feats].notna().all(axis=1)
            if _valid.any():
                estimated_aa[_valid] = _aa_model.predict(_Xaa.loc[_valid, _model_feats].values)

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

    df["arm_angle"] = (
        statcast_aa
        .where(statcast_aa.notna(), other=estimated_aa)
        .where(statcast_aa.notna() | estimated_aa.notna(), other=lookup_aa)
    )

    df["release_pos_z_resid"] = df["release_pos_z"] - (4.467 + 0.03416 * df["arm_angle"])

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

    actual_move_angle = np.degrees(np.arctan2(df["hb_accel_arm"], df["ivb_accel"]))
    raw_dev = actual_move_angle - df["arm_angle"].fillna(45.0)
    dev_rad = np.radians((raw_dev + 180) % 360 - 180)
    df["arm_angle_dev"] = (raw_dev + 180) % 360 - 180
    df["arm_angle_dev_sin"] = np.sin(dev_rad)
    df["arm_angle_dev_cos"] = np.cos(dev_rad)
    df["arm_angle_dev_magnitude"] = df["total_movement"] * np.sin(dev_rad)
    df["lateral_deception_mag"] = np.abs(df["total_movement"] * np.sin(dev_rad))

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

    if "p_throws" in df.columns:
        df["p_throws_r"] = (df["p_throws"] == "R").astype(float)
    else:
        df["p_throws_r"] = 1.0

    if "p_throws" in df.columns and "stand" in df.columns:
        df["same_hand"] = (df["p_throws"] == df["stand"]).astype(float)
    else:
        df["same_hand"] = 0.0

    _hand_sign = np.where(df["p_throws"].astype(str) == "R", -1.0, 1.0) \
        if "p_throws" in df.columns else 1.0
    if "ax" in df.columns:
        df["ax_arm"] = pd.to_numeric(df["ax"], errors="coerce") * _hand_sign
    if "release_pos_x" in df.columns:
        df["release_pos_x_arm"] = pd.to_numeric(df["release_pos_x"], errors="coerce") * _hand_sign

    if all(c in df.columns for c in ("vx0", "vy0", "vz0", "ax", "ay", "az", "p_throws")):
        from model.prob_resid import add_magnus
        add_magnus(df)

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

    if "release_speed" in df.columns and "release_extension" in df.columns:
        ext = df["release_extension"].fillna(df["release_extension"].median())
        df["perceived_velocity"] = df["release_speed"] * (60.5 / (60.5 - ext).clip(lower=50.0))
    else:
        df["perceived_velocity"] = df.get("release_speed", 90.0)

    if "spin_axis" in df.columns:
        spin_rad = np.radians(df["spin_axis"].fillna(180.0))
        df["spin_axis_sin"] = np.sin(spin_rad)
        df["spin_axis_cos"] = np.cos(spin_rad)
    else:
        df["spin_axis_sin"] = 0.0
        df["spin_axis_cos"] = -1.0

    if baselines is None:
        baselines, ff_ivb_coefs, _arm_bin_edges = compute_baselines(df)

    df = merge_deviations(df, baselines)

    if "ivb_adj" in df.columns and "hb_adj" in df.columns:
        df["surprise_total"] = np.sqrt(df["ivb_adj"] ** 2 + df["hb_adj"] ** 2)
    else:
        df["surprise_total"] = 0.0

    if "ssw_pfx_z" in df.columns and "ssw_pfx_x_arm" in df.columns:
        df["ssw_total"] = np.sqrt(df["ssw_pfx_z"] ** 2 + df["ssw_pfx_x_arm"] ** 2)
    else:
        df["ssw_total"] = 0.0

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

    if "pitch_type" in df.columns:
        from config import PITCH_TYPE_MODELS as _PTM
        _pt_to_family = {pt: fam for fam, pts in _PTM.items() for pt in pts}
        df["family"] = df["pitch_type"].map(_pt_to_family).fillna("ff")
    else:
        df["family"] = "ff"

    df["pitch_type_code"] = df["pitch_type"].map(PITCH_TYPE_MAP).fillna(15).astype(int)

    _FB_PTS = {"FF", "FA", "SI", "FC"}
    _BR_PTS = {"SL", "SV", "ST", "SC", "GY", "CU", "KC", "CS"}
    _OS_PTS = {"CH", "FO", "EP", "KN", "FS"}
    _fam_map = {**{pt: 0 for pt in _FB_PTS}, **{pt: 1 for pt in _BR_PTS}, **{pt: 2 for pt in _OS_PTS}}
    df["pitch_family_code"] = df["pitch_type"].map(_fam_map).fillna(0).astype(int)

    for _c in ["hb_accel_arm", "ay", "az"]:
        if _c not in df.columns:
            df[_c] = 0.0
    df["total_movement_3d"] = np.sqrt(df["hb_accel_arm"]**2 + df["az"]**2)

    is_rhp = (df.get("p_throws", pd.Series("R", index=df.index)) == "R")
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
        df["gyro_degree"] = 45.0

    if "gyro_degree" in df.columns:
        df["active_spin_rate"] = df["release_spin_rate"] * np.cos(np.radians(df["gyro_degree"].clip(0, 90)))
    elif "release_spin_rate" in df.columns:
        df["active_spin_rate"] = df["release_spin_rate"]

    if "speed_diff"  not in df.columns: df["speed_diff"]  = 0.0
    if "ax_diff"     not in df.columns: df["ax_diff"]     = 0.0
    if "ay_diff"     not in df.columns: df["ay_diff"]     = 0.0
    if "az_diff"     not in df.columns: df["az_diff"]     = 0.0
    df["ax_diff_abs"] = df["ax_diff"].abs()
    df["az_diff_abs"] = df["az_diff"].abs()
    if "pfx_x_diff"  not in df.columns: df["pfx_x_diff"]  = 0.0
    if "pfx_z_diff"  not in df.columns: df["pfx_z_diff"]  = 0.0

    if "release_speed" in df.columns and "release_extension" in df.columns:
        df["perceived_velocity"] = df["release_speed"] * (60.5 / (60.5 - df["release_extension"].clip(upper=9.0)).clip(lower=1.0))

    _kin_cols = ["vx0", "vy0", "vz0", "ax", "ay", "az"]
    if all(c in df.columns for c in _kin_cols):
        _vy0 = pd.to_numeric(df["vy0"], errors="coerce")
        _ay  = pd.to_numeric(df["ay"],  errors="coerce")
        _a   = 0.5 * _ay
        _b   = _vy0
        _c   = 50.0 - 1.417
        _disc = _b**2 - 4.0 * _a * (-_c)
        _t_plate = (-_b - np.sqrt(_disc.clip(lower=0.0))) / (2.0 * _a.replace(0, np.nan))
        _t_plate = _t_plate.clip(lower=0.0)
        _vx_t = pd.to_numeric(df["vx0"], errors="coerce") + pd.to_numeric(df["ax"], errors="coerce") * _t_plate
        _vy_t = _vy0 + _ay * _t_plate
        _vz_t = pd.to_numeric(df["vz0"], errors="coerce") + pd.to_numeric(df["az"], errors="coerce") * _t_plate
        _speed_fps = np.sqrt(_vx_t**2 + _vy_t**2 + _vz_t**2)
        df["plate_speed"] = (_speed_fps * 3600.0 / 5280.0).clip(40.0, 110.0)
    else:
        df["plate_speed"] = df.get("release_speed", pd.Series(90.0, index=df.index))

    if "pfx_x_in" in df.columns and "pfx_z_in" in df.columns:
        df["pfx_total"] = np.sqrt(df["pfx_x_in"]**2 + df["pfx_z_in"]**2)
    elif "pfx_x" in df.columns and "pfx_z" in df.columns:
        df["pfx_total"] = np.sqrt(df["pfx_x"]**2 + df["pfx_z"]**2) * 12.0
    else:
        df["pfx_total"] = 0.0

    df["release_speed_drop"] = df["release_speed"] - df["plate_speed"]

    if "active_spin_rate" in df.columns and "release_spin_rate" in df.columns:
        df["spin_efficiency"] = (
            df["active_spin_rate"] / (df["release_spin_rate"].replace(0, np.nan) + 1e-8)
        ).clip(0.0, 1.0)
    else:
        df["spin_efficiency"] = 0.5

    df["ivb_accel_adj"]    = df.get("ivb_adj", pd.Series(0.0, index=df.index))
    df["hb_accel_arm_adj"] = df.get("hb_adj",  pd.Series(0.0, index=df.index))

    if "arm_angle" in df.columns and "ivb_accel" in df.columns and "hb_accel_arm" in df.columns:
        move_angle = np.degrees(np.arctan2(df["hb_accel_arm"], df["ivb_accel"]))
        df["accel_arm_angle_dev"] = move_angle - df["arm_angle"]
    else:
        df["accel_arm_angle_dev"] = 0.0

    try:
        from model.deadzone import apply_deadzone
        df = apply_deadzone(df)
    except Exception as _dz_exc:
        logger.warning(f"dead-zone feature skipped ({_dz_exc})")
        for _f in ("deadzone_hb_dev", "deadzone_vert_dev", "deadzone_dist"):
            if _f not in df.columns:
                df[_f] = 0.0

    for col in CORE_FEATURES:
        if col in df.columns:
            med = df[col].median()
            df[col] = df[col].fillna(med if pd.notna(med) else 0.0)

    logger.info(f"Feature engineering done. Shape: {df.shape}")
    return df, (baselines, ff_ivb_coefs, _arm_bin_edges)


_FB_CONTEXT_TYPES = {"FF", "FA", "SI"}



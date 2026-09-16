
import logging

import numpy as np
import pandas as pd

from config import ARM_ANGLE_BINS, MIN_PITCHES_BASELINE

logger = logging.getLogger(__name__)


def fit_ff_ivb_model(df: pd.DataFrame) -> np.ndarray:
    ff_mask = df["pitch_type"].isin({"FF", "FA"})
    sub = df[ff_mask].dropna(subset=["arm_angle", "release_pos_z", "pfx_z_in"])
    if len(sub) < 100:
        return np.array([2.0, 2.0])

    arm_sin = np.sin(np.deg2rad(sub["arm_angle"].values))
    interaction = arm_sin * sub["release_pos_z"].values
    X = interaction.reshape(-1, 1)
    y = sub["pfx_z_in"].values
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    logger.info(
        f"FF IVB model (no-intercept sin×ht): β_sin×release_ht={coefs[0]:.3f}"
    )
    return coefs


def compute_baselines(
    df: pd.DataFrame,
    pitch_col: str = "pitch_type",
    n_bins: int = ARM_ANGLE_BINS,
) -> tuple:
    df = df.copy()

    arm_bin_edges = None
    if "arm_angle" not in df.columns:
        logger.warning("arm_angle missing from df – cannot bin by arm angle")
        df["arm_angle_bin"] = 0
    else:
        df["arm_angle_bin"], arm_bin_edges = pd.qcut(
            df["arm_angle"].fillna(df["arm_angle"].median()),
            q=n_bins,
            labels=False,
            duplicates="drop",
            retbins=True,
        )
        arm_bin_edges[0]  = -np.inf
        arm_bin_edges[-1] =  np.inf

    if "p_throws" not in df.columns:
        df["p_throws"] = "R"

    group_cols = [pitch_col, "p_throws", "arm_angle_bin"]

    agg = (
        df.groupby(group_cols)
        .agg(
            ivb_mean=("pfx_z_in", "mean"),
            ivb_std=("pfx_z_in", "std"),
            hb_mean=("pfx_x_arm", "mean"),
            hb_std=("pfx_x_arm", "std"),
            velo_mean=("release_speed", "mean"),
            velo_std=("release_speed", "std"),
            spin_mean=("release_spin_rate", "mean"),
            spin_std=("release_spin_rate", "std"),
            ext_mean=("release_extension", "mean"),
            ext_std=("release_extension", "std"),
            n=("pfx_z_in", "count"),
        )
        .reset_index()
    )

    agg = agg[agg["n"] >= MIN_PITCHES_BASELINE]

    for col in ["ivb_std", "hb_std", "velo_std", "spin_std", "ext_std"]:
        agg[col] = agg[col].fillna(1.0).clip(lower=0.1)

    logger.info(
        f"Baselines: {len(agg)} (pitch_type × p_throws × arm_angle_bin) buckets"
    )


    def _fit_vaa(sub_df):
        out = {"vaa_slope": 0.0, "vaa_slope_pz2": 0.0, "vaa_slope_px": 0.0,
               "vaa_slope_velo": 0.0, "vaa_slope_pz_velo": 0.0, "vaa_intercept": 0.0}
        req = ["vaa", "plate_z"]
        if all(c in sub_df.columns for c in req):
            s = sub_df.dropna(subset=req)
            if len(s) >= 50:
                pz = s["plate_z"].values
                X = np.column_stack([pz, np.ones(len(s))])
                c, *_ = np.linalg.lstsq(X, s["vaa"].values, rcond=None)
                out.update(vaa_slope=float(c[0]), vaa_intercept=float(c[1]))
        return out

    def _fit_haa(sub_df):
        out = {"haa_slope_px": 0.0, "haa_slope_px2": 0.0, "haa_slope_pz": 0.0,
               "haa_slope_velo": 0.0, "haa_intercept": 0.0}
        req = ["haa", "plate_x"]
        if all(c in sub_df.columns for c in req):
            s = sub_df.dropna(subset=req)
            if len(s) >= 50:
                px = s["plate_x"].values
                X = np.column_stack([px, np.ones(len(s))])
                c, *_ = np.linalg.lstsq(X, s["haa"].values, rcond=None)
                out.update(haa_slope_px=float(c[0]), haa_intercept=float(c[1]))
        return out

    _vaa_coefs = _fit_vaa(df)
    _haa_by_hand = {th: _fit_haa(g) for th, g in df.groupby("p_throws")}
    _default_haa = _fit_haa(df)

    aa_coef_rows = []
    for keys, grp in df.groupby(["pitch_type", "p_throws"]):
        pt, throws = keys
        row = {"pitch_type": pt, "p_throws": throws}
        row.update(_vaa_coefs)
        row.update(_haa_by_hand.get(throws, _default_haa))
        aa_coef_rows.append(row)

    if aa_coef_rows:
        aa_df = pd.DataFrame(aa_coef_rows)
        agg = agg.merge(aa_df, on=["pitch_type", "p_throws"], how="left")
        for col in ["vaa_slope", "vaa_slope_pz2", "vaa_slope_px", "vaa_slope_velo",
                    "vaa_slope_pz_velo", "vaa_intercept",
                    "haa_slope_px", "haa_slope_px2", "haa_slope_pz", "haa_slope_velo",
                    "haa_intercept"]:
            agg[col] = agg[col].fillna(0.0)
        logger.info("Approach angle regression coefficients stored in baselines.")

    velo_arm_rows = []
    for keys, grp in df.groupby(["pitch_type", "p_throws"]):
        pt, throws = keys
        row = {
            "pitch_type":       pt,
            "p_throws":         throws,
            "velo_slope_arm":   0.0,
            "velo_intercept_arm": float(grp["release_speed"].mean()) if len(grp) > 0 else 0.0,
            "velo_resid_std":   float(grp["release_speed"].std()) if len(grp) > 1 else 1.0,
        }
        if "arm_angle" in grp.columns and "release_speed" in grp.columns:
            sub = grp.dropna(subset=["arm_angle", "release_speed"])
            if len(sub) >= 50:
                c = np.polyfit(sub["arm_angle"].values, sub["release_speed"].values, 1)
                row["velo_slope_arm"]     = float(c[0])
                row["velo_intercept_arm"] = float(c[1])
                resid = sub["release_speed"].values - (c[0] * sub["arm_angle"].values + c[1])
                row["velo_resid_std"] = max(float(resid.std()), 0.1)
        velo_arm_rows.append(row)

    if velo_arm_rows:
        velo_arm_df = pd.DataFrame(velo_arm_rows)
        agg = agg.merge(velo_arm_df, on=["pitch_type", "p_throws"], how="left")
        for col in ["velo_slope_arm", "velo_intercept_arm", "velo_resid_std"]:
            agg[col] = agg[col].fillna(0.0)
        agg["velo_resid_std"] = agg["velo_resid_std"].clip(lower=0.1)
        logger.info("Velocity arm-angle regression coefficients stored in baselines.")

    ivb_arm_rows = []
    for keys, grp in df.groupby(["pitch_type", "p_throws"]):
        pt, throws = keys
        row = {
            "pitch_type":           pt,
            "p_throws":             throws,
            "ivb_slope_arm":        0.0,
            "ivb_slope_arm2":       0.0,
            "ivb_slope_rz":         0.0,
            "ivb_intercept_armrz":  float(grp["ivb_accel"].mean()) if "ivb_accel" in grp.columns and len(grp) > 0 else 0.0,
            "ivb_resid_std":        float(grp["ivb_accel"].std())  if "ivb_accel" in grp.columns and len(grp) > 1 else 1.0,
            "hb_slope_arm":         0.0,
            "hb_slope_arm2":        0.0,
            "hb_slope_rz":          0.0,
            "hb_intercept_armrz":   float(grp["hb_accel_arm"].mean()) if "hb_accel_arm" in grp.columns and len(grp) > 0 else 0.0,
            "hb_resid_std":         float(grp["hb_accel_arm"].std())  if "hb_accel_arm" in grp.columns and len(grp) > 1 else 1.0,
        }
        needed = ["arm_angle", "release_pos_z", "release_extension", "ivb_accel", "hb_accel_arm"]
        if all(c in grp.columns for c in needed):
            sub = grp.dropna(subset=needed)
            if len(sub) >= 50:
                arm_vals = sub["arm_angle"].values
                X = np.column_stack([
                    arm_vals,
                    arm_vals ** 2,
                    sub["release_pos_z"].values,
                    sub["release_extension"].values,
                    np.ones(len(sub)),
                ])
                coefs_ivb, *_ = np.linalg.lstsq(X, sub["ivb_accel"].values, rcond=None)
                row["ivb_slope_arm"]        = float(coefs_ivb[0])
                row["ivb_slope_arm2"]       = float(coefs_ivb[1])
                row["ivb_slope_rz"]         = float(coefs_ivb[2])
                row["ivb_slope_ext"]        = float(coefs_ivb[3])
                row["ivb_intercept_armrz"]  = float(coefs_ivb[4])
                resid_ivb = sub["ivb_accel"].values - X @ coefs_ivb
                row["ivb_resid_std"] = max(float(resid_ivb.std()), 0.01)
                coefs_hb, *_ = np.linalg.lstsq(X, sub["hb_accel_arm"].values, rcond=None)
                row["hb_slope_arm"]         = float(coefs_hb[0])
                row["hb_slope_arm2"]        = float(coefs_hb[1])
                row["hb_slope_rz"]          = float(coefs_hb[2])
                row["hb_slope_ext"]         = float(coefs_hb[3])
                row["hb_intercept_armrz"]   = float(coefs_hb[4])
                resid_hb = sub["hb_accel_arm"].values - X @ coefs_hb
                row["hb_resid_std"] = max(float(resid_hb.std()), 0.01)
        ivb_arm_rows.append(row)

    if ivb_arm_rows:
        ivb_arm_df = pd.DataFrame(ivb_arm_rows)
        agg = agg.merge(ivb_arm_df, on=["pitch_type", "p_throws"], how="left")
        for col in [
            "ivb_slope_arm", "ivb_slope_arm2", "ivb_slope_rz", "ivb_intercept_armrz", "ivb_resid_std",
            "hb_slope_arm",  "hb_slope_arm2",  "hb_slope_rz",  "hb_intercept_armrz",  "hb_resid_std",
        ]:
            agg[col] = agg[col].fillna(0.0)
        agg["ivb_resid_std"] = agg["ivb_resid_std"].clip(lower=0.1)
        agg["hb_resid_std"]  = agg["hb_resid_std"].clip(lower=0.1)
        logger.info("IVB/HB arm-angle + release-height regression coefficients stored in baselines.")

    ssw_global_rows = []
    for hand, hand_df in df.groupby("p_throws"):
        row = {
            "p_throws":         hand,
            "ssw_z_global_cos": 0.0,
            "ssw_z_global_sin": 0.0,
            "ssw_z_global_int": 0.0,
            "ssw_x_global_cos": 0.0,
            "ssw_x_global_sin": 0.0,
            "ssw_x_global_int": 0.0,
        }
        needed = ["spin_axis", "release_spin_rate", "az", "ax"]
        if all(c in hand_df.columns for c in needed):
            sub = hand_df.dropna(subset=needed)
            if len(sub) >= 500:
                sa_rad = np.radians(sub["spin_axis"].values)
                sr_vals = (
                    sub["active_spin_rate"].values
                    if "active_spin_rate" in sub.columns
                    else sub["release_spin_rate"].values
                )
                throw_sign = np.where(sub["p_throws"].values == "R", 1.0, -1.0)
                ax_arm = sub["ax"].values * throw_sign
                X = np.column_stack([
                    sr_vals * np.cos(sa_rad),
                    sr_vals * np.sin(sa_rad),
                    np.ones(len(sub)),
                ])
                bz, _, _, _ = np.linalg.lstsq(X, sub["az"].values, rcond=None)
                bx, _, _, _ = np.linalg.lstsq(X, ax_arm, rcond=None)
                row["ssw_z_global_cos"] = float(bz[0])
                row["ssw_z_global_sin"] = float(bz[1])
                row["ssw_z_global_int"] = float(bz[2])
                row["ssw_x_global_cos"] = float(bx[0])
                row["ssw_x_global_sin"] = float(bx[1])
                row["ssw_x_global_int"] = float(bx[2])
        ssw_global_rows.append(row)

    if ssw_global_rows:
        ssw_global_df = pd.DataFrame(ssw_global_rows)
        agg = agg.merge(ssw_global_df, on=["p_throws"], how="left")
        for col in ["ssw_z_global_cos", "ssw_z_global_sin", "ssw_z_global_int",
                    "ssw_x_global_cos", "ssw_x_global_sin", "ssw_x_global_int"]:
            agg[col] = agg[col].fillna(0.0)
        logger.info("SSW global Magnus coefficients (raw az/ax_arm, per handedness, with intercept) stored in baselines.")

    ff_ivb_coefs = fit_ff_ivb_model(df)

    return agg, ff_ivb_coefs, arm_bin_edges


def merge_deviations(
    df: pd.DataFrame,
    baselines: pd.DataFrame,
    pitch_col: str = "pitch_type",
) -> pd.DataFrame:
    join_cols = [pitch_col]
    if "p_throws" in baselines.columns and "p_throws" in df.columns:
        join_cols.append("p_throws")
    elif "p_throws" not in df.columns:
        df = df.copy()
        df["p_throws"] = "R"

    if "arm_angle_bin" in baselines.columns and "arm_angle_bin" in df.columns:
        join_cols.append("arm_angle_bin")

    _baseline_cols = [c for c in baselines.columns if c not in join_cols and c in df.columns]
    if _baseline_cols:
        df = df.drop(columns=_baseline_cols)
    merged = df.merge(baselines, on=join_cols, how="left")

    merged["ivb_z"] = (merged["pfx_z_in"] - merged["ivb_mean"]) / merged["ivb_std"].fillna(1)
    merged["hb_z"] = (merged["pfx_x_arm"] - merged["hb_mean"]) / merged["hb_std"].fillna(1)
    merged["velo_z"] = (merged["release_speed"] - merged["velo_mean"]) / merged["velo_std"].fillna(1)
    merged["spin_z"] = (merged["release_spin_rate"] - merged["spin_mean"]) / merged["spin_std"].fillna(1)
    merged["ext_z"] = (merged["release_extension"] - merged["ext_mean"]) / merged["ext_std"].fillna(1)

    if "gyro_frac" in merged.columns:
        merged["active_spin_z"] = (merged["spin_z"] * (1.0 - merged["gyro_frac"].clip(0, 1))).clip(lower=0)

    _D_group = (60.5 - merged["ext_mean"]).clip(lower=50.0)
    _D_pitch = (60.5 - merged["release_extension"]).clip(lower=50.0)
    merged["perceived_velo"] = merged["release_speed"] * _D_group / _D_pitch
    merged["pv_z"] = (merged["perceived_velo"] - merged["velo_mean"]) / merged["velo_std"].fillna(1)

    if "vaa_slope" in merged.columns and "vaa" in merged.columns and "plate_z" in merged.columns:
        def _get(col): return merged[col] if col in merged.columns else pd.Series(0.0, index=merged.index)
        plate_z       = merged["plate_z"].fillna(merged["plate_z"].median())
        plate_x_v     = merged["plate_x"].fillna(0.0) if "plate_x" in merged.columns else pd.Series(0.0, index=merged.index)
        release_speed = merged["release_speed"].fillna(merged["release_speed"].median()) \
            if "release_speed" in merged.columns else pd.Series(0.0, index=merged.index)
        merged["vaa_adj"] = (
            merged["vaa"]
            - merged["vaa_slope"]        * plate_z
            - _get("vaa_slope_pz2")      * plate_z ** 2
            - _get("vaa_slope_px")       * plate_x_v
            - _get("vaa_slope_velo")     * release_speed
            - _get("vaa_slope_pz_velo")  * (plate_z * release_speed)
            - merged["vaa_intercept"]
        )
    else:
        merged["vaa_adj"] = merged.get("vaa", pd.Series(np.nan, index=merged.index))

    if (
        "haa_slope_px" in merged.columns
        and "haa" in merged.columns
        and "plate_x" in merged.columns
    ):
        _throw_sign = merged["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0)
        merged["haa_adj"] = (
            merged["haa"]
            - _get("haa_slope_px")   * plate_x_v
            - _get("haa_slope_px2")  * plate_x_v ** 2
            - _get("haa_slope_pz")   * plate_z
            - _get("haa_slope_velo") * release_speed
            - merged["haa_intercept"]
        ) * _throw_sign
    else:
        merged["haa_adj"] = merged.get("haa", pd.Series(np.nan, index=merged.index))

    if (
        "velo_slope_arm" in merged.columns
        and "arm_angle" in merged.columns
        and "release_speed" in merged.columns
    ):
        arm = merged["arm_angle"].fillna(merged["arm_angle"].median())
        predicted_velo = (
            merged["velo_slope_arm"] * arm + merged["velo_intercept_arm"]
        )
        resid_std = merged["velo_resid_std"].clip(lower=0.1)
        merged["velo_adj"] = (merged["release_speed"] - predicted_velo) / resid_std
    else:
        merged["velo_adj"] = merged.get("velo_z", pd.Series(0.0, index=merged.index))

    if (
        "ivb_slope_arm" in merged.columns
        and "arm_angle" in merged.columns
        and "release_pos_z" in merged.columns
        and "ivb_accel" in merged.columns
    ):
        arm = merged["arm_angle"].fillna(merged["arm_angle"].median())
        rz  = merged["release_pos_z"].fillna(merged["release_pos_z"].median())
        ext = merged["release_extension"].fillna(merged["release_extension"].median()) if "release_extension" in merged.columns else pd.Series(6.0, index=merged.index)
        arm2_coef = merged["ivb_slope_arm2"] if "ivb_slope_arm2" in merged.columns \
            else pd.Series(0.0, index=merged.index)
        ext_coef_ivb = merged["ivb_slope_ext"] if "ivb_slope_ext" in merged.columns \
            else pd.Series(0.0, index=merged.index)
        pred_ivb = (
            merged["ivb_slope_arm"] * arm
            + arm2_coef * arm ** 2
            + merged["ivb_slope_rz"] * rz
            + ext_coef_ivb * ext
            + merged["ivb_intercept_armrz"]
        )
        ivb_resid_std = merged["ivb_resid_std"].clip(lower=0.01)
        merged["ivb_adj"] = (merged["ivb_accel"] - pred_ivb) / ivb_resid_std
    else:
        merged["ivb_adj"] = merged.get("ivb_z", pd.Series(np.nan, index=merged.index))

    if (
        "hb_slope_arm" in merged.columns
        and "arm_angle" in merged.columns
        and "release_pos_z" in merged.columns
        and "hb_accel_arm" in merged.columns
    ):
        arm = merged["arm_angle"].fillna(merged["arm_angle"].median())
        rz  = merged["release_pos_z"].fillna(merged["release_pos_z"].median())
        ext = merged["release_extension"].fillna(merged["release_extension"].median()) if "release_extension" in merged.columns else pd.Series(6.0, index=merged.index)
        arm2_coef_hb = merged["hb_slope_arm2"] if "hb_slope_arm2" in merged.columns \
            else pd.Series(0.0, index=merged.index)
        ext_coef_hb = merged["hb_slope_ext"] if "hb_slope_ext" in merged.columns \
            else pd.Series(0.0, index=merged.index)
        pred_hb = (
            merged["hb_slope_arm"] * arm
            + arm2_coef_hb * arm ** 2
            + merged["hb_slope_rz"] * rz
            + ext_coef_hb * ext
            + merged["hb_intercept_armrz"]
        )
        hb_resid_std = merged["hb_resid_std"].clip(lower=0.01)
        merged["hb_adj"] = (merged["hb_accel_arm"] - pred_hb) / hb_resid_std
    else:
        merged["hb_adj"] = merged.get("hb_z", pd.Series(np.nan, index=merged.index))

    ssw_coef_cols = ["ssw_z_global_cos", "ssw_z_global_sin", "ssw_z_global_int",
                     "ssw_x_global_cos", "ssw_x_global_sin", "ssw_x_global_int"]
    have_coefs = all(c in merged.columns for c in ssw_coef_cols)
    have_spin  = ("spin_axis" in merged.columns and
                  "release_spin_rate" in merged.columns)
    have_raw   = ("az" in merged.columns and "ax" in merged.columns and
                  "p_throws" in merged.columns)

    if have_coefs and have_spin and have_raw:
        sa_rad = np.radians(merged["spin_axis"].fillna(0.0))
        sr = (
            merged["active_spin_rate"].fillna(0.0)
            if "active_spin_rate" in merged.columns
            else merged["release_spin_rate"].fillna(0.0)
        )
        spin_cos = sr * np.cos(sa_rad)
        spin_sin = sr * np.sin(sa_rad)
        throw_sign = np.where(merged["p_throws"].values == "R", 1.0, -1.0)
        ax_arm = merged["ax"].fillna(0.0).values * throw_sign

        magnus_z = (merged["ssw_z_global_cos"] * spin_cos
                    + merged["ssw_z_global_sin"] * spin_sin
                    + merged["ssw_z_global_int"])
        magnus_x = (merged["ssw_x_global_cos"] * spin_cos
                    + merged["ssw_x_global_sin"] * spin_sin
                    + merged["ssw_x_global_int"])

        merged["ssw_pfx_z"]     = merged["az"].fillna(0.0) - magnus_z
        merged["ssw_pfx_x_arm"] = ax_arm - magnus_x
    else:
        merged["ssw_pfx_z"]     = 0.0
        merged["ssw_pfx_x_arm"] = 0.0

    merged["ssw_pfx_x_glove"]   = -merged["ssw_pfx_x_arm"]
    merged["ssw_pfx_z_abs"]     = merged["ssw_pfx_z"].abs()
    merged["ssw_pfx_x_arm_abs"] = merged["ssw_pfx_x_arm"].abs()

    return merged

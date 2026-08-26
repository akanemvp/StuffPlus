"""
Movement feature engineering:

  • Arm angle estimation from release position
  • Total movement magnitude (combined IVB + HB)
  • Movement balance ratio  – encodes the insight that 12-12 is better than 7-14
    for hybrid sweepers, because multiple Magnus forces acting in opposite planes
    produce different deception than maximizing a single axis.
  • Gyro proxy from spin_axis (high gyro → less Magnus-driven movement)
  • Population baselines: mean/std of IVB and HB by (pitch_type, arm_angle_bin)
  • Z-score deviations from those baselines
"""

import logging

import numpy as np
import pandas as pd

from config import ARM_ANGLE_BINS, MIN_PITCHES_BASELINE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arm angle
# ---------------------------------------------------------------------------

def calculate_arm_angle(df: pd.DataFrame) -> pd.Series:
    """
    Estimate arm angle (degrees above horizontal) from release velocity components.

    Uses atan2(-vz0, |vx0|):
      - vz0 is inverted because overhand pitchers have downward vz0 at release
        (arm is moving downward as it releases the ball), so higher arm angle
        corresponds to more negative vz0.
      - |vx0| is the magnitude of horizontal arm motion (sign varies by handedness
        so we take absolute value for consistent measurement).
    Result: 0° = sidearm, 90° = straight overhead, negative = submarine.

    Validation vs Statcast ground truth (2025, 100k pitches):
      Velocity formula MAE: 12.6°, RMSE: 16.6° (4× better than position formula)
      Position formula MAE: 44.7°, RMSE: 58.0° (systematically wrong)

    Spring training Statcast data does not include the native arm_angle field,
    making this fallback critical for spring training inference accuracy.
    Falls back to the less-accurate release-position formula only when velocity
    components are unavailable.

    Positive = overhand, 0 = sidearm, negative = submarine.
    """
    # Fallback formula — used for pitches where Statcast's native arm_angle field
    # is null. This applies to ~4% of regular-season pitches and ~100% of spring
    # training pitches (Statcast does not populate arm_angle for spring games).
    # See engineering.py step 3 for the primary path (Statcast value when available).
    if "vx0" in df.columns and "vz0" in df.columns:
        vx0 = pd.to_numeric(df["vx0"], errors="coerce")
        vz0 = pd.to_numeric(df["vz0"], errors="coerce")
        # Clip |vx0| to avoid near-zero denominator (which produces ±90° regardless of vz0).
        # Pitches with effectively no horizontal velocity component (rare edge cases) would
        # blow up atan2 — clip to ≥0.5 ft/s which gives a max angle error of ~1°.
        vx0_abs = vx0.abs().clip(lower=0.5)
        arm_angle = np.arctan2(-vz0, vx0_abs) * (180.0 / np.pi)
        # Clip to physically plausible range: Tyler Rogers ≈ −61° (most extreme submarine
        # in MLB); no pitcher releases from beyond overhead (>85°).
        arm_angle = arm_angle.clip(lower=-70.0, upper=85.0)
        # Fill any remaining NaNs (missing vx0/vz0) with position-based estimate
        if arm_angle.isna().any():
            delta_z = df["release_pos_z"] - 5.5
            delta_x = df["release_pos_x"].abs() - 1.5
            pos_fallback = np.arctan2(delta_z, delta_x.abs() + 0.01) * (180.0 / np.pi)
            arm_angle = arm_angle.fillna(pos_fallback)
        return arm_angle
    else:
        # Last resort: position-based estimate (less accurate, kept for backward compat)
        delta_z = df["release_pos_z"] - 5.5
        delta_x = np.abs(df["release_pos_x"]) - 1.5
        arm_angle = np.arctan2(delta_z, np.abs(delta_x) + 0.01) * (180.0 / np.pi)
        return arm_angle


# ---------------------------------------------------------------------------
# Combined movement features
# ---------------------------------------------------------------------------

def total_movement(ivb: pd.Series, hb: pd.Series) -> pd.Series:
    """Euclidean distance in 2-D movement space (inches)."""
    return np.sqrt(ivb ** 2 + hb ** 2)


def movement_balance(ivb: pd.Series, hb: pd.Series) -> pd.Series:
    """
    Balance ratio: min(|IVB|, |HB|) / max(|IVB|, |HB|).

    1.0 = perfectly balanced (e.g. 12-12 on a sweeper/curveball combo).
    0.0 = all movement on one axis only.

    Encodes the heuristic: for breaking balls, balanced dual-plane movement
    (12-12) is more effective than lopsided movement (7-14), even at the same
    total magnitude. See also: sweeper/slurve hybrid note.
    """
    a = np.abs(ivb)
    b = np.abs(hb)
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    return mn / (mx + 1e-6)


def gyro_proxy(spin_axis: pd.Series) -> pd.Series:
    """
    Approximate gyro spin fraction from spin_axis (degrees, 0–360).

    Convention (RHP):
      ~180° → pure backspin (4-seam, high IVB)
      ~0/360° → pure topspin (curve, high drop)
      ~90/270° → pure sidespin (sweeper, high HB)
      Off-axis = gyro component (reduces Magnus effect)

    We map to [0, 1] where 1 = maximum gyro.
    The formula uses the deviation from the nearest cardinal axis.
    """
    axis_rad = np.deg2rad(spin_axis % 360)
    # Distance to nearest 90° multiple (cardinal axes = pure Magnus)
    dist_to_cardinal = np.minimum(axis_rad % (np.pi / 2), np.pi / 2 - (axis_rad % (np.pi / 2)))
    # Normalize: max distance from a cardinal axis is π/4
    gyro_frac = dist_to_cardinal / (np.pi / 4)
    return gyro_frac


# ---------------------------------------------------------------------------
# Population baselines
# ---------------------------------------------------------------------------

def fit_ff_ivb_model(df: pd.DataFrame) -> np.ndarray:
    """
    Fit OLS: FF/FA IVB ~ sin(arm_angle) × release_pos_z

    Used to estimate the "expected 4-seam IVB" for pitchers who never throw
    a FF/FA, based solely on their arm angle and release height.

    Uses the INTERACTION term sin(arm_angle) × release_height rather than
    additive terms, because backspin efficiency is a gate on the release-height
    IVB effect:

      • Overhead (sin≈1): full release_height → IVB relationship applies.
      • Sidearm (sin≈0): release_height → IVB collapses to near zero.
      • Submarine (sin<0): slightly negative expected IVB (topspin geometry).

    A simple additive model (β1*arm_angle + β2*release_height) fails for
    low-slot pitchers because the arm_angle coefficient is tiny (≈0.002° from
    a population dominated by overhand pitchers) and the release_height term
    incorrectly predicts 14–15" IVB for any pitcher releasing at 5.5 ft,
    regardless of arm slot.

    Returns coefficient array [intercept, β_interaction] where
      interaction = sin(arm_angle) × release_height.
    """
    ff_mask = df["pitch_type"].isin({"FF", "FA"})
    sub = df[ff_mask].dropna(subset=["arm_angle", "release_pos_z", "pfx_z_in"])
    if len(sub) < 100:
        return np.array([2.0, 2.0])   # sensible default: ~2 + 2*sin*ht

    arm_sin = np.sin(np.deg2rad(sub["arm_angle"].values))
    interaction = arm_sin * sub["release_pos_z"].values
    # No intercept: forces IVB=0 when arm_angle=0° (sidearm/submarine).
    # Physically correct — a pure sidearm pitcher produces zero backspin
    # so a 4-seam from that slot has near-zero IVB regardless of release height.
    # An intercept lets the OLS settle at the population mean (~15") for all
    # low-slot pitchers, defeating the purpose of the interaction.
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
    """
    Compute expected (mean ± std) of movement, velocity, and spin by
    (pitch_type × p_throws × arm_angle_bin) from the training population.

    arm_angle_bin IS included so that velocity, spin, and extension baselines
    are arm-slot-adjusted.  Submarine pitchers throw much slower than overhand
    pitchers, so comparing Rogers' 83.5 mph sinker to a population mean of
    ~92 mph gives velo_z ≈ -4 and destroys his grade.  With arm-slot
    adjustment, his velo is compared to same-slot peers and velo_z ≈ 0.

    The movement baselines (ivb_mean/std, hb_mean/std) are also arm-slot
    adjusted here (ivb_z, hb_z = "is this unusual for THIS arm slot?").
    The full-population movement deviation ("is this unusual vs what ALL
    hitters have seen?") is captured by ivb_z_raw / hb_z_raw, computed
    separately in engineer_features() via a pitch_type-only groupby.

    Handedness (p_throws) IS kept: RHP and LHP breaking balls move in
    opposite horizontal directions (RHP KC: +8" HB, LHP KC: -3" HB).

    Returns
    -------
    (baselines_df, ff_ivb_coefs, arm_bin_edges)
      baselines_df  – DataFrame keyed by pitch_type × p_throws × arm_angle_bin
      ff_ivb_coefs  – OLS coefficients for estimating 4-seam IVB from mechanics
      arm_bin_edges – bin edges from pd.qcut; save these so inference uses
                      pd.cut with fixed edges rather than recomputing qcut on
                      the (smaller, differently-distributed) inference batch.
    """
    df = df.copy()

    # ── Arm angle bins — kept for ivb_sep_z groupby, not used in baseline ─
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
        # Extend to ±inf so pd.cut at inference handles pitchers outside
        # the training arm-angle range (e.g. new extreme arm slots).
        arm_bin_edges[0]  = -np.inf
        arm_bin_edges[-1] =  np.inf

    # ── Handedness column ─────────────────────────────────────────────────
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

    # Drop under-populated buckets
    agg = agg[agg["n"] >= MIN_PITCHES_BASELINE]

    # Fill any zero std to avoid division by zero
    for col in ["ivb_std", "hb_std", "velo_std", "spin_std", "ext_std"]:
        agg[col] = agg[col].fillna(1.0).clip(lower=0.1)

    logger.info(
        f"Baselines: {len(agg)} (pitch_type × p_throws × arm_angle_bin) buckets"
    )

    # ── Approach angle normalization ──────────────────────────────────────
    # VAA depends heavily on pitch height (plate_z): a pitch at the top of the
    # zone is flatter by physics regardless of arm slot. Raw VAA conflates
    # pitch location with arm-slot quality. We regress out plate_z within
    # each (pitch_type, p_throws) group so the residual (vaa_adj) isolates
    # the arm-slot/velocity-driven approach quality that actually matters.
    #
    # HAA similarly depends on horizontal location (plate_x) and the pitcher's
    # release point (release_pos_x). A pitch released from the left side of the
    # rubber has a different HAA than the same pitch released from the right,
    # independent of movement or approach quality. We regress out both.
    #
    # Coefficients are stored as columns in the baselines DataFrame so that
    # merge_deviations() can apply them at inference time without refitting.
    # --- Approach-angle regressions (NO pitch-type grouping) ---
    # VAA_hat = b0 + b1*plate_z + b2*plate_z^2 + b3*plate_x + b4*release_speed
    #              + b5*(plate_z*release_speed)   -- ONE global fit (vertical
    #   approach is handedness-symmetric, so R/L are pooled).
    # HAA_hat = b0 + b1*plate_x + b2*plate_x^2 + b3*plate_z + b4*release_speed
    #   -- fit PER p_throws only (horizontal approach flips by throwing hand).
    # Neither uses release point / arm slot / extension, so flat approach from a
    # low slot or big extension survives in the residual.

    # LOCATION-ONLY adjustment: remove ONLY the relevant location axis from each angle
    # (plate_z for VAA, plate_x for HAA). Velocity, release point, arm slot, extension —
    # all survive in the residual. So vaa_adj/haa_adj = approach angle with just the
    # location effect stripped out.
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

    _vaa_coefs = _fit_vaa(df)                                  # one global VAA fit
    _haa_by_hand = {th: _fit_haa(g) for th, g in df.groupby("p_throws")}  # HAA per hand
    _default_haa = _fit_haa(df)

    aa_coef_rows = []
    for keys, grp in df.groupby(["pitch_type", "p_throws"]):
        pt, throws = keys
        row = {"pitch_type": pt, "p_throws": throws}
        row.update(_vaa_coefs)                                 # same VAA coefs everywhere
        row.update(_haa_by_hand.get(throws, _default_haa))     # HAA coefs by handedness
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

    # ── Velocity arm-angle normalization ──────────────────────────────────
    # Raw velocity conflates arm-slot mechanics with true pitch quality.
    # Submarine pitchers (e.g. Tyler Rogers, −61°) naturally throw slower
    # than overhand pitchers due to mechanics — not because their pitch is
    # less effective.  The 6-bin qcut baseline partially corrects for this
    # but bin 0 spans −71° to +26° (mean 17.8°), so Rogers at −61° is still
    # compared to sidearm/three-quarter peers at ~85 mph → velo_z ≈ −3σ.
    #
    # Fix: fit release_speed ~ arm_angle (linear) within (pitch_type, p_throws)
    # and store the slope/intercept + residual std.  At inference, velo_adj =
    # (release_speed − predicted_velo) / resid_std.  This gives the
    # continuous arm-slot correction and cleanly parallels vaa_adj / haa_adj.
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
                # Residual std (after removing arm-angle trend)
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

    # ── IVB / HB arm-angle + release-height normalization ─────────────────
    # IVB depends systematically on arm angle and release height:
    #   • High arm angle + high release height → more backspin geometry → more IVB
    #     on ALL pitch types (4-seams, sliders, changeups, splitters, cutters).
    #   • Low arm angle + low release height (e.g. Soriano ~35°, Rogers ~−61°)
    #     → less IVB on ALL pitch types by the same physics.
    #
    # Problem without adjustment: Soriano's FS at −0.46" IVB gets penalised vs
    # the full FS population because overhand pitchers (55°+) generate +4 to +8"
    # IVB on splitters.  His arm slot naturally produces less IVB — comparing raw
    # IVB to a population dominated by higher-arm-slot pitchers is unfair.
    #
    # Fix: fit ivb_accel ~ arm_angle + release_pos_z (multiple linear regression)
    # within each (pitch_type, p_throws).  ivb_accel = az + 32.174 (ft/s²), the
    # instantaneous aerodynamic vertical force — velocity-independent and
    # flight-time-independent.
    #
    # CRITICAL: previously used pfx_z_in (time-integrated inches) as the target.
    # pfx_z_in ∝ t² where t = (60.5 − release_extension) / |vy0|.  Elite-extension
    # pitchers (Glasnow 7.61 ft, Luzardo 7.4 ft) have ~5% shorter flight times,
    # compressing pfx_z_in by ~10% even at equal spin force.  This caused ivb_adj
    # to systematically penalise extreme-extension pitchers: Glasnow's ivb_adj was
    # 23rd pctile (pfx-based) vs 70th pctile (ivb_accel-based) — the ~1.9" pfx
    # deficit was entirely an extension artifact.
    # ivb_accel is immune to this because it is instantaneous, not time-integrated.
    #
    # Effect expected:
    #   • Soriano FS: arm_angle≈35° + lower release_pos_z → predicted ivb_accel
    #     lower than overhand average → his negative residual is closer to neutral
    #   • Rogers SL: arm_angle≈−61° → very different expected ivb_accel → rising
    #     slider gets appropriate credit for slot context
    #   • Glasnow FF: arm_angle 55.9° but release_pos_z average (5.96 ft, 62nd
    #     pctile) → expected ivb_accel reflects that; no flight-time penalty
    ivb_arm_rows = []
    for keys, grp in df.groupby(["pitch_type", "p_throws"]):
        pt, throws = keys
        row = {
            "pitch_type":           pt,
            "p_throws":             throws,
            "ivb_slope_arm":        0.0,
            "ivb_slope_arm2":       0.0,   # quadratic — captures diminishing IVB at extreme arm angles
            "ivb_slope_rz":         0.0,
            "ivb_intercept_armrz":  float(grp["ivb_accel"].mean()) if "ivb_accel" in grp.columns and len(grp) > 0 else 0.0,
            "ivb_resid_std":        float(grp["ivb_accel"].std())  if "ivb_accel" in grp.columns and len(grp) > 1 else 1.0,
            "hb_slope_arm":         0.0,
            "hb_slope_arm2":        0.0,   # quadratic — captures diminishing HB at extreme arm angles
            "hb_slope_rz":          0.0,
            "hb_intercept_armrz":   float(grp["hb_accel_arm"].mean()) if "hb_accel_arm" in grp.columns and len(grp) > 0 else 0.0,
            "hb_resid_std":         float(grp["hb_accel_arm"].std())  if "hb_accel_arm" in grp.columns and len(grp) > 1 else 1.0,
        }
        needed = ["arm_angle", "release_pos_z", "release_extension", "ivb_accel", "hb_accel_arm"]
        if all(c in grp.columns for c in needed):
            sub = grp.dropna(subset=needed)
            if len(sub) >= 50:
                arm_vals = sub["arm_angle"].values
                # Design matrix: arm + arm² + release_pos_z + release_extension + intercept.
                # arm² captures diminishing-returns curvature of IVB vs arm angle.
                # release_extension added per Trevor Thrash's ext:height approach —
                # pitchers with more extension get more movement from the same arm slot.
                X = np.column_stack([
                    arm_vals,
                    arm_vals ** 2,
                    sub["release_pos_z"].values,
                    sub["release_extension"].values,
                    np.ones(len(sub)),
                ])
                # IVB regression — target: ivb_accel (instantaneous, flight-time-independent)
                coefs_ivb, *_ = np.linalg.lstsq(X, sub["ivb_accel"].values, rcond=None)
                row["ivb_slope_arm"]        = float(coefs_ivb[0])
                row["ivb_slope_arm2"]       = float(coefs_ivb[1])
                row["ivb_slope_rz"]         = float(coefs_ivb[2])
                row["ivb_slope_ext"]        = float(coefs_ivb[3])
                row["ivb_intercept_armrz"]  = float(coefs_ivb[4])
                resid_ivb = sub["ivb_accel"].values - X @ coefs_ivb
                row["ivb_resid_std"] = max(float(resid_ivb.std()), 0.01)
                # HB regression — target: hb_accel_arm (instantaneous, flight-time-independent)
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

    # ── Spin-movement residuals (seam-shifted wake — SSW) ─────────────────
    # For each (pitch_type × p_throws) group, fit a no-intercept OLS model:
    #   pfx_z_in = β_z1 × (spin_rate × cos(spin_axis)) + β_z2 × (spin_rate × sin(spin_axis))
    #   pfx_x_arm = β_x1 × (spin_rate × cos(spin_axis)) + β_x2 × (spin_rate × sin(spin_axis))
    #
    # The spin_rate × trig(spin_axis) terms express what each pitch's spin PREDICTS
    # for movement via the Magnus effect. The residuals capture movement the spin
    # cannot explain — this is the seam-shifted wake (SSW) component:
    #   ssw_pfx_z   = unexpected vertical movement (actual − spin-predicted IVB)
    #   ssw_pfx_x_arm = unexpected arm-side run (actual − spin-predicted HB)
    #
    # This is the "unexpected movement" the stuff+ model should credit:
    #   • Skubal CH: actual arm-side run > spin-predicted → positive ssw_pfx_x_arm
    #   • Senga FK: total SSW residual captures tumbling seam effects
    #   • Any FF that rides more than spin axis predicts → positive ssw_pfx_z
    #
    # Audit (2025 pitchers, n≥50 pitches, pitcher-level r vs avg_stuff+):
    #   ssw_pfx_z:     FF r=+0.36, SI r=+0.16, CU r=−0.16, CH r=−0.11, ST r=+0.20
    #   ssw_pfx_x_arm: CH r=−0.27 (LHP −0.41, RHP −0.22), SI r=+0.16
    # Stored in baselines at pitch_type × p_throws level (arm_angle_bin not needed:
    # spin_axis already accounts for slot; model merges via arm_angle_bin but
    # coefficients broadcast uniformly via the merge fill).
    # Global per-handedness fit (NOT per pitch type) so that pitch-type-typical
    # SSW (sinker arm-side run, splitter drop, etc.) shows up as real signal
    # in the residual instead of being absorbed into pitch-type-specific coefs.
    #
    # Targets are RAW accelerations: az (raw vertical, gravity-included, ft/s²)
    # and ax × throw_sign (arm-side-signed horizontal, ft/s²).  Intercept INCLUDED
    # in the vertical fit so it absorbs the constant gravity offset (-32.174);
    # the residual is then pure aerodynamic SSW.  Acceleration targets are
    # velocity- and extension-independent (unlike time-integrated pfx_*).
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

    # ── FF IVB regression (arm_angle + release_height → expected IVB) ────
    ff_ivb_coefs = fit_ff_ivb_model(df)

    return agg, ff_ivb_coefs, arm_bin_edges


def merge_deviations(
    df: pd.DataFrame,
    baselines: pd.DataFrame,
    pitch_col: str = "pitch_type",
) -> pd.DataFrame:
    """
    Left-join baselines onto df and compute z-score deviations.

    Joins on pitch_type × p_throws × arm_angle_bin so baselines are
    arm-slot-adjusted.  This ensures:
      • velo_z: Rogers' 83.5 mph submarine SI is compared to other
        sidearm/submarine SI pitchers, not to the ~92 mph overhand average.
      • ivb_z / hb_z: arm-slot deviation ("is this unusual for THIS slot?").

    The complementary full-population movement signal ("is this unusual vs
    what ALL hitters have seen?") is ivb_z_raw / hb_z_raw, computed
    separately in engineer_features() via a pitch_type-only groupby.

    New columns added:
      ivb_z, hb_z, velo_z, spin_z, ext_z
    """
    join_cols = [pitch_col]
    if "p_throws" in baselines.columns and "p_throws" in df.columns:
        join_cols.append("p_throws")
    elif "p_throws" not in df.columns:
        df = df.copy()
        df["p_throws"] = "R"

    if "arm_angle_bin" in baselines.columns and "arm_angle_bin" in df.columns:
        join_cols.append("arm_angle_bin")

    # Drop any previously-computed baseline columns to prevent _x/_y suffix
    # conflicts when re-engineering a df that already has these columns.
    _baseline_cols = [c for c in baselines.columns if c not in join_cols and c in df.columns]
    if _baseline_cols:
        df = df.drop(columns=_baseline_cols)
    merged = df.merge(baselines, on=join_cols, how="left")

    merged["ivb_z"] = (merged["pfx_z_in"] - merged["ivb_mean"]) / merged["ivb_std"].fillna(1)
    merged["hb_z"] = (merged["pfx_x_arm"] - merged["hb_mean"]) / merged["hb_std"].fillna(1)
    merged["velo_z"] = (merged["release_speed"] - merged["velo_mean"]) / merged["velo_std"].fillna(1)
    merged["spin_z"] = (merged["release_spin_rate"] - merged["spin_mean"]) / merged["spin_std"].fillna(1)
    merged["ext_z"] = (merged["release_extension"] - merged["ext_mean"]) / merged["ext_std"].fillna(1)

    # Active spin z-score — spin_z scaled by the fraction of spin that is actually
    # Magnus-generating (active). For gyro sliders: gyro_frac ≈ 0.85 → only ~15%
    # active, so active_spin_z ≈ 0.15 × spin_z. For standard sliders (gyro_frac 0.79):
    # ~21% active spin — meaningfully rewards pitchers like Burns whose break comes
    # from real seam force. Pure gyros (gyro_frac 0.92) get near-zero credit.
    if "gyro_frac" in merged.columns:
        # Clipped at 0: only reward positive active spin, never penalize pitchers
        # whose spin quality is below average (e.g. Miller — gyro efficiency, not spin).
        merged["active_spin_z"] = (merged["spin_z"] * (1.0 - merged["gyro_frac"].clip(0, 1))).clip(lower=0)

    # ── Perceived velocity ─────────────────────────────────────────────────
    # A pitch released 1 ft closer to the plate gives the batter proportionally
    # less reaction time — equivalent to throwing harder from the standard slot.
    # Physics: if D = 60.5 − extension is the release-to-plate distance, a
    # pitcher with D_pitch < D_group releases from closer and effectively
    # speeds the pitch up:
    #
    #   PV = actual_velo × D_group / D_pitch
    #      = actual_velo × (60.5 − ext_mean) / (60.5 − release_extension)
    #
    # D_group uses the same arm-slot group mean as ext_z, so the PV adjustment
    # is already arm-slot-adjusted (a submarine pitcher is compared to other
    # submarine pitchers' extension norms, not the overhand population).
    #
    # pv_z is z-scored using the same group velo_mean/velo_std as velo_z, since
    # E[perceived_velo] ≈ velo_mean (the extension adjustment averages to 1 at
    # the group mean). Floor D at 50 ft to prevent extreme outliers from blowing
    # up the ratio on data-entry errors.
    #
    # Example – Glasnow FF (95.7 mph, 7.61 ft ext, group mean 6.5 ft):
    #   PV = 95.7 × 54.0 / 52.89 = 97.7 mph  →  +2.0 mph phantom velocity
    #   pv_z rises from ≈ +0.6 (raw) to ≈ +1.6 (perceived)
    _D_group = (60.5 - merged["ext_mean"]).clip(lower=50.0)
    _D_pitch = (60.5 - merged["release_extension"]).clip(lower=50.0)
    merged["perceived_velo"] = merged["release_speed"] * _D_group / _D_pitch
    merged["pv_z"] = (merged["perceived_velo"] - merged["velo_mean"]) / merged["velo_std"].fillna(1)

    # ── Adjusted approach angles ──────────────────────────────────────────
    # Apply regression coefficients stored during compute_baselines().
    # vaa_adj = residual after removing: pitch location, release point, arm slot,
    #   velocity, extension, AND az/ax/ay (in-flight accelerations).
    # Regressing out az/ax/ay is critical: without it, the model simultaneously
    # rewards lift (az) and penalizes the flat plate arrival that lift causes
    # (e.g. Pérez sweeper). The residual now captures approach quality beyond
    # what the pitch's own physics predict.
    # Falls back to raw vaa / haa if coefficients weren't stored (old baselines).
    if "vaa_slope" in merged.columns and "vaa" in merged.columns and "plate_z" in merged.columns:
        def _get(col): return merged[col] if col in merged.columns else pd.Series(0.0, index=merged.index)
        plate_z       = merged["plate_z"].fillna(merged["plate_z"].median())
        plate_x_v     = merged["plate_x"].fillna(0.0) if "plate_x" in merged.columns else pd.Series(0.0, index=merged.index)
        release_speed = merged["release_speed"].fillna(merged["release_speed"].median()) \
            if "release_speed" in merged.columns else pd.Series(0.0, index=merged.index)
        # VAAx = VAA_actual − VAA_hat, where
        # VAA_hat = b0 + b1*plate_z + b2*plate_z^2 + b3*plate_x + b4*release_speed + b5*(plate_z*release_speed)
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
        # Handedness matters for HAA: coefficients are fit separately per p_throws,
        # and the residual is sign-flipped to be arm-side-relative across R/L.
        _throw_sign = merged["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0)
        # HAAx = HAA_actual − HAA_hat, where
        # HAA_hat = b0 + b1*plate_x + b2*plate_x^2 + b3*plate_z + b4*release_speed
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

    # ── Arm-angle-adjusted velocity ───────────────────────────────────────
    # velo_adj removes the arm-slot effect on raw velocity.
    # Computed here as metadata (not in MODEL_FEATURES) — the bin-adjusted
    # velo_z is already handled by the arm_angle_bin groupby in baselines.
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

    # ── Arm-angle + release-height-adjusted IVB / HB ─────────────────────
    # ivb_adj: spin-force residual after removing predictable arm-slot and
    #   release-height effects on ivb_accel (instantaneous aerodynamic force,
    #   ft/s²).  "Is this pitch generating more/less vertical spin force than
    #   expected for a pitcher at THIS arm angle and release height?"
    #
    # Uses ivb_accel (not pfx_z_in) as the target to avoid the extension artifact:
    #   pfx_z_in ∝ t² so extreme-extension pitchers (Glasnow 7.61 ft) accumulate
    #   ~10% less inches even at equal spin force → unfair penalty via pfx-based adj.
    #   ivb_accel is instantaneous and flight-time-independent, so extension has
    #   no effect on the measurement.
    # hb_adj: same logic — uses hb_accel_arm instead of pfx_x_arm.
    if (
        "ivb_slope_arm" in merged.columns
        and "arm_angle" in merged.columns
        and "release_pos_z" in merged.columns
        and "ivb_accel" in merged.columns
    ):
        arm = merged["arm_angle"].fillna(merged["arm_angle"].median())
        rz  = merged["release_pos_z"].fillna(merged["release_pos_z"].median())
        ext = merged["release_extension"].fillna(merged["release_extension"].median()) if "release_extension" in merged.columns else pd.Series(6.0, index=merged.index)
        # ivb_slope_arm2/ext may be absent in old baselines — default to 0
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
        # hb_slope_arm2/ext may be absent in old baselines — default to 0
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

    # ── True Seam-Shifted Wake (SSW) residuals — raw accel (ft/s²) ────────
    # Single global Magnus model per handedness (fit across all pitch types) on
    # RAW az (gravity-included) and arm-side-signed ax. Intercept in the fit
    # absorbs the constant gravity offset, so the residual is pure aerodynamic
    # SSW.  Pitch-type-typical seam effects show up as real signal.
    #
    #   ssw_pfx_z:     vertical SSW (ft/s²)  (+ = more rise force than spin predicts)
    #   ssw_pfx_x_arm: horizontal SSW (ft/s²) arm-side (+ = more fade force)
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

"""Dynamic dead-zone feature.

For each fastball-family pitch type (FF / SI / FC) we model the joint 4-D
distribution of release characteristics and acceleration as a multivariate
normal:

    X = [ hb_accel_arm, az,  arm_angle (d),  scaled_extension (ê = ext / height) ]
        |____ X_acc ____|    |________ X_rel ________|

    X_pitch_type ~ N(μ, Σ),   μ = [μ_acc; μ_rel],
    Σ = [[Σ_acc,  Σ_cross],
         [Σ_crossᵀ, Σ_rel ]]

Given a pitch's observed release characteristics r = [d, ê], the conditional
distribution of its acceleration is

    X_acc | X_rel = r  ~  N(μ̄, Σ̄)
    μ̄ = μ_acc + Σ_cross Σ_rel⁻¹ (r − μ_rel)

A multinomial-logistic gate gives the per-type weights π_k = P(type = k | r),
so the "expected movement for this slot" is the mixture conditional mean

    E[X_acc | r] = Σ_k π_k(r) · μ̄_k(r)        (k ∈ {FF, SI, FC})

This expected movement is the **dynamic dead zone**: where a fastball from this
arm slot / extension is expected to go. The feature is each pitch's deviation
from it:

    deadzone_hb_dev   = hb_accel_arm − E[hb | r]   (extra arm-side run vs slot)
    deadzone_vert_dev = az           − E[az | r]   (extra ride/sink vs slot)
    deadzone_dist     = ‖(deadzone_hb_dev, deadzone_vert_dev)‖

≈0 → the pitch moves exactly as its slot predicts (in the dead zone, easy to
read); large → movement that defies the slot (deceptive).
"""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR

logger = logging.getLogger(__name__)

DEADZONE_PATH  = os.path.join(MODEL_DIR, "deadzone.pkl")
DEADZONE_FEATS = ["deadzone_hb_dev", "deadzone_vert_dev", "deadzone_dist"]
_FAM           = ["FF", "SI", "FC"]
# 4-vector order: [hb_accel_arm, az, arm_angle, ext_scaled]; acc = idx 0:2, rel = idx 2:4
_COLS = ["hb_accel_arm", "az", "arm_angle", "ext_s"]


def _scaled_extension(df: pd.DataFrame, height_lookup: dict, hdef: float) -> pd.Series:
    """ê = release_extension (ft) / pitcher height (ft).  Height lookup is in inches."""
    h_ft = (df["pitcher"].map(height_lookup).fillna(hdef)) / 12.0
    return pd.to_numeric(df["release_extension"], errors="coerce") / h_ft


def train_deadzone(df: pd.DataFrame, save: bool = True) -> dict:
    """Fit per-type (FF/SI/FC) 4-D Gaussians + a multinomial-logistic gate.

    df needs: pitch_type, hb_accel_arm, az, arm_angle, release_extension, pitcher.
    """
    from sklearn.linear_model import LogisticRegression

    # Reuse the arm-angle estimator's pitcher-height lookup (inches).
    hl: dict = {}
    _aa = os.path.join(MODEL_DIR, "arm_angle_estimator.pkl")
    if os.path.exists(_aa):
        try:
            hl = pickle.load(open(_aa, "rb")).get("height_lookup", {})
        except Exception:
            hl = {}
    hdef = float(np.nanmean(list(hl.values()))) if hl else 74.0

    d = df.copy()
    d["ext_s"] = _scaled_extension(d, hl, hdef)
    for c in _COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=_COLS + ["pitch_type"])
    fb = d[d["pitch_type"].isin(_FAM)]

    params: dict = {}
    for k in _FAM:
        g = fb[fb["pitch_type"] == k]
        X  = g[_COLS].values
        mu = X.mean(0)
        S  = np.cov(X.T)
        params[k] = {
            "mu_acc": mu[:2],                       # [E hb, E az]
            "mu_rel": mu[2:],                       # [E d,  E ê]
            "B":      S[:2, 2:] @ np.linalg.inv(S[2:, 2:]),   # Σ_cross Σ_rel⁻¹  (2×2)
        }

    gate = LogisticRegression(max_iter=1000).fit(
        fb[["arm_angle", "ext_s"]].values, fb["pitch_type"].values
    )

    bundle = {
        "params": params, "gate": gate, "classes": list(gate.classes_),
        "height_lookup": hl, "hdef": hdef, "fam": _FAM,
    }
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(DEADZONE_PATH, "wb") as fh:
            pickle.dump(bundle, fh)
        logger.info(f"  Dead-zone model saved ({len(fb):,} FB-family pitches) → {DEADZONE_PATH}")
    return bundle


def apply_deadzone(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Add the three dead-zone deviation columns to df.

    Safe no-op (columns set to 0.0) if the model artifact or any required input
    column is missing — never raises into the feature pipeline.
    """
    for f in DEADZONE_FEATS:
        df[f] = 0.0

    if bundle is None:
        if not os.path.exists(DEADZONE_PATH):
            return df
        try:
            bundle = pickle.load(open(DEADZONE_PATH, "rb"))
        except Exception:
            return df

    need = ["hb_accel_arm", "az", "arm_angle", "release_extension", "pitcher"]
    if not all(c in df.columns for c in need):
        return df

    try:
        params, gate, cls = bundle["params"], bundle["gate"], bundle["classes"]
        hl, hdef = bundle["height_lookup"], bundle["hdef"]

        es = _scaled_extension(df, hl, hdef)
        dd = pd.to_numeric(df["arm_angle"], errors="coerce")
        d_fill = dd.median() if dd.notna().any() else 38.0
        e_fill = es.median() if es.notna().any() else 1.0
        R = np.column_stack([dd.fillna(d_fill).values, es.fillna(e_fill).values])

        pi = gate.predict_proba(R)                       # N × n_types
        E  = np.zeros((len(df), 2))
        for j, k in enumerate(cls):
            p = params[k]
            E += pi[:, j:j + 1] * (p["mu_acc"][None, :] + (R - p["mu_rel"][None, :]) @ p["B"].T)

        hb = pd.to_numeric(df["hb_accel_arm"], errors="coerce").values
        az = pd.to_numeric(df["az"], errors="coerce").values
        dev_hb = hb - E[:, 0]
        dev_vt = az - E[:, 1]
        df["deadzone_hb_dev"]   = np.nan_to_num(dev_hb)
        df["deadzone_vert_dev"] = np.nan_to_num(dev_vt)
        df["deadzone_dist"]     = np.nan_to_num(np.hypot(dev_hb, dev_vt))
    except Exception as exc:
        logger.warning(f"apply_deadzone failed ({exc}); dead-zone features zeroed.")
    return df

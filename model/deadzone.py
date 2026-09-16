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
_COLS = ["hb_accel_arm", "az", "arm_angle", "ext_s"]


def _scaled_extension(df: pd.DataFrame, height_lookup: dict, hdef: float) -> pd.Series:
    h_ft = (df["pitcher"].map(height_lookup).fillna(hdef)) / 12.0
    return pd.to_numeric(df["release_extension"], errors="coerce") / h_ft


def apply_deadzone(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
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

        pi = gate.predict_proba(R)
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

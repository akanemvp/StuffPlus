"""
Stuff+ inference — loads the trained model and turns pitches into grades.

StuffPlusPredictor loads the trained three-group model, then:
  1. engineers shape features for each pitch (unless already engineered),
  2. derives the arm-signed induced shape columns the model scores on,
  3. routes each pitch to its group model (fastball / breaking / offspeed, cutters routed)
     and predicts its expected run value (xRV),
  4. normalizes on the ONE SHARED scale: 100 = league-average pitch, 10 = one SD.
Lower xRV = better pitch = higher grade. All three groups share one scale, so a fastball,
breaking ball, and offspeed pitch are graded on the same scale.

The shape features are arm-normalized and carry no handedness, so a lefty and a righty
throwing physically identical pitches grade identically — one scoring pass per pitch.

One grade column is produced: `stuff_plus`, the raw z-score. No floor, clip, or display
cap — the shallow regularized linear_tree keeps individual pitches bounded on their own
(max ~165), so every pitch shows its true value.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR
from features.engineering import engineer_features
from model.submodels import load_ensemble
from model.prob_resid import grade_pitches, add_shape_features, add_magnus, SHAPE_FEATS

logger = logging.getLogger(__name__)


class StuffPlusPredictor:
    def __init__(self):
        self.ensemble: dict | None = None
        self.router = None
        self.baselines = None
        self._load()

    def _load(self):
        bpath = os.path.join(MODEL_DIR, "movement_baselines.pkl")
        if os.path.exists(bpath):
            with open(bpath, "rb") as f:
                self.baselines = pickle.load(f)

        self.ensemble = load_ensemble("all")
        if self.ensemble is not None:
            logger.info("Global model loaded (ensemble_all.pkl)")
        else:
            logger.warning("No model found — run 'python main.py train' first")

        rpath = os.path.join(MODEL_DIR, "cutter_router.pkl")
        if os.path.exists(rpath):
            with open(rpath, "rb") as f:
                self.router = pickle.load(f)

    def predict(self, df, baselines=None, already_engineered=False, norm_set="current"):
        if not already_engineered:
            bl = baselines if baselines is not None else self.baselines
            df, _ = engineer_features(df, baselines=bl)

        df = df.copy()
        df["stuff_plus"] = np.nan
        if self.ensemble is None:
            logger.warning("Model not loaded — returning NaN")
            return df
        add_magnus(df)          # ind_vert/ind_horiz_arm (also router feats)
        add_shape_features(df)  # induced shape feats

        # A pitch is scorable only when all 7 shape features are present. They derive from
        # the 9-parameter trajectory (kinematics) alone — no spin_axis needed.
        rows = df[SHAPE_FEATS].notna().all(axis=1).values
        sp = np.full(len(df), np.nan)
        if rows.any():
            sp[rows] = grade_pitches(df.loc[rows], self.ensemble, norm_set)

        # No velocity floor, clip, or display cap: every scorable pitch grades on its
        # true value, including slow position-player junk (bad on its own merits).
        df["stuff_plus"] = sp
        return df


_COMPOSITE_PREDICTOR = None


def _composite_pitch_grade(df, norm_set: str = "current") -> dict:
    """Grade each pitch type's AVERAGE pitch — {pitch_type: Stuff+}.

    Averages the model's shape features within a pitch type, routes each averaged pitch
    to its family, and scores it on that family's scale. `df` must already be engineered
    (from a *_scored table). Cutter routing uses this pitcher's cutters (df is per-pitcher).
    """
    global _COMPOSITE_PREDICTOR
    if _COMPOSITE_PREDICTOR is None:
        _COMPOSITE_PREDICTOR = StuffPlusPredictor()
    p = _COMPOSITE_PREDICTOR
    if p.ensemble is None or df is None or df.empty or "pitch_type" not in df.columns:
        logger.warning("composite grade skipped: no model or empty frame")
        return {}

    # Regenerate the raw shape columns from kinematics if a slim (refresh) frame lacks
    # them — never fall back to mean-of-grades just because a stored column is missing.
    if all(c in df.columns for c in ("ax", "az", "release_pos_x", "p_throws")):
        df = add_shape_features(df.copy())
    if all(c in df.columns for c in ("vx0", "vy0", "vz0", "ax", "ay", "az", "p_throws")):
        df = add_magnus(df)   # ind_vert/ind_horiz_arm so cutters route correctly
    missing = [f for f in SHAPE_FEATS if f not in df.columns]
    if missing:
        logger.warning(f"composite grade skipped — missing model features: {missing}")
        return {}

    _agg = list(dict.fromkeys(list(SHAPE_FEATS) + ["ind_vert", "ind_horiz_arm"]))
    comp = df.groupby("pitch_type")[_agg].mean().dropna(subset=SHAPE_FEATS)
    if comp.empty:
        return {}
    comp = comp.reset_index()

    sp = grade_pitches(comp, p.ensemble, norm_set)
    out: dict = {}
    for pt, v in zip(comp["pitch_type"].values, sp):
        if np.isfinite(v):
            out[pt] = float(v)
    return out


def load_baselines():
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No baselines at {path}. Run train first.")
    with open(path, "rb") as f:
        return pickle.load(f)

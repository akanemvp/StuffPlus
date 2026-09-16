import logging
import os
import pickle

import numpy as np

from config import MODEL_DIR
from features.engineering import engineer_features
from model.submodels import load_ensemble
from model.prob_resid import grade_pitches, add_shape_features, add_magnus

logger = logging.getLogger(__name__)


class StuffPlusPredictor:
    def __init__(self):
        self.ensemble: dict | None = None
        self.platoon: dict | None = None
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

        p = os.path.join(MODEL_DIR, "ensemble_platoon_sameoppo.pkl")
        if os.path.exists(p):
            with open(p, "rb") as f:
                self.platoon = pickle.load(f)
            logger.info("Platoon model loaded (ensemble_platoon_sameoppo.pkl)")
        else:
            logger.warning("No platoon model at ensemble_platoon_sameoppo.pkl — handed grades unavailable")

        rpath = os.path.join(MODEL_DIR, "cutter_router.pkl")
        if os.path.exists(rpath):
            with open(rpath, "rb") as f:
                self.router = pickle.load(f)

    def predict(self, df, baselines=None, already_engineered=False, norm_set="current"):
        if not already_engineered:
            bl = baselines if baselines is not None else self.baselines
            df, _ = engineer_features(df, baselines=bl)

        df = df.copy()
        df["stuff_plus_rhb"] = np.nan
        df["stuff_plus_lhb"] = np.nan
        if self.platoon is None:
            logger.warning("Platoon model not loaded — returning NaN")
            return df
        add_magnus(df)
        add_shape_features(df)

        feats = self.platoon["feats"]
        rows = df[feats].notna().all(axis=1).values
        if rows.any():
            sub = df.loc[rows]
            g_same = grade_pitches(sub, self.platoon["models"]["same"], norm_set)
            g_oppo = grade_pitches(sub, self.platoon["models"]["oppo"], norm_set)
            is_rhp = (sub["p_throws"].astype(str) == "R").values
            df.loc[rows, "stuff_plus_rhb"] = np.where(is_rhp, g_same, g_oppo)
            df.loc[rows, "stuff_plus_lhb"] = np.where(is_rhp, g_oppo, g_same)

        return df


def load_baselines():
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No baselines at {path}. Run train first.")
    with open(path, "rb") as f:
        return pickle.load(f)

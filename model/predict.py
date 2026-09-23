import logging
import os
import pickle

import numpy as np

from config import MODEL_DIR
from features.engineering import engineer_features
from model.prob_resid import grade_pitches, add_shape_features, add_magnus

logger = logging.getLogger(__name__)

PLATOON_PATH = os.path.join(MODEL_DIR, "ensemble_platoon_sameoppo.pkl")


class StuffPlusPredictor:
    def __init__(self):
        self.platoon: dict | None = None
        self._load()

    def _load(self):
        if os.path.exists(PLATOON_PATH):
            with open(PLATOON_PATH, "rb") as f:
                self.platoon = pickle.load(f)
            logger.info("Platoon model loaded (ensemble_platoon_sameoppo.pkl)")
        else:
            logger.warning("No platoon model at ensemble_platoon_sameoppo.pkl — run 'python main.py train' first")

    def predict(self, df, already_engineered=False, norm_set="current"):
        if not already_engineered:
            df = engineer_features(df)

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

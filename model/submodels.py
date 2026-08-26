"""
Trained-model I/O for the production Stuff+ pipeline.

The model itself (the seven-head Method A ensemble) lives in model/prob_resid.py;
this module only pickles it to / from disk.

  • save_ensemble / load_ensemble — pickle the trained "all" model.
"""

import os
import pickle

from config import MODEL_DIR

ENSEMBLE_KEY = "all"


def save_ensemble(ensemble: dict, family: str = ENSEMBLE_KEY) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, f"ensemble_{family}.pkl")
    with open(path, "wb") as f:
        pickle.dump(ensemble, f)


def load_ensemble(family: str = ENSEMBLE_KEY) -> "dict | None":
    path = os.path.join(MODEL_DIR, f"ensemble_{family}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

import hashlib
import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR
from features.engineering import engineer_features
from model.prob_resid import (
    SHAPE_FEATS, add_magnus, add_shape_features, _dre_values, _outcome_label,
    _fit_group, predict_group_rv, _SAMPLE_NORM,
)

logger = logging.getLogger(__name__)

MODEL_VERSION = "v800_platoon_sameoppo"
PLATOON_PATH = os.path.join(MODEL_DIR, "ensemble_platoon_sameoppo.pkl")


def train_platoon(df: pd.DataFrame) -> dict:
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("Engineering features …")
    df = engineer_features(df)
    add_magnus(df)
    add_shape_features(df)

    sf = df[SHAPE_FEATS].notna().all(axis=1)
    if int(sf.sum()) < 5000:
        raise RuntimeError(f"Too few shaped pitches to train ({int(sf.sum())}).")

    values = _dre_values(df, _outcome_label(df))
    logger.info(f"  values: whiff={values['whiff']:+.4f} foul={values['foul']:+.4f} "
                f"gb={values['gb']:+.4f} air={values['air']:+.4f}")

    same = df["stand"].astype(str).eq(df["p_throws"].astype(str))
    rng = np.random.RandomState(42)
    models = {}
    for key, mask in (("same", same), ("oppo", ~same)):
        sub = df[mask]
        logger.info(f"  training {key}-hand model on {len(sub):,} pitches …")
        g = _fit_group(sub, SHAPE_FEATS, rng)
        models[key] = {"feats": SHAPE_FEATS, "group_feats": {"ALL": SHAPE_FEATS}, "router": None,
                       "values": values, "weights": values, "groups": {"ALL": g}}
        logger.info(f"    swings={g['n_swings']:,} (rounds={g['sw_iter']})  in-play={g['n_inplay']:,} (rounds={g['gb_iter']})")

    idx = df.index[sf]
    if len(idx) > _SAMPLE_NORM:
        idx = pd.Index(np.random.RandomState(42).choice(idx.values, _SAMPLE_NORM, replace=False))
    rv = np.full(len(idx), np.nan)
    for key, mask in (("same", same), ("oppo", ~same)):
        m = mask.loc[idx].values
        if m.any():
            rv[m] = predict_group_rv(df.loc[idx[m]], models[key])
    norm = {"mean": float(np.nanmean(rv)), "std": float(np.nanstd(rv) + 1e-8)}
    for key in models:
        models[key]["norm"] = norm
    logger.info(f"  shared anchor: mean={norm['mean']:+.5f} std={norm['std']:.5f}")

    ens = {"method": "platoon_2way_sameoppo", "feats": SHAPE_FEATS, "values": values,
           "norm": norm, "models": models}
    with open(PLATOON_PATH, "wb") as f:
        pickle.dump(ens, f)
    logger.info("  saved -> ensemble_platoon_sameoppo.pkl")

    version = hashlib.md5(MODEL_VERSION.encode()).hexdigest()[:12]
    with open(os.path.join(MODEL_DIR, "model_version.txt"), "w") as f:
        f.write(version)
    logger.info(f"Model version: {version}")
    return ens

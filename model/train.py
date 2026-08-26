"""
Stuff+ model training.

Trains the production model: a three-group SPLIT model — fastballs / breaking balls /
offspeed, cutters routed to FB or BR by a Mahalanobis classifier — each group a swing
softmax {whiff, foul, in-play} plus a GB/air contact head, on 8 arm-normalized induced
(Magnus-frame) shape features (see model/prob_resid.py). Outcome run values are ONE GLOBAL
set and all three groups are z-scored on ONE SHARED scale, so a fastball, breaking ball,
and offspeed pitch are graded on the same scale.

train_unified() does the full run:
  1. engineer shape features (cached to model/feature_cache.parquet) + Magnus/shape backfill,
  2. save the movement/spin baselines inference needs,
  3. fit + save the cutter router and the per-hand spin-axis convention offset,
  4. train the three-group split model (train_split_model) and save it,
  5. compute and save the shared normalization (current + historical),
  6. write the model version hash.
"""

import hashlib
import logging
import os
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR
from features.engineering import CORE_FEATURES, engineer_features
from model.submodels import save_ensemble

logger = logging.getLogger(__name__)

MODEL_VERSION = "v700_threegroup_gbair"


def train_unified(df: pd.DataFrame) -> dict:
    """Train the single Driveline run-value regressor and save all inference artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    FEAT_CACHE      = os.path.join(MODEL_DIR, "feature_cache.parquet")
    FEAT_CACHE_META = os.path.join(MODEL_DIR, "feature_cache_meta.pkl")
    _cache_key = hashlib.md5(str(sorted(df["game_year"].unique().tolist()) if "game_year" in df.columns else df.shape).encode()).hexdigest()

    # Raw columns that pass through engineering unchanged — needed for training
    # but not saved in older caches. We restore them from the original df after
    # loading the cache so we never need to re-engineer just because a raw column
    # was added to RAW_COLS.
    _RAW_PASSTHROUGH = ["bb_type", "launch_speed", "launch_angle", "hc_x", "hc_y", "stand"]

    _cache_loaded = False
    if os.path.exists(FEAT_CACHE) and os.path.exists(FEAT_CACHE_META):
        _meta = pickle.load(open(FEAT_CACHE_META, "rb"))
        if _meta.get("key") == _cache_key and _meta.get("stage") == "post_fastball":
            logger.info("Loading cached engineered features (skipping feature engineering) …")
            df_raw = df  # keep original for passthrough cols
            df = pd.read_parquet(FEAT_CACHE)
            logger.info(f"  Loaded {len(df):,} rows from cache.")
            # Restore any raw passthrough columns missing from the cache
            for _col in _RAW_PASSTHROUGH:
                if _col not in df.columns and _col in df_raw.columns:
                    df[_col] = df_raw[_col].values
                    logger.info(f"  Restored raw column '{_col}' from original df.")
            _cache_loaded = True

    if not _cache_loaded:
        logger.info("Engineering features …")
        df, _baselines = engineer_features(df)

        # Save movement baselines so inference uses the same regression coefficients
        # (vaa_adj / haa_adj slopes, arm_bin_edges, etc.) as training.
        if _baselines is not None:
            _bpath = os.path.join(MODEL_DIR, "movement_baselines.pkl")
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(_bpath, "wb") as _fh:
                pickle.dump(_baselines, _fh)
            logger.info(f"Movement baselines saved → {_bpath}")

        logger.info("Saving feature cache …")
        df.to_parquet(FEAT_CACHE, index=False)
        pickle.dump({"key": _cache_key, "stage": "post_fastball"}, open(FEAT_CACHE_META, "wb"))
        logger.info(f"  Feature cache saved ({len(df):,} rows)")

    # Save spin axis population means so inference can compute spin_axis_rel correctly
    # (avoids collapse to 0 when scoring a single pitcher in isolation).
    if "spin_axis_arm" in df.columns:
        _spin_axis_path = os.path.join(MODEL_DIR, "spin_axis_lookup.pkl")
        _spin_lookup: dict = {}
        _gc = [c for c in ["pitch_type", "p_throws", "game_year"] if c in df.columns]
        if _gc:
            _gm = df.groupby(_gc)["spin_axis_arm"].mean()
            for idx, val in _gm.items():
                if len(_gc) >= 3:
                    pt, throws, yr = idx
                    _spin_lookup[(str(pt), str(throws), int(yr))] = float(val)
                elif len(_gc) == 2:
                    pt, throws = idx
                    _spin_lookup[(str(pt), str(throws), 0)] = float(val)
                else:
                    _spin_lookup[(str(idx), "R", 0)] = float(val)
        # Year=0 fallback: multi-year mean per (pt, throws)
        _fb_cols = [c for c in ["pitch_type", "p_throws"] if c in df.columns]
        if _fb_cols:
            for idx, val in df.groupby(_fb_cols)["spin_axis_arm"].mean().items():
                pt, throws = (idx if isinstance(idx, tuple) else (idx, "R"))
                _spin_lookup[(str(pt), str(throws), 0)] = float(val)
        with open(_spin_axis_path, "wb") as _fh:
            pickle.dump(_spin_lookup, _fh)
        logger.info(f"Spin axis lookup saved ({len(_spin_lookup)} entries)")

    if "stand" in df.columns and "stand_r" not in df.columns:
        df["stand_r"] = (df["stand"] == "R").astype(int)

    from model.prob_resid import (add_magnus, add_shape_features, fit_cutter_router,
                                   assign_family, bucket_values, train_split_model,
                                   predict_group_rv, fit_spin_offset, _assign_group,
                                   SHAPE_FEATS, GROUPS)
    # Backfill induced-Magnus (router feats) and derive the shape features. Idempotent.
    add_magnus(df)
    fit_spin_offset(df)
    add_shape_features(df)

    # --- Cutter router (fastball vs breaking family), fit + saved (family tagging only) ---
    router = fit_cutter_router(df)
    with open(os.path.join(MODEL_DIR, "cutter_router.pkl"), "wb") as f:
        pickle.dump(router, f)
    fam = assign_family(df, router)
    _fc = df["pitch_type"].astype(str).eq("FC")
    logger.info(f"  Cutter router: {int((fam[_fc]=='FB').sum()):,} FC->FB, "
                f"{int((fam[_fc]=='BR').sum()):,} FC->BR of {int(_fc.sum()):,}")

    # --- Valued swing-outcome run values (whiff, foul) ---
    V = bucket_values(df)
    logger.info(f"  values: whiff={V['whiff']:+.4f}  foul={V['foul']:+.4f}")

    # --- Train the two-group split model (FB / non-FB swing softmax + SIERA contact head) ---
    _sf = df[SHAPE_FEATS].notna().all(axis=1)
    if int(_sf.sum()) < 5000:
        raise RuntimeError(f"Too few shaped pitches to train ({int(_sf.sum())}).")
    ens = train_split_model(df, V, router)

    # Single shared historical norm (<=2024) so 2023/2024 seasons grade on a clean scale.
    _hist = (df["game_year"] <= 2024) if "game_year" in df.columns else None
    if _hist is not None:
        hi = df.index[_hist & _sf]
        if len(hi):
            if len(hi) > 300000:
                hi = pd.Index(np.random.RandomState(42).choice(hi.values, 300000, replace=False))
            eh = predict_group_rv(df.loc[hi], ens)   # routes each pitch to its group model
            ens["norm_hist"] = {"mean": float(np.nanmean(eh)), "std": float(np.nanstd(eh) + 1e-8)}

    save_ensemble(ens, "all")
    ensembles = {"all": ens}
    logger.info("  saved -> ensemble_all.pkl")

    # Drop any stale per-family model files so predict.py only loads ensemble_all.pkl.
    for _fam_key in ("fb", "os", "br"):
        _fp = os.path.join(MODEL_DIR, f"ensemble_{_fam_key}.pkl")
        if os.path.exists(_fp):
            os.remove(_fp); logger.info(f"  Removed stale ensemble_{_fam_key}.pkl")

    version = hashlib.md5(MODEL_VERSION.encode()).hexdigest()[:12]
    with open(os.path.join(MODEL_DIR, "model_version.txt"), "w") as f:
        f.write(version)
    logger.info(f"Model version: {version}")
    logger.info("Done.")

    return ensembles


def train_all(df: pd.DataFrame) -> dict:
    return train_unified(df)


def save_baselines(baselines):
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    with open(path, "wb") as f:
        pickle.dump(baselines, f)
    logger.info(f"Baselines saved → {path}")


def load_baselines():
    path = os.path.join(MODEL_DIR, "movement_baselines.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

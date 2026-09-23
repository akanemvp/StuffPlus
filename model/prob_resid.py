from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

BALL_DESCS   = {"ball", "blocked_ball", "hit_by_pitch", "pitchout"}
CALLED_DESCS = {"called_strike"}
WHIFF_DESCS  = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL_DESCS   = {"foul", "foul_tip", "bunt_foul_tip", "foul_bunt"}
INPLAY_DESC  = "hit_into_play"

SHAPE_FEATS = [
    "release_speed", "release_spin_rate",
    "ind_horiz_arm", "ind_vert",
    "arm_angle", "release_pos_x_arm", "release_pos_z",
    "release_extension",
]
_G = 32.174

_LGBM = dict(linear_tree=True, n_jobs=-1, verbose=-1, random_state=42)
_SAMPLE_SWING, _SAMPLE_GRID, _SAMPLE_NORM = 2_500_000, 1_500_000, 300_000


class _Identity:
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    s = df.get(name)
    if s is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(s, errors="coerce")


def add_magnus(df: pd.DataFrame) -> pd.DataFrame:
    hs = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0).values
    vx = _num(df, "vx0").values
    vy = _num(df, "vy0").values
    vz = _num(df, "vz0").values
    ax = _num(df, "ax").values
    az = _num(df, "az").values
    ay = _num(df, "ay").values
    vm = np.sqrt(vx * vx + vy * vy + vz * vz)
    aax, aaz = ax, az + _G
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = (aax * vx + ay * vy + aaz * vz) / vm
        pz = dot * vz / vm
        px = dot * vx / vm
    df["ind_vert"] = aaz - pz
    df["ind_horiz"] = aax - px
    df["ind_horiz_arm"] = (aax - px) * hs
    return df


def add_shape_features(df: pd.DataFrame) -> pd.DataFrame:
    if "ind_vert" not in df.columns or "ind_horiz_arm" not in df.columns:
        add_magnus(df)
    hs = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0).values
    df["release_pos_x_arm"] = _num(df, "release_pos_x").values * hs
    if "release_extension" in df.columns:
        df["release_extension"] = _num(df, "release_extension").clip(4.0, 8.5)
    if "arm_angle" in df.columns:
        df["arm_angle"] = _num(df, "arm_angle").clip(upper=100.0)
    return df


def _assign_group(df: pd.DataFrame, router: dict | None) -> pd.Series:
    return pd.Series("ALL", index=df.index)


def _grid_cell(df: pd.DataFrame) -> pd.Series:
    la = _num(df, "launch_angle")
    cell = pd.Series(np.nan, index=df.index)
    cell[la < 10]  = 0
    cell[la >= 10] = 1
    return cell


def _dre_values(sub: pd.DataFrame, lab: pd.Series) -> dict:
    dre = pd.to_numeric(sub["delta_run_exp"], errors="coerce").values
    lv = lab.values

    def m(mask):
        x = dre[mask & np.isfinite(dre)]
        return float(x.mean()) if len(x) else 0.0

    cell = _grid_cell(sub).values
    isip = (lv == _OUT_INPLAY)
    return {"whiff": m(lv == _OUT_WHIFF), "foul": m(lv == _OUT_FOUL),
            "gb": m(isip & (cell == 0)), "air": m(isip & (cell == 1))}


_OUT_WHIFF, _OUT_FOUL, _OUT_INPLAY, _OUT_CALLED, _OUT_BALL = 0, 1, 2, 3, 4


def _outcome_label(df: pd.DataFrame) -> pd.Series:
    dd = df["description"].fillna("").astype(str)
    lab = pd.Series(-1, index=df.index)
    lab[dd.isin(WHIFF_DESCS)]  = _OUT_WHIFF
    lab[dd.isin(FOUL_DESCS)]   = _OUT_FOUL
    lab[dd.eq(INPLAY_DESC)]    = _OUT_INPLAY
    lab[dd.isin(CALLED_DESCS)] = _OUT_CALLED
    lab[dd.isin(BALL_DESCS)]   = _OUT_BALL
    return lab


def _p_class(clf, Xg: np.ndarray, target) -> np.ndarray:
    P = clf.predict_proba(Xg)
    cls = list(clf.classes_)
    return P[:, cls.index(target)] if target in cls else np.zeros(len(Xg))


def _inplay_rv(g: dict, Xg: np.ndarray, V: dict) -> np.ndarray:
    p_gb = _p_class(g["grid"], Xg, 0)
    return p_gb * V["gb"] + (1.0 - p_gb) * V["air"]


def _fit_es(X, y, est_cls, kwargs, eval_metric, seed=0, val_frac=0.15):
    rng = np.random.RandomState(seed)
    n = len(X)
    nval = min(max(2000, int(n * val_frac)), n // 2)
    idx = rng.permutation(n)
    vi, ti = idx[:nval], idx[nval:]
    probe = est_cls(n_estimators=2000, **kwargs)
    probe.fit(X[ti], y[ti], eval_set=[(X[vi], y[vi])], eval_metric=eval_metric,
              callbacks=[lgb.early_stopping(50, verbose=False)])
    best = int(probe.best_iteration_ or 2000)
    final = est_cls(n_estimators=best, **kwargs)
    final.fit(X, y)
    return final, best


def _fit_group(sub: pd.DataFrame, feats: list, rng) -> dict:
    sf = sub[feats].notna().all(axis=1)
    scaler = _Identity().fit(sub.loc[sf, feats].values)
    Xs = scaler.transform(sub[feats].values)

    lab = _outcome_label(sub)
    swing = lab.isin([_OUT_WHIFF, _OUT_FOUL, _OUT_INPLAY]).values
    oi = np.where(swing & sf.values)[0]
    if len(oi) > _SAMPLE_SWING:
        oi = rng.choice(oi, _SAMPLE_SWING, replace=False)
    outcome, sw_iter = _fit_es(Xs[oi], lab.values[oi], lgb.LGBMClassifier,
                               dict(objective="multiclass", num_class=3, **_LGBM), "multi_logloss")

    cell = _grid_cell(sub)
    isip = sub["description"].fillna("").astype(str).eq(INPLAY_DESC)
    ipall = (isip & sf & cell.notna()).values
    gi = np.where(ipall)[0]
    if len(gi) > _SAMPLE_GRID:
        gi = rng.choice(gi, _SAMPLE_GRID, replace=False)
    grid, gb_iter = _fit_es(Xs[gi], cell.values[gi].astype(int), lgb.LGBMClassifier,
                            dict(objective="binary", **_LGBM), "binary_logloss")

    return {"outcome": outcome, "classes_out": list(outcome.classes_), "grid": grid,
            "scaler": scaler, "feats": list(feats),
            "n_swings": int(len(oi)), "n_inplay": int(len(gi)),
            "sw_iter": sw_iter, "gb_iter": gb_iter}


def predict_group_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    out = np.full(len(df), np.nan)
    if len(df) == 0:
        return out
    V = ens["values"]
    grp = _assign_group(df, ens.get("router")).values
    for gname, g in ens["groups"].items():
        feats = g["feats"]
        rows = np.where(grp == gname)[0]
        if len(rows) == 0 or not all(c in df.columns for c in feats):
            continue
        X = df.iloc[rows][feats].apply(pd.to_numeric, errors="coerce").values
        ok = np.isfinite(X).all(axis=1)
        if not ok.any():
            continue
        Xg = g["scaler"].transform(X[ok])
        P = g["outcome"].predict_proba(Xg)
        cls = g["classes_out"]
        def pc(c):
            return P[:, cls.index(c)] if c in cls else 0.0
        inplay_rv = _inplay_rv(g, Xg, V)
        out[rows[ok]] = pc(_OUT_WHIFF) * V["whiff"] + pc(_OUT_FOUL) * V["foul"] + pc(_OUT_INPLAY) * inplay_rv
    return out


def grade_pitches(df: pd.DataFrame, ens: dict, norm_set: str = "current") -> np.ndarray:
    if len(df) == 0:
        return np.full(0, np.nan)
    rv = predict_group_rv(df, ens)
    out = np.full(len(df), np.nan)
    n = ens.get("norm_hist") if (norm_set == "historical" and ens.get("norm_hist")) else ens["norm"]
    ok = np.isfinite(rv)
    out[ok] = 100.0 + (n["mean"] - rv[ok]) / max(n["std"], 1e-6) * 10.0
    return out



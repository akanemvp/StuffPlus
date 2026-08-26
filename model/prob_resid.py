"""Stuff+ — three-group (fastball / breaking / offspeed) probability model.

Three shape models — fastballs, breaking balls, and offspeed (cutters routed to FB or BR
by a Mahalanobis classifier) — each grade pitch shape by expected run value:

    xRV = P(whiff)·V_whiff + P(foul)·V_foul + P(in-play)·[ P(GB)·V_gb + P(air)·V_air ]

  • A swing softmax head predicts {whiff, foul, in-play} from pitch shape.
  • A GB/air contact head predicts whether an in-play ball is a ground ball (<10°) or in
    the air (any non-ground-ball contact). The in-play value is that split weighted by two
    run-value constants. (Exit-velocity heads were tried and removed: pitch shape explains
    ~3% of exit-velocity variance and added no out-of-sample ranking skill — all the
    repeatable contact signal is ground-ball propensity.)

Outcome run values are ONE GLOBAL set (mean delta_run_exp per outcome over all pitches),
shared by all three groups — a whiff/grounder/fly is worth the same regardless of what
pitch produced it. Per-pitch differentiation comes from the PROBABILITIES, not the values.
All three groups are z-scored on ONE SHARED scale (100 = league-average pitch, 10 = one
SD), so a fastball, breaking ball, and offspeed pitch are graded on the same scale.

Shape is 7 arm-normalized features (identical for all three groups): velocity, spin rate,
the induced (Magnus-frame, gravity + drag removed) vertical (ind_vert) and arm-side horizontal
(ind_horiz_arm) accelerations, arm angle, and release side and height. The induced movement is
derived from the pitch's 9-parameter trajectory (its velocity and acceleration vectors), so
— unlike the earlier Magnus/non-Magnus split — a pitch scores from kinematics alone and does
NOT need Statcast's per-pitch spin_axis (minor-league feeds without it still score). No
location, count, or game-state; a lefty and righty throwing physically identical pitches
grade identically. (release_extension and arm_angle_dev were tested and dropped 2026-08-19 —
both near-no-ops; and NEVER fold extension into velocity, it corrupts the velocity signal.)

Lower xRV = better. Heads are LightGBM (num_leaves=8, max_depth=3) with linear_tree,
early-stopped on a held-out split; features are passed through unscaled.
"""
from __future__ import annotations

import logging
import os
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import MODEL_DIR

logger = logging.getLogger(__name__)

BALL_DESCS   = {"ball", "blocked_ball", "hit_by_pitch", "pitchout"}
CALLED_DESCS = {"called_strike"}
WHIFF_DESCS  = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
FOUL_DESCS   = {"foul", "foul_tip", "bunt_foul_tip", "foul_bunt"}
INPLAY_DESC  = "hit_into_play"

# 7 arm-normalized shape features. Movement is the INDUCED (Magnus-frame, gravity + drag
# removed) vertical (ind_vert) and arm-side horizontal (ind_horiz_arm) accelerations;
# release side is arm-signed (release_pos_x_arm). A mirror lefty/righty grade identically.
SHAPE_FEATS = [  # 7 kinematic + release_extension as a RAW feature (NOT folded into velocity)
    "release_speed", "release_spin_rate",
    "ind_horiz_arm", "ind_vert",
    "arm_angle", "release_pos_x_arm", "release_pos_z",
    "release_extension",
]
# Raw kinematic columns add_shape_features consumes to build the Magnus split.
KINEMATIC_COLS = ["vx0", "vy0", "vz0", "ax", "ay", "az",
                  "release_extension", "spin_axis", "release_pos_x", "p_throws"]
# Cutter-router space: velocity + arm-relative movement + spin + slot.
ROUTER_FEATS = ["release_speed", "ind_vert", "ind_horiz_arm", "release_spin_rate", "arm_angle"]

# Family membership by Statcast pitch_type. Cutters (FC) are routed by the router.
FB_TYPES  = {"FF", "FA", "SI"}
BR_TYPES  = {"SL", "ST", "SV", "SC", "GY", "CU", "KC", "CS", "SLV"}
OFF_TYPES = {"CH", "FO", "FS", "EP", "KN"}
FAMILIES  = ("FB", "OFF", "BR")
GROUPS    = ("ALL",)   # ONE unified model over all pitch types (no FB/BR/OFF split)
GROUP_FEATS = {"ALL": SHAPE_FEATS}

ZONE_HALF = 0.83   # half plate-width + ball radius (ft)
_G = 32.174

_SPIN_OFFSET_PATH = os.path.join(MODEL_DIR, "spin_offset.pkl")   # per-hand spin-axis convention offset

# All heads: shallow LightGBM (num_leaves=8, max_depth=3) + linear_tree, early-stopped.
_LGBM = dict(linear_tree=True, n_jobs=-1, verbose=-1, random_state=42)   # base LGBM defaults (num_leaves=31, max_depth=-1, lr=0.1) + linear_tree
_SAMPLE_SWING, _SAMPLE_GRID, _SAMPLE_NORM = 2_500_000, 1_500_000, 300_000


class _Identity:
    """No-op feature scaler — features are passed through raw (tree splits are scale-invariant)."""
    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    """A numeric column as a Series, or an all-NaN Series if the column is absent — so a
    feed without spin_axis (minor leagues) yields NaN shape features, not a crash."""
    s = df.get(name)
    if s is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(s, errors="coerce")


def add_magnus(df: pd.DataFrame) -> pd.DataFrame:
    """Add induced-Magnus accel components from raw kinematics (feeds the cutter router).
    Idempotent. ind_vert / ind_horiz(_arm) are ROUTER_FEATS only — scoring uses SHAPE_FEATS.
    """
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


def _transverse_accel(df: pd.DataFrame):
    """Alan Nathan transverse (Magnus-frame) acceleration components aTx, aTz, and the
    arm sign, from raw 9P kinematics. Drag is projected out along the ball's velocity."""
    hs = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0).values
    vx0 = _num(df, "vx0").values
    vy0 = _num(df, "vy0").values
    vz0 = _num(df, "vz0").values
    ax  = _num(df, "ax").values
    ay  = _num(df, "ay").values
    az  = _num(df, "az").values
    ext = _num(df, "release_extension").clip(4, 8).values
    with np.errstate(invalid="ignore", divide="ignore"):
        yR = 60.5 - ext
        tR = (-vy0 - np.sqrt(np.clip(vy0 ** 2 - 2 * ay * (50 - yR), 0, None))) / ay
        vxr = vx0 + ax * tR; vyr = vy0 + ay * tR; vzr = vz0 + az * tR
        tc = (-vyr - np.sqrt(np.clip(vyr ** 2 - 2 * ay * (yR - 17 / 12), 0, None))) / ay
        vxb = vxr + 0.5 * ax * tc; vyb = vyr + 0.5 * ay * tc; vzb = vzr + 0.5 * az * tc
        vb = np.sqrt(vxb ** 2 + vyb ** 2 + vzb ** 2)
        adrag = -(ax * vxb + ay * vyb + (az + _G) * vzb) / vb
        aTx = ax + adrag * vxb / vb
        aTz = az + adrag * vzb / vb + _G
    return aTx, aTz, hs


def fit_spin_offset(df: pd.DataFrame) -> dict:
    """Per-hand spin-axis convention offset = circular mean(spin_axis − movement_dir) over
    four-seamers, whose movement is pure Magnus. Saved so inference can align a single
    pitcher's spin axis to the movement frame. Returns {'R': deg, 'L': deg}."""
    aTx, aTz, _ = _transverse_accel(df)
    sa = _num(df, "spin_axis").values
    mdir = np.degrees(np.arctan2(aTz, aTx)) % 360
    ff = (df["pitch_type"].astype(str) == "FF").values
    throws = df["p_throws"].astype(str).values
    off = {}
    for h in ("R", "L"):
        mk = ff & (throws == h) & np.isfinite(sa) & np.isfinite(mdir)
        if mk.sum() < 100:
            off[h] = 97.0 if h == "R" else 83.0        # documented fallback
            continue
        dev = np.radians(sa[mk] - mdir[mk])
        off[h] = float(np.degrees(np.arctan2(np.nanmean(np.sin(dev)), np.nanmean(np.cos(dev)))))
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(_SPIN_OFFSET_PATH, "wb") as fh:
        pickle.dump(off, fh)
    global _SPIN_OFFSET
    _SPIN_OFFSET = off
    return off


_SPIN_OFFSET: dict | None = None


def _spin_offset() -> dict:
    global _SPIN_OFFSET
    if _SPIN_OFFSET is None:
        if os.path.exists(_SPIN_OFFSET_PATH):
            with open(_SPIN_OFFSET_PATH, "rb") as fh:
                _SPIN_OFFSET = pickle.load(fh)
        else:
            _SPIN_OFFSET = {"R": 97.0, "L": 83.0}       # documented fallback
    return _SPIN_OFFSET


def add_shape_features(df: pd.DataFrame, spin_off: dict | None = None) -> pd.DataFrame:
    """Add the shape columns the scoring model needs. Idempotent.

    Movement is the induced (Magnus-frame) vertical/arm-side horizontal acceleration
    (ind_vert, ind_horiz_arm); release_pos_x_arm is the arm-side release point.
    Extension and arm_angle are winsorized to physically plausible bands.
    """
    if "ind_vert" not in df.columns or "ind_horiz_arm" not in df.columns:
        add_magnus(df)
    hs = df["p_throws"].map({"R": -1.0, "L": 1.0}).fillna(-1.0).values
    df["release_pos_x_arm"] = _num(df, "release_pos_x").values * hs
    # Winsorize extension into a physically plausible band — values outside are
    # Hawkeye tracking errors; clipping (not dropping) caps leverage on linear_tree leaves.
    if "release_extension" in df.columns:
        df["release_extension"] = _num(df, "release_extension").clip(4.0, 8.5)
    # arm_angle: clip ONLY the impossible high side (>100deg = behind the head, ~5 glitch
    # pitches). Do NOT floor the low end — submariners (Tyler Rogers ~-62deg) are real.
    if "arm_angle" in df.columns:
        df["arm_angle"] = _num(df, "arm_angle").clip(upper=100.0)
    return df


def _basefam(pt: str):
    if pt in FB_TYPES:  return "FB"
    if pt in BR_TYPES:  return "BR"
    if pt in OFF_TYPES: return "OFF"
    return None   # FC handled by the router; unknown types unscored


def _maha(A, m, ci):
    d = A - m
    return np.einsum("ij,jk,ik->i", d, ci, d)


def fit_cutter_router(df: pd.DataFrame) -> dict:
    """Fit Mahalanobis centroids for the fastball vs breaking families in ROUTER_FEATS."""
    Z = df[ROUTER_FEATS].apply(pd.to_numeric, errors="coerce")
    mu, sd = Z.mean(), Z.std().replace(0, 1.0)
    Zs = (Z - mu) / sd
    ok = Zs.notna().all(axis=1)
    pt = df["pitch_type"].astype(str)

    def params(mask):
        A = Zs[ok & mask].values
        return A.mean(0), np.linalg.inv(np.cov(A.T) + 1e-6 * np.eye(A.shape[1]))

    m_fb, ci_fb = params(pt.isin(["FF", "SI"]))
    m_br, ci_br = params(pt.isin(["SL", "ST", "CU", "KC"]))
    return {"mu": mu.values, "sd": sd.values, "m_fb": m_fb, "ci_fb": ci_fb,
            "m_br": m_br, "ci_br": ci_br}


def assign_family(df: pd.DataFrame, router: dict | None) -> pd.Series:
    """Family (FB/OFF/BR) per pitch. Cutters routed by Mahalanobis; per-pitcher majority
    when a pitcher/name column is present, else per-pitch."""
    pt = df["pitch_type"].astype(str)
    fam = pt.map(_basefam)
    fc = pt.eq("FC")
    if fc.any() and router is not None and all(c in df.columns for c in ROUTER_FEATS):
        Z = (df.loc[fc, ROUTER_FEATS].apply(pd.to_numeric, errors="coerce").values
             - router["mu"]) / router["sd"]
        ok = np.isfinite(Z).all(axis=1)
        r = np.full(int(fc.sum()), None, dtype=object)
        if ok.any():
            d_fb = _maha(Z[ok], router["m_fb"], router["ci_fb"])
            d_br = _maha(Z[ok], router["m_br"], router["ci_br"])
            r[ok] = np.where(d_fb < d_br, "FB", "BR")
        rr = pd.Series(r, index=df.index[fc])
        pcol = "player_name" if "player_name" in df.columns else ("pitcher" if "pitcher" in df.columns else None)
        if pcol is not None:
            tmp = pd.DataFrame({"p": df.loc[fc, pcol].values, "r": rr.values})
            tmp = tmp[tmp["r"].notna()]
            if len(tmp):
                maj = tmp.groupby("p")["r"].agg(lambda x: x.value_counts().index[0])
                fam.loc[df.index[fc]] = df.loc[fc, pcol].map(maj).values
        else:
            fam.loc[rr.index] = rr.values
    return fam


def _assign_group(df: pd.DataFrame, router: dict | None) -> pd.Series:
    """One unified model over all pitch types — every pitch is graded by the same heads on
    ONE shared scale + GLOBAL run values, so a fastball, breaking ball, and offspeed pitch
    are directly comparable."""
    return pd.Series("ALL", index=df.index)   # one unified group over all pitch types


def _count_adjusted_rv(df: pd.DataFrame) -> pd.Series:
    """delta_run_exp with the mean removed per (balls, strikes) count — a shape-neutral,
    count-adjusted run value. Base-out state is added when those columns are present."""
    dre = pd.to_numeric(df["delta_run_exp"], errors="coerce")
    b = _num(df, "balls").fillna(0).astype(int)
    s = _num(df, "strikes").fillna(0).astype(int)
    grp = pd.DataFrame({"b": b, "s": s, "d": dre})
    keys = ["b", "s"]
    have_situ = all(c in df.columns for c in ("on_1b", "on_2b", "on_3b", "outs_when_up"))
    if have_situ:
        b1 = df["on_1b"].notna().astype(int); b2 = df["on_2b"].notna().astype(int)
        b3 = df["on_3b"].notna().astype(int)
        outs = pd.to_numeric(df["outs_when_up"], errors="coerce").fillna(0).astype(int)
        grp["situ"] = b1 * 4 + b2 * 2 + b3 + outs * 8
        keys.append("situ")
    return dre - grp.groupby(keys)["d"].transform("mean")


def _is_hr(df: pd.DataFrame) -> pd.Series:
    """True where the batted-ball event is a home run."""
    ev = df.get("events")
    if ev is None:
        return pd.Series(False, index=df.index)
    return ev.fillna("").astype(str).eq("home_run")


def _grid_cell(df: pd.DataFrame) -> pd.Series:
    """In-play GB vs air — 0 ground ball (<10°), 1 air (>=10°: any non-GB contact). NaN if none."""
    la = _num(df, "launch_angle")
    cell = pd.Series(np.nan, index=df.index)
    cell[la < 10]  = 0                     # ground ball
    cell[la >= 10] = 1                     # air (all non-ground-ball contact)
    return cell


def _dre_values(sub: pd.DataFrame, lab: pd.Series) -> dict:
    """GLOBAL run values for every valued outcome — whiff, foul, and the two in-play
    batted-ball types (ground ball, air). All constants (mean delta_run_exp), shape-independent:
    a whiff/grounder/fly is worth the same regardless of what pitch produced it. Per-pitch
    differentiation comes from the PROBABILITIES (swing head + GB/air head), not these values."""
    dre = pd.to_numeric(sub["delta_run_exp"], errors="coerce").values
    lv = lab.values

    def m(mask):
        x = dre[mask & np.isfinite(dre)]
        return float(x.mean()) if len(x) else 0.0

    cell = _grid_cell(sub).values                       # 0 = GB, 1 = air (any non-GB)
    isip = (lv == _OUT_INPLAY)
    return {"whiff": m(lv == _OUT_WHIFF), "foul": m(lv == _OUT_FOUL),
            "gb": m(isip & (cell == 0)), "air": m(isip & (cell == 1))}


def bucket_values(df: pd.DataFrame) -> dict:
    """Count-adjusted run values for the valued swing outcomes (whiff, foul)."""
    dd = df["description"].fillna("").astype(str)
    ca = _count_adjusted_rv(df)
    isw = dd.isin(WHIFF_DESCS); isf = dd.isin(FOUL_DESCS)

    def cav(mask):
        v = ca[mask & ca.notna()]
        return float(v.mean()) if len(v) else 0.0

    return {"whiff": cav(isw), "foul": cav(isf)}


# Per-pitch outcome classes for the outcome head (trained on EVERY outcome).
_OUT_WHIFF, _OUT_FOUL, _OUT_INPLAY, _OUT_CALLED, _OUT_BALL = 0, 1, 2, 3, 4


def _outcome_label(df: pd.DataFrame) -> pd.Series:
    """Per-pitch terminal outcome label 0..4 (whiff/foul/in-play/called/ball), −1 otherwise."""
    dd = df["description"].fillna("").astype(str)
    lab = pd.Series(-1, index=df.index)
    lab[dd.isin(WHIFF_DESCS)]  = _OUT_WHIFF
    lab[dd.isin(FOUL_DESCS)]   = _OUT_FOUL
    lab[dd.eq(INPLAY_DESC)]    = _OUT_INPLAY
    lab[dd.isin(CALLED_DESCS)] = _OUT_CALLED
    lab[dd.isin(BALL_DESCS)]   = _OUT_BALL
    return lab


def _p_class(clf, Xg: np.ndarray, target) -> np.ndarray:
    """P(class==target) from a fitted classifier, aligned to its class order."""
    P = clf.predict_proba(Xg)
    cls = list(clf.classes_)
    return P[:, cls.index(target)] if target in cls else np.zeros(len(Xg))


def _inplay_rv(g: dict, Xg: np.ndarray, V: dict) -> np.ndarray:
    """E[RV | in-play] = P(GB)·V_gb + P(air)·V_air. The GB/air classifier supplies the shape-
    dependent probability; V_gb / V_air are GLOBAL constant run values."""
    p_gb = _p_class(g["grid"], Xg, 0)
    return p_gb * V["gb"] + (1.0 - p_gb) * V["air"]


def _fit_es(X, y, est_cls, kwargs, eval_metric, seed=0, val_frac=0.15):
    """Fit a LightGBM estimator (classifier or regressor) with early stopping on a random
    holdout to pick the round count, then refit on ALL X. Returns (model, best_iteration)."""
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
    """Swing head (3-class {whiff,foul,in-play}) + P(HR|in-play) head, each early-stopped
    (round count picked on a 15% holdout, then refit on all data). One feature set + scaler."""
    sf = sub[feats].notna().all(axis=1)
    scaler = _Identity().fit(sub.loc[sf, feats].values)        # no RobustScaler
    Xs = scaler.transform(sub[feats].values)

    lab = _outcome_label(sub)
    swing = lab.isin([_OUT_WHIFF, _OUT_FOUL, _OUT_INPLAY]).values
    oi = np.where(swing & sf.values)[0]
    if len(oi) > _SAMPLE_SWING:
        oi = rng.choice(oi, _SAMPLE_SWING, replace=False)
    outcome, sw_iter = _fit_es(Xs[oi], lab.values[oi], lgb.LGBMClassifier,
                               dict(objective="multiclass", num_class=3, **_LGBM), "multi_logloss")

    # GB vs air classifier (in-play balls with a launch angle)
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


def train_split_model(df: pd.DataFrame, V: dict, router: dict) -> dict:
    """Three probability models (FB / BR / OFF, cutters routed) — each a swing softmax
    {whiff,foul,in-play} + GB/air contact head on raw (unscaled) features — but ONE GLOBAL set
    of outcome run values (raw mean delta_run_exp over all pitches, so a whiff/grounder is
    worth the same regardless of shape) and ONE shared grading scale."""
    sf = df[SHAPE_FEATS].notna().all(axis=1)
    grp = _assign_group(df, router)
    rng = np.random.RandomState(42)

    # ONE global set of outcome run values, shared by both groups (consistent regardless of shape).
    values = _dre_values(df, _outcome_label(df))

    ens = {"method": "fbsplit_gbair_globalvalues", "feats": SHAPE_FEATS, "shape_feats": SHAPE_FEATS,
           "group_feats": GROUP_FEATS, "router": router, "spin_offset": _spin_offset(),
           "values": values, "weights": V, "groups": {}}
    for gname in GROUPS:
        gm = (grp == gname)
        sub = df[gm]
        ens["groups"][gname] = _fit_group(sub, GROUP_FEATS[gname], rng)

    # --- SINGLE shared scale across both groups (100 = league-avg pitch, 10 = one SD) ---
    idx = df.index[sf]
    if len(idx) > _SAMPLE_NORM:
        idx = pd.Index(np.random.RandomState(42).choice(idx.values, _SAMPLE_NORM, replace=False))
    e = predict_group_rv(df.loc[idx], ens)              # routes each pitch to its group model
    ens["norm"] = {"mean": float(np.nanmean(e)), "std": float(np.nanstd(e) + 1e-8)}

    v = ens["values"]
    logger.info(f"  two-group split (SHARED scale + GLOBAL values): norm mean={ens['norm']['mean']:+.5f} std={ens['norm']['std']:.5f}")
    logger.info(f"    global values: Vwh={v["whiff"]:+.4f} Vfo={v["foul"]:+.4f} Vgb={v["gb"]:+.4f} Vair={v["air"]:+.4f}")
    for gname, g in ens["groups"].items():
        logger.info(f"    [{gname}] ({len(g['feats'])}f) outcome n={g['n_swings']:,} (rounds={g['sw_iter']}), "
                    f"in-play n={g["n_inplay"]:,} (GB/air classifier rounds={g["gb_iter"]})")
    return ens


def predict_group_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    """xRV from the single model, each outcome weighted by its empirical run value:
    P(wh)·V_wh + P(foul)·V_fo + P(in-play)·(P(GB)·V_gb + P(FB)·V_fb). Lower = better
    (whiff/grounder run values are negative, fly balls positive).

    Each group is scored on its OWN feature set + scaler; a pitch missing any of its group's
    features returns NaN (no card)."""
    out = np.full(len(df), np.nan)
    if len(df) == 0:
        return out
    V = ens["values"]                                        # ONE global set of outcome values
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
        P = g["outcome"].predict_proba(Xg)                   # 3-class swing {wh,foul,ip}
        cls = g["classes_out"]
        def pc(c):
            return P[:, cls.index(c)] if c in cls else 0.0
        inplay_rv = _inplay_rv(g, Xg, V)                     # P(GB)*V_gb + P(air)*V_air
        out[rows[ok]] = pc(_OUT_WHIFF) * V["whiff"] + pc(_OUT_FOUL) * V["foul"] + pc(_OUT_INPLAY) * inplay_rv
    return out


def grade_pitches(df: pd.DataFrame, ens: dict, norm_set: str = "current") -> np.ndarray:
    """xRV routed to each group, then z-scored on that group's OWN scale:
    100 = group-average pitch, 10 = one group SD (lower xRV -> higher grade)."""
    if len(df) == 0:
        return np.full(0, np.nan)
    rv = predict_group_rv(df, ens)
    out = np.full(len(df), np.nan)
    n = ens.get("norm_hist") if (norm_set == "historical" and ens.get("norm_hist")) else ens["norm"]
    ok = np.isfinite(rv)
    out[ok] = 100.0 + (n["mean"] - rv[ok]) / max(n["std"], 1e-6) * 10.0   # ONE shared scale
    return out


# --- back-compat aliases: external callers still import these names ---
def predict_global_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    return predict_group_rv(df, ens)


def predict_family_rv(df: pd.DataFrame, ens: dict) -> np.ndarray:
    return predict_group_rv(df, ens)


def _batter_zone(df: pd.DataFrame) -> pd.Series:
    px = _num(df, "plate_x")
    pz = _num(df, "plate_z")
    szt = _num(df, "sz_top")
    szb = _num(df, "sz_bot")
    inz = (px.abs().values <= ZONE_HALF) & (pz.values >= szb.values) & (pz.values <= szt.values)
    return pd.Series(np.where(pz.notna().values, inz, False), index=df.index).fillna(False)

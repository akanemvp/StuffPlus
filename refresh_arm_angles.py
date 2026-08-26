"""Refresh ONLY the arm_angle column in scored tables using the new estimator.
Nothing else changes: no retrain, no re-score, no other columns touched.

Fill chain matches features/engineering.py exactly:
  native Statcast arm_angle  ->  new estimator (null rows)  ->  pitcher lookup (last resort)
Native value sourced from the raw table; only estimated rows differ from current.

Dry-run by default. Pass --write to apply (backs up old arm_angle values first).
"""
import os, sys, pickle, sqlite3
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "data", "statcast.db")
MA = os.path.join(HERE, "model", "artifacts")
WRITE = "--write" in sys.argv
BK = "/Users/akane/Desktop/new_stuff/_armangle_refresh_backup_20260614"

with open(os.path.join(MA, "arm_angle_estimator.pkl"), "rb") as f:
    est = pickle.load(f)
FEATS = est["features"]; HL = est["height_lookup"]
HDEF = float(np.nanmean(list(HL.values()))) if HL else 74.0
try:
    with open(os.path.join(MA, "arm_angle_lookup.pkl"), "rb") as f:
        LK = pickle.load(f)
except Exception:
    LK = None

KEY = ["game_pk", "at_bat_number", "pitch_number"]
PAIRS = [("pitches_2023","pitches_2023_scored"),("pitches_2024","pitches_2024_scored"),
         ("pitches_2025","pitches_2025_scored"),("pitches_spring2026","pitches_spring2026_scored"),
         ("pitches_breakout2026","pitches_breakout2026_scored"),("pitches_2026","pitches_2026_scored"),
         ("pitches_aaa2026","pitches_aaa2026_scored"),("pitches_acl2026","pitches_acl2026_scored")]

def cols(cur, t):
    return [r[1] for r in cur.execute(f"PRAGMA table_info({t})")]

def compute(raw):
    n = len(raw)
    statcast = pd.to_numeric(raw["arm_angle"], errors="coerce") if "arm_angle" in raw.columns \
               else pd.Series(np.nan, index=raw.index)
    # new estimator (in-house elif branch contract)
    est_aa = pd.Series(np.nan, index=raw.index)
    X = pd.DataFrame(index=raw.index)
    X["p_throws_r"] = (raw["p_throws"] == "R").astype(float)
    X["height"] = raw["pitcher"].map(HL).fillna(HDEF).astype(float)
    for c in ["release_extension", "release_pos_x", "release_pos_z"]:
        X[c] = pd.to_numeric(raw[c], errors="coerce")
    mf = [f for f in FEATS if f in X.columns]
    valid = X[mf].notna().all(axis=1)
    if valid.any():
        est_aa[valid] = est["model"].predict(X.loc[valid, mf].values)
    # pitcher lookup last resort
    look = pd.Series(np.nan, index=raw.index)
    if isinstance(LK, dict) and "by_pitcher" in LK and "pitcher" in raw.columns:
        bp, bpt = LK["by_pitcher"], LK.get("by_pitcher_pitch_type", {})
        if "pitch_type" in raw.columns:
            look = raw.apply(lambda r: bpt.get((r["pitcher"], r["pitch_type"]),
                             bp.get(r["pitcher"], np.nan)), axis=1).astype(float)
        else:
            look = raw["pitcher"].map(bp).astype(float)
    out = statcast.where(statcast.notna(), est_aa)
    out = out.where(statcast.notna() | est_aa.notna(), look)
    return out

con = sqlite3.connect(DB); cur = con.cursor()
existing = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if WRITE: os.makedirs(BK, exist_ok=True)
print(f"{'WRITE' if WRITE else 'DRY-RUN'} mode\n")
print(f"{'table':<26}{'rows':>9}{'changed':>9}{'mean|Δ|':>9}{'max|Δ|':>9}")
tot_changed = 0
for raw_t, scored_t in PAIRS:
    if raw_t not in existing or scored_t not in existing:
        continue
    rc = cols(cur, raw_t)
    need = ["p_throws","pitcher","release_extension","release_pos_z","release_pos_x"]
    if not all(c in rc for c in need) or not all(k in rc for k in KEY):
        print(f"{scored_t:<26}  SKIP (missing cols/keys)"); continue
    sel = KEY + need + (["arm_angle"] if "arm_angle" in rc else []) + \
          (["pitch_type"] if "pitch_type" in rc else [])
    raw = pd.read_sql(f"SELECT {','.join(sel)} FROM {raw_t}", con)
    raw["new_aa"] = compute(raw)
    cur_sc = pd.read_sql(f"SELECT {','.join(KEY)}, arm_angle AS old_aa, rowid FROM {scored_t}", con)
    for k in KEY:                       # unify join-key dtypes across tables
        raw[k] = pd.to_numeric(raw[k], errors="coerce")
        cur_sc[k] = pd.to_numeric(cur_sc[k], errors="coerce")
    m = cur_sc.merge(raw[KEY + ["new_aa"]], on=KEY, how="left")
    d = (m["new_aa"] - pd.to_numeric(m["old_aa"], errors="coerce"))
    chg = m["new_aa"].notna() & (d.abs() > 1e-6)
    nchg = int(chg.sum()); tot_changed += nchg
    mad = float(d[chg].abs().mean()) if nchg else 0.0
    mx = float(d[chg].abs().max()) if nchg else 0.0
    print(f"{scored_t:<26}{len(m):>9,}{nchg:>9,}{mad:>9.2f}{mx:>9.1f}")
    if WRITE and nchg:
        m.loc[chg, KEY + ["old_aa", "new_aa"]].to_parquet(os.path.join(BK, f"{scored_t}_armangle.parquet"))
        upd = [(float(r.new_aa), int(r.rowid)) for r in m.loc[chg].itertuples()]
        cur.executemany(f"UPDATE {scored_t} SET arm_angle=? WHERE rowid=?", upd)
        con.commit()
print(f"\ntotal rows that would change: {tot_changed:,}")
if WRITE: print(f"backups -> {BK}")
con.close()

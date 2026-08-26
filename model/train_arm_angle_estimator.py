"""Retrain arm_angle_estimator with release_pos_x (release side) added.

Drops into engineering.py's in-house elif branch with NO code change:
  features = ['p_throws_r','height','release_extension','release_pos_z','release_pos_x']
  raw release_pos_x (tree learns handedness via p_throws_r), predicts arm_angle directly.

Validates on pitcher-held-out split vs the current deployed model, then (if better)
retrains on ALL data and saves, backing up the old bundle.
"""
import os, pickle, sqlite3, shutil, sys
import numpy as np, pandas as pd
import lightgbm as lgb

RNG = np.random.RandomState(42)
HERE = os.path.dirname(os.path.abspath(__file__))
PKL = os.path.join(HERE, "model", "artifacts", "arm_angle_estimator.pkl")

with open(PKL, "rb") as f:
    old = pickle.load(f)
heights = old["height_lookup"]
hdef = float(np.nanmean(list(heights.values()))) if heights else 74.0

con = sqlite3.connect(os.path.join(HERE, "data", "statcast.db"))
parts = []
for tbl in ["pitches_2023", "pitches_2024", "pitches_2025"]:
    parts.append(pd.read_sql(
        f"SELECT pitcher,p_throws,arm_angle,release_pos_x,release_pos_z,release_extension "
        f"FROM {tbl} WHERE arm_angle IS NOT NULL AND release_pos_x IS NOT NULL "
        f"AND release_pos_z IS NOT NULL AND release_extension IS NOT NULL", con))
con.close()
df = pd.concat(parts, ignore_index=True)
for c in ["arm_angle", "release_pos_x", "release_pos_z", "release_extension"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["p_throws_r"] = (df["p_throws"] == "R").astype(float)
df["height"] = df["pitcher"].map(heights).fillna(hdef).astype(float)
df = df.dropna(subset=["arm_angle", "release_pos_x", "release_pos_z", "release_extension"]).reset_index(drop=True)

FEATS_SIDE = ["p_throws_r", "height", "release_extension", "release_pos_z", "release_pos_x"]
FEATS_NO   = ["p_throws_r", "height", "release_extension", "release_pos_z"]
LGB = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, subsample=0.8,
           colsample_bytree=0.9, min_child_samples=200, random_state=0, n_jobs=-1, verbose=-1)

def fit(feats, mask):
    m = lgb.LGBMRegressor(**LGB)
    m.fit(df.loc[mask, feats], df.loc[mask, "arm_angle"])
    return m

def mae_rmse(pred, true):
    e = pred - true
    return np.nanmean(np.abs(e)), np.sqrt(np.nanmean(e ** 2))

# ---- validation: pitcher held-out ----
pitchers = df["pitcher"].unique(); RNG.shuffle(pitchers)
test_p = set(pitchers[:int(len(pitchers) * 0.30)])
tr_mask = ~df["pitcher"].isin(test_p)
te_mask = df["pitcher"].isin(test_p)
yte = df.loc[te_mask, "arm_angle"].values

m_no   = fit(FEATS_NO, tr_mask)
m_side = fit(FEATS_SIDE, tr_mask)
r_no   = mae_rmse(m_no.predict(df.loc[te_mask, FEATS_NO]), yte)
r_side = mae_rmse(m_side.predict(df.loc[te_mask, FEATS_SIDE]), yte)

# old deployed model on same held-out rows (jmaschino: 90-|pred|)
enc = old.get("encoders", {}).get("p_throws", {"R": 0, "L": 1})
ofeats = old["features"]
Xo = pd.DataFrame(index=df.index)
Xo["p_throws"] = df["p_throws"].map(enc).astype(float)
Xo["height"] = df["height"]
Xo["release_extension"] = df["release_extension"]
Xo["release_pos_z"] = df["release_pos_z"]
raw = old["model"].predict(Xo.loc[te_mask, ofeats].values)
r_old = mae_rmse(90.0 - np.abs(raw), yte)

print(f"rows {len(df):,}  test pitchers {len(test_p)}")
print(f"  OLD deployed (jmaschino, leaky-trained):  MAE={r_old[0]:.3f} RMSE={r_old[1]:.3f}")
print(f"  NEW no-side (fair, held-out):             MAE={r_no[0]:.3f} RMSE={r_no[1]:.3f}")
print(f"  NEW +release_side (fair, held-out):       MAE={r_side[0]:.3f} RMSE={r_side[1]:.3f}")
print(f"  fair feature value of release_side: {r_no[0]-r_side[0]:+.3f} MAE (new no-side -> new +side)")

# decisive apples-to-apples: BOTH trained on all data, eval same held-out rows (leaky for both)
m_all = fit(FEATS_SIDE, np.ones(len(df), dtype=bool))
r_all = mae_rmse(m_all.predict(df.loc[te_mask, FEATS_SIDE]), yte)
print(f"\n  LEAKY-vs-LEAKY (both trained on all data):")
print(f"    OLD deployed:        MAE={r_old[0]:.3f} RMSE={r_old[1]:.3f}")
print(f"    NEW +side (all data):MAE={r_all[0]:.3f} RMSE={r_all[1]:.3f}   ({r_old[0]-r_all[0]:+.3f} MAE vs OLD)")
better = r_all[0] < r_old[0]
print(f"    -> NEW {'BEATS' if better else 'does NOT beat'} OLD")

if "--save" not in sys.argv:
    print("\n(dry run — pass --save to retrain on ALL data and write the bundle)")
    sys.exit(0)

# ---- production: reuse all-data model, save (back up old) ----
bundle = {
    "model": m_all,
    "features": FEATS_SIDE,
    "height_lookup": heights,
    "jmaschino": False,
    "trained_on": "2023-2025 statcast.db",
    "note": "in-house direct arm_angle model; adds release_pos_x (release side) vs prior jmaschino bundle",
}
bak = PKL + ".pre_releaseside.bak"
if not os.path.exists(bak):
    shutil.copy2(PKL, bak)
with open(PKL, "wb") as f:
    pickle.dump(bundle, f)
print(f"\nSAVED new bundle -> {PKL}\nbacked up old -> {bak}")

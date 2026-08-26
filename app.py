"""
Flask dashboard for the Stuff+ Model.

Routes
------
GET  /                         – Home / leaderboard selector
GET  /api/leaderboard/<season> – JSON leaderboard for a season
GET  /api/player/<name>/<season> – JSON player profile
GET  /api/pitcher_names/<season> – Autocomplete list
GET  /card/<name>/<season>     – Serve card PNG
"""

import gc
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
from datetime import datetime, timedelta
from functools import lru_cache

import pandas as pd
from flask import Flask, jsonify, render_template, send_file, abort, request, Response
from flask_cors import CORS

from config import DB_PATH, PROFILES_DIR, MODEL_DIR, DATA_DIR
from storage import overrides as pitch_overrides_store

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

logger = logging.getLogger(__name__)

# Model version tracking — detect when new models are deployed and trigger a full
# rescore of pitches_spring2026_scored so profiles reflect the new model.
# _MODEL_VERSION_FILE: deployed with the code (model/artifacts/model_version.txt)
# _SCORED_VERSION_FILE: persists on the Railway Volume (data/.spring_scored_model_version)
_MODEL_VERSION_FILE  = os.path.join(MODEL_DIR, "model_version.txt")
_SCORED_VERSION_FILE = os.path.join(DATA_DIR,  ".spring_scored_model_version")
# Separate marker for the regular-season 2026 table so a model upgrade triggers a
# full rescore of pitches_2026_scored independently of the spring rescore.
_LIVE_SCORED_VERSION_FILE = os.path.join(DATA_DIR, ".live_scored_model_version")
# pitcher_id → canonical display name. Reconciles /gf names (accent-stripped,
# sometimes mis-parsed for suffixes/multi-word surnames) with the authoritative
# official-CSV names so a pitcher's games never split across spellings.
_CANON_NAMES_PATH = os.path.join(DATA_DIR, "pitcher_names.pkl")
try:
    with open(_CANON_NAMES_PATH, "rb") as _f:
        _CANON_NAMES = pickle.load(_f)
except Exception:
    _CANON_NAMES = {}

# ---------------------------------------------------------------------------
# Bootstrap historical editor tables if missing (e.g. fresh Railway volume)
# Downloads minimal per-pitch CSVs (pfx, pitch_type, stuff_plus, keys)
# ---------------------------------------------------------------------------

def _bootstrap_editor_tables() -> None:
    """Download and import pitches_{season}_editor.csv.gz for any missing seasons.

    Per-season done-sentinels prevent re-downloading already-imported tables.
    A running-sentinel (with staleness check) prevents concurrent bootstrap runs.
    45-second initial delay lets models and spring refresh settle first.
    """
    import gzip as _gz, requests as _req
    _running = "/tmp/.editor_bootstrap_running"
    # Prevent concurrent runs; treat lock as stale if older than 20 minutes
    if os.path.exists(_running):
        if time.time() - os.path.getmtime(_running) < 1200:
            return
        os.unlink(_running)
    # Wait for app startup to settle (model loading, spring refresh, etc.)
    time.sleep(45)
    try:
        open(_running, "w").close()
    except Exception:
        return

    base    = "https://arlington-atlas-trustee-ali.trycloudflare.com"
    gz_tmp  = "/tmp/editor_bootstrap.csv.gz"
    csv_tmp = "/tmp/editor_bootstrap.csv"
    try:
        for season in ["2026", "2024", "2023"]:
            tbl      = f"pitches_{season}_editor"
            done_key = f"/tmp/.editor_{season}_done"
            if os.path.exists(done_key):
                continue
            try:
                conn = sqlite3.connect(DB_PATH)
                exists = conn.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{tbl}'"
                ).fetchone()
                conn.close()
                if exists:
                    open(done_key, "w").close()
                    continue
                url = f"{base}/pitches_{season}_editor.csv.gz"
                logger.info(f"Editor bootstrap: downloading {url} ...")
                resp = _req.get(url, timeout=180, stream=True,
                                headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                with open(gz_tmp, "wb") as _f:
                    for _chunk in resp.iter_content(chunk_size=1 << 20):
                        _f.write(_chunk)
                resp.close()
                with _gz.open(gz_tmp, "rb") as src, open(csv_tmp, "wb") as dst:
                    while True:
                        blk = src.read(1 << 20)
                        if not blk:
                            break
                        dst.write(blk)
                os.unlink(gz_tmp)
                conn = sqlite3.connect(DB_PATH)
                total, first = 0, True
                for chunk_df in pd.read_csv(csv_tmp, chunksize=5000, low_memory=False):
                    chunk_df.to_sql(tbl, conn,
                                    if_exists="replace" if first else "append",
                                    index=False)
                    first = False
                    total += len(chunk_df)
                conn.close()
                os.unlink(csv_tmp)
                open(done_key, "w").close()  # mark success AFTER import
                logger.info(f"Editor bootstrap: {tbl} imported ({total:,} rows).")
            except Exception as exc:
                logger.warning(f"Editor bootstrap for {season} failed: {exc}")
                for _f in [gz_tmp, csv_tmp]:
                    if os.path.exists(_f):
                        os.unlink(_f)
    finally:
        if os.path.exists(_running):
            os.unlink(_running)

threading.Thread(target=_bootstrap_editor_tables, daemon=True).start()

# Log available DB tables at startup for diagnostics
try:
    _tables = [r[0] for r in sqlite3.connect(DB_PATH).execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    logger.info(f"DB tables at startup: {_tables}")
except Exception as _e:
    logger.warning(f"Could not list DB tables: {_e}")

# ---------------------------------------------------------------------------
# Columns stored in the scored table (subset of 215 feature cols, enough for
# player cards without carrying all engineered features in the DB).
# ---------------------------------------------------------------------------
_SCORED_COLS = [
    # Player / game identification
    "player_name", "pitcher", "p_throws", "age_pit",
    "game_pk", "game_date", "at_bat_number", "pitch_number", "inning",
    "inning_topbot", "home_team", "away_team",
    # Pitch outcome
    "pitch_type", "events", "bat_score", "post_bat_score",
    "description", "zone", "stand",
    # Raw pitch characteristics
    "release_speed", "release_spin_rate", "release_extension",
    "arm_angle", "release_pos_z", "spin_axis",
    "plate_x", "plate_z",
    # Engineered features needed for display
    "pfx_z_in", "pfx_x_arm", "vaa", "haa",
    # Core model features (needed for diagnostics / SHAP)
    "ivb_adj", "hb_adj", "deadzone_proximity",
    "ssw_pfx_z", "ssw_pfx_x_arm",
    "vaa_adj", "haa_adj",
    "speed_diff", "gyro_deg",
    # Model outputs
    "stuff_plus", "estimated_woba_using_speedangle",
]

# The background refreshes load a SLIM column set for card generation. That slim list
# must include every feature the model uses, because pitch-type grades are computed by
# scoring each type's average pitch — drop a feature and the card builder silently
# falls back to averaging per-pitch grades, which is a different (worse) number.
# Derive the model features from the trained model so this can never drift again when
# SHAPE_FEATS changes.
try:
    from model.prob_resid import SHAPE_FEATS as _MODEL_FEATS, ROUTER_FEATS as _RTR_FEATS
    # Model features + router features + the raw kinematics add_shape_features needs to
    # regenerate the Magnus/non-Magnus shape columns. spin_axis is required for the
    # Magnus split — a pitch with no 3D spin axis can't be scored (NaN grade).
    _needed = list(_MODEL_FEATS) + list(_RTR_FEATS) + \
        ["release_pos_x", "vx0", "vy0", "vz0", "ax", "ay", "az", "spin_axis", "release_extension"]
    for _f in _needed:
        if _f not in _SCORED_COLS:
            _SCORED_COLS.append(_f)
except Exception as _exc:  # pragma: no cover - defensive
    logging.getLogger(__name__).warning(f"could not append model features to _SCORED_COLS: {_exc}")

# ---------------------------------------------------------------------------
# Spring training live refresh (background thread, every 90 s)
# ---------------------------------------------------------------------------

_spring_refresh_lock = threading.Lock()
_spring_last_updated: datetime | None = None

_live_refresh_lock = threading.Lock()
_live_last_updated: datetime | None = None

# Game_pks where we have already attempted a CSV re-score to fix per-pitcher null events.
# Once attempted, we skip them in subsequent cycles to avoid infinite loops for pitchers
# who genuinely have all-null events in the CSV (e.g. mop-up relievers with no outs recorded).
# Cleared on service restart so history games are retried after new CSV data is available.
_events_fix_attempted_pks: set = set()

# ---------------------------------------------------------------------------
# Singleton predictor — load XGBoost models once, reuse across all refresh cycles
# ---------------------------------------------------------------------------
_predictor_instance = None
_predictor_lock = threading.Lock()


def _get_predictor():
    global _predictor_instance
    if _predictor_instance is None:
        with _predictor_lock:
            if _predictor_instance is None:
                from model.predict import StuffPlusPredictor
                _predictor_instance = StuffPlusPredictor()
    return _predictor_instance


def _spring_max_date() -> "str | None":
    """Return the latest game_date already in pitches_spring2026, or None."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT MAX(game_date) FROM pitches_spring2026"
        ).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _get_spring_game_pks(start_date: str = "2026-02-15") -> dict:
    """Return {game_pk: (detailedState, game_date)} for all started/completed spring games."""
    import requests as req
    from zoneinfo import ZoneInfo
    try:
        today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        url = (f"https://statsapi.mlb.com/api/v1/schedule"
               f"?sportId=1&startDate={start_date}&endDate={today_str}"
               f"&gameType=S&hydrate=team")
        data = req.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).json()
        result = {}
        for d in data.get("dates", []):
            game_date = d.get("date", "")
            for g in d.get("games", []):
                state = g.get("status", {}).get("detailedState", "")
                if state not in ("Scheduled", "Pre-Game", "Warmup"):
                    result[g["gamePk"]] = (state, game_date)
        return result
    except Exception as exc:
        logger.warning(f"Could not fetch spring schedule: {exc}")
        return {}


def _live_max_date() -> "str | None":
    """Return the latest game_date already in pitches_2026, or None."""
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT MAX(game_date) FROM pitches_2026"
        ).fetchone()
        conn.close()
        return str(row[0])[:10] if row and row[0] else None
    except Exception:
        return None


def _get_live_game_pks(start_date: str = "2026-03-26") -> dict:
    """Return {game_pk: (detailedState, game_date)} for all started/completed regular-season games."""
    import requests as req
    from zoneinfo import ZoneInfo
    try:
        today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        url = (f"https://statsapi.mlb.com/api/v1/schedule"
               f"?sportId=1&startDate={start_date}&endDate={today_str}"
               f"&gameType=R&hydrate=team")
        data = req.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).json()
        result = {}
        for d in data.get("dates", []):
            game_date = d.get("date", "")
            for g in d.get("games", []):
                state = g.get("status", {}).get("detailedState", "")
                if state not in ("Scheduled", "Pre-Game", "Warmup"):
                    result[g["gamePk"]] = (state, game_date)
        return result
    except Exception as exc:
        logger.warning(f"Could not fetch live schedule: {exc}")
        return {}


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

def _gf_name(name: str | None) -> "str | None":
    """Convert 'First Last' to 'Last, First' to match Statcast CSV convention.

    Handles generational suffixes so 'Samy Natera Jr.' becomes 'Natera Jr., Samy'
    (surname = last two tokens) rather than the misparsed 'Jr., Samy Natera'.
    """
    if not name or ',' in name:
        return name
    parts = name.split()
    if len(parts) < 2:
        return name
    if len(parts) >= 3 and parts[-1].lower().rstrip('.') in _NAME_SUFFIXES:
        first = " ".join(parts[:-2]); last = " ".join(parts[-2:])
    else:
        first = " ".join(parts[:-1]); last = parts[-1]
    return f"{last}, {first}"


def _apply_canonical_names(df: pd.DataFrame) -> pd.DataFrame:
    """Unify a pitcher's display name across data sources by pitcher id.

    /gf strips accents (Ureña→Urena) and mis-parses suffixes/multi-word surnames
    (McCullers Jr.→'Jr., Lance McCullers'), while the official CSV is correct.
    Pick the best variant per pitcher — fewest tokens after the comma (proper
    'Surname, First'), then accented — learning from CSV rows in the batch and a
    persisted lookup, so games never split across spellings.
    """
    if "pitcher" not in df.columns or "player_name" not in df.columns:
        return df

    def _has_accent(s):  return any(ord(c) > 127 for c in str(s))
    def _after_comma(n): return len(str(n).split(",", 1)[1].split()) if "," in str(n) else len(str(n).split())
    def _score(n):       return (-_after_comma(n), _has_accent(n), len(str(n)))

    canon = dict(_CANON_NAMES)
    changed = False
    sub = df.dropna(subset=["pitcher", "player_name"])
    for pid, g in sub.groupby("pitcher"):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        cands = list(g["player_name"].unique())
        if pid in canon:
            cands.append(canon[pid])
        best = max(cands, key=_score)
        if canon.get(pid) != best:
            canon[pid] = best
            changed = True
    if changed:
        try:
            with open(_CANON_NAMES_PATH, "wb") as _fh:
                pickle.dump(canon, _fh)
            _CANON_NAMES.update(canon)
        except Exception:
            pass

    def _pick(row):
        p = row.get("pitcher")
        if pd.isna(p):
            return row["player_name"]
        return canon.get(int(p), row["player_name"])
    df["player_name"] = df.apply(_pick, axis=1)
    return df


def _spin_axis_from_pfx(pfx_x, pfx_z) -> float | None:
    """Estimate spin_axis (Statcast 0-360°) from movement components (inches).

    Movement direction in the pfx_x/pfx_z plane corresponds to tilt:
      atan2(pfx_x, pfx_z) gives degrees from 12-o'clock clockwise.
    spin_axis = (deg_tilt - 180) % 360  (inverse of Statcast's +180 convention).
    """
    try:
        if pfx_x is None or pfx_z is None:
            return None
        deg_tilt = math.atan2(float(pfx_x), float(pfx_z)) * (180.0 / math.pi)
        return round((deg_tilt - 180) % 360, 1)
    except Exception:
        return None


def _compute_zone(plate_x, plate_z, sz_top, sz_bot) -> "int | None":
    """Approximate Statcast zone (1-9 in-zone, 14 out-of-zone) from plate coords."""
    try:
        px, pz = float(plate_x), float(plate_z)
        st, sb = float(sz_top), float(sz_bot)
    except (TypeError, ValueError):
        return None
    if -0.8333 <= px <= 0.8333 and sb <= pz <= st:
        x_frac = (px + 0.8333) / 1.6666  # 0=left, 1=right (catcher view)
        z_frac = (pz - sb) / max(st - sb, 0.01)  # 0=bottom, 1=top
        col = min(int(x_frac * 3), 2)
        row = min(int(z_frac * 3), 2)
        return (2 - row) * 3 + col + 1  # 1-9
    return 14  # out-of-zone


def _gf_to_df(game_pk: int) -> pd.DataFrame:
    """
    Fetch live pitch data from the Savant /gf endpoint for one game and
    return a DataFrame with Statcast-compatible column names.
    """
    import requests as req
    try:
        data = req.get(f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}",
                       timeout=20, headers={"User-Agent": "Mozilla/5.0"}).json()
    except Exception as exc:
        logger.warning(f"gf fetch failed for {game_pk}: {exc}")
        return pd.DataFrame()

    game_date = data.get("game_date", "")
    rows = []
    # team_home = home team pitching (away team bats) = Top of inning
    # team_away = away team pitching (home team bats) = Bot of inning
    for half, topbot in [("team_home", "Top"), ("team_away", "Bot")]:
        for p in data.get(half, []):
            if not p.get("pitch_type"):
                continue
            pfx_x_val = (-p.get("pfxX")) if p.get("pfxX") is not None else None
            pfx_z_val = p.get("pfxZ")
            row = {
                "player_name":        _gf_name(p.get("pitcher_name")),
                "pitcher":            p.get("pitcher"),
                "batter":             p.get("batter"),
                "pitch_type":         p.get("pitch_type"),
                "release_speed":      p.get("start_speed"),
                "pfx_x":              pfx_x_val,
                "pfx_z":              pfx_z_val,
                "spin_axis":          _spin_axis_from_pfx(pfx_x_val, pfx_z_val),
                "release_spin_rate":  p.get("spin_rate"),
                "release_extension":  p.get("extension"),
                "release_pos_x":      p.get("x0"),
                "release_pos_y":      p.get("y0"),
                "release_pos_z":      p.get("z0"),
                "ax":                 p.get("ax"),
                "ay":                 p.get("ay"),
                "az":                 p.get("az"),
                "vx0":                p.get("vx0"),
                "vy0":                p.get("vy0"),
                "vz0":                p.get("vz0"),
                "plate_x":            p.get("plate_x"),
                "plate_z":            p.get("plate_z"),
                "sz_top":             p.get("sz_top"),
                "sz_bot":             p.get("sz_bot"),
                "p_throws":           p.get("p_throws"),
                "stand":              p.get("stand"),
                "description":        p.get("description"),
                "zone":               _compute_zone(p.get("plate_x"), p.get("plate_z"), p.get("sz_top"), p.get("sz_bot")),
                "events":             p.get("events"),
                "bat_score":          p.get("bat_score"),
                "post_bat_score":     p.get("post_bat_score"),
                "balls":              p.get("pre_balls"),
                "strikes":            p.get("pre_strikes"),
                "inning":             p.get("inning"),
                "at_bat_number":      p.get("ab_number"),
                "pitch_number":       p.get("pitch_number"),
                "game_pk":            game_pk,
                "game_date":          game_date,
                "game_type":          "R",   # so engineering pipeline passes
                "inning_topbot":      topbot,
                # Batted-ball fields (None for non-contact pitches)
                "launch_speed":       (float(p["launch_speed"])  if p.get("launch_speed")  not in (None, "", "null") else None),
                "launch_angle":       (float(p["launch_angle"])  if p.get("launch_angle")  not in (None, "", "null") else None),
                "hc_x":               p.get("hc_x"),
                "hc_y":               p.get("hc_y"),
                "is_barrel":          p.get("is_barrel"),
                "estimated_woba_using_speedangle": None,  # filled by _apply_xwoba
                "home_team":          p.get("team_fielding") if topbot == "Top" else p.get("team_batting"),
                "away_team":          p.get("team_batting")  if topbot == "Top" else p.get("team_fielding"),
            }
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


_xwoba_model_cache = None
_XWOBA_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xwoba_knn.pkl")


def _get_xwoba_model():
    """Lazy-build a k-NN regressor (k=400) on (EV, LA) → xwOBA from historical data.

    xwOBAcon is defined as the average wOBA of the ~400 most similar batted balls
    by exit velocity and launch angle (k-nearest neighbors).
    Model is cached to disk after first build.
    """
    global _xwoba_model_cache
    if _xwoba_model_cache is not None:
        return _xwoba_model_cache

    import pickle
    from sklearn.neighbors import KNeighborsRegressor

    if os.path.exists(_XWOBA_MODEL_PATH):
        try:
            with open(_XWOBA_MODEL_PATH, "rb") as f:
                _xwoba_model_cache = pickle.load(f)
            logger.info("xwOBA k-NN model loaded from cache.")
            return _xwoba_model_cache
        except Exception as exc:
            logger.warning(f"Could not load xwOBA model cache: {exc}")

    # Build from historical pitches_2025
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute("""
            SELECT CAST(launch_speed AS REAL), CAST(launch_angle AS REAL),
                   CAST(estimated_woba_using_speedangle AS REAL)
            FROM pitches_2025
            WHERE launch_speed IS NOT NULL
              AND launch_angle IS NOT NULL
              AND estimated_woba_using_speedangle IS NOT NULL
        """).fetchall()
        conn.close()
    except Exception as exc:
        logger.warning(f"xwOBA model build: DB query failed: {exc}")
        return None

    if len(rows) < 1000:
        logger.warning(f"xwOBA model: only {len(rows)} historical rows — skipping build.")
        return None

    X = [[r[0], r[1]] for r in rows]
    y = [r[2] for r in rows]
    model = KNeighborsRegressor(n_neighbors=400, weights="uniform", algorithm="ball_tree")
    model.fit(X, y)

    try:
        with open(_XWOBA_MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"xwOBA k-NN model built and cached ({len(rows):,} historical BBE).")
    except Exception:
        pass

    _xwoba_model_cache = model
    return model


def _apply_xwoba(df: pd.DataFrame) -> pd.DataFrame:
    """Fill estimated_woba_using_speedangle for rows with EV+LA but no value (live /gf data)."""
    if df.empty or not {"launch_speed", "launch_angle"}.issubset(df.columns):
        return df

    if "estimated_woba_using_speedangle" not in df.columns:
        df = df.copy()
        df["estimated_woba_using_speedangle"] = None

    mask = (
        df["launch_speed"].notna() &
        df["launch_angle"].notna() &
        df["estimated_woba_using_speedangle"].isna()
    )
    if not mask.any():
        return df

    model = _get_xwoba_model()
    if model is None:
        return df

    X_pred = df.loc[mask, ["launch_speed", "launch_angle"]].astype(float).values
    preds = model.predict(X_pred)
    # Modify in-place — callers always reassign (e.g. df = _apply_xwoba(df)),
    # so no copy needed. Avoids duplicating large DataFrames in memory.
    df.loc[mask, "estimated_woba_using_speedangle"] = preds.round(3)
    logger.info(f"xwOBA: predicted {int(mask.sum())} batted-ball values via k-NN.")
    return df


def _refresh_spring() -> None:
    """Pull any new spring-training pitches since last stored date, rescore, rebuild profiles."""
    global _spring_last_updated
    if not _spring_refresh_lock.acquire(blocking=False):
        return  # previous refresh still running

    try:
        import io, requests as req
        from scraper.statcast_scraper import load_from_db, save_to_db
        from model.predict import StuffPlusPredictor
        from profiles.player_cards import generate_all_cards

        from zoneinfo import ZoneInfo
        from datetime import timedelta
        today = datetime.now(ZoneInfo("America/New_York"))
        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        refetch_cutoff_str = (today - timedelta(days=4)).strftime("%Y-%m-%d")

        # ── 0. Model version check — force full rescore when new models deployed ──
        # Compares the model_version.txt shipped with the deployment against the
        # version recorded when the scored table was last (re)built on this Volume.
        # If they differ, the old scored data was produced by a different model and
        # must be discarded so every pitch gets re-scored with the new model.
        try:
            _deployed_ver = open(_MODEL_VERSION_FILE).read().strip() if os.path.exists(_MODEL_VERSION_FILE) else ""
            _scored_ver   = open(_SCORED_VERSION_FILE).read().strip() if os.path.exists(_SCORED_VERSION_FILE) else ""
            if _deployed_ver and _deployed_ver != _scored_ver:
                logger.info(
                    f"Spring refresh: model version changed "
                    f"({_scored_ver[:8] or 'none'} → {_deployed_ver[:8]}) — "
                    f"batch-rescoring all spring2026 pitches with new model."
                )
                # Drop the stale scored table so nothing reads old scores.
                _conn_ver = sqlite3.connect(DB_PATH)
                _conn_ver.execute("DROP TABLE IF EXISTS pitches_spring2026_scored")
                _conn_ver.commit()
                # Score in batches of 25 game_pks to avoid OOM.
                _all_pks_ver = [r[0] for r in _conn_ver.execute(
                    "SELECT DISTINCT CAST(game_pk AS INTEGER) FROM pitches_spring2026 "
                    "WHERE game_pk IS NOT NULL ORDER BY game_pk"
                ).fetchall()]
                _conn_ver.close()
                logger.info(f"Spring refresh: rescoring {len(_all_pks_ver)} games in batches.")
                _VER_BATCH = 25
                _pred_ver = _get_predictor()
                _first_ver = True
                for _vi in range(0, len(_all_pks_ver), _VER_BATCH):
                    _vpks = _all_pks_ver[_vi:_vi + _VER_BATCH]
                    _vph  = ",".join("?" * len(_vpks))
                    try:
                        _conn_vr = sqlite3.connect(DB_PATH)
                        _df_vr = pd.read_sql_query(
                            f"SELECT * FROM pitches_spring2026 "
                            f"WHERE CAST(game_pk AS INTEGER) IN ({_vph})",
                            _conn_vr, params=_vpks
                        )
                        _conn_vr.close()
                        if _df_vr.empty:
                            continue
                        _df_vr = _apply_xwoba(_df_vr)
                        _sc_vr = _pred_ver.predict(_df_vr)
                        del _df_vr
                        gc.collect()
                        _slim_vr = [c for c in _SCORED_COLS if c in _sc_vr.columns]
                        _sc_vr   = _sc_vr[_slim_vr].copy()
                        _conn_vw = sqlite3.connect(DB_PATH)
                        _sc_vr.to_sql("pitches_spring2026_scored", _conn_vw,
                                      if_exists="replace" if _first_ver else "append",
                                      index=False)
                        _conn_vw.commit()
                        _conn_vw.close()
                        del _sc_vr
                        gc.collect()
                        _first_ver = False
                        logger.info(
                            f"Spring refresh: version rescore batch "
                            f"{_vi // _VER_BATCH + 1}/"
                            f"{(len(_all_pks_ver) + _VER_BATCH - 1) // _VER_BATCH} done."
                        )
                    except Exception as _vbatch_exc:
                        logger.warning(f"Spring refresh: version rescore batch {_vi} failed: {_vbatch_exc}")
                # Generate profiles from the fully-rescored table.
                _conn_vc = sqlite3.connect(DB_PATH)
                try:
                    _vtbl_cols = [r[1] for r in _conn_vc.execute(
                        "PRAGMA table_info(pitches_spring2026_scored)"
                    ).fetchall()]
                    _vsel = [c for c in _SCORED_COLS if c in _vtbl_cols]
                    _df_vc = pd.read_sql(
                        f"SELECT {', '.join(_vsel)} FROM pitches_spring2026_scored",
                        _conn_vc
                    )
                finally:
                    _conn_vc.close()
                generate_all_cards(_df_vc, season='spring2026', skip_png=True)
                del _df_vc
                gc.collect()
                _cached_leaderboard.cache_clear()
                _spring_last_updated = datetime.now()
                logger.info("Spring refresh: full model-upgrade rescore complete.")
                try:
                    open(_SCORED_VERSION_FILE, "w").write(_deployed_ver)
                except Exception:
                    pass
                return
        except Exception as _ver_exc:
            logger.warning(f"Spring refresh: version check/rescore error ({_ver_exc}); continuing with normal flow.")

        # ── 1. CSV bulk export for completed/historical games ──────────────
        # Check if any PITCHER in the scored table has all-null events — this
        # means they were added via /gf only and need their historical stats fixed.
        # Events are pitcher-specific: a game may pass a game-level check because
        # OTHER pitchers in that game have events, while a new pitcher doesn't.
        max_date = _spring_max_date()
        try:
            _conn_pre = sqlite3.connect(DB_PATH)
            _sc_pre = _conn_pre.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pitches_spring2026_scored'"
            ).fetchone()
            if _sc_pre:
                _sc_cols_pre = {r[1] for r in _conn_pre.execute(
                    "PRAGMA table_info(pitches_spring2026_scored)"
                ).fetchall()}
                if "events" in _sc_cols_pre:
                    # Game_pks where any pitcher has ALL null events — excluding today's games
                    # (those are in-progress and inherently lack events; they'll be fixed when
                    # the next day's CSV is available) and games already attempted this session
                    # (avoids infinite loops for mop-up relievers whose CSV data is also null).
                    _needs_events_game_pks = {int(r[0]) for r in _conn_pre.execute(
                        "SELECT DISTINCT CAST(game_pk AS INTEGER) "
                        "FROM pitches_spring2026_scored "
                        "WHERE game_pk IS NOT NULL AND game_date < ? "
                        "GROUP BY CAST(game_pk AS INTEGER), player_name "
                        "HAVING SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END) = 0",
                        (today_str,)
                    ).fetchall()} - _events_fix_attempted_pks
                else:
                    _needs_events_game_pks = set()
            else:
                _needs_events_game_pks = set()
            _conn_pre.close()
            _scored_needs_events = len(_needs_events_game_pks) > 0
        except Exception:
            _needs_events_game_pks = set()
            _scored_needs_events = False
        fetch_from = (
            "2026-02-15" if _scored_needs_events else
            (datetime.strptime(max_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            if max_date else "2026-02-15"
        )
        if _scored_needs_events:
            logger.info(f"Spring refresh: {len(_needs_events_game_pks)} games have pitchers "
                        f"with all-null events in scored table — fetching full CSV.")
        fetch_to_lt = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        csv_url = (
            "https://baseballsavant.mlb.com/statcast_search/csv"
            f"?all=true&hfGT=S%7C&hfSea=2026%7C"
            f"&game_date_gt={fetch_from}&game_date_lt={fetch_to_lt}"
            "&player_type=pitcher&type=details"
            "&min_pitches=0&min_results=0&sort_col=pitches&sort_order=desc"
        )
        csv_df = pd.DataFrame()
        try:
            resp = req.get(csv_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            csv_df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if "pitch_type" in csv_df.columns:
                csv_df = csv_df[csv_df["pitch_type"].notna() & (csv_df["pitch_type"] != "pitch_type")]
            # Force game_type="R" so feature engineering doesn't filter spring data
            if not csv_df.empty:
                csv_df["game_type"] = "R"
        except Exception as exc:
            logger.warning(f"CSV fetch failed: {exc}")

        # ── 2. What's already in the DB? ──────────────────────────────────
        try:
            conn_q = sqlite3.connect(DB_PATH)
            pk_rows = conn_q.execute(
                "SELECT DISTINCT game_pk, game_date FROM pitches_spring2026"
            ).fetchall()
            conn_q.close()
            existing_pks           = {r[0] for r in pk_rows if r[0] is not None}
            today_existing_pks     = {r[0] for r in pk_rows if r[0] is not None and str(r[1]) == today_str}
            yesterday_existing_pks = {r[0] for r in pk_rows if r[0] is not None and str(r[1]) == yesterday_str}
            recent_existing_pks    = {r[0] for r in pk_rows if r[0] is not None and str(r[1]) >= refetch_cutoff_str}
        except Exception:
            existing_pks = set()
            today_existing_pks = set()
            yesterday_existing_pks = set()
            recent_existing_pks = set()

        # ── 3. /gf for all missing game_pks + today's in-progress ─────────
        all_spring_pks = _get_spring_game_pks()  # {pk: (state, date)}
        csv_game_pks = (
            set(csv_df["game_pk"].dropna().astype(int))
            if not csv_df.empty and "game_pk" in csv_df.columns else set()
        )
        pks_to_fetch = [
            pk for pk, (state, gdate) in all_spring_pks.items()
            if (pk not in existing_pks and pk not in csv_game_pks) or gdate >= refetch_cutoff_str
        ]
        gf_frames = [_gf_to_df(pk) for pk in pks_to_fetch]
        _non_empty_gf = [f for f in gf_frames if not f.empty]
        gf_df = pd.concat(_non_empty_gf, ignore_index=True) if _non_empty_gf else pd.DataFrame()
        if not gf_df.empty:
            logger.info(f"/gf: {len(gf_df):,} pitches from {len(pks_to_fetch)} games")

        # Which game_pks already have events data in the SCORED table?
        # We check the scored table (not raw) because player cards are generated
        # from the scored table. The scored table may be missing events if it was
        # built from a pre-scored CSV that lacked those columns.
        try:
            conn_z = sqlite3.connect(DB_PATH)
            _sc_tbl_exists = conn_z.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pitches_spring2026_scored'"
            ).fetchone() is not None
            if _sc_tbl_exists:
                _sc_cols = {r[1] for r in conn_z.execute(
                    "PRAGMA table_info(pitches_spring2026_scored)"
                ).fetchall()}
                if "events" in _sc_cols:
                    events_ok_pks = {int(r[0]) for r in conn_z.execute(
                        "SELECT DISTINCT CAST(game_pk AS INTEGER) FROM pitches_spring2026_scored "
                        "WHERE game_pk IS NOT NULL "
                        "GROUP BY game_pk HAVING SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END) > 0"
                    ).fetchall()}
                else:
                    events_ok_pks = set()  # scored table has no events column — rescore all
            else:
                events_ok_pks = set()
            conn_z.close()
        except Exception:
            events_ok_pks = set()

        # ── Re-score from raw: fix scored rows missing events ─────────────
        # The scored table may have been built from a pre-scored CSV that lacked
        # events/bat_score/post_bat_score. The raw table already has correct events
        # (from old CSV loads). Re-score those games from raw in batches of 20 so
        # IP and ERA populate without re-downloading the full spring CSV.
        _MAX_RAW_RESCORE = 20
        try:
            conn_rr = sqlite3.connect(DB_PATH)
            _raw_events_pks = {int(r[0]) for r in conn_rr.execute(
                "SELECT DISTINCT CAST(game_pk AS INTEGER) FROM pitches_spring2026 "
                "WHERE game_pk IS NOT NULL AND events IS NOT NULL"
            ).fetchall()}
            conn_rr.close()
            _needs_rescore_pks = _raw_events_pks - events_ok_pks - today_existing_pks
            if _needs_rescore_pks:
                _rr_batch = list(_needs_rescore_pks)[:_MAX_RAW_RESCORE]
                ph_rr = ",".join("?" * len(_rr_batch))
                conn_rr2 = sqlite3.connect(DB_PATH)
                df_rr = pd.read_sql_query(
                    f"SELECT * FROM pitches_spring2026 "
                    f"WHERE CAST(game_pk AS INTEGER) IN ({ph_rr})",
                    conn_rr2, params=[int(p) for p in _rr_batch])
                conn_rr2.close()
                if not df_rr.empty:
                    if "player_name" in df_rr.columns:
                        df_rr["player_name"] = df_rr["player_name"].apply(_gf_name)
                    df_rr = _apply_xwoba(df_rr)
                    _pred_rr = _get_predictor()
                    sc_rr = _pred_rr.predict(df_rr)
                    slim_rr = [c for c in _SCORED_COLS if c in sc_rr.columns]
                    sc_rr = sc_rr[slim_rr].copy()
                    conn_rr3 = sqlite3.connect(DB_PATH)
                    conn_rr3.execute(
                        f"DELETE FROM pitches_spring2026_scored "
                        f"WHERE CAST(game_pk AS INTEGER) IN ({ph_rr})",
                        [int(p) for p in _rr_batch])
                    conn_rr3.commit()
                    sc_rr.to_sql("pitches_spring2026_scored", conn_rr3,
                                 if_exists="append", index=False)
                    conn_rr3.commit()
                    conn_rr3.close()
                    events_ok_pks.update(set(_rr_batch))
                    logger.info(f"Spring re-score from raw: {len(_rr_batch)}/{len(_needs_rescore_pks)} "
                                f"games fixed (events now populated).")
                del df_rr
                gc.collect()
        except Exception as _rr_exc:
            logger.warning(f"Spring re-score from raw failed: {_rr_exc}")

        # From CSV: brand-new games, today's in-progress, and games missing events data.
        # Batch historical re-scores to MAX_RESCORE_PER_CYCLE games to avoid OOM.
        # Today's in-progress games are always included regardless of batch limit.
        _MAX_RESCORE = 20
        to_add_csv = pd.DataFrame()
        csv_pks: set = set()
        if not csv_df.empty and "game_pk" in csv_df.columns:
            _today_mask  = csv_df["game_pk"].isin(today_existing_pks)
            _stale_mask  = ~csv_df["game_pk"].isin(events_ok_pks) & ~_today_mask
            # Batch stale historical games to avoid scoring tens of thousands of rows at once
            _stale_pks   = list(csv_df.loc[_stale_mask, "game_pk"].dropna().astype(int).unique())
            _batch_pks   = set(_stale_pks[:_MAX_RESCORE])
            # Also force-include game_pks where any pitcher has ALL null events in scored.
            # These games pass the game-level events_ok_pks check (other pitchers have events)
            # but that specific pitcher's rows need to be replaced from the full CSV.
            # Include ALL rows for those game_pks so the DELETE+INSERT is a complete game
            # re-score (not a partial one that destroys other pitchers' scored data).
            # Use remaining batch capacity to bound memory usage.
            if _scored_needs_events:
                _remaining = max(0, _MAX_RESCORE - len(_batch_pks))
                # Only include games that are actually in the CSV (can be fixed).
                # Games not in the CSV (exhibition, non-Statcast) are skipped silently.
                _csv_pks_available = set(csv_df["game_pk"].dropna().astype(int)) if "game_pk" in csv_df.columns else set()
                _batch_needs_events = set(
                    sorted((_needs_events_game_pks - _batch_pks) & _csv_pks_available)[:_remaining]
                )
                # Mark these as attempted so they're skipped next cycle even if still null
                _events_fix_attempted_pks.update(_batch_needs_events)
                # Also permanently skip games not in the CSV at all — they're exhibition/non-Statcast
                # and will never be fixable from the CSV regardless of retries.
                _events_fix_attempted_pks.update(_needs_events_game_pks - _csv_pks_available)
            else:
                _batch_needs_events = set()
            needs_update = _today_mask | csv_df["game_pk"].isin(_batch_pks | _batch_needs_events)
            to_add_csv   = csv_df[needs_update].copy()
            csv_pks      = set(to_add_csv["game_pk"].dropna().astype(int))
            if _stale_pks:
                logger.info(f"Spring refresh: {len(_stale_pks)} CSV games need events re-score; "
                            f"processing {len(_batch_pks)} this cycle.")
            if _scored_needs_events and _batch_needs_events:
                logger.info(f"Spring refresh: forcing game re-score for {len(_batch_needs_events)} "
                            f"games with per-pitcher null events (of {len(_needs_events_game_pks)} total).")

        # gf_df first so CSV rows come last — dedup keep="last" lets CSV win
        to_add = pd.concat([gf_df, to_add_csv], ignore_index=True)
        if to_add.empty:
            # No new pitches — but re-score if the scored table is missing
            # (e.g. a previous scoring attempt crashed mid-run).
            try:
                conn_chk = sqlite3.connect(DB_PATH)
                tbl = conn_chk.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='pitches_spring2026_scored'"
                ).fetchone()
                row_count = conn_chk.execute(
                    "SELECT COUNT(*) FROM pitches_spring2026_scored"
                ).fetchone()[0] if tbl else 0
                conn_chk.close()
                has_scored = tbl and row_count > 0
            except Exception:
                has_scored = False
            if has_scored:
                logger.info("Spring refresh: no new data.")
                return
            _sentinel = "/tmp/.spring_rescore_attempted"
            if os.path.exists(_sentinel):
                logger.info("Spring refresh: scored table empty; re-score already attempted this session.")
                return
            open(_sentinel, "w").close()   # write sentinel before starting (survives worker restarts)
            # Try downloading the pre-scored CSV from the local tunnel first
            # (avoids expensive in-process XGBoost scoring that OOMs on Railway).
            _csv_url = "https://arlington-atlas-trustee-ali.trycloudflare.com/spring2026_scored.csv.gz"
            _tmp_csv = "/tmp/spring2026_scored.csv.gz"
            try:
                import requests as _req
                logger.info("Spring refresh: downloading pre-scored CSV to disk...")
                resp = _req.get(_csv_url, timeout=300, stream=True)
                resp.raise_for_status()
                with open(_tmp_csv, "wb") as _f:
                    for _chunk in resp.iter_content(chunk_size=1 << 20):
                        _f.write(_chunk)
                resp.close()
                logger.info("Spring refresh: CSV downloaded; importing in chunks...")
                conn_imp = sqlite3.connect(DB_PATH)
                _total = 0
                _first = True
                for _chunk_df in pd.read_csv(_tmp_csv, compression="gzip", chunksize=5000, low_memory=False):
                    _chunk_df.to_sql("pitches_spring2026_scored", conn_imp,
                                     if_exists="replace" if _first else "append", index=False)
                    _first = False
                    _total += len(_chunk_df)
                conn_imp.close()
                os.unlink(_tmp_csv)
                _cached_leaderboard.cache_clear()
                _spring_last_updated = datetime.now()
                logger.info(f"Spring2026 scored table imported: {_total:,} rows from CSV.")
                if os.path.exists(_MODEL_VERSION_FILE):
                    try:
                        open(_SCORED_VERSION_FILE, "w").write(open(_MODEL_VERSION_FILE).read())
                    except Exception:
                        pass
                return
            except Exception as csv_exc:
                logger.warning(f"Spring refresh: CSV import failed ({csv_exc}), falling back to in-process scoring.")
                if os.path.exists(_tmp_csv):
                    os.unlink(_tmp_csv)
            try:
                existing_all = load_from_db('pitches_spring2026')
            except Exception:
                logger.info("Spring refresh: no new data and no existing data.")
                return
            if existing_all.empty:
                logger.info("Spring refresh: no new data.")
                return
            predictor = _get_predictor()
            df_scored = predictor.predict(existing_all)
            save_to_db(df_scored, 'pitches_spring2026_scored', replace=True)
            generate_all_cards(df_scored, season='spring2026', skip_png=True)
            _cached_leaderboard.cache_clear()
            _spring_last_updated = datetime.now()
            logger.info("Spring2026 profiles re-scored successfully.")
            if os.path.exists(_MODEL_VERSION_FILE):
                try:
                    open(_SCORED_VERSION_FILE, "w").write(open(_MODEL_VERSION_FILE).read())
                except Exception:
                    pass
            return

        # ── 4. Update raw table via SQL DELETE + INSERT ────────────────────
        # Avoids loading the full 70K-row table into memory each cycle.
        _spring_replace_pks = recent_existing_pks | csv_pks
        # Normalise to_add before persisting
        if "player_name" in to_add.columns:
            to_add["player_name"] = to_add["player_name"].apply(_gf_name)
        if "game_type" in to_add.columns:
            to_add["game_type"] = "R"
        conn_raw = sqlite3.connect(DB_PATH)
        try:
            if _spring_replace_pks:
                ph = ",".join("?" * len(_spring_replace_pks))
                conn_raw.execute(
                    f"DELETE FROM pitches_spring2026 WHERE CAST(game_pk AS INTEGER) IN ({ph})",
                    [int(p) for p in _spring_replace_pks],
                )
                conn_raw.commit()
            to_add.to_sql("pitches_spring2026", conn_raw, if_exists="append", index=False)
            conn_raw.commit()
            row_count = conn_raw.execute(
                "SELECT COUNT(*) FROM pitches_spring2026"
            ).fetchone()[0]
        finally:
            conn_raw.close()
        logger.info(f"Spring refresh: +{len(to_add):,} pitches → {row_count:,} total")
        gc.collect()

        # ── 5. Score new pitches, update scored table, regenerate cards ────
        # Free CSV/gf DataFrames before loading XGBoost models — saves ~60 MB.
        del csv_df, gf_df, to_add_csv
        gc.collect()

        to_add = _apply_xwoba(to_add)
        predictor = _get_predictor()
        new_scored = predictor.predict(to_add)
        del to_add
        gc.collect()

        # Slim down to only the columns needed for player cards (~30 vs 215).
        # This keeps the scored table small so loading it never OOMs.
        slim_cols = [c for c in _SCORED_COLS if c in new_scored.columns]
        new_scored = new_scored[slim_cols].copy()

        # Update scored table via SQL DELETE + INSERT — never load the full table.
        conn_sc = sqlite3.connect(DB_PATH)
        try:
            has_tbl = conn_sc.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='pitches_spring2026_scored'"
            ).fetchone() is not None
            if has_tbl and _spring_replace_pks:
                ph = ",".join("?" * len(_spring_replace_pks))
                conn_sc.execute(
                    f"DELETE FROM pitches_spring2026_scored "
                    f"WHERE CAST(game_pk AS INTEGER) IN ({ph})",
                    [int(p) for p in _spring_replace_pks],
                )
                conn_sc.commit()
            new_scored.to_sql(
                "pitches_spring2026_scored", conn_sc,
                if_exists="append" if has_tbl else "replace",
                index=False,
            )
            conn_sc.commit()
        finally:
            conn_sc.close()
        del new_scored
        gc.collect()

        # Load slim scored table for card generation (only needed columns)
        conn_sc2 = sqlite3.connect(DB_PATH)
        try:
            _tbl_cols = [r[1] for r in conn_sc2.execute(
                "PRAGMA table_info(pitches_spring2026_scored)"
            ).fetchall()]
            _sel = [c for c in _SCORED_COLS if c in _tbl_cols]
            df_for_cards = pd.read_sql(
                f"SELECT {', '.join(_sel)} FROM pitches_spring2026_scored",
                conn_sc2,
            )
        finally:
            conn_sc2.close()
        generate_all_cards(df_for_cards, season='spring2026', skip_png=True)
        del df_for_cards
        gc.collect()

        _cached_leaderboard.cache_clear()
        _spring_last_updated = datetime.now()
        logger.info("Spring2026 profiles refreshed.")
        if os.path.exists(_MODEL_VERSION_FILE):
            try:
                open(_SCORED_VERSION_FILE, "w").write(open(_MODEL_VERSION_FILE).read())
            except Exception:
                pass

    except Exception as exc:
        logger.warning(f"Spring refresh failed: {exc}")
    finally:
        _spring_refresh_lock.release()


def _refresh_live() -> None:
    """Pull any new regular-season 2026 pitches since last stored date, rescore, rebuild profiles."""
    global _live_last_updated
    from zoneinfo import ZoneInfo
    today = datetime.now(ZoneInfo("America/New_York"))
    today_str = today.strftime("%Y-%m-%d")

    # Only run during regular season
    if today_str < "2026-03-26":
        return

    if not _live_refresh_lock.acquire(blocking=False):
        return  # previous refresh still running

    try:
        import io, requests as req
        from scraper.statcast_scraper import load_from_db, save_to_db
        from profiles.player_cards import generate_all_cards

        # ── 0. Model version check — force full rescore when new models deployed ──
        # Mirrors the spring trigger but for the regular-season table. If the
        # deployed model_version.txt differs from the version recorded when
        # pitches_2026_scored was last (re)built, every 2026 pitch is re-scored.
        try:
            _deployed_ver = open(_MODEL_VERSION_FILE).read().strip() if os.path.exists(_MODEL_VERSION_FILE) else ""
            _scored_ver   = open(_LIVE_SCORED_VERSION_FILE).read().strip() if os.path.exists(_LIVE_SCORED_VERSION_FILE) else ""
            if _deployed_ver and _deployed_ver != _scored_ver:
                logger.info(
                    f"Live refresh: model version changed "
                    f"({_scored_ver[:8] or 'none'} → {_deployed_ver[:8]}) — "
                    f"batch-rescoring all 2026 pitches with new model."
                )
                _conn_ver = sqlite3.connect(DB_PATH)
                _conn_ver.execute("DROP TABLE IF EXISTS pitches_2026_scored")
                _conn_ver.commit()
                _all_pks_ver = [r[0] for r in _conn_ver.execute(
                    "SELECT DISTINCT CAST(game_pk AS INTEGER) FROM pitches_2026 "
                    "WHERE game_pk IS NOT NULL ORDER BY game_pk"
                ).fetchall()]
                _conn_ver.close()
                logger.info(f"Live refresh: rescoring {len(_all_pks_ver)} games in batches.")
                _VER_BATCH = 25
                _pred_ver = _get_predictor()
                _first_ver = True
                for _vi in range(0, len(_all_pks_ver), _VER_BATCH):
                    _vpks = _all_pks_ver[_vi:_vi + _VER_BATCH]
                    _vph  = ",".join("?" * len(_vpks))
                    try:
                        _conn_vr = sqlite3.connect(DB_PATH)
                        _df_vr = pd.read_sql_query(
                            f"SELECT * FROM pitches_2026 "
                            f"WHERE CAST(game_pk AS INTEGER) IN ({_vph})",
                            _conn_vr, params=_vpks
                        )
                        _conn_vr.close()
                        if _df_vr.empty:
                            continue
                        _df_vr = _apply_xwoba(_df_vr)
                        _sc_vr = _pred_ver.predict(_df_vr)
                        del _df_vr
                        gc.collect()
                        _slim_vr = [c for c in _SCORED_COLS if c in _sc_vr.columns]
                        _sc_vr   = _sc_vr[_slim_vr].copy()
                        _conn_vw = sqlite3.connect(DB_PATH)
                        _sc_vr.to_sql("pitches_2026_scored", _conn_vw,
                                      if_exists="replace" if _first_ver else "append",
                                      index=False)
                        _conn_vw.commit()
                        _conn_vw.close()
                        del _sc_vr
                        gc.collect()
                        _first_ver = False
                        logger.info(
                            f"Live refresh: version rescore batch "
                            f"{_vi // _VER_BATCH + 1}/"
                            f"{(len(_all_pks_ver) + _VER_BATCH - 1) // _VER_BATCH} done."
                        )
                    except Exception as _vbatch_exc:
                        logger.warning(f"Live refresh: version rescore batch {_vi} failed: {_vbatch_exc}")
                _conn_vc = sqlite3.connect(DB_PATH)
                try:
                    _vtbl_cols = [r[1] for r in _conn_vc.execute(
                        "PRAGMA table_info(pitches_2026_scored)"
                    ).fetchall()]
                    _vsel = [c for c in _SCORED_COLS if c in _vtbl_cols]
                    _df_vc = pd.read_sql(
                        f"SELECT {', '.join(_vsel)} FROM pitches_2026_scored",
                        _conn_vc
                    )
                finally:
                    _conn_vc.close()
                generate_all_cards(_df_vc, season='2026', skip_png=True)
                del _df_vc
                gc.collect()
                _cached_leaderboard.cache_clear()
                _live_last_updated = datetime.now()
                logger.info("Live refresh: full model-upgrade rescore complete.")
                try:
                    open(_LIVE_SCORED_VERSION_FILE, "w").write(_deployed_ver)
                except Exception:
                    pass
                return
        except Exception as _ver_exc:
            logger.warning(f"Live refresh: version check/rescore error ({_ver_exc}); continuing with normal flow.")

        # ── 1. CSV bulk export for completed/historical games ──────────────
        max_date = _live_max_date()
        fetch_from = (
            (datetime.strptime(max_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            if max_date else "2026-03-26"
        )
        fetch_to_lt = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        csv_url = (
            "https://baseballsavant.mlb.com/statcast_search/csv"
            f"?all=true&hfGT=R%7C&hfSea=2026%7C"
            f"&game_date_gt={fetch_from}&game_date_lt={fetch_to_lt}"
            "&player_type=pitcher&type=details"
            "&min_pitches=0&min_results=0&sort_col=pitches&sort_order=desc"
        )
        csv_df = pd.DataFrame()
        try:
            resp = req.get(csv_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            csv_df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if "pitch_type" in csv_df.columns:
                csv_df = csv_df[csv_df["pitch_type"].notna() & (csv_df["pitch_type"] != "pitch_type")]
        except Exception as exc:
            logger.warning(f"Live CSV fetch failed: {exc}")

        # ── 2. What's already in the DB? ──────────────────────────────────
        try:
            conn_q = sqlite3.connect(DB_PATH)
            pk_rows = conn_q.execute(
                "SELECT DISTINCT game_pk, game_date FROM pitches_2026"
            ).fetchall()
            conn_q.close()
            existing_pks       = {r[0] for r in pk_rows if r[0] is not None}
            today_existing_pks = {r[0] for r in pk_rows if r[0] is not None and str(r[1]) == today_str}
        except Exception:
            existing_pks = set()
            today_existing_pks = set()

        # ── 3. /gf for all missing game_pks + today's in-progress ─────────
        all_live_pks = _get_live_game_pks()  # {pk: (state, date)}
        csv_game_pks = (
            set(csv_df["game_pk"].dropna().astype(int))
            if not csv_df.empty and "game_pk" in csv_df.columns else set()
        )
        pks_to_fetch = [
            pk for pk, (state, gdate) in all_live_pks.items()
            if (pk not in existing_pks and pk not in csv_game_pks) or gdate == today_str
        ]
        gf_frames = [_gf_to_df(pk) for pk in pks_to_fetch]
        _non_empty_gf = [f for f in gf_frames if not f.empty]
        gf_df = pd.concat(_non_empty_gf, ignore_index=True) if _non_empty_gf else pd.DataFrame()
        if not gf_df.empty:
            logger.info(f"/gf: {len(gf_df):,} pitches from {len(pks_to_fetch)} games")

        # Which game_pks already have events data in the SCORED table?
        try:
            conn_z = sqlite3.connect(DB_PATH)
            _sc_tbl_exists = conn_z.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pitches_2026_scored'"
            ).fetchone() is not None
            if _sc_tbl_exists:
                _sc_cols = {r[1] for r in conn_z.execute(
                    "PRAGMA table_info(pitches_2026_scored)"
                ).fetchall()}
                if "events" in _sc_cols:
                    events_ok_pks = {int(r[0]) for r in conn_z.execute(
                        "SELECT DISTINCT CAST(game_pk AS INTEGER) FROM pitches_2026_scored "
                        "WHERE game_pk IS NOT NULL "
                        "GROUP BY game_pk HAVING SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END) > 0"
                    ).fetchall()}
                else:
                    events_ok_pks = set()
            else:
                events_ok_pks = set()
            conn_z.close()
        except Exception:
            events_ok_pks = set()

        # From CSV: brand-new games, today's in-progress, and games missing events data.
        _MAX_RESCORE = 20
        to_add_csv = pd.DataFrame()
        csv_pks: set = set()
        if not csv_df.empty and "game_pk" in csv_df.columns:
            _today_mask  = csv_df["game_pk"].isin(today_existing_pks)
            _stale_mask  = ~csv_df["game_pk"].isin(events_ok_pks) & ~_today_mask
            _stale_pks   = list(csv_df.loc[_stale_mask, "game_pk"].dropna().astype(int).unique())
            _batch_pks   = set(_stale_pks[:_MAX_RESCORE])
            needs_update = _today_mask | csv_df["game_pk"].isin(_batch_pks)
            to_add_csv   = csv_df[needs_update].copy()
            csv_pks      = set(to_add_csv["game_pk"].dropna().astype(int))
            if _stale_pks:
                logger.info(f"Live refresh: {len(_stale_pks)} games need events re-score; "
                            f"processing {len(_batch_pks)} this cycle.")

        # gf_df first so CSV rows come last — dedup keep="last" lets CSV win
        to_add = pd.concat([gf_df, to_add_csv], ignore_index=True)
        if to_add.empty:
            # No new pitches — check if scored table exists
            try:
                conn_chk = sqlite3.connect(DB_PATH)
                tbl = conn_chk.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='pitches_2026_scored'"
                ).fetchone()
                row_count = conn_chk.execute(
                    "SELECT COUNT(*) FROM pitches_2026_scored"
                ).fetchone()[0] if tbl else 0
                conn_chk.close()
                has_scored = tbl and row_count > 0
            except Exception:
                has_scored = False
            if has_scored:
                logger.info("Live refresh: no new data.")
                return
            # No scored table yet and no new pitches
            try:
                existing_all = load_from_db('pitches_2026')
            except Exception:
                logger.info("Live refresh: no new data and no existing data.")
                return
            if existing_all.empty:
                logger.info("Live refresh: no new data.")
                return
            predictor = _get_predictor()
            df_scored = predictor.predict(existing_all)
            del existing_all
            gc.collect()
            slim_cols = [c for c in _SCORED_COLS if c in df_scored.columns]
            df_scored = df_scored[slim_cols].copy()
            save_to_db(df_scored, 'pitches_2026_scored', replace=True)
            generate_all_cards(df_scored, season='2026', skip_png=True)
            del df_scored
            gc.collect()
            _cached_leaderboard.cache_clear()
            _live_last_updated = datetime.now()
            logger.info("2026 profiles re-scored successfully.")
            return

        # ── 4. Update raw table via SQL DELETE + INSERT ────────────────────
        # Avoids loading the full raw table into memory each cycle.
        _live_replace_pks = today_existing_pks | csv_pks
        if "player_name" in to_add.columns:
            to_add["player_name"] = to_add["player_name"].apply(_gf_name)
            to_add = _apply_canonical_names(to_add)   # reconcile /gf vs CSV spellings by pitcher id
        conn_raw = sqlite3.connect(DB_PATH)
        try:
            _raw_exists = conn_raw.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pitches_2026'"
            ).fetchone() is not None
            if _raw_exists:
                # Ensure the arm_angle column exists so the real (CSV) arm angle is
                # stored when completed games are re-inserted (auto-backfill).
                _cols0 = [r[1] for r in conn_raw.execute("PRAGMA table_info(pitches_2026)").fetchall()]
                if "arm_angle" not in _cols0:
                    conn_raw.execute("ALTER TABLE pitches_2026 ADD COLUMN arm_angle REAL")
                    conn_raw.commit()
            if _live_replace_pks and _raw_exists:
                ph = ",".join("?" * len(_live_replace_pks))
                conn_raw.execute(
                    f"DELETE FROM pitches_2026 WHERE CAST(game_pk AS INTEGER) IN ({ph})",
                    [int(p) for p in _live_replace_pks],
                )
                conn_raw.commit()
            # Align columns with existing table schema to avoid insert errors
            if _raw_exists:
                _existing_cols = [r[1] for r in conn_raw.execute("PRAGMA table_info(pitches_2026)").fetchall()]
                _shared = [c for c in _existing_cols if c in to_add.columns]
                to_add_raw = to_add[_shared]
            else:
                to_add_raw = to_add
            to_add_raw.to_sql("pitches_2026", conn_raw, if_exists="append", index=False)
            conn_raw.commit()
            row_count = conn_raw.execute(
                "SELECT COUNT(*) FROM pitches_2026"
            ).fetchone()[0]
        finally:
            conn_raw.close()
        logger.info(f"Live refresh: +{len(to_add):,} pitches → {row_count:,} total")
        gc.collect()

        # ── 5. Score new pitches, update scored table, regenerate cards ────
        # Free CSV/gf DataFrames before loading XGBoost models — saves ~60 MB.
        del csv_df, gf_df, to_add_csv
        gc.collect()

        to_add = _apply_xwoba(to_add)
        predictor = _get_predictor()
        new_scored = predictor.predict(to_add)
        del to_add
        gc.collect()

        # Slim down to only the columns needed for player cards (~30 vs 215).
        slim_cols = [c for c in _SCORED_COLS if c in new_scored.columns]
        new_scored = new_scored[slim_cols].copy()

        # Update scored table via SQL DELETE + INSERT — never load the full table.
        conn_sc = sqlite3.connect(DB_PATH)
        try:
            has_tbl = conn_sc.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name='pitches_2026_scored'"
            ).fetchone() is not None
            # Align columns to existing table schema — only write columns that exist in both
            if has_tbl:
                existing_cols = [r[1] for r in conn_sc.execute("PRAGMA table_info(pitches_2026_scored)").fetchall()]
                new_scored = new_scored[[c for c in existing_cols if c in new_scored.columns]]
            if has_tbl and _live_replace_pks:
                ph = ",".join("?" * len(_live_replace_pks))
                conn_sc.execute(
                    f"DELETE FROM pitches_2026_scored "
                    f"WHERE CAST(game_pk AS INTEGER) IN ({ph})",
                    [int(p) for p in _live_replace_pks],
                )
                conn_sc.commit()
            new_scored.to_sql(
                "pitches_2026_scored", conn_sc,
                if_exists="append" if has_tbl else "replace",
                index=False,
            )
            conn_sc.commit()
        finally:
            conn_sc.close()
        del new_scored
        gc.collect()

        # Load slim scored table for card generation (only needed columns)
        conn_sc2 = sqlite3.connect(DB_PATH)
        try:
            _tbl_cols = [r[1] for r in conn_sc2.execute(
                "PRAGMA table_info(pitches_2026_scored)"
            ).fetchall()]
            _sel = [c for c in _SCORED_COLS if c in _tbl_cols]
            df_for_cards = pd.read_sql(
                f"SELECT {', '.join(_sel)} FROM pitches_2026_scored",
                conn_sc2,
            )
        finally:
            conn_sc2.close()
        generate_all_cards(df_for_cards, season='2026', skip_png=True)
        del df_for_cards
        gc.collect()

        _cached_leaderboard.cache_clear()
        _live_last_updated = datetime.now()
        logger.info("2026 profiles refreshed.")

    except Exception as exc:
        logger.warning(f"Live refresh failed: {exc}")
    finally:
        _live_refresh_lock.release()


def _spring_refresh_loop():
    """Background thread: refresh spring data every 90 seconds.
    Stops automatically when the regular season starts (2026-03-26).
    """
    while True:
        if datetime.today().strftime("%Y-%m-%d") >= "2026-03-26":
            logger.info("Spring refresh loop: regular season started, shutting down.")
            break
        _refresh_spring()
        time.sleep(90)


# Start the background refresh thread (daemon so it exits with the process)
_spring_thread = threading.Thread(target=_spring_refresh_loop, daemon=True)
_spring_thread.start()


def _backfill_spring_zone_data() -> None:
    """One-time startup backfill: replace /gf-only rows that lack events data.

    Games captured via /gf during live play are missing events/bat_score/
    post_bat_score/description/estimated_woba columns.  After the game ends,
    Baseball Savant's CSV endpoint has the full data.  This function finds
    those games and replaces their rows with complete CSV data so that IP,
    ERA, Zone%, SwStr%, and xwOBA are populated for older spring games.

    Runs once per container (guarded by a sentinel file).  Waits 90 s at
    startup so the first normal spring refresh can complete first.
    """
    _sentinel = "/tmp/.spring_zone_backfill_done"
    if os.path.exists(_sentinel):
        return
    # Let the first normal spring refresh run first
    time.sleep(90)
    if os.path.exists(_sentinel):
        return
    try:
        open(_sentinel, "w").close()
    except Exception:
        return

    try:
        import io, requests as req
        from scraper.statcast_scraper import load_from_db, save_to_db
        from model.predict import StuffPlusPredictor
        from profiles.player_cards import generate_all_cards

        logger.info("Spring events backfill: checking for /gf-only games missing events data...")

        # ── Find game_pks missing events data ──────────────────────────────
        try:
            conn = sqlite3.connect(DB_PATH)
            # Check whether zone column exists at all
            col_names = [r[1] for r in conn.execute(
                "PRAGMA table_info(pitches_spring2026)"
            ).fetchall()]
            if "zone" not in col_names:
                # No zone column at all — every game_pk needs backfill
                backfill_pks = {
                    r[0] for r in conn.execute(
                        "SELECT DISTINCT game_pk FROM pitches_spring2026"
                    ).fetchall() if r[0] is not None
                }
            else:
                # Games where ALL pitches have events IS NULL (no CSV data loaded yet)
                rows = conn.execute(
                    """SELECT game_pk,
                              COUNT(*) AS total,
                              SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END) AS has_events
                       FROM pitches_spring2026
                       GROUP BY game_pk"""
                ).fetchall()
                backfill_pks = {int(r[0]) for r in rows if r[0] is not None and int(r[2]) == 0}
            conn.close()
        except Exception as exc:
            logger.warning(f"Spring events backfill: DB query failed: {exc}")
            return

        if not backfill_pks:
            logger.info("Spring events backfill: no games missing events data — nothing to do.")
            return

        logger.info(f"Spring events backfill: {len(backfill_pks)} games need CSV replacement.")

        # ── Fetch full spring CSV (Feb 15 → tomorrow) ──────────────────────
        today = datetime.today()
        fetch_to_lt = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        csv_url = (
            "https://baseballsavant.mlb.com/statcast_search/csv"
            "?all=true&hfGT=S%7C&hfSea=2026%7C"
            "&game_date_gt=2026-02-15"
            f"&game_date_lt={fetch_to_lt}"
            "&player_type=pitcher&type=details"
            "&min_pitches=0&min_results=0&sort_col=pitches&sort_order=desc"
        )
        try:
            resp = req.get(csv_url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            csv_df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if "pitch_type" in csv_df.columns:
                csv_df = csv_df[csv_df["pitch_type"].notna() & (csv_df["pitch_type"] != "pitch_type")]
            if not csv_df.empty:
                csv_df["game_type"] = "R"
        except Exception as exc:
            logger.warning(f"Spring zone backfill: CSV fetch failed: {exc}")
            return

        if csv_df.empty or "game_pk" in csv_df.columns is False:
            logger.info("Spring zone backfill: empty CSV response.")
            return

        # Filter to only the games that need backfilling AND are in the CSV
        csv_pks_available = set(csv_df["game_pk"].dropna().astype(int))
        to_replace_pks = backfill_pks & csv_pks_available

        if not to_replace_pks:
            if csv_df.empty:
                # CSV returned nothing — retry on next deploy
                try:
                    os.remove(_sentinel)
                except Exception:
                    pass
                logger.info("Spring zone backfill: CSV empty, will retry next deploy.")
            else:
                # CSV has data but none of the /gf-only games appear in Statcast —
                # these are likely exhibition/non-Statcast games; zone data unavailable.
                logger.info(
                    f"Spring zone backfill: {len(backfill_pks)} games not in Statcast "
                    f"(exhibition games — zone/SwStr will be blank). Done."
                )
            return

        replacement_df = csv_df[csv_df["game_pk"].isin(to_replace_pks)].copy()
        logger.info(
            f"Spring zone backfill: replacing {len(to_replace_pks)} games "
            f"({len(replacement_df):,} pitches) with CSV data."
        )

        # Normalise player_name
        if "player_name" in replacement_df.columns:
            replacement_df["player_name"] = replacement_df["player_name"].apply(_gf_name)

        # Fill xwOBA for any batted balls missing it
        replacement_df = _apply_xwoba(replacement_df)

        # ── Acquire lock, replace rows in both tables ──────────────────────
        if not _spring_refresh_lock.acquire(timeout=120):
            logger.warning("Spring events backfill: could not acquire lock — skipping.")
            return
        try:
            ph = ",".join("?" * len(to_replace_pks))
            pk_list = [int(p) for p in to_replace_pks]

            # ── Raw table: DELETE + INSERT (avoids loading 70K-row table) ──
            if "player_name" in replacement_df.columns:
                replacement_df["player_name"] = replacement_df["player_name"].apply(_gf_name)
            if "game_type" in replacement_df.columns:
                replacement_df["game_type"] = "R"
            conn_raw = sqlite3.connect(DB_PATH)
            try:
                conn_raw.execute(
                    f"DELETE FROM pitches_spring2026 "
                    f"WHERE CAST(game_pk AS INTEGER) IN ({ph})", pk_list)
                conn_raw.commit()
                replacement_df.to_sql("pitches_spring2026", conn_raw, if_exists="append", index=False)
                conn_raw.commit()
            finally:
                conn_raw.close()

            # ── Score replacement rows, DELETE + INSERT into scored table ──
            replacement_df = _apply_xwoba(replacement_df)
            predictor = _get_predictor()
            new_scored = predictor.predict(replacement_df)
            del replacement_df
            gc.collect()
            slim_bf = [c for c in _SCORED_COLS if c in new_scored.columns]
            new_scored = new_scored[slim_bf].copy()
            conn_sc = sqlite3.connect(DB_PATH)
            try:
                has_sc_tbl = conn_sc.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='pitches_spring2026_scored'"
                ).fetchone() is not None
                if has_sc_tbl:
                    conn_sc.execute(
                        f"DELETE FROM pitches_spring2026_scored "
                        f"WHERE CAST(game_pk AS INTEGER) IN ({ph})", pk_list)
                    conn_sc.commit()
                new_scored.to_sql(
                    "pitches_spring2026_scored", conn_sc,
                    if_exists="append" if has_sc_tbl else "replace", index=False)
                conn_sc.commit()
                # Read slim scored table for card generation
                _tbl_cols = [r[1] for r in conn_sc.execute(
                    "PRAGMA table_info(pitches_spring2026_scored)"
                ).fetchall()]
                _sel = [c for c in _SCORED_COLS if c in _tbl_cols]
                df_for_cards = pd.read_sql(
                    f"SELECT {', '.join(_sel)} FROM pitches_spring2026_scored", conn_sc)
            finally:
                conn_sc.close()
            del new_scored
            gc.collect()
            generate_all_cards(df_for_cards, season="spring2026", skip_png=True)
            del df_for_cards
            gc.collect()
            _cached_leaderboard.cache_clear()
            logger.info(
                f"Spring events backfill: complete — {len(to_replace_pks)} games updated."
            )
        finally:
            _spring_refresh_lock.release()

    except Exception as exc:
        logger.warning(f"Spring events backfill failed: {exc}")


threading.Thread(target=_backfill_spring_zone_data, daemon=True).start()


def _fill_zone_from_coords() -> None:
    """Fill zone=NULL rows using plate coordinates for all pitch tables.

    Early spring training games often lack zone data in Statcast CSVs.
    plate_x / plate_z / sz_top / sz_bot are available, so we compute an
    approximate zone (1-9 in-zone, 14 out-of-zone) and write it back.
    Runs once per table per container, guarded by sentinel files.
    """
    tables = ["pitches_spring2026", "pitches_2025", "pitches_2024"]
    for tbl in tables:
        sentinel = f"/tmp/.zone_fill_done_{tbl}"
        if os.path.exists(sentinel):
            continue
        try:
            conn = sqlite3.connect(DB_PATH)
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({tbl})").fetchall()}
            if not {"zone", "plate_x", "plate_z", "sz_top", "sz_bot"}.issubset(cols):
                conn.close()
                continue
            rows = conn.execute(
                f"SELECT rowid, plate_x, plate_z, sz_top, sz_bot "
                f"FROM {tbl} WHERE zone IS NULL AND plate_x IS NOT NULL"
            ).fetchall()
            if not rows:
                conn.close()
                open(sentinel, "w").close()
                continue
            updates = []
            for rowid, px, pz, st, sb in rows:
                z = _compute_zone(px, pz, st, sb)
                if z is not None:
                    updates.append((z, rowid))
            if updates:
                conn.executemany(f"UPDATE {tbl} SET zone=? WHERE rowid=?", updates)
                conn.commit()
                logger.info(f"Zone fill: updated {len(updates):,} rows in {tbl}.")
            conn.close()
            open(sentinel, "w").close()
        except Exception as exc:
            logger.warning(f"Zone fill failed for {tbl}: {exc}")


threading.Thread(target=_fill_zone_from_coords, daemon=True).start()


def _repair_scored_table() -> None:
    """Detect and fix truncated scored table by rescoring missing game_pks from raw.

    If the scored table is missing game_pks that exist in the raw table (e.g.,
    due to a failed backfill mid-save), score the missing rows and append them.
    Runs once per container after the backfill settles.
    """
    _sentinel = "/tmp/.spring_scored_repair_done"
    if os.path.exists(_sentinel):
        return
    # Wait for backfill to finish first
    time.sleep(200)
    if os.path.exists(_sentinel):
        return
    try:
        open(_sentinel, "w").close()
    except Exception:
        return

    try:
        from scraper.statcast_scraper import load_from_db, save_to_db
        from model.predict import StuffPlusPredictor
        from profiles.player_cards import generate_all_cards

        conn_q = sqlite3.connect(DB_PATH)
        raw_pks = {r[0] for r in conn_q.execute(
            "SELECT DISTINCT CAST(game_pk AS INTEGER) FROM pitches_spring2026"
        ).fetchall() if r[0] is not None}
        scored_pks = {r[0] for r in conn_q.execute(
            "SELECT DISTINCT CAST(game_pk AS INTEGER) FROM pitches_spring2026_scored"
        ).fetchall() if r[0] is not None}
        conn_q.close()

        missing_pks = raw_pks - scored_pks
        if not missing_pks:
            logger.info("Scored table repair: no missing game_pks.")
            return

        logger.info(f"Scored table repair: {len(missing_pks)} game_pks in raw but not scored.")

        # Score in batches of 5 000 rows to stay within Railway memory limits
        CHUNK = 5_000
        predictor = _get_predictor()
        conn_raw = sqlite3.connect(DB_PATH)
        missing_list = sorted(missing_pks)
        total_added = 0

        for i in range(0, len(missing_list), 20):   # 20 game_pks per batch
            batch_pks = missing_list[i:i + 20]
            placeholders = ",".join("?" * len(batch_pks))
            chunk_df = pd.read_sql_query(
                f"SELECT * FROM pitches_spring2026 WHERE game_pk IN ({placeholders})",
                conn_raw, params=batch_pks
            )
            if chunk_df.empty:
                continue
            chunk_df = _apply_xwoba(chunk_df)
            chunk_scored = predictor.predict(chunk_df)
            del chunk_df

            if not _spring_refresh_lock.acquire(timeout=60):
                logger.warning("Scored table repair: lock timeout, skipping batch.")
                continue
            try:
                # Append-only: avoids loading the full existing scored table (O(n²) OOM fix)
                slim_repair = [c for c in _SCORED_COLS if c in chunk_scored.columns]
                chunk_slim  = chunk_scored[slim_repair].copy()
                del chunk_scored
                gc.collect()
                conn_rep = sqlite3.connect(DB_PATH)
                tbl_rep = conn_rep.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='pitches_spring2026_scored'"
                ).fetchone()
                chunk_slim.to_sql(
                    "pitches_spring2026_scored", conn_rep,
                    if_exists="append" if tbl_rep else "replace",
                    index=False
                )
                conn_rep.commit()
                conn_rep.close()
                del chunk_slim
                gc.collect()
                total_added += len(batch_pks)
                logger.info(f"Scored table repair: batch {i//20+1} done ({total_added}/{len(missing_pks)} pks).")
            finally:
                _spring_refresh_lock.release()
            time.sleep(5)   # brief pause between batches

        conn_raw.close()

        # Final: apply xwOBA + regenerate all cards from full scored table
        if not _spring_refresh_lock.acquire(timeout=180):
            logger.warning("Scored table repair: could not acquire lock for final step.")
            return
        try:
            df_scored = load_from_db("pitches_spring2026_scored")
            df_scored = _apply_xwoba(df_scored)
            save_to_db(df_scored, "pitches_spring2026_scored", replace=True)
            generate_all_cards(df_scored, season="spring2026", skip_png=True)
            _cached_leaderboard.cache_clear()
            logger.info(f"Scored table repair: complete — {total_added} game_pks added.")
        finally:
            _spring_refresh_lock.release()

    except Exception as exc:
        logger.warning(f"Scored table repair failed: {exc}")


threading.Thread(target=_repair_scored_table, daemon=True).start()


def _live_refresh_loop():
    """Background thread: refresh regular-season 2026 data every 60 seconds."""
    while True:
        _refresh_live()
        time.sleep(60)


_live_thread = threading.Thread(target=_live_refresh_loop, daemon=True)
_live_thread.start()


# ---------------------------------------------------------------------------
# Spring Breakout live refresh (sportId=21, every 90 s during event window)
# ---------------------------------------------------------------------------

_breakout_refresh_lock = threading.Lock()
_breakout_last_updated: datetime | None = None


def _get_breakout_game_pks(start_date: str = "2026-03-17") -> dict:
    """Return {game_pk: (detailedState, game_date)} for Spring Breakout games."""
    import requests as req
    from zoneinfo import ZoneInfo
    try:
        today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        url = (f"https://statsapi.mlb.com/api/v1/schedule"
               f"?sportId=21&startDate={start_date}&endDate={today_str}"
               f"&hydrate=team")
        data = req.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}).json()
        result = {}
        for d in data.get("dates", []):
            game_date = d.get("date", "")
            for g in d.get("games", []):
                state = g.get("status", {}).get("detailedState", "")
                if state not in ("Scheduled", "Pre-Game", "Warmup"):
                    result[g["gamePk"]] = (state, game_date)
        return result
    except Exception as exc:
        logger.warning(f"Could not fetch breakout schedule: {exc}")
        return {}


def _refresh_breakout() -> None:
    """Pull any new Spring Breakout pitches, score them, rebuild profiles."""
    global _breakout_last_updated
    if not _breakout_refresh_lock.acquire(blocking=False):
        return

    try:
        from scraper.statcast_scraper import save_to_db
        from profiles.player_cards import generate_all_cards
        from zoneinfo import ZoneInfo
        from datetime import timedelta

        today = datetime.now(ZoneInfo("America/New_York"))
        refetch_cutoff_str = (today - timedelta(days=2)).strftime("%Y-%m-%d")

        # Existing game PKs in DB
        try:
            conn = sqlite3.connect(DB_PATH)
            pk_rows = conn.execute(
                "SELECT DISTINCT CAST(game_pk AS INTEGER), game_date "
                "FROM pitches_breakout2026"
            ).fetchall()
            conn.close()
            existing_pks = {r[0] for r in pk_rows if r[0] is not None}
            recent_existing_pks = {r[0] for r in pk_rows if r[0] is not None and str(r[1]) >= refetch_cutoff_str}
        except Exception:
            existing_pks = set()
            recent_existing_pks = set()

        all_pks = _get_breakout_game_pks()
        pks_to_fetch = [
            pk for pk, (state, gdate) in all_pks.items()
            if pk not in existing_pks or gdate >= refetch_cutoff_str
        ]

        if not pks_to_fetch:
            _breakout_last_updated = datetime.now()
            return

        gf_frames = [_gf_to_df(pk) for pk in pks_to_fetch]
        gf_frames = [f for f in gf_frames if f is not None and not f.empty]
        if not gf_frames:
            _breakout_last_updated = datetime.now()
            return

        import pandas as pd
        new_df = pd.concat(gf_frames, ignore_index=True)

        # Remove rows for games we're re-fetching, then append
        pks_replace = recent_existing_pks | {
            pk for pk in pks_to_fetch if pk in existing_pks
        }
        conn = sqlite3.connect(DB_PATH)
        if pks_replace:
            ph = ",".join("?" * len(pks_replace))
            conn.execute(
                f"DELETE FROM pitches_breakout2026 WHERE CAST(game_pk AS INTEGER) IN ({ph})",
                list(pks_replace)
            )
            conn.commit()
        conn.close()

        new_df = _apply_xwoba(new_df)
        # Dedup-safe append (skips pitches already stored) — Spring Breakout is a
        # finished event, so re-fetches must not re-duplicate the static games.
        from scraper.statcast_scraper import append_to_db as _append_dedup
        _append_dedup(new_df, table="pitches_breakout2026")

        # Score all breakout pitches and regenerate profiles
        conn = sqlite3.connect(DB_PATH)
        existing_all = pd.read_sql_query("SELECT * FROM pitches_breakout2026", conn)
        conn.close()

        pred = _get_predictor()
        df_scored = pred.predict(existing_all)
        slim = [c for c in _SCORED_COLS if c in df_scored.columns]
        df_scored = df_scored[slim]

        conn = sqlite3.connect(DB_PATH)
        df_scored.to_sql("pitches_breakout2026_scored", conn,
                         if_exists="replace", index=False)
        conn.close()

        generate_all_cards(df_scored, season="breakout2026", skip_png=True)
        _cached_leaderboard.cache_clear()
        _breakout_last_updated = datetime.now()
        total = len(existing_all)
        logger.info(f"Breakout refresh: {total:,} total pitches, profiles rebuilt.")

    except Exception as exc:
        logger.warning(f"Breakout refresh error: {exc}", exc_info=True)
    finally:
        _breakout_refresh_lock.release()


def _breakout_refresh_loop():
    """Background thread: refresh Spring Breakout data every 90 seconds."""
    while True:
        _refresh_breakout()
        time.sleep(90)


_breakout_thread = threading.Thread(target=_breakout_refresh_loop, daemon=True)
_breakout_thread.start()


def _clear_preseason_2026_profiles() -> None:
    """Delete any 2026 profiles written before opening day (spring data bleed).
    Runs once at startup if today is still pre-season.
    """
    if datetime.today().strftime("%Y-%m-%d") >= "2026-03-26":
        return  # regular season already started, leave files alone
    import shutil
    dir_2026 = os.path.join(PROFILES_DIR, "2026")
    if os.path.isdir(dir_2026):
        shutil.rmtree(dir_2026, ignore_errors=True)
        logger.info("Pre-season cleanup: removed stale 2026 profiles (will rebuild on opening day).")


_clear_preseason_2026_profiles()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_dir(season: int) -> str:
    return os.path.join(PROFILES_DIR, str(season), "json")


def _leaderboard_path(season: int) -> str:
    return os.path.join(_json_dir(season), f"leaderboard_{season}.json")


def _profile_path(season: int, safe_name: str) -> str:
    return os.path.join(_json_dir(season), f"{safe_name}_{season}.json")


def _safe_name(name: str) -> str:
    return re.sub(r"[',]", "", name.replace(", ", "_").replace(" ", "_"))


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # File may be mid-write (temp rename not yet done) — treat as missing
        return None


@lru_cache(maxsize=64)
def _cached_leaderboard_mt(season: str, mtime: float):
    # mtime is part of the cache key: when an external updater rewrites the
    # leaderboard JSON (live seasons), the mtime changes and this re-reads from
    # disk — so live data refreshes without an app restart or manual cache_clear.
    return _load_json(_leaderboard_path(season))


def _cached_leaderboard(season: str):
    try:
        mt = os.path.getmtime(_leaderboard_path(season))
    except OSError:
        mt = 0.0
    return _cached_leaderboard_mt(season, mt)


# Preserve the `.cache_clear()` API used throughout app.py.
_cached_leaderboard.cache_clear = _cached_leaderboard_mt.cache_clear


def _leaderboard_live(season: str):
    """Load leaderboard, always bypassing cache when the cached value is None.

    lru_cache caches None when the file doesn't exist at request time (e.g.,
    server starts before the first refresh writes the file).  We never want to
    keep a None entry — always re-read from disk and re-populate the cache so
    the next request doesn't need to hit disk.
    """
    data = _cached_leaderboard(season)
    if data is None:
        # Read directly from disk — file may have been generated since the
        # cache entry was set.  Clear the stale None entry so future callers
        # get the cached version instead of hitting disk every time.
        data = _load_json(_leaderboard_path(season))
        if data is not None:
            _cached_leaderboard.cache_clear()
            _cached_leaderboard(season)   # re-populate cache
    return data


def _find_profile(season: str, player_name: str):
    """Try 'Last, First' then 'First Last' safe-name lookup for a player profile."""
    data = _load_json(_profile_path(season, _safe_name(player_name)))
    if data is None and "," not in player_name:
        # "First Last" → try "Last, First"
        parts = player_name.strip().rsplit(" ", 1)
        if len(parts) == 2:
            alt = f"{parts[1]}, {parts[0]}"
            data = _load_json(_profile_path(season, _safe_name(alt)))
    return data


# ---------------------------------------------------------------------------
# API routes  (season is a string: "2025", "2024", "spring2026", etc.)
# ---------------------------------------------------------------------------

@app.route("/api/leaderboard/<season>")
def api_leaderboard(season: str):
    data = _leaderboard_live(season)
    if data is None:
        return jsonify({"error": f"No leaderboard for {season}. Run 'python main.py profiles --season {season}'"}), 404
    return jsonify(data)


@app.route("/api/player/<path:player_name_season>")
def api_player(player_name_season: str):
    # Last path segment is season; everything before is the player name
    parts = player_name_season.rsplit("/", 1)
    if len(parts) != 2:
        abort(404)
    player_name, season = parts
    data = _find_profile(season, player_name)
    if data is None:
        abort(404)
    return jsonify(data)


# Hosts the headshot/logo images come from — allow-list so the proxy can't be
# used to fetch arbitrary URLs.
_IMG_PROXY_HOSTS = ("img.mlbstatic.com", "a.espncdn.com", "statsapi.mlb.com",
                    "www.mlbstatic.com", "securea.mlb.com")


@app.route("/imgproxy")
def imgproxy():
    """Re-serve an allow-listed external image from THIS origin.

    html2canvas taints the export canvas when it draws cross-origin images
    (MLB headshot, ESPN logo), which makes toDataURL() throw an "insecure"
    SecurityError. Swapping those <img> srcs to this same-origin proxy just
    before capture keeps the canvas clean so the PNG can render.
    """
    url = request.args.get("u", "")
    from urllib.parse import urlparse
    host = urlparse(url).netloc.lower()
    if not url.startswith("https://") or host not in _IMG_PROXY_HOSTS:
        abort(400)
    try:
        import requests as req
        r = req.get(url, timeout=10)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "image/png")
        return Response(r.content, mimetype=ctype,
                        headers={"Cache-Control": "public, max-age=86400"})
    except Exception:
        abort(502)


_CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _crop_trailing_bg(png_path: str, pad: int = 28) -> None:
    """Trim the uniform background Chrome leaves below the content (we render at
    a generous height and crop to fit). Keeps full width and the top intact."""
    try:
        import numpy as np
        from PIL import Image
        im = Image.open(png_path).convert("RGB")
        a = np.asarray(im).astype(int)
        trail = a[-3, -3]                                # bottom corner = the trailing fill to trim
        content = (np.abs(a - trail).sum(axis=2) > 24)   # pixels that differ from the trailing fill
        rows = np.where(content.any(axis=1))[0]
        if len(rows):
            bottom = min(a.shape[0], int(rows[-1]) + pad)
            im.crop((0, 0, a.shape[1], bottom)).save(png_path)
    except Exception as exc:
        logger.warning(f"crop_trailing_bg skipped: {exc}")


@app.route("/export_png")
def export_png():
    """Render the pitcher's season/game summary page in headless Chrome and return
    it as a downloadable PNG. Browser-agnostic (no client-side canvas taint)."""
    import subprocess, tempfile, os as _os
    from urllib.parse import quote
    name = request.args.get("name", "")
    season = request.args.get("season", "2026")
    game = request.args.get("game", "")
    if not name:
        abort(400)
    if not _os.path.exists(_CHROME_BIN):
        logger.error("export_png: Chrome not found at %s", _CHROME_BIN)
        abort(500)

    url = f"{request.host_url.rstrip('/')}/player/{quote(name)}/{season}?export=1"
    if game:
        url += f"&game={quote(game)}"

    tmp = tempfile.mktemp(suffix=".png")
    try:
        subprocess.run(
            [_CHROME_BIN, "--headless", "--disable-gpu", "--hide-scrollbars", "--no-sandbox",
             "--force-device-scale-factor=2", f"--screenshot={tmp}",
             "--window-size=1460,3200", "--virtual-time-budget=8000", url],
            capture_output=True, timeout=90,
        )
        if not _os.path.exists(tmp):
            logger.error("export_png: no screenshot produced for %s", url)
            abort(502)
        _crop_trailing_bg(tmp)
        dl = _safe_name(name) + f"_{season}" + (f"_g{game}" if game else "") + ".png"
        return send_file(tmp, mimetype="image/png", as_attachment=True, download_name=dl)
    except subprocess.TimeoutExpired:
        logger.error("export_png: Chrome timed out for %s", url)
        abort(504)
    except Exception as exc:
        logger.warning(f"export_png failed: {exc}")
        abort(500)


@app.route("/api/pitcher_names/<season>")
def api_pitcher_names(season: str):
    lb = _leaderboard_live(season)
    if lb is None:
        return jsonify([])
    return jsonify([r["player_name"] for r in lb])


def _normalize(s: str) -> str:
    """Strip accents and lowercase for accent-insensitive search."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii").lower()


def _name_matches(player_name: str, q: str) -> bool:
    """Match query against both 'Last, First' and 'First Last' formats."""
    stored = _normalize(player_name)
    parts = stored.split(", ", 1)
    first_last = f"{parts[1]} {parts[0]}" if len(parts) == 2 else stored
    return q in stored or q in first_last


@app.route("/api/search")
def api_search():
    q = _normalize(request.args.get("q", "").strip())
    season = request.args.get("season", "2026")
    lb = _leaderboard_live(season)
    if not lb or not q:
        return jsonify([])
    results = [r for r in lb if _name_matches(r["player_name"], q)][:20]
    return jsonify(results)


@app.route("/card/<path:player_name_season>")
def serve_card(player_name_season: str):
    parts = player_name_season.rsplit("/", 1)
    if len(parts) != 2:
        abort(404)
    player_name, season = parts
    # Try given name, then "First Last" → "Last, First" fallback
    def _card_path(name):
        return os.path.join(PROFILES_DIR, season, f"{_safe_name(name)}_{season}.png")
    path = _card_path(player_name)
    if not os.path.exists(path) and "," not in player_name:
        name_parts = player_name.strip().rsplit(" ", 1)
        if len(name_parts) == 2:
            path = _card_path(f"{name_parts[1]}, {name_parts[0]}")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="image/png")


# ---------------------------------------------------------------------------
# Game log helpers
# ---------------------------------------------------------------------------

def _table_for_season(season: str) -> "str | None":
    """Map a season string to its scored SQLite table name (live seasons only)."""
    return {
        "spring2026":   "pitches_spring2026_scored",
        "2026":         "pitches_2026_scored",
        "aaa2026":      "pitches_aaa2026_scored",
        "acl2026":      "pitches_acl2026_scored",
        "fsl2026":      "pitches_fsl2026_scored",
        "college2026":  "pitches_college2026_scored",
        "futures2026":  "pitches_futures2026_scored",
        "breakout2026": "pitches_breakout2026_scored",
        "springall2026": "pitches_springall2026_scored",
    }.get(season)


_boxscore_er_cache: dict[int, dict[int, int]] = {}  # game_pk → {pitcher_id → earned_runs}
_boxscore_ip_cache: dict[int, dict[int, int]] = {}  # game_pk → {pitcher_id → outs}
_boxscore_ts: dict[int, float] = {}                 # game_pk → last successful fetch (epoch s)
_final_games: set[int] = set()                      # game_pk known Final → cache forever
_BOXSCORE_TTL = 90.0  # re-fetch after this many seconds: an in-progress game's IP/ER
                      # keep climbing, so a permanent cache freezes the first read.


def _game_is_final(game_pk: int) -> bool:
    """Cheap schedule-endpoint check for whether a game has ended. A Final game's box
    score never changes again, so we can cache it permanently instead of TTL-polling —
    this keeps a starter's multi-game page from re-fetching every finished box each load."""
    try:
        import urllib.request
        url = f"https://statsapi.mlb.com/api/v1/schedule?gamePk={game_pk}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        return data["dates"][0]["games"][0]["status"]["abstractGameState"] == "Final"
    except Exception:
        return False


def _get_boxscore_earned_runs(game_pk: int) -> dict[int, int]:
    """Fetch pitcher earned runs (and innings→outs) from MLB Stats API box score.
    Cached per game with a short TTL. Also populates _boxscore_ip_cache: the boxscore
    inningsPitched is authoritative for IP — it captures baserunning outs (pickoffs /
    caught stealing) and the trailing out of a double play that the pitch-event stream
    omits, which event-counting alone misses. The TTL matters for LIVE games: without it
    the first mid-game read (e.g. 2 IP in the 2nd) is frozen for the life of the process
    even as the starter pitches into the 7th."""
    if game_pk in _boxscore_er_cache and (
        game_pk in _final_games
        or (time.time() - _boxscore_ts.get(game_pk, 0.0)) < _BOXSCORE_TTL
    ):
        return _boxscore_er_cache[game_pk]
    result: dict[int, int] = {}
    ip_result: dict[int, int] = {}
    fetched = False
    try:
        import urllib.request
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        for team in ("home", "away"):
            players = data["teams"][team]["players"]
            for pitcher_id in data["teams"][team].get("pitchers", []):
                p = players.get(f"ID{pitcher_id}", {})
                stats = p.get("stats", {}).get("pitching", {})
                er = stats.get("earnedRuns")
                if er is not None:
                    result[int(pitcher_id)] = int(er)
                # inningsPitched is a string like "1.0", "0.2" — convert to outs
                ip_str = stats.get("inningsPitched")
                if ip_str is not None:
                    try:
                        parts = str(ip_str).split(".")
                        ip_result[int(pitcher_id)] = (
                            int(parts[0]) * 3 + int(parts[1]) if len(parts) == 2
                            else int(parts[0]) * 3)
                    except Exception:
                        pass
        fetched = True
    except Exception:
        pass
    # Only cache a SUCCESSFUL fetch. Caching a failure (network hiccup / timeout)
    # would permanently poison this game for the life of the process: the boxscore
    # IP override would never fire and the game line would silently fall back to
    # the event-counted outs, which undercounts DP/baserunning outs.
    if fetched:
        _boxscore_er_cache[game_pk] = result
        _boxscore_ip_cache[game_pk] = ip_result
        _boxscore_ts[game_pk] = time.time()
        if _game_is_final(game_pk):
            _final_games.add(game_pk)   # stop re-polling — box score is now frozen for real
    # On a failed refresh, keep serving the last good cache rather than blanking it.
    return _boxscore_er_cache.get(game_pk, result)


def _compute_game_stats(gdf: pd.DataFrame) -> dict:
    """Compute box-score stats from one game's pitch-level DataFrame."""
    # Guard against duplicate rows from combined CSV + /gf pipeline.
    # Only dedup when pitch-level identity columns are all present.
    if all(c in gdf.columns for c in ["game_pk", "at_bat_number", "pitch_number"]):
        gdf = gdf.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])

    opp_team = pitcher_team = ""
    if all(c in gdf.columns for c in ["inning_topbot", "home_team", "away_team"]):
        mode_tb = gdf["inning_topbot"].dropna().mode()
        topbot  = mode_tb.iloc[0] if len(mode_tb) > 0 else "Bot"
        if topbot == "Top":   # away team bats → home team pitches
            pitcher_team = str(gdf["home_team"].dropna().mode().iloc[0]) if gdf["home_team"].notna().any() else ""
            opp_team     = str(gdf["away_team"].dropna().mode().iloc[0]) if gdf["away_team"].notna().any() else ""
        else:
            pitcher_team = str(gdf["away_team"].dropna().mode().iloc[0]) if gdf["away_team"].notna().any() else ""
            opp_team     = str(gdf["home_team"].dropna().mode().iloc[0]) if gdf["home_team"].notna().any() else ""

    total_pitches = len(gdf)

    # Strike% — description column format differs by scraper
    _strike_descs_lower = {"called_strike", "swinging_strike", "swinging_strike_blocked",
                            "foul", "foul_tip", "foul_bunt", "missed_bunt"}
    _strike_descs_cap   = {"Called Strike", "Swinging Strike", "Foul", "Foul Tip",
                            "Foul Bunt", "Missed Bunt"}
    if "description" in gdf.columns:
        desc = gdf["description"]
        strikes = int(desc.isin(_strike_descs_lower).sum() + desc.isin(_strike_descs_cap).sum())
    else:
        strikes = 0
    strike_pct = round(strikes / total_pitches * 100, 1) if total_pitches > 0 else 0.0

    # Outs / box-score stats — two formats:
    #   Statcast format : events is lowercase code on terminal pitch only ("strikeout", "field_out"…)
    #   Stockyard format: events is capitalized label repeated on every pitch in the PA ("Strikeout"…)
    _out_events = {"strikeout", "field_out", "force_out", "sac_fly", "sac_bunt",
                   "fielders_choice_out", "other_out", "caught_stealing_2b",
                   "caught_stealing_3b", "caught_stealing_home",
                   "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
                   "pickoff_caught_stealing_home"}
    _dp_events  = {"double_play", "grounded_into_double_play", "strikeout_double_play",
                   "sac_fly_double_play", "triple_play"}
    _out_events_cap = {"Groundout", "Flyout", "Lineout", "Pop Out", "Forceout", "Strikeout",
                       "Sac Fly", "Sac Bunt", "Double Play",
                       "Grounded Into DP", "Strikeout - DP", "Triple Play"}
    _dp_events_cap  = {"Double Play", "Grounded Into DP", "Strikeout - DP"}

    total_outs = 0
    hits = walks = ks = hrs = 0
    earned_runs = None
    if "events" in gdf.columns:
        ev_series = gdf["events"]
        ev_nonnull = ev_series.dropna()
        # Detect format: Stockyard uses Title-Case labels (e.g. "Strikeout")
        is_stockyard = (
            len(ev_nonnull) > 0
            and isinstance(ev_nonnull.iloc[0], str)
            and ev_nonnull.iloc[0][0].isupper()
        )
        if is_stockyard:
            # Event label is repeated on every pitch in the PA; collapse to one per PA.
            # Group by at_bat_number when available — more reliable than shift-based
            # grouping which fails when consecutive ABs share the same event label
            # (e.g. back-to-back strikeouts collapse into one, undercounting outs).
            if "at_bat_number" in gdf.columns and gdf["at_bat_number"].notna().any():
                ev = gdf.groupby("at_bat_number")["events"].last().dropna()
            else:
                pa_groups = (ev_series != ev_series.shift()).cumsum()
                ev = ev_series.groupby(pa_groups).last().dropna()
            total_outs = int(ev.isin(_out_events_cap).sum() + ev.isin(_dp_events_cap).sum())
            hits       = int(ev.isin({"Single", "Double", "Triple", "Home Run"}).sum())
            walks      = int(ev.isin({"Walk", "Intent Walk", "Hit By Pitch"}).sum())
            ks         = int((ev == "Strikeout").sum())
            hrs        = int((ev == "Home Run").sum())
        else:
            # Standard Statcast format: events only on terminal pitch, lowercase codes
            ev = ev_nonnull
            total_outs = int(ev.isin(_out_events).sum() + ev.isin(_dp_events).sum() * 2)
            hits       = int(ev.isin({"single", "double", "triple", "home_run"}).sum())
            walks      = int(ev.isin({"walk", "intent_walk"}).sum())
            ks         = int(ev.isin({"strikeout", "strikeout_double_play"}).sum())
            hrs        = int((ev == "home_run").sum())

    if all(c in gdf.columns for c in ["events", "bat_score", "post_bat_score"]):
        ab_ends = gdf[gdf["events"].notna() & gdf["bat_score"].notna()]
        if not ab_ends.empty:
            # Statcast format: compute from score differential
            run_diff    = (ab_ends["post_bat_score"] - ab_ends["bat_score"]).clip(lower=0)
            earned_runs = int(run_diff.sum())
    if "game_pk" in gdf.columns and "pitcher" in gdf.columns:
        game_pk_val = gdf["game_pk"].dropna().mode()
        pitcher_val = gdf["pitcher"].dropna().mode()
        if not game_pk_val.empty and not pitcher_val.empty:
            gp_i  = int(game_pk_val.iloc[0])
            pid_i = int(pitcher_val.iloc[0])
            er_map = _get_boxscore_earned_runs(gp_i)  # also fills _boxscore_ip_cache
            # Stockyard format (bat_score null/missing): earned runs from boxscore
            if earned_runs is None:
                er = er_map.get(pid_i)
                if er is not None:
                    earned_runs = er
            # Boxscore IP is authoritative for outs — overrides the event-counted
            # value so DP/baserunning outs (e.g. a pickoff completing a double
            # play) the pitch-event stream misses are credited on the game line.
            bs_outs = _boxscore_ip_cache.get(gp_i, {}).get(pid_i)
            if bs_outs is not None and bs_outs != total_outs:
                total_outs = bs_outs

    return {
        "opp_team":     opp_team,
        "pitcher_team": pitcher_team,
        "total_pitches": total_pitches,
        "ip_str":       f"{total_outs // 3}.{total_outs % 3}",
        "hits":         hits,
        "earned_runs":  earned_runs,
        "walks":        walks,
        "strikeouts":   ks,
        "home_runs":    hrs,
        "strike_pct":   strike_pct,
    }




@app.route("/api/games/<path:player_name_season>")
def api_player_games(player_name_season: str):
    """Chronological list of game appearances for a pitcher (live seasons only)."""
    parts = player_name_season.rsplit("/", 1)
    if len(parts) != 2:
        abort(404)
    player_name, season = parts
    table = _table_for_season(season)
    if not table:
        return jsonify([])

    # Accept both "Last, First" and "First Last" — normalise to "Last, First"
    if "," not in player_name:
        name_parts = player_name.strip().rsplit(" ", 1)
        if len(name_parts) == 2:
            player_name = f"{name_parts[1]}, {name_parts[0]}"

    # Gamefeed-sourced tables (ACL, FSL) don't have every column the MLB Statcast
    # schema does (e.g. bat_score/post_bat_score), so select only the columns that
    # actually exist — otherwise the whole query throws and the game list is empty.
    # _compute_game_stats already guards on column presence for anything missing.
    want = ["game_pk", "game_date", "home_team", "away_team", "inning_topbot",
            "description", "events", "bat_score", "post_bat_score",
            "at_bat_number", "pitch_number", "pitcher"]
    try:
        conn = sqlite3.connect(DB_PATH)
        have = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        cols = [c for c in want if c in have]
        if "game_pk" not in have or "game_date" not in have:
            conn.close()
            return jsonify([])
        df = pd.read_sql_query(
            f"""SELECT {", ".join(cols)}
                FROM {table} WHERE player_name = ?
                ORDER BY game_date, game_pk, at_bat_number, pitch_number""",
            conn, params=[player_name])
        conn.close()
    except Exception as exc:
        logger.error(f"api_player_games: {exc}")
        return jsonify([])

    games = []
    for (game_pk, game_date), gdf in df.groupby(["game_pk", "game_date"], sort=False):
        stats = _compute_game_stats(gdf)
        games.append({"game_pk": int(game_pk), "game_date": str(game_date), **stats})
    games.sort(key=lambda g: g["game_date"])
    return jsonify(games)


@app.route("/api/game_detail")
def api_game_detail():
    """Per-pitch-type summary + scatter for one specific game."""
    player_name = request.args.get("player", "")
    season      = request.args.get("season", "")
    game_pk_str = request.args.get("pk", "")
    if not player_name or not season or not game_pk_str:
        abort(400)
    try:
        game_pk = int(game_pk_str)
    except ValueError:
        abort(400)

    table = _table_for_season(season)
    if not table:
        abort(404)

    # Normalise to "Last, First" for DB query
    if "," not in player_name:
        name_parts = player_name.strip().rsplit(" ", 1)
        if len(name_parts) == 2:
            player_name = f"{name_parts[1]}, {name_parts[0]}"

    profile = _find_profile(season, player_name)  # for age/fallback info

    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(
            f"SELECT * FROM {table} WHERE player_name = ? AND game_pk = ?",
            conn, params=[player_name, game_pk])
        conn.close()
    except Exception as exc:
        logger.error(f"api_game_detail: {exc}")
        abort(500)

    if df.empty:
        abort(404)

    # Pull pitcher_id from the DB row (more reliable than JSON profile)
    pitcher_id = None
    if "pitcher" in df.columns:
        _pid = df["pitcher"].dropna().mode()
        if len(_pid) > 0:
            try: pitcher_id = int(_pid.iloc[0])
            except (ValueError, TypeError): pass

    from profiles.player_cards import summarize_pitcher, stuff_grade, _nan_to_none
    from config import PITCH_TYPES

    game_stats = _compute_game_stats(df)
    summary    = summarize_pitcher(df)

    p_throws = "R"
    if "p_throws" in df.columns:
        m = df["p_throws"].dropna().mode()
        if len(m) > 0:
            p_throws = str(m.iloc[0])

    arm_angle_avg = None
    if "arm_angle" in summary.columns and not summary.empty:
        _aa = summary["arm_angle"].mean()
        arm_angle_avg = round(float(_aa), 1) if _aa == _aa else None

    game_date = str(df["game_date"].dropna().iloc[0]) if "game_date" in df.columns and len(df) > 0 else ""

    pitches = []
    for _, row in summary.iterrows():
        sp    = float(row["stuff_plus"])
        grade, color = stuff_grade(round(sp, 1))
        def _sl(key):
            v = row.get(key); return v if isinstance(v, list) else []

        rh = row.get("release_height", float("nan"))
        rs = row.get("release_side", float("nan"))
        pitches.append({
            "pitch_type":       row["pitch_type"],
            "pitch_name":       PITCH_TYPES.get(row["pitch_type"], row["pitch_type"]),
            "stuff_plus":       round(sp, 1),
            "grade":            grade,
            "color":            color,
            "n":                int(row["n"]),
            "n_vs_r":           int(row.get("n_vs_r", 0) or 0),
            "n_vs_l":           int(row.get("n_vs_l", 0) or 0),
            "usage_pct":        round(float(row.get("usage_pct", 0) or 0), 1),
            "usage_pct_vs_r":   round(float(row.get("usage_pct_vs_r", 0) or 0), 1),
            "usage_pct_vs_l":   round(float(row.get("usage_pct_vs_l", 0) or 0), 1),
            "velo":             round(float(row["velo"]), 1),
            "ivb":              round(float(row["ivb"]), 1),
            "hb":               round(float(row["hb"]), 1),
            "vaa":              round(float(row["vaa"]), 2),
            "haa":              round(float(row["haa"]), 2),
            "spin":             int(row["spin"]),
            "extension":        round(float(row["extension"]), 2),
            "arm_angle":        round(float(row["arm_angle"]), 1) if _nan_to_none(row.get("arm_angle")) is not None else None,
            "release_height":   round(float(rh), 2) if rh == rh else None,
            "release_side":     round(float(rs), 2) if rs == rs else None,
            "spin_axis_clock":  str(row.get("spin_axis_clock", "—")),
            "zone_pct":         round(float(row["zone_pct"]), 1)      if _nan_to_none(row.get("zone_pct"))      is not None else None,
            "whiff_pct":        round(float(row["whiff_pct"]), 1)     if _nan_to_none(row.get("whiff_pct"))     is not None else None,
            "xwoba_contact":    round(float(row["xwoba_contact"]), 3) if _nan_to_none(row.get("xwoba_contact")) is not None else None,
            "locations_x_vs_r": _sl("locations_x_vs_r"),
            "locations_z_vs_r": _sl("locations_z_vs_r"),
            "locations_x_vs_l": _sl("locations_x_vs_l"),
            "locations_z_vs_l": _sl("locations_z_vs_l"),
        })

    # Raw scatter: every pitch location with type + batter hand
    scatter = []
    if all(c in df.columns for c in ["plate_x", "plate_z", "pitch_type", "stand"]):
        sdf = df[["plate_x", "plate_z", "pitch_type", "stand"]].dropna()
        scatter = [{"x": round(float(r.plate_x), 2), "z": round(float(r.plate_z), 2),
                    "pt": str(r.pitch_type), "st": str(r.stand)}
                   for r in sdf.itertuples()]

    # Per-pitch movement dots for the game-view movement chart
    movement_dots = []
    if all(c in df.columns for c in ["pfx_x_arm", "pfx_z_in", "pitch_type"]):
        mdf = df[["pfx_x_arm", "pfx_z_in", "pitch_type"]].dropna()
        movement_dots = [{"hb": round(float(r.pfx_x_arm), 2),
                          "ivb": round(float(r.pfx_z_in), 2),
                          "pt": str(r.pitch_type)}
                         for r in mdf.itertuples()]

    return jsonify({
        "player_name":   player_name,
        "season":        season,
        "game_pk":       game_pk,
        "game_date":     game_date,
        "total_pitches": int(sum(p["n"] for p in pitches)),
        "p_throws":      p_throws,
        "arm_angle":     arm_angle_avg,
        "team":          game_stats["pitcher_team"],
        "pitcher_id":    pitcher_id,
        "age":           profile.get("age") if profile else None,
        "pitches":        pitches,
        "scatter":        scatter,
        "movement_dots":  movement_dots,
        **{k: v for k, v in game_stats.items() if k != "pitcher_team"},
    })


# ---------------------------------------------------------------------------
# Live / spring status endpoints
# ---------------------------------------------------------------------------

@app.route("/api/spring/status")
def spring_status():
    max_date = _spring_max_date()
    try:
        conn = sqlite3.connect(DB_PATH)
        count = conn.execute("SELECT COUNT(*) FROM pitches_spring2026").fetchone()[0]
        conn.close()
    except Exception:
        count = 0
    return jsonify({
        "last_game_date": max_date,
        "total_pitches": count,
        "last_refresh": _spring_last_updated.isoformat() if _spring_last_updated else None,
    })


@app.route("/api/live/status")
def live_status():
    if not os.path.exists(DB_PATH):
        return jsonify({"last_updated": None, "total_pitches_2026": 0})
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT MAX(game_date), COUNT(*) FROM pitches_2026"
        ).fetchone()
        return jsonify({
            "last_updated": row[0],
            "total_pitches_2026": row[1] or 0,
        })
    except Exception:
        return jsonify({"last_updated": None, "total_pitches_2026": 0})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

# Admin token comes ONLY from the environment — no committed fallback. If ADMIN_TOKEN is
# unset the admin routes fail CLOSED (a default in a public repo is not a secret).
_ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")


def _admin_ok() -> bool:
    """True only if ADMIN_TOKEN is configured AND the request token matches it. Unset env
    var => always False, so the destructive admin routes are disabled rather than guarded by
    a known default."""
    return bool(_ADMIN_TOKEN) and request.args.get("token") == _ADMIN_TOKEN


@app.route("/api/admin/rescore-spring", methods=["POST"])
def admin_rescore_spring():
    """Drop pitches_spring2026_scored and clear sentinels so next refresh cycle
    does a full rescore with the current model.  Requires ?token=ADMIN_TOKEN."""
    if not _admin_ok():
        abort(403)
    try:
        conn_adm = sqlite3.connect(DB_PATH)
        conn_adm.execute("DROP TABLE IF EXISTS pitches_spring2026_scored")
        conn_adm.commit()
        conn_adm.close()
        if os.path.exists("/tmp/.spring_rescore_attempted"):
            os.unlink("/tmp/.spring_rescore_attempted")
        # Clear the scored version so the version check also triggers
        if os.path.exists(_SCORED_VERSION_FILE):
            os.unlink(_SCORED_VERSION_FILE)
        logger.info("admin_rescore_spring: scored table dropped; full rescore queued.")
        return jsonify({"status": "ok", "message": "Scored table dropped. Full rescore will run on next refresh cycle (~90s)."})
    except Exception as exc:
        logger.warning(f"admin_rescore_spring failed: {exc}")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/admin/rebuild-all", methods=["POST"])
def admin_rebuild_all():
    """Re-score all seasons and regenerate all profiles in the background.
    Requires ?token=ADMIN_TOKEN."""
    if not _admin_ok():
        abort(403)

    import threading

    def _run():
        try:
            from config import DB_PATH
            from features.engineering import engineer_features
            from model.predict import StuffPlusPredictor
            from profiles.player_cards import generate_all_cards

            predictor = StuffPlusPredictor()
            seasons = ["2024", "spring2026", "breakout2026"]
            conn = sqlite3.connect(DB_PATH)
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            conn.close()

            def _resolve_src(s, tables):
                """Return actual table name, checking _editor suffix for historical seasons."""
                plain = f"pitches_{s}"
                editor = f"pitches_{s}_editor"
                if plain in tables:
                    return plain
                if editor in tables:
                    return editor
                return None

            for s in seasons:
                src = _resolve_src(s, tables)
                dst = f"pitches_{s}_scored"
                if src is None:
                    logger.warning(f"[rebuild-all] no source table for {s} — skipping")
                    continue
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql(f"SELECT * FROM [{src}]", conn)
                conn.close()
                if df.empty:
                    continue
                logger.info(f"[rebuild-all] scoring {src} ({len(df):,} rows) …")
                df_eng, _ = engineer_features(df, baselines=predictor.baselines)
                df_sc = predictor.predict(df_eng, already_engineered=True)
                conn = sqlite3.connect(DB_PATH)
                df_sc.to_sql(dst, conn, if_exists="replace", index=False)
                conn.close()
                logger.info(f"[rebuild-all] written → {dst}")
                # Refresh tables list so newly-created scored tables are visible below
                conn = sqlite3.connect(DB_PATH)
                tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
                conn.close()

            for s in seasons:
                table = f"pitches_{s}_scored"
                if table not in tables:
                    continue
                conn = sqlite3.connect(DB_PATH)
                df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
                conn.close()
                if df.empty:
                    continue
                logger.info(f"[rebuild-all] generating profiles for {s} …")
                generate_all_cards(df, season=s)

            logger.info("[rebuild-all] done.")
        except Exception as exc:
            logger.error(f"[rebuild-all] failed: {exc}", exc_info=True)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "ok", "message": "Rebuild started in background. Check logs for progress (~5-10 min)."})


@app.route("/api/admin/retrain", methods=["POST"])
def admin_retrain():
    """Re-train all family ensembles from scratch using the training DB tables.
    Requires ?token=ADMIN_TOKEN. Runs in background (~15-20 min). Check logs for progress."""
    if not _admin_ok():
        abort(403)

    import threading

    def _run():
        try:
            from config import DB_PATH, TRAINING_SEASONS
            from features.engineering import engineer_features
            from model.train import train_all, save_baselines

            RAW_COLS = (
                "pitch_type, game_date, game_pk, player_name, pitcher, batter, "
                "p_throws, stand, balls, strikes, description, events, "
                "release_speed, release_spin_rate, release_extension, "
                "release_pos_x, release_pos_z, release_pos_y, "
                "pfx_x, pfx_z, plate_x, plate_z, sz_top, sz_bot, "
                "vx0, vy0, vz0, ax, ay, az, "
                "spin_axis, arm_angle, "
                "delta_run_exp, estimated_woba_using_speedangle, "
                "bat_score, home_score, away_score, "
                "post_bat_score, inning, outs_when_up, on_1b, on_2b, on_3b, "
                "game_year"
            )
            logger.info("[retrain] Loading training data …")
            conn = sqlite3.connect(DB_PATH)
            all_tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}

            season_parts = []
            for s in TRAINING_SEASONS:
                # Prefer plain table; fall back to _editor variant
                if f"pitches_{s}" in all_tables:
                    tbl = f"pitches_{s}"
                elif f"pitches_{s}_editor" in all_tables:
                    tbl = f"pitches_{s}_editor"
                else:
                    logger.warning(f"[retrain] No table found for season {s} — skipping")
                    continue
                season_parts.append(f"SELECT {RAW_COLS} FROM [{tbl}]")
                logger.info(f"[retrain] Using {tbl} for season {s}")

            if not season_parts:
                logger.error("[retrain] No training tables found — aborting")
                conn.close()
                return

            df = pd.read_sql(" UNION ALL ".join(season_parts), conn)
            conn.close()
            logger.info(f"[retrain] Loaded {len(df):,} rows from {len(season_parts)} seasons.")

            logger.info("[retrain] Engineering features …")
            df, baselines = engineer_features(df)
            save_baselines(baselines)

            logger.info("[retrain] Training per-family ensembles …")
            train_all(df)

            # Reset predictor singleton so next request reloads new models
            global _predictor_instance
            _predictor_instance = None
            logger.info("[retrain] Done. Predictor singleton reset — will reload on next request.")
        except Exception as exc:
            logger.error(f"[retrain] Failed: {exc}", exc_info=True)

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "ok", "message": "Retrain started in background. Check logs for progress (~15-20 min)."})


@app.route("/api/admin/rebuild-season", methods=["POST"])
def admin_rebuild_season():
    """Re-score a single season and regenerate its profiles.
    Requires ?token=ADMIN_TOKEN&season=2025 (or 2023/2024/spring2026)."""
    if not _admin_ok():
        abort(403)
    season = request.args.get("season", "").strip()
    if not season:
        return jsonify({"status": "error", "message": "season param required"}), 400

    import threading

    def _run(s):
        try:
            from config import DB_PATH
            from features.engineering import engineer_features
            from model.predict import StuffPlusPredictor
            from profiles.player_cards import generate_all_cards

            predictor = StuffPlusPredictor()
            conn = sqlite3.connect(DB_PATH)
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            conn.close()

            plain  = f"pitches_{s}"
            editor = f"pitches_{s}_editor"
            src = plain if plain in tables else (editor if editor in tables else None)
            if src is None:
                logger.error(f"[rebuild-season] no source table for {s}")
                return

            dst = f"pitches_{s}_scored"
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql(f"SELECT * FROM [{src}]", conn)
            conn.close()
            if df.empty:
                logger.warning(f"[rebuild-season] {src} is empty — skipping")
                return

            logger.info(f"[rebuild-season] scoring {src} ({len(df):,} rows) …")
            df_eng, _ = engineer_features(df, baselines=predictor.baselines)
            df_sc = predictor.predict(df_eng, already_engineered=True)
            conn = sqlite3.connect(DB_PATH)
            df_sc.to_sql(dst, conn, if_exists="replace", index=False)
            conn.close()
            logger.info(f"[rebuild-season] written → {dst}")

            logger.info(f"[rebuild-season] generating profiles for {s} …")
            generate_all_cards(df_sc, season=s)
            logger.info(f"[rebuild-season] done for {s}.")
        except Exception as exc:
            logger.error(f"[rebuild-season] failed for {s}: {exc}", exc_info=True)

    threading.Thread(target=_run, args=(season,), daemon=True).start()
    return jsonify({"status": "ok", "message": f"Rebuild of {season} started. Check logs for progress."})


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.after_request
def _no_cache_html(resp):
    """Never let the browser serve a stale HTML page or bundled JS — template
    edits (e.g. the PNG-export code) must take effect on a normal reload."""
    ctype = resp.headers.get("Content-Type", "")
    if ctype.startswith("text/html") or resp.mimetype == "application/javascript" \
            or (request.path or "").endswith(".js"):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.route("/player/<path:player_name_season>")
def player_page(player_name_season: str):
    parts = player_name_season.rsplit("/", 1)
    if len(parts) != 2:
        abort(404)
    player_name, season = parts
    return render_template("player.html", player_name=player_name, season=season)


@app.route("/editor/<path:player_name_season>")
def editor_page(player_name_season: str):
    parts = player_name_season.rsplit("/", 1)
    if len(parts) != 2:
        abort(404)
    player_name, season = parts
    game_pk = request.args.get("game_pk", type=int)
    return render_template("editor.html", player_name=player_name, season=season, game_pk=game_pk)


@app.route("/api/player_pitches_raw/<path:player_name_season>")
def api_player_pitches_raw(player_name_season: str):
    """Per-pitch movement data for scatter overlay: [{hb, ivb, pt, sp}].

    Returns pfx_x_arm (HB, arm-side positive) and pfx_z_in (IVB) in inches —
    the same coordinate space used by the movement-profile blobs in player.html.
    Sampled to ≤1500 rows to keep the response snappy.
    """
    parts = player_name_season.rsplit("/", 1)
    if len(parts) != 2:
        return jsonify([])
    player_name, season = parts
    db_name = _gf_name(player_name)
    game_pk_filter = request.args.get("game_pk", type=int)
    try:
        conn = sqlite3.connect(DB_PATH)
        # Prefer scored table; fall back to raw table if scored doesn't exist
        all_tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        scored_tbl = f"pitches_{season}_scored"
        raw_tbl    = f"pitches_{season}"
        editor_tbl = f"pitches_{season}_editor"
        if scored_tbl in all_tables:
            table = scored_tbl
            sp_expr = "stuff_plus"
        elif raw_tbl in all_tables:
            table = raw_tbl
            sp_expr = "NULL"
        elif editor_tbl in all_tables:
            table = editor_tbl
            sp_expr = "stuff_plus"  # editor CSV includes stuff_plus
        else:
            conn.close()
            return jsonify([])

        # Inspect available columns to handle different table versions gracefully
        col_names = {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if sp_expr == "stuff_plus" and "stuff_plus" not in col_names:
            sp_expr = "NULL"
        ivb_expr  = "pfx_z_in" if "pfx_z_in" in col_names else "pfx_z * 12.0"
        hb_expr   = ("pfx_x_arm" if "pfx_x_arm" in col_names
                     else "pfx_x * 12.0 * CASE p_throws WHEN 'R' THEN -1.0 ELSE 1.0 END")
        vaa_expr  = "vaa"              if "vaa"              in col_names else "NULL"
        haa_expr  = "haa"              if "haa"              in col_names else "NULL"
        zone_expr = "zone"             if "zone"             in col_names else "NULL"
        desc_expr = "description"      if "description"      in col_names else "NULL"
        velo_expr = "release_speed"    if "release_speed"    in col_names else "NULL"
        spin_expr = "release_spin_rate" if "release_spin_rate" in col_names else "NULL"
        ext_expr  = "release_extension" if "release_extension" in col_names else "NULL"
        # Use engineered movement columns for NOT NULL filter when available
        # (scored tables store pfx_x_arm/pfx_z_in, not raw pfx_x/pfx_z)
        if "pfx_x_arm" in col_names and "pfx_z_in" in col_names:
            notnull_clause = "pfx_x_arm IS NOT NULL AND pfx_z_in IS NOT NULL"
        elif "pfx_x_arm" in col_names:
            notnull_clause = "pfx_x_arm IS NOT NULL"
        else:
            notnull_clause = "pfx_z IS NOT NULL AND pfx_x IS NOT NULL"
        gk_clause = " AND CAST(game_pk AS INTEGER) = ?" if game_pk_filter is not None else ""
        gk_params = (db_name, game_pk_filter) if game_pk_filter is not None else (db_name,)
        df = pd.read_sql_query(
            f"SELECT "
            f"  ({ivb_expr}) AS ivb, "
            f"  ({hb_expr}) AS hb, "
            f"  pitch_type AS pt, ({sp_expr}) AS sp, "
            f"  {velo_expr} AS velo, {spin_expr} AS spin, "
            f"  {vaa_expr} AS vaa, {haa_expr} AS haa, "
            f"  {ext_expr} AS ext, "
            f"  {zone_expr} AS zone, {desc_expr} AS description, "
            f"  CAST(game_pk AS INTEGER) AS gk, "
            f"  CAST(at_bat_number AS INTEGER) AS ab, "
            f"  CAST(pitch_number AS INTEGER) AS pn "
            f"FROM {table} "
            f"WHERE player_name = ? AND {notnull_clause}{gk_clause}",
            conn, params=gk_params
        )
        conn.close()
    except Exception as exc:
        logger.warning(f"player_pitches_raw error: {exc}")
        return jsonify([])
    # Apply saved pitch-type corrections so the editor shows the user's saved work
    # after a refresh. pitch_type is not a model feature, so the per-pitch grade is
    # unchanged — only the label (and therefore the aggregation) moves.
    try:
        _ov = pitch_overrides_store.get_overrides(season, player_name=db_name)
        if len(_ov) and {"gk", "ab", "pn"}.issubset(df.columns):
            _key = (_ov["game_pk"].astype(str) + "_" + _ov["at_bat_number"].astype(str)
                    + "_" + _ov["pitch_number"].astype(str))
            _map = dict(zip(_key, _ov["new_type"]))
            _dk = (pd.to_numeric(df["gk"], errors="coerce").astype("Int64").astype(str) + "_"
                   + pd.to_numeric(df["ab"], errors="coerce").astype("Int64").astype(str) + "_"
                   + pd.to_numeric(df["pn"], errors="coerce").astype("Int64").astype(str))
            _hit = _dk.map(_map)
            if _hit.notna().any():
                df.loc[_hit.notna(), "pt"] = _hit[_hit.notna()].values
                logger.info(f"applied {int(_hit.notna().sum())} saved override(s) for {db_name} ({season})")
    except Exception as _exc:
        logger.warning(f"could not apply overrides in player_pitches_raw: {_exc}")

    if game_pk_filter is None and len(df) > 2000:
        df = df.sample(2000, random_state=42)
    df = df.dropna(subset=["ivb", "hb", "pt"])
    # Clean NaN → None so Flask serialises as JSON null (NaN is not valid JSON)
    import math
    records = []
    for row in df.to_dict(orient="records"):
        records.append({k: (None if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in row.items()})
    return jsonify(records)


@app.route("/api/overrides/<season>", methods=["GET"])
def api_overrides_list(season: str):
    """List saved pitch-type corrections, optionally filtered to a pitcher / game."""
    player = request.args.get("player")
    if player and "," not in player:
        parts = player.strip().rsplit(" ", 1)
        if len(parts) == 2:
            player = f"{parts[1]}, {parts[0]}"
    gp = request.args.get("game_pk", type=int)
    try:
        df = pitch_overrides_store.get_overrides(season, player_name=player, game_pk=gp)
        return jsonify(df.to_dict("records"))
    except Exception as exc:
        logger.error(f"api_overrides_list: {exc}")
        return jsonify([]), 500


@app.route("/api/overrides", methods=["POST"])
def api_overrides_save():
    """Persist pitch-type corrections so they survive refresh / restart / re-scrape.

    Body: {player_name, season, pitch_overrides: {"{game_pk}_{at_bat}_{pitch}": new_type}}

    The scraped label is looked up and stored as original_type, giving an audit trail
    and letting us detect later if the key no longer points at the pitch we corrected.
    """
    body = request.get_json(silent=True) or {}
    season = body.get("season", "")
    player_name = body.get("player_name", "")
    ovr = body.get("pitch_overrides", {}) or {}
    if not season or not ovr:
        return jsonify({"error": "missing season or pitch_overrides"}), 400

    db_name = _gf_name(player_name) if player_name else None

    # Look up each pitch's current (scraped) label for the audit trail.
    original: dict = {}
    raw_table = f"pitches_{season}"
    try:
        conn = sqlite3.connect(DB_PATH)
        tbls = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if raw_table in tbls and db_name:
            cur = pd.read_sql_query(
                f"""SELECT CAST(game_pk AS INTEGER) gk, CAST(at_bat_number AS INTEGER) ab,
                           CAST(pitch_number AS INTEGER) pn, pitch_type
                    FROM {raw_table} WHERE player_name = ?""", conn, params=(db_name,))
            original = {f"{r.gk}_{r.ab}_{r.pn}": r.pitch_type for r in cur.itertuples()}
        conn.close()
    except Exception as exc:
        logger.warning(f"api_overrides_save: original lookup failed: {exc}")

    rows = []
    for key, new_type in ovr.items():
        parts = str(key).split("_")
        if len(parts) != 3:
            continue
        try:
            gk, ab, pn = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        rows.append({"game_pk": gk, "at_bat_number": ab, "pitch_number": pn,
                     "new_type": new_type, "player_name": db_name,
                     "original_type": original.get(f"{gk}_{ab}_{pn}")})
    if not rows:
        return jsonify({"error": "no valid pitch keys"}), 400
    try:
        n = pitch_overrides_store.save_many(season, rows)
        _rebuild_pitcher_card(season, db_name)
        return jsonify({"saved": n, "season_total": pitch_overrides_store.count_overrides(season)})
    except Exception as exc:
        logger.error(f"api_overrides_save: {exc}")
        return jsonify({"error": str(exc)}), 500


def _rebuild_pitcher_card(season: str, db_name: "str | None") -> None:
    """Regenerate one pitcher's profile card with saved overrides applied.

    No re-scoring needed: pitch_type is not a model feature, so each pitch keeps its
    grade — the correction only changes which pitch type a pitch is aggregated into.
    Without this the card would keep showing the pre-correction arsenal until the next
    full re-score, which makes a saved correction look like it did nothing.
    """
    table = _table_for_season(season)
    if not table or not db_name:
        return
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(f"SELECT * FROM {table} WHERE player_name = ?", conn, params=(db_name,))
        conn.close()
        if df.empty:
            return
        df = pitch_overrides_store.apply_overrides(df, season)
        # generate_all_cards REWRITES the season leaderboard from whatever frame it is
        # given. Passing one pitcher would replace a 700-entry leaderboard with a
        # single row and break search, so snapshot it and merge this pitcher's entry
        # back in afterwards.
        lb_path = _leaderboard_path(season)
        lb_before = _load_json(lb_path) or []
        from profiles.player_cards import generate_all_cards
        # prune=False: this frame holds ONLY this pitcher, so the stale-file
        # cleanup must NOT run — otherwise it deletes every other pitcher's
        # profile JSON and their pages start 404ing after any editor save.
        generate_all_cards(df, season=season, skip_png=True, prune=False)
        lb_after = _load_json(lb_path) or []
        if lb_before:
            updated = {r.get("player_name"): r for r in lb_after if isinstance(r, dict)}
            merged = [updated.get(r.get("player_name"), r) for r in lb_before if isinstance(r, dict)]
            known = {r.get("player_name") for r in merged}
            merged += [r for r in lb_after if isinstance(r, dict) and r.get("player_name") not in known]
            with open(lb_path, "w") as fh:
                json.dump(merged, fh)
            logger.info(f"leaderboard preserved: {len(lb_before)} -> {len(merged)} entries")
        _cached_leaderboard.cache_clear()
        logger.info(f"rebuilt card for {db_name} ({season}) with overrides applied")
    except Exception as exc:
        logger.warning(f"could not rebuild card for {db_name}: {exc}")


@app.route("/api/overrides", methods=["DELETE"])
def api_overrides_delete():
    """Revert one pitch back to its scraped label."""
    body = request.get_json(silent=True) or {}
    try:
        n = pitch_overrides_store.delete_override(
            body["season"], int(body["game_pk"]),
            int(body["at_bat_number"]), int(body["pitch_number"]))
        if body.get("player_name"):
            _rebuild_pitcher_card(body["season"], _gf_name(body["player_name"]))
        return jsonify({"deleted": n})
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "need season, game_pk, at_bat_number, pitch_number"}), 400
    except Exception as exc:
        logger.error(f"api_overrides_delete: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/overrides/revert", methods=["POST"])
def api_overrides_revert():
    """Revert ALL saved corrections for one pitcher in a season (undo everything).

    Body: {player_name, season}. Deletes their override rows and rebuilds the card so
    the pitcher immediately reads back at their original scraped labels.
    """
    body = request.get_json(silent=True) or {}
    season = body.get("season", "")
    player_name = body.get("player_name", "")
    if not season or not player_name:
        return jsonify({"error": "need season and player_name"}), 400
    db_name = _gf_name(player_name)
    try:
        ov = pitch_overrides_store.get_overrides(season, player_name=db_name)
        n = 0
        for r in ov.itertuples():
            n += pitch_overrides_store.delete_override(
                season, int(r.game_pk), int(r.at_bat_number), int(r.pitch_number))
        _rebuild_pitcher_card(season, db_name)
        return jsonify({"reverted": n, "season_total": pitch_overrides_store.count_overrides(season)})
    except Exception as exc:
        logger.error(f"api_overrides_revert: {exc}")
        return jsonify({"error": str(exc)}), 500


@app.route("/api/reclassify", methods=["POST"])
def api_reclassify():
    """Recalculate Stuff+ with per-pitch or bulk type overrides.

    Body JSON — per-pitch mode (from the editor's drag-select):
      {
        "player_name": "Brown, Hunter",
        "season": "spring2026",
        "pitch_overrides": {"831847_1_1": "SL", "831847_1_2": "SL"}
          // key = "{game_pk}_{at_bat_number}_{pitch_number}"
      }

    Bulk mode (type-to-type) is still accepted for backward compat:
      { ..., "type_map": {"SI": "FF"} }

    Returns per-pitch-type Stuff+ before/after for comparison.
    """
    body = request.get_json(silent=True) or {}
    player_name     = body.get("player_name", "")
    season          = body.get("season", "")
    pitch_overrides = body.get("pitch_overrides", {})   # {gk_ab_pn: new_type}
    type_map        = body.get("type_map", {})           # {old_type: new_type} fallback
    game_pk         = body.get("game_pk")                # optional — filter to one game

    if not player_name or not season or (not pitch_overrides and not type_map):
        return jsonify({"error": "missing fields"}), 400

    db_name = _gf_name(player_name)
    raw_table = f"pitches_{season}"
    try:
        conn = sqlite3.connect(DB_PATH)
        all_tbls = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if raw_table in all_tbls:
            if game_pk is not None:
                df = pd.read_sql_query(
                    f"SELECT * FROM {raw_table} WHERE player_name = ? AND CAST(game_pk AS INTEGER) = ?",
                    conn, params=(db_name, int(game_pk))
                )
            else:
                df = pd.read_sql_query(
                    f"SELECT * FROM {raw_table} WHERE player_name = ?",
                    conn, params=(db_name,)
                )
        else:
            # Raw table absent (e.g. fresh Railway volume) — fetch from Baseball Savant
            import io, requests as _req
            conn.close()
            conn = None
            # Get pitcher MLBAM ID from editor table if available
            pitcher_id = None
            editor_tbl = f"pitches_{season}_editor"
            try:
                c2 = sqlite3.connect(DB_PATH)
                row = c2.execute(
                    f"SELECT pitcher FROM {editor_tbl} WHERE player_name=? AND pitcher IS NOT NULL LIMIT 1",
                    (db_name,)
                ).fetchone()
                c2.close()
                if row:
                    pitcher_id = int(row[0])
            except Exception:
                pass
            if not pitcher_id:
                return jsonify({"error": "no pitch data — pitcher ID not found"}), 404
            savant_url = (
                "https://baseballsavant.mlb.com/statcast_search/csv"
                f"?all=true&hfGT=R%7C&hfSea={season}%7C"
                f"&pitchers_lookup%5B%5D={pitcher_id}"
                "&player_type=pitcher&type=details"
                "&min_pitches=0&min_results=0&sort_col=pitches&sort_order=desc"
            )
            logger.info(f"Reclassify: fetching {db_name} {season} from Baseball Savant ...")
            resp = _req.get(savant_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if df.empty or "pitch_type" not in df.columns:
                return jsonify({"error": "no pitch data from Baseball Savant"}), 404
            df = df[df["pitch_type"].notna() & (df["pitch_type"] != "pitch_type")]
        if conn:
            conn.close()
    except Exception as exc:
        logger.warning(f"reclassify query error: {exc}")
        return jsonify({"error": "could not load pitches"}), 500

    if df.empty:
        return jsonify({"error": "no pitches found"}), 404

    # Capture original per-type Stuff+ for comparison
    orig_sp: dict[str, float] = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        if game_pk is not None:
            scored = pd.read_sql_query(
                f"SELECT pitch_type, stuff_plus FROM pitches_{season}_scored "
                f"WHERE player_name = ? AND CAST(game_pk AS INTEGER) = ?",
                conn, params=(db_name, int(game_pk))
            )
        else:
            scored = pd.read_sql_query(
                f"SELECT pitch_type, stuff_plus FROM pitches_{season}_scored WHERE player_name = ?",
                conn, params=(db_name,)
            )
        conn.close()
        for pt, grp in scored.groupby("pitch_type"):
            valid = grp["stuff_plus"].dropna()
            if not valid.empty:
                orig_sp[pt] = round(float(valid.mean()), 1)
    except Exception:
        pass

    df = df.copy()

    # Per-pitch overrides: match by (game_pk, at_bat_number, pitch_number)
    if pitch_overrides:
        for key_str, new_type in pitch_overrides.items():
            parts = key_str.split("_")
            if len(parts) != 3:
                continue
            try:
                gk, ab, pn = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            mask = (
                (pd.to_numeric(df["game_pk"],      errors="coerce") == gk) &
                (pd.to_numeric(df["at_bat_number"], errors="coerce") == ab) &
                (pd.to_numeric(df["pitch_number"],  errors="coerce") == pn)
            )
            df.loc[mask, "pitch_type"] = new_type

    # Bulk type-map fallback (applied after per-pitch so per-pitch wins)
    if type_map:
        df["pitch_type"] = df["pitch_type"].replace(type_map)

    # Re-score
    try:
        from model.predict import StuffPlusPredictor
        predictor = _get_predictor()
        df_scored = predictor.predict(df)
    except Exception as exc:
        logger.warning(f"reclassify predict error: {exc}")
        return jsonify({"error": f"model error: {exc}"}), 500

    # Build per-type result rows. Use summarize_pitcher so xwOBAcon / zone% / whiff% /
    # velo / Command+ recalculate too (not just Stuff+) after a relabel.
    from profiles.player_cards import summarize_pitcher as _summ, _nan_to_none as _n2n
    try:
        _s = _summ(df_scored)
        _smap = {r["pitch_type"]: r for _, r in _s.iterrows()}
    except Exception as _exc:
        logger.warning(f"reclassify summarize error: {_exc}")
        _smap = {}

    def _stat(sr, key, nd):
        v = sr.get(key) if sr is not None else None
        return round(float(v), nd) if _n2n(v) is not None else None

    results = []
    for pt, grp in df_scored.groupby("pitch_type"):
        valid = grp[grp["stuff_plus"].notna()]
        if valid.empty:
            continue
        sr = _smap.get(pt)
        remapped_from = [k for k, v in (type_map or {}).items() if v == pt]
        results.append({
            "pitch_type":     pt,
            "stuff_plus_new": round(float(valid["stuff_plus"].mean()), 1),
            "stuff_plus_old": orig_sp.get(pt),
            "n":              int(len(grp)),
            "remapped_from":  remapped_from,
            "xwoba_contact":  _stat(sr, "xwoba_contact", 3),
            "zone_pct":       _stat(sr, "zone_pct", 1),
            "whiff_pct":      _stat(sr, "whiff_pct", 1),
            "velo":           _stat(sr, "velo", 1),
        })

    results.sort(key=lambda x: -x["n"])
    return jsonify({"pitches": results})


@app.route("/api/reclassify_profile", methods=["POST"])
def api_reclassify_profile():
    """Same as /api/reclassify but returns a full profile JSON for renderPage().

    Applies pitch_overrides / type_map, rescores, then runs summarize_pitcher
    so the player profile page can render a temporary preview without any DB writes.
    """
    body = request.get_json(silent=True) or {}
    player_name     = body.get("player_name", "")
    season          = body.get("season", "")
    pitch_overrides = body.get("pitch_overrides", {})
    type_map        = body.get("type_map", {})
    game_pk         = body.get("game_pk")

    if not player_name or not season:
        return jsonify({"error": "missing fields"}), 400

    db_name   = _gf_name(player_name)
    raw_table = f"pitches_{season}"

    # Load raw pitches (same logic as api_reclassify)
    try:
        conn = sqlite3.connect(DB_PATH)
        all_tbls = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if raw_table in all_tbls:
            if game_pk is not None:
                df = pd.read_sql_query(
                    f"SELECT * FROM {raw_table} WHERE player_name = ? AND CAST(game_pk AS INTEGER) = ?",
                    conn, params=(db_name, int(game_pk))
                )
            else:
                df = pd.read_sql_query(
                    f"SELECT * FROM {raw_table} WHERE player_name = ?",
                    conn, params=(db_name,)
                )
        else:
            conn.close()
            return jsonify({"error": "raw pitch table not found"}), 404
        conn.close()
    except Exception as exc:
        return jsonify({"error": f"could not load pitches: {exc}"}), 500

    if df.empty:
        return jsonify({"error": "no pitches found"}), 404

    df = df.copy()

    # Apply per-pitch overrides
    if pitch_overrides:
        for key_str, new_type in pitch_overrides.items():
            parts = key_str.split("_")
            if len(parts) != 3:
                continue
            try:
                gk, ab, pn = int(parts[0]), int(parts[1]), int(parts[2])
            except ValueError:
                continue
            mask = (
                (pd.to_numeric(df["game_pk"],      errors="coerce") == gk) &
                (pd.to_numeric(df["at_bat_number"], errors="coerce") == ab) &
                (pd.to_numeric(df["pitch_number"],  errors="coerce") == pn)
            )
            df.loc[mask, "pitch_type"] = new_type

    if type_map:
        df["pitch_type"] = df["pitch_type"].replace(type_map)

    # Rescore
    try:
        df_scored = _get_predictor().predict(df)
    except Exception as exc:
        return jsonify({"error": f"model error: {exc}"}), 500

    # Build profile response (mirrors api_game_detail)
    from profiles.player_cards import summarize_pitcher, stuff_grade, _nan_to_none
    from config import PITCH_TYPES

    summary = summarize_pitcher(df_scored)

    p_throws = "R"
    if "p_throws" in df_scored.columns:
        m = df_scored["p_throws"].dropna().mode()
        if len(m) > 0:
            p_throws = str(m.iloc[0])

    pitcher_id = None
    if "pitcher" in df_scored.columns:
        _pid = df_scored["pitcher"].dropna().mode()
        if len(_pid) > 0:
            try: pitcher_id = int(_pid.iloc[0])
            except (ValueError, TypeError): pass

    arm_angle_avg = None
    if "arm_angle" in summary.columns and not summary.empty:
        _aa = summary["arm_angle"].mean()
        arm_angle_avg = round(float(_aa), 1) if _aa == _aa else None

    pitches = []
    for _, row in summary.iterrows():
        sp = float(row["stuff_plus"])
        grade, color = stuff_grade(round(sp, 1))
        def _sl(key):
            v = row.get(key); return v if isinstance(v, list) else []
        rh = row.get("release_height", float("nan"))
        rs = row.get("release_side", float("nan"))
        pitches.append({
            "pitch_type":       row["pitch_type"],
            "pitch_name":       PITCH_TYPES.get(row["pitch_type"], row["pitch_type"]),
            "stuff_plus":       round(sp, 1),
            "grade":            grade,
            "color":            color,
            "n":                int(row["n"]),
            "n_vs_r":           int(row.get("n_vs_r", 0) or 0),
            "n_vs_l":           int(row.get("n_vs_l", 0) or 0),
            "usage_pct":        round(float(row.get("usage_pct", 0) or 0), 1),
            "usage_pct_vs_r":   round(float(row.get("usage_pct_vs_r", 0) or 0), 1),
            "usage_pct_vs_l":   round(float(row.get("usage_pct_vs_l", 0) or 0), 1),
            "velo":             round(float(row["velo"]), 1),
            "ivb":              round(float(row["ivb"]), 1),
            "hb":               round(float(row["hb"]), 1),
            "vaa":              round(float(row["vaa"]), 2),
            "haa":              round(float(row["haa"]), 2),
            "spin":             int(row["spin"]),
            "extension":        round(float(row["extension"]), 2),
            "arm_angle":        round(float(row["arm_angle"]), 1) if _nan_to_none(row.get("arm_angle")) is not None else None,
            "release_height":   round(float(rh), 2) if rh == rh else None,
            "release_side":     round(float(rs), 2) if rs == rs else None,
            "spin_axis_clock":  str(row.get("spin_axis_clock", "—")),
            "zone_pct":         round(float(row["zone_pct"]), 1)      if _nan_to_none(row.get("zone_pct"))      is not None else None,
            "whiff_pct":        round(float(row["whiff_pct"]), 1)     if _nan_to_none(row.get("whiff_pct"))     is not None else None,
            "xwoba_contact":    round(float(row["xwoba_contact"]), 3) if _nan_to_none(row.get("xwoba_contact")) is not None else None,
            "locations_x_vs_r": _sl("locations_x_vs_r"),
            "locations_z_vs_r": _sl("locations_z_vs_r"),
            "locations_x_vs_l": _sl("locations_x_vs_l"),
            "locations_z_vs_l": _sl("locations_z_vs_l"),
        })

    scatter = []
    if all(c in df_scored.columns for c in ["plate_x", "plate_z", "pitch_type", "stand"]):
        sdf = df_scored[["plate_x", "plate_z", "pitch_type", "stand"]].dropna()
        scatter = [{"x": round(float(r.plate_x), 2), "z": round(float(r.plate_z), 2),
                    "pt": str(r.pitch_type), "st": str(r.stand)}
                   for r in sdf.itertuples()]

    # Per-pitch movement dots carrying the RECLASSIFIED pitch_type, so the movement
    # chart's "each pitch" overlay recolors to the new labels in the preview (the
    # season endpoint would otherwise re-fetch the original DB labels).
    movement_dots = []
    if all(c in df_scored.columns for c in ["pfx_x_arm", "pfx_z_in", "pitch_type"]):
        mdf = df_scored[["pfx_x_arm", "pfx_z_in", "pitch_type"]].dropna()
        movement_dots = [{"hb": round(float(r.pfx_x_arm), 2),
                          "ivb": round(float(r.pfx_z_in), 2),
                          "pt": str(r.pitch_type)}
                         for r in mdf.itertuples()]

    profile = _find_profile(season, player_name)

    # Game-scoped preview: carry the game line (IP / H / ER / BB / K / HR, opponent,
    # date) so the page can re-render as a GAME view. Without these the preview
    # collapses to a season summary and the game's stats disappear. Re-classifying a
    # pitch type doesn't change these — they come from descriptions/events/boxscore —
    # so they're computed from the same game-scoped frame.
    game_fields = {}
    if game_pk is not None:
        try:
            game_fields = dict(_compute_game_stats(df))
            _gd = df["game_date"].dropna() if "game_date" in df.columns else []
            game_fields["game_pk"]   = int(game_pk)
            game_fields["game_date"] = str(_gd.iloc[0]) if len(_gd) else None
        except Exception as exc:
            logger.warning(f"reclassify_profile game stats failed: {exc}")

    return jsonify({
        "player_name":   db_name,
        "season":        season,
        "pitcher_id":    pitcher_id,
        "p_throws":      p_throws,
        "arm_angle":     arm_angle_avg,
        "team":          profile.get("team") if profile else None,
        "age":           profile.get("age")  if profile else None,
        "total_pitches": int(sum(p["n"] for p in pitches)),
        "pitches":       pitches,
        "scatter":       scatter,
        "movement_dots": movement_dots,
        **game_fields,
        "_preview":      True,
    })


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    # use_reloader=False: the reloader would re-import this module in a second
    # process, double-starting every background refresh thread (and racing on the
    # SQLite rescore). Production runs via gunicorn, so this only affects local dev.
    app.run(debug=True, port=port, host="0.0.0.0", use_reloader=False)

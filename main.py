"""
Stuff+ CLI entry point.

Commands
--------
  train     — Load training data, engineer shape features, train the Driveline
              run-value model, and save the frozen normalization baseline.
              Optional --sample=<frac> trains on a random subset.
  score     — Score each season's raw pitch table and write the *_scored tables.
  profiles  — Aggregate scored pitches into per-pitcher cards and leaderboards.
  live      — Start the live 2026 update loop (delegates to live/live_update.py).
"""

import logging
import os
import sqlite3
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def cmd_train():
    from config import DB_PATH, TRAINING_SEASONS
    from features.engineering import engineer_features
    from model.train import train_all, save_baselines

    # Load raw columns only — pitches_train has 200+ pre-engineered columns
    # from the old architecture; we only need the raw inputs.
    RAW_COLS = """
        pitch_type, game_date, game_pk, player_name, pitcher, batter,
        p_throws, stand, balls, strikes, description, events,
        release_speed, release_spin_rate, release_extension,
        release_pos_x, release_pos_z, release_pos_y,
        pfx_x, pfx_z, plate_x, plate_z, sz_top, sz_bot,
        vx0, vy0, vz0, ax, ay, az,
        spin_axis, arm_angle,
        delta_run_exp, estimated_woba_using_speedangle,
        bb_type, launch_speed, launch_angle, hc_x, hc_y,
        bat_score, home_score, away_score,
        post_bat_score, inning, outs_when_up, on_1b, on_2b, on_3b,
        game_year
    """
    sample_frac = None
    for arg in sys.argv:
        if arg.startswith("--sample="):
            sample_frac = float(arg.split("=")[1])

    logger.info("Loading training data …")
    conn = sqlite3.connect(DB_PATH)
    seasons_sql = " UNION ALL ".join(
        f"SELECT {RAW_COLS} FROM pitches_{s}" for s in TRAINING_SEASONS
    )
    df = pd.read_sql(seasons_sql, conn)
    conn.close()
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42)
        logger.info(f"  Loaded {len(df):,} rows (sample={sample_frac}).")
    else:
        logger.info(f"  Loaded {len(df):,} rows.")

    logger.info("Training the model …")
    from model.train import train_unified
    train_unified(df)

    logger.info("Done.")



def cmd_score():
    import os, pickle
    from config import DB_PATH, MODEL_DIR
    from features.engineering import engineer_features
    from model.predict import StuffPlusPredictor

    predictor = StuffPlusPredictor()
    _XWOBA_KNN = os.path.join(os.path.dirname(MODEL_DIR), "xwoba_knn.pkl")

    def _fill_xwoba(dfx):
        """Fill estimated_woba from exit-velo/launch-angle via the cached k-NN for
        /gf-sourced rows that lack it (Statcast has it; /gf does not). Without this,
        re-scoring wipes xwOBAcon for any game pulled from the /gf feed."""
        if not os.path.exists(_XWOBA_KNN) or not {"launch_speed", "launch_angle"}.issubset(dfx.columns):
            return dfx
        if "estimated_woba_using_speedangle" not in dfx.columns:
            dfx["estimated_woba_using_speedangle"] = None
        ev = pd.to_numeric(dfx["launch_speed"], errors="coerce")
        la = pd.to_numeric(dfx["launch_angle"], errors="coerce")
        xw = pd.to_numeric(dfx["estimated_woba_using_speedangle"], errors="coerce")
        m = ev.notna() & la.notna() & xw.isna()
        if m.any():
            with open(_XWOBA_KNN, "rb") as fh:
                knn = pickle.load(fh)
            dfx.loc[m, "estimated_woba_using_speedangle"] = \
                knn.predict(dfx.loc[m, ["launch_speed", "launch_angle"]].astype(float).values).round(3)
            logger.info(f"  xwOBA: filled {int(m.sum())} /gf batted balls via k-NN")
        return dfx

    # 2023/2024/2025: use historical norms (2020-2024 baseline) to avoid contamination
    # 2026 seasons: use current norms (2020-2025 baseline)
    season_norm = {
        "2023": "historical", "2024": "historical",   # 2025 removed: it is training data now
        "spring2026": "current", "breakout2026": "current",
        "2026": "current",
        "aaa2026": "current",
        "acl2026": "current",
        "fsl2026": "current",
        "college2026": "historical",   # completed season — score like 2025 (historical norm)
        "futures2026": "current",       # live Futures Game
    }
    seasons = list(season_norm.keys())
    # Default: rescore 2025 and up (2023/2024 are frozen historical). Pass explicit
    # season names as args to override, e.g. `main.py score 2023 2024`.
    _explicit = [a for a in sys.argv[2:] if not a.startswith("--")]
    if _explicit:
        seasons = [s for s in seasons if s in _explicit]
    else:
        seasons = [s for s in seasons if s not in ("2023", "2024")]
    logger.info(f"Scoring seasons: {seasons}")
    for s in seasons:
        src_table   = f"pitches_{s}"
        dst_table   = f"pitches_{s}_scored"
        conn = sqlite3.connect(DB_PATH)
        try:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if src_table not in tables:
                logger.info(f"  No table {src_table} — skipping.")
                conn.close()
                continue
            df = pd.read_sql(f"SELECT * FROM [{src_table}]", conn)
        finally:
            conn.close()

        if df.empty:
            logger.info(f"  {src_table} is empty — skipping.")
            continue

        # Apply saved pitch-type corrections before scoring. The raw tables are
        # rewritten by re-scrapes, so overrides live in their own table and are
        # re-applied here every time — that's what makes a correction durable.
        from storage.overrides import apply_overrides, count_overrides
        _n_ovr = count_overrides(s)
        if _n_ovr:
            df = apply_overrides(df, s)
            logger.info(f"  Applied {_n_ovr} saved pitch-type override(s) for {s}")

        norm_set = season_norm[s]
        logger.info(f"Scoring {src_table} ({len(df):,} rows) [norm={norm_set}] …")
        df_eng, _ = engineer_features(df, baselines=predictor.baselines)
        df_scored = predictor.predict(df_eng, already_engineered=True, norm_set=norm_set)

        # No clip — every pitch keeps its raw grade (slow junk grades low on merit).
        df_scored = _fill_xwoba(df_scored)

        # SQLite is case-insensitive on column names — drop case-insensitive dupes
        seen, keep = {}, []
        for c in df_scored.columns:
            lc = c.lower()
            if lc in seen: continue
            seen[lc] = c; keep.append(c)
        if len(keep) != len(df_scored.columns):
            df_scored = df_scored[keep]

        conn = sqlite3.connect(DB_PATH, timeout=60)
        df_scored.to_sql(dst_table, conn, if_exists="replace", index=False, chunksize=10000)
        conn.close()
        logger.info(f"  Written → {dst_table}")

    logger.info("Done.")


def cmd_profiles():
    from config import DB_PATH
    from profiles.player_cards import generate_all_cards

    seasons = [
        ("2023",         "2023"),
        ("2024",         "2024"),
        ("spring2026",   "spring2026"),
        ("breakout2026", "breakout2026"),
        ("2026",         "2026"),
        ("aaa2026",      "aaa2026"),
        ("acl2026",      "acl2026"),
        ("fsl2026",      "fsl2026"),
        ("college2026",  "college2026"),
        ("futures2026",  "futures2026"),
        ("springall2026", "springall2026"),  # spring2026 + breakout2026 combined
    ]
    # Default: 2025 and up (matches scoring scope). Pass explicit season names to override.
    _explicit = [a for a in sys.argv[2:] if not a.startswith("--")]
    if _explicit:
        seasons = [t for t in seasons if t[0] in _explicit]
    else:
        seasons = [t for t in seasons if t[0] not in ("2023", "2024")]
    logger.info(f"Building profiles for: {[t[0] for t in seasons]}")

    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    conn.close()

    for table_suffix, season_int in seasons:
        table = f"pitches_{table_suffix}_scored"
        if table not in tables:
            logger.info(f"  No table {table} — skipping season {season_int}.")
            continue

        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
        conn.close()

        if df.empty:
            logger.info(f"  {table} is empty — skipping.")
            continue

        logger.info(f"Building profiles for season {season_int} ({len(df):,} pitches) …")
        generate_all_cards(df, season=season_int, skip_png="--skip-png" in sys.argv)

    logger.info("Done.")


def cmd_live():
    from live.live_update import run_live
    interval = 60   # refresh once a minute by default
    for arg in sys.argv:
        if arg.startswith("--interval="):
            interval = int(arg.split("=")[1])
    run_live(interval_seconds=interval)


if __name__ == "__main__":
    commands = {
        "train":    cmd_train,
        "score":    cmd_score,
        "profiles":     cmd_profiles,
        "live":         cmd_live,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        print(f"Usage: python main.py [{' | '.join(commands)}]")
        sys.exit(1)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    commands[sys.argv[1]]()

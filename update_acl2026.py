"""
ACL2026 unified updater: incremental fetch → score → profile gen.

Mirrors update_aaa2026.py but for the Arizona Complex League (MLB StatsAPI
sportId=16). Designed to be run periodically (every ~30 min via launchd) to
keep the acl2026 data fresh end-to-end.

Pipeline:
  1. Fetch new ACL games (last N days by default) via gamefeed JSON (sportId=16)
  2. Append non-dup pitches to pitches_acl2026
  3. Re-score the affected pitches via the model and UPDATE
     pitches_acl2026_scored with new stuff_plus values
  4. Aggregate scored table into profiles/output/acl2026/json/

Run:
  python update_acl2026.py             # incremental (last 2 days)
  python update_acl2026.py --full      # full season rebuild (slow)
"""
import os, sys, sqlite3, logging, argparse, json, glob
from datetime import datetime, timedelta
import pandas as pd, numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.aaa_scraper import download_aaa
from scraper.statcast_scraper import save_to_db, append_to_db
from features.engineering import engineer_features
from model.predict import StuffPlusPredictor

DB_PATH = "/Users/akane/Desktop/new_stuff/stuff_plus/data/statcast.db"
SPORT_ID = 16  # ACL / complex / rookie level
TABLE = "pitches_acl2026"
TABLE_SCORED = "pitches_acl2026_scored"
PROFILE_DIR = "/Users/akane/Desktop/new_stuff/stuff_plus/profiles/output/acl2026/json"


def step_fetch(full: bool = False, days: int = 2):
    if full:
        start = "2026-03-01"
    else:
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    logger.info(f"step_fetch: pulling {start} → {end}")
    df = download_aaa(2026, start_date=start, end_date=end, sleep_per_game=0.15, sport_id=SPORT_ID)
    if df.empty:
        logger.info("  nothing to add"); return 0
    append_to_db(df, table=TABLE)
    return len(df)


def step_score():
    """Score pitches in TABLE that aren't yet in TABLE_SCORED, using the same
    swing-residual no-foul model as the canonical regen script."""
    con = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if TABLE not in tables:
        logger.warning(f"{TABLE} doesn't exist"); con.close(); return
    if TABLE_SCORED in tables:
        existing = pd.read_sql(
            f"SELECT game_pk, at_bat_number, pitch_number FROM {TABLE_SCORED}", con)
    else:
        existing = pd.DataFrame(columns=["game_pk","at_bat_number","pitch_number"])
    raw_all = pd.read_sql(f"SELECT * FROM {TABLE}", con)
    con.close()
    if len(existing):
        # Build normalized keys for the dedup merge only — fresh gamefeed scrapes
        # can store game_pk/at_bat_number/pitch_number as text while the scored
        # table holds them as int64, which makes the merge fail on mixed dtypes.
        # We compute the mask on copies and index back into the untouched raw_all
        # so feature engineering downstream still sees the original dtypes.
        _KEYS = ["game_pk", "at_bat_number", "pitch_number"]
        left_keys = raw_all[_KEYS].apply(lambda s: pd.to_numeric(s, errors="coerce"))
        right_keys = existing[_KEYS].apply(lambda s: pd.to_numeric(s, errors="coerce")).drop_duplicates()
        merged = left_keys.merge(right_keys, on=_KEYS, how="left", indicator=True)
        to_score = raw_all[merged["_merge"].to_numpy() == "left_only"]
    else:
        to_score = raw_all
    if to_score.empty:
        logger.info("  no new pitches to score"); return
    logger.info(f"  scoring {len(to_score):,} new pitches…")
    pr = StuffPlusPredictor()
    eng, _ = engineer_features(to_score, baselines=pr.baselines)
    scored = pr.predict(eng, already_engineered=True, norm_set="current")
    # SQLite column names are case-insensitive — drop case-insensitive dupes
    # (e.g. VAA_adj vs vaa_adj) so the initial CREATE TABLE doesn't collide.
    seen, keep = {}, []
    for c in scored.columns:
        lc = c.lower()
        if lc in seen:
            continue
        seen[lc] = c; keep.append(c)
    if len(keep) != len(scored.columns):
        scored = scored[keep]
    con = sqlite3.connect(DB_PATH, timeout=60)
    scored.to_sql(TABLE_SCORED, con, if_exists="append" if len(existing) else "replace",
                  index=False, chunksize=5000)
    con.close()


def step_profiles():
    """Aggregate from TABLE_SCORED → profile JSONs in profiles/output/acl2026/json/."""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    from profiles.player_cards import generate_all_cards
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_SCORED}", con)
    con.close()
    if df.empty:
        logger.warning("  no scored data → skipping profile generation"); return
    generate_all_cards(df, season="acl2026", skip_png=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full season rebuild")
    ap.add_argument("--days", type=int, default=2,
                    help="incremental lookback window in days (default 2)")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-profiles", action="store_true")
    args = ap.parse_args()

    if not args.skip_fetch:
        n = step_fetch(full=args.full, days=args.days)
        logger.info(f"step_fetch: added {n:,} rows")
    if not args.skip_score:
        logger.info("step_score…")
        step_score()
    if not args.skip_profiles:
        logger.info("step_profiles…")
        step_profiles()
    logger.info("done.")


if __name__ == "__main__":
    main()

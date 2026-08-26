"""
FSL2026 unified updater: incremental fetch → score → profile gen.

Mirrors update_acl2026.py but for the Florida State League — the Low-A (Single-A,
MLB StatsAPI sportId=14) league played in Florida spring-training parks, which —
unlike most of Low-A — have Hawk-Eye tracking. Restricted to leagueId=123 so the
two other Single-A leagues (California=110, Carolina=122) are excluded.

Pipeline:
  1. Fetch new FSL games (last N days by default) via gamefeed JSON
     (sportId=14, leagueId=123)
  2. Append non-dup pitches to pitches_fsl2026
  3. Re-score the affected pitches via the model and APPEND to pitches_fsl2026_scored
  4. Aggregate scored table into profiles/output/fsl2026/json/

Run:
  python update_fsl2026.py             # incremental (last 2 days)
  python update_fsl2026.py --full      # full season rebuild (slow)
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
SPORT_ID = 14    # Single-A / Low-A
LEAGUE_ID = 123  # Florida State League (tracked subset of Single-A)
TABLE = "pitches_fsl2026"
TABLE_SCORED = "pitches_fsl2026_scored"
PROFILE_DIR = "/Users/akane/Desktop/new_stuff/stuff_plus/profiles/output/fsl2026/json"


def step_fetch(full: bool = False, days: int = 2):
    if full:
        start = "2026-03-01"
    else:
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    logger.info(f"step_fetch: pulling {start} → {end}")
    df = download_aaa(2026, start_date=start, end_date=end, sleep_per_game=0.15,
                      sport_id=SPORT_ID, league_id=LEAGUE_ID)
    if df.empty:
        logger.info("  nothing to add"); return 0
    append_to_db(df, table=TABLE)
    return len(df)


def step_score():
    """Score pitches in TABLE that aren't yet in TABLE_SCORED."""
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
    """Aggregate from TABLE_SCORED → profile JSONs in profiles/output/fsl2026/json/."""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    from profiles.player_cards import generate_all_cards
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_SCORED}", con)
    con.close()
    if df.empty:
        logger.warning("  no scored data → skipping profile generation"); return
    generate_all_cards(df, season="fsl2026", skip_png=True)


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

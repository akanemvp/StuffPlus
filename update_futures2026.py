"""
Futures Game 2026 updater: fetch → score → profile gen.

The MLB All-Star Futures Game (a prospect showcase during All-Star week) is filed
under MLB StatsAPI sportId=21 (Minor League Baseball umbrella), which on game day
returns *only* the Futures Game — so no extra filtering is needed. It is played at
the MLB All-Star venue with Hawk-Eye tracking, so the /gf gamefeed carries full
pitch data.

Because it is a single game (and live while it is being played), each run re-fetches
the whole game and REPLACES the table — always current, no dedup needed. Run it on a
short interval (launchd, every ~5 min) while the game is live.

Pipeline:
  1. Fetch the Futures Game via gamefeed JSON (sportId=21)
  2. Replace pitches_futures2026
  3. Score → pitches_futures2026_scored
  4. Aggregate → profiles/output/futures2026/json/

Run:
  python update_futures2026.py
"""
import os, sys, sqlite3, logging
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.aaa_scraper import download_aaa
from scraper.statcast_scraper import save_to_db
from features.engineering import engineer_features
from model.predict import StuffPlusPredictor

DB_PATH = "/Users/akane/Desktop/new_stuff/stuff_plus/data/statcast.db"
SPORT_ID = 21    # Minor League Baseball umbrella — returns only the Futures Game on game day
TABLE = "pitches_futures2026"
TABLE_SCORED = "pitches_futures2026_scored"
GAME_DATE = "2026-07-12"   # 2026 Futures Game date
PROFILE_DIR = "/Users/akane/Desktop/new_stuff/stuff_plus/profiles/output/futures2026/json"


def step_fetch():
    # small window around game day so a date rollover / late finish never misses it
    start = (datetime.strptime(GAME_DATE, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    end = (datetime.strptime(GAME_DATE, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info(f"step_fetch: pulling Futures Game {start} → {end} (sportId={SPORT_ID})")
    df = download_aaa(2026, start_date=start, end_date=end, sleep_per_game=0.15, sport_id=SPORT_ID)
    if df.empty:
        logger.info("  no tracked pitches yet"); return 0
    save_to_db(df, table=TABLE)          # replace — single game, always refresh whole thing
    return len(df)


def step_score():
    con = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if TABLE not in tables:
        logger.warning(f"{TABLE} doesn't exist"); con.close(); return
    raw = pd.read_sql(f"SELECT * FROM {TABLE}", con)
    con.close()
    if raw.empty:
        logger.info("  nothing to score"); return
    logger.info(f"  scoring {len(raw):,} pitches…")
    pr = StuffPlusPredictor()
    eng, _ = engineer_features(raw, baselines=pr.baselines)
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
    scored.to_sql(TABLE_SCORED, con, if_exists="replace", index=False, chunksize=5000)
    con.close()


def step_profiles():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    from profiles.player_cards import generate_all_cards
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_SCORED}", con)
    con.close()
    if df.empty:
        logger.warning("  no scored data → skipping profile generation"); return
    generate_all_cards(df, season="futures2026", skip_png=True)


def main():
    n = step_fetch()
    logger.info(f"step_fetch: {n:,} pitches")
    if n:
        logger.info("step_score…"); step_score()
        logger.info("step_profiles…"); step_profiles()
    logger.info("done.")


if __name__ == "__main__":
    main()

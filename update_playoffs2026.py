import os, sys, sqlite3, logging, argparse
from datetime import datetime, timedelta
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scraper.aaa_scraper import download_aaa
from scraper.statcast_scraper import download_date_range
from features.engineering import engineer_features
from model.predict import StuffPlusPredictor

DB_PATH = "/Users/akane/Desktop/new_stuff/stuff_plus/data/statcast.db"
SPORT_ID = 1
GAME_TYPES = "F,D,L,W"
SEASON_START = "2026-09-28"
TABLE = "pitches_playoffs2026"
TABLE_SCORED = "pitches_playoffs2026_scored"
PROFILE_DIR = "/Users/akane/Desktop/new_stuff/stuff_plus/profiles/output/playoffs2026/json"
KEYS = ["game_pk", "at_bat_number", "pitch_number"]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for k in KEYS + ["pitcher"]:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], format="mixed").dt.strftime("%Y-%m-%d")
    return df.dropna(subset=KEYS)


def _existing() -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        if TABLE not in tables:
            return pd.DataFrame()
        return _normalize(pd.read_sql(f"SELECT * FROM {TABLE}", con))
    finally:
        con.close()


def step_fetch(full: bool = False, days: int = 4) -> int:
    start = SEASON_START if full else max(
        SEASON_START, (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d"))
    end = datetime.today().strftime("%Y-%m-%d")
    logger.info(f"step_fetch: pulling {start} → {end}")

    feed = download_aaa(2026, start_date=start, end_date=end, sleep_per_game=0.15,
                        sport_id=SPORT_ID, game_types=GAME_TYPES)
    feed = _normalize(feed) if not feed.empty else feed

    try:
        csv = download_date_range(start, end)
    except Exception as exc:
        logger.warning(f"  statcast csv fetch failed: {exc}")
        csv = pd.DataFrame()
    if csv is not None and not csv.empty and "game_type" in csv.columns:
        csv = _normalize(csv[csv["game_type"].isin(GAME_TYPES.split(","))])
    else:
        csv = pd.DataFrame()

    existing = _existing()
    csv_pks = set(csv["game_pk"].unique()) if not csv.empty else set()
    feed_pks = (set(feed["game_pk"].unique()) - csv_pks) if not feed.empty else set()
    parts = []
    if not existing.empty:
        parts.append(existing[~existing["game_pk"].isin(csv_pks | feed_pks)])
    if not csv.empty:
        parts.append(csv)
    if feed_pks:
        parts.append(feed[feed["game_pk"].isin(feed_pks)])
    if not parts:
        logger.info("  nothing to add")
        return 0
    out = pd.concat(parts, ignore_index=True).drop_duplicates(subset=KEYS, keep="last")
    con = sqlite3.connect(DB_PATH, timeout=60)
    out.to_sql(TABLE, con, if_exists="replace", index=False, chunksize=5000)
    con.close()
    logger.info(f"  {len(out):,} pitches stored ({len(csv_pks)} games from statcast csv, {len(feed_pks)} from game feed)")
    return len(out)


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
    eng = engineer_features(raw)
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
    con.execute(f"CREATE INDEX IF NOT EXISTS ix_{TABLE_SCORED}_player ON [{TABLE_SCORED}](player_name)")
    con.execute(f"CREATE INDEX IF NOT EXISTS ix_{TABLE_SCORED}_pkey ON [{TABLE_SCORED}](game_pk, at_bat_number, pitch_number)")
    con.commit()
    con.close()


def step_profiles():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    from profiles.player_cards import generate_all_cards
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {TABLE_SCORED}", con)
    con.close()
    if df.empty:
        logger.warning("  no scored data → skipping profile generation"); return
    generate_all_cards(df, season="playoffs2026", skip_png=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--days", type=int, default=4)
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-profiles", action="store_true")
    args = ap.parse_args()

    if not args.skip_fetch:
        n = step_fetch(full=args.full, days=args.days)
        logger.info(f"step_fetch: {n:,} rows")
    if not args.skip_score:
        logger.info("step_score…")
        step_score()
    if not args.skip_profiles:
        logger.info("step_profiles…")
        step_profiles()
    logger.info("done.")


if __name__ == "__main__":
    main()

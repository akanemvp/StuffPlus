import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import schedule

from config import DB_PATH, PROFILES_DIR
from scraper.statcast_scraper import download_date_range, append_to_db, load_from_db, table_exists
from model.predict import StuffPlusPredictor
from profiles.player_cards import generate_all_cards

logger = logging.getLogger(__name__)

LIVE_TABLE   = "pitches_2026"


def _last_stored_date() -> str:
    season_start = "2026-03-26"
    if not table_exists(LIVE_TABLE):
        return season_start
    conn = sqlite3.connect(DB_PATH)
    try:
        result = conn.execute(
            f"SELECT MAX(game_date) FROM [{LIVE_TABLE}]"
        ).fetchone()
        last = result[0] if result and result[0] else None
    finally:
        conn.close()
    if last:
        return last[:10]
    return season_start


class LiveUpdater:

    def __init__(self):
        self.predictor = StuffPlusPredictor()
        self._json_dir = os.path.join(PROFILES_DIR, "2026", "json")
        os.makedirs(self._json_dir, exist_ok=True)

    @staticmethod
    def _cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if hasattr(df[col], "dtype") and hasattr(df[col].dtype, "numpy_dtype"):
                try:
                    df[col] = df[col].astype(df[col].dtype.numpy_dtype)
                except Exception:
                    df[col] = df[col].astype(object)
        return df

    def update(self):
        today = datetime.today().strftime("%Y-%m-%d")
        start = _last_stored_date()
        affected = set()

        if start <= today:
            logger.info(f"Fetching Statcast: {start} → {today}")
            df = download_date_range(start, today)
            if df is not None and not df.empty:
                df = self._only_new(self._cast_dtypes(df), LIVE_TABLE)
                if not df.empty:
                    logger.info(f"Scoring {len(df):,} new pitches…")
                    df_scored = self.predictor.predict(df)
                    append_to_db(df_scored, table=LIVE_TABLE)
                    append_to_db(df_scored, table=f"{LIVE_TABLE}_scored")
                    logger.info(f"Appended {len(df_scored):,} new pitches to '{LIVE_TABLE}' (+scored)")
                    affected |= set(df_scored["player_name"].dropna().unique())
                else:
                    logger.info("No genuinely new pitches.")
        else:
            logger.info("Already up to date through today.")

        try:
            affected |= set(self.refresh_recent())
        except Exception as e:
            logger.error(f"refresh_recent (arm_angle backfill) failed: {e}")

        if affected:
            self._update_cards(list(affected))

    def refresh_recent(self, days: int = 10) -> list:
        today = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        logger.info(f"Trailing refresh (arm_angle backfill): re-fetching {start} → {today}")
        fresh = download_date_range(start, today)
        if fresh is None or fresh.empty:
            return []
        fresh = self._cast_dtypes(fresh)

        append_to_db(fresh, LIVE_TABLE)

        conn = sqlite3.connect(DB_PATH)
        try:
            raw_win = pd.read_sql(
                f"SELECT * FROM [{LIVE_TABLE}] WHERE game_date >= ?", conn, params=(start,))
        finally:
            conn.close()
        if raw_win.empty:
            return []
        from storage.overrides import apply_overrides, count_overrides
        if count_overrides("2026"):
            raw_win = apply_overrides(raw_win, "2026")
        df_scored = self.predictor.predict(raw_win)

        conn = sqlite3.connect(DB_PATH, timeout=60)
        try:
            cols = [r[1] for r in conn.execute(
                f"PRAGMA table_info([{LIVE_TABLE}_scored])").fetchall()]
            keep = [c for c in cols if c in df_scored.columns]
            conn.execute(f"DELETE FROM [{LIVE_TABLE}_scored] WHERE game_date >= ?", (start,))
            conn.commit()
            df_scored[keep].to_sql(f"{LIVE_TABLE}_scored", conn,
                                   if_exists="append", index=False, chunksize=10000)
        finally:
            conn.close()
        logger.info(f"  refreshed {len(df_scored):,} pitches (arm_angle backfilled + re-scored)")
        return list(df_scored["player_name"].dropna().unique())

    def _only_new(self, df, table):
        key = ["game_pk", "at_bat_number", "pitch_number"]
        if not all(k in df.columns for k in key):
            return df
        conn = sqlite3.connect(DB_PATH)
        try:
            keys = pd.read_sql(f"SELECT {', '.join(key)} FROM [{table}]", conn).drop_duplicates()
        except Exception:
            return df
        finally:
            conn.close()
        if keys.empty:
            return df
        merged = df.merge(keys, on=key, how="left", indicator=True)
        return df[merged["_merge"].to_numpy() == "left_only"]

    def _update_cards(self, pitcher_names: list):
        try:
            df_all = load_from_db(f"{LIVE_TABLE}_scored")
            if df_all.empty:
                return
            generate_all_cards(df_all, season="2026")
            logger.info(f"Regenerated 2026 cards after update ({len(pitcher_names)} new pitchers)")
        except Exception as e:
            logger.error(f"Error updating 2026 cards: {e}")


    def run_scheduled(self, interval_seconds: int = 90):
        logger.info(f"Live updater starting (interval = {interval_seconds}s)")
        self.update()

        schedule.every(interval_seconds).seconds.do(self.update)
        while True:
            schedule.run_pending()
            time.sleep(10)


def run_live(interval_seconds: int = 90):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    updater = LiveUpdater()
    updater.run_scheduled(interval_seconds=interval_seconds)

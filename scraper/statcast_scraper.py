
import io
import os
import logging
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import requests
import pybaseball

from config import DB_PATH, DATA_DIR

logger = logging.getLogger(__name__)

SEASON_DATES = {
    2020: ("2020-07-23", "2020-09-27"),
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-20", "2024-09-29"),
    2025: ("2025-03-18", "2025-09-28"),
    2026: ("2026-03-26", None),
}


def _ensure_db_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def download_spring(year: int) -> pd.DataFrame:
    start_date = datetime(year, 2, 15)
    end_date   = datetime.today()

    frames = []
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(chunk_start + timedelta(days=2), end_date)
        s   = chunk_start.strftime("%Y-%m-%d")
        e   = chunk_end.strftime("%Y-%m-%d")
        e_lt = (chunk_end + timedelta(days=1)).strftime("%Y-%m-%d")
        url = (
            "https://baseballsavant.mlb.com/statcast_search/csv"
            f"?all=true&hfGT=S%7C&hfSea={year}%7C"
            f"&game_date_gt={s}&game_date_lt={e_lt}"
            "&player_type=pitcher&type=details"
            "&min_pitches=0&min_results=0&sort_col=pitches&sort_order=desc"
        )
        logger.info(f"Fetching spring training {s} → {e}")
        try:
            resp = requests.get(url, timeout=120,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            chunk_df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
            if not chunk_df.empty and "pitch_type" in chunk_df.columns:
                frames.append(chunk_df)
                logger.info(f"  Got {len(chunk_df):,} pitches")
        except Exception as exc:
            logger.warning(f"  Failed {s}→{e}: {exc}")

        chunk_start = chunk_end + timedelta(days=1)

    if not frames:
        logger.warning("No spring training data retrieved.")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if "pitch_type" in df.columns:
        df = df[df["pitch_type"].notna() & (df["pitch_type"] != "pitch_type")]
    logger.info(f"Spring training total: {len(df):,} pitches")
    return df


def download_date_range(start: str, end: str) -> pd.DataFrame:
    pybaseball.cache.enable()
    logger.info(f"Downloading {start} → {end}")
    df = pybaseball.statcast(start_dt=start, end_dt=end)
    return df


def save_to_db(df: pd.DataFrame, table: str = "pitches_train", replace: bool = True):
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    mode = "replace" if replace else "append"
    df.to_sql(table, conn, if_exists=mode, index=False)
    conn.close()
    logger.info(f"Saved {len(df):,} rows → table '{table}' (mode={mode})")


def append_to_db(df: pd.DataFrame, table: str):
    _ensure_db_dir()
    _KEY = ["game_pk", "at_bat_number", "pitch_number"]
    conn = sqlite3.connect(DB_PATH)
    try:
        existing = pd.read_sql(f"SELECT * FROM [{table}] LIMIT 0", conn)
        shared = [c for c in existing.columns if c in df.columns]
        df = df[shared]
        if all(k in df.columns for k in _KEY):
            keys = pd.read_sql(f"SELECT {', '.join(_KEY)} FROM [{table}]", conn).drop_duplicates()
            if len(keys):
                merged = df.merge(keys, on=_KEY, how="left", indicator=True)
                existing_rows = df[merged["_merge"].to_numpy() == "both"]
                new_rows      = df[merged["_merge"].to_numpy() == "left_only"]
                if len(existing_rows):
                    _backfill_null_columns(conn, table, existing_rows, _KEY)
                df = new_rows
    except Exception as exc:
        logger.warning(f"append_to_db: dedup/backfill path failed ({exc}); falling back to plain append")
    finally:
        conn.close()
    if len(df):
        save_to_db(df, table=table, replace=False)


def _backfill_null_columns(conn: sqlite3.Connection, table: str,
                           df_new: pd.DataFrame, key_cols: list) -> None:
    if df_new.empty:
        return
    tmp = "_aa_backfill_tmp"
    cur = conn.cursor()
    cur.execute(f"DROP TABLE IF EXISTS {tmp}")
    df_new.to_sql(tmp, conn, if_exists="replace", index=False)
    key_csv = ", ".join(key_cols)
    cur.execute(f"CREATE INDEX _aa_backfill_idx ON {tmp}({key_csv})")
    cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_bf_key ON {table}({key_csv})")
    non_key = [c for c in df_new.columns if c not in key_cols]
    if not non_key:
        cur.execute(f"DROP TABLE {tmp}")
        return
    set_clauses = ", ".join(f'"{c}" = COALESCE("{table}"."{c}", {tmp}."{c}")' for c in non_key)
    join_cond   = " AND ".join(f'"{table}"."{k}" = {tmp}."{k}"' for k in key_cols)
    sql = f'UPDATE "{table}" SET {set_clauses} FROM {tmp} WHERE {join_cond}'
    cur.execute(sql)
    cur.execute(f"DROP TABLE {tmp}")
    conn.commit()
    logger.info(f"  backfill: examined {len(df_new):,} existing rows in {table} for NULL→value updates")


def load_from_db(table: str = "pitches_train") -> pd.DataFrame:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(f"SELECT * FROM [{table}]", conn)
    finally:
        conn.close()
    logger.info(f"Loaded {len(df):,} rows from '{table}'")
    return df


def table_exists(table: str) -> bool:
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    )
    result = cursor.fetchone() is not None
    conn.close()
    return result



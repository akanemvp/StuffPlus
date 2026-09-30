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


def _ensure_db_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


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



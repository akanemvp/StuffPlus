
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from config import DB_PATH

TABLE = "pitch_overrides"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    season          TEXT    NOT NULL,          -- season key, e.g. '2026', 'aaa2026'
    game_pk         INTEGER NOT NULL,
    at_bat_number   INTEGER NOT NULL,
    pitch_number    INTEGER NOT NULL,
    player_name     TEXT,                      -- pitcher (denormalized for fast lookup)
    original_type   TEXT,                      -- label before the correction
    new_type        TEXT    NOT NULL,          -- the corrected pitch type
    source          TEXT    NOT NULL DEFAULT 'editor',
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    UNIQUE (season, game_pk, at_bat_number, pitch_number)
);
"""

INDEXES = [
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_season_player ON {TABLE} (season, player_name);",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_season_game   ON {TABLE} (season, game_pk);",
]


def _conn(conn: Optional[sqlite3.Connection] = None):
    if conn is not None:
        return conn, False
    return sqlite3.connect(DB_PATH, timeout=30), True


def init_schema(conn: Optional[sqlite3.Connection] = None) -> None:
    c, close = _conn(conn)
    try:
        c.execute(SCHEMA)
        for ix in INDEXES:
            c.execute(ix)
        c.commit()
    finally:
        if close:
            c.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_override(season: str, game_pk: int, at_bat_number: int, pitch_number: int,
                  new_type: str, player_name: Optional[str] = None,
                  original_type: Optional[str] = None, source: str = "editor",
                  conn: Optional[sqlite3.Connection] = None) -> None:
    c, close = _conn(conn)
    try:
        init_schema(c)
        now = _now()
        c.execute(
            f"""INSERT INTO {TABLE}
                (season, game_pk, at_bat_number, pitch_number, player_name,
                 original_type, new_type, source, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(season, game_pk, at_bat_number, pitch_number)
                DO UPDATE SET new_type   = excluded.new_type,
                              player_name= COALESCE(excluded.player_name, {TABLE}.player_name),
                              source     = excluded.source,
                              updated_at = excluded.updated_at""",
            (str(season), int(game_pk), int(at_bat_number), int(pitch_number),
             player_name, original_type, str(new_type), source, now, now))
        c.commit()
    finally:
        if close:
            c.close()


def save_many(season: str, rows: list, source: str = "editor",
              conn: Optional[sqlite3.Connection] = None) -> int:
    c, close = _conn(conn)
    try:
        init_schema(c)
        n = 0
        for r in rows:
            save_override(season, r["game_pk"], r["at_bat_number"], r["pitch_number"],
                          r["new_type"], r.get("player_name"), r.get("original_type"),
                          source, conn=c)
            n += 1
        return n
    finally:
        if close:
            c.close()


def get_overrides(season: str, player_name: Optional[str] = None,
                  game_pk: Optional[int] = None,
                  conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    c, close = _conn(conn)
    try:
        init_schema(c)
        q = f"SELECT * FROM {TABLE} WHERE season = ?"
        params: list = [str(season)]
        if player_name:
            q += " AND player_name = ?"
            params.append(player_name)
        if game_pk is not None:
            q += " AND game_pk = ?"
            params.append(int(game_pk))
        return pd.read_sql_query(q, c, params=params)
    finally:
        if close:
            c.close()


def delete_override(season: str, game_pk: int, at_bat_number: int, pitch_number: int,
                    conn: Optional[sqlite3.Connection] = None) -> int:
    c, close = _conn(conn)
    try:
        init_schema(c)
        cur = c.execute(
            f"""DELETE FROM {TABLE}
                WHERE season=? AND game_pk=? AND at_bat_number=? AND pitch_number=?""",
            (str(season), int(game_pk), int(at_bat_number), int(pitch_number)))
        c.commit()
        return cur.rowcount
    finally:
        if close:
            c.close()


def apply_overrides(df: pd.DataFrame, season: str,
                    conn: Optional[sqlite3.Connection] = None) -> pd.DataFrame:
    need = {"game_pk", "at_bat_number", "pitch_number", "pitch_type"}
    if df.empty or not need.issubset(df.columns):
        return df
    ov = get_overrides(season, conn=conn)
    if ov.empty:
        return df
    df = df.copy()
    key = lambda d: (pd.to_numeric(d["game_pk"], errors="coerce").astype("Int64").astype(str) + "_" +
                     pd.to_numeric(d["at_bat_number"], errors="coerce").astype("Int64").astype(str) + "_" +
                     pd.to_numeric(d["pitch_number"], errors="coerce").astype("Int64").astype(str))
    mapping = dict(zip(key(ov), ov["new_type"]))
    k = key(df)
    hit = k.map(mapping)
    df.loc[hit.notna(), "pitch_type"] = hit[hit.notna()].values
    return df


def count_overrides(season: Optional[str] = None,
                    conn: Optional[sqlite3.Connection] = None) -> int:
    c, close = _conn(conn)
    try:
        init_schema(c)
        if season:
            return c.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE season=?",
                             (str(season),)).fetchone()[0]
        return c.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
    finally:
        if close:
            c.close()

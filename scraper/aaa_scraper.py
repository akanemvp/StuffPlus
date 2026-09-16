from __future__ import annotations
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import requests

logger = logging.getLogger(__name__)

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
GAMEFEED_URL = "https://baseballsavant.mlb.com/gf"
HEADERS = {"User-Agent": "Mozilla/5.0"}

GAMEFEED_TO_STATCAST = {
    "pitch_type":       "pitch_type",
    "pitch_name":       "pitch_name",
    "start_speed":      "release_speed",
    "extension":        "release_extension",
    "spin_rate":        "release_spin_rate",
    "x0":               "release_pos_x",
    "y0":               "release_pos_y",
    "z0":               "release_pos_z",
    "ax":               "ax",  "ay": "ay", "az": "az",
    "vx0":              "vx0", "vy0":"vy0","vz0":"vz0",
    "plate_x":          "plate_x",
    "plate_z":          "plate_z",
    "pfxX":             "pfx_x",
    "pfxZ":             "pfx_z",
    "sz_top":           "sz_top",
    "sz_bot":           "sz_bot",
    "zone":             "zone",
    "pitch_call":       "description",
    "events":           "events",
    "hc_x":             "hc_x",
    "hc_y":             "hc_y",
    "launch_speed":     "launch_speed",
    "launch_angle":     "launch_angle",
    "ab_number":        "at_bat_number",
    "pitch_number":     "pitch_number",
    "pre_balls":        "balls",
    "pre_strikes":      "strikes",
    "batter":           "batter",
    "pitcher":          "pitcher",
    "p_throws":         "p_throws",
    "stand":            "stand",
    "inning":           "inning",
    "game_pk":          "game_pk",
    "year":             "game_year",
    "is_barrel":        "is_barrel",
}


def get_aaa_game_pks(start_date: str, end_date: str, sport_id: int = 11,
                     league_id: Optional[int] = None) -> List[dict]:
    out = []
    cur = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while cur <= end:
        ds = cur.strftime("%Y-%m-%d")
        try:
            params = {"sportId": sport_id, "date": ds}
            if league_id is not None:
                params["leagueId"] = league_id
            r = requests.get(SCHEDULE_URL, params=params, timeout=30, headers=HEADERS)
            r.raise_for_status()
            d = r.json()
            for date_block in d.get("dates", []):
                for g in date_block.get("games", []):
                    out.append({
                        "gamePk":   g.get("gamePk"),
                        "gameDate": ds,
                        "status":   g.get("status", {}).get("detailedState"),
                        "home":     g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", ""),
                        "away":     g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", ""),
                    })
        except Exception as e:
            logger.warning(f"schedule fetch failed for {ds}: {e}")
        cur += timedelta(days=1)
    return out


def fetch_gamefeed(game_pk: int) -> Optional[dict]:
    try:
        r = requests.get(GAMEFEED_URL, params={"game_pk": game_pk}, timeout=60, headers=HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"gamefeed fetch failed for {game_pk}: {e}")
        return None


def parse_gamefeed_to_pitches(d: dict) -> pd.DataFrame:
    rows = []
    game_date = d.get("gameDate", "")
    try:
        game_date = datetime.strptime(game_date, "%m/%d/%Y").strftime("%Y-%m-%d")
    except Exception:
        pass
    home_team = d.get("home_team_data", {}).get("abbreviation", "")
    away_team = d.get("away_team_data", {}).get("abbreviation", "")
    home_team_id = d.get("home_team_data", {}).get("team_id")
    away_team_id = d.get("away_team_data", {}).get("team_id")

    for side in ("away_pitchers", "home_pitchers"):
        for pid_str, plist in d.get(side, {}).items():
            for p in plist:
                half = p.get("half_inning", "")
                inning_topbot = "Top" if half == "top" else "Bot"
                row = {}
                for src, dst in GAMEFEED_TO_STATCAST.items():
                    row[dst] = p.get(src)
                if row.get("pfx_x") is not None:
                    try: row["pfx_x"] = -float(row["pfx_x"])
                    except Exception: pass
                row["game_date"]     = game_date
                _pname = p.get("pitcher_name", "")
                if _pname and "," not in _pname:
                    _np = _pname.strip().split()
                    if len(_np) >= 3 and _np[-1].lower().rstrip(".") in {"jr", "sr", "ii", "iii", "iv"}:
                        _pname = f"{' '.join(_np[-2:])}, {' '.join(_np[:-2])}"
                    elif len(_np) >= 2:
                        _pname = f"{_np[-1]}, {' '.join(_np[:-1])}"
                row["player_name"]   = _pname
                row["inning_topbot"] = inning_topbot
                row["home_team"] = home_team
                row["away_team"] = away_team
                rows.append(row)
    return pd.DataFrame(rows)


def download_aaa(year: int, start_date: Optional[str] = None,
                 end_date: Optional[str] = None, sleep_per_game: float = 0.0,
                 workers: int = 6, sport_id: int = 11,
                 league_id: Optional[int] = None) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    if start_date is None:
        start_date = f"{year}-03-01"
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    logger.info(f"Discovering game_pks (sportId={sport_id}, leagueId={league_id}) {start_date} → {end_date}")
    games = get_aaa_game_pks(start_date, end_date, sport_id=sport_id, league_id=league_id)
    interesting = [g for g in games if g.get("status") in ("Final", "Completed Early", "In Progress", "Game Over", "Manager challenge")]
    logger.info(f"Found {len(games)} games; {len(interesting)} have pitch data")

    def _fetch_and_parse(g):
        d = fetch_gamefeed(g["gamePk"])
        if d is None:
            return None
        return parse_gamefeed_to_pitches(d)

    frames = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_and_parse, g): g for g in interesting}
        for fut in as_completed(futures):
            done += 1
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as exc:
                logger.warning(f"fetch worker failed: {exc}")
            if done % 50 == 0:
                logger.info(f"  fetched {done}/{len(interesting)} games")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "pitch_type" in out.columns:
        out = out[out["pitch_type"].notna() & (out["pitch_type"] != "")]
    logger.info(f"Total pitches (sportId={sport_id}): {len(out):,}")
    return out



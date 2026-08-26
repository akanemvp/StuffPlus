"""
Player card generator.

For each pitcher, produces:
  • A dark-theme matplotlib figure showing per-pitch-type Stuff+ grades,
    movement profile, VAA, HAA, velocity, spin, and extension.
  • A JSON summary file for the dashboard.
"""

import json
import logging
import os
import tempfile
import time
import urllib.request

import numpy as np
import pandas as pd

# matplotlib is imported lazily inside _generate_card_from_summary so that
# the 30 MB fontManager cost is only paid when actually generating PNG cards
# (skip_png=True refresh cycles never call that function).

from config import PITCH_TYPES, PROFILES_DIR

logger = logging.getLogger(__name__)


def _atomic_json_write(path: str, obj) -> None:
    """Write *obj* as JSON to *path* atomically (write temp → rename).

    A plain ``open(path, 'w')`` leaves a partial file visible to concurrent
    readers for the entire duration of the write.  On typical spring-training
    leaderboards (~1 300 entries, ~100 KB) the write takes only a few ms, but
    that is still long enough to race with the 90-second background refresh.
    Writing to a sibling temp file and renaming is atomic on POSIX filesystems
    so readers always see either the old complete file or the new complete file,
    never a partial one.
    """
    dir_ = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp, path)   # atomic on POSIX; overwrites destination
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Color palette (dark theme)
# ---------------------------------------------------------------------------
BG_DARK = "#0d1117"
BG_CARD = "#161b22"
BORDER_DEFAULT = "#30363d"

GRADE_COLORS = {
    "elite":     "#00b4d8",   # > +2σ  (120+)
    "great":     "#4cc9f0",   # +1–2σ  (110–120)
    "above_avg": "#7bf1a8",   # +0.5–1σ(105–110)
    "average":   "#c9d1d9",   # ±0.5σ  (95–105)
    "below_avg": "#f4a261",   # -1–0σ  (90–95)
    "poor":      "#ef233c",   # < -1σ  (< 90)
}

# Scouting grade scale anchored to each pitch type's training-data mean (pt_mean).
# Grade 50 = average for THAT pitch type, not a universal flat 100.
# Thresholds apply after shifting sp by (100 - pt_mean):
#   80 = ≥ pt_mean + 30  (top ~0.1%)
#   70 = ≥ pt_mean + 20  (top ~2%)    ← elite pitch
#   60 = ≥ pt_mean + 10  (top ~16%)
#   55 = ≥ pt_mean +  5  (top ~31%)
#   50 = ≥ pt_mean       (average for type)
#   45 = ≥ pt_mean -  5  (below avg)
#   40 = ≥ pt_mean - 10  (well below avg)
#   30 = <  pt_mean - 20
def stuff_grade(sp: float, pt_mean: float = 100.0) -> tuple:
    """
    Return (20-80 scouting grade, hex color) for a Stuff+ value.

    Grade thresholds are anchored to the pitch type's own training-data mean
    so that Grade 50 always means "approximately average for that pitch type".
    Pitch types whose model-learned mean differs from 100 (e.g., CU ≈ 94,
    KC ≈ 95) no longer receive systematically suppressed grades.

    pt_mean anchors the thresholds to a pitch type's own mean; it defaults to
    100.0 (flat scale), which is what all current callers pass since grades are
    globally calibrated to 100.
    """
    # Shift so that pt_mean maps to 100, then apply standard flat thresholds
    s = sp + (100.0 - pt_mean)
    if s >= 130: return ("80", GRADE_COLORS["elite"])
    if s >= 120: return ("70", GRADE_COLORS["elite"])
    if s >= 110: return ("60", GRADE_COLORS["great"])
    if s >= 105: return ("55", GRADE_COLORS["above_avg"])
    if s >= 100: return ("50", GRADE_COLORS["average"])
    if s >= 95:  return ("45", GRADE_COLORS["average"])
    if s >= 90:  return ("40", GRADE_COLORS["below_avg"])
    if s >= 80:  return ("35", GRADE_COLORS["poor"])
    return ("30", GRADE_COLORS["poor"])


def _nan_to_none(v):
    """Convert NaN / numpy scalars to a JSON-safe Python float or None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f   # f != f is True only for NaN
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Boxscore helper (MLB Stats API fallback for /gf-only games without bat_score)
# ---------------------------------------------------------------------------

_boxscore_er_cache: dict[int, dict[int, int]] = {}


_boxscore_ip_cache: dict[int, dict[int, int]] = {}  # game_pk → {pitcher_id → outs}

def _get_boxscore_earned_runs(game_pk: int) -> dict[int, int]:
    """Return {pitcher_id: earned_runs} and {pitcher_id: outs} from MLB Stats API.
    Cached permanently per game_pk: batch profile builds fetch each game once (a TTL here
    would re-fetch games shared across pitchers during a long build and stall it). The live
    game-line IP for in-progress games is refreshed separately in app.py's own TTL'd copy."""
    if game_pk in _boxscore_er_cache:
        return _boxscore_er_cache[game_pk]
    result: dict[int, int] = {}
    ip_result: dict[int, int] = {}
    fetched = False
    try:
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        for team in ("home", "away"):
            players = data["teams"][team]["players"]
            for pid in data["teams"][team].get("pitchers", []):
                p = players.get(f"ID{pid}", {})
                stats = p.get("stats", {}).get("pitching", {})
                er = stats.get("earnedRuns")
                if er is not None:
                    result[int(pid)] = int(er)
                # inningsPitched is a string like "1.0", "0.2" — convert to outs
                ip_str = stats.get("inningsPitched")
                if ip_str is not None:
                    try:
                        parts = str(ip_str).split(".")
                        ip_outs = int(parts[0]) * 3 + int(parts[1]) if len(parts) == 2 else int(parts[0]) * 3
                        ip_result[int(pid)] = ip_outs
                    except Exception:
                        pass
        fetched = True
    except Exception:
        pass
    # Only cache a successful fetch; on failure keep serving the last good cache.
    if fetched:
        _boxscore_er_cache[game_pk] = result
        _boxscore_ip_cache[game_pk] = ip_result
    return _boxscore_er_cache.get(game_pk, result)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def summarize_pitcher(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate pitch-level data to one row per pitch type.

    All FS pitches are aggregated together — the unified splitter model
    handles both hard zapballs and traditional splitters, so no threshold
    split is needed at the card level.

    Includes all pitch types with at least 1 scored pitch. Every scorable pitch
    now counts (no velocity floor) — n uses ("stuff_plus", "count") which still
    skips any NaN (e.g. a pitch missing shape features). Sorted by usage (count desc).

    New fields added for profile visualizations:
      - p_throws        : pitcher handedness ("R" or "L")
      - usage_pct       : % of total pitches (float)
      - usage_pct_vs_r  : % of pitches thrown vs RHB (of all pitches vs RHB)
      - usage_pct_vs_l  : % of pitches thrown vs LHB (of all pitches vs LHB)
      - spin_axis_clock : spin_axis in clock-face format ("10:30")
      - zone_pct        : % of pitches in strike zone (zones 1-9)
      - chase_pct       : % of out-of-zone pitches swung at
      - whiff_pct       : % of swings that were whiffs
      - xwoba_contact   : avg xwOBA on balls in play
      - locations_x/z_vs_r/l : sampled plate coords by batter handedness
    """
    df = df.copy()

    # --- Pitcher handedness (same for all rows) ---
    p_throws_val = "R"
    if "p_throws" in df.columns:
        _mode = df["p_throws"].dropna().mode()
        if len(_mode) > 0:
            p_throws_val = str(_mode.iloc[0])

    # --- Pitcher's team: the MOST-RECENT game's team (handles mid-season trades) ---
    team_val = ""
    if all(c in df.columns for c in ["inning_topbot", "home_team", "away_team"]):
        # Top of inning = away team bats = home team pitches; Bot = away team pitches
        pitcher_teams = df.apply(
            lambda r: r["home_team"] if r["inning_topbot"] == "Top" else r["away_team"],
            axis=1,
        )
        if "game_date" in df.columns and df["game_date"].notna().any():
            _t = pd.DataFrame({"team": pitcher_teams,
                               "d": pd.to_datetime(df["game_date"], errors="coerce")}).dropna()
            if len(_t):
                # among the latest date, take that day's most common team (double-headers etc.)
                _latest = _t[_t["d"] == _t["d"].max()]["team"].mode()
                if len(_latest):
                    team_val = str(_latest.iloc[0])
        if not team_val:                              # fallback: mode across all pitches
            _tm = pitcher_teams.dropna().mode()
            if len(_tm) > 0:
                team_val = str(_tm.iloc[0])

    # --- Pitcher MLB ID ---
    pitcher_id_val = None
    if "pitcher" in df.columns:
        _pid = df["pitcher"].dropna().mode()
        if len(_pid) > 0:
            try:
                pitcher_id_val = int(_pid.iloc[0])
            except (ValueError, TypeError):
                pass

    # --- Age ---
    age_val = None
    if "age_pit" in df.columns:
        _age = df["age_pit"].dropna()
        if len(_age) > 0:
            try:
                age_val = int(round(float(_age.mean())))
            except (ValueError, TypeError):
                pass

    # --- Innings pitched and ERA ---
    _out_events = {
        "strikeout", "field_out", "force_out", "sac_fly", "sac_bunt",
        "fielders_choice_out", "fielders_choice", "other_out",
        "caught_stealing_2b", "caught_stealing_3b", "caught_stealing_home",
        "pickoff_1b", "pickoff_2b", "pickoff_3b",
        "pickoff_caught_stealing_2b", "pickoff_caught_stealing_3b",
        "pickoff_caught_stealing_home",
    }
    _dp_events = {
        "double_play", "grounded_into_double_play", "strikeout_double_play",
        "sac_fly_double_play", "triple_play",
    }
    # Stockyard (Baseball Savant /gf) format: Title-Case, repeated on every pitch.
    # _out_events_cap: each matching event = 1 out.
    # _dp_events_cap: each matching event = +1 extra out (so DP events in both = 2 total).
    _out_events_cap = {
        # Single outs
        "Groundout", "Flyout", "Lineout", "Pop Out", "Forceout", "Strikeout",
        "Sac Fly", "Sac Bunt", "Bunt Groundout",
        "Fielders Choice", "Fielders Choice Out",
        # Double plays (also in _dp_events_cap → 2 outs)
        "Double Play", "GIDP", "Strikeout Double Play", "Sac Fly Double Play",
        # Triple play
        "Triple Play",
        # Caught stealing / pickoffs
        "Caught Stealing 2B", "Caught Stealing 3B", "Caught Stealing Home",
        "Pickoff 1B", "Pickoff 2B", "Pickoff 3B",
        "Pickoff Caught Stealing 2B", "Pickoff Caught Stealing 3B",
        "Pickoff Caught Stealing Home",
    }
    _dp_events_cap = {
        "Double Play", "GIDP", "Strikeout Double Play", "Sac Fly Double Play",
        "Triple Play",
    }

    total_outs = 0
    total_runs = 0
    _has_run_data = False
    if "events" in df.columns:
        def _event_outs(ev_series):
            """Outs from a per-at-bat-collapsed event series. Handles Statcast
            (lowercase) + /gf Title-Case sets; DP events count as 2 outs."""
            single = int(ev_series.isin(_out_events | _out_events_cap).sum())
            dp_gf  = int(ev_series.isin(_dp_events_cap).sum())      # /gf DP: +1 (already 1 in _out_events_cap)
            dp_sc  = int(ev_series.isin(_dp_events).sum() * 2)      # Statcast DP: not in single set → ×2
            return single + dp_gf + dp_sc

        pitcher_id_int = None
        if "pitcher" in df.columns:
            _pid_mode = df["pitcher"].dropna().mode()
            if not _pid_mode.empty:
                try:
                    pitcher_id_int = int(_pid_mode.iloc[0])
                except (ValueError, TypeError):
                    pass

        # ── Innings pitched + runs, aggregated PER GAME ──
        # Outs are summed per game using the official boxscore as authoritative
        # (it captures baserunning outs — pickoffs / caught stealing — and the
        # trailing out of a double play that the pitch-event stream omits).
        # Event-counting is the per-game fallback when the boxscore lacks the
        # pitcher. Doing this per game is essential: comparing a single game's
        # boxscore outs against the season-wide event total (the old bug) both
        # failed to add DP/baserunning outs and could clobber the season total.
        if "game_pk" in df.columns:
            for gp in df["game_pk"].dropna().unique():
                gp_int  = int(gp)
                gp_df   = df[df["game_pk"] == gp]
                if "at_bat_number" in gp_df.columns and gp_df["at_bat_number"].notna().any():
                    ev_g = gp_df.groupby("at_bat_number")["events"].last().dropna()
                else:
                    _e = gp_df["events"]
                    ev_g = _e.groupby((_e != _e.shift()).cumsum()).last().dropna()
                event_outs = _event_outs(ev_g)

                # Runs: bat_score diff first (Statcast CSV), else boxscore ER
                game_runs = None
                if all(c in gp_df.columns for c in ["bat_score", "post_bat_score"]):
                    ab_ends = gp_df[gp_df["events"].notna() & gp_df["bat_score"].notna()]
                    if not ab_ends.empty:
                        run_diff = (pd.to_numeric(ab_ends["post_bat_score"], errors="coerce")
                                    - pd.to_numeric(ab_ends["bat_score"], errors="coerce")).clip(lower=0)
                        game_runs = int(run_diff.sum())

                bs_outs = None
                if pitcher_id_int is not None:
                    er_map = _get_boxscore_earned_runs(gp_int)  # also fills _boxscore_ip_cache
                    if game_runs is None:
                        er = er_map.get(pitcher_id_int)
                        if er is not None:
                            game_runs = er
                    bs_outs = _boxscore_ip_cache.get(gp_int, {}).get(pitcher_id_int)

                total_outs += bs_outs if bs_outs is not None else event_outs
                if game_runs is not None:
                    total_runs += game_runs
                    _has_run_data = True
        else:
            # No game_pk to group on — collapse consecutive identical events.
            _e = df["events"]
            ev_all = _e.groupby((_e != _e.shift()).cumsum()).last().dropna()
            total_outs = _event_outs(ev_all)

    ip_str_val = f"{total_outs // 3}.{total_outs % 3}"
    era_val = round((total_runs / (total_outs / 3)) * 9, 2) if (total_outs > 0 and _has_run_data) else None

    # --- Core aggregation ---
    agg_dict = dict(
        stuff_plus=("stuff_plus", "mean"),
        n=("stuff_plus", "count"),
        velo=("release_speed", "mean"),
        ivb=("pfx_z_in", "mean"),
        hb=("pfx_x_arm", "mean"),   # arm-side positive for both RHP and LHP
        vaa=("vaa", "mean"),
        haa=("haa", "mean"),
        spin=("release_spin_rate", "mean"),
        extension=("release_extension", "mean"),
        arm_angle=("arm_angle", "mean"),
        release_height=("release_pos_z", "mean"),
        release_side=("release_pos_x", "mean"),   # raw horizontal release (catcher view)
    )
    agg = df.groupby("pitch_type").agg(**agg_dict).reset_index()

    # --- Pitch-type grade = MEAN of the per-pitch grades ---
    # agg["stuff_plus"] is already the per-pitch mean (agg_dict above). Kept as-is on
    # purpose: the pitch-type score is the average of each pitch's own stuff_plus, not
    # the grade of the averaged shape. Individual pitches keep their own stuff_plus for
    # per-pitch leaderboards and the editor.

    # Circular mean for spin_axis (0–360° circular scale).
    # Arithmetic mean is wrong near the 0°/360° boundary: two pitches at 355° and
    # 5° would average to 180° instead of 0°. Use sin/cos component averaging.
    if "spin_axis" in df.columns:
        import numpy as _np
        _rad = _np.deg2rad(df["spin_axis"].fillna(180))
        _sa_circ = df.assign(_sin=_np.sin(_rad), _cos=_np.cos(_rad)) \
            .groupby("pitch_type")[["_sin", "_cos"]].mean()
        _angles = _np.degrees(_np.arctan2(_sa_circ["_sin"], _sa_circ["_cos"])) % 360
        # % 360 can yield 360.0 (not 0.0) for near-zero negative floats; clamp it.
        _sa_circ["spin_axis"] = _np.where(_np.abs(_angles - 360.0) < 1e-9, 0.0, _angles)
        agg = agg.merge(_sa_circ[["spin_axis"]], on="pitch_type", how="left")

    agg = agg[agg["n"] >= 1].reset_index(drop=True)
    agg = agg.sort_values("n", ascending=False).reset_index(drop=True)

    agg["p_throws"]    = p_throws_val
    agg["team"]        = team_val
    agg["pitcher_id"]  = pitcher_id_val
    agg["age"]         = age_val
    agg["ip_str"]      = ip_str_val
    agg["era"]         = era_val

    # --- Usage % ---
    total_n = agg["n"].sum()
    agg["usage_pct"] = (agg["n"] / total_n * 100).round(1) if total_n > 0 else 0.0

    # --- Spin axis → clock face ---
    def axis_to_clock(deg):
        if pd.isna(deg):
            return "—"
        # Statcast spin_axis is the gyro axis direction; adding 180° converts
        # to the movement tilt direction (what Baseball Savant displays as "tilt").
        deg_tilt = (float(deg) + 180) % 360
        total_min = round(deg_tilt * 2)
        h = (total_min // 60) % 12
        if h == 0:
            h = 12
        m = total_min % 60
        return f"{h}:{m:02d}"

    if "spin_axis" in agg.columns:
        agg["spin_axis_clock"] = agg["spin_axis"].apply(axis_to_clock)
    else:
        agg["spin_axis_clock"] = "—"

    # --- Per-pitch-type outcome stats and split heatmap locations ---
    _has_zone  = "zone" in df.columns and "description" in df.columns
    _has_xwoba = "estimated_woba_using_speedangle" in df.columns
    _has_locs  = "plate_x" in df.columns and "plate_z" in df.columns
    _has_stand = "stand" in df.columns

    # Total pitches thrown vs each handedness (across all pitch types)
    _total_vs_r = int((df["stand"] == "R").sum()) if _has_stand else 0
    _total_vs_l = int((df["stand"] == "L").sum()) if _has_stand else 0

    _swing_desc = {
        # /gf snake_case
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "foul_bunt", "missed_bunt",
        "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
        # Baseball Savant CSV title-case
        "Swinging Strike", "Swinging Strike (Blocked)",
        "Foul", "Foul Tip", "Foul Bunt", "Missed Bunt",
        "In play, out(s)", "In play, no out", "In play, runs(s)",
    }
    _contact_desc = {
        # /gf snake_case
        "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
        # Baseball Savant CSV / /gf title-case
        "In play, out(s)", "In play, no out", "In play, run(s)", "In play, runs(s)",
    }

    def _sample_locs(sub_df, max_n=400):
        if not _has_locs or sub_df.empty:
            return [], []
        valid = sub_df[["plate_x", "plate_z"]].dropna()
        if len(valid) > max_n:
            valid = valid.sample(max_n, random_state=42)
        return (
            [round(v, 2) for v in valid["plate_x"].tolist()],
            [round(v, 2) for v in valid["plate_z"].tolist()],
        )

    extra_rows = []
    for pt, pdata in df.groupby("pitch_type"):
        row: dict = {"pitch_type": pt}

        # Outcome stats
        if _has_zone and len(pdata) > 0:
            in_zone     = pdata["zone"].between(1, 9)
            is_swing    = pdata["description"].isin(_swing_desc)
            is_whiff    = pdata["description"].isin(
                {"swinging_strike", "swinging_strike_blocked",
                 "Swinging Strike", "Swinging Strike (Blocked)"}
            )
            out_of_zone = ~in_zone

            row["zone_pct"] = float(in_zone.mean() * 100)

            ooz_n = int(out_of_zone.sum())
            row["chase_pct"] = (
                float((is_swing & out_of_zone).sum() / ooz_n * 100)
                if ooz_n > 0 else None
            )
            swing_n = int(is_swing.sum())
            row["whiff_pct"] = (
                float(is_whiff.sum() / swing_n * 100)
                if swing_n > 0 else None
            )
        else:
            row["zone_pct"] = row["chase_pct"] = row["whiff_pct"] = None

        if _has_xwoba and _has_zone and len(pdata) > 0:
            is_contact = pdata["description"].isin(_contact_desc)
            xw_vals = pd.to_numeric(
                pdata.loc[is_contact, "estimated_woba_using_speedangle"], errors="coerce"
            ).dropna()
            row["xwoba_contact"] = round(float(xw_vals.mean()), 3) if len(xw_vals) > 0 else None
        else:
            row["xwoba_contact"] = None

        # Handedness splits
        vs_r = pdata[pdata["stand"] == "R"] if _has_stand else pd.DataFrame()
        vs_l = pdata[pdata["stand"] == "L"] if _has_stand else pd.DataFrame()

        row["n_vs_r"] = len(vs_r)
        row["n_vs_l"] = len(vs_l)
        row["usage_pct_vs_r"] = (
            round(len(vs_r) / _total_vs_r * 100, 1) if _total_vs_r > 0 else 0.0
        )
        row["usage_pct_vs_l"] = (
            round(len(vs_l) / _total_vs_l * 100, 1) if _total_vs_l > 0 else 0.0
        )

        lx_r, lz_r = _sample_locs(vs_r)
        lx_l, lz_l = _sample_locs(vs_l)
        row["locations_x_vs_r"] = lx_r
        row["locations_z_vs_r"] = lz_r
        row["locations_x_vs_l"] = lx_l
        row["locations_z_vs_l"] = lz_l

        # Combined locations (still keep for any legacy use)
        lx_all, lz_all = _sample_locs(pdata, max_n=500)
        row["locations_x"] = lx_all
        row["locations_z"] = lz_all

        extra_rows.append(row)

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows)
        agg = agg.merge(extra_df, on="pitch_type", how="left")

    return agg


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def _pitch_panel(ax, row: pd.Series):
    """Draw one pitch-type panel inside a given Axes."""
    pitch_name = PITCH_TYPES.get(row["pitch_type"], row["pitch_type"])
    sp = float(row["stuff_plus"])
    grade, color = stuff_grade(sp, pt_mean=float(row.get("pt_mean", 100.0)))

    ax.set_facecolor(BG_CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1.8)

    # Pitch type header
    ax.text(0.5, 0.97, pitch_name, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, fontweight="bold", color="white")

    # Big Stuff+ number
    ax.text(0.5, 0.72, f"{sp:.0f}", transform=ax.transAxes,
            ha="center", va="center", fontsize=30, fontweight="bold", color=color)

    # Scouting grade
    ax.text(0.5, 0.55, f"Grade {grade}", transform=ax.transAxes,
            ha="center", va="center", fontsize=8, color=color)

    # Stats block
    sign_hb  = "+" if row["hb"]  >= 0 else ""
    sign_ivb = "+" if row["ivb"] >= 0 else ""
    rel_ht   = row.get("release_height", float("nan"))
    rel_ht_s = f"{rel_ht:.1f} ft" if not (rel_ht != rel_ht) else "—"
    lines = [
        f"{row['velo']:.1f} mph",
        f"IVB {sign_ivb}{row['ivb']:.1f}\"  HB {sign_hb}{row['hb']:.1f}\"",
        f"VAA {row['vaa']:.2f}°   HAA {row['haa']:.2f}°",
        f"Spin {row['spin']:.0f} rpm",
        f"Ext {row['extension']:.1f} ft  Rel Ht {rel_ht_s}",
        f"N = {row['n']:.0f}",
    ]
    ax.text(0.5, 0.34, "\n".join(lines), transform=ax.transAxes,
            ha="center", va="center", fontsize=7, color="#cccccc",
            linespacing=1.6, family="monospace")

    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def _canonicalize_names(df: pd.DataFrame) -> pd.DataFrame:
    """Unify a pitcher's display name across data-source spellings so one pitcher
    never splits into two cards. /gf mis-parses suffixes (Samy Natera Jr. →
    'Jr., Samy Natera') and strips accents; pick per pitcher id the variant with the
    fewest tokens after the comma (proper 'Surname, First'), then accented."""
    if "pitcher" not in df.columns or "player_name" not in df.columns:
        return df
    def _after_comma(n): return len(str(n).split(",", 1)[1].split()) if "," in str(n) else len(str(n).split())
    def _has_accent(s):  return any(ord(c) > 127 for c in str(s))
    def _score(n):       return (-_after_comma(n), _has_accent(n), len(str(n)))
    sub = df.dropna(subset=["pitcher", "player_name"])
    best = {pid: max(g["player_name"].unique(), key=_score) for pid, g in sub.groupby("pitcher")}
    if not best:
        return df
    df = df.copy()
    pit = df["pitcher"]
    df["player_name"] = [best.get(p, n) if pd.notna(p) else n
                         for p, n in zip(pit, df["player_name"])]
    return df


def generate_all_cards(df: pd.DataFrame, season: int, skip_png: bool = False,
                       prune: bool = True) -> list:
    """
    Generate cards for every pitcher in df.

    Steps:
      1. Aggregate each pitcher's raw pitch data into per-pitch-type summaries.
      2. Recalibrate Stuff+ ACROSS ALL PITCHERS per pitch type
         (so 100 = league average for that pitch, 120 = elite).
      3. Generate PNG card + JSON profile for each pitcher.
    """
    df = _canonicalize_names(df)  # merge /gf name-spelling variants by pitcher id
    out_dir  = os.path.join(PROFILES_DIR, str(season))
    json_dir = os.path.join(out_dir, "json")
    os.makedirs(json_dir, exist_ok=True)

    import glob as _glob
    # Snapshot existing files BEFORE generating — we'll remove stale ones at the end
    existing_json = set(_glob.glob(os.path.join(json_dir, f"*_{season}.json")))
    existing_png  = set(_glob.glob(os.path.join(out_dir,  f"*_{season}.png")))

    # --- Step 1: raw summaries for all pitchers ---
    # No velocity/eephus gate: every pitcher gets a card, position players included.
    # Their fastballs grade badly on merit and sink to the bottom of the boards.
    raw_summaries = {}
    for name, pdata in df.groupby("player_name"):
        s = summarize_pitcher(pdata)
        if not s.empty:
            s = s.copy()
            s["player_name"] = name
            raw_summaries[name] = s

    if not raw_summaries:
        logger.warning("No valid pitcher summaries — check that model was trained and profiles re-run.")
        return []

    logger.info(f"Summarised {len(raw_summaries)} pitchers for season {season}")

    # --- Step 2: combine summaries ---
    # Per-pitch scores are already calibrated to mean=100, std=10 per pitch type
    # by recalibrate() in train.py. The average of those per-pitch scores IS the
    # profile score — no secondary pitcher-level normalization is applied.
    combined = pd.concat(raw_summaries.values(), ignore_index=True)
    combined["pt_mean"] = 100.0  # all pitch types calibrated to 100; used by stuff_grade()

    # Rebuild per-pitcher lookup with recalibrated scores
    calibrated = {
        name: grp.drop(columns="player_name").reset_index(drop=True)
        for name, grp in combined.groupby("player_name")
    }

    # --- Step 3: generate cards + JSON (write first, remove stale at the end) ---
    paths = []
    new_json = set()
    new_png  = set()
    for name, summary in calibrated.items():
        path = None if skip_png else _generate_card_from_summary(name, summary, season, out_dir)
        profile = _build_profile_from_summary(name, summary, season, card_path=path)

        if path:
            paths.append(path)
            new_png.add(path)

        safe  = name.replace(", ", "_").replace(" ", "_").replace("'", "")
        jpath = os.path.join(json_dir, f"{safe}_{season}.json")
        _atomic_json_write(jpath, profile)
        new_json.add(jpath)

    # Leaderboard — written atomically so a concurrent read never sees partial JSON
    leaderboard = _build_leaderboard_from_calibrated(combined, season)
    lb_path = os.path.join(json_dir, f"leaderboard_{season}.json")
    _atomic_json_write(lb_path, leaderboard)
    new_json.add(lb_path)

    # Remove stale files that were NOT regenerated (players no longer in data).
    # Only when pruning a FULL-season frame: a single-pitcher rebuild (from an
    # editor save) passes prune=False so it never deletes everyone else's cards.
    if prune:
        for old in existing_json - new_json:
            try:
                os.remove(old)
            except OSError:
                pass
        for old in existing_png - new_png:
            try:
                os.remove(old)
            except OSError:
                pass

    logger.info(f"Generated {len(new_json) - 1} cards  |  leaderboard → {lb_path}")
    return paths


# ---------------------------------------------------------------------------
# Card / profile builders that accept a pre-built summary row
# ---------------------------------------------------------------------------

def _generate_card_from_summary(
    player_name: str,
    summary: pd.DataFrame,
    season: int,
    output_dir: str,
) -> "str | None":
    """Generate PNG card from an already-built (and recalibrated) summary."""
    if summary.empty:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches  # noqa: F401 used below

    n_pitches = len(summary)
    cols = min(n_pitches, 4)
    rows = (n_pitches + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3.2 + 0.4, rows * 3.0 + 1.2),
                             facecolor=BG_DARK)

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    total_pitches = int(summary["n"].sum())
    fig.text(0.5, 0.98, player_name, ha="center", va="top",
             fontsize=16, fontweight="bold", color="white")
    fig.text(0.5, 0.94,
             f"{season} Season  |  {total_pitches:,} pitches  |  Stuff+ Model",
             ha="center", va="top", fontsize=9, color="#8b949e")

    for i, (_, row) in enumerate(summary.iterrows()):
        r, c = divmod(i, cols)
        _pitch_panel(axes[r][c], row)

    for j in range(n_pitches, rows * cols):
        r, c = divmod(j, cols)
        axes[r][c].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    safe  = player_name.replace(", ", "_").replace(" ", "_").replace("'", "")
    fpath = os.path.join(output_dir, f"{safe}_{season}.png")
    plt.savefig(fpath, dpi=130, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    return fpath


def _build_profile_from_summary(
    player_name: str,
    summary: pd.DataFrame,
    season: int,
    card_path: str = None,
) -> dict:
    """Build JSON profile from a recalibrated summary DataFrame."""
    # --- Top-level pitcher metadata ---
    p_throws = "R"
    if "p_throws" in summary.columns:
        _pt = summary["p_throws"].dropna()
        if len(_pt) > 0:
            p_throws = str(_pt.iloc[0])

    _aa_mean = summary["arm_angle"].mean() if "arm_angle" in summary.columns else float("nan")
    arm_angle_avg = round(float(_aa_mean), 1) if _aa_mean == _aa_mean else None

    team = ""
    if "team" in summary.columns and len(summary) > 0:
        _t = summary["team"].dropna()
        if len(_t) > 0:
            team = str(_t.iloc[0])

    pitcher_id = None
    if "pitcher_id" in summary.columns:
        _pid = summary["pitcher_id"].dropna()
        if len(_pid) > 0:
            try:
                pitcher_id = int(_pid.iloc[0])
            except (ValueError, TypeError):
                pass

    age = None
    if "age" in summary.columns:
        _age = summary["age"].dropna()
        if len(_age) > 0:
            try:
                age = int(_age.iloc[0])
            except (ValueError, TypeError):
                pass

    ip_str = "0.0"
    if "ip_str" in summary.columns:
        _ip = summary["ip_str"].dropna()
        if len(_ip) > 0:
            ip_str = str(_ip.iloc[0])

    era = None
    if "era" in summary.columns:
        _era = summary["era"].dropna()
        if len(_era) > 0:
            try:
                era = float(_era.iloc[0])
            except (ValueError, TypeError):
                pass

    pitches = []
    for _, row in summary.iterrows():
        sp = float(row["stuff_plus"])
        sp_rounded = round(sp, 1)
        pt_mean_for_type = float(row.get("pt_mean", 100.0))
        grade, color = stuff_grade(sp_rounded, pt_mean=pt_mean_for_type)

        # --- Outcome stats ---
        zone_pct      = _nan_to_none(row.get("zone_pct"))
        chase_pct     = _nan_to_none(row.get("chase_pct"))
        whiff_pct     = _nan_to_none(row.get("whiff_pct"))
        xwoba_contact = _nan_to_none(row.get("xwoba_contact"))
        usage_pct     = _nan_to_none(row.get("usage_pct")) or 0.0

        # --- Handedness usage ---
        usage_vs_r = _nan_to_none(row.get("usage_pct_vs_r")) or 0.0
        usage_vs_l = _nan_to_none(row.get("usage_pct_vs_l")) or 0.0
        n_vs_r = int(row.get("n_vs_r") or 0)
        n_vs_l = int(row.get("n_vs_l") or 0)

        # --- Location arrays (combined + per-handedness) ---
        def _safe_list(key):
            v = row.get(key)
            return v if isinstance(v, list) else []

        pitches.append({
            "pitch_type":        row["pitch_type"],
            "pitch_name":        PITCH_TYPES.get(row["pitch_type"], row["pitch_type"]),
            "stuff_plus":        sp_rounded,
            "grade":             grade,
            "color":             color,
            "n":                 int(row["n"]),
            "n_vs_r":            n_vs_r,
            "n_vs_l":            n_vs_l,
            "usage_pct":         round(float(usage_pct), 1),
            "usage_pct_vs_r":    round(float(usage_vs_r), 1),
            "usage_pct_vs_l":    round(float(usage_vs_l), 1),
            "velo":              round(float(row["velo"]), 1),
            "ivb":               round(float(row["ivb"]), 1),
            "hb":                round(float(row["hb"]), 1),
            "vaa":               round(float(row["vaa"]), 2),
            "haa":               round(float(row["haa"]), 2),
            "spin":              int(row["spin"]),
            "extension":         round(float(row["extension"]), 2),
            "arm_angle":         _nan_to_none(round(float(row["arm_angle"]), 1) if row.get("arm_angle") == row.get("arm_angle") else None),
            "release_height":    round(float(row.get("release_height", float("nan"))), 2),
            "release_side":      _nan_to_none(round(float(row.get("release_side")), 2) if row.get("release_side") == row.get("release_side") else None),
            "spin_axis_clock":   str(row.get("spin_axis_clock", "—")),
            "zone_pct":          round(zone_pct, 1) if zone_pct is not None else None,
            "chase_pct":         round(chase_pct, 1) if chase_pct is not None else None,
            "whiff_pct":         round(whiff_pct, 1) if whiff_pct is not None else None,
            "xwoba_contact":     round(xwoba_contact, 3) if xwoba_contact is not None else None,
            "locations_x":       _safe_list("locations_x"),
            "locations_z":       _safe_list("locations_z"),
            "locations_x_vs_r":  _safe_list("locations_x_vs_r"),
            "locations_z_vs_r":  _safe_list("locations_z_vs_r"),
            "locations_x_vs_l":  _safe_list("locations_x_vs_l"),
            "locations_z_vs_l":  _safe_list("locations_z_vs_l"),
        })

    return {
        "player_name":   player_name,
        "season":        season,
        "total_pitches": int(summary["n"].sum()),
        "p_throws":      p_throws,
        "arm_angle":     arm_angle_avg,
        "team":          team,
        "pitcher_id":    pitcher_id,
        "age":           age,
        "ip_str":        ip_str,
        "era":           era,
        "card_image":    card_path,
        "pitches":       pitches,
    }


def _build_leaderboard_from_calibrated(combined: pd.DataFrame, season: int) -> list:
    """
    Build leaderboard from the cross-pitcher recalibrated summary.
    Overall Stuff+ per pitcher = usage-weighted average across pitch types.
    """
    rows = []
    for name, grp in combined.groupby("player_name"):
        valid = grp[grp["stuff_plus"].notna()]
        if valid.empty:
            continue
        total_n = int(grp["n"].sum())
        # Usage-weighted average (clip before grading so extreme outliers don't distort)
        avg_sp = float((valid["stuff_plus"] * valid["n"]).sum() / valid["n"].sum())
        grade, color = stuff_grade(avg_sp)
        rows.append({
            "player_name":  name,
            "season":       str(season),
            "stuff_plus":   round(float(avg_sp), 1),
            "grade":        grade,
            "color":        color,
            "total_pitches": total_n,
        })
    rows.sort(key=lambda x: x["stuff_plus"] or 0, reverse=True)
    return rows

"""Assemble :class:`trade_value.TradeValueInputs` from live sources.

Each source is loaded independently inside ``_attempt``; a failure records the error in
``freshness`` and leaves that input ``None`` so scoring degrades transparently instead
of raising. Sources:

- nflverse player_stats     weekly PPR points, target share, air yards (refreshed nightly)
- nflverse snap_counts      offensive snap % (mapped pfr -> gsis via ff_playerids/players)
- nflverse pbp              weekly red-zone touches (local parquet, TTL-refreshed)
- nflverse participation    route participation (usually lags; optional)
- nflverse ff_opportunity   expected PPR points
- nflverse depth_charts     latest team depth chart + snapshot ~3 weeks earlier
- nflverse injuries         weekly report statuses (missed-time history)
- nflverse ff_rankings      FantasyPros weekly PPR positional ECR (market)
- ESPN game summaries       current-week designations, incl. official inactives on game day
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import polars as pl

from . import metrics
from .data import DataStore
from .feeds import EspnFeed
from .league import normalize_name
from .trade_value import PPR_SCORING, SKILL_POSITIONS, TradeValueInputs

log = logging.getLogger("nfl_metrics.trade_value")

DEPTH_PREV_DAYS = 21


def _attempt(freshness: dict[str, Any], name: str, fn: Callable[[], pl.DataFrame | None]) -> pl.DataFrame | None:
    try:
        out = fn()
    except Exception as exc:
        log.warning("trade_value source %s unavailable: %s", name, exc)
        freshness[name] = {"available": False, "error": str(exc)[:200]}
        return None
    if out is None or out.is_empty():
        freshness[name] = {"available": False, "error": "no rows"}
        return None
    freshness.setdefault(name, {})["available"] = True
    return out


def build_inputs(store: DataStore, espn: EspnFeed | None, season: int, through_week: int | None, include_pbp: bool = True) -> TradeValueInputs:
    """``through_week=None`` uses the latest week with published stats."""
    fresh: dict[str, Any] = {}
    current_season = store.current_season()
    current_week = store.current_week()
    is_current = season == current_season

    stats = store.player_stats(season)
    if "season_type" in stats.columns:
        stats = stats.filter(pl.col("season_type") == metrics.REG)
    weekly_points = store.derived("weekly_points", season, {"scoring": PPR_SCORING}, lambda: metrics.weekly_fantasy_points(store.player_stats(season), PPR_SCORING))
    max_week = int(weekly_points["week"].max() or 0) if not weekly_points.is_empty() else 0
    fresh["player_stats"] = {"available": not weekly_points.is_empty(), "latest_week": max_week}
    if through_week is None:
        through_week = max_week or 1
    elif max_week and max_week < through_week:
        fresh["player_stats"]["note"] = f"stats published through week {max_week}; through_week {through_week} requested"
        through_week = max_week
    if is_current and max_week and max_week >= current_week:
        fresh["player_stats"]["note"] = f"week {max_week} may still be in progress; its stats are partial until the last game finishes"

    usage_cols = [c for c in ("player_id", "week", "team", "target_share", "air_yards_share", "receiving_air_yards", "targets", "carries") if c in stats.columns]
    weekly_usage = stats.select(usage_cols)

    prior = _attempt(
        fresh,
        "prior_season",
        lambda: store.derived("weekly_points", season - 1, {"scoring": PPR_SCORING}, lambda: metrics.weekly_fantasy_points(store.player_stats(season - 1), PPR_SCORING)),
    )

    id_map = _attempt(fresh, "ff_playerids", lambda: store.ff_playerids().select(["gsis_id", "pfr_id", "fantasypros_id", "name", "position"]).filter(pl.col("gsis_id").is_not_null()))
    players = _attempt(fresh, "players", lambda: store.players().select(["gsis_id", "pfr_id", "display_name", "position"]))

    def load_snaps() -> pl.DataFrame:
        sn = store.snap_counts(season)
        sn = sn.filter((pl.col("game_type") == metrics.REG) & pl.col("position").is_in(list(SKILL_POSITIONS)))
        pfr_map = _pfr_to_gsis(id_map, players)
        sn = sn.join(pfr_map, left_on="pfr_player_id", right_on="pfr_id", how="left")
        # fall back to name+team when the pfr id is unmapped
        names = weekly_points.select("player_id", pl.col("player").map_elements(normalize_name, return_dtype=pl.Utf8).alias("_n"), "team").unique(["_n", "team"])
        sn = sn.with_columns(pl.col("player").map_elements(normalize_name, return_dtype=pl.Utf8).alias("_n")).join(names, on=["_n", "team"], how="left", suffix="_byname")
        sn = sn.with_columns(pl.coalesce(["gsis_id", "player_id"]).alias("player_id")).filter(pl.col("player_id").is_not_null())
        fresh["snap_counts"] = {"latest_week": int(sn["week"].max()) if not sn.is_empty() else None}
        return sn.select("player_id", "week", pl.col("offense_pct").alias("snap_pct"))

    snaps = _attempt(fresh, "snap_counts", load_snaps)

    red_zone = None
    if include_pbp:
        red_zone = _attempt(fresh, "red_zone", lambda: store.derived("red_zone_weekly", season, {}, lambda: metrics.red_zone_weekly(store.scan_pbp(season))))
    else:
        fresh["red_zone"] = {"available": False, "error": "skipped (include_pbp=False)"}

    routes = _attempt(
        fresh,
        "routes",
        lambda: store.derived(
            "routes", season, {"week": None, "weeks": None, "min": 1}, lambda: metrics.route_participation(store.participation(season), store.scan_pbp(season), store.players(), min_routes=1)
        ),
    ) if include_pbp else None
    if not include_pbp:
        fresh["routes"] = {"available": False, "error": "skipped (include_pbp=False)"}

    def load_depth() -> tuple[pl.DataFrame, pl.DataFrame | None]:
        dc = store.depth_charts(season).filter(pl.col("pos_abb").is_in(list(SKILL_POSITIONS)) & pl.col("gsis_id").is_not_null())
        dc = dc.with_columns(pl.col("dt").str.to_datetime(strict=False, time_zone="UTC").alias("_dt")) if dc.schema["dt"] == pl.Utf8 else dc.with_columns(pl.col("dt").alias("_dt"))
        latest_per_team = dc.group_by("team").agg(pl.col("_dt").max().alias("_latest"))
        cur = dc.join(latest_per_team, on="team").filter(pl.col("_dt") == pl.col("_latest"))
        overall_latest = cur["_dt"].max()
        fresh["depth_chart"] = {"as_of": str(overall_latest)}
        cutoff = overall_latest - timedelta(days=DEPTH_PREV_DAYS)
        older = dc.filter(pl.col("_dt") <= cutoff)
        prev = None
        if not older.is_empty():
            prev_latest = older.group_by("team").agg(pl.col("_dt").max().alias("_prev"))
            prev = older.join(prev_latest, on="team").filter(pl.col("_dt") == pl.col("_prev"))
            fresh["depth_chart"]["previous_as_of"] = str(prev["_dt"].max())
            prev = _depth_select(prev)
        return _depth_select(cur), prev

    depth_chart = depth_prev = None
    try:
        depth_chart, depth_prev = load_depth()
        fresh["depth_chart"]["available"] = True
    except Exception as exc:
        log.warning("trade_value source depth_chart unavailable: %s", exc)
        fresh["depth_chart"] = {"available": False, "error": str(exc)[:200]}

    injuries = _attempt(
        fresh,
        "injuries",
        lambda: store.injuries(season).select(pl.col("gsis_id").alias("player_id"), "week", "report_status", "practice_status", "full_name", "team").filter(pl.col("player_id").is_not_null()),
    )

    expected = _attempt(
        fresh,
        "expected_points",
        lambda: store.ff_opportunity(season)
        .filter(pl.col("player_id").is_not_null())
        .select("player_id", pl.col("week").cast(pl.Int32), pl.col("total_fantasy_points_exp").alias("exp_points")),
    )

    def load_market() -> pl.DataFrame:
        rk = store.ff_rankings("week").filter(pl.col("pos").is_in(list(SKILL_POSITIONS)))
        fresh["market"] = {"scrape_date": str(rk["scrape_date"].max()), "note": "FantasyPros weekly PPR positional ECR via nflverse"}
        rk = rk.with_columns(pl.col("fantasypros_id").cast(pl.Int64, strict=False))
        if id_map is not None:
            fp = id_map.select(pl.col("fantasypros_id").cast(pl.Int64, strict=False), "gsis_id").filter(pl.col("fantasypros_id").is_not_null()).unique("fantasypros_id")
            rk = rk.join(fp, on="fantasypros_id", how="left")
        else:
            rk = rk.with_columns(pl.lit(None, dtype=pl.Utf8).alias("gsis_id"))
        names = weekly_points.select("player_id", pl.col("player").map_elements(normalize_name, return_dtype=pl.Utf8).alias("_n"), "position").unique(["_n", "position"])
        rk = rk.with_columns(pl.col("player_name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("_n")).join(names, left_on=["_n", "pos"], right_on=["_n", "position"], how="left")
        rk = rk.with_columns(pl.coalesce(["gsis_id", "player_id"]).alias("player_id")).filter(pl.col("player_id").is_not_null())
        return rk.select("player_id", pl.col("rank").cast(pl.Float64).alias("ecr_pos_rank"), pl.col("ecr").cast(pl.Float64)).unique("player_id")

    market = _attempt(fresh, "market", load_market)

    designations: dict[str, dict[str, Any]] = {}
    # designations describe the next game to be played: the in-progress week, else the one after through_week
    upcoming = max(current_week, through_week + 1) if is_current else through_week + 1
    if is_current and espn is not None:
        reports = espn.injuries(season, upcoming)
        if reports:
            by_name = {(normalize_name(r["player"]), r["team"]): r["player_id"] for r in weekly_points.select("player", "team", "player_id").unique().to_dicts() if r["player"]}
            if depth_chart is not None:
                for r in depth_chart.to_dicts():
                    by_name.setdefault((normalize_name(r["player"]), r["team"]), r["player_id"])
            for rep in reports:
                pid = by_name.get((normalize_name(rep.player), rep.team))
                if pid:
                    designations[pid] = {"status": rep.status, "detail": rep.detail, "source": "espn", "week": upcoming}
            fresh["espn_designations"] = {"available": True, "week": upcoming, "fetched_at": espn.last_fetch_iso(), "games": espn.game_states(season, upcoming)}
        else:
            fresh["espn_designations"] = {"available": False, "week": upcoming, "error": "ESPN returned no injury data"}
    else:
        fresh["espn_designations"] = {"available": False, "error": "only fetched for the current season"}
    if injuries is not None:
        for r in injuries.filter(pl.col("week") == upcoming).to_dicts():
            if r.get("report_status") and r["player_id"] not in designations:
                designations[r["player_id"]] = {"status": r["report_status"], "detail": r.get("practice_status"), "source": "nflverse", "week": upcoming}

    team_games: dict[str, int] = {}
    try:
        sched = store.schedules(season).filter((pl.col("game_type") == metrics.REG) & (pl.col("week") <= through_week))
        for r in sched.select("home_team", "away_team").to_dicts():
            team_games[r["home_team"]] = team_games.get(r["home_team"], 0) + 1
            team_games[r["away_team"]] = team_games.get(r["away_team"], 0) + 1
        fresh["schedules"] = {"available": True}
    except Exception as exc:
        fresh["schedules"] = {"available": False, "error": str(exc)[:200]}

    fresh["as_of"] = {"season": season, "through_week": through_week, "is_current_season": is_current, "scoring": "standard_ppr"}
    return TradeValueInputs(
        season=season,
        through_week=through_week,
        weekly_points=weekly_points,
        weekly_usage=weekly_usage,
        prior_points=prior,
        snaps=snaps,
        red_zone=red_zone,
        routes=routes,
        depth_chart=depth_chart,
        depth_chart_prev=depth_prev,
        injuries=injuries.select("player_id", "week", "report_status") if injuries is not None else None,
        expected_points=expected,
        market=market,
        current_designations=designations,
        team_games=team_games,
        freshness=fresh,
    )


def _depth_select(dc: pl.DataFrame) -> pl.DataFrame:
    return (
        dc.sort("pos_rank")
        .unique(["team", "gsis_id"], keep="first")
        .select(pl.col("gsis_id").alias("player_id"), pl.col("player_name").alias("player"), "team", pl.col("pos_abb").alias("position"), pl.col("pos_rank").cast(pl.Int32))
        .unique("player_id", keep="first")
    )


def _pfr_to_gsis(id_map: pl.DataFrame | None, players: pl.DataFrame | None) -> pl.DataFrame:
    parts = []
    if id_map is not None:
        parts.append(id_map.select("pfr_id", "gsis_id"))
    if players is not None:
        parts.append(players.select("pfr_id", "gsis_id"))
    if not parts:
        return pl.DataFrame(schema={"pfr_id": pl.Utf8, "gsis_id": pl.Utf8})
    return pl.concat(parts).filter(pl.col("pfr_id").is_not_null() & pl.col("gsis_id").is_not_null()).unique("pfr_id", keep="first")

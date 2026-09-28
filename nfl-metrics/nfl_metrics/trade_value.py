"""Rest-of-season trade value (0-100) for skill players on a fixed standard-PPR baseline.

Pure functions over Polars frames so they are unit-testable without network access.
``server.py`` assembles the inputs (nflverse player stats, snap counts, participation,
ff_opportunity expected points, depth charts, injury reports, FantasyPros ECR via
nflverse, ESPN live designations) into :class:`TradeValueInputs`; everything here is
arithmetic.

Scoring is intentionally *not* league-configurable: every number is standard PPR
(``PPR_SCORING``) so values are comparable across leagues and agents.

Pipeline per player
-------------------
1. production      season PPG, last-3 PPG, prior-season PPG, ff_opportunity expected PPG
                   -> blended ``projected_ppg`` (weights shift toward the prior-season
                   baseline early in the season)
2. usage trend     snap %, target share, air-yards share, red-zone touch share, route
                   participation: season vs last-3; a relative decline shrinks value
                   more than a rise inflates it
3. depth chart     real-life ``pos_rank`` on the team's latest depth chart; demotion vs an
                   earlier snapshot is penalised on top of the rank multiplier
4. injury          current designation (ESPN/nflverse), games missed this season and
                   last season, and injury-report frequency -> availability multiplier
5. value           adjusted PPG -> value over positional replacement -> 0-100 model score
6. market          FantasyPros weekly PPR positional ECR -> market score; final value is a
                   0.75/0.25 model/market blend with the gap surfaced as a flag

Every missing source is reported in ``data_coverage`` and lowers ``confidence``; it
never silently defaults to a favourable value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from . import gm

PPR_SCORING: dict[str, float] = {
    "pass_yd": 0.04,
    "pass_td": 4,
    "pass_int": -2,
    "rush_yd": 0.1,
    "rush_td": 6,
    "rec": 1.0,
    "rec_yd": 0.1,
    "rec_td": 6,
    "fum_lost": -2,
    "two_pt": 2,
}

SKILL_POSITIONS = gm.SKILL_POSITIONS
RECENT_GAMES = 3

# depth-chart rank -> role multiplier, by position
DEPTH_MULTIPLIERS: dict[str, dict[int, float]] = {
    "QB": {1: 1.0, 2: 0.35},
    "RB": {1: 1.0, 2: 0.82, 3: 0.6},
    "WR": {1: 1.0, 2: 0.95, 3: 0.82, 4: 0.6},
    "TE": {1: 1.0, 2: 0.6},
}
DEPTH_FLOOR = 0.45
DEMOTION_MULTIPLIER = 0.9

STATUS_MULTIPLIERS: dict[str, float] = {
    "out": 0.6,
    "injured reserve": 0.55,
    "ir": 0.55,
    "pup": 0.55,
    "suspended": 0.6,
    "inactive": 0.6,
    "doubtful": 0.8,
    "questionable": 0.93,
}

# ECR positional rank at which market score has decayed to ~37% (1/e)
MARKET_DECAY: dict[str, float] = {"QB": 8.0, "RB": 14.0, "WR": 18.0, "TE": 6.0}
MARKET_WEIGHT = 0.25
MARKET_GAP_FLAG = 25.0

USAGE_METRICS = ("snap_pct", "target_share", "air_yards_share", "rz_touch_share")
DECLINE_FLAG = 0.15  # relative drop that raises a *_declining flag
CONSOLIDATION_WEIGHTS = (1.0, 0.85, 0.7, 0.55, 0.45, 0.35)
FAIR_TRADE_BAND = 0.06


@dataclass
class TradeValueInputs:
    """Normalised frames. Every frame except ``weekly_points`` is optional.

    weekly_points   player_id, player, position, team, week, fantasy_points   (standard PPR)
    weekly_usage    player_id, week, target_share, air_yards_share, receiving_air_yards,
                    targets, carries   (from nflverse player_stats)
    prior_points    weekly_points for the previous season
    snaps           player_id, week, snap_pct   (offense_pct in 0-1)
    red_zone        player_id, week, rz_touches, team_rz_plays
    routes          player_id, route_participation, targets_per_route_run (season aggregate)
    depth_chart     player_id, team, position, pos_rank   (latest snapshot)
    depth_chart_prev  same columns from an earlier snapshot, for demotion detection
    injuries        player_id, week, report_status   (nflverse weekly reports)
    expected_points player_id, week, exp_points   (ff_opportunity total_fantasy_points_exp, PPR)
    market          player_id, ecr_pos_rank, ecr   (FantasyPros weekly PPR positional ECR)
    current_designations  player_id -> {"status", "detail", "source"} for the upcoming week
    team_games      team -> number of games played through ``through_week``
    freshness       source -> as-of / fetched metadata passed straight through to callers
    """

    season: int
    through_week: int
    weekly_points: pl.DataFrame
    weekly_usage: pl.DataFrame | None = None
    prior_points: pl.DataFrame | None = None
    snaps: pl.DataFrame | None = None
    red_zone: pl.DataFrame | None = None
    routes: pl.DataFrame | None = None
    depth_chart: pl.DataFrame | None = None
    depth_chart_prev: pl.DataFrame | None = None
    injuries: pl.DataFrame | None = None
    expected_points: pl.DataFrame | None = None
    market: pl.DataFrame | None = None
    current_designations: dict[str, dict[str, Any]] = field(default_factory=dict)
    team_games: dict[str, int] = field(default_factory=dict)
    freshness: dict[str, Any] = field(default_factory=dict)

    def coverage(self) -> dict[str, bool]:
        return {
            "weekly_points": not self.weekly_points.is_empty(),
            "usage": _present(self.weekly_usage),
            "prior_season": _present(self.prior_points),
            "snaps": _present(self.snaps),
            "red_zone": _present(self.red_zone),
            "routes": _present(self.routes),
            "depth_chart": _present(self.depth_chart),
            "injuries": _present(self.injuries),
            "expected_points": _present(self.expected_points),
            "market": _present(self.market),
            "current_designations": bool(self.current_designations),
        }


def _present(df: pl.DataFrame | None) -> bool:
    return df is not None and not df.is_empty()


def _season_recent(df: pl.DataFrame | None, col: str, through_week: int, out: str) -> pl.DataFrame | None:
    """Per player: season mean and last-``RECENT_GAMES``-games mean of ``col``."""
    if not _present(df) or col not in df.columns:
        return None
    lf = df.lazy().filter(pl.col("week") <= through_week).filter(pl.col(col).is_not_null()).sort("week")
    return (
        lf.group_by("player_id")
        .agg(
            pl.col(col).mean().alias(f"{out}_season"),
            pl.col(col).tail(RECENT_GAMES).mean().alias(f"{out}_recent"),
            pl.col(col).tail(RECENT_GAMES).first().alias(f"{out}_recent_first"),
            pl.col(col).last().alias(f"{out}_last"),
        )
        .collect()
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _r(v: float | None, n: int = 3) -> float | None:
    return None if v is None else round(v, n)


# --------------------------------------------------------------------- players
def player_trade_values(
    inputs: TradeValueInputs,
    league_size: int = 12,
    starters_per_team: dict[str, float] | None = None,
) -> pl.DataFrame:
    """One row per skill player with every component plus ``trade_value`` (0-100).

    Column groups: identity (player_id, player, position, team), production
    (games, season_ppg, recent_ppg, prior_ppg, expected_ppg, projected_ppg, adjusted_ppg,
    vor), usage (``<metric>_season`` / ``<metric>_recent`` / ``<metric>_delta``),
    role (pos_rank, prev_pos_rank), injury (status, games_missed, prior_games_missed,
    injury_reports, injury_risk), multipliers (usage_multiplier, depth_multiplier,
    injury_multiplier), scores (model_score, market_score, trade_value, confidence),
    flags (list[str]).
    """
    tw = inputs.through_week
    pts = inputs.weekly_points.filter(pl.col("week") <= tw).filter(pl.col("position").is_in(list(SKILL_POSITIONS)))
    if pts.is_empty():
        return _empty_values()
    base = (
        pts.lazy()
        .sort("week")
        .group_by("player_id")
        .agg(
            pl.col("player").last(),
            pl.col("position").last(),
            pl.col("team").last(),
            pl.len().alias("games"),
            pl.col("week").min().alias("first_week"),
            pl.col("fantasy_points").mean().alias("season_ppg"),
            pl.col("fantasy_points").tail(RECENT_GAMES).mean().alias("recent_ppg"),
            pl.col("fantasy_points").std().fill_null(0.0).alias("ppg_std"),
        )
        .collect()
    )

    frames: list[pl.DataFrame | None] = []
    if _present(inputs.prior_points):
        frames.append(
            inputs.prior_points.lazy()
            .filter(pl.col("position").is_in(list(SKILL_POSITIONS)))
            .group_by("player_id")
            .agg(pl.col("fantasy_points").mean().alias("prior_ppg"), pl.len().alias("prior_games"))
            .collect()
        )
    frames.append(_season_recent(inputs.expected_points, "exp_points", tw, "expected_ppg"))
    frames.append(_season_recent(inputs.snaps, "snap_pct", tw, "snap_pct"))
    frames.append(_season_recent(inputs.weekly_usage, "target_share", tw, "target_share"))
    frames.append(_season_recent(inputs.weekly_usage, "air_yards_share", tw, "air_yards_share"))
    frames.append(_season_recent(inputs.weekly_usage, "receiving_air_yards", tw, "air_yards"))
    frames.append(_season_recent(inputs.weekly_usage, "targets", tw, "targets"))
    frames.append(_season_recent(inputs.weekly_usage, "carries", tw, "carries"))
    if _present(inputs.red_zone):
        rz = inputs.red_zone.with_columns(
            pl.when(pl.col("team_rz_plays") > 0).then(pl.col("rz_touches") / pl.col("team_rz_plays")).otherwise(None).alias("rz_touch_share")
        )
        frames.append(_season_recent(rz, "rz_touch_share", tw, "rz_touch_share"))
        frames.append(_season_recent(rz, "rz_touches", tw, "rz_touches"))
    if _present(inputs.routes):
        frames.append(inputs.routes.select(["player_id", "route_participation", "targets_per_route_run"]).unique("player_id"))
    if _present(inputs.depth_chart):
        frames.append(inputs.depth_chart.select(["player_id", "pos_rank"]).unique("player_id"))
    if _present(inputs.depth_chart_prev):
        frames.append(inputs.depth_chart_prev.select(["player_id", pl.col("pos_rank").alias("prev_pos_rank")]).unique("player_id"))
    if _present(inputs.injuries):
        frames.append(
            inputs.injuries.lazy()
            .filter(pl.col("week") <= tw)
            .filter(pl.col("report_status").is_not_null() & (pl.col("report_status") != ""))
            .group_by("player_id")
            .agg(pl.len().alias("injury_reports"), (pl.col("report_status").str.to_lowercase() == "out").sum().alias("out_reports"))
            .collect()
        )
    if _present(inputs.market):
        frames.append(inputs.market.select(["player_id", "ecr_pos_rank", "ecr"]).unique("player_id"))

    df = base
    for fr in frames:
        if fr is not None and not fr.is_empty():
            df = df.join(fr, on="player_id", how="left")

    rows = [_score_row(r, inputs) for r in df.to_dicts()]
    scored = pl.DataFrame(rows, infer_schema_length=None) if rows else _empty_values()
    if scored.is_empty():
        return scored
    scored = scored.with_columns(
        [pl.col(c).cast(pl.Float64) for c in ("market_score", "ecr_pos_rank", "ecr", "recent_ppg", "prior_ppg", "expected_ppg") if c in scored.columns]
        + [pl.col(c).cast(pl.Int64) for c in ("pos_rank", "prev_pos_rank", "prior_games", "prior_games_missed") if c in scored.columns]
        + [pl.col(c).cast(pl.Utf8) for c in ("status", "status_detail", "status_source", "team") if c in scored.columns]
    )

    proj = scored.select(pl.col("player_id"), pl.col("position"), pl.col("adjusted_ppg").alias("projection"))
    levels = gm.replacement_levels(proj, league_size=league_size, starters_per_team=starters_per_team)
    level_expr = pl.col("position").replace_strict(levels, default=0.0, return_dtype=pl.Float64)
    scored = scored.with_columns(level_expr.alias("replacement_ppg"), (pl.col("adjusted_ppg") - level_expr).alias("vor"))

    positive = scored.filter(pl.col("vor") > 0).sort("vor", descending=True)
    ref = float(positive["vor"][min(2, positive.height - 1)]) if not positive.is_empty() else 1.0
    ref = max(ref, 1e-6)

    def model_score(vor: float, adj: float, level: float) -> float:
        if vor > 0:
            return _clamp(100.0 * (vor / ref) ** 0.75, 0.0, 100.0)
        return 15.0 * _clamp(adj / level, 0.0, 1.0) if level > 0 else 0.0

    scored = scored.with_columns(
        pl.struct(["vor", "adjusted_ppg", "replacement_ppg"])
        .map_elements(lambda s: model_score(s["vor"], s["adjusted_ppg"], s["replacement_ppg"]), return_dtype=pl.Float64)
        .alias("model_score")
    )
    scored = scored.with_columns(
        pl.when(pl.col("market_score").is_not_null())
        .then((1 - MARKET_WEIGHT) * pl.col("model_score") + MARKET_WEIGHT * pl.col("market_score"))
        .otherwise(pl.col("model_score"))
        .alias("trade_value"),
        (pl.col("model_score") - pl.col("market_score")).alias("market_gap"),
    )
    scored = scored.with_columns(
        pl.struct(["flags", "market_gap"])
        .map_elements(lambda s: _market_flags(s["flags"], s["market_gap"]), return_dtype=pl.List(pl.Utf8))
        .alias("flags")
    )
    return scored.with_columns(
        pl.col("trade_value").round(1),
        pl.col("model_score").round(1),
        pl.col("market_score").round(1),
        pl.col("market_gap").round(1),
        pl.col("vor").round(2),
        pl.col("replacement_ppg").round(2),
    ).sort("trade_value", descending=True)


def _market_flags(flags: list[str], gap: float | None) -> list[str]:
    flags = list(flags or [])
    if gap is not None:
        if gap >= MARKET_GAP_FLAG:
            flags.append("market_undervalues")
        elif gap <= -MARKET_GAP_FLAG:
            flags.append("market_overvalues")
    return flags


def _score_row(r: dict[str, Any], inputs: TradeValueInputs) -> dict[str, Any]:
    pos = r["position"]
    games = int(r["games"])
    flags: list[str] = []
    tw = inputs.through_week

    # ---- production blend
    season_ppg = _f(r.get("season_ppg")) or 0.0
    recent_ppg = _f(r.get("recent_ppg"))
    prior_ppg = _f(r.get("prior_ppg"))
    prior_games = int(r.get("prior_games") or 0)
    if prior_ppg is not None and prior_games < 4:
        prior_ppg = None  # too small to be a baseline
    expected_ppg = _f(r.get("expected_ppg_season"))
    prior_w = 0.35 if games <= 4 else 0.15
    parts = {"recent": (recent_ppg, 0.40), "season": (season_ppg, 0.25), "expected": (expected_ppg, 0.20), "prior": (prior_ppg, prior_w)}
    avail = {k: v for k, v in parts.items() if v[0] is not None}
    wsum = sum(w for _, w in avail.values())
    projected = sum(v * w for v, w in avail.values()) / wsum if wsum else 0.0
    if games <= 2:
        flags.append("small_sample")

    # ---- usage trend (season vs last 3): asymmetric, declines bite harder
    declines: list[float] = []
    rises: list[float] = []
    usage: dict[str, Any] = {}
    for m in USAGE_METRICS:
        s, rec = _f(r.get(f"{m}_season")), _f(r.get(f"{m}_recent"))
        usage[f"{m}_season"] = _r(s)
        usage[f"{m}_recent"] = _r(rec)
        usage[f"{m}_delta"] = _r(rec - s) if s is not None and rec is not None else None
        if s is None or rec is None or games < RECENT_GAMES + 1:
            continue  # with <=3 games recent == season; no trend information
        if s <= 0:
            continue
        rel = (rec - s) / s
        if rel < 0:
            declines.append(-rel)
            if -rel >= DECLINE_FLAG:
                flags.append(f"{m}_declining")
        else:
            rises.append(rel)
    decline = sum(declines) / len(declines) if declines else 0.0
    rise = sum(rises) / len(rises) if rises else 0.0
    usage_mult = _clamp(1.0 - 0.6 * decline + 0.25 * rise, 0.6, 1.15)
    snap_recent, snap_season = _f(r.get("snap_pct_recent")), _f(r.get("snap_pct_season"))
    if snap_recent is not None and snap_season and snap_recent < 0.5 * snap_season and games > RECENT_GAMES:
        flags.append("role_reduced")
        usage_mult = min(usage_mult, 0.7)
    usage_trend = "declining" if decline >= DECLINE_FLAG and decline > rise else ("rising" if rise >= DECLINE_FLAG else "stable")
    usage.update(
        {
            "route_participation": _r(_f(r.get("route_participation"))),
            "targets_per_route_run": _r(_f(r.get("targets_per_route_run"))),
            "air_yards_per_game": _r(_f(r.get("air_yards_season")), 1),
            "air_yards_per_game_recent": _r(_f(r.get("air_yards_recent")), 1),
            "targets_per_game": _r(_f(r.get("targets_season")), 2),
            "targets_per_game_recent": _r(_f(r.get("targets_recent")), 2),
            "carries_per_game": _r(_f(r.get("carries_season")), 2),
            "carries_per_game_recent": _r(_f(r.get("carries_recent")), 2),
            "rz_touches_per_game": _r(_f(r.get("rz_touches_season")), 2),
            "rz_touches_per_game_recent": _r(_f(r.get("rz_touches_recent")), 2),
        }
    )

    # ---- depth chart role
    pos_rank = r.get("pos_rank")
    prev_rank = r.get("prev_pos_rank")
    pos_rank = int(pos_rank) if pos_rank is not None else None
    prev_rank = int(prev_rank) if prev_rank is not None else None
    if pos_rank is None:
        depth_mult = 1.0
        role = "unknown"
    else:
        table = DEPTH_MULTIPLIERS.get(pos, {1: 1.0})
        depth_mult = table.get(pos_rank, DEPTH_FLOOR if pos_rank > max(table) else 1.0)
        role = f"{pos}{pos_rank}"
        if pos_rank >= 3 and pos != "WR":
            flags.append("depth_chart_backup")
        if prev_rank is not None and pos_rank > prev_rank:
            flags.append("depth_chart_demoted")
            depth_mult *= DEMOTION_MULTIPLIER
        elif prev_rank is not None and pos_rank < prev_rank:
            flags.append("depth_chart_promoted")

    # ---- injury / availability
    desig = inputs.current_designations.get(r["player_id"]) or {}
    status = (desig.get("status") or "").strip().lower() or None
    status_mult = STATUS_MULTIPLIERS.get(status, 1.0) if status else 1.0
    if status and status in STATUS_MULTIPLIERS:
        flags.append(f"designation_{status.replace(' ', '_')}")
    team_games = inputs.team_games.get(r.get("team") or "", tw)
    first_week = int(r.get("first_week") or 1)
    games_missed = max(0, team_games - games)
    missed_rate = games_missed / team_games if team_games else 0.0
    prior_missed = max(0, 17 - prior_games) if prior_games else None
    prior_rate = (prior_missed / 17) if prior_missed is not None else 0.0
    reports = int(r.get("injury_reports") or 0)
    report_rate = min(1.0, reports / tw) if tw else 0.0
    injury_risk = _clamp(0.5 * missed_rate + 0.3 * prior_rate + 0.2 * report_rate, 0.0, 1.0)
    if games_missed >= 2:
        flags.append("missed_time_this_season")
    injury_mult = status_mult * (1.0 - 0.3 * injury_risk)

    adjusted = projected * usage_mult * depth_mult * injury_mult

    # ---- market
    ecr_rank = _f(r.get("ecr_pos_rank"))
    market_score = 100.0 * math.exp(-(ecr_rank - 1) / MARKET_DECAY.get(pos, 12.0)) if ecr_rank is not None else None

    coverage = inputs.coverage()
    have = [
        coverage["usage"],
        coverage["snaps"],
        coverage["red_zone"],
        coverage["routes"],
        coverage["depth_chart"] and pos_rank is not None,
        coverage["injuries"],
        coverage["expected_points"] and expected_ppg is not None,
        coverage["market"] and ecr_rank is not None,
        coverage["prior_season"] and prior_ppg is not None,
    ]
    coverage_ratio = sum(1 for h in have if h) / len(have)
    confidence = round(coverage_ratio * (0.5 + 0.5 * min(games, 5) / 5), 2)

    return {
        "player_id": r["player_id"],
        "player": r["player"],
        "position": pos,
        "team": r.get("team"),
        "games": games,
        "first_week": first_week,
        "season_ppg": round(season_ppg, 2),
        "recent_ppg": _r(recent_ppg, 2),
        "prior_ppg": _r(prior_ppg, 2),
        "prior_games": prior_games or None,
        "expected_ppg": _r(expected_ppg, 2),
        "projected_ppg": round(projected, 2),
        "adjusted_ppg": round(adjusted, 2),
        "ppg_std": round(_f(r.get("ppg_std")) or 0.0, 2),
        **usage,
        "usage_trend": usage_trend,
        "usage_multiplier": round(usage_mult, 3),
        "pos_rank": pos_rank,
        "prev_pos_rank": prev_rank,
        "role": role,
        "depth_multiplier": round(depth_mult, 3),
        "status": status,
        "status_detail": desig.get("detail"),
        "status_source": desig.get("source"),
        "games_missed": games_missed,
        "prior_games_missed": prior_missed,
        "injury_reports": reports,
        "injury_risk": round(injury_risk, 3),
        "injury_multiplier": round(injury_mult, 3),
        "ecr_pos_rank": ecr_rank,
        "ecr": _r(_f(r.get("ecr")), 1),
        "market_score": market_score,
        "confidence": confidence,
        "flags": flags,
    }


def _empty_values() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "player_id": pl.Utf8,
            "player": pl.Utf8,
            "position": pl.Utf8,
            "team": pl.Utf8,
            "trade_value": pl.Float64,
            "model_score": pl.Float64,
            "market_score": pl.Float64,
            "adjusted_ppg": pl.Float64,
            "vor": pl.Float64,
            "flags": pl.List(pl.Utf8),
        }
    )


def explain(row: dict[str, Any]) -> dict[str, Any]:
    """Regroup a flat ``player_trade_values`` row into a readable dossier."""
    g = row.get

    def pick(*keys: str) -> dict[str, Any]:
        return {k: g(k) for k in keys}

    return {
        "player_id": g("player_id"),
        "player": g("player"),
        "position": g("position"),
        "team": g("team"),
        "trade_value": g("trade_value"),
        "model_score": g("model_score"),
        "market_score": g("market_score"),
        "market_gap": g("market_gap"),
        "confidence": g("confidence"),
        "flags": g("flags") or [],
        "production": pick("games", "season_ppg", "recent_ppg", "prior_ppg", "prior_games", "expected_ppg", "projected_ppg", "adjusted_ppg", "replacement_ppg", "vor", "ppg_std"),
        "usage": {
            k: g(k)
            for k in row
            if k.startswith(USAGE_METRICS)
            or k
            in (
                "route_participation",
                "targets_per_route_run",
                "air_yards_per_game",
                "air_yards_per_game_recent",
                "targets_per_game",
                "targets_per_game_recent",
                "carries_per_game",
                "carries_per_game_recent",
                "rz_touches_per_game",
                "rz_touches_per_game_recent",
                "usage_trend",
                "usage_multiplier",
            )
        },
        "role": pick("pos_rank", "prev_pos_rank", "role", "depth_multiplier"),
        "injury": pick("status", "status_detail", "status_source", "games_missed", "prior_games_missed", "injury_reports", "injury_risk", "injury_multiplier"),
        "market": pick("ecr_pos_rank", "ecr", "market_score"),
    }


# ---------------------------------------------------------------------- trades
def side_value(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Consolidation-weighted package value: the best asset counts fully, depth pieces
    progressively less, because a roster spot only holds one starter."""
    ordered = sorted(rows, key=lambda x: -(x.get("trade_value") or 0.0))
    total = 0.0
    for i, r in enumerate(ordered):
        w = CONSOLIDATION_WEIGHTS[min(i, len(CONSOLIDATION_WEIGHTS) - 1)]
        total += w * float(r.get("trade_value") or 0.0)
    return {
        "package_value": round(total, 1),
        "raw_sum": round(sum(float(r.get("trade_value") or 0.0) for r in rows), 1),
        "best_asset": ordered[0]["player"] if ordered else None,
        "best_asset_value": ordered[0].get("trade_value") if ordered else None,
        "adjusted_ppg_total": round(sum(float(r.get("adjusted_ppg") or 0.0) for r in rows), 2),
        "players": [explain(r) for r in ordered],
        "risk_flags": sorted({f for r in rows for f in (r.get("flags") or []) if f != "depth_chart_promoted"}),
    }


def compare_trade(
    values: pl.DataFrame,
    side_a_ids: list[str],
    side_b_ids: list[str],
    labels: tuple[str, str] = ("side_a", "side_b"),
    roster_a: set[str] | None = None,
    roster_b: set[str] | None = None,
    starters_per_team: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare the two packages of a trade. ``side_a_ids`` is what side A *gives*.

    If rosters are supplied, also reports each team's starting-lineup projection
    (adjusted PPG) before and after the swap so a 2-for-1 that upgrades a starter
    is judged on lineup impact, not just summed values.
    """
    by_id = {r["player_id"]: r for r in values.to_dicts()}
    a_rows = [by_id[i] for i in side_a_ids if i in by_id]
    b_rows = [by_id[i] for i in side_b_ids if i in by_id]
    missing = [i for i in side_a_ids + side_b_ids if i not in by_id]
    a, b = side_value(a_rows), side_value(b_rows)
    va, vb = a["package_value"], b["package_value"]
    top = max(va, vb, 1e-9)
    diff = (vb - va) / top  # positive => side A receives more than it gives
    if abs(diff) <= FAIR_TRADE_BAND:
        verdict, winner = "fair", None
    else:
        winner = labels[0] if diff > 0 else labels[1]
        verdict = f"favors_{winner}"
    out: dict[str, Any] = {
        labels[0]: {"gives": a, "receives_value": vb},
        labels[1]: {"gives": b, "receives_value": va},
        "value_gap": round(vb - va, 1),
        "value_gap_pct": round(100 * diff, 1),
        "verdict": verdict,
        "winner": winner,
        "unresolved_ids": missing,
        "scoring": "standard_ppr",
    }
    if roster_a is not None or roster_b is not None:
        starters = starters_per_team or gm.DEFAULT_STARTERS_PER_TEAM
        impact = {}
        for label, roster, gives, gets in ((labels[0], roster_a, side_a_ids, side_b_ids), (labels[1], roster_b, side_b_ids, side_a_ids)):
            if roster is None:
                continue
            before = lineup_projection(values, roster, starters)
            after = lineup_projection(values, (set(roster) - set(gives)) | set(gets), starters)
            not_owned = [i for i in gives if i not in roster]
            impact[label] = {
                "starting_lineup_ppg_before": before,
                "starting_lineup_ppg_after": after,
                "starting_lineup_delta": round(after - before, 2),
                "players_not_on_roster": [by_id.get(i, {}).get("player", i) for i in not_owned],
            }
        out["lineup_impact"] = impact
    return out


def lineup_projection(values: pl.DataFrame, roster_ids: set[str], starters: dict[str, float]) -> float:
    """Greedy best-lineup adjusted PPG for a roster (position slots then FLEX)."""
    mine = values.filter(pl.col("player_id").is_in(list(roster_ids))).sort("adjusted_ppg", descending=True)
    used: set[str] = set()
    total = 0.0
    flex_share = 0.0
    for pos in SKILL_POSITIONS:
        n = starters.get(pos, 0.0)
        whole = int(n)
        flex_share += n - whole
        pos_rows = mine.filter(pl.col("position") == pos).head(whole)
        for r in pos_rows.iter_rows(named=True):
            used.add(r["player_id"])
            total += float(r["adjusted_ppg"])
    flex_slots = int(round(flex_share))
    rest = mine.filter(~pl.col("player_id").is_in(list(used)) & pl.col("position").is_in(["RB", "WR", "TE"])).head(flex_slots)
    total += float(rest["adjusted_ppg"].sum()) if not rest.is_empty() else 0.0
    return round(total, 2)


def evaluate_proposals(
    values: pl.DataFrame,
    proposals: list[dict[str, Any]],
    rosters: dict[str, set[str]] | None = None,
    starters_per_team: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Score a batch of league trade proposals. Each proposal:
    ``{"team_a", "team_b", "a_gives": [ids], "b_gives": [ids], "label"?}``.

    With ``rosters`` (owner -> ids), ownership is verified against ground truth and
    lineup impact is computed for both teams.
    """
    results = []
    for i, p in enumerate(proposals):
        ta, tb = p.get("team_a") or "team_a", p.get("team_b") or "team_b"
        ra = rosters.get(ta) if rosters else None
        rb = rosters.get(tb) if rosters else None
        cmp = compare_trade(values, list(p.get("a_gives") or []), list(p.get("b_gives") or []), labels=(ta, tb), roster_a=ra, roster_b=rb, starters_per_team=starters_per_team)
        issues: list[str] = []
        if rosters:
            if ra is None:
                issues.append(f"unknown team {ta}")
            if rb is None:
                issues.append(f"unknown team {tb}")
            for label in (ta, tb):
                bad = (cmp.get("lineup_impact") or {}).get(label, {}).get("players_not_on_roster") or []
                if bad:
                    issues.append(f"{label} does not roster: {', '.join(bad)}")
        if cmp["unresolved_ids"]:
            issues.append(f"unresolved players: {cmp['unresolved_ids']}")
        impact = cmp.get("lineup_impact") or {}
        both_gain = bool(impact) and all(v["starting_lineup_delta"] >= 0 for v in impact.values())
        results.append(
            {
                "index": i,
                "label": p.get("label") or f"{ta} <-> {tb}",
                "verdict": cmp["verdict"],
                "winner": cmp["winner"],
                "value_gap": cmp["value_gap"],
                "value_gap_pct": cmp["value_gap_pct"],
                "both_lineups_improve": both_gain if impact else None,
                "issues": issues,
                "detail": cmp,
            }
        )
    results.sort(key=lambda x: (bool(x["issues"]), abs(float(x["value_gap_pct"]))))
    fair = [r for r in results if r["verdict"] == "fair" and not r["issues"]]
    return {
        "evaluated": len(results),
        "fair_count": len(fair),
        "lopsided": [r["label"] for r in results if r["verdict"] != "fair" and not r["issues"]],
        "proposals": results,
        "scoring": "standard_ppr",
    }

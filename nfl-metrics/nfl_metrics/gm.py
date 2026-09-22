"""Autonomous-GM analytics: projections, replacement values, waivers, trades,
game environments and a MILP lineup optimizer.

Everything here is pure computation over polars frames / plain dicts so it can
be unit-tested with synthetic data. Data fetching lives in ``data.py`` and
``feeds.py``; ``server.py`` glues them together.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
from scipy.optimize import Bounds, LinearConstraint, milp

Position = Literal["QB", "RB", "WR", "TE", "K", "DST"]
SKILL_POSITIONS = ("QB", "RB", "WR", "TE")

DEFAULT_SCORING: dict[str, float] = {
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

SCORING_PRESETS: dict[str, dict[str, float]] = {
    "ppr": DEFAULT_SCORING,
    "half_ppr": {**DEFAULT_SCORING, "rec": 0.5},
    "standard": {**DEFAULT_SCORING, "rec": 0.0},
    "tep": {**DEFAULT_SCORING, "te_rec_bonus": 0.5},
    "superflex_ppr": {**DEFAULT_SCORING, "pass_td": 6},
}

DEFAULT_ROSTER_SLOTS: dict[str, int] = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}
FLEX_ELIGIBLE: dict[str, tuple[str, ...]] = {
    "FLEX": ("RB", "WR", "TE"),
    "SUPER_FLEX": ("QB", "RB", "WR", "TE"),
    "REC_FLEX": ("WR", "TE"),
    "WRRB_FLEX": ("RB", "WR"),
}

# Typical starters across a 12-team league; used for replacement-level baselines.
DEFAULT_STARTERS_PER_TEAM: dict[str, float] = {"QB": 1.0, "RB": 2.5, "WR": 3.0, "TE": 1.0}
# Typical share of flex-type slots that ends up at each position.
FLEX_SPLIT: dict[str, dict[str, float]] = {
    "FLEX": {"RB": 0.45, "WR": 0.45, "TE": 0.10},
    "WRRB_FLEX": {"RB": 0.5, "WR": 0.5},
    "REC_FLEX": {"WR": 0.8, "TE": 0.2},
    "SUPER_FLEX": {"QB": 0.85, "RB": 0.05, "WR": 0.1},
}


def starters_from_slots(slots: dict[str, int]) -> dict[str, float]:
    """Expected starters per position for a lineup, spreading flex slots by FLEX_SPLIT."""
    out = {p: float(slots.get(p, 0)) for p in SKILL_POSITIONS}
    for slot, n in slots.items():
        for pos, share in FLEX_SPLIT.get(slot, {}).items():
            out[pos] += n * share
    return out


def _n_starters(starters: dict[str, float], pos: str) -> int:
    """Whole starting slots at a position; fractional flex shares round to nearest."""
    x = starters.get(pos, 1.0)
    return max(1, int(round(x))) if x > 0 else 0


def resolve_scoring(scoring: str | dict[str, float] | None) -> dict[str, float]:
    if scoring is None:
        return DEFAULT_SCORING
    if isinstance(scoring, str):
        key = scoring.lower().replace("-", "_")
        if key not in SCORING_PRESETS:
            raise ValueError(f"Unknown scoring preset {scoring!r}; choose from {sorted(SCORING_PRESETS)} or pass a dict")
        return SCORING_PRESETS[key]
    return {**DEFAULT_SCORING, **scoring}


# ------------------------------------------------------------------ projections
def player_projections(
    weekly_points: pl.DataFrame,
    through_week: int | None = None,
    recent_weeks: int = 4,
    recent_weight: float = 0.6,
) -> pl.DataFrame:
    """Blend season-to-date mean with a recent-form mean; report floor/ceiling from the
    observed distribution.

    Columns: player_id, player, position, team, games, mean_pts, recent_pts,
    projection, std, floor (20th pct), ceiling (80th pct).
    """
    lf = weekly_points.lazy()
    if through_week is not None:
        lf = lf.filter(pl.col("week") <= through_week)
    max_week = lf.select(pl.col("week").max()).collect().item()
    if max_week is None:
        return pl.DataFrame(
            schema={
                "player_id": pl.Utf8,
                "player": pl.Utf8,
                "position": pl.Utf8,
                "team": pl.Utf8,
                "games": pl.UInt32,
                "mean_pts": pl.Float64,
                "recent_pts": pl.Float64,
                "projection": pl.Float64,
                "std": pl.Float64,
                "floor": pl.Float64,
                "ceiling": pl.Float64,
            }
        )
    recent_cutoff = max_week - recent_weeks + 1
    out = (
        lf.group_by("player_id")
        .agg(
            pl.col("player").last(),
            pl.col("position").last(),
            pl.col("team").last(),
            pl.len().alias("games"),
            pl.col("fantasy_points").mean().alias("mean_pts"),
            pl.col("fantasy_points").filter(pl.col("week") >= recent_cutoff).mean().alias("recent_pts"),
            pl.col("fantasy_points").std().alias("std"),
            pl.col("fantasy_points").quantile(0.2).alias("floor"),
            pl.col("fantasy_points").quantile(0.8).alias("ceiling"),
        )
        .with_columns(
            pl.when(pl.col("recent_pts").is_null())
            .then(pl.col("mean_pts"))
            .otherwise(recent_weight * pl.col("recent_pts") + (1 - recent_weight) * pl.col("mean_pts"))
            .alias("projection"),
            pl.col("std").fill_null(0.0),
        )
        .with_columns(pl.col("floor").fill_null(pl.col("projection")), pl.col("ceiling").fill_null(pl.col("projection")))
        .sort("projection", descending=True)
        .collect()
    )
    return out


def replacement_levels(
    projections: pl.DataFrame,
    league_size: int = 12,
    starters_per_team: dict[str, float] | None = None,
    rostered_ids: set[str] | None = None,
) -> dict[str, float]:
    """Projection of the best freely-available (or N+1th) player per position.

    If ``rostered_ids`` is provided the replacement level is the best unrostered
    player; otherwise it's the player ranked just outside the league's starter pool.
    """
    starters = starters_per_team or DEFAULT_STARTERS_PER_TEAM
    levels: dict[str, float] = {}
    for pos in SKILL_POSITIONS:
        pos_df = projections.filter(pl.col("position") == pos).sort("projection", descending=True)
        if pos_df.is_empty():
            levels[pos] = 0.0
            continue
        if rostered_ids is not None:
            free = pos_df.filter(~pl.col("player_id").is_in(list(rostered_ids)))
            levels[pos] = float(free["projection"][0]) if not free.is_empty() else 0.0
        else:
            idx = min(int(round(league_size * starters.get(pos, 1.0))), pos_df.height - 1)
            levels[pos] = float(pos_df["projection"][idx])
    return levels


def value_over_replacement(projections: pl.DataFrame, levels: dict[str, float]) -> pl.DataFrame:
    level_expr = pl.col("position").replace_strict(levels, default=0.0, return_dtype=pl.Float64)
    return projections.with_columns((pl.col("projection") - level_expr).alias("vor")).sort("vor", descending=True)


# --------------------------------------------------------------------- injuries
INACTIVE_STATUSES = {"out", "inactive", "doubtful", "injured reserve", "ir", "pup", "suspended"}


@dataclass
class InjuryReport:
    player: str
    player_id: str | None
    team: str
    position: str
    status: str
    detail: str | None
    source: str


def injury_leverage(
    reports: list[InjuryReport],
    projections: pl.DataFrame,
    my_roster_ids: set[str],
    available_ids: set[str] | None,
    levels: dict[str, float],
    same_team_boost: float = 0.15,
    top_n_pivots: int = 5,
) -> dict[str, object]:
    """Turn inactives into pivot recommendations.

    For each inactive on my roster: rank replacements from ``available_ids`` (or all
    unrostered players when None) at the same position by projection, and add a
    usage bump for teammates of the inactive player (vacated targets/carries).
    Also lists beneficiaries league-wide so the caller can spot streaming targets.
    """
    inactive = [r for r in reports if r.status.lower() in INACTIVE_STATUSES]
    proj_by_id = {row["player_id"]: row for row in projections.iter_rows(named=True)}
    proj_by_name = {row["player"].lower(): row for row in projections.iter_rows(named=True) if row["player"]}

    def lookup(r: InjuryReport) -> dict | None:
        if r.player_id and r.player_id in proj_by_id:
            return proj_by_id[r.player_id]
        return proj_by_name.get(r.player.lower())

    inactive_teams: dict[str, list[str]] = {}
    for r in inactive:
        inactive_teams.setdefault(r.team, []).append(r.position)

    def replacement_pool(position: str) -> pl.DataFrame:
        pool = projections.filter(pl.col("position") == position)
        if available_ids is not None:
            pool = pool.filter(pl.col("player_id").is_in(list(available_ids)))
        else:
            pool = pool.filter(~pl.col("player_id").is_in(list(my_roster_ids)))
        inactive_ids = {lookup(r)["player_id"] for r in inactive if lookup(r)}
        pool = pool.filter(~pl.col("player_id").is_in(list(inactive_ids)))
        boost = pl.col("team").is_in([t for t, ps in inactive_teams.items() if position in ps])
        return (
            pool.with_columns(
                pl.when(boost).then(pl.col("projection") * (1 + same_team_boost)).otherwise(pl.col("projection")).alias("adjusted_projection"),
                boost.alias("vacated_usage_beneficiary"),
            )
            .sort("adjusted_projection", descending=True)
            .head(top_n_pivots)
        )

    my_hits = []
    for r in inactive:
        row = lookup(r)
        if not row or row["player_id"] not in my_roster_ids:
            continue
        pivots = replacement_pool(r.position)
        lost = float(row["projection"])
        my_hits.append(
            {
                "inactive": {"player": r.player, "team": r.team, "position": r.position, "status": r.status, "detail": r.detail, "source": r.source},
                "projected_points_lost": round(lost, 2),
                "replacement_level": round(levels.get(r.position, 0.0), 2),
                "pivots": [
                    {
                        "player": p["player"],
                        "player_id": p["player_id"],
                        "team": p["team"],
                        "projection": round(float(p["projection"]), 2),
                        "adjusted_projection": round(float(p["adjusted_projection"]), 2),
                        "value_vs_inactive": round(float(p["adjusted_projection"]) - lost, 2),
                        "vacated_usage_beneficiary": bool(p["vacated_usage_beneficiary"]),
                    }
                    for p in pivots.iter_rows(named=True)
                ],
                "urgency": "emergency" if lost >= levels.get(r.position, 0.0) + 3 else "routine",
            }
        )

    beneficiaries = []
    for team, positions in inactive_teams.items():
        for pos in set(positions):
            teammates = projections.filter((pl.col("team") == team) & (pl.col("position") == pos)).sort("projection", descending=True).head(3)
            for t in teammates.iter_rows(named=True):
                if any(lookup(r) and lookup(r)["player_id"] == t["player_id"] for r in inactive):
                    continue
                beneficiaries.append(
                    {
                        "player": t["player"],
                        "player_id": t["player_id"],
                        "team": team,
                        "position": pos,
                        "projection": round(float(t["projection"]), 2),
                        "rostered_by_me": t["player_id"] in my_roster_ids,
                        "available": available_ids is None or t["player_id"] in available_ids,
                    }
                )

    return {
        "inactives_considered": len(inactive),
        "inactives": [r.__dict__ for r in inactive if r.position in SKILL_POSITIONS][:60],
        "my_roster_impacts": my_hits,
        "league_wide_beneficiaries": sorted(beneficiaries, key=lambda b: -b["projection"])[:15],
    }


# ---------------------------------------------------------------------- waivers
@dataclass
class WaiverSettings:
    waiver_type: Literal["faab", "rolling", "reverse_standings", "none"] = "faab"
    faab_budget: float = 100.0
    faab_remaining: float | None = None
    waiver_position: int | None = None
    league_size: int = 12
    weeks_remaining: int = 10
    roster_size: int = 15


def waiver_recommendations(
    projections: pl.DataFrame,
    my_roster_ids: set[str],
    available_ids: set[str] | None,
    levels: dict[str, float],
    settings: WaiverSettings,
    max_targets: int = 10,
    max_drops: int = 5,
) -> dict[str, object]:
    """Rank free agents by value over replacement and translate that into FAAB
    dollars or waiver-priority tiers, plus drop candidates from my roster."""
    vor = value_over_replacement(projections, levels)
    pool = vor.filter(~pl.col("player_id").is_in(list(my_roster_ids)))
    if available_ids is not None:
        pool = pool.filter(pl.col("player_id").is_in(list(available_ids)))
    pool = pool.filter(pl.col("position").is_in(list(SKILL_POSITIONS))).head(max_targets)

    mine = vor.filter(pl.col("player_id").is_in(list(my_roster_ids))).sort("vor")
    drops = mine.head(max_drops)

    remaining = settings.faab_remaining if settings.faab_remaining is not None else settings.faab_budget
    total_vor = float(pool.filter(pl.col("vor") > 0)["vor"].sum()) or 1.0
    horizon = max(1, settings.weeks_remaining)

    targets = []
    for i, p in enumerate(pool.iter_rows(named=True)):
        vor_pts = max(float(p["vor"]), 0.0)
        ros_value = vor_pts * horizon
        entry: dict[str, object] = {
            "rank": i + 1,
            "player": p["player"],
            "player_id": p["player_id"],
            "team": p["team"],
            "position": p["position"],
            "projection": round(float(p["projection"]), 2),
            "vor_per_week": round(vor_pts, 2),
            "ros_value_points": round(ros_value, 1),
            "ceiling": round(float(p["ceiling"]), 2),
        }
        if settings.waiver_type == "faab":
            # Spend proportional to share of positive VOR in the pool, scaled by how
            # much of the season remains; cap the top target at 45% of remaining budget.
            share = vor_pts / total_vor
            season_fraction = min(1.0, horizon / 17)
            bid = remaining * share * (0.5 + 0.5 * season_fraction)
            bid = min(bid, 0.45 * remaining)
            if vor_pts >= 6:
                bid = max(bid, 0.25 * remaining)
            entry["recommended_bid"] = round(bid) if remaining >= 10 else round(bid, 1)
            entry["bid_range"] = [round(bid * 0.8), round(min(bid * 1.25, remaining))]
            entry["priority_tier"] = "must-add" if vor_pts >= 6 else "strong" if vor_pts >= 3 else "speculative" if vor_pts > 0 else "bench-depth"
        else:
            tier = "use-priority" if vor_pts >= 5 else "claim-if-cheap" if vor_pts >= 2 else "free-agent-after-waivers"
            entry["priority_tier"] = tier
            entry["use_waiver_priority"] = tier == "use-priority" and (settings.waiver_position is None or settings.waiver_position <= max(3, settings.league_size // 3))
        targets.append(entry)

    return {
        "waiver_type": settings.waiver_type,
        "faab_remaining": remaining if settings.waiver_type == "faab" else None,
        "waiver_position": settings.waiver_position if settings.waiver_type != "faab" else None,
        "replacement_levels": {k: round(v, 2) for k, v in levels.items()},
        "targets": targets,
        "drop_candidates": [
            {
                "player": d["player"],
                "player_id": d["player_id"],
                "position": d["position"],
                "team": d["team"],
                "projection": round(float(d["projection"]), 2),
                "vor_per_week": round(float(d["vor"]), 2),
            }
            for d in drops.iter_rows(named=True)
        ],
        "claim_pairs": [
            {"add": t["player"], "drop": d["player"], "net_vor_per_week": round(float(t["vor_per_week"]) - float(d["vor"]), 2)}
            for t, d in zip(targets[:max_drops], drops.iter_rows(named=True))
            if float(t["vor_per_week"]) > float(d["vor"])
        ],
    }


# ----------------------------------------------------------------------- trades
def positional_needs(
    roster_ids: set[str],
    vor: pl.DataFrame,
    starters_per_team: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-position starter strength, depth and surplus/deficit versus league-typical starters."""
    starters = starters_per_team or DEFAULT_STARTERS_PER_TEAM
    mine = vor.filter(pl.col("player_id").is_in(list(roster_ids)))
    needs: dict[str, dict[str, float]] = {}
    for pos in SKILL_POSITIONS:
        pos_df = mine.filter(pl.col("position") == pos).sort("projection", descending=True)
        n_start = _n_starters(starters, pos)
        starter_vor = float(pos_df.head(n_start)["vor"].sum()) if not pos_df.is_empty() else 0.0
        bench_vor = float(pos_df.slice(n_start)["vor"].clip(lower_bound=0).sum()) if pos_df.height > n_start else 0.0
        missing = max(0, n_start - pos_df.height)
        needs[pos] = {
            "count": pos_df.height,
            "starter_vor": round(starter_vor, 2),
            "bench_surplus_vor": round(bench_vor, 2),
            "missing_starters": missing,
            # positive => surplus to trade away; negative => deficit to fill
            "balance": round(bench_vor - 6.0 * missing - max(0.0, -starter_vor), 2),
        }
    return needs


def scan_trades(
    vor: pl.DataFrame,
    my_roster_ids: set[str],
    rival_rosters: dict[str, set[str]],
    starters_per_team: dict[str, float] | None = None,
    max_proposals: int = 10,
    min_gain: float = 0.5,
) -> dict[str, object]:
    """Propose 1-for-1 (and 2-for-1 consolidation) trades that send from my surplus
    positions to a rival's deficit positions, requiring both sides gain starter VOR."""
    my_needs = positional_needs(my_roster_ids, vor, starters_per_team)
    my_players = vor.filter(pl.col("player_id").is_in(list(my_roster_ids)))
    surplus_positions = [p for p, n in my_needs.items() if n["balance"] > 0]
    deficit_positions = [p for p, n in my_needs.items() if n["balance"] < 0]

    starters = starters_per_team or DEFAULT_STARTERS_PER_TEAM

    def weakest_starter_vor(players: pl.DataFrame, pos: str) -> float:
        n = _n_starters(starters, pos)
        pos_df = players.filter(pl.col("position") == pos).sort("projection", descending=True)
        if pos_df.height < n:
            return 0.0  # empty starting slot => replacement level
        return float(pos_df["vor"][n - 1])

    proposals: list[dict[str, object]] = []
    for rival, rival_ids in rival_rosters.items():
        rival_needs = positional_needs(rival_ids, vor, starters_per_team)
        rival_players = vor.filter(pl.col("player_id").is_in(list(rival_ids)))
        # positions worth acquiring: my deficits, or anywhere the rival has bench surplus
        # that would upgrade my weakest starter.
        get_positions = sorted(set(deficit_positions) | {p for p, n in rival_needs.items() if n["balance"] > 0})
        for give_pos in surplus_positions:
            if rival_needs[give_pos]["balance"] >= 0:
                continue
            gives = my_players.filter(pl.col("position") == give_pos).sort("projection", descending=True)
            n_start = _n_starters(starters, give_pos)
            gives = gives.slice(n_start).filter(pl.col("vor") > 0)  # only trade bench pieces with real value
            for get_pos in get_positions:
                if get_pos == give_pos or rival_needs[get_pos]["balance"] <= 0:
                    continue
                gets = rival_players.filter(pl.col("position") == get_pos).sort("projection", descending=True)
                n_start_get = _n_starters(starters, get_pos)
                gets = gets.slice(n_start_get)
                my_weakest = weakest_starter_vor(my_players, get_pos)
                rival_weakest = weakest_starter_vor(rival_players, give_pos)
                for g in gives.iter_rows(named=True):
                    for r in gets.iter_rows(named=True):
                        # gain = lineup upgrade at the position received minus a bench-depth haircut for what leaves
                        my_upgrade = float(r["vor"]) - my_weakest
                        rival_upgrade = float(g["vor"]) - rival_weakest
                        if my_upgrade <= 0 or rival_upgrade <= 0:
                            continue
                        my_gain = my_upgrade - 0.35 * max(0.0, float(g["vor"]))
                        rival_gain = rival_upgrade - 0.35 * max(0.0, float(r["vor"]))
                        if my_gain >= min_gain and rival_gain >= min_gain:
                            proposals.append(
                                {
                                    "partner": rival,
                                    "give": [{"player": g["player"], "position": give_pos, "projection": round(float(g["projection"]), 2), "vor": round(float(g["vor"]), 2)}],
                                    "get": [{"player": r["player"], "position": get_pos, "projection": round(float(r["projection"]), 2), "vor": round(float(r["vor"]), 2)}],
                                    "my_starter_gain": round(my_gain, 2),
                                    "partner_starter_gain": round(rival_gain, 2),
                                    "fairness": round(min(my_gain, rival_gain) / max(my_gain, rival_gain), 2),
                                    "type": "1-for-1",
                                }
                            )
                # 2-for-1 consolidation: two bench pieces for one rival starter-level player
                gets_top = rival_players.filter(pl.col("position") == get_pos).sort("projection", descending=True).head(n_start_get)
                if gives.height >= 2 and not gets_top.is_empty():
                    g1, g2 = gives.row(0, named=True), gives.row(1, named=True)
                    for r in gets_top.iter_rows(named=True):
                        my_upgrade = float(r["vor"]) - my_weakest
                        if my_upgrade <= 0:
                            continue
                        my_gain = my_upgrade - 0.35 * (float(g1["vor"]) + float(g2["vor"]))
                        rival_gain = max(0.0, float(g1["vor"]) - rival_weakest) + 0.6 * float(g2["vor"]) - max(0.0, float(r["vor"]))
                        if my_gain >= min_gain and rival_gain >= min_gain:
                            proposals.append(
                                {
                                    "partner": rival,
                                    "give": [
                                        {"player": g1["player"], "position": give_pos, "projection": round(float(g1["projection"]), 2), "vor": round(float(g1["vor"]), 2)},
                                        {"player": g2["player"], "position": give_pos, "projection": round(float(g2["projection"]), 2), "vor": round(float(g2["vor"]), 2)},
                                    ],
                                    "get": [{"player": r["player"], "position": get_pos, "projection": round(float(r["projection"]), 2), "vor": round(float(r["vor"]), 2)}],
                                    "my_starter_gain": round(my_gain, 2),
                                    "partner_starter_gain": round(rival_gain, 2),
                                    "fairness": round(min(my_gain, rival_gain) / max(my_gain, rival_gain), 2),
                                    "type": "2-for-1",
                                }
                            )
    proposals.sort(key=lambda p: (-(float(p["my_starter_gain"]) * float(p["fairness"])), -float(p["my_starter_gain"])))
    return {
        "my_positional_needs": my_needs,
        "surplus_positions": surplus_positions,
        "deficit_positions": deficit_positions,
        "proposals": proposals[:max_proposals],
    }


# ------------------------------------------------------------ game environments
def game_environments(schedule: pl.DataFrame, week: int, team: str | None = None) -> list[dict[str, object]]:
    """Implied totals and game-script flags from spread/total lines in nflverse schedules."""
    games = schedule.filter(pl.col("week") == week)
    if team:
        games = games.filter((pl.col("home_team") == team.upper()) | (pl.col("away_team") == team.upper()))
    rows = []
    for g in games.iter_rows(named=True):
        spread = g.get("spread_line")  # positive => home favoured by that many
        total = g.get("total_line")
        home_implied = away_implied = None
        if spread is not None and total is not None:
            home_implied = round(total / 2 + spread / 2, 2)
            away_implied = round(total / 2 - spread / 2, 2)
        favourite = None if spread is None or spread == 0 else (g["home_team"] if spread > 0 else g["away_team"])
        rows.append(
            {
                "game_id": g["game_id"],
                "kickoff": f"{g.get('gameday')} {g.get('gametime') or ''}".strip(),
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "spread_line": spread,
                "favourite": favourite,
                "total_line": total,
                "home_implied_total": home_implied,
                "away_implied_total": away_implied,
                "home_moneyline": g.get("home_moneyline"),
                "away_moneyline": g.get("away_moneyline"),
                "roof": g.get("roof"),
                "surface": g.get("surface"),
                "temp": g.get("temp"),
                "wind": g.get("wind"),
                "div_game": bool(g.get("div_game")) if g.get("div_game") is not None else None,
                "game_script": _game_script(spread, total),
                "final_score": None
                if g.get("home_score") is None
                else {"home": g.get("home_score"), "away": g.get("away_score")},
            }
        )
    return rows


def _game_script(spread: float | None, total: float | None) -> dict[str, str]:
    if spread is None or total is None:
        return {"pace": "unknown", "spread_class": "unknown", "confidence": "none", "note": "no betting line available yet"}
    pace = "shootout" if total >= 50 else "high" if total >= 46 else "average" if total >= 42 else "low"
    lopsided = abs(spread) >= 7
    fav_side = "home" if spread > 0 else "away"
    dog_side = "away" if spread > 0 else "home"
    if lopsided:
        note = f"{fav_side} favourite likely to lean on the run late; {dog_side} passing volume boosted by negative game script"
    elif abs(spread) <= 3:
        note = "close game: balanced scripts, RB and WR usage near baseline"
    else:
        note = "moderate favourite; mild run lean for favourite, mild pass lean for underdog"
    return {
        "pace": pace,
        "spread_class": "lopsided" if lopsided else "competitive",
        "favourite_side": fav_side,
        "underdog_side": dog_side,
        "confidence": "line",
        "note": note,
    }


# ----------------------------------------------------------------- lineup MILP
@dataclass
class LineupPlayer:
    player_id: str
    name: str
    position: str
    team: str
    projection: float
    floor: float | None = None
    ceiling: float | None = None
    std: float | None = None
    opponent: str | None = None
    locked: bool = False
    excluded: bool = False


@dataclass
class LineupRequest:
    players: list[LineupPlayer]
    slots: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_ROSTER_SLOTS))
    objective: Literal["projection", "floor", "ceiling"] = "projection"
    variance_weight: float = 0.0  # >0 rewards std (ceiling chasing); <0 penalises it (floor)
    stack_bonus: float = 0.0  # points added per QB + same-team pass-catcher pair
    max_from_team: int | None = None
    bring_back_bonus: float = 0.0  # bonus for opposing pass-catcher alongside a QB stack


def optimize_lineup(req: LineupRequest) -> dict[str, object]:
    """Mixed-integer program (HiGHS via scipy) that picks starters into slots.

    Variables: x[p,s] in {0,1} for every eligible (player, slot) pair; y[q,r] in {0,1}
    for QB q paired with same-team pass-catcher r (stacking bonus), linked so a pair
    only counts when both are started.
    """
    players = [p for p in req.players if not p.excluded]
    if not players:
        raise ValueError("no eligible players supplied")

    def score(p: LineupPlayer) -> float:
        base = p.projection
        if req.objective == "floor":
            base = p.floor if p.floor is not None else p.projection - (p.std or 0.0)
        elif req.objective == "ceiling":
            base = p.ceiling if p.ceiling is not None else p.projection + (p.std or 0.0)
        return base + req.variance_weight * (p.std or 0.0)

    slot_names = [s for s, n in req.slots.items() for _ in range(n)]
    eligible: list[tuple[int, int]] = []  # (player idx, slot idx)
    for pi, p in enumerate(players):
        for si, slot in enumerate(slot_names):
            allowed = FLEX_ELIGIBLE.get(slot, (slot,))
            if p.position in allowed:
                eligible.append((pi, si))
    if not eligible:
        raise ValueError("no player is eligible for any slot")

    n_x = len(eligible)
    pairs: list[tuple[int, int]] = []
    if req.stack_bonus or req.bring_back_bonus:
        for qi, q in enumerate(players):
            if q.position != "QB":
                continue
            for ri, r in enumerate(players):
                if r.position in ("WR", "TE") and (r.team == q.team or (req.bring_back_bonus and q.opponent and r.team == q.opponent)):
                    pairs.append((qi, ri))
    n_y = len(pairs)
    n = n_x + n_y

    c = np.zeros(n)
    for k, (pi, _) in enumerate(eligible):
        c[k] = -score(players[pi])
    for k, (qi, ri) in enumerate(pairs):
        bonus = req.stack_bonus if players[ri].team == players[qi].team else req.bring_back_bonus
        c[n_x + k] = -bonus

    rows: list[np.ndarray] = []
    lbs: list[float] = []
    ubs: list[float] = []

    def add(row: np.ndarray, lb: float, ub: float) -> None:
        rows.append(row)
        lbs.append(lb)
        ubs.append(ub)

    # each slot filled at most once (exactly once if enough players)
    for si in range(len(slot_names)):
        row = np.zeros(n)
        for k, (_, s) in enumerate(eligible):
            if s == si:
                row[k] = 1
        if row.any():
            add(row, 1 if row.sum() >= 1 else 0, 1)
    # each player at most one slot; locked players must start
    for pi, p in enumerate(players):
        row = np.zeros(n)
        for k, (pp, _) in enumerate(eligible):
            if pp == pi:
                row[k] = 1
        if row.any():
            add(row, 1 if p.locked else 0, 1)
        elif p.locked:
            raise ValueError(f"locked player {p.name} is not eligible for any slot")
    # stack linking: y <= started(q), y <= started(r)
    for k, (qi, ri) in enumerate(pairs):
        for who in (qi, ri):
            row = np.zeros(n)
            row[n_x + k] = 1
            for kk, (pp, _) in enumerate(eligible):
                if pp == who:
                    row[kk] = -1
            add(row, -np.inf, 0)
    # team cap
    if req.max_from_team:
        for team in {p.team for p in players}:
            row = np.zeros(n)
            for k, (pi, _) in enumerate(eligible):
                if players[pi].team == team:
                    row[k] = 1
            if row.any():
                add(row, 0, req.max_from_team)

    res = milp(
        c,
        constraints=LinearConstraint(np.vstack(rows), np.array(lbs), np.array(ubs)),
        integrality=np.ones(n),
        bounds=Bounds(0, 1),
    )
    if res.status != 0 or res.x is None:
        raise ValueError(f"lineup optimisation infeasible: {res.message}")

    x = np.round(res.x)
    starters = []
    for k, (pi, si) in enumerate(eligible):
        if x[k] > 0.5:
            p = players[pi]
            starters.append(
                {
                    "slot": slot_names[si],
                    "player": p.name,
                    "player_id": p.player_id,
                    "position": p.position,
                    "team": p.team,
                    "projection": round(p.projection, 2),
                    "floor": p.floor,
                    "ceiling": p.ceiling,
                    "objective_score": round(score(p), 2),
                }
            )
    started_ids = {s["player_id"] for s in starters}
    stacks = [
        {"qb": players[qi].name, "pass_catcher": players[ri].name, "team": players[ri].team, "kind": "stack" if players[ri].team == players[qi].team else "bring-back"}
        for k, (qi, ri) in enumerate(pairs)
        if x[n_x + k] > 0.5
    ]
    bench = [
        {"player": p.name, "position": p.position, "team": p.team, "projection": round(p.projection, 2)}
        for p in players
        if p.player_id not in started_ids
    ]
    slot_order = list(req.slots.keys())
    starters.sort(key=lambda s: (slot_order.index(s["slot"]), -s["projection"]))
    return {
        "objective": req.objective,
        "variance_weight": req.variance_weight,
        "stack_bonus": req.stack_bonus,
        "total_projection": round(sum(s["projection"] for s in starters), 2),
        "total_objective_score": round(-float(res.fun), 2),
        "starters": starters,
        "stacks": stacks,
        "bench": bench,
        "solver": {"status": int(res.status), "message": str(res.message)},
    }

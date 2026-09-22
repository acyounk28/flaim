"""Unit tests for the GM tool suite on small synthetic frames (no network)."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_metrics import gm


def _proj(rows: list[tuple[str, str, str, str, float]]) -> pl.DataFrame:
    """rows: (player_id, player, position, team, projection)"""
    df = pl.DataFrame(
        rows,
        schema=["player_id", "player", "position", "team", "projection"],
        orient="row",
    )
    return df.with_columns(
        games=pl.lit(6, dtype=pl.UInt32),
        mean_pts=pl.col("projection"),
        recent_pts=pl.col("projection"),
        std=pl.lit(4.0),
        floor=pl.col("projection") - 4,
        ceiling=pl.col("projection") + 4,
    )


POOL = _proj(
    [
        ("qb1", "QB One", "QB", "KC", 22.0),
        ("qb2", "QB Two", "QB", "BUF", 20.0),
        ("qb3", "QB Three", "QB", "NYJ", 14.0),
        ("rb1", "RB One", "RB", "SF", 18.0),
        ("rb2", "RB Two", "RB", "DET", 16.0),
        ("rb3", "RB Three", "RB", "KC", 14.0),
        ("rb4", "RB Four", "RB", "PHI", 12.0),
        ("rb5", "RB Five", "RB", "NE", 9.0),
        ("rb6", "RB Six", "RB", "SF", 6.0),
        ("wr1", "WR One", "WR", "KC", 19.0),
        ("wr2", "WR Two", "WR", "SF", 17.0),
        ("wr3", "WR Three", "WR", "MIA", 15.0),
        ("wr4", "WR Four", "WR", "DET", 13.0),
        ("wr5", "WR Five", "WR", "BUF", 10.0),
        ("wr6", "WR Six", "WR", "KC", 7.0),
        ("te1", "TE One", "TE", "KC", 14.0),
        ("te2", "TE Two", "TE", "SF", 10.0),
        ("te3", "TE Three", "TE", "NYJ", 6.0),
    ]
)
STARTERS = {"QB": 1.0, "RB": 2.0, "WR": 2.0, "TE": 1.0}


# ------------------------------------------------------------------ helpers
def test_starters_from_slots_spreads_flex():
    s = gm.starters_from_slots({"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 2, "SUPER_FLEX": 1})
    assert s["QB"] == pytest.approx(1.85)
    assert s["RB"] == pytest.approx(2 + 0.9 + 0.05)
    assert s["TE"] == pytest.approx(1.2)
    assert gm._n_starters(s, "TE") == 1
    assert gm._n_starters(s, "RB") == 3
    assert gm._n_starters({"K": 0.0}, "K") == 0


def test_resolve_scoring_presets_and_custom():
    assert gm.resolve_scoring("half_ppr")["rec"] == 0.5
    assert gm.resolve_scoring("tep")["te_rec_bonus"] == 0.5
    custom = gm.resolve_scoring({"rec": 2.0})
    assert custom["rec"] == 2.0 and custom["pass_td"] == gm.DEFAULT_SCORING["pass_td"]
    with pytest.raises(ValueError):
        gm.resolve_scoring("nope")


def test_player_projections_blends_recent_form():
    weekly = pl.DataFrame(
        {
            "player_id": ["a"] * 6,
            "player": ["A"] * 6,
            "position": ["WR"] * 6,
            "team": ["KC"] * 6,
            "week": [1, 2, 3, 4, 5, 6],
            "fantasy_points": [5.0, 5.0, 5.0, 20.0, 20.0, 20.0],
        }
    )
    proj = gm.player_projections(weekly, recent_weeks=3, recent_weight=0.5)
    row = proj.row(0, named=True)
    assert row["mean_pts"] == pytest.approx(12.5)
    assert row["recent_pts"] == pytest.approx(20.0)
    assert row["projection"] == pytest.approx(16.25)
    assert row["floor"] <= row["projection"] <= row["ceiling"]
    assert gm.player_projections(weekly, through_week=0).height == 0


def test_replacement_levels_use_starter_pool_and_rostered_ids():
    levels = gm.replacement_levels(POOL, league_size=2, starters_per_team=STARTERS)
    # 2 teams x 2 RB starters => 4 starters; replacement is the next RB (rb5 = 9)
    assert levels["RB"] == pytest.approx(9.0)
    levels2 = gm.replacement_levels(POOL, league_size=2, starters_per_team=STARTERS, rostered_ids={"rb1", "rb2", "rb3", "rb4", "rb5"})
    assert levels2["RB"] == pytest.approx(6.0)  # best unrostered RB


# ------------------------------------------------------------- injuries
def test_injury_leverage_only_pivots_for_inactive_statuses():
    levels = {"QB": 14.0, "RB": 9.0, "WR": 10.0, "TE": 6.0}
    reports = [
        gm.InjuryReport("WR One", "wr1", "KC", "WR", "Out", "hamstring", "espn"),
        gm.InjuryReport("RB One", "rb1", "SF", "RB", "Questionable", "ankle", "espn"),
    ]
    out = gm.injury_leverage(reports, POOL, my_roster_ids={"wr1", "rb1", "te1"}, available_ids={"wr3", "wr6", "rb5"}, levels=levels)
    assert out["inactives_considered"] == 1
    hits = out["my_roster_impacts"]
    assert len(hits) == 1 and hits[0]["inactive"]["player"] == "WR One"
    assert hits[0]["projected_points_lost"] == pytest.approx(19.0)
    pivots = hits[0]["pivots"]
    names = [p["player"] for p in pivots]
    assert names[0] == "WR Three"
    assert all(p["player"] != "RB Five" for p in pivots)  # same position only
    wr6 = next(p for p in pivots if p["player"] == "WR Six")
    assert wr6["vacated_usage_beneficiary"] is True and wr6["adjusted_projection"] > wr6["projection"]


def test_injury_leverage_no_roster_impact_when_healthy():
    out = gm.injury_leverage([], POOL, {"wr1"}, None, {"WR": 10.0})
    assert out["my_roster_impacts"] == [] and out["inactives_considered"] == 0


# --------------------------------------------------------------- waivers
def test_waiver_faab_bids_scale_with_vor_and_budget():
    levels = gm.replacement_levels(POOL, league_size=2, starters_per_team=STARTERS)
    settings = gm.WaiverSettings(waiver_type="faab", faab_budget=100, faab_remaining=50, league_size=2, weeks_remaining=8)
    out = gm.waiver_recommendations(POOL, my_roster_ids={"rb5", "wr5", "te3"}, available_ids={"rb1", "rb6", "wr1", "te1"}, levels=levels, settings=settings)
    targets = out["targets"]
    assert [t["player"] for t in targets][:2] == ["RB One", "WR One"] or targets[0]["vor_per_week"] >= targets[1]["vor_per_week"]
    top = targets[0]
    assert 0 < top["recommended_bid"] <= 50
    assert top["bid_range"][0] <= top["recommended_bid"] <= top["bid_range"][1]
    weak = next(t for t in targets if t["player"] == "RB Six")
    assert weak["recommended_bid"] <= 1
    assert out["waiver_type"] == "faab"
    drops = out["drop_candidates"]
    assert drops[0]["player"] == "TE Three"  # lowest VOR on my roster


def test_waiver_rolling_uses_priority_tiers_and_claim_pairs():
    levels = gm.replacement_levels(POOL, league_size=2, starters_per_team=STARTERS)
    settings = gm.WaiverSettings(waiver_type="rolling", waiver_position=1, league_size=2, weeks_remaining=8)
    out = gm.waiver_recommendations(POOL, my_roster_ids={"rb5", "wr5", "te3"}, available_ids={"rb1", "rb6", "wr1"}, levels=levels, settings=settings)
    top = out["targets"][0]
    assert "recommended_bid" not in top
    assert top["priority_tier"] == "use-priority" and top["use_waiver_priority"] is True
    assert out["claim_pairs"][0]["add"] == top["player"]
    assert out["claim_pairs"][0]["drop"] == "TE Three"
    assert out["claim_pairs"][0]["net_vor_per_week"] > 0


# ---------------------------------------------------------------- trades
def test_scan_trades_maps_surplus_to_rival_deficit():
    levels = gm.replacement_levels(POOL, league_size=2, starters_per_team=STARTERS)
    vor = gm.value_over_replacement(POOL, levels)
    me = {"qb1", "rb1", "rb2", "rb3", "rb4", "wr5", "wr6", "te3"}  # RB-rich, WR/TE-poor
    rival = {"qb2", "rb5", "rb6", "wr1", "wr2", "wr3", "wr4", "te1", "te2"}  # WR/TE-rich, RB-poor
    out = gm.scan_trades(vor, me, {"Rival": rival}, starters_per_team=STARTERS)
    assert "RB" in out["surplus_positions"]
    assert set(out["deficit_positions"]) >= {"WR"}
    assert out["proposals"], "expected at least one mutually beneficial proposal"
    p = out["proposals"][0]
    assert p["partner"] == "Rival"
    assert all(g["position"] == "RB" for g in p["give"])
    assert p["my_starter_gain"] > 0 and p["partner_starter_gain"] > 0
    # never trade a starter, only bench surplus
    assert all(g["player"] not in {"RB One", "RB Two"} for g in p["give"])


def test_scan_trades_empty_when_no_fit():
    levels = gm.replacement_levels(POOL, league_size=2, starters_per_team=STARTERS)
    vor = gm.value_over_replacement(POOL, levels)
    out = gm.scan_trades(vor, {"qb1", "rb1", "wr1", "te1"}, {"R": {"qb2", "rb2", "wr2", "te2"}}, starters_per_team=STARTERS)
    assert out["proposals"] == []


# --------------------------------------------------------- game environments
def test_game_environments_implied_totals_and_missing_lines():
    sched = pl.DataFrame(
        {
            "game_id": ["2025_05_KC_JAX", "2025_05_SF_LA"],
            "week": [5, 5],
            "gameday": ["2025-10-06", "2025-10-05"],
            "gametime": ["20:15", None],
            "home_team": ["JAX", "LA"],
            "away_team": ["KC", "SF"],
            "spread_line": [-3.5, None],
            "total_line": [46.5, None],
            "home_moneyline": [150, None],
            "away_moneyline": [-180, None],
            "roof": ["outdoors", "dome"],
            "surface": ["grass", "sportturf"],
            "temp": [70, None],
            "wind": [5, None],
            "div_game": [0, 1],
            "home_score": [None, None],
            "away_score": [None, None],
        }
    )
    rows = gm.game_environments(sched, week=5)
    kc = next(r for r in rows if r["away_team"] == "KC")
    assert kc["favourite"] == "KC"
    assert kc["home_implied_total"] == pytest.approx(21.5)
    assert kc["away_implied_total"] == pytest.approx(25.0)
    assert kc["home_implied_total"] + kc["away_implied_total"] == pytest.approx(46.5)
    la = next(r for r in rows if r["home_team"] == "LA")
    assert la["home_implied_total"] is None and la["favourite"] is None
    assert la["game_script"]["confidence"] == "none"
    assert gm.game_environments(sched, week=5, team="sf")[0]["game_id"] == "2025_05_SF_LA"


def test_game_script_labels():
    assert gm._game_script(10.0, 52.0)["pace"] == "shootout"
    gs = gm._game_script(-9.0, 38.0)
    assert gs["favourite_side"] == "away" and gs["spread_class"] == "lopsided" and gs["pace"] == "low"
    assert gm._game_script(None, None)["confidence"] == "none"


# ------------------------------------------------------------------ lineup
def _lp(pid, pos, team, proj, floor=None, ceil=None, opp=None, **kw):
    return gm.LineupPlayer(pid, pid, pos, team, proj, floor=floor, ceiling=ceil, opponent=opp, **kw)


LINEUP_POOL = [
    _lp("QB-KC", "QB", "KC", 22, 15, 30, opp="LV"),
    _lp("QB-BUF", "QB", "BUF", 23, 12, 34, opp="MIA"),
    _lp("RB-A", "RB", "SF", 18, 14, 22),
    _lp("RB-B", "RB", "DET", 16, 10, 22),
    _lp("RB-C", "RB", "NE", 12, 11, 13),
    _lp("WR-KC", "WR", "KC", 15, 8, 24, opp="LV"),
    _lp("WR-LV", "WR", "LV", 14, 9, 20, opp="KC"),
    _lp("WR-MIA", "WR", "MIA", 16, 7, 26, opp="BUF"),
    _lp("WR-DET", "WR", "DET", 13, 12, 14),
    _lp("TE-KC", "TE", "KC", 11, 6, 16, opp="LV"),
    _lp("TE-SF", "TE", "SF", 9, 8, 10),
]
SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1}


def test_optimize_lineup_projection_objective_fills_slots():
    out = gm.optimize_lineup(gm.LineupRequest(players=LINEUP_POOL, slots=SLOTS))
    assert out["solver"]["status"] == 0
    starters = out["starters"]
    assert len(starters) == 7
    assert sum(1 for s in starters if s["slot"] == "FLEX") == 1
    names = {s["player"] for s in starters}
    assert names == {"QB-BUF", "RB-A", "RB-B", "WR-MIA", "WR-KC", "WR-LV", "TE-KC"} or out["total_projection"] == pytest.approx(
        23 + 18 + 16 + 16 + 15 + 14 + 11
    )
    assert len(out["bench"]) == len(LINEUP_POOL) - 7


def test_optimize_lineup_floor_objective_prefers_safe_players():
    out = gm.optimize_lineup(gm.LineupRequest(players=LINEUP_POOL, slots=SLOTS, objective="floor"))
    names = {s["player"] for s in out["starters"]}
    assert "QB-KC" in names  # floor 15 > BUF floor 12
    assert "WR-DET" in names  # floor 12 beats boom/bust WRs


def test_optimize_lineup_ceiling_and_variance_weight():
    ceil = gm.optimize_lineup(gm.LineupRequest(players=LINEUP_POOL, slots=SLOTS, objective="ceiling"))
    assert "QB-BUF" in {s["player"] for s in ceil["starters"]}
    risky = gm.optimize_lineup(gm.LineupRequest(players=LINEUP_POOL, slots=SLOTS, variance_weight=1.0))
    safe = gm.optimize_lineup(gm.LineupRequest(players=LINEUP_POOL, slots=SLOTS, variance_weight=-1.0))
    assert risky["total_objective_score"] >= safe["total_objective_score"]


def test_optimize_lineup_stack_and_bring_back_bonus():
    base = gm.optimize_lineup(gm.LineupRequest(players=LINEUP_POOL, slots=SLOTS))
    stacked = gm.optimize_lineup(gm.LineupRequest(players=LINEUP_POOL, slots=SLOTS, stack_bonus=5.0, bring_back_bonus=3.0))
    names = {s["player"] for s in stacked["starters"]}
    assert "QB-KC" in names and "WR-KC" in names and "TE-KC" in names
    assert "WR-LV" in names  # bring-back from the opponent
    assert stacked["stacks"], "stack metadata should be reported"
    assert stacked["total_projection"] <= base["total_projection"]
    assert stacked["total_objective_score"] > stacked["total_projection"]


def test_optimize_lineup_locks_excludes_and_infeasible():
    players = [gm.LineupPlayer(**{**p.__dict__}) for p in LINEUP_POOL]
    players[0].locked = True  # QB-KC
    players[1].excluded = True  # QB-BUF
    players[2].excluded = True  # RB-A
    out = gm.optimize_lineup(gm.LineupRequest(players=players, slots=SLOTS))
    names = {s["player"] for s in out["starters"]}
    assert "QB-KC" in names and "QB-BUF" not in names and "RB-A" not in names
    with pytest.raises(ValueError):
        gm.optimize_lineup(gm.LineupRequest(players=LINEUP_POOL, slots={"QB": 3}))


def test_optimize_lineup_max_from_team():
    out = gm.optimize_lineup(gm.LineupRequest(players=LINEUP_POOL, slots=SLOTS, stack_bonus=5.0, max_from_team=2))
    teams = [s["team"] for s in out["starters"]]
    assert max(teams.count(t) for t in set(teams)) <= 2

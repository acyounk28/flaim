"""Unit tests for the standard-PPR trade-value model on small synthetic frames (no network)."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_metrics import metrics
from nfl_metrics import trade_value as tv
from nfl_metrics.trade_value import PPR_SCORING, TradeValueInputs

THROUGH = 6
PLAYERS = [
    # id, name, pos, team
    ("qb1", "QB One", "QB", "KC"),
    ("qb2", "QB Two", "QB", "BUF"),
    ("qb3", "QB Three", "QB", "NYJ"),
    ("rb1", "RB One", "RB", "SF"),
    ("rb2", "RB Two", "RB", "DET"),
    ("rb3", "RB Three", "RB", "KC"),
    ("rb4", "RB Four", "RB", "PHI"),
    ("rb5", "RB Five", "RB", "NE"),
    ("rb6", "RB Six", "RB", "SF"),
    ("wr1", "WR One", "WR", "KC"),
    ("wr2", "WR Two", "WR", "SF"),
    ("wr3", "WR Three", "WR", "MIA"),
    ("wr4", "WR Four", "WR", "DET"),
    ("wr5", "WR Five", "WR", "BUF"),
    ("wr6", "WR Six", "WR", "KC"),
    ("te1", "TE One", "TE", "KC"),
    ("te2", "TE Two", "TE", "SF"),
    ("te3", "TE Three", "TE", "NYJ"),
]
PPG = {"qb1": 24, "qb2": 20, "qb3": 14, "rb1": 20, "rb2": 17, "rb3": 14, "rb4": 12, "rb5": 9, "rb6": 6, "wr1": 19, "wr2": 17, "wr3": 15, "wr4": 13, "wr5": 10, "wr6": 7, "te1": 14, "te2": 10, "te3": 6}


def _weekly(ppg: dict[str, float], weeks: range = range(1, THROUGH + 1), skip: dict[str, set[int]] | None = None) -> pl.DataFrame:
    rows = []
    for pid, name, pos, team in PLAYERS:
        for w in weeks:
            if skip and w in skip.get(pid, set()):
                continue
            rows.append((pid, name, pos, team, w, float(ppg[pid])))
    return pl.DataFrame(rows, schema=["player_id", "player", "position", "team", "week", "fantasy_points"], orient="row")


def _usage(target_share: dict[str, list[float]]) -> pl.DataFrame:
    rows = [(pid, w + 1, ts, ts * 0.9, 40.0, 6.0, 0.0) for pid, shares in target_share.items() for w, ts in enumerate(shares)]
    return pl.DataFrame(rows, schema=["player_id", "week", "target_share", "air_yards_share", "receiving_air_yards", "targets", "carries"], orient="row")


def _inputs(**overrides) -> TradeValueInputs:
    kw = dict(season=2025, through_week=THROUGH, weekly_points=_weekly(PPG), team_games={t: THROUGH for *_, t in PLAYERS})
    kw.update(overrides)
    return TradeValueInputs(**kw)


def _row(values: pl.DataFrame, pid: str) -> dict:
    return values.filter(pl.col("player_id") == pid).row(0, named=True)


def test_ppr_scoring_is_standard_and_used_for_weekly_points():
    assert PPR_SCORING["rec"] == 1.0 and PPR_SCORING["rec_yd"] == 0.1 and PPR_SCORING["pass_td"] == 4
    stats = pl.DataFrame(
        {
            "player_id": ["a"], "player_display_name": ["A"], "position": ["WR"], "team": ["KC"], "season": [2025], "week": [1], "season_type": ["REG"],
            "passing_yards": [0], "passing_tds": [0], "passing_interceptions": [0], "rushing_yards": [0], "rushing_tds": [0],
            "receptions": [5], "receiving_yards": [100], "receiving_tds": [1], "sack_fumbles_lost": [0], "rushing_fumbles_lost": [0], "receiving_fumbles_lost": [0],
            "passing_2pt_conversions": [0], "rushing_2pt_conversions": [0], "receiving_2pt_conversions": [0],
        }
    )
    out = metrics.weekly_fantasy_points(stats, PPR_SCORING)
    assert out["fantasy_points"][0] == pytest.approx(5 + 10 + 6)


def test_values_are_bounded_ranked_and_replacement_relative():
    values = tv.player_trade_values(_inputs())
    assert values.height == len(PLAYERS)
    assert values["trade_value"].max() <= 100 and values["trade_value"].min() >= 0
    assert values["trade_value"].is_sorted(descending=True)
    # sub-replacement players are capped at bench value; the best RB out-values the best QB (positional scarcity)
    rb6 = _row(values, "rb6")
    assert rb6["vor"] <= 0 and rb6["trade_value"] <= 15
    assert rb6["trade_value"] < _row(values, "rb5")["trade_value"]
    assert _row(values, "rb1")["trade_value"] > _row(values, "qb1")["trade_value"]
    r = _row(values, "rb1")
    assert r["projected_ppg"] == pytest.approx(20.0) and r["adjusted_ppg"] == pytest.approx(20.0)
    assert r["usage_multiplier"] == 1 and r["depth_multiplier"] == 1 and r["injury_multiplier"] == 1
    assert r["role"] == "unknown" and r["market_score"] is None


def test_missing_sources_lower_confidence_not_value():
    bare = tv.player_trade_values(_inputs())
    full = tv.player_trade_values(
        _inputs(
            weekly_usage=_usage({p[0]: [0.2] * THROUGH for p in PLAYERS}),
            snaps=pl.DataFrame({"player_id": [p[0] for p in PLAYERS] * THROUGH, "week": [w for w in range(1, THROUGH + 1) for _ in PLAYERS], "snap_pct": [0.8] * (len(PLAYERS) * THROUGH)}),
            depth_chart=pl.DataFrame({"player_id": [p[0] for p in PLAYERS], "team": [p[3] for p in PLAYERS], "position": [p[2] for p in PLAYERS], "pos_rank": [1] * len(PLAYERS)}),
            injuries=pl.DataFrame({"player_id": ["rb6"], "week": [1], "report_status": ["Questionable"]}),
        )
    )
    assert _row(bare, "rb1")["confidence"] < _row(full, "rb1")["confidence"]
    assert _row(bare, "rb1")["projected_ppg"] == _row(full, "rb1")["projected_ppg"]
    assert bare.height == full.height
    cov = _inputs().coverage()
    assert cov["weekly_points"] and not cov["usage"] and not cov["market"]


def test_small_sample_flag_and_prior_season_baseline():
    weekly = _weekly(PPG, weeks=range(1, 3))
    prior = _weekly({**PPG, "rb1": 10.0}, weeks=range(1, 17))
    values = tv.player_trade_values(_inputs(through_week=2, weekly_points=weekly, prior_points=prior, team_games={t: 2 for *_, t in PLAYERS}))
    r = _row(values, "rb1")
    assert "small_sample" in r["flags"]
    assert r["prior_ppg"] == 10 and r["prior_games"] == 16
    assert 10 < r["projected_ppg"] < 20  # blended toward the prior with only two games
    # a prior season with <4 games is not a baseline
    tiny = tv.player_trade_values(_inputs(through_week=2, weekly_points=weekly, prior_points=_weekly(PPG, weeks=range(1, 3))))
    assert _row(tiny, "rb1")["prior_ppg"] is None


def test_target_share_decline_is_penalised_and_flagged():
    steady = {p[0]: [0.25] * THROUGH for p in PLAYERS}
    declining = {**steady, "wr1": [0.30, 0.30, 0.30, 0.15, 0.15, 0.15]}
    rising = {**steady, "wr2": [0.10, 0.10, 0.10, 0.25, 0.25, 0.25]}
    base = tv.player_trade_values(_inputs(weekly_usage=_usage(steady)))
    dec = tv.player_trade_values(_inputs(weekly_usage=_usage(declining)))
    ris = tv.player_trade_values(_inputs(weekly_usage=_usage(rising)))
    w1 = _row(dec, "wr1")
    assert "target_share_declining" in w1["flags"] and "air_yards_share_declining" in w1["flags"]
    assert w1["usage_trend"] == "declining"
    assert w1["usage_multiplier"] < 1 and w1["trade_value"] < _row(base, "wr1")["trade_value"]
    assert w1["target_share_recent"] == pytest.approx(0.15) and w1["target_share_season"] == pytest.approx(0.225)
    w2 = _row(ris, "wr2")
    assert w2["usage_trend"] == "rising" and w2["usage_multiplier"] > 1
    # rises are rewarded less than declines are punished
    assert (1 - w1["usage_multiplier"]) > (w2["usage_multiplier"] - 1)


def test_usage_trend_needs_more_games_than_the_recent_window():
    weekly = _weekly(PPG, weeks=range(1, 4))
    usage = _usage({"wr1": [0.30, 0.10, 0.10]})
    values = tv.player_trade_values(_inputs(through_week=3, weekly_points=weekly, weekly_usage=usage))
    r = _row(values, "wr1")
    assert r["usage_trend"] == "stable" and not [f for f in r["flags"] if f.endswith("_declining")]


def test_snap_collapse_caps_usage_multiplier():
    snaps = pl.DataFrame({"player_id": ["rb1"] * THROUGH, "week": list(range(1, THROUGH + 1)), "snap_pct": [0.9, 0.9, 0.9, 0.2, 0.2, 0.2]})
    r = _row(tv.player_trade_values(_inputs(snaps=snaps)), "rb1")
    assert "role_reduced" in r["flags"] and "snap_pct_declining" in r["flags"]
    assert r["usage_multiplier"] <= 0.7


def test_red_zone_share_is_computed_per_team_and_declines_flag():
    rz = pl.DataFrame(
        {
            "player_id": ["rb1"] * THROUGH + ["rb6"] * THROUGH,
            "week": list(range(1, THROUGH + 1)) * 2,
            "rz_touches": [8, 8, 8, 2, 2, 2] + [2, 2, 2, 8, 8, 8],
            "team_rz_plays": [10] * THROUGH * 2,
        }
    )
    values = tv.player_trade_values(_inputs(red_zone=rz))
    r = _row(values, "rb1")
    assert r["rz_touch_share_season"] == pytest.approx(0.5) and r["rz_touch_share_recent"] == pytest.approx(0.2)
    assert "rz_touch_share_declining" in r["flags"]
    assert r["rz_touches_per_game"] == pytest.approx(5.0)


def test_depth_chart_role_backup_and_demotion():
    dc = pl.DataFrame({"player_id": ["rb1", "rb2", "wr1", "qb2"], "team": ["SF", "DET", "KC", "BUF"], "position": ["RB", "RB", "WR", "QB"], "pos_rank": [2, 1, 3, 2]})
    prev = pl.DataFrame({"player_id": ["rb1", "rb2", "wr1"], "team": ["SF", "DET", "KC"], "position": ["RB", "RB", "WR"], "pos_rank": [1, 2, 3]})
    base = tv.player_trade_values(_inputs())
    values = tv.player_trade_values(_inputs(depth_chart=dc, depth_chart_prev=prev))
    rb1 = _row(values, "rb1")
    assert rb1["role"] == "RB2" and "depth_chart_demoted" in rb1["flags"]
    assert rb1["depth_multiplier"] == pytest.approx(tv.DEPTH_MULTIPLIERS["RB"][2] * tv.DEMOTION_MULTIPLIER)
    assert rb1["trade_value"] < _row(base, "rb1")["trade_value"]
    rb2 = _row(values, "rb2")
    assert rb2["role"] == "RB1" and "depth_chart_promoted" in rb2["flags"] and rb2["depth_multiplier"] == 1
    wr1 = _row(values, "wr1")
    assert wr1["role"] == "WR3" and "depth_chart_backup" not in wr1["flags"]  # WR3s start in 3WR sets
    qb2 = _row(values, "qb2")
    assert qb2["depth_multiplier"] == pytest.approx(0.35) and qb2["trade_value"] < _row(base, "qb2")["trade_value"]
    assert _row(values, "te1")["role"] == "unknown"


def test_designations_and_missed_time_reduce_value():
    skip = {"rb1": {3, 4}}
    weekly = _weekly(PPG, skip=skip)
    desig = {"rb2": {"status": "Out", "detail": "knee", "source": "espn"}, "wr1": {"status": "Questionable", "detail": None, "source": "espn"}}
    injuries = pl.DataFrame({"player_id": ["rb1", "rb1", "rb1"], "week": [2, 3, 4], "report_status": ["Questionable", "Out", "Out"]})
    values = tv.player_trade_values(_inputs(weekly_points=weekly, current_designations=desig, injuries=injuries))
    base = tv.player_trade_values(_inputs())
    rb1 = _row(values, "rb1")
    assert rb1["games"] == 4 and rb1["games_missed"] == 2 and "missed_time_this_season" in rb1["flags"]
    assert rb1["injury_reports"] == 3 and rb1["injury_risk"] > 0 and rb1["injury_multiplier"] < 1
    rb2 = _row(values, "rb2")
    assert rb2["status"] == "out" and rb2["status_source"] == "espn" and "designation_out" in rb2["flags"]
    assert rb2["injury_multiplier"] == pytest.approx(tv.STATUS_MULTIPLIERS["out"])
    assert rb2["trade_value"] < _row(base, "rb2")["trade_value"]
    wr1 = _row(values, "wr1")
    assert "designation_questionable" in wr1["flags"] and wr1["injury_multiplier"] == pytest.approx(tv.STATUS_MULTIPLIERS["questionable"])


def test_expected_points_feed_the_projection():
    exp = pl.DataFrame({"player_id": ["rb1"] * THROUGH, "week": list(range(1, THROUGH + 1)), "exp_points": [30.0] * THROUGH})
    r = _row(tv.player_trade_values(_inputs(expected_points=exp)), "rb1")
    assert r["expected_ppg"] == 30 and r["projected_ppg"] > 20


def test_market_blend_and_gap_flags():
    # market loves rb5 (RB1 overall) and hates rb1 (RB30)
    market = pl.DataFrame({"player_id": ["rb5", "rb1", "wr1"], "ecr_pos_rank": [1.0, 30.0, 1.0], "ecr": [1.0, 60.0, 3.0]})
    base = tv.player_trade_values(_inputs())
    values = tv.player_trade_values(_inputs(market=market))
    rb5, rb1, wr1 = _row(values, "rb5"), _row(values, "rb1"), _row(values, "wr1")
    assert rb5["market_score"] == pytest.approx(100) and "market_overvalues" in rb5["flags"]
    assert rb5["trade_value"] > _row(base, "rb5")["trade_value"]
    assert "market_undervalues" in rb1["flags"] and rb1["trade_value"] < _row(base, "rb1")["trade_value"]
    assert rb1["model_score"] > rb1["market_score"]
    assert wr1["market_gap"] is not None and abs(wr1["market_gap"]) < tv.MARKET_GAP_FLAG
    assert not [f for f in wr1["flags"] if f.startswith("market_")]
    # trade_value is the documented blend
    assert rb1["trade_value"] == pytest.approx((1 - tv.MARKET_WEIGHT) * rb1["model_score"] + tv.MARKET_WEIGHT * rb1["market_score"], abs=0.06)


def test_explain_groups_dossier_sections():
    values = tv.player_trade_values(_inputs())
    d = tv.explain(_row(values, "rb1"))
    assert d["player"] == "RB One"
    assert {"production", "usage", "role", "injury", "market"} <= set(d)
    assert d["production"]["projected_ppg"] == 20 and d["usage"]["usage_trend"] == "stable"
    assert d["trade_value"] == _row(values, "rb1")["trade_value"]


def test_side_value_consolidates_depth_pieces():
    values = tv.player_trade_values(_inputs())
    rows = {r["player_id"]: r for r in values.to_dicts()}
    one = tv.side_value([rows["rb1"]])
    two = tv.side_value([rows["rb3"], rows["rb4"]])
    assert one["package_value"] == one["raw_sum"] == rows["rb1"]["trade_value"]
    assert two["package_value"] < two["raw_sum"]
    assert two["package_value"] == pytest.approx(rows["rb3"]["trade_value"] + 0.85 * rows["rb4"]["trade_value"], abs=0.1)
    assert two["best_asset"] == "RB Three"


def test_compare_trade_verdicts_and_unresolved():
    values = tv.player_trade_values(_inputs())
    fair = tv.compare_trade(values, ["rb1"], ["rb1"])
    assert fair["verdict"] == "fair" and fair["winner"] is None and fair["value_gap"] == 0
    lop = tv.compare_trade(values, ["rb6"], ["rb1"], labels=("me", "rival"))
    assert lop["verdict"] == "favors_me" and lop["winner"] == "me" and lop["value_gap"] > 0
    assert lop["me"]["receives_value"] == lop["rival"]["gives"]["package_value"]
    assert lop["scoring"] == "standard_ppr"
    unk = tv.compare_trade(values, ["nobody"], ["rb1"])
    assert unk["unresolved_ids"] == ["nobody"]


def test_compare_trade_lineup_impact_flags_unowned_players():
    values = tv.player_trade_values(_inputs())
    starters = {"QB": 1.0, "RB": 2.0, "WR": 2.0, "TE": 1.0}
    roster_a = {"qb1", "rb1", "rb5", "wr1", "wr5", "te1"}
    roster_b = {"qb2", "rb2", "rb3", "wr2", "wr3", "te2"}
    out = tv.compare_trade(values, ["rb5", "wr5"], ["rb2"], labels=("A", "B"), roster_a=roster_a, roster_b=roster_b, starters_per_team=starters)
    impact = out["lineup_impact"]
    assert impact["A"]["starting_lineup_ppg_before"] == pytest.approx(24 + 20 + 9 + 19 + 10 + 14)
    assert impact["A"]["starting_lineup_ppg_after"] == pytest.approx(24 + 20 + 17 + 19 + 0 + 14)
    assert impact["A"]["players_not_on_roster"] == []
    assert impact["B"]["starting_lineup_delta"] < 0
    bad = tv.compare_trade(values, ["rb2"], ["rb1"], labels=("A", "B"), roster_a=roster_a, roster_b=roster_b)
    assert bad["lineup_impact"]["A"]["players_not_on_roster"] == ["RB Two"]


def test_lineup_projection_uses_flex_slots():
    values = tv.player_trade_values(_inputs())
    total = tv.lineup_projection(values, {"qb1", "rb1", "rb2", "rb3", "wr1", "wr2", "te1"}, {"QB": 1, "RB": 2.5, "WR": 2.5, "TE": 1})
    assert total == pytest.approx(24 + 20 + 17 + 19 + 17 + 14 + 14)


def test_evaluate_proposals_orders_fair_first_and_reports_issues():
    values = tv.player_trade_values(_inputs())
    rosters = {"A": {"rb1", "wr1"}, "B": {"rb2", "wr2"}, "C": {"rb6"}}
    proposals = [
        {"team_a": "A", "team_b": "B", "a_gives": ["rb1"], "b_gives": ["rb2", "wr2"], "label": "big"},
        {"team_a": "A", "team_b": "B", "a_gives": ["wr1"], "b_gives": ["wr2"], "label": "swap"},
        {"team_a": "A", "team_b": "C", "a_gives": ["rb2"], "b_gives": ["rb6"], "label": "bogus"},
        {"team_a": "A", "team_b": "Z", "a_gives": ["rb1"], "b_gives": ["nobody"]},
    ]
    out = tv.evaluate_proposals(values, proposals, rosters=rosters)
    assert out["evaluated"] == 4 and out["scoring"] == "standard_ppr"
    by = {p["label"]: p for p in out["proposals"]}
    assert by["bogus"]["issues"] == ["A does not roster: RB Two"]
    assert any("unknown team Z" in i for i in by["A <-> Z"]["issues"]) and any("unresolved" in i for i in by["A <-> Z"]["issues"])
    clean = [p for p in out["proposals"] if not p["issues"]]
    assert [p["label"] for p in out["proposals"][: len(clean)]] == [p["label"] for p in clean]
    assert clean == sorted(clean, key=lambda p: abs(p["value_gap_pct"]))
    assert by["swap"]["both_lineups_improve"] is not None
    assert set(out["lopsided"]) <= {"big", "swap"}


def test_empty_inputs_return_typed_empty_frame():
    empty = pl.DataFrame(schema={"player_id": pl.Utf8, "player": pl.Utf8, "position": pl.Utf8, "team": pl.Utf8, "week": pl.Int64, "fantasy_points": pl.Float64})
    values = tv.player_trade_values(_inputs(weekly_points=empty))
    assert values.is_empty() and "trade_value" in values.columns
    out = tv.compare_trade(values, ["x"], [])
    assert out["unresolved_ids"] == ["x"] and out["verdict"] == "fair"

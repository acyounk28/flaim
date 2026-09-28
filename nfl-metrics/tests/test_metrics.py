"""Metric functions on a tiny hand-built play-by-play frame (no network)."""

from __future__ import annotations

import polars as pl
import pytest

from nfl_metrics import metrics
from nfl_metrics.data import PBP_COLUMNS


def _pbp(rows: list[dict]) -> pl.LazyFrame:
    base = {c: None for c in PBP_COLUMNS}
    filled = [{**base, "season": 2025, "season_type": "REG", "week": 1, "game_id": "g1", "play_id": i, **r} for i, r in enumerate(rows)]
    df = pl.DataFrame(filled, infer_schema_length=None)
    return df.with_columns(
        [pl.col(c).cast(pl.Float64) for c in ("air_yards", "yards_after_catch", "yards_gained", "epa", "qb_epa", "cpoe", "yardline_100")]
        + [pl.col(c).cast(pl.Int64) for c in ("pass", "rush", "qb_dropback", "qb_scramble", "sack", "complete_pass", "success", "pass_touchdown", "rush_touchdown", "interception", "touchdown")]
    ).lazy()


def _pass(qb, rec, ay, yds, complete=1, epa=1.0, cpoe=5.0, team="KC", yl=50, td=0):
    return {
        "posteam": team, "defteam": "LV", "play_type": "pass", "pass": 1, "rush": 0, "qb_dropback": 1, "qb_scramble": 0, "sack": 0,
        "complete_pass": complete, "incomplete_pass": 1 - complete, "interception": 0, "air_yards": ay,
        "yards_after_catch": max(0, yds - ay) if complete else 0, "yards_gained": yds if complete else 0, "yardline_100": yl,
        "epa": epa, "qb_epa": epa, "cpoe": cpoe, "success": 1 if epa > 0 else 0, "touchdown": td, "pass_touchdown": td,
        "passer_player_id": qb, "passer_player_name": qb, "receiver_player_id": rec, "receiver_player_name": rec,
    }


PBP = _pbp(
    [
        _pass("QB1", "WR1", 10, 15, epa=2.0, cpoe=10),
        _pass("QB1", "WR1", 20, 0, complete=0, epa=-1.0, cpoe=-20),
        _pass("QB1", "WR2", 5, 8, epa=0.5, cpoe=4, yl=15),
        _pass("QB1", "TE1", 5, 5, epa=0.5, cpoe=6, yl=4, td=1),
        # scramble: QB shows up as rusher, passer fields null
        {"posteam": "KC", "defteam": "LV", "play_type": "run", "pass": 0, "rush": 1, "qb_dropback": 1, "qb_scramble": 1, "sack": 0,
         "complete_pass": 0, "yards_gained": 12, "epa": 1.5, "qb_epa": 1.5, "success": 1, "yardline_100": 40,
         "rusher_player_id": "QB1", "rusher_player_name": "QB1", "touchdown": 0, "rush_touchdown": 0, "interception": 0, "pass_touchdown": 0},
        # sack
        {"posteam": "KC", "defteam": "LV", "play_type": "pass", "pass": 1, "rush": 0, "qb_dropback": 1, "qb_scramble": 0, "sack": 1,
         "complete_pass": 0, "yards_gained": -7, "epa": -2.0, "qb_epa": -2.0, "success": 0, "yardline_100": 60,
         "passer_player_id": "QB1", "passer_player_name": "QB1", "touchdown": 0, "rush_touchdown": 0, "interception": 0, "pass_touchdown": 0},
        # RB carries incl red zone
        {"posteam": "KC", "defteam": "LV", "play_type": "run", "pass": 0, "rush": 1, "qb_dropback": 0, "qb_scramble": 0, "sack": 0,
         "complete_pass": 0, "yards_gained": 3, "epa": 0.1, "success": 1, "yardline_100": 8, "rusher_player_id": "RB1", "rusher_player_name": "RB1",
         "touchdown": 0, "rush_touchdown": 0, "interception": 0, "pass_touchdown": 0},
        {"posteam": "KC", "defteam": "LV", "play_type": "run", "pass": 0, "rush": 1, "qb_dropback": 0, "qb_scramble": 0, "sack": 0,
         "complete_pass": 0, "yards_gained": 2, "epa": 0.8, "success": 1, "yardline_100": 2, "rusher_player_id": "RB1", "rusher_player_name": "RB1",
         "touchdown": 1, "rush_touchdown": 1, "interception": 0, "pass_touchdown": 0},
        # another team's pass so shares are per-team
        _pass("QB9", "WR9", 30, 40, epa=3.0, cpoe=15, team="BUF"),
    ]
)


def test_qb_efficiency_counts_scrambles_and_sacks_as_dropbacks():
    out = metrics.qb_efficiency(PBP, min_dropbacks=1)
    qb = out.filter(pl.col("player_id") == "QB1").row(0, named=True)
    assert qb["dropbacks"] == 6
    assert qb["attempts"] == 4
    assert qb["scrambles"] == 1 and qb["sacks"] == 1
    assert qb["completions"] == 3
    assert qb["completion_pct"] == pytest.approx(0.75)
    assert qb["epa_per_dropback"] == pytest.approx((2 - 1 + 0.5 + 0.5 + 1.5 - 2) / 6, abs=1e-3)
    assert qb["cpoe"] == pytest.approx(0.0)  # (10-20+4+6)/4
    assert qb["air_yards_per_attempt"] == pytest.approx(10.0)
    assert qb["pass_td"] == 1
    assert metrics.qb_efficiency(PBP, min_dropbacks=5).height == 1  # QB9 filtered out


def test_receiver_usage_target_and_air_yard_shares():
    out = metrics.receiver_usage(PBP)
    wr1 = out.filter(pl.col("player_id") == "WR1").row(0, named=True)
    assert wr1["targets"] == 2 and wr1["receptions"] == 1
    assert wr1["target_share"] == pytest.approx(0.5)  # 2 of 4 KC attempts
    assert wr1["air_yards_share"] == pytest.approx(30 / 40)
    assert wr1["adot"] == pytest.approx(15.0)
    assert wr1["wopr"] == pytest.approx(1.5 * 0.5 + 0.7 * 0.75, abs=1e-3)
    assert wr1["deep_targets"] == 1
    wr9 = out.filter(pl.col("player_id") == "WR9").row(0, named=True)
    assert wr9["target_share"] == pytest.approx(1.0)
    assert wr9["team"] == "BUF"


def test_red_zone_usage_counts_carries_and_targets():
    out = metrics.red_zone_usage(PBP)
    rb1 = out.filter(pl.col("player_id") == "RB1").row(0, named=True)
    assert rb1["rz_touches"] == 2 and rb1["inside10_touches"] == 2 and rb1["inside5_touches"] == 1
    assert rb1["rz_td"] == 1
    te1 = out.filter(pl.col("player_id") == "TE1").row(0, named=True)
    assert te1["rz_touches"] == 1 and te1["inside5_touches"] == 1
    wr2 = out.filter(pl.col("player_id") == "WR2").row(0, named=True)
    assert wr2["rz_touches"] == 1 and wr2["inside10_touches"] == 0
    assert rb1["rz_touch_share"] == pytest.approx(0.5)  # 2 of 4 KC red-zone touches
    assert "WR1" not in out["player_id"].to_list()


def test_red_zone_weekly_touches_and_team_plays():
    out = metrics.red_zone_weekly(PBP)
    rb1 = out.filter(pl.col("player_id") == "RB1").row(0, named=True)
    assert rb1["week"] == 1 and rb1["team"] == "KC"
    assert rb1["rz_carries"] == 2 and rb1["rz_targets"] == 0 and rb1["rz_touches"] == 2
    assert rb1["team_rz_plays"] == 4
    te1 = out.filter(pl.col("player_id") == "TE1").row(0, named=True)
    assert te1["rz_targets"] == 1 and te1["rz_carries"] == 0
    assert "WR1" not in out["player_id"].to_list()


def test_week_filters():
    assert metrics.qb_efficiency(PBP, week=2, min_dropbacks=1).height == 0
    assert metrics.receiver_usage(PBP, weeks=(1, 3)).height == 4


def test_snap_share_and_route_participation():
    snaps = pl.DataFrame(
        {
            "season": [2025] * 3, "game_type": ["REG"] * 3, "week": [1, 1, 1], "player": ["A", "B", "C"], "pfr_player_id": ["a", "b", "c"],
            "position": ["WR", "WR", "RB"], "team": ["KC", "KC", "KC"], "opponent": ["LV"] * 3,
            "offense_snaps": [60, 30, 45], "offense_pct": [1.0, 0.5, 0.75], "defense_snaps": [0, 0, 0], "defense_pct": [0.0] * 3, "st_snaps": [0, 5, 0], "st_pct": [0.0, 0.2, 0.0],
        }
    )
    out = metrics.snap_share(snaps, week=1)
    a = out.filter(pl.col("player") == "A").row(0, named=True)
    assert a["offense_snaps"] == 60 and a["offense_snap_pct"] == pytest.approx(1.0)
    assert out.filter(pl.col("player") == "B").row(0, named=True)["offense_snap_pct"] == pytest.approx(0.5)

    part = pl.DataFrame(
        {
            "nflverse_game_id": ["g1"] * 3, "play_id": [1, 2, 3], "possession_team": ["KC"] * 3, "offense_players": ["r1;r2;q1", "r1;q1", "r1;r2;q1"],
            "n_offense": [11, 11, 11], "ngs_air_yards": [5.0, 10.0, 3.0], "was_pressure": [False] * 3, "route": ["GO", None, "SLANT"],
        }
    )
    pbp = pl.DataFrame(
        {
            "game_id": ["g1"] * 3, "play_id": [1, 2, 3], "season": [2025] * 3, "season_type": ["REG"] * 3, "week": [1, 1, 1], "posteam": ["KC"] * 3,
            "play_type": ["pass", "pass", "run"], "pass": [1, 1, 0], "qb_dropback": [1, 1, 0], "sack": [0, 0, 0], "receiver_player_id": ["r1", None, None], "receiver_player_name": ["R1", None, None],
            "complete_pass": [1, 0, 0], "yards_gained": [9, 0, 4], "air_yards": [5.0, None, None], "epa": [1.0, -0.5, 0.2], "qb_scramble": [0, 0, 0],
        }
    )
    rosters = pl.DataFrame({"gsis_id": ["r1", "r2", "q1"], "display_name": ["R One", "R Two", "Q One"], "position": ["WR", "WR", "QB"], "team": ["KC"] * 3, "season": [2025] * 3})
    out = metrics.route_participation(part, pbp.lazy(), rosters, week=1, min_routes=1)
    r1 = out.filter(pl.col("player_id") == "r1").row(0, named=True)
    r2 = out.filter(pl.col("player_id") == "r2").row(0, named=True)
    assert r1["routes"] == 2 and r1["route_participation"] == pytest.approx(1.0)
    assert r2["routes"] == 1 and r2["route_participation"] == pytest.approx(0.5)
    assert r1["targets_per_route_run"] == pytest.approx(0.5)
    assert "q1" not in out["player_id"].to_list()


def test_weekly_fantasy_points_scoring():
    stats = pl.DataFrame(
        {
            "player_id": ["a", "a"], "player_display_name": ["A", "A"], "position": ["TE", "TE"], "team": ["KC", "KC"], "week": [1, 2], "season_type": ["REG", "REG"],
            "passing_yards": [0, 0], "passing_tds": [0, 0], "passing_interceptions": [0, 0], "rushing_yards": [0, 10], "rushing_tds": [0, 0],
            "receptions": [5, 4], "receiving_yards": [50, 40], "receiving_tds": [1, 0], "rushing_fumbles_lost": [0, 1], "receiving_fumbles_lost": [0, 0], "sack_fumbles_lost": [0, 0],
            "passing_2pt_conversions": [0, 0], "rushing_2pt_conversions": [0, 0], "receiving_2pt_conversions": [0, 0],
        }
    )
    ppr = metrics.weekly_fantasy_points(stats, {"rec": 1, "rec_yd": 0.1, "rec_td": 6, "rush_yd": 0.1, "fum_lost": -2})
    assert ppr.sort("week")["fantasy_points"].to_list() == pytest.approx([16.0, 7.0])
    tep = metrics.weekly_fantasy_points(stats, {"rec": 1, "rec_yd": 0.1, "rec_td": 6, "rush_yd": 0.1, "fum_lost": -2, "te_rec_bonus": 0.5})
    assert tep.sort("week")["fantasy_points"].to_list() == pytest.approx([18.5, 9.0])

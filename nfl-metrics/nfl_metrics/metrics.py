"""Pure polars computations for efficiency and usage metrics.

Every function takes frames (or lazy frames) and returns a DataFrame; no I/O
happens here so the logic is unit-testable with tiny synthetic inputs.
"""

from __future__ import annotations

import polars as pl

REG = "REG"


def _week_filter(lf: pl.LazyFrame, week: int | None, weeks: tuple[int, int] | None) -> pl.LazyFrame:
    if week is not None:
        return lf.filter(pl.col("week") == week)
    if weeks is not None:
        return lf.filter(pl.col("week").is_between(weeks[0], weeks[1]))
    return lf


def _round(df: pl.DataFrame, digits: int = 3) -> pl.DataFrame:
    return df.with_columns([pl.col(c).round(digits) for c, dt in df.schema.items() if dt in (pl.Float32, pl.Float64)])


# --------------------------------------------------------------------------- QB
def qb_efficiency(
    pbp: pl.LazyFrame,
    week: int | None = None,
    weeks: tuple[int, int] | None = None,
    min_dropbacks: int = 10,
    season_type: str = REG,
) -> pl.DataFrame:
    """EPA per dropback, CPOE, air yards per attempt and related QB efficiency."""
    lf = _week_filter(pbp, week, weeks)
    if season_type:
        lf = lf.filter(pl.col("season_type") == season_type)
    # Scrambles carry the QB as rusher (passer fields are null); fold them back in.
    scramble = pl.col("qb_scramble") == 1
    dropbacks = (
        lf.filter(pl.col("qb_dropback") == 1)
        .with_columns(
            pl.coalesce([pl.col("passer_player_id"), pl.when(scramble).then(pl.col("rusher_player_id"))]).alias("qb_id"),
            pl.coalesce([pl.col("passer_player_name"), pl.when(scramble).then(pl.col("rusher_player_name"))]).alias("qb_name"),
        )
        .filter(pl.col("qb_id").is_not_null())
    )
    attempts = (pl.col("pass") == 1) & ~scramble
    out = (
        dropbacks.group_by("qb_id")
        .agg(
            pl.col("qb_name").drop_nulls().last().alias("player"),
            pl.col("posteam").drop_nulls().last().alias("team"),
            pl.len().alias("dropbacks"),
            pl.col("pass").filter(attempts & (pl.col("sack") == 0)).len().alias("attempts"),
            pl.col("complete_pass").sum().alias("completions"),
            pl.col("sack").sum().alias("sacks"),
            pl.col("qb_scramble").sum().alias("scrambles"),
            pl.col("qb_epa").sum().alias("total_epa"),
            pl.col("qb_epa").mean().alias("epa_per_dropback"),
            pl.col("qb_epa").filter(attempts).mean().alias("epa_per_pass"),
            pl.col("cpoe").drop_nulls().mean().alias("cpoe"),
            pl.col("air_yards").filter(attempts).sum().alias("air_yards"),
            pl.col("air_yards").filter(attempts & (pl.col("sack") == 0)).mean().alias("air_yards_per_attempt"),
            pl.col("yards_gained").filter(attempts).sum().alias("passing_yards"),
            pl.col("pass_touchdown").sum().alias("pass_td"),
            pl.col("interception").sum().alias("interceptions"),
            pl.col("success").mean().alias("success_rate"),
        )
        .filter(pl.col("dropbacks") >= min_dropbacks)
        .with_columns(
            (pl.col("completions") / pl.col("attempts")).alias("completion_pct"),
            (pl.col("sacks") / pl.col("dropbacks")).alias("sack_rate"),
        )
        .rename({"qb_id": "player_id"})
        .sort("epa_per_dropback", descending=True)
        .collect(engine="streaming")
    )
    return _round(out)


# --------------------------------------------------------------- receivers
def receiver_usage(
    pbp: pl.LazyFrame,
    week: int | None = None,
    weeks: tuple[int, int] | None = None,
    min_targets: int = 1,
    season_type: str = REG,
) -> pl.DataFrame:
    """Targets, target share, air yards, air-yards share, WOPR, RACR, EPA per target."""
    lf = _week_filter(pbp, week, weeks)
    if season_type:
        lf = lf.filter(pl.col("season_type") == season_type)
    passes = lf.filter((pl.col("pass") == 1) & (pl.col("sack") == 0) & pl.col("posteam").is_not_null())
    team_totals = passes.group_by("posteam").agg(
        pl.len().alias("team_pass_attempts"),
        pl.col("air_yards").sum().alias("team_air_yards"),
    )
    targeted = passes.filter(pl.col("receiver_player_id").is_not_null())
    players = targeted.group_by(["receiver_player_id", "posteam"]).agg(
        pl.col("receiver_player_name").drop_nulls().last().alias("player"),
        pl.len().alias("targets"),
        pl.col("complete_pass").sum().alias("receptions"),
        pl.col("yards_gained").filter(pl.col("complete_pass") == 1).sum().alias("receiving_yards"),
        pl.col("air_yards").sum().alias("air_yards"),
        pl.col("air_yards").mean().alias("adot"),
        pl.col("yards_after_catch").sum().alias("yac"),
        pl.col("epa").sum().alias("total_epa"),
        pl.col("epa").mean().alias("epa_per_target"),
        pl.col("pass_touchdown").sum().alias("receiving_td"),
        (pl.col("yardline_100") <= 20).sum().alias("red_zone_targets"),
        (pl.col("air_yards") >= 20).sum().alias("deep_targets"),
    )
    out = (
        players.join(team_totals, on="posteam", how="left")
        .with_columns(
            (pl.col("targets") / pl.col("team_pass_attempts")).alias("target_share"),
            (pl.col("air_yards") / pl.col("team_air_yards")).alias("air_yards_share"),
            (pl.col("receptions") / pl.col("targets")).alias("catch_rate"),
        )
        .with_columns(
            (1.5 * pl.col("target_share") + 0.7 * pl.col("air_yards_share")).alias("wopr"),
            pl.when(pl.col("air_yards") > 0).then(pl.col("receiving_yards") / pl.col("air_yards")).otherwise(None).alias("racr"),
        )
        .filter(pl.col("targets") >= min_targets)
        .rename({"receiver_player_id": "player_id", "posteam": "team"})
        .sort("wopr", descending=True, nulls_last=True)
        .collect(engine="streaming")
    )
    return _round(out)


# ---------------------------------------------------------------- red zone
def red_zone_usage(
    pbp: pl.LazyFrame,
    week: int | None = None,
    weeks: tuple[int, int] | None = None,
    season_type: str = REG,
) -> pl.DataFrame:
    """Red-zone (<=20), inside-10 and inside-5 touches (carries + targets) per player with team shares."""
    lf = _week_filter(pbp, week, weeks)
    if season_type:
        lf = lf.filter(pl.col("season_type") == season_type)
    rz = lf.filter((pl.col("yardline_100") <= 20) & ((pl.col("pass") == 1) | (pl.col("rush") == 1)) & (pl.col("sack") == 0))

    def touches(id_col: str, name_col: str, kind: str) -> pl.LazyFrame:
        src = rz.filter(pl.col(id_col).is_not_null())
        return src.group_by([id_col, "posteam"]).agg(
            pl.col(name_col).drop_nulls().last().alias("player"),
            pl.len().alias(f"rz_{kind}"),
            (pl.col("yardline_100") <= 10).sum().alias(f"inside10_{kind}"),
            (pl.col("yardline_100") <= 5).sum().alias(f"inside5_{kind}"),
            pl.col("touchdown").sum().alias(f"rz_{kind}_td"),
            pl.col("epa").sum().alias(f"rz_{kind}_epa"),
        ).rename({id_col: "player_id"})

    rushes = touches("rusher_player_id", "rusher_player_name", "carries")
    targets = touches("receiver_player_id", "receiver_player_name", "targets")
    team = rz.group_by("posteam").agg(pl.len().alias("team_rz_plays"))

    out = (
        rushes.join(targets, on=["player_id", "posteam"], how="full", coalesce=True)
        .with_columns(pl.coalesce(["player", "player_right"]).alias("player"))
        .drop("player_right")
        .fill_null(0)
        .with_columns(
            (pl.col("rz_carries") + pl.col("rz_targets")).alias("rz_touches"),
            (pl.col("inside10_carries") + pl.col("inside10_targets")).alias("inside10_touches"),
            (pl.col("inside5_carries") + pl.col("inside5_targets")).alias("inside5_touches"),
            (pl.col("rz_carries_td") + pl.col("rz_targets_td")).alias("rz_td"),
        )
        .join(team, on="posteam", how="left")
        .with_columns((pl.col("rz_touches") / pl.col("team_rz_plays")).alias("rz_touch_share"))
        .rename({"posteam": "team"})
        .sort("rz_touches", descending=True)
        .collect(engine="streaming")
    )
    return _round(out)


# ------------------------------------------------------------ snap counts
def snap_share(
    snaps: pl.DataFrame,
    week: int | None = None,
    weeks: tuple[int, int] | None = None,
    positions: list[str] | None = None,
    game_type: str = REG,
) -> pl.DataFrame:
    lf = _week_filter(snaps.lazy(), week, weeks)
    if game_type:
        lf = lf.filter(pl.col("game_type") == game_type)
    if positions:
        lf = lf.filter(pl.col("position").is_in([p.upper() for p in positions]))
    out = (
        lf.group_by(["pfr_player_id", "player", "team", "position"])
        .agg(
            pl.len().alias("games"),
            pl.col("offense_snaps").sum().alias("offense_snaps"),
            pl.col("offense_pct").mean().alias("offense_snap_pct"),
            pl.col("offense_pct").min().alias("min_offense_snap_pct"),
            pl.col("offense_pct").max().alias("max_offense_snap_pct"),
            pl.col("defense_snaps").sum().alias("defense_snaps"),
            pl.col("defense_pct").mean().alias("defense_snap_pct"),
            pl.col("st_snaps").sum().alias("st_snaps"),
        )
        .sort(["offense_snap_pct", "defense_snap_pct"], descending=[True, True])
        .collect()
    )
    return _round(out)


# ------------------------------------------------------- route participation
def route_participation(
    participation: pl.DataFrame,
    pbp: pl.LazyFrame,
    players: pl.DataFrame,
    week: int | None = None,
    weeks: tuple[int, int] | None = None,
    min_routes: int = 5,
    season_type: str = REG,
) -> pl.DataFrame:
    """Routes run (on field for a team dropback), route participation %, targets per route run.

    nflverse participation lists every offensive player on the field per play;
    a "route" is credited to non-QB/OL skill players on plays where the offense
    dropped back (excluding QB spikes/kneels via play_type filter).
    """
    plays = _week_filter(pbp, week, weeks)
    if season_type:
        plays = plays.filter(pl.col("season_type") == season_type)
    dropbacks = plays.filter((pl.col("qb_dropback") == 1) & pl.col("play_type").is_in(["pass", "run"])).select(
        ["game_id", "play_id", "posteam", "week", "receiver_player_id"]
    )
    part = (
        participation.lazy()
        .select(["nflverse_game_id", "play_id", "offense_players"])
        .rename({"nflverse_game_id": "game_id"})
        .with_columns(pl.col("play_id").cast(pl.Int64))
    )
    joined = dropbacks.with_columns(pl.col("play_id").cast(pl.Int64)).join(part, on=["game_id", "play_id"], how="inner")
    team_dropbacks = joined.group_by("posteam").agg(pl.len().alias("team_dropbacks"))

    exploded = joined.with_columns(pl.col("offense_players").str.split(";").alias("gsis_id")).explode("gsis_id").filter(
        pl.col("gsis_id").is_not_null() & (pl.col("gsis_id") != "")
    )
    skill = players.lazy().select(
        pl.col("gsis_id"),
        pl.col("display_name").alias("player"),
        pl.col("position"),
    ).filter(pl.col("position").is_in(["WR", "TE", "RB", "FB"]))
    routes = (
        exploded.join(skill, on="gsis_id", how="inner")
        .group_by(["gsis_id", "player", "position", "posteam"])
        .agg(
            pl.len().alias("routes"),
            (pl.col("receiver_player_id") == pl.col("gsis_id")).sum().alias("targets_on_routes"),
        )
        .join(team_dropbacks, on="posteam", how="left")
        .with_columns(
            (pl.col("routes") / pl.col("team_dropbacks")).alias("route_participation"),
            (pl.col("targets_on_routes") / pl.col("routes")).alias("targets_per_route_run"),
        )
        .filter(pl.col("routes") >= min_routes)
        .rename({"gsis_id": "player_id", "posteam": "team"})
        .sort("route_participation", descending=True)
        .collect(engine="streaming")
    )
    return _round(routes)


# -------------------------------------------------------- weekly fantasy pts
def weekly_fantasy_points(
    stats: pl.DataFrame,
    scoring: dict[str, float],
    season_type: str = REG,
) -> pl.DataFrame:
    """Per player-week fantasy points using a scoring dict (see gm.DEFAULT_SCORING)."""
    lf = stats.lazy()
    if season_type and "season_type" in stats.columns:
        lf = lf.filter(pl.col("season_type") == season_type)

    def col(name: str) -> pl.Expr:
        return pl.col(name).fill_null(0) if name in stats.columns else pl.lit(0.0)

    points = (
        col("passing_yards") * scoring.get("pass_yd", 0.04)
        + col("passing_tds") * scoring.get("pass_td", 4)
        + col("passing_interceptions") * scoring.get("pass_int", -2)
        + col("rushing_yards") * scoring.get("rush_yd", 0.1)
        + col("rushing_tds") * scoring.get("rush_td", 6)
        + col("receptions") * scoring.get("rec", 1)
        + col("receiving_yards") * scoring.get("rec_yd", 0.1)
        + col("receiving_tds") * scoring.get("rec_td", 6)
        + (col("rushing_fumbles_lost") + col("receiving_fumbles_lost") + col("sack_fumbles_lost")) * scoring.get("fum_lost", -2)
        + (col("passing_2pt_conversions") + col("rushing_2pt_conversions") + col("receiving_2pt_conversions")) * scoring.get("two_pt", 2)
    )
    if "te_rec_bonus" in scoring:
        points = points + pl.when(pl.col("position") == "TE").then(col("receptions") * scoring["te_rec_bonus"]).otherwise(0.0)
    return lf.select(
        pl.col("player_id"),
        pl.col("player_display_name").alias("player"),
        pl.col("position"),
        pl.col("team"),
        pl.col("week"),
        points.alias("fantasy_points"),
    ).collect()

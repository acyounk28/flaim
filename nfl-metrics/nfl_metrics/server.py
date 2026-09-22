"""FastMCP server: nflverse efficiency/usage metrics + autonomous-GM tools.

Transport is SSE by default (``/sse`` + ``/messages/``) or Streamable HTTP
(``/mcp``) via NFL_MCP_TRANSPORT. Every request must carry
``Authorization: Bearer $NFL_MCP_TOKEN``; ``/health`` is public.
"""

from __future__ import annotations

import hmac
import logging
import sys
import time
from typing import Annotated, Any, Literal

import polars as pl
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from . import gm, metrics
from .config import ConfigError, Settings, load_settings
from .data import DataStore
from .feeds import EspnFeed
from .league import SleeperLeague, load_sleeper_league, normalize_name, resolve_players

log = logging.getLogger("nfl_metrics")

Week = Annotated[int | None, Field(ge=1, le=22, description="Single week; omit for season-to-date")]
Season = Annotated[int | None, Field(ge=1999, le=2100, description="NFL season; defaults to the current season")]
ScoringArg = Annotated[
    str | dict[str, float] | None,
    Field(description="Scoring preset ('ppr', 'half_ppr', 'standard', 'tep', 'superflex_ppr') or a dict of per-stat points"),
]


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = DataStore(settings)
        self.espn = EspnFeed()
        self._sleeper_cache: dict[str, tuple[float, SleeperLeague]] = {}
        self._id_map: pl.DataFrame | None = None

    def sleeper_league(self, league_id: str, ttl_seconds: int = 300) -> SleeperLeague:
        hit = self._sleeper_cache.get(league_id)
        if hit and time.time() - hit[0] < ttl_seconds:
            return hit[1]
        league = load_sleeper_league(league_id, self.id_map())
        self._sleeper_cache[league_id] = (time.time(), league)
        return league

    # ------------------------------------------------------------ helpers
    def season(self, season: int | None) -> int:
        return season or self.store.current_season()

    def to_records(self, df: pl.DataFrame, limit: int | None) -> list[dict[str, Any]]:
        if limit:
            df = df.head(limit)
        return df.to_dicts()

    def weekly_points(self, season: int, scoring: dict[str, float]) -> pl.DataFrame:
        key = {"scoring": scoring}
        return self.store.derived(
            "weekly_points",
            season,
            key,
            lambda: metrics.weekly_fantasy_points(self.store.player_stats(season), scoring),
        )

    def projections(self, season: int, scoring: dict[str, float], through_week: int | None = None) -> pl.DataFrame:
        return self.store.derived(
            "projections",
            season,
            {"scoring": scoring, "through_week": through_week},
            lambda: gm.player_projections(self.weekly_points(season, scoring), through_week=through_week),
        )

    def with_full_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """pbp only carries abbreviated names (A.Brown); swap in nflverse display names."""
        if "player_id" not in df.columns:
            return df
        names = self.store.players().select(pl.col("gsis_id").alias("player_id"), pl.col("display_name").alias("_full")).unique("player_id")
        return df.join(names, on="player_id", how="left").with_columns(
            pl.coalesce(["_full", "player"]).alias("player"),
        ).drop("_full")

    def id_map(self) -> pl.DataFrame:
        if self._id_map is None:
            import nflreadpy as nfl

            self._id_map = nfl.load_ff_playerids().select(["sleeper_id", "gsis_id", "espn_id", "name"])
        return self._id_map


class LeagueContext(BaseModel):
    """How to find rosters. Provide a Sleeper league id (auto-import) or explicit rosters."""

    sleeper_league_id: str | None = Field(None, description="Sleeper league id; rosters, scoring and waiver settings are imported automatically")
    my_team: str | None = Field(None, description="My team/owner display name inside the Sleeper league (required with sleeper_league_id)")
    my_roster: list[str] | None = Field(None, description="Player names or gsis ids on my roster (ESPN/Yahoo or manual)")
    rival_rosters: dict[str, list[str]] | None = Field(None, description="Owner name -> player names/ids for other teams (manual mode)")
    available_players: list[str] | None = Field(None, description="Free agents (names/ids). Omit to treat every unrostered player as available")
    league_size: int = Field(12, ge=2, le=32)
    scoring: str | dict[str, float] | None = Field(None, description="Overrides imported scoring")


class LineupPlayerIn(BaseModel):
    name: str
    position: Literal["QB", "RB", "WR", "TE", "K", "DST"]
    team: str
    projection: float | None = Field(None, description="Override; otherwise derived from season data")
    floor: float | None = None
    ceiling: float | None = None
    opponent: str | None = None
    locked: bool = False
    excluded: bool = False


def build_server(services: Services) -> FastMCP:
    mcp = FastMCP(
        "nfl-metrics",
        instructions=(
            "Advanced NFL efficiency/usage metrics from nflverse (EPA/play, CPOE, air yards, "
            "target share, route participation, red-zone touches, snap counts) plus fantasy GM "
            "tools (injury pivots, waiver/FAAB, trade scan, game environments, MILP lineup "
            "optimizer). Data is cached locally as parquet; first call for a season downloads it."
        ),
        host=services.settings.host,
        port=services.settings.port,
        stateless_http=True,
        json_response=False,
        # Requests arrive via a Cloudflare Tunnel with the public hostname in Host, so the
        # SDK's localhost-only DNS-rebinding check would reject them; the bearer middleware
        # is the access control here.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    store = services.store

    # ------------------------------------------------------------ metrics
    @mcp.tool()
    def get_qb_efficiency(
        season: Season = None,
        week: Week = None,
        start_week: int | None = None,
        end_week: int | None = None,
        team: str | None = None,
        player: str | None = None,
        min_dropbacks: int = 10,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        """QB efficiency: EPA per dropback, EPA per pass, CPOE, air yards per attempt, success rate, sack rate."""
        s = services.season(season)
        weeks = (start_week, end_week) if start_week and end_week else None
        df = store.derived(
            "qb_efficiency",
            s,
            {"week": week, "weeks": weeks, "min": min_dropbacks},
            lambda: services.with_full_names(metrics.qb_efficiency(store.scan_pbp(s), week=week, weeks=weeks, min_dropbacks=min_dropbacks)),
        )
        return services.to_records(_filter(df, team, player), limit)

    @mcp.tool()
    def get_receiver_usage(
        season: Season = None,
        week: Week = None,
        start_week: int | None = None,
        end_week: int | None = None,
        team: str | None = None,
        player: str | None = None,
        min_targets: int = 5,
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        """Target share, air yards, air-yards share, aDOT, WOPR, RACR, EPA/target, red-zone and deep targets for pass catchers."""
        s = services.season(season)
        weeks = (start_week, end_week) if start_week and end_week else None
        df = store.derived(
            "receiver_usage",
            s,
            {"week": week, "weeks": weeks, "min": min_targets},
            lambda: services.with_full_names(metrics.receiver_usage(store.scan_pbp(s), week=week, weeks=weeks, min_targets=min_targets)),
        )
        return services.to_records(_filter(df, team, player), limit)

    @mcp.tool()
    def get_red_zone_usage(
        season: Season = None,
        week: Week = None,
        start_week: int | None = None,
        end_week: int | None = None,
        team: str | None = None,
        player: str | None = None,
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        """Red-zone (<=20 yd), inside-10 and inside-5 carries + targets per player with team red-zone touch share."""
        s = services.season(season)
        weeks = (start_week, end_week) if start_week and end_week else None
        df = store.derived(
            "red_zone",
            s,
            {"week": week, "weeks": weeks},
            lambda: services.with_full_names(metrics.red_zone_usage(store.scan_pbp(s), week=week, weeks=weeks)),
        )
        return services.to_records(_filter(df, team, player), limit)

    @mcp.tool()
    def get_snap_counts(
        season: Season = None,
        week: Week = None,
        start_week: int | None = None,
        end_week: int | None = None,
        team: str | None = None,
        player: str | None = None,
        positions: list[str] | None = None,
        limit: int = 80,
    ) -> list[dict[str, Any]]:
        """Offensive/defensive snap counts and snap percentage (avg/min/max) per player. Defaults to offensive skill positions."""
        s = services.season(season)
        weeks = (start_week, end_week) if start_week and end_week else None
        pos = positions or ["QB", "RB", "WR", "TE"]
        df = store.derived(
            "snaps",
            s,
            {"week": week, "weeks": weeks, "pos": pos},
            lambda: metrics.snap_share(store.snap_counts(s), week=week, weeks=weeks, positions=pos),
        )
        return services.to_records(_filter(df, team, player), limit)

    @mcp.tool()
    def get_route_participation(
        season: Season = None,
        week: Week = None,
        start_week: int | None = None,
        end_week: int | None = None,
        team: str | None = None,
        player: str | None = None,
        min_routes: int = 10,
        limit: int = 60,
    ) -> dict[str, Any]:
        """Routes run, route participation % of team dropbacks, targets per route run (from nflverse participation, 2016-latest published season)."""
        s = services.season(season)
        weeks = (start_week, end_week) if start_week and end_week else None
        try:
            df = store.derived(
                "routes",
                s,
                {"week": week, "weeks": weeks, "min": min_routes},
                lambda: metrics.route_participation(
                    store.participation(s), store.scan_pbp(s), store.players(), week=week, weeks=weeks, min_routes=min_routes
                ),
            )
        except Exception as exc:  # participation lags a season behind
            return {"available": False, "season": s, "error": str(exc), "hint": "nflverse participation data may not exist yet for this season; try the prior season."}
        return {"available": True, "season": s, "rows": services.to_records(_filter(df, team, player), limit)}

    @mcp.tool()
    def get_player_usage_profile(player: str, season: Season = None, week: Week = None) -> dict[str, Any]:
        """One-call usage dossier for a player: efficiency, receiving usage, red-zone touches, snaps and routes."""
        s = services.season(season)
        out: dict[str, Any] = {"player": player, "season": s, "week": week}
        out["qb_efficiency"] = _filter(
            store.derived("qb_efficiency", s, {"week": week, "weeks": None, "min": 1}, lambda: services.with_full_names(metrics.qb_efficiency(store.scan_pbp(s), week=week, min_dropbacks=1))),
            None,
            player,
        ).to_dicts()
        out["receiving"] = _filter(
            store.derived("receiver_usage", s, {"week": week, "weeks": None, "min": 1}, lambda: services.with_full_names(metrics.receiver_usage(store.scan_pbp(s), week=week, min_targets=1))),
            None,
            player,
        ).to_dicts()
        out["red_zone"] = _filter(store.derived("red_zone", s, {"week": week, "weeks": None}, lambda: services.with_full_names(metrics.red_zone_usage(store.scan_pbp(s), week=week))), None, player).to_dicts()
        out["snaps"] = _filter(
            store.derived("snaps", s, {"week": week, "weeks": None, "pos": ["QB", "RB", "WR", "TE"]}, lambda: metrics.snap_share(store.snap_counts(s), week=week, positions=["QB", "RB", "WR", "TE"])),
            None,
            player,
        ).to_dicts()
        try:
            out["routes"] = _filter(
                store.derived("routes", s, {"week": week, "weeks": None, "min": 1}, lambda: metrics.route_participation(store.participation(s), store.scan_pbp(s), store.players(), week=week, min_routes=1)),
                None,
                player,
            ).to_dicts()
        except Exception as exc:
            out["routes"] = {"available": False, "error": str(exc)}
        return out

    # ------------------------------------------------------------ GM suite
    def _resolve_league(ctx: LeagueContext, season: int, through_week: int | None) -> dict[str, Any]:
        scoring_dict = gm.resolve_scoring(ctx.scoring) if ctx.scoring else None
        rosters: dict[str, set[str]] = {}
        my_ids: set[str] = set()
        waiver = gm.WaiverSettings(league_size=ctx.league_size)
        slots = dict(gm.DEFAULT_ROSTER_SLOTS)
        notes: list[str] = []
        if ctx.sleeper_league_id:
            league = services.sleeper_league(ctx.sleeper_league_id)
            if scoring_dict is None:
                scoring_dict = league.scoring
            rosters = league.rosters
            slots = league.slots() or slots
            if not ctx.my_team or ctx.my_team not in rosters:
                raise ValueError(f"my_team must be one of {sorted(rosters)}")
            my_ids = rosters[ctx.my_team]
            waiver = gm.WaiverSettings(
                waiver_type=league.waiver_type,  # type: ignore[arg-type]
                faab_budget=league.faab_budget,
                faab_remaining=league.faab_remaining.get(ctx.my_team),
                waiver_position=league.waiver_positions.get(ctx.my_team),
                league_size=league.league_size,
                weeks_remaining=max(1, 17 - (through_week or store.current_week())),
            )
            if league.unresolved_sleeper_ids:
                notes.append(f"{len(league.unresolved_sleeper_ids)} Sleeper player ids had no gsis mapping and were skipped")
        scoring_dict = scoring_dict or gm.DEFAULT_SCORING
        proj = services.projections(season, scoring_dict, through_week)
        if ctx.my_roster:
            my_ids, unresolved = resolve_players(ctx.my_roster, proj)
            if unresolved:
                notes.append(f"unresolved my_roster entries: {unresolved}")
        if ctx.rival_rosters:
            for owner, names in ctx.rival_rosters.items():
                ids, unresolved = resolve_players(names, proj)
                rosters[owner] = ids
                if unresolved:
                    notes.append(f"unresolved entries for {owner}: {unresolved}")
        if not my_ids:
            raise ValueError("Could not determine my roster: pass sleeper_league_id+my_team or my_roster")
        available: set[str] | None = None
        if ctx.available_players:
            available, unresolved = resolve_players(ctx.available_players, proj)
            if unresolved:
                notes.append(f"unresolved available_players: {unresolved}")
        all_rostered = (set().union(*rosters.values()) if rosters else set()) | my_ids
        complete_picture = bool(ctx.sleeper_league_id) and len(rosters) >= waiver.league_size
        if available is None:
            available = set(proj["player_id"].to_list()) - all_rostered
            if not complete_picture:
                notes.append("available_players not supplied and rosters are incomplete: every player not on a listed roster is treated as a free agent")
        starters = gm.starters_from_slots(slots)
        levels = gm.replacement_levels(proj, league_size=waiver.league_size, starters_per_team=starters)
        return {
            "projections": proj,
            "starters_per_team": starters,
            "scoring": scoring_dict,
            "my_ids": my_ids,
            "rosters": {k: v for k, v in rosters.items() if v != my_ids},
            "available": available,
            "levels": levels,
            "waiver": waiver,
            "slots": slots,
            "notes": notes,
        }

    @mcp.tool()
    def check_injury_leverage(
        league: LeagueContext,
        season: Season = None,
        week: Week = None,
        statuses: list[str] | None = None,
        include_questionable: bool = False,
    ) -> dict[str, Any]:
        """Pull pregame inactives/injury designations (ESPN game summaries, which mirror the official 90-minute inactives on game day, merged with nflverse injury reports) and produce emergency pivot recommendations with replacement values for my roster."""
        s = services.season(season)
        w = week or store.current_week()
        wanted = {x.lower() for x in statuses} if statuses else set(gm.INACTIVE_STATUSES) | ({"questionable"} if include_questionable else set())
        reports: list[gm.InjuryReport] = []
        sources: list[str] = []
        freshness: dict[str, Any] = {"official_inactives": False}
        is_current = s == store.current_season() and w == store.current_week()
        if is_current:
            # ESPN's game summary reflects the live injury list; on game day it carries the
            # official 90-minute inactives as status 'Out'. For past weeks it is not historical.
            reports = services.espn.injuries(s, w, wanted)
            if reports:
                sources.append("espn-game-summary")
                states = services.espn.game_states(s, w)
                freshness = {
                    "official_inactives": any(st in ("in", "post") or st == "pre-inactives" for st in states.values()),
                    "games": states,
                    "fetched_at": services.espn.last_fetch_iso(),
                    "note": "Official inactives are published ~90 minutes before kickoff; before that these are practice-report designations.",
                }
        else:
            freshness["note"] = "Requested week is not the current week; only nflverse historical injury reports are used."
        reports = [r for r in reports if not r.position or r.position in gm.SKILL_POSITIONS or r.position in ("K", "DST")]
        try:
            inj = store.injuries(s).filter(pl.col("week") == w)
            seen = {(r.player.lower(), r.team) for r in reports}
            for row in inj.iter_rows(named=True):
                status = (row.get("report_status") or "").strip()
                if not status or status.lower() not in wanted:
                    continue
                if ((row.get("full_name") or "").lower(), row.get("team")) in seen:
                    continue
                reports.append(
                    gm.InjuryReport(
                        player=row.get("full_name") or "",
                        player_id=row.get("gsis_id"),
                        team=row.get("team") or "",
                        position=row.get("position") or "",
                        status=status,
                        detail=row.get("report_primary_injury"),
                        source="nflverse",
                    )
                )
            sources.append("nflverse-injuries")
        except Exception as exc:
            log.warning("nflverse injuries unavailable: %s", exc)
        resolved = _resolve_league(league, s, through_week=w - 1 if w > 1 else None)
        result = gm.injury_leverage(reports, resolved["projections"], resolved["my_ids"], resolved["available"], resolved["levels"])
        result.update({"season": s, "week": w, "sources": sources, "freshness": freshness, "notes": resolved["notes"], "statuses_considered": sorted(wanted)})
        return result

    @mcp.tool()
    def get_waiver_recommendations(
        league: LeagueContext,
        season: Season = None,
        through_week: Week = None,
        waiver_type: Literal["faab", "rolling", "reverse_standings", "auto"] = "auto",
        faab_budget: float | None = None,
        faab_remaining: float | None = None,
        waiver_position: int | None = None,
        weeks_remaining: int | None = None,
        max_targets: int = 10,
    ) -> dict[str, Any]:
        """Waiver targets ranked by value over replacement. FAAB leagues get recommended dollar bids and ranges; rolling/reverse-standings leagues get priority tiers and whether to burn waiver position. Includes drop candidates and add/drop pairs."""
        s = services.season(season)
        resolved = _resolve_league(league, s, through_week)
        ws: gm.WaiverSettings = resolved["waiver"]
        if waiver_type != "auto":
            ws.waiver_type = waiver_type
        if faab_budget is not None:
            ws.faab_budget = faab_budget
        if faab_remaining is not None:
            ws.faab_remaining = faab_remaining
        if waiver_position is not None:
            ws.waiver_position = waiver_position
        if weeks_remaining is not None:
            ws.weeks_remaining = weeks_remaining
        out = gm.waiver_recommendations(resolved["projections"], resolved["my_ids"], resolved["available"], resolved["levels"], ws, max_targets=max_targets)
        out.update({"season": s, "scoring": resolved["scoring"], "notes": resolved["notes"]})
        return out

    @mcp.tool()
    def scan_trade_opportunities(
        league: LeagueContext,
        season: Season = None,
        through_week: Week = None,
        max_proposals: int = 10,
        min_gain: float = 0.5,
    ) -> dict[str, Any]:
        """Map my positional surpluses to rivals' deficits using rest-of-season value-over-replacement baselines and propose 1-for-1 and 2-for-1 trades both sides gain from."""
        s = services.season(season)
        resolved = _resolve_league(league, s, through_week)
        if not resolved["rosters"]:
            raise ValueError("Trade scanning needs rival rosters: pass sleeper_league_id or rival_rosters")
        vor = gm.value_over_replacement(resolved["projections"], resolved["levels"])
        out = gm.scan_trades(vor, resolved["my_ids"], resolved["rosters"], starters_per_team=resolved["starters_per_team"], max_proposals=max_proposals, min_gain=min_gain)
        out.update({"season": s, "replacement_levels": resolved["levels"], "notes": resolved["notes"]})
        return out

    @mcp.tool()
    def get_game_environments(season: Season = None, week: Week = None, team: str | None = None, include_live_odds: bool = True) -> dict[str, Any]:
        """Point spreads, over/unders, implied team totals, weather/roof and a game-script read for each game in a week (nflverse schedule lines, refreshed with ESPN odds when available)."""
        s = services.season(season)
        w = week or store.current_week()
        games = gm.game_environments(store.schedules(s), w, team)
        live = services.espn.odds(s, w) if include_live_odds else {}
        for g in games:
            key = f"{g['away_team']}@{g['home_team']}"
            if key in live:
                g["live_odds"] = live[key]
                ou = live[key].get("over_under")
                if ou is not None and g.get("total_line") != ou:
                    g["total_line_source"] = "espn-live"
                    g["total_line"] = ou
                    if g.get("spread_line") is not None:
                        g["home_implied_total"] = round(ou / 2 + g["spread_line"] / 2, 2)
                        g["away_implied_total"] = round(ou / 2 - g["spread_line"] / 2, 2)
        return {"season": s, "week": w, "games": games, "sources": ["nflverse-schedules"] + (["espn-odds"] if live else [])}

    @mcp.tool()
    def optimize_lineup(
        roster: list[LineupPlayerIn] | None = None,
        league: LeagueContext | None = None,
        season: Season = None,
        week: Week = None,
        scoring: ScoringArg = None,
        slots: dict[str, int] | None = None,
        objective: Literal["projection", "floor", "ceiling"] = "projection",
        variance_weight: float = 0.0,
        stack_bonus: float = 1.5,
        bring_back_bonus: float = 0.0,
        max_from_team: int | None = None,
    ) -> dict[str, Any]:
        """MILP lineup optimizer (HiGHS). Supports custom scoring, roster slots incl. FLEX/SUPER_FLEX, QB + pass-catcher stacking bonus, bring-back stacks, and a floor-vs-ceiling toggle (objective + variance_weight: negative = safer, positive = chase upside)."""
        s = services.season(season)
        scoring_dict = gm.resolve_scoring(scoring)
        use_slots = slots or dict(gm.DEFAULT_ROSTER_SLOTS)
        players: list[gm.LineupPlayer] = []
        notes: list[str] = []
        proj_df: pl.DataFrame | None = None
        if league is not None:
            resolved = _resolve_league(league, s, through_week=(week - 1) if week and week > 1 else None)
            proj_df = resolved["projections"]
            if not scoring:
                scoring_dict = resolved["scoring"]
                proj_df = services.projections(s, scoring_dict, (week - 1) if week and week > 1 else None)
            use_slots = slots or resolved["slots"]
            mine = proj_df.filter(pl.col("player_id").is_in(list(resolved["my_ids"])))
            for r in mine.iter_rows(named=True):
                players.append(gm.LineupPlayer(r["player_id"], r["player"], r["position"], r["team"] or "", float(r["projection"]), float(r["floor"]), float(r["ceiling"]), float(r["std"])))
            notes.extend(resolved["notes"])
        if roster:
            if proj_df is None:
                proj_df = services.projections(s, scoring_dict, (week - 1) if week and week > 1 else None)
            for p in roster:
                ids, _ = resolve_players([p.name], proj_df)
                row = proj_df.filter(pl.col("player_id").is_in(list(ids))).to_dicts()[0] if ids else None
                players.append(
                    gm.LineupPlayer(
                        player_id=(row or {}).get("player_id", p.name),
                        name=p.name,
                        position=p.position,
                        team=p.team.upper(),
                        projection=p.projection if p.projection is not None else float((row or {}).get("projection", 0.0)),
                        floor=p.floor if p.floor is not None else (row or {}).get("floor"),
                        ceiling=p.ceiling if p.ceiling is not None else (row or {}).get("ceiling"),
                        std=(row or {}).get("std"),
                        opponent=p.opponent,
                        locked=p.locked,
                        excluded=p.excluded,
                    )
                )
                if row is None and p.projection is None:
                    notes.append(f"{p.name}: no season data found; projection defaulted to 0")
        if not players:
            raise ValueError("Provide roster players and/or a league context")
        # attach opponents from the schedule so bring-back stacks work
        if week:
            try:
                sched = store.schedules(s).filter(pl.col("week") == week)
                opp = {}
                for g in sched.iter_rows(named=True):
                    opp[g["home_team"]] = g["away_team"]
                    opp[g["away_team"]] = g["home_team"]
                for p in players:
                    if not p.opponent:
                        p.opponent = opp.get(p.team)
                    if p.team and p.team not in opp:
                        notes.append(f"{p.name} ({p.team}) is on bye in week {week}")
                        p.projection, p.floor, p.ceiling = 0.0, 0.0, 0.0
            except Exception as exc:
                notes.append(f"schedule unavailable for bye/opponent detection: {exc}")
        # drop slots nobody on the roster can fill (e.g. K/DEF when only skill players are known)
        fillable = {}
        for slot, n in use_slots.items():
            allowed = gm.FLEX_ELIGIBLE.get(slot, (slot,))
            if any(p.position in allowed and not p.excluded for p in players):
                fillable[slot] = n
            else:
                notes.append(f"slot {slot} skipped: no eligible players supplied")
        use_slots = fillable
        req = gm.LineupRequest(
            players=players,
            slots=use_slots,
            objective=objective,
            variance_weight=variance_weight,
            stack_bonus=stack_bonus,
            bring_back_bonus=bring_back_bonus,
            max_from_team=max_from_team,
        )
        out = gm.optimize_lineup(req)
        out.update({"season": s, "week": week, "scoring": scoring_dict, "slots": use_slots, "notes": notes})
        return out

    # ------------------------------------------------------------ cache ops
    @mcp.tool()
    def get_cache_status() -> dict[str, Any]:
        """List cached parquet files, sizes and ages plus TTL settings."""
        return store.cache_summary()

    @mcp.tool()
    def refresh_season_data(season: Season = None, clear_derived: bool = True) -> dict[str, Any]:
        """Force re-download of a season's play-by-play and drop derived aggregates so the next queries recompute."""
        s = services.season(season)
        path = store.ensure_pbp(s, force=True)
        removed = store.clear_derived(s) if clear_derived else 0
        return {"season": s, "pbp_file": str(path), "bytes": path.stat().st_size, "derived_removed": removed}

    return mcp


def _filter(df: pl.DataFrame, team: str | None, player: str | None) -> pl.DataFrame:
    if team and "team" in df.columns:
        df = df.filter(pl.col("team") == team.upper())
    if player and "player" in df.columns:
        target = normalize_name(player)
        df = df.filter(pl.col("player").map_elements(lambda x: normalize_name(x) if x else "", return_dtype=pl.Utf8).str.contains(target, literal=True))
    return df


class BearerAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str, public_paths: tuple[str, ...] = ("/health", "/healthz")):
        super().__init__(app)
        self.token = token
        self.public_paths = public_paths

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.public_paths:
            return await call_next(request)
        header = request.headers.get("authorization", "")
        supplied = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if not supplied or not hmac.compare_digest(supplied, self.token):
            return JSONResponse(
                {"error": "unauthorized", "error_description": "Bearer token required"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="nfl-metrics"'},
            )
        return await call_next(request)


def build_app(settings: Settings, services: Services | None = None) -> Starlette:
    services = services or Services(settings)
    mcp = build_server(services)
    inner = mcp.sse_app() if settings.transport == "sse" else mcp.streamable_http_app()

    async def health(_: Request) -> Response:
        return JSONResponse(
            {
                "status": "healthy",
                "service": "nfl-metrics",
                "transport": settings.transport,
                "endpoint": "/sse" if settings.transport == "sse" else "/mcp",
                "current_season": services.store.current_season(),
                "current_week": services.store.current_week(),
            }
        )

    app = Starlette(
        routes=[Route("/health", health), Route("/healthz", health), Mount("/", app=inner)],
        lifespan=inner.router.lifespan_context,  # streamable-http session manager needs it
    )
    app.add_middleware(BearerAuthMiddleware, token=settings.token)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"[nfl-metrics] {exc}", file=sys.stderr)
        sys.exit(1)
    app = build_app(settings)
    log.info(
        "nfl-metrics MCP listening on http://%s:%s%s (transport=%s, data_dir=%s)",
        settings.host,
        settings.port,
        "/sse" if settings.transport == "sse" else "/mcp",
        settings.transport,
        settings.data_dir,
    )
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()

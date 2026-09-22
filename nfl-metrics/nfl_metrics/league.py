"""League roster import + player-name resolution.

Sleeper's API is public, so a league id is enough to pull every roster, the
scoring settings and waiver configuration. ESPN/Yahoo rosters must be passed in
explicitly (names or gsis ids) because they need private cookies/OAuth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import httpx
import polars as pl

SLEEPER_API = "https://api.sleeper.app/v1"
_SUFFIXES = (" jr", " sr", " ii", " iii", " iv", " v")


def normalize_name(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"[.'`\-]", "", n)
    n = re.sub(r"\s+", " ", n)
    for suf in _SUFFIXES:
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n.strip()


def resolve_players(tokens: list[str], projections: pl.DataFrame) -> tuple[set[str], list[str]]:
    """Map names or gsis ids to player_ids; returns (ids, unresolved)."""
    ids = set(projections["player_id"].to_list())
    by_name: dict[str, list[str]] = {}
    for row in projections.select(["player_id", "player"]).iter_rows():
        if row[1]:
            by_name.setdefault(normalize_name(row[1]), []).append(row[0])
    resolved: set[str] = set()
    unresolved: list[str] = []
    for token in tokens:
        t = token.strip()
        if t in ids:
            resolved.add(t)
            continue
        matches = by_name.get(normalize_name(t), [])
        if len(matches) >= 1:
            resolved.add(matches[0])
        else:
            unresolved.append(token)
    return resolved, unresolved


@dataclass
class SleeperLeague:
    league_id: str
    name: str
    season: int
    scoring: dict[str, float]
    roster_positions: list[str]
    waiver_type: str  # faab | rolling | reverse_standings | none
    faab_budget: float
    league_size: int
    rosters: dict[str, set[str]] = field(default_factory=dict)  # owner display name -> gsis ids
    faab_remaining: dict[str, float] = field(default_factory=dict)
    waiver_positions: dict[str, int] = field(default_factory=dict)
    unresolved_sleeper_ids: list[str] = field(default_factory=list)

    def slots(self) -> dict[str, int]:
        slots: dict[str, int] = {}
        for pos in self.roster_positions:
            if pos in ("BN", "IR", "TAXI"):
                continue
            key = {"FLEX": "FLEX", "SUPER_FLEX": "SUPER_FLEX", "REC_FLEX": "REC_FLEX", "WRRB_FLEX": "WRRB_FLEX"}.get(pos, pos)
            slots[key] = slots.get(key, 0) + 1
        return slots


def sleeper_scoring_to_dict(settings: dict[str, float]) -> dict[str, float]:
    return {
        "pass_yd": settings.get("pass_yd", 0.04),
        "pass_td": settings.get("pass_td", 4),
        "pass_int": settings.get("pass_int", -1),
        "rush_yd": settings.get("rush_yd", 0.1),
        "rush_td": settings.get("rush_td", 6),
        "rec": settings.get("rec", 0),
        "rec_yd": settings.get("rec_yd", 0.1),
        "rec_td": settings.get("rec_td", 6),
        "fum_lost": settings.get("fum_lost", -2),
        "two_pt": settings.get("pass_2pt", 2),
        **({"te_rec_bonus": settings["bonus_rec_te"]} if settings.get("bonus_rec_te") else {}),
    }


def load_sleeper_league(league_id: str, id_map: pl.DataFrame, timeout: float = 20.0) -> SleeperLeague:
    """``id_map`` needs columns sleeper_id, gsis_id (nflreadpy.load_ff_playerids)."""
    with httpx.Client(timeout=timeout, base_url=SLEEPER_API) as client:
        league = client.get(f"/league/{league_id}").json()
        if not league or "league_id" not in league:
            raise ValueError(f"Sleeper league {league_id} not found")
        rosters = client.get(f"/league/{league_id}/rosters").json() or []
        users = client.get(f"/league/{league_id}/users").json() or []

    settings = league.get("settings", {}) or {}
    waiver_type_code = settings.get("waiver_type", 0)
    waiver_type = {0: "rolling", 1: "reverse_standings", 2: "faab"}.get(waiver_type_code, "rolling")
    faab_budget = float(settings.get("waiver_budget", 100) or 100)

    mapping = {
        str(r[0]): str(r[1])
        for r in id_map.select(["sleeper_id", "gsis_id"]).drop_nulls().iter_rows()
    }
    user_names = {u["user_id"]: (u.get("metadata", {}) or {}).get("team_name") or u.get("display_name") or u["user_id"] for u in users}
    out = SleeperLeague(
        league_id=str(league["league_id"]),
        name=league.get("name", league_id),
        season=int(league.get("season", 0) or 0),
        scoring=sleeper_scoring_to_dict(league.get("scoring_settings", {}) or {}),
        roster_positions=list(league.get("roster_positions", []) or []),
        waiver_type=waiver_type,
        faab_budget=faab_budget,
        league_size=int(settings.get("num_teams", len(rosters)) or len(rosters)),
    )
    unresolved: set[str] = set()
    for roster in rosters:
        owner = user_names.get(roster.get("owner_id"), f"roster_{roster.get('roster_id')}")
        gsis: set[str] = set()
        for sid in roster.get("players") or []:
            g = mapping.get(str(sid))
            if g:
                gsis.add(g)
            elif not str(sid).isalpha():  # team defenses are alphabetic codes; skip silently
                unresolved.add(str(sid))
        out.rosters[owner] = gsis
        rs = roster.get("settings", {}) or {}
        out.faab_remaining[owner] = faab_budget - float(rs.get("waiver_budget_used", 0) or 0)
        if rs.get("waiver_position") is not None:
            out.waiver_positions[owner] = int(rs["waiver_position"])
    out.unresolved_sleeper_ids = sorted(unresolved)
    return out

"""Live feeds that are not in nflverse parquet: ESPN's public scoreboard/summary
API for pregame injury/inactive status and current odds.

These endpoints are unauthenticated and undocumented; every call is wrapped so a
feed outage degrades to nflverse data instead of failing the tool.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import httpx

from .gm import InjuryReport

log = logging.getLogger("nfl_metrics.feeds")

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"

# ESPN uses a few abbreviations that differ from nflverse.
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA", "JAX": "JAX"}


def _team(abbr: str | None) -> str:
    if not abbr:
        return ""
    return ESPN_TO_NFLVERSE.get(abbr.upper(), abbr.upper())


class EspnFeed:
    def __init__(self, timeout: float = 20.0, ttl_seconds: int = 300):
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "flaim-nfl-metrics/0.1"})
        self._cache: dict[str, tuple[float, object]] = {}
        self._ttl = ttl_seconds
        self._last_fetch: float | None = None

    def last_fetch_iso(self) -> str | None:
        return datetime.fromtimestamp(self._last_fetch, tz=timezone.utc).isoformat() if self._last_fetch else None

    def _get(self, url: str, params: dict[str, object]) -> dict:
        key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self._ttl:
            return hit[1]  # type: ignore[return-value]
        response = self._client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        self._last_fetch = time.time()
        self._cache[key] = (self._last_fetch, data)
        return data

    def scoreboard(self, season: int, week: int) -> list[dict]:
        data = self._get(ESPN_SCOREBOARD, {"week": week, "seasontype": 2, "dates": season})
        return list(data.get("events", []))

    def game_states(self, season: int, week: int) -> dict[str, str]:
        """'AWAY@HOME' -> 'pre' | 'pre-inactives' (within 90 min of kickoff) | 'in' | 'post'."""
        out: dict[str, str] = {}
        try:
            events = self.scoreboard(season, week)
        except Exception as exc:
            log.warning("ESPN scoreboard unavailable: %s", exc)
            return out
        now = datetime.now(tz=timezone.utc)
        for event in events:
            for comp in event.get("competitions", []):
                teams = {c.get("homeAway"): _team(c.get("team", {}).get("abbreviation")) for c in comp.get("competitors", [])}
                state = str(comp.get("status", {}).get("type", {}).get("state") or "pre")
                if state == "pre":
                    try:
                        kickoff = datetime.fromisoformat(str(comp.get("date") or event.get("date")).replace("Z", "+00:00"))
                        if (kickoff - now).total_seconds() <= 90 * 60:
                            state = "pre-inactives"
                    except ValueError:
                        pass
                out[f"{teams.get('away')}@{teams.get('home')}"] = state
        return out

    def odds(self, season: int, week: int) -> dict[str, dict[str, object]]:
        """Current ESPN odds keyed by 'AWAY@HOME'. Empty for completed games."""
        out: dict[str, dict[str, object]] = {}
        try:
            events = self.scoreboard(season, week)
        except Exception as exc:  # network / schema drift
            log.warning("ESPN scoreboard unavailable: %s", exc)
            return out
        for event in events:
            for comp in event.get("competitions", []):
                teams = {c.get("homeAway"): _team(c.get("team", {}).get("abbreviation")) for c in comp.get("competitors", [])}
                key = f"{teams.get('away')}@{teams.get('home')}"
                odds = (comp.get("odds") or [{}])[0]
                if not odds:
                    continue
                out[key] = {
                    "provider": odds.get("provider", {}).get("name"),
                    "details": odds.get("details"),
                    "over_under": odds.get("overUnder"),
                    "spread": odds.get("spread"),
                    "home_favourite": odds.get("homeTeamOdds", {}).get("favorite"),
                    "status": comp.get("status", {}).get("type", {}).get("description"),
                }
        return out

    def injuries(self, season: int, week: int, statuses: set[str] | None = None) -> list[InjuryReport]:
        """Injury/inactive designations for every game in the week from ESPN game summaries.

        On game day ESPN flips the 90-minute inactives to status 'Out'; earlier in
        the week you get the practice-report designations (Questionable/Doubtful/Out).
        """
        reports: list[InjuryReport] = []
        try:
            events = self.scoreboard(season, week)
        except Exception as exc:
            log.warning("ESPN scoreboard unavailable: %s", exc)
            return reports
        for event in events:
            try:
                summary = self._get(ESPN_SUMMARY, {"event": event["id"]})
            except Exception as exc:
                log.warning("ESPN summary %s unavailable: %s", event.get("id"), exc)
                continue
            for team_block in summary.get("injuries", []) or []:
                team = _team(team_block.get("team", {}).get("abbreviation"))
                for item in team_block.get("injuries", []) or []:
                    status = str(item.get("status") or item.get("type", {}).get("description") or "").strip()
                    if statuses and status.lower() not in statuses:
                        continue
                    athlete = item.get("athlete", {}) or {}
                    detail = item.get("details", {}) or {}
                    reports.append(
                        InjuryReport(
                            player=str(athlete.get("displayName") or ""),
                            player_id=None,
                            team=team,
                            position=str((athlete.get("position") or {}).get("abbreviation") or ""),
                            status=status,
                            detail=" ".join(str(v) for v in [detail.get("type"), detail.get("detail"), detail.get("returnDate") and f"return {detail['returnDate']}"] if v) or None,
                            source="espn",
                        )
                    )
        return reports

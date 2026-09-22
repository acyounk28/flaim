"""Parquet-backed data access tuned for a Raspberry Pi.

Three layers:

1. ``raw/``      - nflverse play-by-play parquet files, streamed to disk with httpx
                   and read with ``polars.scan_parquet`` so only the ~25 columns a
                   metric needs are ever decoded (a full season is ~50k rows x 370
                   columns; projecting keeps peak RSS in the low hundreds of MB).
2. ``nflreadpy/``- nflreadpy's own filesystem cache for the smaller datasets
                   (player stats, snap counts, participation, schedules, injuries,
                   rosters, players).
3. ``derived/``  - small aggregated frames written back as parquet keyed by
                   (name, season, week) so repeated MCP calls are file reads, not
                   recomputation.

Completed seasons never expire; the current season honours the TTLs from
``Settings``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import polars as pl

from .config import Settings

log = logging.getLogger("nfl_metrics.data")

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

PBP_COLUMNS = [
    "game_id",
    "play_id",
    "season",
    "week",
    "season_type",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "play_type",
    "pass",
    "rush",
    "qb_dropback",
    "qb_scramble",
    "sack",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "air_yards",
    "yards_after_catch",
    "yards_gained",
    "yardline_100",
    "epa",
    "qb_epa",
    "cpoe",
    "success",
    "touchdown",
    "pass_touchdown",
    "rush_touchdown",
    "passer_player_id",
    "passer_player_name",
    "receiver_player_id",
    "receiver_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "two_point_attempt",
    "down",
]


class DataStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._configure_nflreadpy()

    # ------------------------------------------------------------------ helpers
    def _configure_nflreadpy(self) -> None:
        from nflreadpy.config import update_config

        update_config(
            cache_mode="filesystem",
            cache_dir=self.settings.nflreadpy_dir,
            cache_duration=int(self.settings.raw_ttl_hours * 3600),
            verbose=False,
            timeout=120,
        )

    @staticmethod
    def current_season() -> int:
        from nflreadpy import get_current_season

        return int(get_current_season())

    @staticmethod
    def current_week() -> int:
        from nflreadpy import get_current_week

        return int(get_current_week())

    def _is_stale(self, path: Path, season: int, ttl_hours: float) -> bool:
        if not path.exists():
            return True
        if season < self.current_season():
            return False
        return (time.time() - path.stat().st_mtime) > ttl_hours * 3600

    # ---------------------------------------------------------------- raw pbp
    def pbp_path(self, season: int) -> Path:
        return self.settings.raw_dir / f"pbp_{season}.parquet"

    def ensure_pbp(self, season: int, force: bool = False) -> Path:
        path = self.pbp_path(season)
        if not force and not self._is_stale(path, season, self.settings.raw_ttl_hours):
            return path
        with self._lock:
            if not force and not self._is_stale(path, season, self.settings.raw_ttl_hours):
                return path
            url = PBP_URL.format(season=season)
            tmp = path.with_suffix(".parquet.part")
            log.info("downloading %s", url)
            with httpx.stream("GET", url, follow_redirects=True, timeout=180) as response:
                if response.status_code == 404:
                    raise FileNotFoundError(f"nflverse has no play-by-play for season {season}")
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in response.iter_bytes(1 << 20):
                        fh.write(chunk)
            tmp.replace(path)
        return path

    def scan_pbp(self, season: int, columns: list[str] | None = None) -> pl.LazyFrame:
        """Lazy frame over one season with column projection pushed into the parquet reader."""
        path = self.ensure_pbp(season)
        lf = pl.scan_parquet(path)
        wanted = columns or PBP_COLUMNS
        available = set(lf.collect_schema().names())
        return lf.select([c for c in wanted if c in available])

    # ------------------------------------------------------------ nflreadpy
    def player_stats(self, season: int) -> pl.DataFrame:
        import nflreadpy as nfl

        return nfl.load_player_stats(season)

    def snap_counts(self, season: int) -> pl.DataFrame:
        import nflreadpy as nfl

        return nfl.load_snap_counts(season)

    def participation(self, season: int) -> pl.DataFrame:
        import nflreadpy as nfl

        return nfl.load_participation(season)

    def schedules(self, season: int) -> pl.DataFrame:
        import nflreadpy as nfl

        return nfl.load_schedules(season)

    def injuries(self, season: int) -> pl.DataFrame:
        import nflreadpy as nfl

        return nfl.load_injuries(season)

    def players(self) -> pl.DataFrame:
        import nflreadpy as nfl

        return nfl.load_players()

    def rosters_weekly(self, season: int) -> pl.DataFrame:
        import nflreadpy as nfl

        return nfl.load_rosters_weekly(season)

    def ff_opportunity(self, season: int) -> pl.DataFrame:
        import nflreadpy as nfl

        return nfl.load_ff_opportunity(season)

    # -------------------------------------------------------------- derived
    def derived(
        self,
        name: str,
        season: int,
        params: dict[str, object],
        compute: Callable[[], pl.DataFrame],
        force: bool = False,
    ) -> pl.DataFrame:
        key = hashlib.sha1(json.dumps({"n": name, "s": season, **params}, sort_keys=True, default=str).encode()).hexdigest()[:16]
        path = self.settings.derived_dir / f"{name}_{season}_{key}.parquet"
        if not force and not self._is_stale(path, season, self.settings.derived_ttl_hours):
            try:
                return pl.read_parquet(path)
            except Exception:  # corrupt cache file - recompute
                path.unlink(missing_ok=True)
        frame = compute()
        tmp = path.with_suffix(".parquet.part")
        frame.write_parquet(tmp, compression="zstd")
        tmp.replace(path)
        return frame

    def cache_summary(self) -> dict[str, object]:
        def listing(directory: Path) -> list[dict[str, object]]:
            items = []
            for p in sorted(directory.glob("*.parquet")):
                st = p.stat()
                items.append({"file": p.name, "bytes": st.st_size, "age_hours": round((time.time() - st.st_mtime) / 3600, 2)})
            return items

        return {
            "data_dir": str(self.settings.data_dir),
            "raw": listing(self.settings.raw_dir),
            "derived": listing(self.settings.derived_dir),
            "nflreadpy_files": len(list(self.settings.nflreadpy_dir.glob("*.parquet"))),
            "raw_ttl_hours": self.settings.raw_ttl_hours,
            "derived_ttl_hours": self.settings.derived_ttl_hours,
        }

    def clear_derived(self, season: int | None = None) -> int:
        removed = 0
        for p in self.settings.derived_dir.glob("*.parquet"):
            if season is None or f"_{season}_" in p.name:
                p.unlink()
                removed += 1
        return removed

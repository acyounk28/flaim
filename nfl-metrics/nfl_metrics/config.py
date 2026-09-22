"""Environment-driven settings for the nfl-metrics MCP server."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


@dataclass(frozen=True)
class Settings:
    token: str
    data_dir: Path
    host: str
    port: int
    transport: str  # "sse" | "streamable-http"
    raw_ttl_hours: float
    derived_ttl_hours: float
    max_seasons_in_memory: int

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def derived_dir(self) -> Path:
        return self.data_dir / "derived"

    @property
    def nflreadpy_dir(self) -> Path:
        return self.data_dir / "nflreadpy"


def load_settings() -> Settings:
    token = _env("NFL_MCP_TOKEN")
    if not token:
        raise ConfigError("NFL_MCP_TOKEN is required (generate with: openssl rand -hex 32)")
    if len(token) < 24:
        raise ConfigError("NFL_MCP_TOKEN must be at least 24 characters")

    transport = (_env("NFL_MCP_TRANSPORT", "sse") or "sse").lower()
    if transport not in {"sse", "streamable-http"}:
        raise ConfigError("NFL_MCP_TRANSPORT must be 'sse' or 'streamable-http'")

    port_raw = _env("PORT", "8800") or "8800"
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise ConfigError(f"PORT must be an integer, got {port_raw!r}") from exc

    settings = Settings(
        token=token,
        data_dir=Path(_env("NFL_DATA_DIR", "/data/nfl") or "/data/nfl"),
        host=_env("HOST", "0.0.0.0") or "0.0.0.0",
        port=port,
        transport=transport,
        raw_ttl_hours=float(_env("NFL_RAW_TTL_HOURS", "12") or "12"),
        derived_ttl_hours=float(_env("NFL_DERIVED_TTL_HOURS", "6") or "6"),
        max_seasons_in_memory=int(_env("NFL_MAX_SEASONS_IN_MEMORY", "1") or "1"),
    )
    for directory in (settings.raw_dir, settings.derived_dir, settings.nflreadpy_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return settings

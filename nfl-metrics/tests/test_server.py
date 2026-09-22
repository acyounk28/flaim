"""HTTP-level tests: auth middleware, health, tool listing over both transports."""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
from starlette.testclient import TestClient

from nfl_metrics.config import ConfigError, Settings, load_settings
from nfl_metrics.server import build_app

TOKEN = "test-token-0123456789abcdef0123456789"
EXPECTED_TOOLS = {
    "get_qb_efficiency",
    "get_receiver_usage",
    "get_red_zone_usage",
    "get_snap_counts",
    "get_route_participation",
    "get_player_usage_profile",
    "check_injury_leverage",
    "get_waiver_recommendations",
    "scan_trade_opportunities",
    "get_game_environments",
    "optimize_lineup",
    "get_cache_status",
    "refresh_season_data",
}


def _settings(tmp_path: Path, transport: str) -> Settings:
    return Settings(
        token=TOKEN,
        data_dir=tmp_path,
        host="127.0.0.1",
        port=8800,
        transport=transport,
        raw_ttl_hours=12,
        derived_ttl_hours=6,
        max_seasons_in_memory=1,
    )


def test_load_settings_requires_long_token(monkeypatch, tmp_path):
    monkeypatch.setenv("NFL_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("NFL_MCP_TOKEN", raising=False)
    with pytest.raises(ConfigError):
        load_settings()
    monkeypatch.setenv("NFL_MCP_TOKEN", "short")
    with pytest.raises(ConfigError):
        load_settings()
    monkeypatch.setenv("NFL_MCP_TOKEN", TOKEN)
    monkeypatch.setenv("NFL_MCP_TRANSPORT", "bogus")
    with pytest.raises(ConfigError):
        load_settings()
    monkeypatch.setenv("NFL_MCP_TRANSPORT", "sse")
    s = load_settings()
    assert s.transport == "sse" and s.port == 8800
    assert (tmp_path / "raw").is_dir() and (tmp_path / "derived").is_dir()


def test_health_is_public_and_mcp_requires_bearer(tmp_path):
    app = build_app(_settings(tmp_path, "sse"))
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "healthy" and body["transport"] == "sse" and body["endpoint"] == "/sse"
        assert body["current_season"] >= 2025

        r = client.post("/messages/", json={})
        assert r.status_code == 401
        assert r.headers["WWW-Authenticate"].startswith("Bearer")
        r = client.post("/messages/", json={}, headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401


class _LiveServer:
    """Real uvicorn server in a thread: long-lived SSE streams can't be driven by TestClient."""

    def __init__(self, app):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        self.url = f"http://127.0.0.1:{self.port}"
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)


def test_sse_endpoint_streams_session_endpoint_and_lists_tools(tmp_path):
    app = build_app(_settings(tmp_path, "sse"))
    with _LiveServer(app) as srv, httpx.Client(base_url=srv.url, timeout=10) as client:
        assert client.get("/sse").status_code == 401
        with client.stream("GET", "/sse", headers={"Authorization": f"Bearer {TOKEN}"}) as r:
            assert r.status_code == 200
            assert r.headers["content-type"].startswith("text/event-stream")
            lines = r.iter_lines()
            assert next(lines).startswith("event: endpoint")
            endpoint = next(lines).split("data:", 1)[1].strip()
            assert endpoint.startswith("/messages/?session_id=")

            headers = {"Authorization": f"Bearer {TOKEN}"}
            init = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
            }
            assert client.post(endpoint, json=init, headers=headers).status_code == 202
            client.post(endpoint, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=headers)
            client.post(endpoint, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, headers=headers)
            messages = []
            for line in lines:
                if line.startswith("data:"):
                    messages.append(json.loads(line[5:]))
                if any(m.get("id") == 2 for m in messages):
                    break
            tools = next(m for m in messages if m.get("id") == 2)["result"]["tools"]
            assert EXPECTED_TOOLS <= {t["name"] for t in tools}


def _mcp_post(client: TestClient, payload: dict, session: str | None = None):
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if session:
        headers["mcp-session-id"] = session
    return client.post("/mcp", json=payload, headers=headers)


def _sse_json(text: str) -> dict:
    for line in text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:].strip())
    raise AssertionError(f"no data frame in {text!r}")


def test_streamable_http_initialize_and_list_tools(tmp_path):
    app = build_app(_settings(tmp_path, "streamable-http"))
    with TestClient(app) as client:
        assert client.get("/health").json()["endpoint"] == "/mcp"
        assert client.post("/mcp", json={}).status_code == 401

        init = _mcp_post(
            client,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "pytest", "version": "0"},
                },
            },
        )
        assert init.status_code == 200, init.text
        session = init.headers.get("mcp-session-id")  # absent: server runs stateless
        result = _sse_json(init.text)["result"]
        assert result["serverInfo"]["name"] == "nfl-metrics"
        assert "tools" in result["capabilities"]

        _mcp_post(client, {"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
        listed = _mcp_post(client, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session)
        tools = _sse_json(listed.text)["result"]["tools"]
        names = {t["name"] for t in tools}
        assert EXPECTED_TOOLS <= names, names - EXPECTED_TOOLS
        lineup = next(t for t in tools if t["name"] == "optimize_lineup")
        props = lineup["inputSchema"]["properties"]
        assert {"roster", "league", "objective", "stack_bonus", "variance_weight", "scoring", "slots"} <= set(props)

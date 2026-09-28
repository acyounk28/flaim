# Self-Hosting Flaim on a Raspberry Pi 5 (Docker + Cloudflare Tunnel)

This guide runs Flaim's MCP tools on your own hardware instead of the hosted
`api.flaim.app` Workers, and publishes them to an MCP client such as Poke through
a Cloudflare Tunnel. Nothing here changes the Cloudflare Worker deployment; the
self-host stack is an additional, opt-in way to run the same tool code.

The stack has two MCP servers:

| Service | Image | Port | Transport | Endpoint | What it does |
|---|---|---|---|---|---|
| `flaim-mcp` | `Dockerfile` (root) | 8790 | Streamable HTTP (SSE responses) | `/mcp` | The existing `fantasy-mcp` gateway + ESPN/Sleeper/Yahoo client Workers, run in-process on Node 24 with a local auth stub and file-backed KV. Rosters, matchups, standings, transactions, available players. |
| `nfl-metrics` | `nfl-metrics/Dockerfile` | 8800 | SSE (default) or Streamable HTTP | `/sse` (or `/mcp`) | Python FastMCP server. nflverse advanced metrics (EPA/play, CPOE, air yards, target share, route participation, red-zone touches, snap counts) plus the autonomous-GM tools (injury leverage, waivers/FAAB, trade scan, game environments, MILP lineup optimizer). Data cached locally as parquet. |
| `cloudflared` | `cloudflare/cloudflared` | – | – | – | Optional tunnel connector (`--profile tunnel`). |

Both images are multi-arch (`linux/arm64`, `linux/amd64`) and run as non-root users.

## 1. Prerequisites on the Pi

- Raspberry Pi 5 (4 GB works; 8 GB recommended if you query several seasons), 64-bit Raspberry Pi OS (Bookworm).
- Docker Engine + Compose plugin:

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
docker compose version
```

- Fast storage. Play-by-play parquet is ~40–60 MB per season; the first query for a season downloads it. An SSD over USB 3 is much faster than an SD card for parquet scans.

## 2. Configure

```bash
git clone https://github.com/acyounk28/flaim.git && cd flaim
cp .env.example .env   # tokens, ESPN cookies, ESPN_LEAGUE_IDS / SLEEPER_LEAGUE_IDS
openssl rand -hex 32   # -> FLAIM_MCP_TOKEN
openssl rand -hex 32   # -> NFL_MCP_TOKEN
```

### Environment variables (`.env`)

| Variable | Service | Default | Notes |
|---|---|---|---|
| `FLAIM_MCP_TOKEN` | flaim-mcp | (required) | Bearer token clients send to `/mcp`. Min 24 chars. Alias: `FLAIM_MCP_AUTH_TOKEN`. |
| `ESPN_LEAGUE_IDS` | flaim-mcp | – | Comma-separated ESPN league ids, optionally `leagueId:teamId`. Sport/season from `ESPN_SPORT` (default `football`) and `ESPN_SEASON_YEAR` (default: current season). |
| `SLEEPER_LEAGUE_IDS` | flaim-mcp | – | Comma-separated Sleeper league ids, optionally `leagueId:rosterId`. `SLEEPER_SPORT` / `SLEEPER_SEASON_YEAR` as above; `SLEEPER_USERNAME` optional. |
| `ESPN_SWID`, `ESPN_S2` | flaim-mcp | – | ESPN cookies, needed for ESPN tools. Aliases: `SWID`, `espn_s2`. Missing or placeholder values do **not** stop startup: ESPN tools return a credentials error, Sleeper keeps working. May also live under `espn.swid`/`espn.s2` in `leagues.json`; env wins. |
| `FLAIM_LEAGUES_FILE` | flaim-mcp | `/config/leagues.json` | Optional JSON file (see below). Missing/invalid/example content is logged as a warning and ignored. `FLAIM_LEAGUES_JSON` (inline JSON) is also accepted. |
| `INTERNAL_SERVICE_TOKEN` | flaim-mcp | random | Token used between the in-process Worker apps. Leave blank. |
| `FLAIM_CACHE_DIR` | flaim-mcp | `/data/cache` | File-backed KV (league metadata cache). |
| `PORT` / `HOST` | both | `8790` / `8800`, `0.0.0.0` | Set by Compose. flaim-mcp also honours `FLAIM_MCP_PORT` / `FLAIM_MCP_HOST` (they take precedence). |
| `NFL_MCP_TOKEN` | nfl-metrics | (required) | Bearer token for `/sse` and `/messages`. Min 24 chars. |
| `NFL_MCP_TRANSPORT` | nfl-metrics | `sse` | `sse` → `/sse`; `streamable-http` → `/mcp`. |
| `NFL_DATA_DIR` | nfl-metrics | `/data/nfl` | Parquet cache root (`raw/`, `derived/`, `nflreadpy/`). |
| `NFL_RAW_TTL_HOURS` | nfl-metrics | `12` | Refresh interval for the *current* season's raw files. Completed seasons never re-download. |
| `NFL_DERIVED_TTL_HOURS` | nfl-metrics | `6` | Refresh interval for cached aggregates of the current season. |
| `NFL_MAX_SEASONS_IN_MEMORY` | nfl-metrics | `1` | Cached LazyFrame handles; keep at 1 on a Pi. |
| `POLARS_MAX_THREADS` | nfl-metrics | `3` | Leave one core for the OS and tunnel. |
| `FLAIM_BIND` | compose | `127.0.0.1` | Host interface for published ports. Only change to `0.0.0.0` on a trusted LAN. |
| `FLAIM_PORT`, `NFL_PORT` | compose | `8790`, `8800` | Host ports. |
| `FLAIM_MEM_LIMIT`, `NFL_MEM_LIMIT` | compose | `512m`, `1536m` | Container memory limits. |
| `CLOUDFLARE_TUNNEL_TOKEN` | cloudflared | – | From Zero Trust → Networks → Tunnels. |

### Optional league file (`config/leagues.json`)

Most setups need only `.env`. The file adds what env vars cannot express:
league/team names, mixed sports or seasons, and explicit default leagues. It
is merged with the env lists (file entries win on the same league) and is
git-ignored, so a local copy on the Pi never conflicts with `git pull`. The
gateway treats every league as belonging to the single self-host operator;
there are no user accounts. Example with **2 ESPN leagues and 3 Sleeper
leagues** (this is `config/leagues.example.json`; its ids are recognised as
placeholders and skipped if you copy it unchanged):

```json
{
  "espn": {
    "leagues": [
      { "leagueId": "123456",    "sport": "football", "seasonYear": 2026, "teamId": "4",  "leagueName": "Sunday Night Degenerates", "teamName": "Pi Powered" },
      { "leagueId": "987654321", "sport": "football", "seasonYear": 2026, "teamId": "10", "leagueName": "Work League (ESPN)",       "teamName": "Container Ship" }
    ]
  },
  "sleeper": {
    "username": "your_sleeper_username",
    "leagues": [
      { "leagueId": "1124838275649073152", "sport": "football", "seasonYear": 2026, "rosterId": 3, "leagueName": "Dynasty Forever", "recurringLeagueId": "dynasty-forever" },
      { "leagueId": "1124910048231305216", "sport": "football", "seasonYear": 2026, "rosterId": 7, "leagueName": "Family Redraft" },
      { "leagueId": "1125000934567890123", "sport": "football", "seasonYear": 2026, "rosterId": 1, "leagueName": "Best Ball Buddies" }
    ]
  },
  "preferences": {
    "defaultSport": "football",
    "defaultFootball": { "platform": "sleeper", "leagueId": "1124838275649073152", "seasonYear": 2026 }
  }
}
```

Field notes:

- **ESPN** — `leagueId` is in the league URL (`leagueId=`), `teamId` is the `teamId=` in your team URL. Find `SWID` and `espn_s2` in browser cookies for `espn.com` while logged in; `SWID` keeps its braces. ESPN sports: `football`, `baseball`, `basketball`, `hockey`.
- **Sleeper** — `leagueId` is the long number in `sleeper.com/leagues/<id>`. `rosterId` is your roster slot (1-based; `GET https://api.sleeper.app/v1/league/<id>/rosters` lists them with `owner_id`). Sleeper's API is public, so no credentials are needed. Sleeper sports: `football`, `basketball`.
- `preferences` sets which league is used when a tool call omits `platform`/`leagueId`.

## 3. Build and run

```bash
docker compose up -d --build           # builds arm64 images natively on the Pi
docker compose ps
curl -s localhost:8790/health
curl -s localhost:8800/health
```

`flaim-mcp`'s `/health` reports league counts, per-provider status
(`espn: ready | missing-credentials`) and any configuration `warnings`; the
same warnings are printed at startup as `[selfhost] warning: ...`. The
container only refuses to start when the bearer token is missing/placeholder
or the port is invalid.

First build on a Pi 5 takes ~6–10 minutes (Node workspace install + Python wheels; all
dependencies ship aarch64 wheels, so nothing compiles).

### Cross-building on a workstation (optional)

```bash
docker buildx create --use --name flaim
docker buildx build --platform linux/arm64,linux/amd64 -t ghcr.io/<you>/flaim-mcp:latest --push .
docker buildx build --platform linux/arm64,linux/amd64 -t ghcr.io/<you>/flaim-nfl-metrics:latest --push nfl-metrics
```

Then replace the `build:` blocks in `docker-compose.yml` with the pushed `image:` tags.

### Smoke-test MCP locally

```bash
# flaim-mcp (Streamable HTTP)
curl -sN localhost:8790/mcp -H "Authorization: Bearer $FLAIM_MCP_TOKEN" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'

# nfl-metrics (SSE): open the stream, note the /messages?session_id=... endpoint event
curl -sN localhost:8800/sse -H "Authorization: Bearer $NFL_MCP_TOKEN"
```

Unauthenticated requests return `401`; `/health` and `/healthz` are public.

## 4. Cloudflare Tunnel

1. Zero Trust dashboard → **Networks → Tunnels → Create a tunnel** (Cloudflared connector). Copy the token into `CLOUDFLARE_TUNNEL_TOKEN`.
2. Add **Public hostnames** (the connector runs inside the compose network, so use service names):

   | Hostname | Service |
   |---|---|
   | `fantasy.example.com` | `http://flaim-mcp:8790` |
   | `nfl.example.com` | `http://nfl-metrics:8800` |

   Single-hostname alternative: path `/mcp` → `flaim-mcp:8790`, paths `/sse` and `/messages*` → `nfl-metrics:8800`.
   In **Additional application settings → HTTP Settings** disable "Disable chunked encoding" (leave default) and set **Connection timeout** / **TCP keep-alive** high; SSE streams are long-lived. Cloudflare does not buffer `text/event-stream` responses.
3. Start the connector:

```bash
docker compose --profile tunnel up -d
docker compose logs -f cloudflared     # look for "Registered tunnel connection"
```

No inbound ports are opened on your router; the Pi dials out to Cloudflare.

### Hardening (recommended)

- Keep `FLAIM_BIND=127.0.0.1` so the services are reachable only via the tunnel and localhost.
- Both services already require a bearer token. Additionally put a **Cloudflare Access** service-token or policy in front of the hostnames, or a **WAF rule** allowing only your MCP client's egress IPs, so brute-force traffic never reaches the Pi.
- Rotate `FLAIM_MCP_TOKEN` / `NFL_MCP_TOKEN` by editing `.env` and `docker compose up -d`.
- ESPN cookies (`espn_s2`) expire roughly yearly; refresh them from the browser when ESPN tools start returning auth errors.
- `nfl-metrics` disables FastMCP's DNS-rebinding host check (it must accept the tunnel's public hostname) and relies on the bearer token, so never expose the ports directly without the tunnel/Access layer.

## 5. Connect Poke (or any MCP client)

Add two MCP servers:

| Name | URL | Transport | Header |
|---|---|---|---|
| Flaim Fantasy | `https://fantasy.example.com/mcp` | Streamable HTTP | `Authorization: Bearer <FLAIM_MCP_TOKEN>` |
| NFL Metrics & GM | `https://nfl.example.com/sse` | SSE | `Authorization: Bearer <NFL_MCP_TOKEN>` |

If the client only supports Streamable HTTP, set `NFL_MCP_TRANSPORT=streamable-http` and use `https://nfl.example.com/mcp` instead.

## 6. nfl-metrics tool reference

All tools take `season` (default: current) and optional `week` / `start_week`–`end_week` ranges.

### Advanced metrics

| Tool | Key outputs |
|---|---|
| `get_qb_efficiency` | dropbacks, EPA/play, CPOE, completion %, success rate, aDOT, sack rate (scrambles count as dropbacks) |
| `get_receiver_usage` | targets, target share, air yards, air-yards share, aDOT, WOPR, RACR, catch rate, EPA/target, deep targets |
| `get_route_participation` | routes run, route participation %, targets per route run (nflverse participation data; available 2016–present with lag) |
| `get_red_zone_usage` | red-zone touches / targets, inside-10 and inside-5 touches, RZ TDs, share of team RZ plays |
| `get_snap_counts` | offensive snaps and snap %, by player/week |
| `get_player_usage_profile` | one player's combined snap/route/target/RZ profile |
| `get_cache_status`, `refresh_season_data` | inspect or force-refresh the parquet cache |

### GM tools

Every GM tool accepts a `league` object:

```json
{
  "sleeper_league_id": "1124838275649073152",
  "my_team": "Pi Powered",
  "scoring": "half_ppr"
}
```

With `sleeper_league_id`, rosters, scoring rules, roster slots, waiver type and FAAB budget are imported from Sleeper's public API. For ESPN/Yahoo (or offline), pass `my_roster`, `rival_rosters`, `available_players` and `scoring` manually; the `flaim-mcp` roster tools can supply those lists.

Scoring presets: `ppr`, `half_ppr`, `standard`, `tep`, `superflex_ppr`, or a dict such as `{"rec": 1, "rec_td": 6, "pass_td": 4, "int": -2}`.

| Tool | Behaviour |
|---|---|
| `check_injury_leverage` | Pulls nflverse injury reports and, once the ESPN scoreboard shows games within the ~90-minute pregame window, the official inactives. Flags affected starters, proposes pivots from bench/free agents with replacement values and same-team beneficiaries. `freshness.official_inactives` tells you whether the data is official or still practice-report designations. |
| `get_waiver_recommendations` | Detects `faab` / `rolling` / `reverse_standings` from league settings (or `waiver_type`). FAAB: recommended bid and range vs remaining budget by priority tier. Rolling: claim order ranked by value over replacement and whether spending your position is worth it. Always includes drop candidates. |
| `scan_trade_opportunities` | Computes ROS value-over-replacement baselines per position, maps your positional surpluses to rivals' deficits, and returns 1-for-1 / 2-for-1 proposals only when both starting lineups improve (`min_gain`). |
| `get_game_environments` | Per game: spread, total, moneylines, implied team totals (`total/2 ± spread/2`), roof, surface, weather, kickoff, and a `game_script` classification (pace, favourite/underdog side, confidence). nflverse lines are supplemented with ESPN odds when `include_live_odds` is true; games without a line are flagged rather than guessed. |
| `optimize_lineup` | MILP (SciPy/HiGHS). `slots` e.g. `{"QB":1,"RB":2,"WR":2,"TE":1,"FLEX":1}`; `objective` = `projection` / `floor` / `ceiling`; `variance_weight` shifts toward ceiling (>0) or floor (<0); `stack_bonus` rewards QB + same-team pass catchers; `bring_back_bonus` rewards an opposing-team receiver; `locked` / `excluded` per player; `max_from_team`. Returns starters, bench, objective and solver status. |

Example lineup request:

```json
{
  "roster": [
    {"name": "Josh Allen", "position": "QB", "team": "BUF", "projection": 22.1, "floor": 15, "ceiling": 33},
    {"name": "Khalil Shakir", "position": "WR", "team": "BUF", "projection": 12.4},
    {"name": "Bijan Robinson", "position": "RB", "team": "ATL", "projection": 18.0, "locked": true}
  ],
  "slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
  "scoring": "half_ppr",
  "objective": "ceiling",
  "variance_weight": 0.3,
  "stack_bonus": 1.5
}
```

Omit `projection` and pass `league` + `season` to derive projections from nflverse weekly fantasy points instead.

### Trade value tools

Trade value is a live, standard-PPR (1 pt/reception, 0.1/yd, 6/TD, 4/pass TD) 0–100 index for every QB/RB/WR/TE. It is deliberately **not** affected by league scoring settings: a `league` object is only used for rosters. Each value blends:

- **Production** — season PPG, last-3-games PPG, prior-season PPG and nflverse expected fantasy points (`ff_opportunity`), weighted toward recent form.
- **Usage trend** — offensive snap %, target share, air-yards share and red-zone touch share, season vs last three games. Declines cut value roughly twice as hard as rises add it, and a snap share that halves caps the multiplier (`role_reduced`). Route participation and targets per route run are included when nflverse participation data exists for the season.
- **Depth-chart role** — the latest nflverse team depth chart (`RB2`, `WR3`, …) with a further penalty when the player has been demoted since the snapshot ~3 weeks earlier.
- **Availability** — ESPN designation for the next game (`questionable`, `doubtful`, `out`, IR…), games missed this season, prior-season games missed and injury-report frequency.
- **Market** — FantasyPros weekly PPR positional ECR (via nflverse) blended at 25%; `market_overvalues` / `market_undervalues` flag large model-vs-market gaps.

Model value is value over positional replacement (12-team defaults), so a QB12 is worth far less than an RB12. Every response carries `data_coverage`, `freshness` (source as-of dates, ESPN fetch time, per-source errors) and a per-player `confidence`; a source that fails to load lowers confidence instead of silently changing the number. Set `include_pbp: false` to skip the play-by-play download (red-zone and route metrics become unavailable).

| Tool | Behaviour |
|---|---|
| `get_player_trade_value` | Dossiers for one or more players (names or gsis ids): value, model/market scores, production, usage trend, role, injury risk, flags. |
| `rank_trade_values` | League-wide board, filterable by `position`, `team` and `flag` (e.g. `target_share_declining`, `depth_chart_demoted`, `market_overvalues`). |
| `compare_trade` | Both sides of a trade: consolidation-weighted package values (the best asset counts fully, depth pieces less), gap, verdict. With `league` + `side_a_team`/`side_b_team`, verifies ownership and reports each starting lineup's PPG before/after. |
| `evaluate_trade_proposals` | Batch of `{team_a, team_b, a_gives, b_gives}` proposals ranked by fairness, with ownership and unresolved-player issues when league rosters are supplied. |

## 7. Caching and Pi resource behaviour

- Raw nflverse files are stored once per season under `NFL_DATA_DIR/raw/pbp_<season>.parquet` etc. Completed seasons are immutable; the current season re-downloads after `NFL_RAW_TTL_HOURS`.
- Queries use Polars lazy scans with column projection, so a full-season EPA query touches ~20 of 370+ columns and stays well under 500 MB RSS.
- Derived aggregates (per season/week/filters) are cached as small parquet files under `derived/` with `NFL_DERIVED_TTL_HOURS`.
- Writes are atomic (temp file + rename) so a power cut on the Pi never leaves a corrupt cache. `refresh_season_data` clears derived files.
- The Node gateway's KV cache lives in the `flaim-cache` volume.

## 8. Limitations

- **Single operator.** The self-host gateway has no user accounts, OAuth or Supabase; every configured league belongs to the one token holder. Yahoo OAuth is not implemented in the local auth stub (Yahoo leagues need the hosted service).
- **ESPN cookies are manual.** The Chrome extension flow targets the hosted service; on the Pi you paste `SWID`/`espn_s2` yourself. Without them the server still runs, but ESPN tools report a credentials error.
- **No LLM calls.** Neither container calls an AI model; model choice and cost are decided entirely by the MCP client (Poke, Claude, ChatGPT) you connect.
- **Official inactives** depend on ESPN's public scoreboard/summary JSON, which is unauthenticated and undocumented; before the 90-minute window the tool returns practice-report designations and says so.
- **Betting lines**: nflverse schedule lines can lag or be missing for the current week; ESPN odds fill gaps when available. No paid odds provider is used.
- **Route participation** relies on nflverse participation data, which is released with a delay and may be missing for the newest weeks.
- **Projections** for the GM tools are derived from recent nflverse production (blended per-game averages), not a consensus projection feed; pass your own `projection` values for best results.
- `cloudflare/cloudflared:latest` is pinned by tag only; pin a digest for reproducible deployments.

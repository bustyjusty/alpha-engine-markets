# Market Intel

A market intelligence platform for **discretionary event-driven investing**: data
aggregation, news collection, AI research summaries, event calendars, relative-value
scanning and theme tracking — surfaced through a Streamlit dashboard. This is a
research and monitoring tool, not a trading bot.

## Architecture

```
market-intel/
├── app.py                        # Streamlit entrypoint (st.navigation)
├── market_intel/
│   ├── config.py                 # Settings from env vars / .env (pydantic-settings)
│   ├── logging_conf.py           # Console + rotating-file logging
│   ├── exceptions.py             # Typed error hierarchy (ProviderError, RateLimitError, ...)
│   ├── cache.py                  # DB-backed TTL cache for API responses
│   ├── models/                   # Provider-agnostic domain dataclasses
│   ├── database/
│   │   ├── engine.py             # Database wrapper: engine + transactional sessions
│   │   ├── orm.py                # SQLAlchemy 2.0 schema (dialect-portable, 12 tables)
│   │   └── repositories/         # Only layer that touches the ORM
│   ├── providers/                # Adapters: yfinance (prices, news, earnings, ETF holdings)
│   ├── services/                 # market data, news, calendar, ETF, scanner,
│   │                             # AI research, watchlists, themes, journal
│   └── analysis/                 # Pure computation: returns, beta, relative-value scan
├── ui/
│   ├── context.py                # Service graph, cached per server process
│   ├── figures.py                # Plotly figure builders (pure)
│   └── pages/                    # Overview, Charts, News, Calendar, Scanner,
│                                 # ETF Explorer, Themes, Research, Journal
└── tests/                        # 80 tests incl. offline UI smoke tests
```

**Dependency direction:** `ui → services → (providers, repositories, analysis) → models`.
Layers never reach upward, and only repositories import the ORM.

### Key design decisions

- **Provider abstraction.** Each data category (market data, news, fundamentals,
  macro) has an abstract interface in `providers/`; concrete adapters (yfinance,
  Finnhub, NewsAPI, FRED, ...) return **domain dataclasses**, never raw payloads.
  Swapping providers = writing one adapter + changing one env var.
- **Database portability.** SQLAlchemy 2.0 with dialect-agnostic types only, UTC
  datetimes, and portable upserts in repositories. Migrating to PostgreSQL is a
  `MIP_DATABASE_URL` change.
- **Data provenance.** Stored market data carries `source` and `fetched_at`
  columns so you always know where a number came from and how stale it is.
- **Two-level caching.** A DB-backed TTL cache for provider responses (survives
  restarts, respects rate limits) plus `st.cache_data` at the UI layer.
- **Graceful degradation.** Provider failures raise typed exceptions; services
  fall back to cached/stored data and the UI shows staleness instead of crashing.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env   # then fill in the API keys you have
```

All configuration is via `MIP_*` environment variables — see `.env.example`.
Every API key is optional; features that need a missing key are disabled, not fatal.

## Run the dashboard

```powershell
streamlit run app.py
```

Default providers are keyless (Yahoo Finance), so the dashboard works out of
the box: add symbols to a watchlist on **Overview**, then use **Refresh** on
the News/Calendar pages to collect data. AI note generation on the
**Research** page needs `MIP_ANTHROPIC_API_KEY`.

## Tests

```powershell
pytest                     # offline suite (fakes, no network)
pytest -m integration      # live yfinance smoke test
```

## Module roadmap

1. ✅ Foundation: config, logging, exceptions, database schema + repositories
2. ✅ Market data: provider interface, yfinance adapter, incremental sync, caching
3. ✅ News collection and storage (URL-deduplicated, linked to securities)
4. ✅ Event calendar (earnings + manual macro) + ETF constituents & exposure
5. ✅ Analysis engine + relative-value scanner (z-scored move vs benchmark)
6. ✅ AI research service (Anthropic API) + archive, themes, watchlists, journal
7. ✅ Streamlit dashboard (9 pages, Plotly charts, offline smoke-tested)
8. Next ideas: FRED macro series, Finnhub fundamentals, event-reaction studies
   around stored earnings dates, pair z-score tracking, PostgreSQL deployment

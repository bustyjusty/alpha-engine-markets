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
│   ├── universe.py               # Cross-asset recap universe (region x asset class)
│   ├── providers/                # Adapters: yfinance, FRED, Alpha Vantage, fallback chain
│   ├── services/                 # market data, news, calendar, ETF, scanner,
│   │                             # AI research, watchlists, themes, journal
│   ├── services/                 # ... plus the global market recap service
│   └── analysis/                 # Pure computation: returns, beta, relative-value
│                                 # scan, recap moves, session clock, narrative,
│                                 # MKR frameworks (fibs, FVGs, ADX, Monte Carlo)
├── ui/
│   ├── context.py                # Service graph, cached per server process
│   ├── figures.py                # Plotly figure builders (pure)
│   └── pages/                    # Overview, Global Recap, MKR Framework, Charts,
│                                 # News, Calendar, Scanner, ETF Explorer, Themes,
│                                 # Research, Journal
└── tests/                        # 303 tests incl. offline UI smoke tests
```

**Dependency direction:** `ui → services → (providers, repositories, analysis) → models`.
Layers never reach upward, and only repositories import the ORM.

### Key design decisions

- **Provider abstraction.** Each data category (market data, news, fundamentals,
  macro) has an abstract interface in `providers/`; concrete adapters (yfinance,
  FRED, Alpha Vantage, NewsAPI, ...) return **domain dataclasses**, never raw
  payloads. Swapping providers = writing one adapter + changing one env var.
- **No single point of data failure.** No free market-data source is at once
  broad, current and officially supported, so market data defaults to a
  **fallback chain**: yfinance (current, covers everything, unofficial scrape)
  -> FRED (official Fed service, ~30 series, publishes with a lag) ->
  Alpha Vantage (licensed API, 25 requests/day on the free tier). Whichever
  provider served a bar is recorded in its `source` column.
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

## Global Recap

A morning cross-asset wrap, built for forming a view fast.

**Snapshot (no API key needed).** 93 instruments spanning APAC, UK, Europe, US and
global, grouped by asset class, with 1D / 5D / 1M / YTD moves, breadth per region,
and top gainers, losers and biggest movers.

**Session clock.** At 11:00 in Singapore, APAC is mid-session while London and New
York last printed the day before. Every region is stamped live / closed / weekend,
so a prior close is never presented as today's trading.

**Correct units per asset class.** Indices and FX move in percent; yields and
spreads move in basis points. They are ranked in separate tables — a 4bp move on
the 10y-2y curve is +8.7% in percentage terms, and ranking it against equities
would put a trivial rates move at the top of the board.

**Labelled proxies.** Yahoo publishes live yields for US Treasuries only. Other
rates markets fall back to listed bond ETFs, flagged as proxies with the caveat
that a price rally means yields fell — never presented as a yield. Credit spreads
(HY/IG OAS), the 10y-2y curve and 10-year breakevens come from FRED, which has no
Yahoo equivalent.

**News and archive on the page.** Headlines load on demand for the day's biggest movers and the standing benchmarks — a mover without a story is usually the one worth chasing. Every report built here is archived and searchable from the same page, alongside notes written on the Research page.

**Data-quality guards.** A market twenty minutes into its open prints a real bar
on negligible volume; compared with a full prior session it shows a huge move nobody
traded. Those bars are detected by volume and kept out of the movers ranking — but
still named, since hiding them would mislead too. A zero-volume bar is treated as a
bad print outright: the provider occasionally serves one carrying a wildly wrong
close. Recent bars are re-fetched on every sync so today's price updates intraday and
provider corrections land, rather than freezing at the first print ever stored.

## Report pipeline

Type what you want; the pipeline runs five stages and hands back a written report.

1. **Interpret** — resolve the free-text prompt into regions, asset classes and
   depth. Generous matching: "the yen" means APAC and currencies. Anything the
   prompt does not pin down stays unset, so unmatched words widen scope rather
   than silently narrowing it.
2. **Gather** — price the requested slice into a snapshot, and collect
   headlines for the movers plus a few standing benchmarks. News is
   best-effort: an outage costs you the headlines, never the report.
3. **Analyse** — derive breadth, leaders, laggards, rate moves and the
   cross-asset divergences worth acting on.
4. **Write** — hand the findings to Claude with web search enabled, or, with no
   API key, render them as prose directly.
5. **Archive** — store the report with the research notes, searchable from
   the Global Recap page itself.

Stage 4 is the point: **a report is always produced.** Without a credential you get
real written analysis computed from the data, not a table and an apology. With one,
the same findings go to the model as established arithmetic so its effort goes into
researching causation instead of re-deriving what the snapshot already says. If the
model call fails, the pipeline falls back to the deterministic writer and says so
rather than losing the report.

## MKR 14-Framework

Single-name analysis on the same principle as the recap: **compute everything
computable, then ask the model only for judgement.** One ticker in, a structured
trade plan out — scorecard, levels, entry timing, option legs, Monte Carlo.

Nine of the fourteen frameworks are pure arithmetic and are never left to a
language model: the Fibonacci grid and extensions, RSI on two timeframes with
divergence classification, three-bar fair value gaps (daily plus 4-hour where
intraday bars load), volume and OBV, the moving-average stack with swing-derived
support and resistance, MACD/ADX/Bollinger/ATR, the daily-weekly-monthly
alignment, the confirmation stack including the death-cross override, and the
capital-rotation sizing rules. Three more come from the option chain and
fundamentals. Elliott Wave and chart patterns are labelled by heuristic and
**marked ambiguous when the evidence is thin** — a confident wave count from a
machine that cannot see the chart is worse than no count.

Deliberate choices worth knowing about:

- **Option expiries are picked per maturity bucket, not nearest-first.** On a
  name with daily expiries the nearest six are all inside a month, which leaves
  the wheel with nothing in its 21-60 day window and no LEAPS at all. Chain
  statistics also skip anything under a week out, because a zero-day put/call
  ratio is same-day scalping and its max pain is meaningless by tomorrow.
- **IV rank is never quoted.** A real one needs a year of implied-vol history,
  which no free source publishes. ATM IV is compared against *realised*
  volatility and its one-year percentile instead, and both the page and the
  model prompt say that is what it is.
- **LEAPS scenarios reprice with Black-Scholes at the shorter maturity** rather
  than multiplying by delta, so decay is included — which matters precisely for
  a holder who exits inside two months.
- **The stop is structural.** A confirmed swing low or the wave invalidation,
  never a moving average; the 2x ATR level is reported alongside it for sizing.
  Where structure sits further away than that, the answer is a smaller
  position, not a tighter stop placed where noise will hit it.
- **Targets must clear the noise.** Any candidate closer than one ATR is
  skipped, and a full-size entry is downgraded to a starter whenever the first
  target is nearer than the invalidation — a green board with a 0.6x payout is
  still a bad trade.
- **Monte Carlo bootstraps the security's own daily returns**, so its fat tails
  survive instead of being flattened into a normal distribution. The
  thesis-tilted run recentres the drift on the path to T1 and leaves volatility
  alone; when that implies *less* upside than the stock has actually delivered,
  the report says so rather than letting the reader assume a bug.
- **Two UNI legs cannot be measured** — scarcity/monopoly IP and, without margin
  data, pricing power. They score a neutral 3 and are flagged as placeholders,
  so a 21/25 built on two guesses does not read like a 21/25 that was measured.

With `MIP_ANTHROPIC_API_KEY` set, the computed pack goes to Claude with web
search for the judgement layer: the wave count in context, the patterns
arithmetic cannot see, the two UNI legs, and the researched catalyst and risk
picture. Without a key the computed report *is* the report — the same contract
as the recap pipeline, for the same reason.

## Module roadmap

1. ✅ Foundation: config, logging, exceptions, database schema + repositories
2. ✅ Market data: provider interface, yfinance adapter, incremental sync, caching
3. ✅ News collection and storage (URL-deduplicated, linked to securities)
4. ✅ Event calendar (earnings + manual macro) + ETF constituents & exposure
5. ✅ Analysis engine + relative-value scanner (z-scored move vs benchmark)
6. ✅ AI research service (Anthropic API) + archive, themes, watchlists, journal
7. ✅ Streamlit dashboard (11 pages, Plotly charts, offline smoke-tested)
8. ✅ Global Recap: 93-instrument cross-asset universe, session-aware snapshot,
   multi-source provider chain, and an agentic AI wrap with live web research
9. ✅ MKR 14-Framework: single-name analysis with computed frameworks, option
   legs priced from the live chain, bootstrapped Monte Carlo, and an AI
   judgement layer over the numbers
10. Next ideas: Finnhub fundamentals, event-reaction studies around stored
   earnings dates, pair z-score tracking, PostgreSQL deployment

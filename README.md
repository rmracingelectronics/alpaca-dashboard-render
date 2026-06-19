# Alpaca Momentum Dashboard V33 - Render Paper Trading

V33 is the Render deployment package for the Symbol/Event Playbook dashboard and Alpaca paper-trading worker.

It keeps the historical backtest lab, keeps the single **Best Report 153601** strategy preset separate from sizing, and adds a Render worker that can automatically submit Alpaca **paper** bracket orders from the live algorithm. The dashboard now reads a shared database so it can show live signal plans, open positions, recent Alpaca orders, closed-position events, account snapshots, and worker events.

## What changed in this package

- Added a shared persistence layer: `src/live_store.py`.
- Added a Render Postgres database in `render.yaml` and wires `DATABASE_URL` into both services.
- Updated the dashboard live monitor to read the shared worker database instead of relying on a local file.
- Added tables for:
  - open live/paper positions
  - recent signal plans / order plans
  - recent Alpaca paper orders
  - closed / recently closed positions
  - worker events
- Updated the paper worker to:
  - fetch recent 5-minute and daily bars online from Alpaca
  - apply the Best Report 153601 signal/filter stack to the latest fully closed 5-minute bar
  - submit market bracket orders automatically when enabled
  - sync account snapshots, positions, orders, and closed-position detections to the shared database
  - enforce the 12-bar max-hold rule by canceling open orders for the symbol and submitting a close-position request
- Kept sizing outside the strategy preset with `LIVE_*` environment variables.

## Render process layout

`render.yaml` defines three Render resources:

```text
web       -> gunicorn app:server --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120
worker    -> python trading_worker.py
database  -> alpaca-live-trading-db-v33
```

The database is required because Render web services and workers do not share a reliable local filesystem. The worker writes live state to `DATABASE_URL`; the dashboard reads from the same `DATABASE_URL`.

## Automatic Alpaca paper trading

The package is configured for Alpaca paper trading only:

```env
ALPACA_TRADING_BASE_URL=https://paper-api.alpaca.markets/v2
ALLOW_LIVE_TRADING=false
PAPER_TRADING_ENABLED=true
PAPER_TRADING_DRY_RUN=false
```

With those settings, the worker places paper orders automatically from the algorithm. You do not need to go to the Alpaca website to place the trades.

To temporarily log trade plans without sending orders, set:

```env
PAPER_TRADING_DRY_RUN=true
```

## Strategy preset used by the worker

The worker uses one strategy preset only: **Best Report 153601**.

Strategy parameters:

- Strategy profile: `symbol_playbook_v25`
- Direction: long + short
- Top trades/day: 2
- Candlestick mode: selective
- Min score: 2
- News/catalyst proxy: on
- News/catalyst filter: skip catalyst
- QQQ stress filter: skip QQQ stress days
- QQQ stress threshold: 4.2
- Macro filter: off
- Symbol/side kill switch: off
- Target: +0.75R
- Stop: -1.00R
- Max hold: 12 bars
- Timeframe: 5-minute
- Volume Profile reaction filter: active

The preset does **not** change sizing, compounding, account value, max daily loss, or live risk caps.

## Live algorithm behavior

The live worker uses the same V25 Symbol/Event Playbook signal path and the same baseline filter stack used by the historical lab:

1. Pull recent 5-minute and daily bars from Alpaca for QQQ and the selected symbols.
2. Build the same intraday, daily, QQQ, relative-strength, ATR, VWAP, and volume-profile features.
3. Run `compute_signals(..., strategy_profile="symbol_playbook_v25")`.
4. Apply score, selective candle, news/catalyst proxy, QQQ stress skip, and top-2 rules.
5. Evaluate only the latest fully closed 5-minute bar.
6. Size the order using the separate `LIVE_RISK_MODE` and `LIVE_*` variables.
7. Submit a market bracket order to Alpaca paper when `PAPER_TRADING_DRY_RUN=false`.

Important live-vs-backtest note: the historical report can rank all candidates after the day is complete. Live trading cannot know future candidates that have not happened yet, so the default `LIVE_SELECTION_MODE=seen_so_far_top_n` applies the top-2 rule only to signals available so far and then enforces a hard two-trades-per-day cap. This avoids look-ahead behavior while keeping the same signal/filter logic.

## Risk sizing remains separate

Main live risk variables:

```env
LIVE_RISK_MODE=fixed_dollar_risk
LIVE_FIXED_RISK_DOLLARS=100
LIVE_ACCOUNT_VALUE_FALLBACK=10000
LIVE_BASE_RISK_PCT=1.0
LIVE_MIN_RISK_DOLLARS=10
LIVE_MAX_RISK_DOLLARS=100
LIVE_DD1_RISK_PCT=0.75
LIVE_DD2_RISK_PCT=0.50
LIVE_PAUSE_DD_PCT=15
LIVE_ALLOW_FRACTIONAL_SHARES=false
LIVE_MAX_DAILY_TRADES=2
LIVE_MAX_OPEN_POSITIONS=2
LIVE_MAX_ORDERS_PER_SYMBOL_PER_DAY=1
LIVE_MAX_DAILY_LOSS_DOLLARS=500
```

Supported sizing modes:

```text
fixed_dollar_risk
percent_equity
controlled_compounding
```

## Watchlist / symbols

Default live universe:

```env
LIVE_WATCHLIST_PRESET=v25_playbook
LIVE_SYMBOLS=
```

To override it, set a comma-separated list:

```env
LIVE_SYMBOLS=NVDA,TSLA,AMD,PLTR,SOFI
```

## Deploying on Render

1. Push this package to a GitHub repository.
2. In Render, create a Blueprint from `render.yaml`.
3. Enter your Alpaca paper API key, secret key, and account ID when Render asks for secret environment variables.
4. Confirm the Postgres database is created and both services have `DATABASE_URL` populated.
5. Confirm the dashboard web service opens.
6. Confirm the worker logs show a validated paper worker.
7. During market hours, watch the **Live Alpaca Paper Monitor** section in the dashboard.

The dashboard should populate as the worker writes heartbeat, account, signal, order, and position data into the shared database.

## Local run

Copy `.env.example` to `.env`, fill in your Alpaca paper keys, then run the dashboard:

```bash
pip install -r requirements.txt
python app.py
```

Run the worker locally in a separate terminal:

```bash
python trading_worker.py
```

If `DATABASE_URL` is blank locally, the app uses SQLite at:

```text
data/live_trading/live_trading.sqlite3
```

On Render, use the managed Postgres database from `render.yaml`.

## Safety gates

The worker refuses to start if:

- Alpaca keys are missing.
- `PAPER_TRADING_ENABLED=false`.
- `ALPACA_TRADING_BASE_URL` is not the paper endpoint and `ALLOW_LIVE_TRADING=false`.

This package is intended for Alpaca paper testing, not live-money trading.

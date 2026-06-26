# V33 V10 - free-feed live monitor and extended-hours diagnostics

This package keeps the V5/V6/V7/V8/V9 fixes and adds a stricter solution for the remaining blank Live Symbol Intelligence rows on Alpaca Basic/free paper accounts.

## What was actually wrong

The worker settings were already saved and loaded from Postgres. The worker was also running in `all_strategies` mode with extended-hours enabled.

The remaining blank rows came from the live market-data/indicator path:

1. **The strategy schedule was extended, but the indicator diagnostics could still end up with no same-day usable feature row.**
   The worker was showing `Waiting - no latest bar` / `0/0 checks` when the final strategy output did not align with the exact wall-clock 5-minute slot. This is common with IEX in pre/after-market because some symbols do not print every five minutes.

2. **The live daily-feature merge required a same-session daily row.**
   Alpaca's current `1Day` row may be missing or partial during live/pre-market. The old merge could leave `prev_close`, `daily_atr14_percent`, and `avg_20d_dollar_volume` blank for today's intraday bars. Those fields are required by the strategy gates.

3. **The app needed to be explicit about Alpaca Basic/free data limitations.**
   Free/no-subscription historical equity bars are reliable only on IEX. IEX is one exchange, so extended-hours bars can be sparse or missing by symbol. SIP is broader, but recent SIP requires a data entitlement or a delayed query.

## What V10 changes

### 1. Live-safe daily features

The live worker now attaches daily features from the most recent completed daily bar before the current intraday session. It no longer requires today's `1Day` row to exist before computing:

- previous close
- previous day high/low
- 20-day average dollar volume
- daily ATR percent

### 2. Latest usable bar diagnostics

When no strategy signal exists, the Live Symbol Intelligence panel now falls back to the latest usable indicator row within `live_max_bar_age_minutes` and still shows:

- close
- RVOL
- daily ATR percent
- day/open relative strength
- VWAP extension in ATR
- check count
- reason no setup was active

So the panel should show `Watching - no setup` or `Blocked - ...` with indicator values instead of blank `Waiting - no latest bar` whenever usable bars exist.

### 3. Alpaca free-feed aware fetching

The worker keeps `IEX` as the safest free/default feed. In all-strategies extended-hours mode it also attempts a best-effort SIP request with an end time at least 16 minutes delayed. If Alpaca rejects it, the worker continues using IEX and stores the rejection in diagnostics.

A dashboard feed dropdown is available:

- `IEX - free/no-subscription, single-exchange`
- `SIP delayed - broad market, 16+ min delayed`
- `SIP real-time - paid/unlimited entitlement`

For `SIP delayed`, the code uses `feed=sip` with a delayed `end` time for historical bars. It does not pass `feed=delayed_sip` to `/v2/stocks/bars` because the historical bars endpoint documents `iex`, `sip`, `otc`, and `boats` as allowed bars feeds.

### 4. Better diagnostics

Added/kept these no-Dash endpoints:

- `/healthz`
- `/debug/db-ping`
- `/debug/live-state`
- `/debug/live-data`
- `/debug/live-symbol-monitor`
- `/debug/live-data-readiness`
- `/debug/live-bar-health`

`/debug/live-bar-health` is the key endpoint for this issue. It shows:

- configured feed
- primary API feed actually used
- effective feed
- 5-minute row count
- daily row count
- missing/stale symbols
- latest bar per symbol
- SIP fallback rows
- Alpaca API errors/rejections if any
- symbol monitor preview

## What this does not hide

The current strategy presets are still mainly regular-session/morning-trained strategies. Extended-hours mode is now technically monitored correctly, but the strategies may still produce no trades because their gates were researched on regular-market behavior. That is different from the panel being blank.

Expected healthy extended-hours states are now:

- `Watching - no setup`
- `Blocked - score`
- `Blocked - liquidity`
- `Blocked - quality gate`
- `Candidate - passed latest-bar checks`
- `Selected - order planning`

A row should only stay `Waiting - symbol data` / `Waiting - no latest bar` if Alpaca did not provide usable bars for that symbol/feed/session.

## Commit commands

```powershell
cd C:\render_apps\alpaca_dashboard_v33
git status --short
git add app.py src/alpaca_rest.py src/live_engine.py src/live_store.py src/live_dashboard.py assets/styles.css Procfile render.yaml README_FIX_LIVE_SETTINGS.md
git commit -m "Fix free-feed live monitor and extended-hours diagnostics"
git push origin master --verbose --progress
```

If Render watches `main` instead of `master`:

```powershell
git push origin master:main --verbose --progress
```

## After deploy

Open these in order:

```text
https://alpaca-momentum-dashboard-v33.onrender.com/healthz
https://alpaca-momentum-dashboard-v33.onrender.com/debug/db-ping
https://alpaca-momentum-dashboard-v33.onrender.com/debug/live-state
https://alpaca-momentum-dashboard-v33.onrender.com/debug/live-bar-health
https://alpaca-momentum-dashboard-v33.onrender.com/debug/live-symbol-monitor
https://alpaca-momentum-dashboard-v33.onrender.com/debug/live-data
```

Then open the dashboard Live tab and click `Refresh live data now`.

## Operational recommendation

For free Alpaca paper experiments:

- Use `IEX` if you want the real-time free feed but accept incomplete extended-hours coverage.
- Use `SIP delayed` if you want broader paper-research diagnostics and can accept 16+ minute delayed decisions.
- Use `SIP real-time` only if the account has the paid/unlimited data entitlement.

For real-money trading, do not use the all-strategies extended-hours experiment without separate risk controls, spread checks, and dedicated extended-hours strategy validation.

## V11 live-data safety correction

V10 added free-feed diagnostics for Alpaca Basic/IEX accounts. V11 tightens the live trading path so order decisions always use a real-time feed only:

- Live order decisions use `iex` or `sip` only.
- If an old saved database config contains `delayed_sip`, the live worker automatically uses `iex` for live scans instead of trading from delayed bars.
- The worker heartbeat and `last_bar_fetch` now expose `live_data_policy = real_time_only_for_order_decisions` and `delayed_data_used_for_orders = false`.
- The delayed SIP idea is kept out of the live order path. Delayed/broad-market data is only appropriate for diagnostics/research/backtest, not for live entries.
- Regular-session entries still submit market bracket paper orders.
- Extended-hours entries still submit Alpaca-compatible marketable limit paper orders with `extended_hours=true`.

## V12 - All-strategies monitor scaffold fix

The V11 diagnostics showed `live_config_override` correctly saved `live_strategy_run_mode=all_strategies` with 14 active variants, and the heartbeat later showed the worker running all 14 variants, but the Live Symbol Intelligence table still displayed only one or two strategy views during a scan.

Root cause: the worker wrote `live_strategy_symbol_monitor` rows one strategy at a time. The dashboard reads the newest run_id, so while a scan was still in progress it could show a partial run: 16 rows for one strategy, 32 rows for two strategies, etc. This looked like all-strategies mode was not working even though the worker had started the all-strategies scan.

Fix: at the start of every all-strategies scan, the worker now publishes a complete scaffold of all active strategies x all configured symbols using status `Scanning - queued`. Each strategy then overwrites its own scaffold rows with real indicator/check data as it finishes. This makes the dashboard immediately show the expected coverage, e.g. 14 strategies x 16 symbols = 224 monitor rows, instead of showing only the first strategy that finished writing.

This does not change live trading decisions, risk sizing, Alpaca order submission, data feed policy, or backtesting. It only fixes the visibility/diagnostic layer for all-strategies mode.

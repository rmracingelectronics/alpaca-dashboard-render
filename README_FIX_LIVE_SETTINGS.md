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

## V13 - order budget / Alpaca rejection fix

This build fixes order submission failures observed in the worker events table:

- Alpaca 403 insufficient Reg-T buying power on repeated AMD short attempts.
- Alpaca 422 bracket short stop-loss price too close to the base price.

Changes:

- Added conservative account buying-power detection using Alpaca account buckets (`regt_buying_power`, `buying_power`, `daytrading_buying_power`, `non_marginable_buying_power`).
- Added a per-order notional cap so tiny stop distances cannot create very large positions. Default is 10% of equity in all-strategies mode and 25% in single-strategy mode. Override with `live_max_position_notional_pct` if needed.
- Added an 80% buying-power safety reserve, overrideable with `live_buying_power_safety_pct`.
- When a signal is capped, the order plan/report records `notional_capped`, `notional_cap`, `uncapped_qty`, and `estimated_notional`.
- Added a practical minimum stop/target distance before sizing. This prevents Alpaca bracket rejections where a short stop rounds to the same cent as Alpaca's base price.
- Prevents repeated order attempts for the same symbol inside the same worker scan after an attempted submission, reducing repeated 403/422 spam in all-strategies mode.

The live Alpaca order path remains active:

- Regular session: market bracket order.
- Extended hours: simple limit order with `extended_hours=true` and `time_in_force=day`.


## V14 risk/compounding correction

V13 introduced a fixed default notional cap to prevent Alpaca buying-power rejects. That was too blunt for this project because the dashboard already has an explicit risk model: fixed-dollar risk, percent-equity risk, and controlled compounding with min/max risk dollars and drawdown throttles.

V14 removes the hard-coded 10%/25% order cap. The live worker now keeps the existing risk budget calculation as the source of truth:

- `fixed_dollar_risk` uses the configured fixed risk dollars.
- `percent_equity` uses current Alpaca account equity and the configured base risk percent.
- `controlled_compounding` uses current Alpaca equity, high-water mark, drawdown thresholds, min risk, max risk, and pause drawdown.

After the target risk budget is calculated, the worker applies an execution affordability guard based on the live Alpaca paper account's current buying-power buckets. In all-strategies mode, the default allocation is based on the number of remaining configured strategy slots, derived from `max_open_positions`, `max_daily_trades`, and active strategy count. This prevents one tight-stop setup from consuming the whole paper account while keeping the user's risk model intact.

The order plan now reports:

- `risk_budget`: target risk from the selected risk/compounding model.
- `actual_risk_dollars`: actual risk after any affordability sizing.
- `risk_budget_shortfall`: difference between target and actual risk.
- `notional_cap_reason`: why the quantity was capped, if it was capped.
- `notional_capped`: whether the order size was reduced by the affordability guard.

No strategy preset hides sizing or compounding. Risk settings remain separate from strategy selection.

## V15 risk UI/compounding clarification

- Clarifies that live mode uses Alpaca account equity when available. The Account value field is a fallback/backtest value.
- Adds a live risk preview showing the effective intended risk budget for the selected risk mode.
- Makes fixed risk, percent-equity, and controlled compounding explicit:
  - Fixed dollar risk uses the selected fixed $ radio only.
  - Percent-equity uses Base risk % x live Alpaca equity and ignores fixed-risk and min/max fields.
  - Controlled compounding uses Base risk % plus min/max risk and drawdown brakes.
- Stops saving controlled-compounding min/max values as active settings when the selected mode is fixed-dollar or percent-equity, so DB/debug output no longer implies hidden $300 caps.
- Keeps the Alpaca affordability guard as an execution-layer guard only; it does not replace the selected risk/compounding model.

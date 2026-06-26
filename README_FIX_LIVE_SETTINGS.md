# V20 live worker scan + diagnostics fix

This build keeps all prior fixes (database-backed settings, Alpaca paper trade history, live symbol intelligence, all-strategies paper experiment mode, real-time-only live decisions, risk/compounding guard, bracket auto-repair retry, monitor value carry-forward, and worker-independent scan progress).

## Main V20 changes

### 1) Faster all-strategies scan
All-strategies mode was recomputing the same intraday/daily/QQQ indicator feature frames for every strategy. With 14 strategies x 16 symbols, that made scans slow enough for the dashboard to show old or partial monitor rows.

V20 precomputes the per-symbol market/indicator feature frames once per scan and reuses them for every strategy variant. The strategy rules still run independently and keep their own strategy attribution.

New DB diagnostic state:

- `live_feature_cache_summary`

### 2) Monitor table no longer shows queued placeholders as if they were final values
The strategy-symbol monitor table is a current-state table keyed by symbol/strategy/gate. Earlier queued scaffolds overwrote useful indicator values during scans. V20 makes scaffold/progress state progress-only and leaves the monitor table to store real indicator/check rows.

The dashboard should now keep the last useful RVOL/ATR/RS/VWAP/check values visible while a new scan is running.

### 3) `signals_filtered` now explains why
The previous event only said:

`Signals existed but were duplicate, already open, or failed sizing/capacity checks.`

V20 stores and displays reason counts/examples, such as:

- `duplicate_signal_already_submitted`
- `symbol_daily_order_limit`
- `position_already_open`
- `buying_power_reserve_limit`
- `sizing_or_invalid_order_plan`
- `symbol_already_attempted_this_scan`

New DB diagnostic state:

- `last_signal_filter_summary`

## What this does not change

- Does not require the browser to stay open.
- Does not change live/backtest separation.
- Does not remove Alpaca order submission.
- Does not change fixed-risk / percent-equity / controlled-compounding logic.
- Does not add user settings.
- Does not use delayed/historical data for live order decisions.
- Does not change extended-hours order rules.

## Useful checks after deploy

Open these URLs after Render redeploys:

- `/healthz`
- `/debug/live-state`
- `/debug/live-symbol-monitor`
- `/debug/live-bar-health`

In `/debug/live-state`, check:

- `heartbeat.status`
- `heartbeat.strategy_run_mode`
- `live_scan_progress`
- `last_completed_strategy_scan`
- `live_feature_cache_summary`
- `last_signal_filter_summary`


## V20 - signal filter diagnostics and complete strategy-symbol monitor view

This build keeps all previous live fixes and adds two operational corrections:

1. `signals_filtered` is no longer a vague event.  When strategies produce signals but no Alpaca order is submitted, the worker now writes `last_signal_filter_summary` with `filter_reason_counts` and examples.  The dashboard/debug endpoint `/debug/live-submit-filters` shows whether the blocker was duplicate signal, open position, symbol daily order limit, buying-power reserve, invalid sizing, or capacity.

2. Live Symbol Intelligence now reads `live_strategy_symbol_monitor` as a latest-state table keyed by strategy-symbol, not as a single `run_id` batch.  This prevents the dashboard from showing only a partial in-progress scan such as 44 rows when all 14 strategies x 16 symbols should be represented.  Queued rows continue to carry forward the last completed indicator values.

New diagnostic endpoint:

- `/debug/live-submit-filters`

Deployment command reminder:

```powershell
cd C:\render_apps\alpaca_dashboard_v33
git status --short
git add app.py src/live_store.py src/live_engine.py README_FIX_LIVE_SETTINGS.md
git commit -m "Improve live signal filter diagnostics and monitor coverage"
git push origin master --verbose --progress
```

## V21 - Extended-hours protective exits and scan lifecycle hardening

This build keeps the V20 settings/database/all-strategies fixes and adds the missing execution safety required for unattended paper testing:

- Extended-hours entries are still submitted as Alpaca-compatible simple limit orders.
- Those extended-hours positions are now protected client-side by the worker:
  - if current price touches the stored stop, the worker submits an opposite-side exit order;
  - if current price touches the stored target, the worker submits an opposite-side exit order;
  - in extended hours the exit is a marketable limit order with `extended_hours=true`;
  - during the regular session it uses Alpaca close-position after cancelling open symbol orders.
- The worker stores stop/target/order-mode/strategy metadata on `live_positions` so protective exits survive browser closure and worker cycles.
- `/debug/live-state` and `/debug/live-symbol-monitor` now include the latest client-side protective-exit diagnostics.

The browser is still only a viewer/config UI. The Render worker performs scanning, order placement, and protective exits from the shared database state.

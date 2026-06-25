# V33 Render live/settings/trade display fix v6

This package keeps the v5 database-backed settings fix and restores the live Alpaca/paper history display.

## What was fixed

1. Settings persistence remains database-backed.
   - The dashboard loads `live_worker_state.key = 'live_config_override'` from Postgres.
   - Clicking **Apply current settings to live worker** writes `live_config_override`, reads it back, and only shows success after the readback `applied_at_utc` matches.
   - The legacy `settings` row is merged instead of overwritten, so applying dashboard settings no longer erases Alpaca/risk/account diagnostics written by the worker.

2. The Live tab now refreshes on demand and when opened.
   - Added **Refresh live data now**.
   - The live table callback is triggered by the Live tab selection, the refresh button, and the timer.
   - The status text now shows real loaded counts: trades, orders, signal plans, positions, and events.

3. Historical paper trades now pair filled exits correctly.
   - Alpaca bracket child legs can arrive with blank `parent_order_id` even when nested under the parent order.
   - `LiveStore.upsert_orders_recursive()` now repairs `parent_order_id` for future syncs.
   - `build_live_trade_report()` now also repairs historical display by reading nested raw order legs and by inferring same-symbol opposite-side filled exits when older rows have blank `parent_order_id`.
   - This restores realized P/L in the Live / Paper Trade P&L table for previous paper trades.

4. Diagnostics were expanded.
   - `/healthz` basic service health.
   - `/debug/db-ping` real Postgres write/read test.
   - `/debug/live-state` worker/config/heartbeat/settings plus table counts.
   - `/debug/live-data`, `/debug/live-tables`, and `/debug/live-snapshot` show the same tables the Live tab uses, including row counts and recent rows.
   - `/debug/alpaca-connection` verifies Alpaca trading API credentials from the web service without exposing secrets.

## Local validation performed

The uploaded Render database export contained historical `live_orders`, `live_signal_plans`, `live_events`, and account snapshots.  I loaded that export into a local SQLite live-store simulation and verified:

- 18 recent Alpaca paper orders load.
- 6 signal plans load.
- 6 Live / Paper Trade P&L rows are built.
- Previously pending rows now display `closed_filled_exit` with realized P/L and R values.
- The no-Dash debug routes return JSON successfully in Flask's test client.
- `python -m py_compile` passes for `app.py`, `src/live_store.py`, `src/live_dashboard.py`, and `src/live_engine.py`.

## Render deploy notes

Both Render services must still share the same `DATABASE_URL`:

- `alpaca-momentum-dashboard-v33` web service
- `alpaca-paper-trading-worker-v33` worker service

The web service should use one gunicorn worker:

```bash
gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 120
```

After deploy, open:

- `/healthz`
- `/debug/db-ping`
- `/debug/live-state`
- `/debug/live-data`
- `/debug/alpaca-connection`

Then open the dashboard, click the Live tab, and click **Refresh live data now**.


## V7 - Live Symbol Intelligence panel

This build adds a lightweight per-symbol live monitor so the Live tab shows which
symbols are currently monitored and how the active strategy indicators/gates are
behaving on the latest closed 5-minute bar.

New shared DB table:

- `live_symbol_monitor`: one row per configured live symbol, written by the worker
  and read by the dashboard.

New endpoints:

- `/debug/live-symbol-monitor`: no-Dash diagnostic view of the same symbol monitor
  rows used by the Live tab.
- `/debug/live-data` now also includes the symbol monitor preview/count.

The dashboard does not recalculate indicators on refresh.  The worker computes
signals as it already does, writes a compact monitor snapshot, and the dashboard
only reads the latest rows.  This keeps the Live tab mobile-friendly and avoids
another heavy Dash refresh loop.

Commit commands after copying files to `C:ender_appslpaca_dashboard_v33`:

```powershell
cd C:ender_appslpaca_dashboard_v33
git status --short
git add app.py src/live_store.py src/live_engine.py src/live_dashboard.py assets/styles.css README_FIX_LIVE_SETTINGS.md
git commit -m "Add live symbol intelligence monitor"
git push origin master --verbose --progress
```

If Render watches `main` instead of `master`:

```powershell
git push origin master:main --verbose --progress
```

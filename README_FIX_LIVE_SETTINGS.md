# V33 live/dashboard stability fix v4

This package addresses the Render dashboard permanently showing **Updating...** and the live settings not visibly saving.

## What changed

- Web service now runs with **one gunicorn worker** instead of two. The Render logs showed Dash component-suite 500 errors during page load, which is consistent with multi-worker Dash asset registration/race issues.
- Dash `update_title` is disabled, so the browser tab will no longer get stuck showing `Updating...` while diagnosing.
- Initial heavy live-table refresh is no longer run during first page load. It will run on the normal interval after the page has loaded.
- Added no-Dash diagnostics endpoints:
  - `/healthz`
  - `/debug/live-state`
- Added server-log prints for the settings load, live refresh, and apply-settings callbacks.
- Settings still save to the database only when clicking **Apply current settings to live worker**. The database row remains `live_worker_state.key = live_config_override`.

## After deploy

Open:

```text
https://alpaca-momentum-dashboard-v33.onrender.com/debug/live-state
```

This bypasses Dash and shows whether the web service can read the database, heartbeat, live_config_override, recent events, recent plans and recent orders.

## Render service settings

Both web and worker services must use the same `DATABASE_URL`. Rotate the DB password because it was pasted in chat.


## V5 fix

- Live settings load/save now uses `LiveStore(initialize_schema=False)` so the dashboard no longer runs full schema/index migrations during page load or settings save.
- Added fast diagnostics: `/debug/live-state` and `/debug/db-ping`. These avoid large table scans and should return quickly.
- Settings Apply now verifies the Postgres write by reading `live_config_override` back and comparing `applied_at_utc`; the UI confirmation is a real DB verification, not just a button condition.
- Added Postgres `statement_timeout` default of 5000 ms to prevent a blocked DB statement from leaving Dash stuck forever.

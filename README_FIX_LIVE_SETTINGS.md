# V33 live settings/database persistence fix

This package was fixed from the uploaded `alpaca_dashboard_v33.zip`.

## Main changes

- `app.py` now treats Postgres/`DATABASE_URL` as the source of truth for live settings.
- On page load, the dashboard reads `live_worker_state.key = 'live_config_override'` from the shared database first.
- Browser local storage is now only a fallback when the database row does not exist yet.
- Changing live/backtest/risk controls automatically saves the complete live configuration to the shared database.
- The `Apply current settings to live worker` button remains as a manual force-sync/status button and writes the same database row.
- The same config is also mirrored to `live_worker_state.key = 'settings'` under `settings.live` for older dashboard/report code.
- The worker already reads `live_config_override` on every `run_once()`, so it will pick up changes on the next scan if the web service and worker share the same `DATABASE_URL`.

## Database row to verify on Render

```sql
SELECT key, value_json, updated_at_utc
FROM live_worker_state
WHERE key IN ('live_config_override', 'settings', 'heartbeat');
```

`live_config_override.updated_at_utc` should update whenever a setting changes or when Apply is clicked.

`heartbeat.value_json` should eventually show:

```json
"config_source": "dashboard_db"
```

That confirms the worker is reading the database config.

# V33 Live Settings / Render Database Fix v3

This version keeps the Render PostgreSQL database as the source of truth for live-worker settings, but avoids saving every individual keystroke to the database.

## What changed in v3

- Dashboard loads saved live settings from `live_worker_state.key = 'live_config_override'` on page open.
- Clicking **Apply current settings to live worker** writes the current dashboard config to:
  - `live_worker_state.key = 'live_config_override'`
  - `live_worker_state.key = 'settings'`
- Removed the previous automatic database-save callback that fired on every control change/keystroke. That callback could keep the browser tab stuck on **Updating...** and could save partial typed values such as `2` instead of `20:00`.
- Added validation for live entry time fields so invalid/partial times do not get saved to the shared database.
- Worker still reads `live_config_override` during its scan loop and reports `config_source = dashboard_db` in the heartbeat when it applies dashboard settings.

## Expected behavior

1. Open dashboard.
2. Dashboard controls load from Postgres first, browser storage only as fallback.
3. Change settings.
4. Click **Apply current settings to live worker**.
5. Worker reads the saved database config on the next scan.

## Check from Render shell

Run this from the web service shell from the project root:

```bash
cd ~/project
python - <<'PY'
from src.live_store import LiveStore
store = LiveStore()
for key in ["live_config_override", "settings", "heartbeat", "market_clock"]:
    print("\n---", key, "---")
    print(store.get_state(key, None))
PY
```

If you paste into the Render shell and see `^[[200~python`, clear the line with Ctrl+C and paste only the command starting with `cd ~/project`.

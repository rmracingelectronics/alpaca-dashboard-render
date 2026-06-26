# V33 V9 live all-strategies / extended-hours fix

This package builds on the V5-V8 fixes:

- Settings save/load from Render Postgres (`live_worker_state.live_config_override`) remains the source of truth.
- The web dashboard and worker keep using the same shared `DATABASE_URL`.
- All-strategies paper experiment mode remains available.
- Live Symbol Intelligence remains available.

## V9 fixes

### 1. Extended-hours indicator pipeline now actually uses extended-hours bars

V8 allowed the worker schedule to run from 04:00-20:00 ET, but the indicator pipeline still called:

```python
add_intraday_features(...)
build_qqq_context(...)
```

without passing `session_mode`, so those functions defaulted to `regular_only` and filtered out all premarket/after-hours bars. That caused the Live Symbol Intelligence rows to show zero checks and messages such as `No same-day indicator row...` before 09:30 ET.

V9 now uses:

```python
session_mode = "extended_hours" if live_allow_extended_hours_entries else "regular_only"
add_intraday_features(..., session_mode=session_mode)
build_qqq_context(..., session_mode=session_mode)
```

When extended-hours entries are enabled, premarket and after-hours bars can now create real indicator/check rows.

### 2. Better live diagnostics

`/debug/live-state` now includes:

- `last_bar_fetch`
- `last_strategy_scan_summary`
- heartbeat `bar_session_mode`

This makes it clear whether the worker is using `regular_only` or `extended_hours` bars, how many rows Alpaca returned, and how many signals each strategy selected.

### 3. Extended-hours order compatibility

Alpaca extended-hours eligible equity orders must be limit orders with `time_in_force=day` or `gtc`, using `extended_hours=true`. The older worker always submitted market bracket orders, which are regular-session style entries.

V9 keeps regular-session entries as market bracket orders, but when a signal occurs outside 09:30-16:00 ET and extended-hours entries are enabled, the worker submits a simple marketable limit order with `extended_hours=true` and stores the target/stop in the signal plan/report payload.

This is intended for paper-account experimentation. Regular-session bracket behavior is unchanged.

### 4. Dashboard status explains data session/feed

The Live panel now reports the data session and feed, for example:

```text
Data session: extended_hours; feed: iex
```

Because the free Alpaca paper account uses IEX data, some extended-hours symbols may have sparse or missing bars. The symbol monitor now says that explicitly instead of making it look like the strategy is broken.

## Test notes

Before packaging, this build was tested with:

```bash
python -m py_compile app.py trading_worker.py src/*.py
```

A local synthetic premarket test confirmed that V9 produces populated Live Symbol Intelligence checks from 04:00-09:30 ET bars. The same data produces zero rows with the old default `regular_only` path, confirming the root cause.

## Commit commands

After copying the files into `C:\render_apps\alpaca_dashboard_v33`:

```powershell
cd C:\render_apps\alpaca_dashboard_v33
git status --short
git add app.py src/live_engine.py src/alpaca_trading.py src/live_dashboard.py README_FIX_LIVE_SETTINGS.md
git commit -m "Fix extended-hours all-strategies indicator pipeline"
git push origin master --verbose --progress
```

If Render watches `main` instead of `master`:

```powershell
git push origin master:main --verbose --progress
```

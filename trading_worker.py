from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.live_engine import LivePaperTradingEngine, load_live_settings_from_env


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    live, risk = load_live_settings_from_env()
    engine = LivePaperTradingEngine(live, risk)
    engine.validate_environment()
    print(
        f"[{datetime.now(timezone.utc).isoformat()}] Starting paper trading worker: "
        f"enabled={live.enabled}, dry_run={live.dry_run}, feed={live.feed}, "
        f"symbols={len(live.symbols or [])}, risk_mode={risk.position_sizing_mode}",
        flush=True,
    )
    while True:
        try:
            rows = engine.run_once()
            for row in rows:
                print(row, flush=True)
        except KeyboardInterrupt:
            print("Worker stopped by KeyboardInterrupt.", flush=True)
            return 0
        except Exception as exc:
            print(f"[{datetime.now(timezone.utc).isoformat()}] Worker error: {exc}", file=sys.stderr, flush=True)
        time.sleep(max(30, int(live.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())

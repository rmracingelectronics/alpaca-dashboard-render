from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import PROJECT_ROOT


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        return json.dumps({"repr": repr(value)}, sort_keys=True)


def _coerce_database_url(url: str | None) -> str:
    if not url:
        return f"sqlite:///{PROJECT_ROOT / 'data' / 'live_trading' / 'live_trading.sqlite3'}"
    if url.startswith("postgres://"):
        return "postgresql://" + url[len("postgres://") :]
    return url


class LiveStore:
    """Persistence shared by the Dash app and the paper-trading worker.

    Local development falls back to SQLite. Render should set DATABASE_URL to a
    managed Postgres database so the web service and the worker can read/write
    the same live account, order, signal-plan, and position state.
    """

    def __init__(self, database_url: str | None = None):
        self.database_url = _coerce_database_url(database_url or os.getenv("DATABASE_URL"))
        self.is_postgres = self.database_url.startswith("postgresql://")
        self.sqlite_path: Path | None = None
        if not self.is_postgres:
            if self.database_url.startswith("sqlite:///"):
                self.sqlite_path = Path(self.database_url.replace("sqlite:///", "", 1))
            else:
                self.sqlite_path = PROJECT_ROOT / "data" / "live_trading" / "live_trading.sqlite3"
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        if self.is_postgres:
            try:
                import psycopg2
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("DATABASE_URL is Postgres, but psycopg2-binary is not installed.") from exc
            conn = psycopg2.connect(self.database_url)
            conn.autocommit = True
            try:
                yield conn
            finally:
                conn.close()
        else:
            assert self.sqlite_path is not None
            conn = sqlite3.connect(str(self.sqlite_path), timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _ph(self) -> str:
        return "%s" if self.is_postgres else "?"

    def _execute(self, sql: str, params: Iterable[Any] | None = None) -> None:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, list(params or []))

    def _fetch(self, sql: str, params: Iterable[Any] | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, list(params or []))
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            if isinstance(row, sqlite3.Row):
                out.append(dict(row))
            else:
                out.append(dict(zip(cols, row)))
        return out

    def initialize(self) -> None:
        id_col = "SERIAL PRIMARY KEY" if self.is_postgres else "INTEGER PRIMARY KEY AUTOINCREMENT"
        statements = [
            f"""
            CREATE TABLE IF NOT EXISTS live_events (
                id {id_col},
                created_at_utc TEXT NOT NULL,
                event TEXT NOT NULL,
                symbol TEXT,
                strategy_side TEXT,
                status TEXT,
                message TEXT,
                signal_time_utc TEXT,
                order_id TEXT,
                client_order_id TEXT,
                payload_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_positions (
                symbol TEXT PRIMARY KEY,
                is_open INTEGER NOT NULL DEFAULT 1,
                side TEXT,
                qty REAL,
                avg_entry_price REAL,
                market_value REAL,
                current_price REAL,
                cost_basis REAL,
                unrealized_pl REAL,
                unrealized_plpc REAL,
                opened_at_utc TEXT,
                last_seen_utc TEXT,
                closed_at_utc TEXT,
                max_hold_until_utc TEXT,
                entry_client_order_id TEXT,
                entry_order_id TEXT,
                raw_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_orders (
                order_id TEXT PRIMARY KEY,
                client_order_id TEXT,
                parent_order_id TEXT,
                symbol TEXT,
                side TEXT,
                type TEXT,
                order_class TEXT,
                qty REAL,
                filled_qty REAL,
                limit_price REAL,
                stop_price REAL,
                filled_avg_price REAL,
                status TEXT,
                time_in_force TEXT,
                submitted_at TEXT,
                filled_at TEXT,
                expired_at TEXT,
                canceled_at TEXT,
                created_at TEXT,
                updated_at TEXT,
                raw_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_signal_plans (
                client_order_id TEXT PRIMARY KEY,
                symbol TEXT,
                strategy_side TEXT,
                alpaca_side TEXT,
                signal_time_utc TEXT,
                signal_time_et TEXT,
                submitted_at_utc TEXT,
                max_hold_until_utc TEXT,
                trigger_type TEXT,
                candidate_score REAL,
                qty REAL,
                entry_reference_price REAL,
                risk_per_share REAL,
                risk_budget REAL,
                stop_price REAL,
                target_price REAL,
                status TEXT,
                alpaca_order_id TEXT,
                dry_run INTEGER,
                payload_json TEXT
            )
            """,
            f"""
            CREATE TABLE IF NOT EXISTS live_account_snapshots (
                id {id_col},
                created_at_utc TEXT NOT NULL,
                equity REAL,
                last_equity REAL,
                buying_power REAL,
                cash REAL,
                portfolio_value REAL,
                status TEXT,
                raw_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_worker_state (
                key TEXT PRIMARY KEY,
                value_json TEXT,
                updated_at_utc TEXT NOT NULL
            )
            """,
        ]
        for stmt in statements:
            self._execute(stmt)
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_live_events_created ON live_events(created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_live_events_symbol ON live_events(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_orders_symbol ON live_orders(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_orders_status ON live_orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_live_plans_symbol ON live_signal_plans(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_plans_status ON live_signal_plans(status)",
            "CREATE INDEX IF NOT EXISTS idx_live_positions_open ON live_positions(is_open)",
        ]
        for stmt in indexes:
            self._execute(stmt)
        self._ensure_schema_migrations()

    def _column_exists(self, table: str, column: str) -> bool:
        try:
            if self.is_postgres:
                rows = self._fetch(
                    "SELECT column_name FROM information_schema.columns WHERE table_name=%s AND column_name=%s",
                    [table, column],
                )
                return bool(rows)
            rows = self._fetch(f"PRAGMA table_info({table})")
            return any(str(r.get("name", "")).lower() == column.lower() for r in rows)
        except Exception:
            return False

    def _add_column_if_missing(self, table: str, column: str, definition: str) -> None:
        if self._column_exists(table, column):
            return
        try:
            self._execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        except Exception:
            # Safe on concurrent service startup or partially migrated databases.
            pass

    def _ensure_schema_migrations(self) -> None:
        # Existing local/Render databases from older builds used CREATE TABLE IF NOT EXISTS.
        # Additive migrations keep the web service and worker compatible after redeploys.
        migrations = {
            "live_positions": [
                ("cost_basis", "REAL"),
                ("max_hold_until_utc", "TEXT"),
                ("entry_client_order_id", "TEXT"),
                ("entry_order_id", "TEXT"),
                ("raw_json", "TEXT"),
            ],
            "live_orders": [
                ("parent_order_id", "TEXT"),
                ("filled_avg_price", "REAL"),
                ("expired_at", "TEXT"),
                ("canceled_at", "TEXT"),
                ("created_at", "TEXT"),
                ("updated_at", "TEXT"),
                ("raw_json", "TEXT"),
            ],
            "live_signal_plans": [
                ("submitted_at_utc", "TEXT"),
                ("max_hold_until_utc", "TEXT"),
                ("risk_per_share", "REAL"),
                ("dry_run", "INTEGER"),
                ("strategy_variant", "TEXT"),
                ("strategy_preset", "TEXT"),
                ("strategy_profile", "TEXT"),
                ("quality_gate", "TEXT"),
                ("pattern_mode", "TEXT"),
                ("selection_mode", "TEXT"),
                ("realized_pl", "REAL"),
                ("realized_r", "REAL"),
                ("payload_json", "TEXT"),
            ],
            "live_events": [
                ("payload_json", "TEXT"),
                ("message", "TEXT"),
            ],
        }
        for table, cols in migrations.items():
            for column, definition in cols:
                self._add_column_if_missing(table, column, definition)

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    def upsert_signal_plan(self, plan: dict[str, Any], status: str | None = None) -> None:
        if not plan or not plan.get("client_order_id"):
            return
        ph = self._ph()
        data = {
            "client_order_id": plan.get("client_order_id"),
            "symbol": str(plan.get("symbol") or "").upper(),
            "strategy_side": plan.get("strategy_side"),
            "alpaca_side": plan.get("alpaca_side"),
            "signal_time_utc": plan.get("signal_time_utc"),
            "signal_time_et": plan.get("signal_time_et"),
            "submitted_at_utc": plan.get("submitted_at_utc") or utc_now_iso(),
            "max_hold_until_utc": plan.get("max_hold_until_utc"),
            "trigger_type": plan.get("trigger_type"),
            "candidate_score": self._float_or_none(plan.get("candidate_score")),
            "qty": self._float_or_none(plan.get("qty")),
            "entry_reference_price": self._float_or_none(plan.get("entry_reference_price")),
            "risk_per_share": self._float_or_none(plan.get("risk_per_share")),
            "risk_budget": self._float_or_none(plan.get("risk_budget")),
            "stop_price": self._float_or_none(plan.get("stop_price")),
            "target_price": self._float_or_none(plan.get("target_price")),
            "status": status or plan.get("alpaca_status") or plan.get("status") or "planned",
            "alpaca_order_id": plan.get("alpaca_order_id"),
            "dry_run": 1 if bool(plan.get("dry_run", False)) else 0,
            "strategy_variant": plan.get("strategy_variant"),
            "strategy_preset": plan.get("strategy_preset"),
            "strategy_profile": plan.get("strategy_profile"),
            "quality_gate": plan.get("quality_gate"),
            "pattern_mode": plan.get("pattern_mode"),
            "selection_mode": plan.get("selection_mode"),
            "realized_pl": self._float_or_none(plan.get("realized_pl")),
            "realized_r": self._float_or_none(plan.get("realized_r")),
            "payload_json": _json_dumps(plan),
        }
        cols = list(data.keys())
        placeholders = ",".join([ph] * len(cols))
        update_cols = [c for c in cols if c != "client_order_id"]
        sql = f"""
            INSERT INTO live_signal_plans ({','.join(cols)}) VALUES ({placeholders})
            ON CONFLICT(client_order_id) DO UPDATE SET {','.join([f'{c}=excluded.{c}' for c in update_cols])}
        """
        self._execute(sql, [data[c] for c in cols])

    def insert_event(self, event: str, payload: dict[str, Any] | None = None, **fields: Any) -> None:
        payload = payload or {}
        ph = self._ph()
        sql = f"""
            INSERT INTO live_events
            (created_at_utc, event, symbol, strategy_side, status, message, signal_time_utc, order_id, client_order_id, payload_json)
            VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
        """
        self._execute(
            sql,
            [
                fields.get("created_at_utc") or utc_now_iso(),
                event,
                fields.get("symbol") or payload.get("symbol"),
                fields.get("strategy_side") or payload.get("strategy_side") or payload.get("side"),
                fields.get("status") or payload.get("status") or payload.get("alpaca_status"),
                fields.get("message") or payload.get("message"),
                fields.get("signal_time_utc") or payload.get("signal_time_utc"),
                fields.get("order_id") or payload.get("alpaca_order_id") or payload.get("id"),
                fields.get("client_order_id") or payload.get("client_order_id"),
                _json_dumps(payload),
            ],
        )

    def upsert_order(self, order: dict[str, Any]) -> None:
        if not order:
            return
        order_id = str(order.get("id") or order.get("order_id") or "")
        if not order_id:
            return
        ph = self._ph()
        data = {
            "order_id": order_id,
            "client_order_id": order.get("client_order_id"),
            "parent_order_id": order.get("parent_order_id"),
            "symbol": str(order.get("symbol") or "").upper(),
            "side": order.get("side"),
            "type": order.get("type"),
            "order_class": order.get("order_class"),
            "qty": self._float_or_none(order.get("qty")),
            "filled_qty": self._float_or_none(order.get("filled_qty")),
            "limit_price": self._float_or_none(order.get("limit_price")),
            "stop_price": self._float_or_none(order.get("stop_price")),
            "filled_avg_price": self._float_or_none(order.get("filled_avg_price")),
            "status": order.get("status"),
            "time_in_force": order.get("time_in_force"),
            "submitted_at": order.get("submitted_at"),
            "filled_at": order.get("filled_at"),
            "expired_at": order.get("expired_at"),
            "canceled_at": order.get("canceled_at"),
            "created_at": order.get("created_at"),
            "updated_at": order.get("updated_at"),
            "raw_json": _json_dumps(order),
        }
        cols = list(data.keys())
        placeholders = ",".join([ph] * len(cols))
        update_cols = [c for c in cols if c != "order_id"]
        sql = f"""
            INSERT INTO live_orders ({','.join(cols)}) VALUES ({placeholders})
            ON CONFLICT(order_id) DO UPDATE SET {','.join([f'{c}=excluded.{c}' for c in update_cols])}
        """
        self._execute(sql, [data[c] for c in cols])

    def upsert_orders_recursive(self, orders: Iterable[dict[str, Any]]) -> None:
        for order in orders or []:
            if not isinstance(order, dict):
                continue
            self.upsert_order(order)
            legs = order.get("legs") or []
            if isinstance(legs, list):
                for leg in legs:
                    if isinstance(leg, dict):
                        self.upsert_order(leg)

    def latest_plan_for_symbol(self, symbol: str) -> dict[str, Any] | None:
        ph = self._ph()
        rows = self._fetch(
            f"SELECT * FROM live_signal_plans WHERE symbol={ph} ORDER BY submitted_at_utc DESC LIMIT 1",
            [symbol.upper()],
        )
        if not rows:
            return None
        row = dict(rows[0])
        try:
            payload = json.loads(row.get("payload_json") or "{}")
            if isinstance(payload, dict):
                payload.update({k: v for k, v in row.items() if v is not None})
                return payload
        except Exception:
            pass
        return row

    def upsert_position(self, position: dict[str, Any], plan: dict[str, Any] | None = None) -> None:
        symbol = str(position.get("symbol") or "").upper()
        if not symbol:
            return
        if plan is None:
            plan = self.latest_plan_for_symbol(symbol) or {}
        now = utc_now_iso()
        ph = self._ph()
        side = "long"
        qty = self._float_or_none(position.get("qty")) or 0.0
        if qty < 0:
            side = "short"
        if plan and plan.get("strategy_side"):
            side = str(plan.get("strategy_side"))
        data = {
            "symbol": symbol,
            "is_open": 1,
            "side": side,
            "qty": abs(qty),
            "avg_entry_price": self._float_or_none(position.get("avg_entry_price")),
            "market_value": self._float_or_none(position.get("market_value")),
            "current_price": self._float_or_none(position.get("current_price")),
            "cost_basis": self._float_or_none(position.get("cost_basis")),
            "unrealized_pl": self._float_or_none(position.get("unrealized_pl")),
            "unrealized_plpc": self._float_or_none(position.get("unrealized_plpc")),
            "opened_at_utc": now,
            "last_seen_utc": now,
            "closed_at_utc": None,
            "max_hold_until_utc": (plan or {}).get("max_hold_until_utc"),
            "entry_client_order_id": (plan or {}).get("client_order_id"),
            "entry_order_id": (plan or {}).get("alpaca_order_id"),
            "raw_json": _json_dumps(position),
        }
        cols = list(data.keys())
        placeholders = ",".join([ph] * len(cols))
        update_cols = [
            "is_open",
            "side",
            "qty",
            "avg_entry_price",
            "market_value",
            "current_price",
            "cost_basis",
            "unrealized_pl",
            "unrealized_plpc",
            "last_seen_utc",
            "closed_at_utc",
            "raw_json",
        ]
        update_exprs = [f"{c}=excluded.{c}" for c in update_cols]
        update_exprs.extend(
            [
                "opened_at_utc=COALESCE(opened_at_utc, excluded.opened_at_utc)",
                "max_hold_until_utc=COALESCE(excluded.max_hold_until_utc, max_hold_until_utc)",
                "entry_client_order_id=COALESCE(excluded.entry_client_order_id, entry_client_order_id)",
                "entry_order_id=COALESCE(excluded.entry_order_id, entry_order_id)",
            ]
        )
        sql = f"""
            INSERT INTO live_positions ({','.join(cols)}) VALUES ({placeholders})
            ON CONFLICT(symbol) DO UPDATE SET {','.join(update_exprs)}
        """
        self._execute(sql, [data[c] for c in cols])

    def mark_missing_positions_closed(self, current_symbols: set[str]) -> list[str]:
        rows = self._fetch("SELECT symbol FROM live_positions WHERE is_open=1")
        missing = [str(r["symbol"]).upper() for r in rows if str(r["symbol"]).upper() not in current_symbols]
        if not missing:
            return []
        ph = self._ph()
        now = utc_now_iso()
        for symbol in missing:
            self._execute(f"UPDATE live_positions SET is_open=0, closed_at_utc={ph} WHERE symbol={ph}", [now, symbol])
            self.insert_event(
                "position_closed_detected",
                {"symbol": symbol, "message": "Position is no longer returned by Alpaca /positions and was marked closed."},
                symbol=symbol,
                status="closed",
            )
        return missing

    def insert_account_snapshot(self, account: dict[str, Any]) -> None:
        if not account:
            return
        ph = self._ph()
        sql = f"""
            INSERT INTO live_account_snapshots
            (created_at_utc, equity, last_equity, buying_power, cash, portfolio_value, status, raw_json)
            VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})
        """
        self._execute(
            sql,
            [
                utc_now_iso(),
                self._float_or_none(account.get("equity")),
                self._float_or_none(account.get("last_equity")),
                self._float_or_none(account.get("buying_power")),
                self._float_or_none(account.get("cash")),
                self._float_or_none(account.get("portfolio_value")),
                account.get("status"),
                _json_dumps(account),
            ],
        )

    def set_state(self, key: str, value: Any) -> None:
        ph = self._ph()
        sql = f"""
            INSERT INTO live_worker_state (key, value_json, updated_at_utc) VALUES ({ph},{ph},{ph})
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at_utc=excluded.updated_at_utc
        """
        self._execute(sql, [key, _json_dumps(value), utc_now_iso()])

    def get_state(self, key: str, default: Any = None) -> Any:
        ph = self._ph()
        rows = self._fetch(f"SELECT value_json FROM live_worker_state WHERE key={ph}", [key])
        if not rows:
            return default
        try:
            return json.loads(rows[0].get("value_json") or "null")
        except Exception:
            return default

    def recent_events(self, limit: int = 100) -> pd.DataFrame:
        rows = self._fetch(f"SELECT * FROM live_events ORDER BY id DESC LIMIT {int(limit)}")
        return pd.DataFrame(rows)

    def open_positions(self) -> pd.DataFrame:
        return pd.DataFrame(self._fetch("SELECT * FROM live_positions WHERE is_open=1 ORDER BY symbol"))

    def closed_positions(self, limit: int = 100) -> pd.DataFrame:
        return pd.DataFrame(self._fetch(f"SELECT * FROM live_positions WHERE is_open=0 ORDER BY closed_at_utc DESC LIMIT {int(limit)}"))

    def recent_orders(self, limit: int = 100) -> pd.DataFrame:
        return pd.DataFrame(self._fetch(f"SELECT * FROM live_orders ORDER BY COALESCE(updated_at, created_at, submitted_at) DESC LIMIT {int(limit)}"))

    def recent_signal_plans(self, limit: int = 100) -> pd.DataFrame:
        return pd.DataFrame(self._fetch(f"SELECT * FROM live_signal_plans ORDER BY submitted_at_utc DESC LIMIT {int(limit)}"))

    def latest_account(self) -> pd.DataFrame:
        rows = self._fetch("SELECT * FROM live_account_snapshots ORDER BY id DESC LIMIT 1")
        return pd.DataFrame(rows)

    def dashboard_snapshot(self) -> dict[str, pd.DataFrame | Any]:
        return {
            "events": self.recent_events(100),
            "open_positions": self.open_positions(),
            "closed_positions": self.closed_positions(100),
            "orders": self.recent_orders(100),
            "plans": self.recent_signal_plans(100),
            "account": self.latest_account(),
            "heartbeat": self.get_state("heartbeat", {}),
            "settings": self.get_state("settings", {}),
        }

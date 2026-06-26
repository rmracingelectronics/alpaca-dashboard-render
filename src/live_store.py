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

    def __init__(self, database_url: str | None = None, initialize_schema: bool = True):
        self.database_url = _coerce_database_url(database_url or os.getenv("DATABASE_URL"))
        self.is_postgres = self.database_url.startswith("postgresql://")
        self.sqlite_path: Path | None = None
        self._state_table_checked = False
        if not self.is_postgres:
            if self.database_url.startswith("sqlite:///"):
                self.sqlite_path = Path(self.database_url.replace("sqlite:///", "", 1))
            else:
                self.sqlite_path = PROJECT_ROOT / "data" / "live_trading" / "live_trading.sqlite3"
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        # The full schema/index migration can be expensive on a live Render DB.
        # Worker/report code still uses the default full initialization.
        # Dashboard settings load/save uses initialize_schema=False so the page
        # cannot get stuck on Dash "Updating..." while indexes/migrations run.
        if initialize_schema:
            self.initialize()

    @contextmanager
    def connect(self):
        if self.is_postgres:
            try:
                import psycopg2
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("DATABASE_URL is Postgres, but psycopg2-binary is not installed.") from exc
            conn = psycopg2.connect(
                self.database_url,
                connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "5")),
                application_name=os.getenv("RENDER_SERVICE_NAME", "alpaca-dashboard"),
                options=f"-c statement_timeout={int(os.getenv('DB_STATEMENT_TIMEOUT_MS', '5000'))}",
            )
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
                strategy_variant TEXT,
                strategy_code TEXT,
                strategy_preset TEXT,
                strategy_profile TEXT,
                quality_gate TEXT,
                pattern_mode TEXT,
                selection_mode TEXT,
                realized_pl REAL,
                realized_r REAL,
                payload_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_candidate_audit (
                audit_key TEXT PRIMARY KEY,
                run_id TEXT,
                created_at_utc TEXT,
                updated_at_utc TEXT,
                candidate_time_utc TEXT,
                candidate_time_et TEXT,
                session_date TEXT,
                symbol TEXT,
                strategy_side TEXT,
                trigger_type TEXT,
                strategy_variant TEXT,
                strategy_code TEXT,
                strategy_preset TEXT,
                strategy_profile TEXT,
                quality_gate TEXT,
                pattern_mode TEXT,
                selection_mode TEXT,
                audit_stage TEXT,
                decision_status TEXT,
                reject_reason TEXT,
                rank_before_filter INTEGER,
                rank_after_filter INTEGER,
                candidate_score REAL,
                final_rank_score REAL,
                entry_reference_price REAL,
                stop_price REAL,
                target_price REAL,
                risk_budget REAL,
                rvol_time_of_day REAL,
                daily_atr14_percent REAL,
                gap_percent REAL,
                day_relative_strength REAL,
                open_relative_strength REAL,
                vwap_extension_atr REAL,
                qqq_change_from_open REAL,
                qqq_day_change_percent REAL,
                atr5m14 REAL,
                entry_candle_pattern TEXT,
                candle_pattern_score REAL,
                payload_json TEXT
            )
            """,

            """
            CREATE TABLE IF NOT EXISTS live_symbol_monitor (
                symbol TEXT PRIMARY KEY,
                run_id TEXT,
                updated_at_utc TEXT,
                latest_bar_time_utc TEXT,
                latest_bar_time_et TEXT,
                session_date TEXT,
                strategy_variant TEXT,
                strategy_code TEXT,
                strategy_label TEXT,
                strategy_preset TEXT,
                strategy_profile TEXT,
                quality_gate TEXT,
                pattern_mode TEXT,
                selection_mode TEXT,
                feed TEXT,
                monitor_status TEXT,
                decision_status TEXT,
                reject_reason TEXT,
                setup_signal INTEGER,
                selected_signal INTEGER,
                in_entry_window INTEGER,
                liquidity_ok INTEGER,
                score_ok INTEGER,
                quality_gate_ok INTEGER,
                candle_ok INTEGER,
                checks_passed INTEGER,
                checks_total INTEGER,
                check_summary TEXT,
                symbol_side_bias TEXT,
                strategy_side TEXT,
                trigger_type TEXT,
                candidate_score REAL,
                final_rank_score REAL,
                close_price REAL,
                volume REAL,
                rvol_time_of_day REAL,
                daily_atr14_percent REAL,
                gap_percent REAL,
                day_relative_strength REAL,
                open_relative_strength REAL,
                vwap_extension_atr REAL,
                qqq_change_from_open REAL,
                qqq_day_change_percent REAL,
                atr5m14 REAL,
                ema9 REAL,
                ema20 REAL,
                session_vwap REAL,
                payload_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS live_strategy_symbol_monitor (
                monitor_key TEXT PRIMARY KEY,
                symbol TEXT,
                run_id TEXT,
                updated_at_utc TEXT,
                latest_bar_time_utc TEXT,
                latest_bar_time_et TEXT,
                session_date TEXT,
                strategy_variant TEXT,
                strategy_code TEXT,
                strategy_label TEXT,
                strategy_preset TEXT,
                strategy_profile TEXT,
                quality_gate TEXT,
                pattern_mode TEXT,
                selection_mode TEXT,
                feed TEXT,
                monitor_status TEXT,
                decision_status TEXT,
                reject_reason TEXT,
                setup_signal INTEGER,
                selected_signal INTEGER,
                in_entry_window INTEGER,
                liquidity_ok INTEGER,
                score_ok INTEGER,
                quality_gate_ok INTEGER,
                candle_ok INTEGER,
                checks_passed INTEGER,
                checks_total INTEGER,
                check_summary TEXT,
                symbol_side_bias TEXT,
                strategy_side TEXT,
                trigger_type TEXT,
                candidate_score REAL,
                final_rank_score REAL,
                close_price REAL,
                volume REAL,
                rvol_time_of_day REAL,
                daily_atr14_percent REAL,
                gap_percent REAL,
                day_relative_strength REAL,
                open_relative_strength REAL,
                vwap_extension_atr REAL,
                qqq_change_from_open REAL,
                qqq_day_change_percent REAL,
                atr5m14 REAL,
                ema9 REAL,
                ema20 REAL,
                session_vwap REAL,
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
        # Migrate older Render/local tables before creating indexes that reference
        # newly added columns.  This keeps redeploys safe even if an older build
        # created a partial live_* schema.
        self._ensure_schema_migrations()
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_live_events_created ON live_events(created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_live_events_symbol ON live_events(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_orders_symbol ON live_orders(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_orders_status ON live_orders(status)",
            "CREATE INDEX IF NOT EXISTS idx_live_plans_symbol ON live_signal_plans(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_plans_status ON live_signal_plans(status)",
            "CREATE INDEX IF NOT EXISTS idx_live_positions_open ON live_positions(is_open)",
            "CREATE INDEX IF NOT EXISTS idx_live_candidate_audit_time ON live_candidate_audit(candidate_time_utc)",
            "CREATE INDEX IF NOT EXISTS idx_live_candidate_audit_status ON live_candidate_audit(decision_status)",
            "CREATE INDEX IF NOT EXISTS idx_live_candidate_audit_symbol ON live_candidate_audit(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_symbol_monitor_updated ON live_symbol_monitor(updated_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_live_symbol_monitor_status ON live_symbol_monitor(monitor_status)",
            "CREATE INDEX IF NOT EXISTS idx_live_symbol_monitor_strategy ON live_symbol_monitor(strategy_variant, quality_gate)",
            "CREATE INDEX IF NOT EXISTS idx_live_strategy_symbol_monitor_symbol ON live_strategy_symbol_monitor(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_strategy_symbol_monitor_strategy ON live_strategy_symbol_monitor(strategy_variant, quality_gate)",
            "CREATE INDEX IF NOT EXISTS idx_live_strategy_symbol_monitor_updated ON live_strategy_symbol_monitor(updated_at_utc)",
        ]
        for stmt in indexes:
            self._execute(stmt)

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
                ("strategy_code", "TEXT"),
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
            "live_symbol_monitor": [
                ("run_id", "TEXT"),
                ("updated_at_utc", "TEXT"),
                ("latest_bar_time_utc", "TEXT"),
                ("latest_bar_time_et", "TEXT"),
                ("session_date", "TEXT"),
                ("strategy_variant", "TEXT"),
                ("strategy_preset", "TEXT"),
                ("strategy_profile", "TEXT"),
                ("quality_gate", "TEXT"),
                ("pattern_mode", "TEXT"),
                ("selection_mode", "TEXT"),
                ("feed", "TEXT"),
                ("monitor_status", "TEXT"),
                ("decision_status", "TEXT"),
                ("reject_reason", "TEXT"),
                ("setup_signal", "INTEGER"),
                ("selected_signal", "INTEGER"),
                ("in_entry_window", "INTEGER"),
                ("liquidity_ok", "INTEGER"),
                ("score_ok", "INTEGER"),
                ("quality_gate_ok", "INTEGER"),
                ("candle_ok", "INTEGER"),
                ("checks_passed", "INTEGER"),
                ("checks_total", "INTEGER"),
                ("check_summary", "TEXT"),
                ("symbol_side_bias", "TEXT"),
                ("strategy_side", "TEXT"),
                ("trigger_type", "TEXT"),
                ("candidate_score", "REAL"),
                ("final_rank_score", "REAL"),
                ("close_price", "REAL"),
                ("volume", "REAL"),
                ("rvol_time_of_day", "REAL"),
                ("daily_atr14_percent", "REAL"),
                ("gap_percent", "REAL"),
                ("day_relative_strength", "REAL"),
                ("open_relative_strength", "REAL"),
                ("vwap_extension_atr", "REAL"),
                ("qqq_change_from_open", "REAL"),
                ("qqq_day_change_percent", "REAL"),
                ("atr5m14", "REAL"),
                ("ema9", "REAL"),
                ("ema20", "REAL"),
                ("session_vwap", "REAL"),
                ("strategy_code", "TEXT"),
                ("strategy_label", "TEXT"),
                ("payload_json", "TEXT"),
            ],
            "live_strategy_symbol_monitor": [
                ("monitor_key", "TEXT"),
                ("symbol", "TEXT"),
                ("run_id", "TEXT"),
                ("updated_at_utc", "TEXT"),
                ("latest_bar_time_utc", "TEXT"),
                ("latest_bar_time_et", "TEXT"),
                ("session_date", "TEXT"),
                ("strategy_variant", "TEXT"),
                ("strategy_code", "TEXT"),
                ("strategy_label", "TEXT"),
                ("strategy_preset", "TEXT"),
                ("strategy_profile", "TEXT"),
                ("quality_gate", "TEXT"),
                ("pattern_mode", "TEXT"),
                ("selection_mode", "TEXT"),
                ("feed", "TEXT"),
                ("monitor_status", "TEXT"),
                ("decision_status", "TEXT"),
                ("reject_reason", "TEXT"),
                ("setup_signal", "INTEGER"),
                ("selected_signal", "INTEGER"),
                ("in_entry_window", "INTEGER"),
                ("liquidity_ok", "INTEGER"),
                ("score_ok", "INTEGER"),
                ("quality_gate_ok", "INTEGER"),
                ("candle_ok", "INTEGER"),
                ("checks_passed", "INTEGER"),
                ("checks_total", "INTEGER"),
                ("check_summary", "TEXT"),
                ("symbol_side_bias", "TEXT"),
                ("strategy_side", "TEXT"),
                ("trigger_type", "TEXT"),
                ("candidate_score", "REAL"),
                ("final_rank_score", "REAL"),
                ("close_price", "REAL"),
                ("volume", "REAL"),
                ("rvol_time_of_day", "REAL"),
                ("daily_atr14_percent", "REAL"),
                ("gap_percent", "REAL"),
                ("day_relative_strength", "REAL"),
                ("open_relative_strength", "REAL"),
                ("vwap_extension_atr", "REAL"),
                ("qqq_change_from_open", "REAL"),
                ("qqq_day_change_percent", "REAL"),
                ("atr5m14", "REAL"),
                ("ema9", "REAL"),
                ("ema20", "REAL"),
                ("session_vwap", "REAL"),
                ("payload_json", "TEXT"),
            ],
            "live_candidate_audit": [
                ("run_id", "TEXT"),
                ("created_at_utc", "TEXT"),
                ("updated_at_utc", "TEXT"),
                ("candidate_time_utc", "TEXT"),
                ("candidate_time_et", "TEXT"),
                ("session_date", "TEXT"),
                ("symbol", "TEXT"),
                ("strategy_side", "TEXT"),
                ("trigger_type", "TEXT"),
                ("strategy_variant", "TEXT"),
                ("strategy_code", "TEXT"),
                ("strategy_preset", "TEXT"),
                ("strategy_profile", "TEXT"),
                ("quality_gate", "TEXT"),
                ("pattern_mode", "TEXT"),
                ("selection_mode", "TEXT"),
                ("audit_stage", "TEXT"),
                ("decision_status", "TEXT"),
                ("reject_reason", "TEXT"),
                ("rank_before_filter", "INTEGER"),
                ("rank_after_filter", "INTEGER"),
                ("candidate_score", "REAL"),
                ("final_rank_score", "REAL"),
                ("entry_reference_price", "REAL"),
                ("stop_price", "REAL"),
                ("target_price", "REAL"),
                ("risk_budget", "REAL"),
                ("rvol_time_of_day", "REAL"),
                ("daily_atr14_percent", "REAL"),
                ("gap_percent", "REAL"),
                ("day_relative_strength", "REAL"),
                ("open_relative_strength", "REAL"),
                ("vwap_extension_atr", "REAL"),
                ("qqq_change_from_open", "REAL"),
                ("qqq_day_change_percent", "REAL"),
                ("atr5m14", "REAL"),
                ("entry_candle_pattern", "TEXT"),
                ("candle_pattern_score", "REAL"),
                ("payload_json", "TEXT"),
            ],
        }
        for table, cols in migrations.items():
            for column, definition in cols:
                self._add_column_if_missing(table, column, definition)

        # Dashboard and worker read these tables continuously.  These indexes keep
        # Render Postgres responsive as live_events/orders/plans grow.
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_live_events_id_desc ON live_events (id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_live_events_created_at ON live_events (created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_live_orders_updated ON live_orders (updated_at, created_at, submitted_at)",
            "CREATE INDEX IF NOT EXISTS idx_live_orders_symbol ON live_orders (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_signal_plans_submitted ON live_signal_plans (submitted_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_live_signal_plans_symbol ON live_signal_plans (symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_positions_open ON live_positions (is_open, symbol)",
            "CREATE INDEX IF NOT EXISTS idx_live_account_snapshots_id_desc ON live_account_snapshots (id DESC)",
            "CREATE INDEX IF NOT EXISTS idx_live_candidate_audit_updated ON live_candidate_audit (updated_at_utc, created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_live_symbol_monitor_updated_desc ON live_symbol_monitor (updated_at_utc DESC)",
            "CREATE INDEX IF NOT EXISTS idx_live_symbol_monitor_status_idx ON live_symbol_monitor (monitor_status)",
            "CREATE INDEX IF NOT EXISTS idx_live_symbol_monitor_strategy_idx ON live_symbol_monitor (strategy_variant, quality_gate)",
            "CREATE INDEX IF NOT EXISTS idx_live_strategy_symbol_monitor_updated_desc ON live_strategy_symbol_monitor (updated_at_utc DESC)",
            "CREATE INDEX IF NOT EXISTS idx_live_strategy_symbol_monitor_strategy_idx ON live_strategy_symbol_monitor (strategy_variant, quality_gate)",
            "CREATE INDEX IF NOT EXISTS idx_live_strategy_symbol_monitor_symbol_idx ON live_strategy_symbol_monitor (symbol)",
        ]
        for sql in index_statements:
            try:
                self._execute(sql)
            except Exception:
                pass

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
            "strategy_code": plan.get("strategy_code"),
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
        """Upsert Alpaca orders and nested bracket/OCO legs.

        Alpaca's /orders?nested=true response often nests take-profit/stop legs
        under the parent bracket order but does not populate parent_order_id on
        each child leg.  Earlier dashboard builds saved those child legs with a
        blank parent_order_id, which made the live trade report unable to pair
        filled exits back to their entry order.  Preserve/repair that parent
        relationship here for all future syncs.
        """

        def visit(order: dict[str, Any], parent_order_id: str | None = None) -> None:
            if not isinstance(order, dict):
                return
            row = dict(order)
            if parent_order_id and not row.get("parent_order_id"):
                row["parent_order_id"] = parent_order_id
            self.upsert_order(row)
            oid = str(row.get("id") or row.get("order_id") or "") or parent_order_id
            legs = row.get("legs") or []
            if isinstance(legs, list):
                for leg in legs:
                    if isinstance(leg, dict):
                        visit(leg, oid)

        for order in orders or []:
            visit(order)

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

    def _ensure_state_table(self) -> None:
        if self._state_table_checked:
            return
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS live_worker_state (
                key TEXT PRIMARY KEY,
                value_json TEXT,
                updated_at_utc TEXT NOT NULL
            )
            """
        )
        self._state_table_checked = True

    def set_state(self, key: str, value: Any) -> None:
        self._ensure_state_table()
        ph = self._ph()
        sql = f"""
            INSERT INTO live_worker_state (key, value_json, updated_at_utc) VALUES ({ph},{ph},{ph})
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at_utc=excluded.updated_at_utc
        """
        self._execute(sql, [key, _json_dumps(value), utc_now_iso()])

    def get_state_row(self, key: str) -> dict[str, Any] | None:
        self._ensure_state_table()
        ph = self._ph()
        rows = self._fetch(f"SELECT key, value_json, updated_at_utc FROM live_worker_state WHERE key={ph}", [key])
        return rows[0] if rows else None

    def get_state_with_updated_at(self, key: str, default: Any = None) -> tuple[Any, str | None]:
        row = self.get_state_row(key)
        if not row:
            return default, None
        try:
            return json.loads(row.get("value_json") or "null"), row.get("updated_at_utc")
        except Exception:
            return default, row.get("updated_at_utc")

    def get_state(self, key: str, default: Any = None) -> Any:
        value, _updated_at = self.get_state_with_updated_at(key, default)
        return value

    def upsert_candidate_audit(self, record: dict[str, Any]) -> None:
        if not record:
            return
        audit_key = str(record.get("audit_key") or "").strip()
        if not audit_key:
            return
        ph = self._ph()
        now = utc_now_iso()
        data = {
            "audit_key": audit_key,
            "run_id": record.get("run_id"),
            "created_at_utc": record.get("created_at_utc") or now,
            "updated_at_utc": now,
            "candidate_time_utc": record.get("candidate_time_utc"),
            "candidate_time_et": record.get("candidate_time_et"),
            "session_date": record.get("session_date"),
            "symbol": str(record.get("symbol") or "").upper(),
            "strategy_side": record.get("strategy_side"),
            "trigger_type": record.get("trigger_type"),
            "strategy_variant": record.get("strategy_variant"),
            "strategy_code": record.get("strategy_code"),
            "strategy_preset": record.get("strategy_preset"),
            "strategy_profile": record.get("strategy_profile"),
            "quality_gate": record.get("quality_gate"),
            "pattern_mode": record.get("pattern_mode"),
            "selection_mode": record.get("selection_mode"),
            "audit_stage": record.get("audit_stage"),
            "decision_status": record.get("decision_status"),
            "reject_reason": record.get("reject_reason"),
            "rank_before_filter": record.get("rank_before_filter"),
            "rank_after_filter": record.get("rank_after_filter"),
            "candidate_score": self._float_or_none(record.get("candidate_score")),
            "final_rank_score": self._float_or_none(record.get("final_rank_score")),
            "entry_reference_price": self._float_or_none(record.get("entry_reference_price")),
            "stop_price": self._float_or_none(record.get("stop_price")),
            "target_price": self._float_or_none(record.get("target_price")),
            "risk_budget": self._float_or_none(record.get("risk_budget")),
            "rvol_time_of_day": self._float_or_none(record.get("rvol_time_of_day")),
            "daily_atr14_percent": self._float_or_none(record.get("daily_atr14_percent")),
            "gap_percent": self._float_or_none(record.get("gap_percent")),
            "day_relative_strength": self._float_or_none(record.get("day_relative_strength")),
            "open_relative_strength": self._float_or_none(record.get("open_relative_strength")),
            "vwap_extension_atr": self._float_or_none(record.get("vwap_extension_atr")),
            "qqq_change_from_open": self._float_or_none(record.get("qqq_change_from_open")),
            "qqq_day_change_percent": self._float_or_none(record.get("qqq_day_change_percent")),
            "atr5m14": self._float_or_none(record.get("atr5m14")),
            "entry_candle_pattern": record.get("entry_candle_pattern"),
            "candle_pattern_score": self._float_or_none(record.get("candle_pattern_score")),
            "payload_json": _json_dumps(record.get("payload", record)),
        }
        cols = list(data.keys())
        placeholders = ",".join([ph] * len(cols))
        update_cols = [c for c in cols if c not in {"audit_key", "created_at_utc"}]
        sql = f"""
            INSERT INTO live_candidate_audit ({','.join(cols)}) VALUES ({placeholders})
            ON CONFLICT(audit_key) DO UPDATE SET {','.join([f'{c}=excluded.{c}' for c in update_cols])}
        """
        self._execute(sql, [data[c] for c in cols])

    def upsert_candidate_audits(self, records: Iterable[dict[str, Any]]) -> None:
        for rec in records or []:
            try:
                self.upsert_candidate_audit(rec)
            except Exception:
                pass

    def _ensure_strategy_symbol_monitor_table(self) -> None:
        """Create/lightly migrate the per-strategy symbol monitor table.

        The dashboard diagnostic endpoints use LiveStore(initialize_schema=False), so
        this helper must be safe and fast: it creates the table when missing and
        adds any missing additive columns without running the full live schema.
        """
        self._execute(
            """
            CREATE TABLE IF NOT EXISTS live_strategy_symbol_monitor (
                monitor_key TEXT PRIMARY KEY,
                symbol TEXT, run_id TEXT, updated_at_utc TEXT, latest_bar_time_utc TEXT, latest_bar_time_et TEXT, session_date TEXT,
                strategy_variant TEXT, strategy_code TEXT, strategy_label TEXT, strategy_preset TEXT, strategy_profile TEXT, quality_gate TEXT, pattern_mode TEXT, selection_mode TEXT, feed TEXT,
                monitor_status TEXT, decision_status TEXT, reject_reason TEXT, setup_signal INTEGER, selected_signal INTEGER, in_entry_window INTEGER,
                liquidity_ok INTEGER, score_ok INTEGER, quality_gate_ok INTEGER, candle_ok INTEGER, checks_passed INTEGER, checks_total INTEGER,
                check_summary TEXT, symbol_side_bias TEXT, strategy_side TEXT, trigger_type TEXT, candidate_score REAL, final_rank_score REAL, close_price REAL,
                volume REAL, rvol_time_of_day REAL, daily_atr14_percent REAL, gap_percent REAL, day_relative_strength REAL, open_relative_strength REAL,
                vwap_extension_atr REAL, qqq_change_from_open REAL, qqq_day_change_percent REAL, atr5m14 REAL, ema9 REAL, ema20 REAL, session_vwap REAL, payload_json TEXT
            )
            """
        )
        for column, definition in [
            ("strategy_code", "TEXT"),
            ("strategy_label", "TEXT"),
            ("strategy_preset", "TEXT"),
            ("strategy_profile", "TEXT"),
            ("quality_gate", "TEXT"),
            ("pattern_mode", "TEXT"),
            ("selection_mode", "TEXT"),
            ("payload_json", "TEXT"),
        ]:
            self._add_column_if_missing("live_strategy_symbol_monitor", column, definition)

    def upsert_strategy_symbol_monitor_snapshot(self, record: dict[str, Any], prepared: dict[str, Any] | None = None) -> None:
        if not record:
            return
        data = dict(prepared or {})
        if not data:
            # Reuse the main normalizer by building only the fields needed here.
            symbol = str(record.get("symbol") or "").upper().strip()
            if not symbol:
                return
            data = {
                "symbol": symbol,
                "run_id": record.get("run_id"),
                "updated_at_utc": record.get("updated_at_utc") or utc_now_iso(),
                "latest_bar_time_utc": record.get("latest_bar_time_utc"),
                "latest_bar_time_et": record.get("latest_bar_time_et"),
                "session_date": record.get("session_date"),
                "strategy_variant": record.get("strategy_variant"),
                "strategy_code": record.get("strategy_code"),
                "strategy_label": record.get("strategy_label"),
                "strategy_preset": record.get("strategy_preset"),
                "strategy_profile": record.get("strategy_profile"),
                "quality_gate": record.get("quality_gate"),
                "pattern_mode": record.get("pattern_mode"),
                "selection_mode": record.get("selection_mode"),
                "feed": record.get("feed"),
                "monitor_status": record.get("monitor_status"),
                "decision_status": record.get("decision_status"),
                "reject_reason": record.get("reject_reason"),
                "setup_signal": int(bool(record.get("setup_signal"))) if record.get("setup_signal") is not None else None,
                "selected_signal": int(bool(record.get("selected_signal"))) if record.get("selected_signal") is not None else None,
                "in_entry_window": int(bool(record.get("in_entry_window"))) if record.get("in_entry_window") is not None else None,
                "liquidity_ok": int(bool(record.get("liquidity_ok"))) if record.get("liquidity_ok") is not None else None,
                "score_ok": int(bool(record.get("score_ok"))) if record.get("score_ok") is not None else None,
                "quality_gate_ok": int(bool(record.get("quality_gate_ok"))) if record.get("quality_gate_ok") is not None else None,
                "candle_ok": int(bool(record.get("candle_ok"))) if record.get("candle_ok") is not None else None,
                "checks_passed": record.get("checks_passed"),
                "checks_total": record.get("checks_total"),
                "check_summary": record.get("check_summary"),
                "symbol_side_bias": record.get("symbol_side_bias"),
                "strategy_side": record.get("strategy_side"),
                "trigger_type": record.get("trigger_type"),
                "candidate_score": self._float_or_none(record.get("candidate_score")),
                "final_rank_score": self._float_or_none(record.get("final_rank_score")),
                "close_price": self._float_or_none(record.get("close_price")),
                "volume": self._float_or_none(record.get("volume")),
                "rvol_time_of_day": self._float_or_none(record.get("rvol_time_of_day")),
                "daily_atr14_percent": self._float_or_none(record.get("daily_atr14_percent")),
                "gap_percent": self._float_or_none(record.get("gap_percent")),
                "day_relative_strength": self._float_or_none(record.get("day_relative_strength")),
                "open_relative_strength": self._float_or_none(record.get("open_relative_strength")),
                "vwap_extension_atr": self._float_or_none(record.get("vwap_extension_atr")),
                "qqq_change_from_open": self._float_or_none(record.get("qqq_change_from_open")),
                "qqq_day_change_percent": self._float_or_none(record.get("qqq_day_change_percent")),
                "atr5m14": self._float_or_none(record.get("atr5m14")),
                "ema9": self._float_or_none(record.get("ema9")),
                "ema20": self._float_or_none(record.get("ema20")),
                "session_vwap": self._float_or_none(record.get("session_vwap")),
                "payload_json": _json_dumps(record.get("payload", record)),
            }
        symbol = str(data.get("symbol") or "").upper().strip()
        if not symbol:
            return
        variant = str(data.get("strategy_variant") or "unknown").strip().lower() or "unknown"
        code = str(data.get("strategy_code") or "").strip().lower()
        gate = str(data.get("quality_gate") or "off").strip().lower() or "off"
        data["monitor_key"] = str(record.get("monitor_key") or f"{symbol}|{code or variant}|{gate}")

        # Scaffold/queued rows are diagnostic placeholders written at the start
        # of a long all-strategies scan.  They must NOT wipe the last real
        # indicator values from the table.  Otherwise the dashboard flickers
        # back to blank RVOL/ATR/RS columns while the worker is still scanning.
        # Preserve the previous completed measurements for the same
        # symbol-strategy key and only update the status/reason/run metadata.
        payload_obj = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        is_scaffold = bool(payload_obj.get("scaffold")) or str(data.get("monitor_status") or "").lower().startswith("scanning - queued")
        if is_scaffold:
            try:
                self._ensure_strategy_symbol_monitor_table()
                existing_rows = self._fetch(f"SELECT * FROM live_strategy_symbol_monitor WHERE monitor_key = {self._ph()} LIMIT 1", [data["monitor_key"]])
                if existing_rows:
                    existing = existing_rows[0]
                    preserve_cols = [
                        "latest_bar_time_utc", "latest_bar_time_et", "session_date",
                        "setup_signal", "selected_signal", "in_entry_window", "liquidity_ok", "score_ok", "quality_gate_ok", "candle_ok",
                        "checks_passed", "checks_total", "symbol_side_bias", "strategy_side", "trigger_type",
                        "candidate_score", "final_rank_score", "close_price", "volume", "rvol_time_of_day", "daily_atr14_percent",
                        "gap_percent", "day_relative_strength", "open_relative_strength", "vwap_extension_atr", "qqq_change_from_open",
                        "qqq_day_change_percent", "atr5m14", "ema9", "ema20", "session_vwap",
                    ]
                    for col in preserve_cols:
                        prev = existing.get(col)
                        if prev not in (None, ""):
                            data[col] = prev
                    prev_summary = str(existing.get("check_summary") or "").strip()
                    queued_msg = str(data.get("check_summary") or data.get("reject_reason") or "Queued for current scan.").strip()
                    if prev_summary:
                        data["check_summary"] = f"{queued_msg} Last completed values are carried forward until this strategy finishes scanning. Previous: {prev_summary}"[:1000]
                    else:
                        data["check_summary"] = f"{queued_msg} Last completed values are carried forward until this strategy finishes scanning."[:1000]
                    data["payload_json"] = _json_dumps({
                        "scaffold": True,
                        "carried_forward_previous_values": True,
                        "queued_reason": queued_msg,
                        "previous_run_id": existing.get("run_id"),
                        "current_run_id": data.get("run_id"),
                        "previous_updated_at_utc": existing.get("updated_at_utc"),
                    })
            except Exception:
                pass

        cols = ["monitor_key"] + [c for c in data.keys() if c != "monitor_key"]
        ph = self._ph()
        self._ensure_strategy_symbol_monitor_table()
        placeholders = ",".join([ph] * len(cols))
        update_cols = [c for c in cols if c != "monitor_key"]
        sql = f"""
            INSERT INTO live_strategy_symbol_monitor ({','.join(cols)}) VALUES ({placeholders})
            ON CONFLICT(monitor_key) DO UPDATE SET {','.join([f'{c}=excluded.{c}' for c in update_cols])}
        """
        self._execute(sql, [data.get(c) for c in cols])


    def upsert_symbol_monitor_snapshot(self, record: dict[str, Any]) -> None:
        """Persist one lightweight per-symbol live indicator/decision snapshot.

        The worker writes this table; the Dash app only reads it.  It is one row
        per monitored symbol, so it stays small and safe to refresh on mobile.
        """
        if not record:
            return
        symbol = str(record.get("symbol") or "").upper().strip()
        if not symbol:
            return
        ph = self._ph()
        now = utc_now_iso()
        data = {
            "symbol": symbol,
            "run_id": record.get("run_id"),
            "updated_at_utc": record.get("updated_at_utc") or now,
            "latest_bar_time_utc": record.get("latest_bar_time_utc"),
            "latest_bar_time_et": record.get("latest_bar_time_et"),
            "session_date": record.get("session_date"),
            "strategy_variant": record.get("strategy_variant"),
            "strategy_code": record.get("strategy_code"),
            "strategy_label": record.get("strategy_label"),
            "strategy_preset": record.get("strategy_preset"),
            "strategy_profile": record.get("strategy_profile"),
            "quality_gate": record.get("quality_gate"),
            "pattern_mode": record.get("pattern_mode"),
            "selection_mode": record.get("selection_mode"),
            "feed": record.get("feed"),
            "monitor_status": record.get("monitor_status"),
            "decision_status": record.get("decision_status"),
            "reject_reason": record.get("reject_reason"),
            "setup_signal": int(bool(record.get("setup_signal"))) if record.get("setup_signal") is not None else None,
            "selected_signal": int(bool(record.get("selected_signal"))) if record.get("selected_signal") is not None else None,
            "in_entry_window": int(bool(record.get("in_entry_window"))) if record.get("in_entry_window") is not None else None,
            "liquidity_ok": int(bool(record.get("liquidity_ok"))) if record.get("liquidity_ok") is not None else None,
            "score_ok": int(bool(record.get("score_ok"))) if record.get("score_ok") is not None else None,
            "quality_gate_ok": int(bool(record.get("quality_gate_ok"))) if record.get("quality_gate_ok") is not None else None,
            "candle_ok": int(bool(record.get("candle_ok"))) if record.get("candle_ok") is not None else None,
            "checks_passed": record.get("checks_passed"),
            "checks_total": record.get("checks_total"),
            "check_summary": record.get("check_summary"),
            "symbol_side_bias": record.get("symbol_side_bias"),
            "strategy_side": record.get("strategy_side"),
            "trigger_type": record.get("trigger_type"),
            "candidate_score": self._float_or_none(record.get("candidate_score")),
            "final_rank_score": self._float_or_none(record.get("final_rank_score")),
            "close_price": self._float_or_none(record.get("close_price")),
            "volume": self._float_or_none(record.get("volume")),
            "rvol_time_of_day": self._float_or_none(record.get("rvol_time_of_day")),
            "daily_atr14_percent": self._float_or_none(record.get("daily_atr14_percent")),
            "gap_percent": self._float_or_none(record.get("gap_percent")),
            "day_relative_strength": self._float_or_none(record.get("day_relative_strength")),
            "open_relative_strength": self._float_or_none(record.get("open_relative_strength")),
            "vwap_extension_atr": self._float_or_none(record.get("vwap_extension_atr")),
            "qqq_change_from_open": self._float_or_none(record.get("qqq_change_from_open")),
            "qqq_day_change_percent": self._float_or_none(record.get("qqq_day_change_percent")),
            "atr5m14": self._float_or_none(record.get("atr5m14")),
            "ema9": self._float_or_none(record.get("ema9")),
            "ema20": self._float_or_none(record.get("ema20")),
            "session_vwap": self._float_or_none(record.get("session_vwap")),
            "payload_json": _json_dumps(record.get("payload", record)),
        }
        cols = list(data.keys())
        placeholders = ",".join([ph] * len(cols))
        update_cols = [c for c in cols if c != "symbol"]
        sql = f"""
            INSERT INTO live_symbol_monitor ({','.join(cols)}) VALUES ({placeholders})
            ON CONFLICT(symbol) DO UPDATE SET {','.join([f'{c}=excluded.{c}' for c in update_cols])}
        """
        self._execute(sql, [data[c] for c in cols])
        try:
            self.upsert_strategy_symbol_monitor_snapshot(record, data)
        except Exception:
            pass

    def upsert_symbol_monitor_snapshots(self, records: Iterable[dict[str, Any]]) -> None:
        rows = list(records or [])
        if not rows:
            return
        # Keep every in-progress run until the worker has completed the full
        # all-strategies scan.  Older versions deleted the previous completed
        # run as soon as the first non-scaffold strategy batch arrived.  That made
        # the dashboard show a partial run such as 3 strategies x 16 symbols while
        # the worker was still scanning.  We now keep prior runs and let
        # latest_symbol_monitor() choose the newest complete run when possible.
        def _is_scaffold_row(rec: dict[str, Any]) -> bool:
            payload_obj = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
            return bool(payload_obj.get("scaffold")) or str(rec.get("monitor_status") or "").lower().startswith("scanning - queued")

        for rec in rows:
            try:
                payload_obj = rec.get("payload") if isinstance(rec.get("payload"), dict) else {}
                is_scaffold = bool(payload_obj.get("scaffold")) or str(rec.get("monitor_status") or "").lower().startswith("scanning - queued")
                if is_scaffold:
                    # Strategy-level scaffolds are useful for progress visibility,
                    # but the legacy one-row-per-symbol table should keep the last
                    # real per-symbol indicator snapshot instead of being blanked by
                    # a queued placeholder.
                    self.upsert_strategy_symbol_monitor_snapshot(rec)
                else:
                    self.upsert_symbol_monitor_snapshot(rec)
            except Exception:
                pass

    def _fetch_df(self, sql: str, params: Iterable[Any] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
        """Fetch a DataFrame, returning an empty frame if a live table is not created yet.

        The web dashboard can use LiveStore(initialize_schema=False) so page load
        never blocks on schema/index migrations.  On a brand-new Render database
        the worker may not have created all live_* tables yet; the dashboard should
        show empty sections instead of failing the whole page.
        """
        try:
            return pd.DataFrame(self._fetch(sql, params))
        except Exception as exc:
            msg = str(exc).lower()
            if "does not exist" in msg or "no such table" in msg or "undefinedtable" in msg:
                return pd.DataFrame(columns=columns or [])
            raise

    def table_count(self, table: str) -> int | None:
        allowed = {
            "live_events",
            "live_positions",
            "live_orders",
            "live_signal_plans",
            "live_account_snapshots",
            "live_worker_state",
            "live_candidate_audit",
            "live_symbol_monitor",
            "live_strategy_symbol_monitor",
        }
        if table not in allowed:
            raise ValueError(f"Unsupported table for count: {table}")
        try:
            rows = self._fetch(f"SELECT COUNT(*) AS n FROM {table}")
            return int(rows[0].get("n", 0)) if rows else 0
        except Exception as exc:
            msg = str(exc).lower()
            if "does not exist" in msg or "no such table" in msg or "undefinedtable" in msg:
                return None
            raise

    def recent_candidate_audit(self, limit: int = 10000) -> pd.DataFrame:
        return self._fetch_df(
            f"SELECT * FROM live_candidate_audit ORDER BY COALESCE(updated_at_utc, created_at_utc) DESC LIMIT {int(limit)}"
        )

    def _expected_strategy_symbol_monitor_rows(self) -> int:
        """Expected rows for a complete strategy-symbol monitor snapshot."""
        try:
            heartbeat = self.get_state("heartbeat", {}) or {}
            active = int(float(heartbeat.get("active_strategy_count") or 0))
            symbols = int(float(heartbeat.get("symbols") or 0))
            if active > 0 and symbols > 0:
                return active * symbols
        except Exception:
            pass
        return 0

    def _active_monitor_filter_values(self) -> tuple[list[str], list[str], int]:
        """Return active strategy variants/symbols and expected monitor rows.

        live_strategy_symbol_monitor is intentionally keyed by monitor_key
        (symbol + strategy + gate), not by run_id, so it behaves like the latest
        state for every strategy-symbol pair.  The dashboard should therefore
        read the latest row for each active key, not the newest run_id group.
        """
        variants: list[str] = []
        symbols: list[str] = []
        try:
            heartbeat = self.get_state("heartbeat", {}) or {}
            cfg = self.get_state("live_config_override", {}) or {}
            raw_variants = heartbeat.get("active_strategy_variants") or cfg.get("active_strategy_variants") or []
            if isinstance(raw_variants, list):
                variants = [str(v).strip().lower() for v in raw_variants if str(v).strip()]
            raw_symbols = cfg.get("symbols") or []
            if isinstance(raw_symbols, list):
                symbols = [str(s).strip().upper() for s in raw_symbols if str(s).strip()]
            if not symbols and heartbeat.get("symbols"):
                # Heartbeat only has a count, so leave symbols empty rather than guessing.
                symbols = []
        except Exception:
            pass
        variants = list(dict.fromkeys(variants))
        symbols = list(dict.fromkeys(symbols))
        expected = (len(variants) * len(symbols)) if variants and symbols else self._expected_strategy_symbol_monitor_rows()
        return variants, symbols, int(expected or 0)

    def latest_symbol_monitor(self, limit: int = 250) -> pd.DataFrame:
        cols = [
            "monitor_key", "symbol", "updated_at_utc", "latest_bar_time_utc", "latest_bar_time_et", "session_date",
            "strategy_variant", "strategy_code", "strategy_label", "strategy_preset", "strategy_profile", "quality_gate", "pattern_mode",
            "selection_mode", "feed", "monitor_status", "decision_status", "reject_reason",
            "setup_signal", "selected_signal", "in_entry_window", "liquidity_ok", "score_ok",
            "quality_gate_ok", "candle_ok", "checks_passed", "checks_total", "check_summary",
            "symbol_side_bias", "strategy_side", "trigger_type", "candidate_score", "final_rank_score",
            "close_price", "volume", "rvol_time_of_day", "daily_atr14_percent", "gap_percent",
            "day_relative_strength", "open_relative_strength", "vwap_extension_atr", "qqq_change_from_open",
            "qqq_day_change_percent", "atr5m14", "ema9", "ema20", "session_vwap", "payload_json",
        ]
        try:
            # live_strategy_symbol_monitor is a current-state table keyed by
            # monitor_key (symbol|strategy|gate), not a historical run table.
            # Returning all current rows gives the dashboard the latest completed
            # value for every strategy-symbol pair while a new scan progresses.
            df = self._fetch_df(
                f"SELECT * FROM live_strategy_symbol_monitor ORDER BY strategy_variant ASC, symbol ASC LIMIT {int(limit)}",
                columns=cols,
            )
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        old_cols = [c for c in cols if c != "monitor_key"]
        return self._fetch_df(
            f"SELECT * FROM live_symbol_monitor ORDER BY symbol ASC LIMIT {int(limit)}",
            columns=old_cols,
        )

    def prune_strategy_symbol_monitor_runs(self, keep: int = 8) -> None:
        """Delete old monitor runs after a scan completes, keeping recent history."""
        try:
            self._ensure_strategy_symbol_monitor_table()
            rows = self._fetch(
                """
                SELECT run_id, MAX(updated_at_utc) AS max_updated
                FROM live_strategy_symbol_monitor
                WHERE run_id IS NOT NULL AND run_id <> ''
                GROUP BY run_id
                ORDER BY max_updated DESC
                """
            )
            stale = [str(r.get("run_id") or "") for r in rows[int(max(1, keep)):] if str(r.get("run_id") or "")]
            if not stale:
                return
            phs = ",".join([self._ph()] * len(stale))
            self._execute(f"DELETE FROM live_strategy_symbol_monitor WHERE run_id IN ({phs})", stale)
        except Exception:
            pass

    def recent_events(self, limit: int = 100) -> pd.DataFrame:
        return self._fetch_df(f"SELECT * FROM live_events ORDER BY id DESC LIMIT {int(limit)}")

    def open_positions(self) -> pd.DataFrame:
        return self._fetch_df("SELECT * FROM live_positions WHERE is_open=1 ORDER BY symbol")

    def closed_positions(self, limit: int = 100) -> pd.DataFrame:
        return self._fetch_df(f"SELECT * FROM live_positions WHERE is_open=0 ORDER BY closed_at_utc DESC LIMIT {int(limit)}")

    def recent_orders(self, limit: int = 100) -> pd.DataFrame:
        return self._fetch_df(f"SELECT * FROM live_orders ORDER BY COALESCE(updated_at, created_at, submitted_at) DESC LIMIT {int(limit)}")

    def recent_signal_plans(self, limit: int = 100) -> pd.DataFrame:
        return self._fetch_df(f"SELECT * FROM live_signal_plans ORDER BY submitted_at_utc DESC LIMIT {int(limit)}")

    def latest_account(self) -> pd.DataFrame:
        return self._fetch_df("SELECT * FROM live_account_snapshots ORDER BY id DESC LIMIT 1")

    def dashboard_snapshot(self) -> dict[str, pd.DataFrame | Any]:
        return {
            "events": self.recent_events(100),
            # Candidate audit is intentionally not loaded on every dashboard refresh;
            # it can become large and make the Dash page stay in "Updating".
            # Full audit rows are still included when generating the live report ZIP.
            "candidate_audit": pd.DataFrame(),
            "symbol_monitor": self.latest_symbol_monitor(1000),
            "open_positions": self.open_positions(),
            "closed_positions": self.closed_positions(100),
            "orders": self.recent_orders(500),
            "plans": self.recent_signal_plans(500),
            "account": self.latest_account(),
            "heartbeat": self.get_state("heartbeat", {}),
            "settings": self.get_state("settings", {}),
        }

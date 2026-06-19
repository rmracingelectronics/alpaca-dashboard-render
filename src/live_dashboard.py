from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .live_store import LiveStore


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if pd.notna(out):
            return out
    except Exception:
        pass
    return default


def _short_time(value: Any) -> str:
    if value is None or value == "":
        return ""
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return str(value)
    return ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M:%S")


def _age_text(value: Any) -> str:
    if not value:
        return "no heartbeat yet"
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return "unknown age"
    seconds = max(0, int((pd.Timestamp.now(tz="UTC") - ts).total_seconds()))
    if seconds < 90:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}m ago"
    hours = minutes // 60
    return f"{hours}h ago"


def _format_money(value: Any) -> str:
    return f"${_safe_float(value):,.2f}"


def _format_qty(value: Any) -> str:
    qty = _safe_float(value)
    if abs(qty - round(qty)) < 1e-8:
        return f"{int(round(qty)):,}"
    return f"{qty:,.4f}"


def _format_pct_from_decimal(value: Any) -> str:
    return f"{_safe_float(value) * 100:,.2f}%"


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame()
    for col in cols:
        if col not in out.columns:
            out[col] = ""
    return out[cols].copy()


def _apply_time_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for source, target in mapping.items():
        if source in out.columns:
            out[target] = out[source].map(_short_time)
    return out


def format_open_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "side", "qty", "avg_entry_price", "current_price", "market_value", "unrealized_pl", "unrealized_plpc", "opened_at_et", "last_seen_et", "max_hold_until_et"])
    out = _apply_time_columns(df, {"opened_at_utc": "opened_at_et", "last_seen_utc": "last_seen_et", "max_hold_until_utc": "max_hold_until_et"})
    for col in ["avg_entry_price", "current_price", "market_value", "unrealized_pl"]:
        if col in out.columns:
            out[col] = out[col].map(_format_money)
    if "unrealized_plpc" in out.columns:
        out["unrealized_plpc"] = out["unrealized_plpc"].map(_format_pct_from_decimal)
    if "qty" in out.columns:
        out["qty"] = out["qty"].map(_format_qty)
    return _ensure_columns(out, ["symbol", "side", "qty", "avg_entry_price", "current_price", "market_value", "unrealized_pl", "unrealized_plpc", "opened_at_et", "last_seen_et", "max_hold_until_et", "entry_client_order_id"])


def format_closed_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "side", "qty", "avg_entry_price", "current_price", "unrealized_pl", "opened_at_et", "closed_at_et", "entry_client_order_id"])
    out = _apply_time_columns(df, {"opened_at_utc": "opened_at_et", "closed_at_utc": "closed_at_et", "last_seen_utc": "last_seen_et"})
    for col in ["avg_entry_price", "current_price", "market_value", "unrealized_pl"]:
        if col in out.columns:
            out[col] = out[col].map(_format_money)
    if "qty" in out.columns:
        out["qty"] = out["qty"].map(_format_qty)
    return _ensure_columns(out, ["symbol", "side", "qty", "avg_entry_price", "current_price", "unrealized_pl", "opened_at_et", "closed_at_et", "entry_client_order_id"])


def format_signal_plans(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["submitted_at_et", "symbol", "strategy_side", "status", "dry_run", "qty", "entry_reference_price", "stop_price", "target_price", "risk_budget", "signal_time_et", "max_hold_until_et", "client_order_id"])
    out = _apply_time_columns(df, {"submitted_at_utc": "submitted_at_et", "max_hold_until_utc": "max_hold_until_et"})
    for col in ["entry_reference_price", "stop_price", "target_price", "risk_budget"]:
        if col in out.columns:
            out[col] = out[col].map(_format_money)
    if "qty" in out.columns:
        out["qty"] = out["qty"].map(_format_qty)
    if "dry_run" in out.columns:
        out["dry_run"] = out["dry_run"].map(lambda x: "yes" if str(x).lower() in {"1", "true", "yes"} else "no")
    return _ensure_columns(out, ["submitted_at_et", "symbol", "strategy_side", "status", "dry_run", "qty", "entry_reference_price", "stop_price", "target_price", "risk_budget", "signal_time_et", "max_hold_until_et", "client_order_id"])


def format_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["updated_at_et", "symbol", "side", "type", "order_class", "status", "qty", "filled_qty", "limit_price", "stop_price", "filled_avg_price", "client_order_id", "order_id"])
    out = _apply_time_columns(df, {"updated_at": "updated_at_et", "submitted_at": "submitted_at_et", "filled_at": "filled_at_et", "canceled_at": "canceled_at_et"})
    if "updated_at_et" not in out.columns or out["updated_at_et"].eq("").all():
        out["updated_at_et"] = out.get("submitted_at_et", "")
    for col in ["limit_price", "stop_price", "filled_avg_price"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: _format_money(x) if x not in (None, "") and pd.notna(x) else "")
    for col in ["qty", "filled_qty"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: _format_qty(x) if x not in (None, "") and pd.notna(x) else "")
    return _ensure_columns(out, ["updated_at_et", "symbol", "side", "type", "order_class", "status", "qty", "filled_qty", "limit_price", "stop_price", "filled_avg_price", "client_order_id", "order_id"])


def format_events(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["created_at_et", "event", "symbol", "strategy_side", "status", "message", "client_order_id"])
    out = _apply_time_columns(df, {"created_at_utc": "created_at_et"})
    return _ensure_columns(out, ["created_at_et", "event", "symbol", "strategy_side", "status", "message", "client_order_id"])


def _metrics_from_snapshot(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    heartbeat = snapshot.get("heartbeat") if isinstance(snapshot.get("heartbeat"), dict) else {}
    settings = snapshot.get("settings") if isinstance(snapshot.get("settings"), dict) else {}
    account = snapshot.get("account")
    open_positions = snapshot.get("open_positions")
    plans = snapshot.get("plans")
    orders = snapshot.get("orders")
    events = snapshot.get("events")

    latest_account = account.iloc[0].to_dict() if isinstance(account, pd.DataFrame) and not account.empty else {}
    latest_event = events.iloc[0].to_dict() if isinstance(events, pd.DataFrame) and not events.empty else {}
    risk = settings.get("risk", {}) if isinstance(settings, dict) else {}
    live = settings.get("live", {}) if isinstance(settings, dict) else {}

    status = str(heartbeat.get("status") or "not started")
    updated_at = heartbeat.get("updated_at_utc")
    dry_run = heartbeat.get("dry_run", live.get("dry_run", ""))
    dry_run_text = "dry-run" if str(dry_run).lower() in {"1", "true", "yes"} else "submitting paper orders"
    open_count = len(open_positions) if isinstance(open_positions, pd.DataFrame) else 0
    plan_count = len(plans) if isinstance(plans, pd.DataFrame) else 0
    order_count = len(orders) if isinstance(orders, pd.DataFrame) else 0

    return [
        {"label": "Worker", "value": status.title(), "help": f"Heartbeat {_age_text(updated_at)}; {dry_run_text}"},
        {"label": "Paper Equity", "value": _format_money(latest_account.get("equity")), "help": f"Buying power {_format_money(latest_account.get('buying_power'))}"},
        {"label": "Open Positions", "value": str(open_count), "help": "Shared DB state from worker / Alpaca"},
        {"label": "Signal Plans", "value": str(plan_count), "help": f"Recent orders tracked: {order_count}"},
        {"label": "Risk Mode", "value": str(risk.get("position_sizing_mode") or "--"), "help": f"Fixed risk {_format_money(risk.get('fixed_risk_dollars'))}"},
        {"label": "Last Event", "value": str(latest_event.get("event") or "--")[:28], "help": str(latest_event.get("message") or "")[:60]},
    ]


def load_live_paper_snapshot(days: int = 7) -> dict[str, Any]:
    """Load live paper-trading state written by the Render worker.

    The web service and worker are separate Render services, so this reads from
    the shared DATABASE_URL store instead of relying on a local file.
    """
    try:
        store = LiveStore()
        snapshot = store.dashboard_snapshot()
        metrics = _metrics_from_snapshot(snapshot)
        heartbeat = snapshot.get("heartbeat") if isinstance(snapshot.get("heartbeat"), dict) else {}
        if heartbeat:
            status = f"Live monitor refreshed {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}. Worker status: {heartbeat.get('status', 'unknown')} ({_age_text(heartbeat.get('updated_at_utc'))})."
        else:
            status = "Live monitor is connected to the shared store, but the worker has not written a heartbeat yet."
        return {
            "ok": True,
            "status": status,
            "metrics": metrics,
            "open_positions": format_open_positions(snapshot.get("open_positions", pd.DataFrame())),
            "plans": format_signal_plans(snapshot.get("plans", pd.DataFrame())),
            "orders": format_orders(snapshot.get("orders", pd.DataFrame())),
            "closed_positions": format_closed_positions(snapshot.get("closed_positions", pd.DataFrame())),
            "events": format_events(snapshot.get("events", pd.DataFrame())),
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": f"Live paper monitor error: {exc}",
            "metrics": [
                {"label": "Live Monitor", "value": "Error", "help": str(exc)[:80]},
                {"label": "Worker", "value": "--", "help": "Check DATABASE_URL and worker logs"},
                {"label": "Open Positions", "value": "--", "help": ""},
                {"label": "Orders", "value": "--", "help": ""},
            ],
            "open_positions": pd.DataFrame(),
            "plans": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "closed_positions": pd.DataFrame(),
            "events": pd.DataFrame(),
        }

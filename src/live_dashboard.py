from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import zipfile
from typing import Any

import pandas as pd

from .live_store import LiveStore
from .config import REPORTS_DIR


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



def _json_loads_safe(value: Any) -> dict[str, Any]:
    try:
        out = json.loads(value or "{}")
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def build_live_trade_report(days: int = 3650) -> pd.DataFrame:
    """Build an audit report of live/paper trades from the shared store.

    It pairs signal plans with Alpaca parent/exit orders where available and
    includes the strategy/preset/gate used for the trade plus signal-time
    indicator context captured at entry.  Open trades show unrealized P/L;
    closed trades show realized P/L when filled exit legs are available.
    """
    store = LiveStore()
    plans = store.recent_signal_plans(5000)
    orders = store.recent_orders(10000)
    open_positions = store.open_positions()
    closed_positions = store.closed_positions(5000)
    if plans is None or plans.empty:
        return pd.DataFrame()
    if orders is None:
        orders = pd.DataFrame()
    if open_positions is None:
        open_positions = pd.DataFrame()
    if closed_positions is None:
        closed_positions = pd.DataFrame()

    by_client = {}
    by_order_id = {}
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    if not orders.empty:
        for _, o in orders.iterrows():
            od = o.to_dict()
            oid = str(od.get("order_id") or "")
            cid = str(od.get("client_order_id") or "")
            parent = str(od.get("parent_order_id") or "")
            if oid:
                by_order_id[oid] = od
            if cid:
                by_client[cid] = od
            if parent:
                children_by_parent.setdefault(parent, []).append(od)

    open_by_symbol = {str(r.get("symbol", "")).upper(): r.to_dict() for _, r in open_positions.iterrows()} if not open_positions.empty else {}
    closed_by_client = {str(r.get("entry_client_order_id", "")): r.to_dict() for _, r in closed_positions.iterrows()} if not closed_positions.empty and "entry_client_order_id" in closed_positions.columns else {}

    rows = []
    for _, p in plans.iterrows():
        plan = p.to_dict()
        payload = _json_loads_safe(plan.get("payload_json"))
        ctx = payload.get("signal_context") if isinstance(payload.get("signal_context"), dict) else {}
        cid = str(plan.get("client_order_id") or payload.get("client_order_id") or "")
        sym = str(plan.get("symbol") or payload.get("symbol") or "").upper()
        side = str(plan.get("strategy_side") or payload.get("strategy_side") or "").lower()
        entry_order = None
        if cid and cid in by_client:
            entry_order = by_client[cid]
        elif str(plan.get("alpaca_order_id") or "") in by_order_id:
            entry_order = by_order_id[str(plan.get("alpaca_order_id"))]
        entry_order_id = str((entry_order or {}).get("order_id") or plan.get("alpaca_order_id") or payload.get("alpaca_order_id") or "")
        entry_fill = _safe_float((entry_order or {}).get("filled_avg_price"), _safe_float(plan.get("entry_reference_price"), 0.0))
        qty = _safe_float((entry_order or {}).get("filled_qty"), _safe_float(plan.get("qty"), 0.0))
        exit_orders = [x for x in children_by_parent.get(entry_order_id, []) if str(x.get("status", "")).lower() == "filled"]
        exit_orders.sort(key=lambda x: str(x.get("filled_at") or x.get("updated_at") or ""))
        exit_order = exit_orders[-1] if exit_orders else None
        exit_fill = _safe_float((exit_order or {}).get("filled_avg_price"), 0.0) if exit_order else 0.0
        realized_pl = None
        realized_r = None
        status = str(plan.get("status") or payload.get("status") or "planned")
        exit_time = (exit_order or {}).get("filled_at") or (exit_order or {}).get("updated_at") or ""
        if exit_order and qty > 0 and entry_fill > 0 and exit_fill > 0:
            realized_pl = (exit_fill - entry_fill) * qty if side == "long" else (entry_fill - exit_fill) * qty
            risk_budget = _safe_float(plan.get("risk_budget"), _safe_float(payload.get("risk_budget"), 0.0))
            realized_r = realized_pl / risk_budget if risk_budget > 0 else None
            status = "closed_filled_exit"
        unrealized_pl = None
        if sym in open_by_symbol:
            unrealized_pl = _safe_float(open_by_symbol[sym].get("unrealized_pl"), 0.0)
            if realized_pl is None:
                status = "open"
        if realized_pl is None and cid in closed_by_client:
            cp = closed_by_client[cid]
            # Older rows only know the last seen unrealized P/L. Keep it as an
            # approximate close P/L instead of hiding it.
            approx = cp.get("unrealized_pl")
            if approx not in (None, ""):
                realized_pl = _safe_float(approx, 0.0)
                risk_budget = _safe_float(plan.get("risk_budget"), _safe_float(payload.get("risk_budget"), 0.0))
                realized_r = realized_pl / risk_budget if risk_budget > 0 else None
                status = "closed_detected_approx_pl"
        trade_pl = realized_pl if realized_pl is not None else unrealized_pl
        row = {
            "submitted_at_utc": plan.get("submitted_at_utc"),
            "signal_time_utc": plan.get("signal_time_utc"),
            "signal_time_et": plan.get("signal_time_et"),
            "symbol": sym,
            "strategy_side": side,
            "trigger_type": plan.get("trigger_type"),
            "strategy_variant": plan.get("strategy_variant") or payload.get("strategy_variant"),
            "strategy_preset": plan.get("strategy_preset") or payload.get("strategy_preset"),
            "strategy_profile": plan.get("strategy_profile") or payload.get("strategy_profile"),
            "quality_gate": plan.get("quality_gate") or payload.get("quality_gate"),
            "pattern_mode": plan.get("pattern_mode") or payload.get("pattern_mode"),
            "selection_mode": plan.get("selection_mode") or payload.get("selection_mode"),
            "status": status,
            "qty": qty,
            "entry_fill_or_reference": entry_fill,
            "exit_fill": exit_fill if exit_order else "",
            "risk_budget": _safe_float(plan.get("risk_budget"), _safe_float(payload.get("risk_budget"), 0.0)),
            "risk_per_share": _safe_float(plan.get("risk_per_share"), _safe_float(payload.get("risk_per_share"), 0.0)),
            "target_price": _safe_float(plan.get("target_price"), _safe_float(payload.get("target_price"), 0.0)),
            "stop_price": _safe_float(plan.get("stop_price"), _safe_float(payload.get("stop_price"), 0.0)),
            "realized_pl": realized_pl if realized_pl is not None else "",
            "realized_r": realized_r if realized_r is not None else "",
            "unrealized_pl": unrealized_pl if unrealized_pl is not None else "",
            "trade_pl_live": trade_pl if trade_pl is not None else "",
            "entry_order_id": entry_order_id,
            "exit_order_id": (exit_order or {}).get("order_id", ""),
            "exit_time": exit_time,
            "client_order_id": cid,
        }
        for k, v in ctx.items():
            row[f"signal_{k}"] = v
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["submitted_at_utc"] = pd.to_datetime(out["submitted_at_utc"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days or 3650))
        out = out[(out["submitted_at_utc"].isna()) | (out["submitted_at_utc"] >= cutoff)].copy()
        out = out.sort_values("submitted_at_utc", ascending=False)
    return out.reset_index(drop=True)


def generate_live_report_zip(days: int = 3650) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = REPORTS_DIR / f"live_paper_report_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    store = LiveStore()
    trade_report = build_live_trade_report(days=days)
    tables = {
        "live_trade_report.csv": trade_report,
        "live_signal_plans.csv": store.recent_signal_plans(5000),
        "live_orders.csv": store.recent_orders(10000),
        "live_open_positions.csv": store.open_positions(),
        "live_closed_positions.csv": store.closed_positions(5000),
        "live_events.csv": store.recent_events(5000),
    }
    for name, df in tables.items():
        clean = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
        for col in clean.columns:
            if pd.api.types.is_datetime64_any_dtype(clean[col]):
                clean[col] = clean[col].astype(str)
        clean.to_csv(folder / name, index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "days": days,
        "settings": store.get_state("settings", {}),
        "live_config_override": store.get_state("live_config_override", {}),
        "heartbeat": store.get_state("heartbeat", {}),
        "notes": [
            "live_trade_report.csv includes each paper/live strategy trade with strategy_variant, strategy_preset, quality_gate, realized/unrealized P/L, and signal-time indicators where captured.",
            "This report is generated from the shared DATABASE_URL live store, so it works on Render when the web and worker use the same database.",
        ],
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    zip_path = REPORTS_DIR / f"{folder.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in folder.iterdir():
            zf.write(file, arcname=file.name)
    latest = REPORTS_DIR / "latest_live_paper_report.zip"
    try:
        latest.write_bytes(zip_path.read_bytes())
    except Exception:
        pass
    return zip_path


def format_live_trade_report(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["submitted_at_et", "symbol", "strategy_side", "strategy_variant", "quality_gate", "status", "trade_pl_live", "realized_r", "qty", "entry_fill_or_reference", "exit_fill", "trigger_type", "client_order_id"])
    out = _apply_time_columns(df, {"submitted_at_utc": "submitted_at_et", "signal_time_utc": "signal_time_utc_fmt"})
    for col in ["trade_pl_live", "realized_pl", "unrealized_pl", "entry_fill_or_reference", "exit_fill", "risk_budget"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: _format_money(x) if x not in (None, "") and pd.notna(x) else "")
    if "realized_r" in out.columns:
        out["realized_r"] = out["realized_r"].map(lambda x: f"{_safe_float(x):.2f}R" if x not in (None, "") and pd.notna(x) else "")
    if "qty" in out.columns:
        out["qty"] = out["qty"].map(lambda x: _format_qty(x) if x not in (None, "") and pd.notna(x) else "")
    return _ensure_columns(out, ["submitted_at_et", "symbol", "strategy_side", "strategy_variant", "quality_gate", "status", "trade_pl_live", "realized_r", "qty", "entry_fill_or_reference", "exit_fill", "trigger_type", "client_order_id"])

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

    trade_report = snapshot.get("trade_report") if isinstance(snapshot.get("trade_report"), pd.DataFrame) else pd.DataFrame()
    live_pl = 0.0
    if not trade_report.empty and "trade_pl_live" in trade_report.columns:
        live_pl = pd.to_numeric(trade_report["trade_pl_live"], errors="coerce").fillna(0.0).sum()
    current_variant = heartbeat.get("strategy_variant") or live.get("strategy_variant") or "--"
    current_gate = heartbeat.get("quality_gate") or "--"
    return [
        {"label": "Worker", "value": status.title(), "help": f"Heartbeat {_age_text(updated_at)}; {dry_run_text}"},
        {"label": "Live Strategy", "value": str(current_variant)[:28], "help": f"Gate {current_gate}; source {heartbeat.get('config_source', live.get('live_config_source', '--'))}"},
        {"label": "Paper Equity", "value": _format_money(latest_account.get("equity")), "help": f"Buying power {_format_money(latest_account.get('buying_power'))}"},
        {"label": "Live Trade P/L", "value": _format_money(live_pl), "help": "Realized where exits are filled, otherwise open unrealized P/L"},
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
        snapshot["trade_report"] = build_live_trade_report(days=days)
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
            "trade_report": format_live_trade_report(snapshot.get("trade_report", pd.DataFrame())),
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
            "trade_report": pd.DataFrame(),
            "events": pd.DataFrame(),
        }

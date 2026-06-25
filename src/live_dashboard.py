from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
import zipfile
from typing import Any

import pandas as pd

from .live_store import LiveStore
from .config import REPORTS_DIR
from .live_engine import all_live_strategy_specs


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


_STRATEGY_CID_RE = re.compile(r"^(rmv\d+|rm\d+)-(?:(?P<strategy_code>[a-z0-9]{2,8})-)?(?P<symbol>[A-Z0-9.]+)-(?P<stamp>\d{12})-(?P<side>[ls])", flags=re.IGNORECASE)


def _strategy_spec_by_code(code: Any) -> dict[str, Any]:
    code_l = str(code or "").strip().lower()
    for spec in all_live_strategy_specs():
        if code_l and code_l == str(spec.get("code", "")).lower():
            return dict(spec)
    return {}


def _parse_strategy_client_order_id(client_order_id: Any) -> dict[str, Any] | None:
    cid = str(client_order_id or "").strip()
    if not cid:
        return None
    match = _STRATEGY_CID_RE.match(cid)
    if not match:
        return None
    symbol = match.group("symbol")
    stamp = match.group("stamp")
    side_code = match.group("side")
    strategy_code = (match.group("strategy_code") or "").lower()
    spec = _strategy_spec_by_code(strategy_code)
    try:
        ts_et = pd.Timestamp(datetime.strptime(stamp, "%Y%m%d%H%M"), tz="America/New_York")
        signal_time_utc = ts_et.tz_convert("UTC").isoformat()
        signal_time_et = ts_et.strftime("%Y-%m-%d %H:%M")
    except Exception:
        signal_time_utc = ""
        signal_time_et = ""
    return {
        "client_order_id": cid,
        "strategy_code": strategy_code,
        "strategy_variant": spec.get("variant", ""),
        "strategy_preset": spec.get("preset", ""),
        "quality_gate": spec.get("quality_gate", ""),
        "symbol": symbol.upper(),
        "strategy_side": "long" if side_code.lower() == "l" else "short",
        "alpaca_side": "buy" if side_code.lower() == "l" else "sell",
        "signal_time_utc": signal_time_utc,
        "signal_time_et": signal_time_et,
    }


def _order_row_from_raw(raw: dict[str, Any], parent_order_id: str = "") -> dict[str, Any]:
    """Convert an Alpaca raw order/leg JSON object into live_orders-like fields."""
    if not isinstance(raw, dict):
        return {}
    return {
        "order_id": raw.get("id") or raw.get("order_id") or "",
        "client_order_id": raw.get("client_order_id"),
        "parent_order_id": raw.get("parent_order_id") or parent_order_id or None,
        "symbol": str(raw.get("symbol") or "").upper(),
        "side": raw.get("side"),
        "type": raw.get("type") or raw.get("order_type"),
        "order_class": raw.get("order_class"),
        "qty": raw.get("qty"),
        "filled_qty": raw.get("filled_qty"),
        "limit_price": raw.get("limit_price"),
        "stop_price": raw.get("stop_price"),
        "filled_avg_price": raw.get("filled_avg_price"),
        "status": raw.get("status"),
        "time_in_force": raw.get("time_in_force"),
        "submitted_at": raw.get("submitted_at"),
        "filled_at": raw.get("filled_at"),
        "expired_at": raw.get("expired_at"),
        "canceled_at": raw.get("canceled_at"),
        "created_at": raw.get("created_at"),
        "updated_at": raw.get("updated_at"),
        "raw_json": _json_dumps_for_diag(raw),
    }


def _json_dumps_for_diag(value: Any) -> str:
    try:
        return json.dumps(value, default=str, sort_keys=True)
    except Exception:
        return "{}"


def _augment_orders_with_nested_legs(orders: pd.DataFrame) -> pd.DataFrame:
    """Return order rows plus nested Alpaca bracket legs with parent_order_id restored.

    Alpaca's nested order response often includes exit legs inside the parent
    order JSON, but the leg objects do not always arrive with parent_order_id set.
    Older dashboard code only joined exits by parent_order_id, so closed trades
    looked like pending/open trades and P/L disappeared.  This reconstructs that
    parent relationship from the raw nested JSON.
    """
    if orders is None or orders.empty:
        return pd.DataFrame()
    by_id: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []

    def add_row(row: dict[str, Any]) -> None:
        oid = str(row.get("order_id") or row.get("id") or "")
        if oid:
            old = by_id.get(oid, {})
            merged = dict(old)
            merged.update({k: v for k, v in row.items() if v not in (None, "", "\\N")})
            if not merged.get("parent_order_id") and row.get("parent_order_id"):
                merged["parent_order_id"] = row.get("parent_order_id")
            by_id[oid] = merged
        else:
            anonymous.append(row)

    for _, rec in orders.iterrows():
        od = rec.to_dict()
        add_row(od)
        raw = _json_loads_safe(od.get("raw_json"))
        parent_id = str(od.get("order_id") or raw.get("id") or "")
        for leg in raw.get("legs") or []:
            if isinstance(leg, dict):
                add_row(_order_row_from_raw(leg, parent_order_id=parent_id))
    rows = list(by_id.values()) + anonymous
    return pd.DataFrame(rows)


def _plans_from_strategy_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Build fallback signal-plan rows from Alpaca orders when plan rows are absent.

    This keeps older paper trades visible even if live_signal_plans was empty or
    partially lost during a redesign/redeploy.  Only orders with our rm/rmv
    strategy client_order_id shape are converted.
    """
    if orders is None or orders.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, rec in orders.iterrows():
        od = rec.to_dict()
        parsed = _parse_strategy_client_order_id(od.get("client_order_id"))
        if not parsed:
            continue
        status = str(od.get("status") or "")
        qty = _safe_float(od.get("filled_qty"), _safe_float(od.get("qty"), 0.0))
        ref = _safe_float(od.get("filled_avg_price"), _safe_float(od.get("limit_price"), _safe_float(od.get("stop_price"), 0.0)))
        row = {
            **parsed,
            "submitted_at_utc": od.get("submitted_at") or od.get("created_at") or parsed.get("signal_time_utc"),
            "max_hold_until_utc": "",
            "trigger_type": "alpaca_order_history",
            "candidate_score": "",
            "qty": qty,
            "entry_reference_price": ref,
            "risk_per_share": "",
            "risk_budget": "",
            "stop_price": od.get("stop_price") or "",
            "target_price": od.get("limit_price") or "",
            "status": status or "alpaca_order_history",
            "alpaca_order_id": od.get("order_id"),
            "dry_run": 0,
            "payload_json": _json_dumps_for_diag({"derived_from": "live_orders", "order": od}),
            "strategy_variant": parsed.get("strategy_variant", ""),
            "strategy_code": parsed.get("strategy_code", ""),
            "strategy_preset": parsed.get("strategy_preset", ""),
            "strategy_profile": "",
            "quality_gate": parsed.get("quality_gate", ""),
            "pattern_mode": "",
            "selection_mode": "",
            "realized_pl": "",
            "realized_r": "",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _opposite_alpaca_side(strategy_side: str) -> str:
    return "sell" if str(strategy_side).lower() == "long" else "buy"


def _find_fallback_exit_order(entry_order: dict[str, Any], orders: list[dict[str, Any]], strategy_side: str) -> dict[str, Any] | None:
    """Find a filled opposite-side exit when parent_order_id was not preserved."""
    if not entry_order:
        return None
    sym = str(entry_order.get("symbol") or "").upper()
    entry_id = str(entry_order.get("order_id") or "")
    wanted_side = _opposite_alpaca_side(strategy_side)
    entry_ts = pd.to_datetime(entry_order.get("filled_at") or entry_order.get("submitted_at") or entry_order.get("created_at"), utc=True, errors="coerce")
    candidates = []
    for od in orders:
        if str(od.get("order_id") or "") == entry_id:
            continue
        if str(od.get("symbol") or "").upper() != sym:
            continue
        if str(od.get("side") or "").lower() != wanted_side:
            continue
        if str(od.get("status") or "").lower() != "filled":
            continue
        if _safe_float(od.get("filled_qty"), 0.0) <= 0 or _safe_float(od.get("filled_avg_price"), 0.0) <= 0:
            continue
        od_ts = pd.to_datetime(od.get("filled_at") or od.get("updated_at") or od.get("submitted_at"), utc=True, errors="coerce")
        if not pd.isna(entry_ts) and not pd.isna(od_ts) and od_ts < entry_ts:
            continue
        candidates.append(od)
    candidates.sort(key=lambda x: str(x.get("filled_at") or x.get("updated_at") or x.get("submitted_at") or ""))
    return candidates[-1] if candidates else None


def _order_time(order: dict[str, Any], *fields: str) -> pd.Timestamp:
    for field in fields:
        value = order.get(field)
        if value:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if not pd.isna(ts):
                return ts
    return pd.NaT


def _order_status(order: dict[str, Any]) -> str:
    return str((order or {}).get("status") or "").lower()


def _order_side(order: dict[str, Any]) -> str:
    return str((order or {}).get("side") or "").lower()


def _order_symbol(order: dict[str, Any]) -> str:
    return str((order or {}).get("symbol") or "").upper()


def _extract_raw_legs(order: dict[str, Any]) -> list[dict[str, Any]]:
    """Return child legs stored inside the raw Alpaca parent order JSON."""
    raw = order.get("raw_json")
    if not raw:
        return []
    payload = _json_loads_safe(raw)
    legs = payload.get("legs") if isinstance(payload, dict) else None
    if not isinstance(legs, list):
        return []
    parent_id = str(order.get("order_id") or payload.get("id") or "")
    out: list[dict[str, Any]] = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        row = {
            "order_id": leg.get("id") or leg.get("order_id"),
            "client_order_id": leg.get("client_order_id"),
            "parent_order_id": leg.get("parent_order_id") or parent_id,
            "symbol": leg.get("symbol") or order.get("symbol"),
            "side": leg.get("side"),
            "type": leg.get("type") or leg.get("order_type"),
            "order_class": leg.get("order_class") or order.get("order_class"),
            "qty": leg.get("qty"),
            "filled_qty": leg.get("filled_qty"),
            "limit_price": leg.get("limit_price"),
            "stop_price": leg.get("stop_price"),
            "filled_avg_price": leg.get("filled_avg_price"),
            "status": leg.get("status"),
            "time_in_force": leg.get("time_in_force"),
            "submitted_at": leg.get("submitted_at"),
            "filled_at": leg.get("filled_at"),
            "expired_at": leg.get("expired_at"),
            "canceled_at": leg.get("canceled_at"),
            "created_at": leg.get("created_at"),
            "updated_at": leg.get("updated_at"),
            "raw_json": json.dumps(leg, default=str),
        }
        out.append(row)
    return out


def _infer_exit_order(entry_order: dict[str, Any] | None, plan: dict[str, Any], orders: pd.DataFrame, side: str, qty: float) -> dict[str, Any] | None:
    """Find a filled exit when Alpaca did not provide parent_order_id on child legs.

    Historical Render data shows Alpaca bracket exit legs can be returned with a
    blank parent_order_id even though the parent order raw JSON includes the legs.
    The old report paired exits only by parent_order_id, so previous filled paper
    trades appeared as pending with blank P/L.  This fallback pairs by same symbol,
    opposite side, filled status, and timestamp after the entry order.
    """
    if orders is None or orders.empty:
        return None
    sym = str(plan.get("symbol") or (entry_order or {}).get("symbol") or "").upper()
    if not sym:
        return None
    wanted_side = "sell" if str(side).lower() == "long" else "buy"
    entry_ts = _order_time(entry_order or {}, "filled_at", "submitted_at", "created_at", "updated_at")
    if pd.isna(entry_ts):
        entry_ts = pd.to_datetime(plan.get("submitted_at_utc") or plan.get("signal_time_utc"), utc=True, errors="coerce")
    candidates: list[dict[str, Any]] = []
    for _, row in orders.iterrows():
        od = row.to_dict()
        if str(od.get("order_id") or "") == str((entry_order or {}).get("order_id") or ""):
            continue
        if _order_symbol(od) != sym:
            continue
        if _order_side(od) != wanted_side:
            continue
        if _order_status(od) != "filled":
            continue
        fill_ts = _order_time(od, "filled_at", "updated_at", "submitted_at", "created_at")
        if not pd.isna(entry_ts) and not pd.isna(fill_ts) and fill_ts < entry_ts:
            continue
        filled_qty = _safe_float(od.get("filled_qty"), _safe_float(od.get("qty"), 0.0))
        if qty > 0 and filled_qty > 0 and abs(filled_qty - qty) / max(qty, 1.0) > 0.35:
            # Avoid pairing an unrelated same-symbol order of a very different size.
            continue
        candidates.append(od)
    if not candidates:
        return None
    candidates.sort(key=lambda x: str(x.get("filled_at") or x.get("updated_at") or x.get("submitted_at") or ""))
    return candidates[0]


def build_live_trade_report(days: int = 3650) -> pd.DataFrame:
    """Build an audit report of live/paper trades from the shared store.

    It pairs signal plans with Alpaca parent/exit orders where available and
    includes the strategy/preset/gate used for the trade plus signal-time
    indicator context captured at entry.  Open trades show unrealized P/L;
    closed trades show realized P/L when filled exit legs are available.
    """
    store = LiveStore(initialize_schema=False)
    # For the live dashboard refresh, only a recent window is needed.  Full
    # historical exports still use the larger limits through days=3650.
    full_history = int(days or 0) >= 365
    plans = store.recent_signal_plans(5000 if full_history else 500)
    orders = store.recent_orders(10000 if full_history else 1000)
    open_positions = store.open_positions()
    closed_positions = store.closed_positions(5000 if full_history else 500)
    if plans is None or plans.empty:
        return pd.DataFrame()
    if orders is None:
        orders = pd.DataFrame()
    if open_positions is None:
        open_positions = pd.DataFrame()
    if closed_positions is None:
        closed_positions = pd.DataFrame()

    by_client: dict[str, dict[str, Any]] = {}
    by_order_id: dict[str, dict[str, Any]] = {}
    children_by_parent: dict[str, list[dict[str, Any]]] = {}
    if not orders.empty:
        extra_leg_rows: list[dict[str, Any]] = []
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
            extra_leg_rows.extend(_extract_raw_legs(od))
        # Parent raw_json often contains legs with the parent relationship even
        # when the child rows themselves have parent_order_id blank.  Add those
        # extracted leg rows to the lookup maps without changing the DB.
        for od in extra_leg_rows:
            oid = str(od.get("order_id") or "")
            cid = str(od.get("client_order_id") or "")
            parent = str(od.get("parent_order_id") or "")
            if oid and oid not in by_order_id:
                by_order_id[oid] = od
            if cid and cid not in by_client:
                by_client[cid] = od
            if parent:
                children_by_parent.setdefault(parent, []).append(od)

    open_by_symbol = {str(r.get("symbol", "")).upper(): r.to_dict() for _, r in open_positions.iterrows()} if not open_positions.empty else {}
    closed_by_client = {str(r.get("entry_client_order_id", "")): r.to_dict() for _, r in closed_positions.iterrows()} if not closed_positions.empty and "entry_client_order_id" in closed_positions.columns else {}
    heartbeat = store.get_state("heartbeat", {}) or {}
    live_cfg = store.get_state("live_config_override", {}) or {}
    settings = store.get_state("settings", {}) or {}
    live_settings = settings.get("live", {}) if isinstance(settings, dict) else {}
    fallback_variant = heartbeat.get("strategy_variant") or live_cfg.get("strategy_variant") or live_settings.get("strategy_variant") or "unknown"
    fallback_preset = heartbeat.get("strategy_preset") or live_cfg.get("settings_preset") or "unknown"
    fallback_gate = heartbeat.get("quality_gate") or live_cfg.get("live_quality_gate") or "unknown"
    fallback_profile = live_cfg.get("strategy_profile") or "symbol_playbook_v25"
    fallback_pattern = live_cfg.get("live_quality_gate") or ""

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
        exit_orders = [x for x in children_by_parent.get(entry_order_id, []) if _order_status(x) == "filled"]
        if not exit_orders:
            inferred = _infer_exit_order(entry_order, plan, orders, side, qty)
            if inferred:
                exit_orders = [inferred]
        exit_orders.sort(key=lambda x: str(x.get("filled_at") or x.get("updated_at") or x.get("submitted_at") or ""))
        exit_order = exit_orders[-1] if exit_orders else None
        exit_fill = _safe_float((exit_order or {}).get("filled_avg_price"), 0.0) if exit_order else 0.0
        realized_pl = None
        realized_r = None
        entry_status = _order_status(entry_order or {})
        status = str(plan.get("status") or payload.get("status") or "planned")
        exit_time = (exit_order or {}).get("filled_at") or (exit_order or {}).get("updated_at") or ""
        if exit_order and qty > 0 and entry_fill > 0 and exit_fill > 0:
            realized_pl = (exit_fill - entry_fill) * qty if side == "long" else (entry_fill - exit_fill) * qty
            risk_budget = _safe_float(plan.get("risk_budget"), _safe_float(payload.get("risk_budget"), 0.0))
            realized_r = realized_pl / risk_budget if risk_budget > 0 else None
            status = "closed_filled_exit"
        elif entry_status:
            if entry_status == "filled":
                status = "entry_filled_exit_unmatched"
            else:
                status = entry_status
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
            "strategy_variant": plan.get("strategy_variant") or payload.get("strategy_variant") or fallback_variant,
            "strategy_preset": plan.get("strategy_preset") or payload.get("strategy_preset") or fallback_preset,
            "strategy_profile": plan.get("strategy_profile") or payload.get("strategy_profile") or fallback_profile,
            "quality_gate": plan.get("quality_gate") or payload.get("quality_gate") or fallback_gate,
            "pattern_mode": plan.get("pattern_mode") or payload.get("pattern_mode") or fallback_pattern,
            "selection_mode": plan.get("selection_mode") or payload.get("selection_mode") or live_cfg.get("selection_mode") or "seen_so_far_top_n",
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
            "entry_slippage_dollars": (entry_fill - _safe_float(plan.get("entry_reference_price"), entry_fill)) if entry_fill else "",
            "entry_slippage_bps": ((entry_fill - _safe_float(plan.get("entry_reference_price"), entry_fill)) / _safe_float(plan.get("entry_reference_price"), entry_fill) * 10000.0) if entry_fill and _safe_float(plan.get("entry_reference_price"), 0.0) else "",
            "bars_held": int(max(0, (pd.to_datetime(exit_time, utc=True, errors="coerce") - pd.to_datetime(plan.get("signal_time_utc"), utc=True, errors="coerce")).total_seconds() // 300)) if exit_time and not pd.isna(pd.to_datetime(exit_time, utc=True, errors="coerce")) and not pd.isna(pd.to_datetime(plan.get("signal_time_utc"), utc=True, errors="coerce")) else "",
            "mfe_r": "",
            "mae_r": "",
            "max_favorable_price": "",
            "max_adverse_price": "",
            "hit_target": bool(exit_order and str((exit_order or {}).get("type", "")).lower() == "limit"),
            "hit_stop": bool(exit_order and str((exit_order or {}).get("type", "")).lower() == "stop"),
            "hit_max_hold": "max_hold" in str(status).lower(),
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


def _strategy_performance_summary(trade_report: pd.DataFrame) -> pd.DataFrame:
    if trade_report is None or trade_report.empty:
        return pd.DataFrame(columns=["strategy_variant", "strategy_preset", "quality_gate", "trades", "closed_trades", "wins", "losses", "win_rate_pct", "total_pl", "avg_pl", "total_r", "avg_r"])
    df = trade_report.copy()
    for col in ["trade_pl_live", "realized_pl", "realized_r"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    rows = []
    group_cols = [c for c in ["strategy_variant", "strategy_preset", "quality_gate"] if c in df.columns]
    if not group_cols:
        group_cols = ["strategy_variant"]
        df["strategy_variant"] = "unknown"
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        pl = pd.to_numeric(grp.get("trade_pl_live", grp.get("realized_pl", 0)), errors="coerce")
        r = pd.to_numeric(grp.get("realized_r", 0), errors="coerce")
        closed_mask = grp.get("status", pd.Series([""] * len(grp))).astype(str).str.contains("closed|filled_exit|approx", case=False, na=False)
        closed = grp[closed_mask]
        closed_pl = pd.to_numeric(closed.get("trade_pl_live", closed.get("realized_pl", 0)), errors="coerce") if not closed.empty else pd.Series(dtype="float64")
        wins = int((closed_pl > 0).sum())
        losses = int((closed_pl < 0).sum())
        closed_n = int(len(closed))
        row.update({
            "trades": int(len(grp)),
            "closed_trades": closed_n,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round((wins / closed_n * 100.0), 2) if closed_n else "",
            "total_pl": float(pl.fillna(0.0).sum()),
            "avg_pl": float(pl.dropna().mean()) if pl.notna().any() else "",
            "total_r": float(r.fillna(0.0).sum()),
            "avg_r": float(r.dropna().mean()) if r.notna().any() else "",
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["total_pl", "total_r"], ascending=[False, False], ignore_index=True) if rows else pd.DataFrame()


def _candidate_strategy_summary(candidate_audit: pd.DataFrame) -> pd.DataFrame:
    if candidate_audit is None or candidate_audit.empty:
        return pd.DataFrame(columns=["strategy_variant", "quality_gate", "decision_status", "rows"])
    df = candidate_audit.copy()
    group_cols = [c for c in ["strategy_variant", "quality_gate", "decision_status", "reject_reason"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame({"rows": [len(df)]})
    return df.groupby(group_cols, dropna=False).size().reset_index(name="rows").sort_values("rows", ascending=False, ignore_index=True)

def generate_live_report_zip(days: int = 3650) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    folder = REPORTS_DIR / f"live_paper_report_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    store = LiveStore(initialize_schema=False)
    trade_report = build_live_trade_report(days=days)
    candidate_audit = store.recent_candidate_audit(20000)
    symbol_monitor = store.latest_symbol_monitor(1000)
    strategy_summary = _strategy_performance_summary(trade_report)
    candidate_strategy_summary = _candidate_strategy_summary(candidate_audit)
    summary_rows = []
    if isinstance(trade_report, pd.DataFrame) and not trade_report.empty:
        pl = pd.to_numeric(trade_report.get("trade_pl_live", 0), errors="coerce").fillna(0.0)
        r = pd.to_numeric(trade_report.get("realized_r", 0), errors="coerce")
        closed = trade_report[trade_report.get("status", "").astype(str).str.contains("closed|filled_exit|approx", case=False, na=False)] if "status" in trade_report.columns else trade_report
        summary_rows.append({"metric": "trades_in_report", "value": len(trade_report)})
        summary_rows.append({"metric": "closed_trades", "value": len(closed)})
        summary_rows.append({"metric": "total_trade_pl_live", "value": float(pl.sum())})
        summary_rows.append({"metric": "total_realized_r", "value": float(r.fillna(0.0).sum())})
        if "strategy_variant" in trade_report.columns:
            for variant, grp in trade_report.groupby("strategy_variant", dropna=False):
                summary_rows.append({"metric": f"strategy_{variant}_trades", "value": len(grp)})
                summary_rows.append({"metric": f"strategy_{variant}_pl", "value": float(pd.to_numeric(grp.get("trade_pl_live", 0), errors="coerce").fillna(0).sum())})
    if isinstance(candidate_audit, pd.DataFrame) and not candidate_audit.empty:
        summary_rows.append({"metric": "candidate_audit_rows", "value": len(candidate_audit)})
        if "decision_status" in candidate_audit.columns:
            for status, grp in candidate_audit.groupby("decision_status", dropna=False):
                summary_rows.append({"metric": f"candidates_{status}", "value": len(grp)})
    tables = {
        "live_trade_report.csv": trade_report,
        "live_candidate_audit.csv": candidate_audit,
        "live_symbol_monitor.csv": symbol_monitor,
        "live_report_summary.csv": pd.DataFrame(summary_rows),
        "strategy_performance_summary.csv": strategy_summary,
        "candidate_strategy_summary.csv": candidate_strategy_summary,
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
            "live_trade_report.csv includes each paper/live strategy trade with strategy_variant, strategy_preset, quality_gate, realized/unrealized P/L, execution diagnostics, and signal-time indicators where captured.",
            "live_candidate_audit.csv includes accepted and rejected candidates with reject_reason and universal signal-time indicator fields for diagnosing any selected strategy.",
            "live_symbol_monitor.csv is one row per configured live symbol and strategy with latest indicator values, threshold checks, status and reason displayed in the Live Symbol Intelligence panel.",
            "strategy_performance_summary.csv groups filled/planned paper performance by strategy_variant, strategy_preset and quality_gate for multi-day all-strategies experiments.",
            "candidate_strategy_summary.csv groups accepted/rejected candidates by strategy and reject reason so weak filters can be improved even when no order was placed.",
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
    return _ensure_columns(out, ["submitted_at_et", "symbol", "strategy_side", "strategy_variant", "strategy_preset", "quality_gate", "pattern_mode", "status", "trade_pl_live", "realized_r", "qty", "entry_fill_or_reference", "exit_fill", "entry_slippage_bps", "bars_held", "trigger_type", "client_order_id"])

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
    cols = ["submitted_at_et", "symbol", "strategy_side", "strategy_variant", "strategy_preset", "quality_gate", "status", "dry_run", "qty", "entry_reference_price", "stop_price", "target_price", "risk_budget", "signal_time_et", "max_hold_until_et", "client_order_id"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = _apply_time_columns(df, {"submitted_at_utc": "submitted_at_et", "max_hold_until_utc": "max_hold_until_et"})
    for col in ["entry_reference_price", "stop_price", "target_price", "risk_budget"]:
        if col in out.columns:
            out[col] = out[col].map(_format_money)
    if "qty" in out.columns:
        out["qty"] = out["qty"].map(_format_qty)
    if "dry_run" in out.columns:
        out["dry_run"] = out["dry_run"].map(lambda x: "yes" if str(x).lower() in {"1", "true", "yes"} else "no")
    for col in ["strategy_variant", "strategy_preset", "quality_gate"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if x in (None, "None") or pd.isna(x) else str(x))
    return _ensure_columns(out, cols)


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



def _format_decimal(value: Any, places: int = 2, suffix: str = "") -> str:
    try:
        if value in (None, "") or pd.isna(value):
            return ""
        return f"{float(value):,.{places}f}{suffix}"
    except Exception:
        return ""


def format_symbol_monitor(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "symbol", "strategy", "status", "bar_et", "checks", "close", "rvol", "daily_atr_%",
        "day_rs", "open_rs", "vwap_atr", "score", "side", "trigger", "gate", "reason"
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    if "latest_bar_time_et" in out.columns:
        out["bar_et"] = out["latest_bar_time_et"].map(lambda x: "" if x in (None, "", "None") or pd.isna(x) else str(x).replace("T", " ")[:16])
    else:
        out = _apply_time_columns(out, {"latest_bar_time_utc": "bar_et"})
    out["strategy"] = out.get("strategy_variant", "").map(lambda x: "" if x in (None, "None") or pd.isna(x) else str(x).replace("live_", "").replace("_", " ")) if "strategy_variant" in out.columns else ""
    out["status"] = out.get("monitor_status", "").astype(str)
    out["checks"] = out.apply(lambda r: f"{int(_safe_float(r.get('checks_passed'), 0))}/{int(_safe_float(r.get('checks_total'), 0))}" if str(r.get("checks_total", "")) not in {"", "nan", "None"} else "", axis=1)
    for src, dest, places, suffix in [
        ("close_price", "close", 2, ""),
        ("rvol_time_of_day", "rvol", 2, "x"),
        ("daily_atr14_percent", "daily_atr_%", 2, "%"),
        ("day_relative_strength", "day_rs", 2, ""),
        ("open_relative_strength", "open_rs", 2, ""),
        ("vwap_extension_atr", "vwap_atr", 2, " ATR"),
        ("candidate_score", "score", 2, ""),
    ]:
        out[dest] = out[src].map(lambda x, p=places, s=suffix: _format_decimal(x, p, s)) if src in out.columns else ""
    out["side"] = out.get("strategy_side", "").map(lambda x: "" if x in (None, "None") or pd.isna(x) else str(x)) if "strategy_side" in out.columns else ""
    out["trigger"] = out.get("trigger_type", "").map(lambda x: "" if x in (None, "None") or pd.isna(x) else str(x).replace("v25_", "")) if "trigger_type" in out.columns else ""
    out["gate"] = out.get("quality_gate", "").map(lambda x: "" if x in (None, "None") or pd.isna(x) else str(x)) if "quality_gate" in out.columns else ""
    out["reason"] = out.get("check_summary", out.get("reject_reason", "")).map(lambda x: "" if x in (None, "None") or pd.isna(x) else str(x)[:180]) if ("check_summary" in out.columns or "reject_reason" in out.columns) else ""
    return _ensure_columns(out, cols)


def symbol_monitor_summary(df: pd.DataFrame, heartbeat: dict[str, Any] | None = None) -> str:
    heartbeat = heartbeat or {}
    configured = int(heartbeat.get("symbols", 0) or 0)
    run_mode = str(heartbeat.get("strategy_run_mode") or "single")
    active_count = int(_safe_float(heartbeat.get("active_strategy_count"), 1) or 1)
    if df is None or df.empty:
        return f"No symbol monitor snapshots yet. Worker is configured for {configured} symbols and {active_count} active strategy view(s); the table fills after the next worker scan."
    status = df.get("monitor_status", pd.Series([], dtype="object")).astype(str)
    selected = int(status.str.contains("selected", case=False, na=False).sum())
    candidates = int(status.str.contains("candidate", case=False, na=False).sum())
    blocked = int(status.str.contains("blocked", case=False, na=False).sum())
    watching = int(status.str.contains("watching", case=False, na=False).sum())
    paused = int(status.str.contains("paused|waiting", case=False, na=False).sum())
    updated = df.get("updated_at_utc", pd.Series([], dtype="object")).dropna()
    last = _age_text(updated.max()) if not updated.empty else "unknown age"
    symbols_seen = int(df["symbol"].astype(str).str.upper().nunique()) if "symbol" in df.columns else len(df)
    strategies_seen = int(df["strategy_variant"].astype(str).nunique()) if "strategy_variant" in df.columns else active_count
    mode_text = "all strategies" if run_mode == "all_strategies" else "single strategy"
    return f"Monitoring {symbols_seen}/{configured or symbols_seen} symbols across {strategies_seen} strategy view(s) ({mode_text}), {len(df)} rows. Selected/candidate: {selected + candidates}; blocked: {blocked}; watching: {watching}; paused/waiting: {paused}. Last symbol snapshot {last}."

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
    run_mode = str(heartbeat.get("strategy_run_mode") or live.get("strategy_run_mode") or "single")
    active_count = int(_safe_float(heartbeat.get("active_strategy_count"), 1) or 1)
    strategy_value = "All strategies" if run_mode == "all_strategies" else str(current_variant)[:28]
    strategy_help = f"{active_count} active; gate {current_gate}; source {heartbeat.get('config_source', live.get('live_config_source', '--'))}"
    return [
        {"label": "Worker", "value": status.title(), "help": f"Heartbeat {_age_text(updated_at)}; {dry_run_text}"},
        {"label": "Live Strategy", "value": strategy_value, "help": strategy_help},
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
        store = LiveStore(initialize_schema=False)
        snapshot = store.dashboard_snapshot()
        settings = snapshot.get("settings") if isinstance(snapshot.get("settings"), dict) else {}
        live_cfg = settings.get("live", {}) if isinstance(settings.get("live"), dict) else {}
        configured_symbols = [str(s).upper() for s in (live_cfg.get("symbols") or []) if str(s).strip()]
        if configured_symbols and isinstance(snapshot.get("symbol_monitor"), pd.DataFrame) and not snapshot["symbol_monitor"].empty and "symbol" in snapshot["symbol_monitor"].columns:
            snapshot["symbol_monitor"] = snapshot["symbol_monitor"][snapshot["symbol_monitor"]["symbol"].astype(str).str.upper().isin(configured_symbols)].copy()
        snapshot["trade_report"] = build_live_trade_report(days=days)
        metrics = _metrics_from_snapshot(snapshot)
        heartbeat = snapshot.get("heartbeat") if isinstance(snapshot.get("heartbeat"), dict) else {}
        counts = {
            "plans": len(snapshot.get("plans")) if isinstance(snapshot.get("plans"), pd.DataFrame) else 0,
            "orders": len(snapshot.get("orders")) if isinstance(snapshot.get("orders"), pd.DataFrame) else 0,
            "trades": len(snapshot.get("trade_report")) if isinstance(snapshot.get("trade_report"), pd.DataFrame) else 0,
            "events": len(snapshot.get("events")) if isinstance(snapshot.get("events"), pd.DataFrame) else 0,
            "open_positions": len(snapshot.get("open_positions")) if isinstance(snapshot.get("open_positions"), pd.DataFrame) else 0,
            "closed_positions": len(snapshot.get("closed_positions")) if isinstance(snapshot.get("closed_positions"), pd.DataFrame) else 0,
            "symbol_monitor": len(snapshot.get("symbol_monitor")) if isinstance(snapshot.get("symbol_monitor"), pd.DataFrame) else 0,
        }
        if heartbeat:
            status = (
                f"Live monitor refreshed {datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S')}. "
                f"Worker status: {heartbeat.get('status', 'unknown')} ({_age_text(heartbeat.get('updated_at_utc'))}). "
                f"Loaded {counts['trades']} trades, {counts['orders']} orders, {counts['plans']} signal plans, "
                f"{counts['open_positions']} open positions, {counts['closed_positions']} closed positions, "
                f"{counts['symbol_monitor']} symbol monitor rows, and {counts['events']} events from the shared DB."
            )
        else:
            status = (
                "Live monitor is connected to the shared store, but the worker has not written a heartbeat yet. "
                f"Loaded {counts['orders']} orders, {counts['plans']} signal plans and {counts['events']} events from the shared DB."
            )
        return {
            "ok": True,
            "status": status,
            "metrics": metrics,
            "symbol_monitor": format_symbol_monitor(snapshot.get("symbol_monitor", pd.DataFrame())),
            "symbol_monitor_summary": symbol_monitor_summary(snapshot.get("symbol_monitor", pd.DataFrame()), heartbeat),
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
            "symbol_monitor": pd.DataFrame(),
            "symbol_monitor_summary": "Live symbol monitor unavailable because dashboard refresh failed.",
            "open_positions": pd.DataFrame(),
            "plans": pd.DataFrame(),
            "orders": pd.DataFrame(),
            "closed_positions": pd.DataFrame(),
            "trade_report": pd.DataFrame(),
            "events": pd.DataFrame(),
        }

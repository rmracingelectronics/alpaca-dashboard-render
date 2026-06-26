from __future__ import annotations

import os
import json
import traceback
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update, ctx
from flask import send_file, jsonify

from src.backtest import run_backtest
from src.config import AlpacaSettings, StrategyParams
from src.live_dashboard import load_live_paper_snapshot, generate_live_report_zip
from src.live_store import LiveStore
from src.live_engine import live_variant_from_dashboard, all_live_strategy_specs
from src.symbols import WATCHLISTS, parse_symbols

app = Dash(__name__, suppress_callback_exceptions=True, title="Alpaca Momentum Dashboard V38.8", update_title=None)
server = app.server


def _sanitize_diag_value(value):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            key = str(k).lower()
            if "secret" in key or key in {"api_key", "key_id"}:
                out[k] = "***"
            elif key in {"account_id", "account_number", "id"} and isinstance(v, str) and len(v) > 4:
                out[k] = "***" + v[-4:]
            elif key in {"raw_json", "payload_json"}:
                out[k] = "[omitted]"
            else:
                out[k] = _sanitize_diag_value(v)
        return out
    if isinstance(value, list):
        return [_sanitize_diag_value(v) for v in value]
    return value


def _safe_diag_value(value):
    try:
        return json.loads(json.dumps(_sanitize_diag_value(value), default=str))
    except Exception:
        return str(value)


def _df_preview(df, limit: int = 10):
    try:
        if df is None or getattr(df, "empty", True):
            return {"count_loaded": 0, "rows": []}
        clean = df.head(int(limit)).copy()
        for col in ["raw_json", "payload_json"]:
            if col in clean.columns:
                clean = clean.drop(columns=[col])
        for col in clean.columns:
            if pd.api.types.is_datetime64_any_dtype(clean[col]):
                clean[col] = clean[col].astype(str)
        return {"count_loaded": int(len(df)), "rows": _safe_diag_value(clean.to_dict("records"))}
    except Exception as exc:
        return {"count_loaded": 0, "rows": [], "error": str(exc)}


def _settings_snapshot_from_cfg(cfg: dict, existing_settings: dict | None = None) -> dict:
    """Build the worker/report-compatible settings row without erasing Alpaca/risk info."""
    existing = existing_settings if isinstance(existing_settings, dict) else {}
    try:
        alpaca = existing.get("alpaca") if isinstance(existing.get("alpaca"), dict) else AlpacaSettings().redacted()
    except Exception:
        alpaca = existing.get("alpaca", {}) if isinstance(existing, dict) else {}
    live_existing = existing.get("live", {}) if isinstance(existing.get("live"), dict) else {}
    risk_existing = existing.get("risk", {}) if isinstance(existing.get("risk"), dict) else {}
    live = dict(live_existing)
    live.update({
        "enabled": bool(cfg.get("enabled", True)),
        "dry_run": bool(cfg.get("dry_run", False)),
        "feed": cfg.get("feed", live_existing.get("feed", os.getenv("ALPACA_FEED", "iex"))),
        "symbols": cfg.get("symbols", live_existing.get("symbols", [])),
        "strategy_variant": cfg.get("strategy_variant", live_existing.get("strategy_variant", "best_report_153601")),
        "strategy_run_mode": cfg.get("live_strategy_run_mode", live_existing.get("strategy_run_mode", "single")),
        "active_strategy_count": cfg.get("active_strategy_count", live_existing.get("active_strategy_count", 1)),
        "active_strategy_variants": cfg.get("active_strategy_variants", live_existing.get("active_strategy_variants", [])),
        "live_config_source": "dashboard_db",
        "entry_start_time_et": cfg.get("live_entry_start_time_et", live_existing.get("entry_start_time_et", "09:35")),
        "entry_end_time_et": cfg.get("live_entry_end_time_et", live_existing.get("entry_end_time_et", "15:55")),
        "force_market_open": bool(cfg.get("live_require_market_open", live_existing.get("force_market_open", True))),
        "allow_extended_hours_entries": bool(cfg.get("live_allow_extended_hours_entries", live_existing.get("allow_extended_hours_entries", False))),
        "enable_max_hold_exit": bool(cfg.get("live_enable_max_hold_exit", live_existing.get("enable_max_hold_exit", True))),
        "max_daily_trades": int(cfg.get("max_daily_trades", cfg.get("max_trades", live_existing.get("max_daily_trades", 1))) or 1),
        "max_open_positions": int(cfg.get("max_open_positions", live_existing.get("max_open_positions", 1)) or 1),
        "max_orders_per_symbol_per_day": int(cfg.get("max_orders_per_symbol_per_day", live_existing.get("max_orders_per_symbol_per_day", 1)) or 1),
        "selection_mode": cfg.get("selection_mode", live_existing.get("selection_mode", "seen_so_far_top_n")),
    })
    risk = dict(risk_existing)
    mode = str(cfg.get("risk_mode", risk_existing.get("position_sizing_mode", "percent_equity")) or "percent_equity")
    risk.update({
        "position_sizing_mode": mode,
        "fixed_risk_dollars": float(cfg.get("risk_dollars", risk_existing.get("fixed_risk_dollars", 100)) or 100),
        "account_value_fallback": float(cfg.get("account_value", risk_existing.get("account_value_fallback", 10000)) or 10000),
        "base_risk_pct": float(cfg.get("base_risk_pct", risk_existing.get("base_risk_pct", 1.0)) or 1.0),
        # These only apply to controlled_compounding. For fixed/percent modes they are
        # deliberately stored as null so the DB/debug output does not imply a hidden cap.
        "min_risk_dollars": (float(cfg.get("min_risk_dollars") or 0.0) if mode == "controlled_compounding" else None),
        "max_risk_dollars": (float(cfg.get("max_risk_dollars") or 0.0) if mode == "controlled_compounding" else None),
        "dd1_risk_pct": (float(cfg.get("dd1_risk_pct") or 0.0) if mode == "controlled_compounding" else None),
        "dd2_risk_pct": (float(cfg.get("dd2_risk_pct") or 0.0) if mode == "controlled_compounding" else None),
        "pause_dd_pct": (float(cfg.get("pause_dd_pct") or 0.0) if mode == "controlled_compounding" else None),
    })
    return {"live": live, "risk": risk, "alpaca": alpaca}


@server.route("/healthz")
def healthz():
    return jsonify({"ok": True, "service": os.getenv("RENDER_SERVICE_NAME", "unknown")})


@server.route("/debug/live-state")
def debug_live_state():
    """Fast no-Dash diagnostic endpoint.

    This route intentionally avoids full LiveStore schema initialization and
    avoids large history queries.  It should open quickly even if the Dash UI is
    stuck or the live tables are large.
    """
    out = {
        "ok": False,
        "service": os.getenv("RENDER_SERVICE_NAME", "unknown"),
        "database_url_present": bool(os.getenv("DATABASE_URL")),
        "database_url_type": "postgres" if str(os.getenv("DATABASE_URL", "")).startswith(("postgresql://", "postgres://")) else "sqlite_or_missing",
        "alpaca_env_present": {
            "ALPACA_API_KEY": bool(os.getenv("ALPACA_API_KEY")),
            "ALPACA_SECRET_KEY": bool(os.getenv("ALPACA_SECRET_KEY")),
            "ALPACA_TRADING_BASE_URL": bool(os.getenv("ALPACA_TRADING_BASE_URL")),
            "ALPACA_DATA_BASE_URL": bool(os.getenv("ALPACA_DATA_BASE_URL")),
            "ALPACA_FEED": os.getenv("ALPACA_FEED", ""),
        },
    }
    try:
        store = LiveStore(initialize_schema=False)
        out["store"] = {"is_postgres": store.is_postgres}
        for key in ["live_config_override", "settings", "heartbeat", "market_clock", "last_bar_fetch", "last_strategy_scan_summary", "last_completed_strategy_scan", "live_scan_progress", "live_feature_cache_summary", "last_signal_filter_summary", "last_client_side_protective_exits", "worker_error"]:
            value, updated_at = store.get_state_with_updated_at(key, None)
            out[key] = {"updated_at_utc": updated_at, "value": _safe_diag_value(value)}
        out["table_counts"] = {
            name: store.table_count(name)
            for name in ["live_account_snapshots", "live_positions", "live_orders", "live_signal_plans", "live_events", "live_candidate_audit", "live_symbol_monitor", "live_strategy_symbol_monitor"]
        }
        out["ok"] = True
    except Exception as exc:
        out["error"] = str(exc)
        out["traceback"] = traceback.format_exc()
    return jsonify(out)




@server.route("/debug/live-bar-health")
def debug_live_bar_health():
    """No-Dash data-feed diagnostic for live scans.

    Shows whether the worker is receiving same-day bars, which feed was used,
    and whether the free IEX/delayed-SIP fallback is active.
    """
    out = {"ok": False, "service": os.getenv("RENDER_SERVICE_NAME", "unknown")}
    try:
        store = LiveStore(initialize_schema=False)
        last_bar_fetch = store.get_state("last_bar_fetch", {}) or {}
        heartbeat = store.get_state("heartbeat", {}) or {}
        monitor = store.latest_symbol_monitor(1000)
        summary = {}
        if monitor is not None and not monitor.empty:
            work = monitor.copy()
            strategies_seen = int(work["strategy_variant"].astype(str).nunique()) if "strategy_variant" in work.columns else 0
            symbols_seen = int(work["symbol"].astype(str).str.upper().nunique()) if "symbol" in work.columns else 0
            summary["rows"] = int(len(work))
            summary["strategies_seen"] = strategies_seen
            summary["symbols_seen"] = symbols_seen
            active_count = int(float((heartbeat or {}).get("active_strategy_count") or 0) or 0)
            configured_symbols = int(float((heartbeat or {}).get("symbols") or 0) or 0)
            expected_rows = active_count * configured_symbols if active_count and configured_symbols else 0
            summary["expected_rows_from_heartbeat"] = expected_rows
            summary["row_coverage_ok"] = bool(not expected_rows or len(work) >= expected_rows)
            if expected_rows and len(work) < expected_rows:
                summary["coverage_warning"] = "Latest monitor row count is below active_strategy_count * symbols. Check last_strategy_scan_summary for strategy errors or an incomplete worker scan."
            if "monitor_status" in work.columns:
                summary["status_counts"] = _safe_diag_value(work["monitor_status"].astype(str).value_counts().head(30).to_dict())
            if "latest_bar_time_et" in work.columns:
                summary["latest_bar_time_et_counts"] = _safe_diag_value(work["latest_bar_time_et"].astype(str).value_counts().head(20).to_dict())
        out.update({
            "ok": True,
            "heartbeat": _safe_diag_value(heartbeat),
            "last_bar_fetch": _safe_diag_value(last_bar_fetch),
            "monitor_summary": summary,
            "symbol_monitor_preview": _df_preview(monitor, 50),
        })
    except Exception as exc:
        out.update({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    return jsonify(out)


@server.route("/debug/db-ping")
def debug_db_ping():
    out = {"ok": False, "database_url_present": bool(os.getenv("DATABASE_URL"))}
    try:
        store = LiveStore(initialize_schema=False)
        store.set_state("debug_ping", {"at_utc": pd.Timestamp.now(tz="UTC").isoformat(), "service": os.getenv("RENDER_SERVICE_NAME", "unknown")})
        value, updated_at = store.get_state_with_updated_at("debug_ping", None)
        out.update({"ok": True, "is_postgres": store.is_postgres, "updated_at_utc": updated_at, "value": _safe_diag_value(value)})
    except Exception as exc:
        out.update({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    return jsonify(out)


@server.route("/debug/live-strategies")
def debug_live_strategies():
    """List deterministic strategies available to all-strategies experiment mode."""
    try:
        specs = all_live_strategy_specs()
        return jsonify({
            "ok": True,
            "count": len(specs),
            "strategies": specs,
            "notes": [
                "Set Live strategy mode to all_strategies and click Apply current settings to live worker.",
                "The worker tags each signal plan, order client id, candidate audit row, symbol monitor row, and report row with strategy_variant/strategy_preset/quality_gate.",
            ],
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}), 500


@server.route("/debug/live-submit-filters")
def debug_live_submit_filters():
    """Explain why selected live signals did or did not become Alpaca orders."""
    out = {"ok": False, "service": os.getenv("RENDER_SERVICE_NAME", "unknown"), "database_url_present": bool(os.getenv("DATABASE_URL"))}
    try:
        store = LiveStore(initialize_schema=False)
        summary = store.get_state("last_signal_filter_summary", {}) or {}
        audit = store.recent_candidate_audit(500)
        recent_rejects = audit
        if recent_rejects is not None and not recent_rejects.empty:
            if "decision_status" in recent_rejects.columns:
                recent_rejects = recent_rejects[recent_rejects["decision_status"].astype(str).str.lower().eq("rejected")].copy()
            keep = [
                "updated_at_utc", "candidate_time_et", "symbol", "strategy_variant", "strategy_code", "strategy_side",
                "trigger_type", "audit_stage", "decision_status", "reject_reason", "candidate_score", "final_rank_score",
                "entry_reference_price", "qty", "risk_budget", "stop_price", "target_price",
            ]
            recent_rejects = recent_rejects[[c for c in keep if c in recent_rejects.columns]].head(100)
        out.update({
            "ok": True,
            "store": {"is_postgres": store.is_postgres},
            "last_signal_filter_summary": _safe_diag_value(summary),
            "recent_rejected_candidates": _df_preview(recent_rejects, 100),
            "interpretation": [
                "signals_filtered is not an Alpaca/API error. It means at least one strategy produced a signal, but none survived the final submit checks.",
                "Use filter_reason_counts to see whether the blocker was duplicate signal, open position, symbol daily order limit, buying-power reserve, invalid sizing, or capacity.",
                "Candidate audit keeps per-symbol/per-strategy rows so reports can show which strategy generated the signal and why it was rejected.",
            ],
        })
    except Exception as exc:
        out.update({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    return jsonify(out)


@server.route("/debug/live-data")
@server.route("/debug/live-tables")
@server.route("/debug/live-snapshot")
def debug_live_data():
    """No-Dash diagnostic endpoint for the live dashboard tables."""
    out = {"ok": False, "service": os.getenv("RENDER_SERVICE_NAME", "unknown"), "database_url_present": bool(os.getenv("DATABASE_URL"))}
    try:
        from src.live_dashboard import build_live_trade_report
        store = LiveStore(initialize_schema=False)
        events = store.recent_events(25)
        orders = store.recent_orders(50)
        plans = store.recent_signal_plans(50)
        open_positions = store.open_positions()
        closed_positions = store.closed_positions(50)
        account = store.latest_account()
        trade_report = build_live_trade_report(days=3650)
        symbol_monitor = store.latest_symbol_monitor(1000)
        out.update({
            "ok": True,
            "store": {"is_postgres": store.is_postgres},
            "counts": {
                "events_loaded": int(len(events)),
                "orders_loaded": int(len(orders)),
                "signal_plans_loaded": int(len(plans)),
                "open_positions_loaded": int(len(open_positions)),
                "closed_positions_loaded": int(len(closed_positions)),
                "account_snapshots_loaded": int(len(account)),
                "trade_report_rows": int(len(trade_report)),
                "symbol_monitor_rows": int(len(symbol_monitor)),
            },
            "latest_account": _df_preview(account, 3),
            "recent_signal_plans": _df_preview(plans, 10),
            "recent_orders": _df_preview(orders, 10),
            "live_trade_report": _df_preview(trade_report, 10),
            "symbol_monitor": _df_preview(symbol_monitor, 50),
            "recent_events": _df_preview(events, 10),
        })
    except Exception as exc:
        out.update({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    return jsonify(out)


@server.route("/debug/live-symbol-monitor")
def debug_live_symbol_monitor():
    """No-Dash diagnostic endpoint for the professional Live Symbol Intelligence panel."""
    out = {"ok": False, "service": os.getenv("RENDER_SERVICE_NAME", "unknown"), "database_url_present": bool(os.getenv("DATABASE_URL"))}
    try:
        store = LiveStore(initialize_schema=False)
        cfg = store.get_state("live_config_override", {}) or {}
        heartbeat = store.get_state("heartbeat", {}) or {}
        monitor = store.latest_symbol_monitor(1000)
        out.update({
            "ok": True,
            "store": {"is_postgres": store.is_postgres},
            "configured_symbols": cfg.get("symbols") or heartbeat.get("symbols"),
            "heartbeat": _safe_diag_value(heartbeat),
            "live_scan_progress": _safe_diag_value(store.get_state("live_scan_progress", {}) or {}),
            "last_completed_strategy_scan": _safe_diag_value(store.get_state("last_completed_strategy_scan", {}) or {}),
            "last_signal_filter_summary": _safe_diag_value(store.get_state("last_signal_filter_summary", {}) or {}),
            "last_client_side_protective_exits": _safe_diag_value(store.get_state("last_client_side_protective_exits", {}) or {}),
            "symbol_monitor": _df_preview(monitor, 1000),
        })
    except Exception as exc:
        out.update({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    return jsonify(out)


@server.route("/debug/live-data-readiness")
def debug_live_data_readiness():
    """Explain whether the current free/IEX extended-hours data is usable for scans.

    This endpoint is intentionally JSON-only and avoids Dash so it remains usable
    even while diagnosing UI refresh problems.
    """
    out = {"ok": False, "service": os.getenv("RENDER_SERVICE_NAME", "unknown")}
    try:
        store = LiveStore(initialize_schema=False)
        heartbeat = store.get_state("heartbeat", {}) or {}
        last_bar_fetch = store.get_state("last_bar_fetch", {}) or {}
        scan_summary = store.get_state("last_strategy_scan_summary", {}) or {}
        monitor = store.latest_symbol_monitor(1000)
        status_counts = {}
        if monitor is not None and not monitor.empty and "monitor_status" in monitor.columns:
            status_counts = monitor["monitor_status"].astype(str).value_counts().to_dict()
        strategies_seen = 0
        symbols_seen = 0
        latest_bar = None
        if monitor is not None and not monitor.empty:
            if "strategy_variant" in monitor.columns:
                strategies_seen = int(monitor["strategy_variant"].astype(str).nunique())
            if "symbol" in monitor.columns:
                symbols_seen = int(monitor["symbol"].astype(str).str.upper().nunique())
            if "latest_bar_time_utc" in monitor.columns:
                vals = monitor["latest_bar_time_utc"].dropna().astype(str)
                latest_bar = vals.max() if not vals.empty else None
        out.update({
            "ok": True,
            "database": {"is_postgres": store.is_postgres},
            "heartbeat": _safe_diag_value(heartbeat),
            "last_bar_fetch": _safe_diag_value(last_bar_fetch),
            "last_strategy_scan_summary": _safe_diag_value(scan_summary),
            "monitor_counts": {
                "rows": int(0 if monitor is None else len(monitor)),
                "symbols_seen": symbols_seen,
                "strategies_seen": strategies_seen,
                "expected_rows_from_heartbeat": int(float((heartbeat or {}).get("active_strategy_count") or 0) * float((heartbeat or {}).get("symbols") or 0)) if heartbeat else 0,
                "latest_bar_time_utc": latest_bar,
                "status_counts": _safe_diag_value(status_counts),
            },
            "sample_monitor_rows": _df_preview(monitor, 25),
            "interpretation": [
                "For Alpaca Basic/free accounts, feed=iex is only one exchange and may be sparse in extended hours.",
                "The worker now uses the newest available bar within live_max_bar_age_minutes instead of requiring a wall-clock exact 5-minute slot.",
                "If rows show Watching with non-zero checks and indicator values, the data pipeline is working; no trade means no strategy setup passed.",
                "If rows show Waiting - symbol data/no latest bar, the selected feed did not provide usable bars for that symbol/session.",
            ],
        })
    except Exception as exc:
        out.update({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    return jsonify(out)


@server.route("/debug/alpaca-connection")
def debug_alpaca_connection():
    """Verify the Alpaca trading API from the web service without exposing secrets."""
    out = {
        "ok": False,
        "alpaca_env_present": {
            "ALPACA_API_KEY": bool(os.getenv("ALPACA_API_KEY")),
            "ALPACA_SECRET_KEY": bool(os.getenv("ALPACA_SECRET_KEY")),
            "ALPACA_TRADING_BASE_URL": bool(os.getenv("ALPACA_TRADING_BASE_URL")),
            "ALPACA_DATA_BASE_URL": bool(os.getenv("ALPACA_DATA_BASE_URL")),
            "ALPACA_FEED": os.getenv("ALPACA_FEED", ""),
        },
    }
    try:
        from src.alpaca_trading import AlpacaTradingClient
        settings = AlpacaSettings()
        out["configured"] = bool(settings.is_configured)
        out["trading_base_url"] = str(settings.trading_base_url)
        if not settings.is_configured:
            out["error"] = "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY."
            return jsonify(out)
        account = AlpacaTradingClient(settings).get_account()
        account_id = str(account.get("account_number") or account.get("id") or "")
        out.update({
            "ok": True,
            "paper_endpoint": "paper-api" in str(settings.trading_base_url).lower(),
            "account_status": account.get("status"),
            "account_id_last4": account_id[-4:] if account_id else "",
            "equity_present": bool(account.get("equity")),
            "buying_power_present": bool(account.get("buying_power")),
        })
    except Exception as exc:
        out.update({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    return jsonify(out)


@server.route("/download-live-report.zip")
def download_live_report_direct():
    """Direct download endpoint for Render/browser cases where dcc.Download does not trigger."""
    zip_path = generate_live_report_zip(days=3650)
    return send_file(str(zip_path), as_attachment=True, download_name=zip_path.name, mimetype="application/zip")

DEFAULT_END = date.today()
DEFAULT_START = DEFAULT_END - timedelta(days=90)


def live_strategy_run_mode_options():
    specs = all_live_strategy_specs()
    label = f"All live strategies in parallel ({len(specs)} strategies)"
    return [
        {"label": "Single selected strategy", "value": "single"},
        {"label": label, "value": "all_strategies"},
    ]


def metric_card(label: str, value: str, help_text: str = "") -> html.Div:
    return html.Div(
        className="metric-card",
        children=[html.Div(label, className="metric-label"), html.Div(value, className="metric-value"), html.Div(help_text, className="metric-help") if help_text else None],
    )


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=330,
        margin=dict(l=40, r=20, t=55, b=40),
        annotations=[dict(text="Run a backtest to populate this chart", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")],
    )
    return fig


def fmt_money(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "$0.00"
    return f"${x:,.2f}"


def fmt_pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "0.00%"
    return f"{x:,.2f}%"




def _collect_unavailable_symbols(result: dict) -> list[str]:
    """Collect symbols that Alpaca/cache could not provide, without stopping the backtest."""
    symbols: set[str] = set()
    for status in result.get("diagnostics", {}).get("cache_status", []) or []:
        for rec in (status or {}).get("unavailable_symbols", []) or []:
            sym = str((rec or {}).get("symbol", "")).upper().strip()
            if sym:
                reason = str((rec or {}).get("reason", "unavailable")).strip()
                symbols.add(f"{sym}: {reason[:140]}")
    skipped_df = result.get("skipped_symbols", pd.DataFrame())
    if skipped_df is not None and not skipped_df.empty:
        for _, row in skipped_df.iterrows():
            sym = str(row.get("symbol", "")).upper().strip()
            reason = str(row.get("reason", "")).strip()
            if sym and reason in {"missing_bars", "no_rows_in_requested_window"}:
                symbols.add(f"{sym}: {reason}")
            elif sym and reason.startswith("error:") and ("404" in reason or "symbol" in reason.lower() or "not found" in reason.lower()):
                symbols.add(f"{sym}: {reason[:140]}")
    return sorted(symbols)


def _format_user_alerts(result: dict, custom_symbols_active: bool, max_items: int = 12) -> str:
    unavailable = _collect_unavailable_symbols(result)
    parts: list[str] = []
    if unavailable:
        shown = unavailable[:max_items]
        more = len(unavailable) - len(shown)
        parts.append("⚠️ Unavailable/no-data symbols skipped while continuing with the rest: " + "; ".join(shown) + (f"; +{more} more" if more > 0 else ""))
    parts.append("Note: Symbol Summary is portfolio-selected P&L after Top trades/day ranking. A symbol can be positive inside a preset because only its best ranked trades were selected, but negative when tested alone because all eligible trades for that symbol can be taken.")
    if custom_symbols_active:
        parts.append("Custom symbols override the preset and use the raw Alpaca local bar cache/fetch path; unavailable symbols are skipped, not fatal.")
    return " | ".join(parts)

def table_card(title: str, subtitle: str, table_id: str, page_size: int = 10, filterable: bool = False) -> html.Div:
    return html.Div(
        className="card",
        children=[
            html.Div(className="section-head", children=[html.H3(title), html.Span(subtitle)]),
            dash_table.DataTable(
                id=table_id,
                page_size=page_size,
                sort_action="native",
                filter_action="native" if filterable else "none",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "10px", "fontFamily": "Inter, Arial", "fontSize": "12px"},
                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
            ),
        ],
    )


def symbol_monitor_card() -> html.Div:
    return html.Div(
        className="card symbol-monitor-card",
        children=[
            html.Div(
                className="section-head",
                children=[
                    html.Div(children=[html.H3("Live Symbol Intelligence"), html.Span("One lightweight row per monitored symbol: latest closed bar, strategy gate, indicator readings and pass/fail checks.")]),
                    html.Div(className="intel-legend", children=[html.Span("Selected"), html.Span("Candidate"), html.Span("Blocked"), html.Span("Watching")]),
                ],
            ),
            html.Div(className="symbol-intel-intro", children=[
                html.Div(id="live-symbol-monitor-summary", className="symbol-intel-summary", children="Symbol monitor will populate after the worker writes its next scan."),
            ]),
            dash_table.DataTable(
                id="live-symbol-monitor-table",
                page_size=16,
                sort_action="native",
                filter_action="native",
                fixed_rows={"headers": True},
                style_table={"overflowX": "auto", "maxHeight": "520px", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "9px 10px", "fontFamily": "Inter, Arial", "fontSize": "12px", "minWidth": "82px", "whiteSpace": "normal", "height": "auto"},
                style_header={"fontWeight": "800", "backgroundColor": "#f8fafc", "borderBottom": "1px solid #e2e8f0"},
                style_data_conditional=[
                    {"if": {"filter_query": "{status} contains Selected"}, "backgroundColor": "#ecfdf5", "borderLeft": "4px solid #10b981"},
                    {"if": {"filter_query": "{status} contains Candidate"}, "backgroundColor": "#eff6ff", "borderLeft": "4px solid #2563eb"},
                    {"if": {"filter_query": "{status} contains Blocked"}, "backgroundColor": "#fff7ed", "borderLeft": "4px solid #f97316"},
                    {"if": {"filter_query": "{status} contains Paused"}, "backgroundColor": "#f8fafc", "color": "#64748b"},
                    {"if": {"filter_query": "{status} contains Waiting"}, "backgroundColor": "#f8fafc", "color": "#64748b"},
                ],
            ),
        ],
    )


settings = AlpacaSettings()
status_text = "Configured" if settings.is_configured else "Missing API keys"
status_class = "status-pill ok" if settings.is_configured else "status-pill warn"

app.layout = html.Div(
    className="page",
    children=[
        html.Div(
            className="hero",
            children=[
                html.Div(children=[html.Div("ALPACA MOMENTUM LAB", className="eyebrow"), html.H1("Day-Trading Research Dashboard V38.2"), html.P("Backtest, tune, and monitor the symbol playbook with editable strategy settings, separate risk sizing, and trade context reports.")]),
                html.Div(className=status_class, children=status_text),
            ],
        ),
        html.Div(
            className="layout",
            children=[
                html.Div(
                    className="sidebar card",
                    children=[
                        html.H3("Backtest Controls"),
                        html.Div(title="Uses the Symbol/Event Playbook engine for the current best strategy work.", children=[
                            html.Label("Strategy mode"),
                            dcc.Dropdown(
                                id="strategy-profile",
                                options=[
                                    {"label": "Symbol/Event Playbook - Best Report 153601 strategy + separate sizing", "value": "symbol_playbook_v25"},
                                ],
                                value="symbol_playbook_v25",
                                clearable=False,
                            ),
                        ]),
                        html.Div(title="Applies only the winning strategy filters. It does not change account value, fixed risk, compounding, or risk sizing settings.", children=[
                            html.Label("Strategy preset"),
                            dcc.Dropdown(
                                id="settings-preset",
                                options=[
                                    {"label": "Manual / custom - keep current controls", "value": "manual"},
                                    {"label": "Best Report 153601 baseline", "value": "best_qqq_news"},
                                    {"label": "V35.8 Raw quality gate", "value": "live_raw_optimized_v358"},
                                    {"label": "V35.9 Live Hunter", "value": "live_hunter_v359"},
                                    {"label": "V36.2 Long-run robust", "value": "live_longrun_robust_v362"},
                                    {"label": "V36.3 Grid-tested robust", "value": "live_grid_robust_v363"},
                                    {"label": "V36.4 Pro momentum hybrid", "value": "live_professional_momentum_v364"},
                                    {"label": "V37.8 Mined pattern matcher", "value": "live_positive_context_v377"},
                                    {"label": "V37.9 Indicator pattern scorer", "value": "live_indicator_pattern_v379"},
                                    {"label": "V38 Active pattern scorer", "value": "live_active_pattern_v38"},
                                    {"label": "V38 Stable pattern scorer", "value": "live_stable_pattern_v38"},
                                    {"label": "V38.2 Active Plus - more trades", "value": "live_active_plus_v382"},
                                    {"label": "V38.3 Adaptive Composite - regime routed", "value": "live_adaptive_composite_v383"},
                                    {"label": "V38.4 Failure-aware reversal router", "value": "live_failure_reversal_v384"},
                                    {"label": "V38.5 Adaptive Plus - V38.3 with damage control", "value": "live_adaptive_plus_v385"},
                                    {"label": "V38.2 More Trades Research", "value": "live_more_trades_v382"},
                                ],
                                value="best_qqq_news",
                                clearable=False,
                            ),
                        ]),
                        html.Div(title="Selects the stock universe to test. Best Report 153601 uses the V25 playbook universe.", children=[html.Label("Watchlist preset"), dcc.Dropdown(id="preset", options=[{"label": k.replace("_", " ").title(), "value": k} for k in WATCHLISTS.keys()], value="v25_playbook", clearable=False)]),
                        html.Div(title="Optional: enter a comma-separated custom symbol list. When populated, this overrides the preset and uses the raw local Alpaca bar cache/fetch path for those symbols.", children=[html.Label("Custom symbols, comma separated - overrides preset"), dcc.Textarea(id="custom-symbols", placeholder="Example: ADTX, GDC, GPUS, SRXH, CDT", value="", className="textarea")]),
                        html.Div(className="two-col", children=[html.Div(title="Backtest start date.", children=[html.Label("Start"), dcc.DatePickerSingle(id="start-date", date=DEFAULT_START.isoformat())]), html.Div(title="Backtest end date.", children=[html.Label("End"), dcc.DatePickerSingle(id="end-date", date=DEFAULT_END.isoformat())])]),
                        html.Div(title="LIVE trading uses real-time order-decision data only. IEX is the free/no-subscription feed but it is a single exchange and can be sparse in extended hours. SIP real-time requires paid/unlimited data entitlement.", children=[html.Label("Alpaca data feed"), dcc.Dropdown(id="feed", options=[{"label": "IEX - free/no-subscription, single-exchange", "value": "iex"}, {"label": "SIP real-time - paid/unlimited entitlement", "value": "sip"}], value=os.getenv("ALPACA_FEED", "iex"), clearable=False)]),
                        html.Div(title="Regular hours preserves the original V33 backtest/replay/cache behavior. Extended hours uses the same local Alpaca bar store and fetches missing chunks only when they are not available locally.", children=[
                            html.Label("Backtest data session"),
                            dcc.Dropdown(
                                id="backtest-session-mode",
                                options=[
                                    {"label": "Regular hours only - original baseline behavior", "value": "regular_only"},
                                    {"label": "Extended hours - 04:00 to 20:00 ET research", "value": "extended_hours"},
                                    {"label": "24/5 available bars - research only", "value": "twenty_four_five"},
                                ],
                                value="regular_only",
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="hint", children="Regular hours uses the original V33 V25 replay/backtest path. Extended-hours modes use raw local bars from data/local_bars/<feed>/5Min/split and the same Alpaca cache/fetch mechanism as before."),
                        html.Div(title="Controls trade selection timing. Classic preserves the original research result by ranking the full day after all candidates are known. Live simulation walks forward historically and only ranks candidates seen up to each timestamp, like live trading.", children=[
                            html.Label("Backtest decision mode"),
                            dcc.Dropdown(
                                id="backtest-decision-mode",
                                options=[
                                    {"label": "Classic research replay - end-of-day Top N ranking", "value": "end_of_day_top_n"},
                                    {"label": "Live simulation - walk-forward from historical candidates", "value": "live_simulated"},
                                    {"label": "Full raw-bar replay - rebuild signals from stored 5Min bars", "value": "raw_bar_replay"},
                                ],
                                value="end_of_day_top_n",
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="hint", children="Use Full raw-bar replay for the strongest live-style test: it rebuilds signals from stored 5-minute bars and walks forward candle by candle. Live simulation uses historical candidates but no future candidate ranking. Classic keeps the original baseline comparable."),
                        html.Div(className="control-section", children=[
                            html.Div(className="section-mini-head", children=[html.H4("Live raw replay quality gate"), html.Span("optional, live-safe")]),
                            html.Div(title="Choose a live-safe quality gate. Presets load defaults here, but you can change every field before running.", children=[
                                html.Label("Quality gate mode"),
                                dcc.Dropdown(
                                    id="live-quality-gate",
                                    options=[
                                        {"label": "Off - no extra live quality gate", "value": "off"},
                                        {"label": "V35.8 strict morning gate", "value": "v358"},
                                        {"label": "V35.9 live hunter gate", "value": "v359"},
                                        {"label": "V36.4 professional momentum hybrid gate", "value": "v364"},
                                        {"label": "V37.8 mined profitable-indicator pattern", "value": "v377"},
                                                {"label": "V37.9 decision-time indicator pattern scorer", "value": "v379"},
                                        {"label": "V38 active pattern scorer - more trades", "value": "v38_active"},
                                        {"label": "V38 stable pattern scorer", "value": "v38_stable"},
                                        {"label": "V38.2 active plus - more trades", "value": "v382_active_plus"},
                                        {"label": "V38.3 adaptive composite - regime routed", "value": "v383_adaptive"},
                                        {"label": "V38.4 failure-aware reversal router", "value": "v384_failure_reversal"},
                                        {"label": "V38.5 adaptive plus - V38.3 damage control", "value": "v385_adaptive_plus"},
                                        {"label": "V38.2 more trades research", "value": "v382_more_trades"},
                                        {"label": "Custom gate - use the values below", "value": "custom"},
                                    ],
                                    value="off",
                                    clearable=False,
                                ),
                            ]),
                            html.Div(className="two-col", children=[
                                html.Div(title="First eligible entry time for the quality gate, New York time HH:MM.", children=[html.Label("Gate start ET"), dcc.Input(id="quality-start-time", value="10:00", type="text")]),
                                html.Div(title="Last eligible entry time for the quality gate, New York time HH:MM.", children=[html.Label("Gate end ET"), dcc.Input(id="quality-end-time", value="11:00", type="text")]),
                            ]),
                            html.Div(className="three-col", children=[
                                html.Div(title="Minimum relative volume at this time of day.", children=[html.Label("Min RVOL"), dcc.Input(id="quality-min-rvol", value=1.0, type="number", min=0.0, max=20.0, step=0.05)]),
                                html.Div(title="Minimum prior-day ATR percent. This avoids low-range days that were the largest long-run raw-replay loss source.", children=[html.Label("Min daily ATR %"), dcc.Input(id="quality-min-daily-atr", value=0.0, type="number", min=0.0, max=20.0, step=0.05)]),
                                html.Div(title="Minimum directional relative strength. Long uses positive RS; short uses relative weakness converted to positive.", children=[html.Label("Min dir RS"), dcc.Input(id="quality-min-dir-rs", value=0.0, type="number", min=-20.0, max=20.0, step=0.05)]),
                                html.Div(title="Maximum directional relative strength. Use this to avoid overextended moves.", children=[html.Label("Max dir RS"), dcc.Input(id="quality-max-dir-rs", value=999.0, type="number", min=-20.0, max=999.0, step=0.05)]),
                            ]),
                            html.Div(className="three-col", children=[
                                html.Div(title="Minimum directional open-relative strength.", children=[html.Label("Min open RS"), dcc.Input(id="quality-min-dir-open-rs", value=-999.0, type="number", min=-999.0, max=999.0, step=0.05)]),
                                html.Div(title="Maximum directional open-relative strength.", children=[html.Label("Max open RS"), dcc.Input(id="quality-max-dir-open-rs", value=999.0, type="number", min=-999.0, max=999.0, step=0.05)]),
                                html.Div(title="Maximum absolute VWAP extension in ATR units.", children=[html.Label("Max abs VWAP ATR"), dcc.Input(id="quality-max-abs-vwap", value=1.5, type="number", min=0.0, max=20.0, step=0.05)]),
                            ]),
                            html.Div(className="two-col", children=[
                                html.Div(title="Minimum directional VWAP extension in ATR units. For shorts, below-VWAP extension is converted to positive.", children=[html.Label("Min dir VWAP ATR"), dcc.Input(id="quality-min-dir-vwap", value=0.5, type="number", min=-20.0, max=20.0, step=0.05)]),
                                html.Div(title="Maximum directional VWAP extension in ATR units.", children=[html.Label("Max dir VWAP ATR"), dcc.Input(id="quality-max-dir-vwap", value=2.0, type="number", min=-20.0, max=20.0, step=0.05)]),
                            ]),
                            html.Div(className="hint", children="These gate fields are independent from risk/compounding and can be used with any preset. They only use signal-bar data available at that timestamp."),
                        ]),
                        html.Div(title="Research-only filter. The algorithm creates candidates first, then sends the candidate list in batch prompts to the OpenAI API. Approved candidates continue into the normal simulator. Leave OFF for baseline tests.", children=[
                            html.Label("OpenAI trade review filter"),
                            dcc.Dropdown(
                                id="openai-filter-mode",
                                options=[
                                    {"label": "Off - baseline algorithm only", "value": "off"},
                                    {"label": "On - OpenAI reviews candidates before portfolio selection", "value": "on"},
                                ],
                                value="off",
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="three-col", children=[
                            html.Div(title="OpenAI model name. You can change this if your API account uses a different model.", children=[html.Label("OpenAI model"), dcc.Input(id="openai-model", value=os.getenv("OPENAI_TRADE_FILTER_MODEL", "gpt-5-mini"), type="text")]),
                            html.Div(title="V35.7: OpenAI reviews all candidate decisions in one API call when they fit this limit. Raw-bar replay now calls OpenAI before walk-forward selection, not once per trade/timestamp. If candidates exceed this limit, only then it chunks.", children=[html.Label("AI max independent decisions per API call"), dcc.Input(id="openai-max-candidates", value=5000, type="number", min=1, max=20000, step=1)]),
                            html.Div(title="Reject OpenAI-approved trades below this confidence. 0.0 accepts all approved trades.", children=[html.Label("AI min confidence"), dcc.Input(id="openai-min-confidence", value=0.0, type="number", min=0.0, max=1.0, step=0.05)]),
                        ]),
                        html.Div(className="hint", children="OpenAI trade review is for research only. V35.3 batches candidates for API efficiency, but every candidate is reviewed as an independent real-time decision at its own entry timestamp. It does not ask AI to choose the best of the day/batch and does not send future P&L or exit outcome. It requires OPENAI_API_KEY in .env."),
                        html.Div(className="two-col", children=[html.Div(title="Starting account size used for the equity curve and risk calculations.", children=[html.Label("Account value"), dcc.Input(id="account-value", value=10000, type="number", min=500, step=100)]), html.Div(title="Used only when Risk mode is Fixed dollar risk. This disables compounding and risks the same dollars on every trade.", children=[html.Label("Fixed risk $/trade"), dcc.RadioItems(id="risk-dollars-v12", options=[{"label": "$10", "value": 10}, {"label": "$25", "value": 25}, {"label": "$50", "value": 50}, {"label": "$100", "value": 100}, {"label": "$200", "value": 200}, {"label": "$500", "value": 500}], value=100, inline=True)])]),
                        html.Hr(),
                        html.H3("Risk sizing / compounding", className="section-title"),
                        html.Div(className="hint", id="risk-mode-help", children="Risk sizing is independent from the strategy preset. Change it to compare fixed risk vs compounding without changing the trade signals."),
                        html.Div(title="Fixed dollar risk = no compounding. Percent of equity = full compounding. Controlled compounding = compounding with cap and drawdown brakes.", children=[
                            html.Label("Risk / compounding mode"),
                            dcc.Dropdown(id="risk-mode", options=[
                                {"label": "Fixed dollar risk - no compounding", "value": "fixed_dollar_risk"},
                                {"label": "Percent of equity - full compounding", "value": "percent_equity"},
                                {"label": "Controlled compounding - small-account friendly", "value": "controlled_compounding"},
                            ], value="percent_equity", clearable=False),
                        ]),
                        html.Div(id="risk-percent-panel", className="three-col", children=[
                            html.Div(title="Used by Percent of equity and Controlled compounding. 1.0 means risk 1% of current equity per trade.", children=[html.Label("Base risk %"), dcc.Input(id="base-risk-pct", value=1.0, type="number", min=0.1, max=5.0, step=0.05)]),
                        ]),
                        html.Div(id="controlled-compounding-panel", children=[
                            html.Div(className="three-col", children=[
                                html.Div(title="Controlled compounding only. Minimum dollar risk per trade. Use a small value like $5 or $10 for small accounts.", children=[html.Label("Min risk $"), dcc.Input(id="min-risk-dollars", value=10, type="number", min=0, max=1000, step=5)]),
                                html.Div(title="Controlled compounding only. Maximum dollar risk per trade. This prevents profits from increasing position size too quickly.", children=[html.Label("Max risk $"), dcc.Input(id="max-risk-dollars", value=300, type="number", min=0, max=5000, step=25)]),
                            ]),
                            html.Div(className="three-col", children=[
                                html.Div(title="Controlled compounding only. When account drawdown reaches about 5%, use this reduced risk percentage.", children=[html.Label("DD 5% risk %"), dcc.Input(id="dd1-risk-pct", value=0.75, type="number", min=0.0, max=5.0, step=0.05)]),
                                html.Div(title="Controlled compounding only. When account drawdown reaches about 10%, use this smaller risk percentage.", children=[html.Label("DD 10% risk %"), dcc.Input(id="dd2-risk-pct", value=0.50, type="number", min=0.0, max=5.0, step=0.05)]),
                                html.Div(title="Controlled compounding only. If drawdown reaches this level, new trades are skipped until equity recovers.", children=[html.Label("Pause DD %"), dcc.Input(id="pause-dd-pct", value=15.0, type="number", min=1.0, max=50.0, step=0.5)]),
                            ]),
                        ]),
                        html.Div(className="hint", children="To reproduce the high-equity uploaded report, select Risk mode = Percent of equity - full compounding and Base risk = 1.0%. To disable compounding, select Fixed dollar risk. To use safer compounding for smaller accounts, select Controlled compounding.") ,
                        html.Div(className="two-col", children=[html.Div(title="Minimum playbook score. 0 disables the score filter. The winning report used 2.", children=[html.Label("Min score (V25 raw score, 0=off)"), dcc.Input(id="min-score", value=2, type="number", min=0, max=60, step=1)]), html.Div(title="Maximum selected trades per day.", children=[html.Label("Top trades/day"), dcc.Dropdown(id="max-trades", options=[{"label": "Top 1", "value": 1}, {"label": "Top 2", "value": 2}, {"label": "Top 3", "value": 3}, {"label": "Top 5", "value": 5}, {"label": "Top 7", "value": 7}, {"label": "Top 10", "value": 10}, {"label": "Top 15", "value": 15}], value=2, clearable=False)])]),
                        html.Div(className="two-col", children=[html.Div(title="Estimated execution slippage in basis points. 3 bps means about 0.03% per execution adjustment.", children=[html.Label("Slippage bps"), dcc.Input(id="slippage-bps", value=3, type="number", min=0, max=50, step=1)]), html.Div(title="Turns on the historical news/catalyst proxy used by the catalyst filter. Winning report used Yes.", children=[html.Label("Use news / catalyst proxy?"), dcc.Dropdown(id="use-news", options=[{"label": "No - ignore news proxy", "value": "false"}, {"label": "Yes - activate news/catalyst flags", "value": "true"}], value="true", clearable=False)])]),
                        html.Div(title="Optional macro calendar filter. Winning report left this Off.", children=[
                            html.Label("Macro/news risk filters - optional"),
                            dcc.Dropdown(
                                id="macro-filter",
                                options=[
                                    {"label": "Off - do not filter macro calendar days", "value": "off"},
                                    {"label": "Top 1 only on macro calendar days", "value": "top1"},
                                    {"label": "Skip macro calendar days", "value": "skip"},
                                ],
                                value="off",
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="two-col", children=[
                            html.Div(title="Controls what happens on high QQQ stress days. Winning report skipped QQQ stress days.", children=[html.Label("QQQ stress filter"), dcc.Dropdown(id="stress-filter", options=[{"label": "Off", "value": "off"}, {"label": "Top 1 only on QQQ stress days", "value": "top1"}, {"label": "Skip QQQ stress days", "value": "skip"}], value="skip", clearable=False)]),
                            html.Div(title="Controls what happens when the news/catalyst proxy flags a candidate. Winning report skipped catalyst candidates.", children=[html.Label("News/catalyst filter"), dcc.Dropdown(id="news-filter", options=[{"label": "Off", "value": "off"}, {"label": "Top 1 only on catalyst proxy days", "value": "top1"}, {"label": "Skip catalyst proxy candidates", "value": "skip"}], value="skip", clearable=False)]),
                        ]),
                        html.Div(className="two-col", children=[
                            html.Div(title="QQQ stress threshold used by the stress filter. Winning report used 4.2.", children=[html.Label("QQQ stress threshold %"), dcc.Input(id="qqq-stress-threshold", value=4.2, type="number", min=0.25, max=5.0, step=0.05)]),
                            html.Div(title="Optional rolling symbol/side pause after losses. Winning report left this Off.", children=[html.Label("Symbol/side kill switch"), dcc.Dropdown(id="kill-switch", options=[{"label": "Off", "value": "off"}, {"label": "Moderate: pause after -3R / 20 trades", "value": "moderate"}, {"label": "Strict: pause after -2R / 10 trades", "value": "strict"}], value="off", clearable=False)]),
                        ]),
                        html.Div(title="Entry/exit candlestick filter. Winning report used Selective; Broad score gave the same result in your test.", children=[
                            html.Label("Candlestick patterns"),
                            dcc.Dropdown(
                                id="candle-mode",
                                options=[
                                    {"label": "Exit-only - candle reversal exits only", "value": "exit_only"},
                                    {"label": "Off - ignore candle patterns", "value": "off"},
                                    {"label": "Selective - rejection filter + reversal exits", "value": "selective"},
                                    {"label": "Broad confirm + exits - comparison", "value": "confirm"},
                                    {"label": "Broad score + exits - comparison", "value": "score"},
                                ],
                                value="selective",
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="two-col", children=[
                            html.Div(children=[html.Label("Mean reversion"), dcc.Dropdown(id="enable-mr", options=[{"label": "On", "value": "true"}, {"label": "Off", "value": "false"}], value="false", clearable=False)]),
                            html.Div(children=[html.Label("OR/retest setup"), dcc.Dropdown(id="enable-or", options=[{"label": "Off", "value": "false"}, {"label": "Selective only", "value": "true"}], value="false", clearable=False)]),
                        ]),
                        html.Label("Direction mode"),
                        dcc.Dropdown(id="direction-mode", options=[{"label": "Long only", "value": "long_only"}, {"label": "Short only", "value": "short_only"}, {"label": "Long + short", "value": "long_short"}], value="long_short", clearable=False),
                        html.Button("Run Backtest", id="run-btn", className="primary-btn", n_clicks=0),
                        html.Div(id="run-status", className="run-status"),
                        html.Hr(),
                        html.H4("Notes"),
                        html.Ul(
                            className="rule-list compact-notes",
                            children=[
                                html.Li("Strategy settings and risk sizing are separate."),
                                html.Li("Entries use the next 5-minute bar open with conservative stop-before-target sequencing."),
                                html.Li("Top trades/day now supports 1, 2, 3, 5, 7, 10, and 15."),
                                html.Li("Backtest reports include selected_trade_market_conditions.csv for trade context."),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="main",
                    children=[
                        dcc.Interval(id="live-refresh-interval", interval=30_000, n_intervals=0),
                        html.Div(className="card", children=[
                            html.Div(className="section-head", children=[html.H3("Live Alpaca Paper Monitor"), html.Span("Auto-refreshes every 30 seconds")]),
                            html.Div(id="live-status", className="run-status"),
                            html.Div(className="two-col", children=[
                                html.Button("Apply current settings to live worker", id="apply-live-settings-btn", className="primary-btn", n_clicks=0),
                                html.Div(children=[
                                    html.Button("Generate live report ZIP", id="generate-live-report-btn", className="secondary-btn", n_clicks=0),
                                    html.A("Direct download", href="/download-live-report.zip", target="_blank", className="secondary-btn", style={"marginLeft": "8px", "display": "inline-block", "textDecoration": "none"}),
                                ]),
                            ]),
                            dcc.Download(id="live-report-download"),
                            html.Div("Use Generate live report ZIP first. If your browser does not start a download, click Direct download.", id="live-action-status", className="run-status"),
                            html.Div(id="live-metrics-row", className="metrics-grid"),
                        ]),
                        html.Div(className="grid-2", children=[
                            table_card("Open Live/Paper Positions", "Positions currently open in Alpaca paper and synced by the worker", "live-positions-table", page_size=10),
                            table_card("Recent Signal Plans", "Algorithm decisions, sizing, stop, target, and paper-order submission status", "live-plans-table", page_size=12, filterable=True),
                        ]),
                        html.Div(className="grid-2", children=[
                            table_card("Recent Alpaca Paper Orders", "Parent bracket orders and exit legs synced from Alpaca", "live-orders-table", page_size=12, filterable=True),
                            table_card("Closed / Recently Closed Positions", "Positions that the worker saw open and later missing from Alpaca /positions", "live-closed-positions-table", page_size=12, filterable=True),
                        ]),
                        table_card("Live/Paper Trade P&L", "Every strategy trade with realized/unrealized profit and the strategy/gate used", "live-trade-report-table", page_size=15, filterable=True),
                        table_card("Worker Events", "Scans, blocked entries, submitted paper orders, sync errors, and max-hold exits", "live-events-table", page_size=12, filterable=True),
                        html.Div(className="section-head", children=[html.H3("Historical Backtest Lab"), html.Span("Backtest results and trade context")]),
                        html.Div(id="metrics-row", className="metrics-grid"),
                        html.Div(className="grid-2", children=[html.Div(className="card", children=[dcc.Graph(id="equity-fig", figure=empty_figure("Equity Curve"))]), html.Div(className="card", children=[dcc.Graph(id="drawdown-fig", figure=empty_figure("Drawdown"))])]),
                        html.Div(className="grid-2", children=[html.Div(className="card", children=[dcc.Graph(id="symbol-fig", figure=empty_figure("P&L by Symbol (Portfolio-Selected Trades)"))]), html.Div(className="card", children=[dcc.Graph(id="r-fig", figure=empty_figure("R-Multiple Distribution"))])]),
                        html.Div(className="grid-2", children=[html.Div(className="card", children=[dcc.Graph(id="setup-fig", figure=empty_figure("P&L by Setup Type"))]), html.Div(className="card", children=[dcc.Graph(id="daily-fig", figure=empty_figure("Trades by Day"))])]),
                        table_card("Symbol Summary", "Portfolio-selected contribution after Top trades/day. This is not the same as a standalone one-symbol backtest.", "symbol-table", page_size=12),
                        html.Div(className="grid-2", children=[table_card("Setup Summary", "Which trigger type works?", "setup-table", page_size=10), table_card("Daily Summary", "Frequency and daily consistency", "daily-table", page_size=10)]),
                        html.Div(className="grid-2", children=[table_card("Exit Reason Summary", "Are exits helping or hurting?", "exit-table", page_size=10), table_card("MFE / MAE Diagnosis", "Entry vs exit failure clues", "mfe-table", page_size=10)]),
                        html.Div(className="grid-2", children=[table_card("Candlestick Pattern Summary", "Do candle patterns improve entries/exits?", "candle-table", page_size=10), table_card("Score Band Summary", "Is score predictive?", "score-table", page_size=10)]),
                        table_card("Time Bucket Summary", "Best time of day", "time-table", page_size=10),
                        table_card("Trades", "Selected trades. Full context is saved in selected_trade_market_conditions.csv", "trades-table", page_size=15, filterable=True),
                    ],
                ),
            ],
        ),
    ],
)




# -----------------------------------------------------------------------------
# V38.7 clean customer-facing layout
# -----------------------------------------------------------------------------
# The original V38.6 layout kept all research, backtest and live controls on one
# long page.  This override keeps every callback id intact, but presents the app
# as a cleaner two-tab product: Backtest Lab and Live Monitor.

STRATEGY_PRESET_OPTIONS = [
    {"label": "Manual / custom", "value": "manual"},
    {"label": "Best Report 153601 baseline", "value": "best_qqq_news"},
    {"label": "V35.8 Raw quality gate", "value": "live_raw_optimized_v358"},
    {"label": "V35.9 Live Hunter", "value": "live_hunter_v359"},
    {"label": "V36.2 Long-run robust", "value": "live_longrun_robust_v362"},
    {"label": "V36.3 Grid-tested robust", "value": "live_grid_robust_v363"},
    {"label": "V36.4 Pro momentum hybrid", "value": "live_professional_momentum_v364"},
    {"label": "V37.8 Mined pattern matcher", "value": "live_positive_context_v377"},
    {"label": "V37.9 Indicator pattern scorer", "value": "live_indicator_pattern_v379"},
    {"label": "V38 Active pattern scorer", "value": "live_active_pattern_v38"},
    {"label": "V38 Stable pattern scorer", "value": "live_stable_pattern_v38"},
    {"label": "V38.2 Active Plus - more trades", "value": "live_active_plus_v382"},
    {"label": "V38.2 More Trades Research", "value": "live_more_trades_v382"},
    {"label": "V38.3 Adaptive Composite", "value": "live_adaptive_composite_v383"},
    {"label": "V38.4 Failure-aware router", "value": "live_failure_reversal_v384"},
    {"label": "V38.5 Adaptive Plus", "value": "live_adaptive_plus_v385"},
]

QUALITY_GATE_OPTIONS = [
    {"label": "Off", "value": "off"},
    {"label": "V35.8 strict morning gate", "value": "v358"},
    {"label": "V35.9 live hunter gate", "value": "v359"},
    {"label": "V36.4 professional momentum hybrid", "value": "v364"},
    {"label": "V37.8 mined profitable-indicator pattern", "value": "v377"},
    {"label": "V37.9 decision-time indicator pattern scorer", "value": "v379"},
    {"label": "V38 active pattern scorer", "value": "v38_active"},
    {"label": "V38 stable pattern scorer", "value": "v38_stable"},
    {"label": "V38.2 active plus", "value": "v382_active_plus"},
    {"label": "V38.2 more trades research", "value": "v382_more_trades"},
    {"label": "V38.3 adaptive composite", "value": "v383_adaptive"},
    {"label": "V38.4 failure-aware reversal router", "value": "v384_failure_reversal"},
    {"label": "V38.5 adaptive plus", "value": "v385_adaptive_plus"},
    {"label": "Custom gate", "value": "custom"},
]


def field(label: str, component, className: str = "field"):
    return html.Div(className=className, children=[html.Label(label), component])


def pro_card(title: str, subtitle: str = "", children=None, className: str = "card pro-card"):
    return html.Div(
        className=className,
        children=[
            html.Div(className="section-head compact", children=[html.H3(title), html.Span(subtitle) if subtitle else None]),
            html.Div(className="card-body", children=children or []),
        ],
    )


def details_block(title: str, children, open_: bool = False):
    return html.Details(className="details-card", open=open_, children=[html.Summary(title), html.Div(className="details-body", children=children)])


def strategy_controls():
    return pro_card(
        "Strategy & Universe",
        "Shared by backtest and live worker",
        [
            html.Div(style={"display": "none"}, children=[dcc.Dropdown(id="strategy-profile", options=[{"label": "Symbol/Event Playbook", "value": "symbol_playbook_v25"}], value="symbol_playbook_v25")]),
            html.Div(className="form-grid four", children=[
                field("Strategy preset", dcc.Dropdown(id="settings-preset", options=STRATEGY_PRESET_OPTIONS, value="best_qqq_news", clearable=False)),
                field("Live strategy mode", dcc.Dropdown(id="live-strategy-run-mode", options=live_strategy_run_mode_options(), value="single", clearable=False)),
                field("Watchlist", dcc.Dropdown(id="preset", options=[{"label": k.replace("_", " ").title(), "value": k} for k in WATCHLISTS.keys()], value="v25_playbook", clearable=False)),
                field("Feed", dcc.Dropdown(id="feed", options=[{"label": "IEX - free real-time paper feed", "value": "iex"}, {"label": "SIP real-time - paid/unlimited", "value": "sip"}], value=os.getenv("ALPACA_FEED", "iex"), clearable=False)),
            ]),
            html.Div(className="subtle-note", children="Live strategy mode is saved to Postgres with the rest of the live settings. Single mode trades only the selected preset. All-strategies mode runs every live preset in parallel, tags each signal/order/report row by strategy, and still respects the global risk/capacity controls."),
            field("Custom symbols - optional, overrides watchlist", dcc.Textarea(id="custom-symbols", placeholder="Example: AAPL, TSLA, MU, AMD", value="", className="textarea compact-textarea")),
            html.Div(className="form-grid three", children=[
                field("Max trades/day", dcc.Dropdown(id="max-trades", options=[{"label": f"Top {x}", "value": x} for x in [1,2,3,5,7,10,15]], value=2, clearable=False)),
                field("Direction", dcc.Dropdown(id="direction-mode", options=[{"label": "Long + short", "value": "long_short"}, {"label": "Long only", "value": "long_only"}, {"label": "Short only", "value": "short_only"}], value="long_short", clearable=False)),
                field("Min score", dcc.Input(id="min-score", value=2, type="number", min=0, max=60, step=1)),
            ]),
        ],
    )


def risk_controls():
    return pro_card(
        "Risk & Position Sizing",
        "Kept separate from strategy presets",
        [
            html.Div(id="risk-mode-help", className="subtle-note", children="Risk sizing is independent from the strategy preset. In live mode the worker uses Alpaca equity when available; Account value is only a fallback/backtest value."),
            html.Div(id="live-risk-preview", className="status-pill good", children="Live risk preview loads after the first worker/account snapshot."),
            html.Div(className="form-grid three", children=[
                field("Account value fallback", dcc.Input(id="account-value", value=10000, type="number", min=500, step=100)),
                field("Fixed risk $/trade", dcc.RadioItems(id="risk-dollars-v12", options=[{"label": "$10", "value": 10}, {"label": "$25", "value": 25}, {"label": "$50", "value": 50}, {"label": "$100", "value": 100}, {"label": "$200", "value": 200}, {"label": "$500", "value": 500}], value=100, inline=True)),
                field("Risk mode", dcc.Dropdown(id="risk-mode", options=[{"label": "Fixed dollar risk", "value": "fixed_dollar_risk"}, {"label": "Percent of equity", "value": "percent_equity"}, {"label": "Controlled compounding", "value": "controlled_compounding"}], value="percent_equity", clearable=False)),
            ]),
            html.Div(id="risk-percent-panel", className="form-grid three", children=[
                field("Base risk %", dcc.Input(id="base-risk-pct", value=1.0, type="number", min=0.1, max=5.0, step=0.05)),
            ]),
            html.Div(id="controlled-compounding-panel", children=[
                html.Div(className="form-grid five", children=[
                    field("Min risk $", dcc.Input(id="min-risk-dollars", value=10, type="number", min=0, max=1000, step=5)),
                    field("Max risk $", dcc.Input(id="max-risk-dollars", value=300, type="number", min=0, max=5000, step=25)),
                    field("DD 5% risk %", dcc.Input(id="dd1-risk-pct", value=0.75, type="number", min=0.0, max=5.0, step=0.05)),
                    field("DD 10% risk %", dcc.Input(id="dd2-risk-pct", value=0.50, type="number", min=0.0, max=5.0, step=0.05)),
                    field("Pause DD %", dcc.Input(id="pause-dd-pct", value=15.0, type="number", min=1.0, max=50.0, step=0.5)),
                ])
            ]),
        ],
    )


def live_quality_controls(open_: bool = False):
    return details_block(
        "Advanced live-safe quality gate",
        [
            html.Div(className="form-grid three", children=[
                field("Quality gate", dcc.Dropdown(id="live-quality-gate", options=QUALITY_GATE_OPTIONS, value="off", clearable=False)),
                field("Start ET", dcc.Input(id="quality-start-time", value="10:00", type="text")),
                field("End ET", dcc.Input(id="quality-end-time", value="11:00", type="text")),
            ]),
            html.Div(className="form-grid five", children=[
                field("Min RVOL", dcc.Input(id="quality-min-rvol", value=1.0, type="number", min=0.0, max=20.0, step=0.05)),
                field("Min daily ATR %", dcc.Input(id="quality-min-daily-atr", value=0.0, type="number", min=0.0, max=20.0, step=0.05)),
                field("Min dir RS", dcc.Input(id="quality-min-dir-rs", value=0.0, type="number", min=-20.0, max=20.0, step=0.05)),
                field("Max dir RS", dcc.Input(id="quality-max-dir-rs", value=999.0, type="number", min=-20.0, max=999.0, step=0.05)),
                field("Max abs VWAP", dcc.Input(id="quality-max-abs-vwap", value=1.5, type="number", min=0.0, max=20.0, step=0.05)),
            ]),
            html.Div(className="form-grid four", children=[
                field("Min open RS", dcc.Input(id="quality-min-dir-open-rs", value=-999.0, type="number", min=-999.0, max=999.0, step=0.05)),
                field("Max open RS", dcc.Input(id="quality-max-dir-open-rs", value=999.0, type="number", min=-999.0, max=999.0, step=0.05)),
                field("Min dir VWAP", dcc.Input(id="quality-min-dir-vwap", value=0.5, type="number", min=-20.0, max=20.0, step=0.05)),
                field("Max dir VWAP", dcc.Input(id="quality-max-dir-vwap", value=2.0, type="number", min=-20.0, max=20.0, step=0.05)),
            ]),
        ],
        open_=open_,
    )


def playbook_advanced_controls():
    return details_block(
        "Advanced playbook filters",
        [
            html.Div(className="form-grid four", children=[
                field("Slippage bps", dcc.Input(id="slippage-bps", value=3, type="number", min=0, max=50, step=1)),
                field("News proxy", dcc.Dropdown(id="use-news", options=[{"label": "Off", "value": "false"}, {"label": "On", "value": "true"}], value="false", clearable=False)),
                field("Candle mode", dcc.Dropdown(id="candle-mode", options=[{"label": "Off", "value": "off"}, {"label": "Selective", "value": "selective"}, {"label": "Broad", "value": "broad"}], value="off", clearable=False)),
                field("Macro filter", dcc.Dropdown(id="macro-filter", options=[{"label": "Off", "value": "off"}, {"label": "On", "value": "on"}], value="off", clearable=False)),
            ]),
            html.Div(className="form-grid four", children=[
                field("QQQ stress filter", dcc.Dropdown(id="stress-filter", options=[{"label": "Off", "value": "off"}, {"label": "Skip stress days", "value": "skip"}], value="off", clearable=False)),
                field("Stress threshold", dcc.Input(id="qqq-stress-threshold", value=4.2, type="number", min=0.0, max=20.0, step=0.1)),
                field("Catalyst filter", dcc.Dropdown(id="news-filter", options=[{"label": "Off", "value": "off"}, {"label": "Skip catalyst", "value": "skip"}], value="off", clearable=False)),
                field("Kill switch", dcc.Dropdown(id="kill-switch", options=[{"label": "Off", "value": "off"}, {"label": "On", "value": "on"}], value="off", clearable=False)),
            ]),
            html.Div(className="form-grid two", children=[
                field("Mean reversion", dcc.Dropdown(id="enable-mr", options=[{"label": "Off", "value": "false"}, {"label": "On", "value": "true"}], value="false", clearable=False)),
                field("Opening range / retest", dcc.Dropdown(id="enable-or", options=[{"label": "Off", "value": "false"}, {"label": "On", "value": "true"}], value="true", clearable=False)),
            ]),
        ],
    )


def openai_controls():
    return details_block(
        "Optional OpenAI review filter - research only",
        [
            html.Div(className="form-grid four", children=[
                field("OpenAI filter", dcc.Dropdown(id="openai-filter-mode", options=[{"label": "Off", "value": "off"}, {"label": "On", "value": "on"}], value="off", clearable=False)),
                field("Model", dcc.Input(id="openai-model", value=os.getenv("OPENAI_TRADE_FILTER_MODEL", "gpt-5-mini"), type="text")),
                field("Max decisions/API call", dcc.Input(id="openai-max-candidates", value=5000, type="number", min=1, max=20000, step=1)),
                field("Min confidence", dcc.Input(id="openai-min-confidence", value=0.0, type="number", min=0.0, max=1.0, step=0.05)),
            ])
        ],
    )


def backtest_tab():
    return html.Div(className="tab-panel", children=[
        html.Div(className="panel-grid two-one", children=[
            pro_card("Backtest Setup", "Date range and replay mode", [
                html.Div(className="form-grid four", children=[
                    field("Start", dcc.DatePickerSingle(id="start-date", date=DEFAULT_START.isoformat())),
                    field("End", dcc.DatePickerSingle(id="end-date", date=DEFAULT_END.isoformat())),
                    field("Data session", dcc.Dropdown(id="backtest-session-mode", options=[{"label": "Regular hours only", "value": "regular_only"}, {"label": "Extended hours", "value": "extended_hours"}, {"label": "24/5 available bars", "value": "twenty_four_five"}], value="regular_only", clearable=False)),
                    field("Decision mode", dcc.Dropdown(id="backtest-decision-mode", options=[{"label": "Classic research replay", "value": "end_of_day_top_n"}, {"label": "Live simulation", "value": "live_simulated"}, {"label": "Full raw-bar replay", "value": "raw_bar_replay"}], value="end_of_day_top_n", clearable=False)),
                ]),
                html.Button("Run backtest", id="run-btn", n_clicks=0, className="primary-btn action-btn"),
                html.Div(id="run-status", className="run-status"),
            ]),
            pro_card("Report Output", "Backtest ZIP is generated automatically after each run", [
                html.P("After a backtest finishes, download/open the report ZIP path shown in the status line. It includes selected_trade_market_conditions.csv, selected_trades.csv, symbol/setup summaries, and full context for analysis.", className="subtle-note"),
                html.P("Use Full raw-bar replay for live-style validation. Classic replay is research only.", className="subtle-note strong"),
            ]),
        ]),
        html.Div(id="metrics-row", className="metrics-grid compact-metrics"),
        html.Div(className="grid-2", children=[html.Div(className="card chart-card", children=[dcc.Graph(id="equity-fig", figure=empty_figure("Equity Curve"))]), html.Div(className="card chart-card", children=[dcc.Graph(id="drawdown-fig", figure=empty_figure("Drawdown"))])]),
        html.Div(className="grid-2", children=[html.Div(className="card chart-card", children=[dcc.Graph(id="symbol-fig", figure=empty_figure("P&L by Symbol"))]), html.Div(className="card chart-card", children=[dcc.Graph(id="r-fig", figure=empty_figure("R-Multiple Distribution"))])]),
        table_card("Trades", "Selected trades. Full detail is saved in selected_trade_market_conditions.csv", "trades-table", page_size=12, filterable=True),
        html.Div(className="grid-2", children=[table_card("Symbol Summary", "Which symbols helped or hurt?", "symbol-table", page_size=10), table_card("Setup Summary", "Which trigger type works?", "setup-table", page_size=10)]),
        details_block("Advanced backtest diagnostics", [
            html.Div(className="grid-2", children=[html.Div(className="card chart-card", children=[dcc.Graph(id="setup-fig", figure=empty_figure("P&L by Setup Type"))]), html.Div(className="card chart-card", children=[dcc.Graph(id="daily-fig", figure=empty_figure("Trades by Day"))])]),
            html.Div(className="grid-2", children=[table_card("Daily Summary", "Frequency and daily consistency", "daily-table", page_size=8), table_card("Exit Reason Summary", "Are exits helping or hurting?", "exit-table", page_size=8)]),
            html.Div(className="grid-2", children=[table_card("MFE / MAE Diagnosis", "Entry vs exit failure clues", "mfe-table", page_size=8), table_card("Candlestick Pattern Summary", "Candle contribution", "candle-table", page_size=8)]),
            html.Div(className="grid-2", children=[table_card("Score Band Summary", "Is score predictive?", "score-table", page_size=8), table_card("Time Bucket Summary", "Best time of day", "time-table", page_size=8)]),
        ]),
    ])


def live_tab():
    return html.Div(className="tab-panel", children=[
        dcc.Interval(id="live-refresh-interval", interval=60_000, n_intervals=0),
        dcc.Download(id="live-report-download"),
        pro_card("Live Worker Controls", "Apply selected dashboard settings to Render and export live reports", [
            html.Div(id="live-status", className="live-status", children="Open this Live tab or click Refresh live data now to load the shared Postgres/Alpaca paper history."),
            html.Div(className="form-grid four", children=[
                field("Live entries from ET", dcc.Input(id="live-entry-start-time", value="09:35", type="text")),
                field("Live entries until ET", dcc.Input(id="live-entry-end-time", value="15:55", type="text")),
                field("Require market open", dcc.Dropdown(id="live-require-market-open", options=[{"label": "Yes - Alpaca clock", "value": "true"}, {"label": "No - schedule only", "value": "false"}], value="true", clearable=False)),
                field("Extended-hours entries", dcc.Dropdown(id="live-allow-extended-hours", options=[{"label": "No", "value": "false"}, {"label": "Yes", "value": "true"}], value="false", clearable=False)),
            ]),
            html.Div(className="form-grid two", children=[
                field("Max-hold exits", dcc.Dropdown(id="live-enable-max-hold-exit", options=[{"label": "Enabled", "value": "true"}, {"label": "Disabled", "value": "false"}], value="true", clearable=False)),
                html.Div(className="subtle-note", children="These live schedule controls apply globally to every strategy before strategy-specific filters."),
            ]),
            html.Div(className="live-actions", children=[
                html.Button("Refresh live data now", id="refresh-live-now-btn", n_clicks=0, className="secondary-btn action-btn"),
                html.Button("Apply current settings to live worker", id="apply-live-settings-btn", n_clicks=0, className="primary-btn action-btn"),
                html.Button("Generate live report ZIP", id="generate-live-report-btn", n_clicks=0, className="secondary-btn action-btn"),
                html.A("Direct report download", href="/download-live-report.zip", className="secondary-link", target="_blank"),
            ]),
            html.Div(id="live-action-status", className="run-status"),
        ]),
        html.Div(id="live-metrics-row", className="metrics-grid compact-metrics"),
        symbol_monitor_card(),
        table_card("Live / Paper Trade P&L", "Every strategy trade with realized/unrealized profit and the strategy/gate used", "live-trade-report-table", page_size=12, filterable=True),
        html.Div(className="grid-2", children=[
            table_card("Open Positions", "Positions currently open in Alpaca paper and synced by the worker", "live-positions-table", page_size=8, filterable=True),
            table_card("Recent Signal Plans", "Algorithm decisions, sizing, stop, target and submission status", "live-plans-table", page_size=8, filterable=True),
        ]),
        details_block("Operational details", [
            html.Div(className="grid-2", children=[
                table_card("Recent Alpaca Paper Orders", "Parent brackets and exit legs synced from Alpaca", "live-orders-table", page_size=8, filterable=True),
                table_card("Closed / Recently Closed Positions", "Positions the worker saw open and later missing from Alpaca /positions", "live-closed-positions-table", page_size=8, filterable=True),
            ]),
            table_card("Worker Events", "Scans, blocked entries, submitted paper orders, sync errors and max-hold exits", "live-events-table", page_size=10, filterable=True),
        ]),
    ])


app.layout = html.Div(
    className="page clean-page",
    children=[
        dcc.Interval(id="settings-load-interval", interval=500, n_intervals=0, max_intervals=1),
        dcc.Store(id="live-settings-store", storage_type="local"),
        html.Div(className="hero clean-hero", children=[
            html.Div(children=[html.Div("ALPACA MOMENTUM LAB", className="eyebrow"), html.H1("Trading Research & Live Monitor V38.8"), html.P("Clean two-tab workspace for backtesting, live paper execution, strategy settings and analysis reports.")]),
            html.Div(className=status_class, children=status_text),
        ]),
        html.Div(className="clean-shell", children=[
            strategy_controls(),
            risk_controls(),
            live_quality_controls(open_=False),
            playbook_advanced_controls(),
            openai_controls(),
            dcc.Tabs(id="main-tabs", value="tab-backtest", className="main-tabs", children=[
                dcc.Tab(label="Backtest", value="tab-backtest", className="custom-tab", selected_className="custom-tab selected", children=backtest_tab()),
                dcc.Tab(label="Live", value="tab-live", className="custom-tab", selected_className="custom-tab selected", children=live_tab()),
            ]),
        ]),
    ],
)


def _cfg_value(cfg: dict, key: str, default=None):
    value = (cfg or {}).get(key, default)
    return default if value is None else value

def _cfg_bool_string(cfg: dict, key: str, default: bool = False) -> str:
    value = (cfg or {}).get(key, default)
    return "true" if str(value).lower() in {"1", "true", "yes", "on"} or value is True else "false"


def _normalize_time_et(value, default: str) -> str:
    """Normalize HH:MM dashboard time fields and reject partial typing values.

    This prevents a half-typed value like "2" from being saved to the shared
    Render database and blocking the live worker entry window.
    """
    text = str(value or "").strip()
    if not text or ":" not in text:
        return default
    try:
        hh, mm = text.split(":", 1)
        h = int(hh)
        m = int(mm)
        if 0 <= h <= 23 and 0 <= m <= 59:
            return f"{h:02d}:{m:02d}"
    except Exception:
        pass
    return default


@app.callback(
    Output("settings-preset", "value", allow_duplicate=True),
    Output("live-strategy-run-mode", "value", allow_duplicate=True),
    Output("preset", "value", allow_duplicate=True),
    Output("custom-symbols", "value", allow_duplicate=True),
    Output("feed", "value", allow_duplicate=True),
    Output("backtest-session-mode", "value", allow_duplicate=True),
    Output("live-quality-gate", "value", allow_duplicate=True),
    Output("quality-start-time", "value", allow_duplicate=True),
    Output("quality-end-time", "value", allow_duplicate=True),
    Output("quality-min-rvol", "value", allow_duplicate=True),
    Output("quality-min-daily-atr", "value", allow_duplicate=True),
    Output("quality-min-dir-rs", "value", allow_duplicate=True),
    Output("quality-max-dir-rs", "value", allow_duplicate=True),
    Output("quality-min-dir-open-rs", "value", allow_duplicate=True),
    Output("quality-max-dir-open-rs", "value", allow_duplicate=True),
    Output("quality-min-dir-vwap", "value", allow_duplicate=True),
    Output("quality-max-dir-vwap", "value", allow_duplicate=True),
    Output("quality-max-abs-vwap", "value", allow_duplicate=True),
    Output("account-value", "value", allow_duplicate=True),
    Output("risk-dollars-v12", "value", allow_duplicate=True),
    Output("risk-mode", "value", allow_duplicate=True),
    Output("base-risk-pct", "value", allow_duplicate=True),
    Output("min-risk-dollars", "value", allow_duplicate=True),
    Output("max-risk-dollars", "value", allow_duplicate=True),
    Output("dd1-risk-pct", "value", allow_duplicate=True),
    Output("dd2-risk-pct", "value", allow_duplicate=True),
    Output("pause-dd-pct", "value", allow_duplicate=True),
    Output("min-score", "value", allow_duplicate=True),
    Output("max-trades", "value", allow_duplicate=True),
    Output("slippage-bps", "value", allow_duplicate=True),
    Output("use-news", "value", allow_duplicate=True),
    Output("candle-mode", "value", allow_duplicate=True),
    Output("macro-filter", "value", allow_duplicate=True),
    Output("stress-filter", "value", allow_duplicate=True),
    Output("news-filter", "value", allow_duplicate=True),
    Output("qqq-stress-threshold", "value", allow_duplicate=True),
    Output("kill-switch", "value", allow_duplicate=True),
    Output("enable-mr", "value", allow_duplicate=True),
    Output("enable-or", "value", allow_duplicate=True),
    Output("direction-mode", "value", allow_duplicate=True),
    Output("live-entry-start-time", "value", allow_duplicate=True),
    Output("live-entry-end-time", "value", allow_duplicate=True),
    Output("live-require-market-open", "value", allow_duplicate=True),
    Output("live-allow-extended-hours", "value", allow_duplicate=True),
    Output("live-enable-max-hold-exit", "value", allow_duplicate=True),
    Input("settings-load-interval", "n_intervals"),
    State("live-settings-store", "data"),
    prevent_initial_call="initial_duplicate",
)
def load_saved_live_settings(n_intervals, store_data=None):
    print("[dashboard] load_saved_live_settings start", flush=True)
    # Render has two separate processes: the Dash web service and trading_worker.py.
    # The database row is therefore the source of truth for live trading settings.
    # Browser local storage is only a fallback for local/dev cases where the DB row
    # does not exist yet.  This prevents stale browser storage from overwriting or
    # visually reverting a config that was already applied to the live worker.
    cfg = None
    try:
        db_cfg, db_updated_at = LiveStore(initialize_schema=False).get_state_with_updated_at("live_config_override", None)
        if isinstance(db_cfg, dict) and db_cfg:
            cfg = db_cfg
            print(f"[dashboard] load_saved_live_settings db row updated_at={db_updated_at}", flush=True)
    except Exception as exc:
        print(f"[dashboard] load_saved_live_settings db error: {exc}", flush=True)
        cfg = None
    if not isinstance(cfg, dict) or not cfg:
        if isinstance(store_data, dict) and store_data:
            cfg = store_data
    if not isinstance(cfg, dict) or not cfg:
        print("[dashboard] load_saved_live_settings no saved config", flush=True)
        return tuple([no_update] * 45)
    print(f"[dashboard] load_saved_live_settings loaded preset={cfg.get('settings_preset')} variant={cfg.get('strategy_variant')}", flush=True)
    values = (
        _cfg_value(cfg, "settings_preset", "best_qqq_news"),
        _cfg_value(cfg, "live_strategy_run_mode", "single"),
        _cfg_value(cfg, "watchlist_preset", "v25_playbook"),
        _cfg_value(cfg, "custom_symbols", ""),
        _cfg_value(cfg, "feed", os.getenv("ALPACA_FEED", "iex")),
        _cfg_value(cfg, "backtest_session_mode", "regular_only"),
        _cfg_value(cfg, "live_quality_gate", "off"),
        _cfg_value(cfg, "quality_start_time", "10:00"),
        _cfg_value(cfg, "quality_end_time", "11:00"),
        _cfg_value(cfg, "quality_min_rvol", 1.0),
        _cfg_value(cfg, "quality_min_daily_atr", 0.0),
        _cfg_value(cfg, "quality_min_dir_rs", 0.0),
        _cfg_value(cfg, "quality_max_dir_rs", 999.0),
        _cfg_value(cfg, "quality_min_dir_open_rs", -999.0),
        _cfg_value(cfg, "quality_max_dir_open_rs", 999.0),
        _cfg_value(cfg, "quality_min_dir_vwap", 0.5),
        _cfg_value(cfg, "quality_max_dir_vwap", 2.0),
        _cfg_value(cfg, "quality_max_abs_vwap", 1.5),
        _cfg_value(cfg, "account_value", 10000),
        _cfg_value(cfg, "risk_dollars", 100),
        _cfg_value(cfg, "risk_mode", "percent_equity"),
        _cfg_value(cfg, "base_risk_pct", 1.0),
        _cfg_value(cfg, "min_risk_dollars", 10),
        _cfg_value(cfg, "max_risk_dollars", 300),
        _cfg_value(cfg, "dd1_risk_pct", 0.75),
        _cfg_value(cfg, "dd2_risk_pct", 0.50),
        _cfg_value(cfg, "pause_dd_pct", 15.0),
        _cfg_value(cfg, "min_score", 2),
        _cfg_value(cfg, "max_trades", 2),
        _cfg_value(cfg, "slippage_bps", 2.0),
        "true" if bool(cfg.get("use_news")) else "false",
        _cfg_value(cfg, "candle_mode", "selective"),
        _cfg_value(cfg, "macro_filter", "off"),
        _cfg_value(cfg, "stress_filter", "off"),
        _cfg_value(cfg, "news_filter", "skip"),
        _cfg_value(cfg, "qqq_stress_threshold", 4.2),
        _cfg_value(cfg, "kill_switch", "off"),
        "true" if bool(cfg.get("enable_mr")) else "false",
        "true" if bool(cfg.get("enable_or")) else "false",
        _cfg_value(cfg, "direction_mode", "long_short"),
        _normalize_time_et(_cfg_value(cfg, "live_entry_start_time_et", "09:35"), "09:35"),
        _normalize_time_et(_cfg_value(cfg, "live_entry_end_time_et", "15:55"), "15:55"),
        _cfg_bool_string(cfg, "live_require_market_open", True),
        _cfg_bool_string(cfg, "live_allow_extended_hours_entries", False),
        _cfg_bool_string(cfg, "live_enable_max_hold_exit", True),
    )
    print("[dashboard] load_saved_live_settings ok", flush=True)
    return values

@app.callback(
    Output("live-metrics-row", "children"),
    Output("live-symbol-monitor-summary", "children"),
    Output("live-symbol-monitor-table", "data"), Output("live-symbol-monitor-table", "columns"),
    Output("live-positions-table", "data"), Output("live-positions-table", "columns"),
    Output("live-plans-table", "data"), Output("live-plans-table", "columns"),
    Output("live-orders-table", "data"), Output("live-orders-table", "columns"),
    Output("live-closed-positions-table", "data"), Output("live-closed-positions-table", "columns"),
    Output("live-trade-report-table", "data"), Output("live-trade-report-table", "columns"),
    Output("live-events-table", "data"), Output("live-events-table", "columns"),
    Output("live-status", "children"),
    Input("main-tabs", "value"),
    Input("refresh-live-now-btn", "n_clicks"),
    Input("live-refresh-interval", "n_intervals"),
    prevent_initial_call=False,
)
def refresh_live_paper_monitor(active_tab, refresh_clicks, n_intervals):
    trigger = ctx.triggered_id
    print(f"[dashboard] refresh_live_paper_monitor trigger={trigger} tab={active_tab} n={n_intervals}", flush=True)
    if active_tab != "tab-live":
        return tuple([no_update] * 17)
    try:
        snapshot = load_live_paper_snapshot(days=3650)
        metrics = [metric_card(m.get("label", "Metric"), m.get("value", "--"), m.get("help", "")) for m in snapshot.get("metrics", [])]
        if not metrics:
            metrics = [
                metric_card("Paper Monitor", "No data", "Start the Render worker to populate the shared database"),
                metric_card("Open Positions", "--"),
                metric_card("Signal Plans", "--"),
                metric_card("Worker", "--"),
            ]
        monitor_summary = snapshot.get("symbol_monitor_summary", "Symbol monitor snapshot unavailable.")
        monitor_data, monitor_cols = table_payload(snapshot.get("symbol_monitor", pd.DataFrame()))
        pos_data, pos_cols = table_payload(snapshot.get("open_positions", pd.DataFrame()))
        plan_data, plan_cols = table_payload(snapshot.get("plans", pd.DataFrame()))
        order_data, order_cols = table_payload(snapshot.get("orders", pd.DataFrame()))
        closed_data, closed_cols = table_payload(snapshot.get("closed_positions", pd.DataFrame()))
        live_trade_data, live_trade_cols = table_payload(snapshot.get("trade_report", pd.DataFrame()))
        event_data, event_cols = table_payload(snapshot.get("events", pd.DataFrame()))
        print(f"[dashboard] refresh_live_paper_monitor ok monitor={len(monitor_data)} plans={len(plan_data)} orders={len(order_data)} trades={len(live_trade_data)} events={len(event_data)}", flush=True)
        monitor_symbols = len({str(r.get("symbol", "")).upper() for r in monitor_data if r.get("symbol")}) if isinstance(monitor_data, list) else len(monitor_data)
        monitor_strategies = len({str(r.get("strategy", r.get("strategy_variant", ""))) for r in monitor_data if r.get("strategy") or r.get("strategy_variant")}) if isinstance(monitor_data, list) else 0
        strategy_suffix = f", strategies={monitor_strategies}" if monitor_strategies > 1 else ""
        status_text = snapshot.get("status", "Live paper monitor refreshed.") + f" Loaded symbols={monitor_symbols}{strategy_suffix}, strategy-symbol rows={len(monitor_data)}, trades={len(live_trade_data)}, plans={len(plan_data)}, orders={len(order_data)}, events={len(event_data)}."
        return metrics, monitor_summary, monitor_data, monitor_cols, pos_data, pos_cols, plan_data, plan_cols, order_data, order_cols, closed_data, closed_cols, live_trade_data, live_trade_cols, event_data, event_cols, status_text
    except Exception as exc:
        print(f"[dashboard] refresh_live_paper_monitor error: {exc}", flush=True)
        metrics = [metric_card("Live Monitor", "Error", str(exc)[:80]), metric_card("Open Positions", "--"), metric_card("Signal Plans", "--"), metric_card("Worker", "--")]
        blank = ([], [])
        return metrics, "Live symbol monitor unavailable because dashboard refresh failed.", *blank, *blank, *blank, *blank, *blank, *blank, *blank, f"Live paper monitor error: {exc}"


def _bool_from_dropdown(value) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def _live_config_from_controls(strategy_profile, settings_preset, live_strategy_run_mode, preset, custom_symbols, feed, backtest_session_mode, live_quality_gate,
                               quality_start_time, quality_end_time, quality_min_rvol, quality_min_daily_atr, quality_min_dir_rs, quality_max_dir_rs,
                               quality_min_dir_open_rs, quality_max_dir_open_rs, quality_min_dir_vwap, quality_max_dir_vwap, quality_max_abs_vwap,
                               account_value, risk_dollars, risk_mode, base_risk_pct, min_risk_dollars, max_risk_dollars, dd1_risk_pct, dd2_risk_pct, pause_dd_pct,
                               min_score, max_trades, slippage_bps, use_news, candle_mode, macro_filter, stress_filter, news_filter, qqq_stress_threshold,
                               kill_switch, enable_mr, enable_or, direction_mode, live_entry_start_time, live_entry_end_time, live_require_market_open, live_allow_extended_hours, live_enable_max_hold_exit):
    custom_symbols_active = bool(str(custom_symbols or "").strip())
    symbols = parse_symbols(custom_symbols or "", preset=preset or "v25_playbook")
    if not symbols:
        symbols = WATCHLISTS.get("v25_playbook", [])
    variant = live_variant_from_dashboard(settings_preset, live_quality_gate)
    run_mode = str(live_strategy_run_mode or "single").strip().lower()
    if run_mode not in {"single", "all_strategies"}:
        run_mode = "single"
    active_specs = all_live_strategy_specs() if run_mode == "all_strategies" else []
    return {
        "enabled": True,
        "applied_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "strategy_profile": strategy_profile or "symbol_playbook_v25",
        "settings_preset": settings_preset or "manual",
        "strategy_variant": variant,
        "live_strategy_run_mode": run_mode,
        "active_strategy_count": len(active_specs) if active_specs else 1,
        "active_strategy_variants": [spec.get("variant") for spec in active_specs] if active_specs else [variant],
        "active_strategy_presets": [spec.get("preset") for spec in active_specs] if active_specs else [settings_preset or "manual"],
        "watchlist_preset": preset or "v25_playbook",
        "custom_symbols": custom_symbols or "",
        "custom_symbols_active": custom_symbols_active,
        "symbols": symbols,
        "feed": feed or os.getenv("ALPACA_FEED", "iex"),
        "backtest_session_mode": backtest_session_mode or "regular_only",
        "live_quality_gate": live_quality_gate or "off",
        "quality_start_time": quality_start_time,
        "quality_end_time": quality_end_time,
        "quality_min_rvol": quality_min_rvol,
        "quality_min_daily_atr": quality_min_daily_atr,
        "quality_min_dir_rs": quality_min_dir_rs,
        "quality_max_dir_rs": quality_max_dir_rs,
        "quality_min_dir_open_rs": quality_min_dir_open_rs,
        "quality_max_dir_open_rs": quality_max_dir_open_rs,
        "quality_min_dir_vwap": quality_min_dir_vwap,
        "quality_max_dir_vwap": quality_max_dir_vwap,
        "quality_max_abs_vwap": quality_max_abs_vwap,
        "account_value": account_value,
        "risk_dollars": risk_dollars,
        "risk_mode": risk_mode,
        "base_risk_pct": base_risk_pct,
        "min_risk_dollars": min_risk_dollars if str(risk_mode or "").lower() == "controlled_compounding" else None,
        "max_risk_dollars": max_risk_dollars if str(risk_mode or "").lower() == "controlled_compounding" else None,
        "dd1_risk_pct": dd1_risk_pct if str(risk_mode or "").lower() == "controlled_compounding" else None,
        "dd2_risk_pct": dd2_risk_pct if str(risk_mode or "").lower() == "controlled_compounding" else None,
        "pause_dd_pct": pause_dd_pct if str(risk_mode or "").lower() == "controlled_compounding" else None,
        "min_score": min_score,
        "max_trades": max_trades,
        "max_daily_trades": max_trades,
        "max_open_positions": max_trades,
        "max_orders_per_symbol_per_day": 1,
        "slippage_bps": slippage_bps,
        "use_news": _bool_from_dropdown(use_news),
        "candle_mode": candle_mode,
        "macro_filter": macro_filter,
        "stress_filter": stress_filter,
        "news_filter": news_filter,
        "qqq_stress_threshold": qqq_stress_threshold,
        "kill_switch": kill_switch,
        "enable_mr": _bool_from_dropdown(enable_mr),
        "enable_or": _bool_from_dropdown(enable_or),
        "direction_mode": direction_mode or "long_short",
        "live_entry_start_time_et": _normalize_time_et(live_entry_start_time, "09:35"),
        "live_entry_end_time_et": _normalize_time_et(live_entry_end_time, "15:55"),
        "live_require_market_open": _bool_from_dropdown(live_require_market_open),
        "live_allow_extended_hours_entries": _bool_from_dropdown(live_allow_extended_hours),
        "live_enable_max_hold_exit": _bool_from_dropdown(live_enable_max_hold_exit),
        "selection_mode": "seen_so_far_top_n",
    }


def _save_live_config_to_shared_db(cfg: dict) -> str:
    """Persist dashboard settings to the shared database used by Render worker.

    The real confirmation is: write live_config_override, read it back, and
    verify the unique applied_at_utc token.  The settings row is merged rather
    than overwritten so Alpaca/risk dashboard status is not erased by clicking
    Apply.
    """
    store = LiveStore(initialize_schema=False)
    existing_settings = store.get_state("settings", {})
    store.set_state("live_config_override", cfg)
    store.set_state("settings", _settings_snapshot_from_cfg(cfg, existing_settings))
    saved, updated_at = store.get_state_with_updated_at("live_config_override", None)
    if not isinstance(saved, dict):
        raise RuntimeError("Postgres write verification failed: live_config_override row could not be read back.")
    if str(saved.get("applied_at_utc")) != str(cfg.get("applied_at_utc")):
        raise RuntimeError("Postgres write verification failed: applied_at_utc readback mismatch.")
    return str(updated_at or cfg.get("applied_at_utc") or "")


# Do not save every control keystroke to Postgres.
# The shared Render database is updated only when the user clicks
# "Apply current settings to live worker" below.  This avoids Dash callback
# loops, partial typed times like "2", and the browser tab staying in
# "Updating..." forever.


@app.callback(
    Output("live-action-status", "children"),
    Output("live-report-download", "data"),
    Output("live-settings-store", "data", allow_duplicate=True),
    Input("apply-live-settings-btn", "n_clicks"),
    Input("generate-live-report-btn", "n_clicks"),
    State("strategy-profile", "value"), State("settings-preset", "value"), State("live-strategy-run-mode", "value"), State("preset", "value"), State("custom-symbols", "value"), State("feed", "value"), State("backtest-session-mode", "value"),
    State("live-quality-gate", "value"), State("quality-start-time", "value"), State("quality-end-time", "value"),
    State("quality-min-rvol", "value"), State("quality-min-daily-atr", "value"), State("quality-min-dir-rs", "value"), State("quality-max-dir-rs", "value"),
    State("quality-min-dir-open-rs", "value"), State("quality-max-dir-open-rs", "value"),
    State("quality-min-dir-vwap", "value"), State("quality-max-dir-vwap", "value"), State("quality-max-abs-vwap", "value"),
    State("account-value", "value"), State("risk-dollars-v12", "value"), State("risk-mode", "value"), State("base-risk-pct", "value"), State("min-risk-dollars", "value"), State("max-risk-dollars", "value"), State("dd1-risk-pct", "value"), State("dd2-risk-pct", "value"), State("pause-dd-pct", "value"),
    State("min-score", "value"), State("max-trades", "value"), State("slippage-bps", "value"), State("use-news", "value"), State("candle-mode", "value"),
    State("macro-filter", "value"), State("stress-filter", "value"), State("news-filter", "value"), State("qqq-stress-threshold", "value"), State("kill-switch", "value"), State("enable-mr", "value"), State("enable-or", "value"), State("direction-mode", "value"),
    State("live-entry-start-time", "value"), State("live-entry-end-time", "value"), State("live-require-market-open", "value"), State("live-allow-extended-hours", "value"), State("live-enable-max-hold-exit", "value"),
    prevent_initial_call=True,
)
def live_actions(apply_clicks, report_clicks, strategy_profile, settings_preset, live_strategy_run_mode, preset, custom_symbols, feed, backtest_session_mode, live_quality_gate, quality_start_time, quality_end_time, quality_min_rvol, quality_min_daily_atr, quality_min_dir_rs, quality_max_dir_rs, quality_min_dir_open_rs, quality_max_dir_open_rs, quality_min_dir_vwap, quality_max_dir_vwap, quality_max_abs_vwap, account_value, risk_dollars, risk_mode, base_risk_pct, min_risk_dollars, max_risk_dollars, dd1_risk_pct, dd2_risk_pct, pause_dd_pct, min_score, max_trades, slippage_bps, use_news, candle_mode, macro_filter, stress_filter, news_filter, qqq_stress_threshold, kill_switch, enable_mr, enable_or, direction_mode, live_entry_start_time, live_entry_end_time, live_require_market_open, live_allow_extended_hours, live_enable_max_hold_exit):
    trigger = ctx.triggered_id
    print(f"[dashboard] live_actions trigger={trigger}", flush=True)
    if trigger == "generate-live-report-btn":
        try:
            zip_path = generate_live_report_zip(days=3650)
            return f"Live report generated and download triggered: {zip_path.name}. If the browser does not download it, click Direct download.", dcc.send_file(str(zip_path)), no_update
        except Exception as exc:
            return f"Live report generation failed: {exc}", no_update, no_update
    if trigger == "apply-live-settings-btn":
        cfg = _live_config_from_controls(strategy_profile, settings_preset, live_strategy_run_mode, preset, custom_symbols, feed, backtest_session_mode, live_quality_gate, quality_start_time, quality_end_time, quality_min_rvol, quality_min_daily_atr, quality_min_dir_rs, quality_max_dir_rs, quality_min_dir_open_rs, quality_max_dir_open_rs, quality_min_dir_vwap, quality_max_dir_vwap, quality_max_abs_vwap, account_value, risk_dollars, risk_mode, base_risk_pct, min_risk_dollars, max_risk_dollars, dd1_risk_pct, dd2_risk_pct, pause_dd_pct, min_score, max_trades, slippage_bps, use_news, candle_mode, macro_filter, stress_filter, news_filter, qqq_stress_threshold, kill_switch, enable_mr, enable_or, direction_mode, live_entry_start_time, live_entry_end_time, live_require_market_open, live_allow_extended_hours, live_enable_max_hold_exit)
        try:
            saved_at = _save_live_config_to_shared_db(cfg)
            print(f"[dashboard] live_actions saved verified at {saved_at}", flush=True)
            hb = LiveStore(initialize_schema=False).get_state("heartbeat", {}) or {}
            worker_note = f" Worker heartbeat: {hb.get('status', 'unknown')} / source={hb.get('config_source', 'unknown')}." if isinstance(hb, dict) else ""
            return f"✅ Verified database save at {saved_at}. Saved and read back from Postgres: live_strategy_mode={cfg.get('live_strategy_run_mode', 'single')}, strategies={cfg.get('active_strategy_count', 1)}, strategy_variant={cfg['strategy_variant']}, gate={cfg['live_quality_gate']}, symbols={len(cfg['symbols'])}, max daily trades={cfg['max_daily_trades']}, live entries {cfg['live_entry_start_time_et']}-{cfg['live_entry_end_time_et']} ET.{worker_note}", no_update, cfg
        except Exception as exc:
            # Still persist in this browser so a database issue does not make the UI forget values on refresh.
            print(f"[dashboard] live_actions save error: {exc}", flush=True)
            return f"❌ Database save failed. The worker was NOT updated. Error: {exc}", no_update, cfg
    return no_update, no_update, no_update


@app.callback(
    Output("risk-percent-panel", "style"),
    Output("controlled-compounding-panel", "style"),
    Output("risk-mode-help", "children"),
    Input("risk-mode", "value"),
)
def update_risk_mode_visibility(risk_mode):
    mode = str(risk_mode or "fixed_dollar_risk")
    hidden = {"display": "none"}
    shown = {}
    if mode == "fixed_dollar_risk":
        return hidden, hidden, "Fixed dollar risk is selected: compounding is OFF. The Fixed risk $/trade radio buttons set the same dollar risk for every trade. Base risk %, min/max risk, and DD brakes are ignored."
    if mode == "percent_equity":
        return shown, hidden, "Percent of equity is selected: full compounding is ON. In live mode the worker uses current Alpaca equity; Account value is fallback only. The fixed-risk radio and controlled-compounding min/max are ignored."
    return shown, shown, "Controlled compounding is selected: compounding is ON and uses current Alpaca equity, with min/max risk and drawdown brakes below controlling how much profits can increase risk."


@app.callback(
    Output("live-risk-preview", "children"),
    Input("risk-mode", "value"),
    Input("risk-dollars-v12", "value"),
    Input("base-risk-pct", "value"),
    Input("account-value", "value"),
    Input("min-risk-dollars", "value"),
    Input("max-risk-dollars", "value"),
    Input("dd1-risk-pct", "value"),
    Input("dd2-risk-pct", "value"),
    Input("pause-dd-pct", "value"),
    Input("live-refresh-interval", "n_intervals"),
)
def update_live_risk_preview(risk_mode, fixed_risk, base_pct, account_value, min_risk, max_risk, dd1, dd2, pause_dd, _n):
    mode = str(risk_mode or "fixed_dollar_risk")
    fallback_equity = float(account_value or 0.0)
    equity = fallback_equity
    equity_source = "fallback input"
    buying_power = None
    try:
        acct = LiveStore(initialize_schema=False).latest_account()
        if acct is not None and not acct.empty:
            row = acct.iloc[0]
            eq = float(row.get("equity") or row.get("portfolio_value") or 0.0)
            if eq > 0:
                equity = eq
                equity_source = "Alpaca live equity"
            try:
                buying_power = float(row.get("buying_power") or 0.0)
            except Exception:
                buying_power = None
    except Exception:
        pass
    if equity <= 0:
        equity = fallback_equity or 0.0
    if mode == "fixed_dollar_risk":
        budget = float(fixed_risk or 0.0)
        detail = f"Fixed risk mode: ${budget:,.2f} risk/trade. No compounding. Equity source: {equity_source}."
    elif mode == "percent_equity":
        pct = float(base_pct or 0.0)
        budget = equity * pct / 100.0
        detail = f"Percent-equity mode: {pct:.2f}% x ${equity:,.2f} ({equity_source}) = ${budget:,.2f} intended risk/trade. Fixed-risk radio and min/max fields are ignored."
    else:
        pct = float(base_pct or 0.0)
        raw = equity * pct / 100.0
        min_v = float(min_risk or 0.0)
        max_v = float(max_risk or 0.0)
        budget = raw
        if min_v > 0:
            budget = max(budget, min_v)
        if max_v > 0:
            budget = min(budget, max_v)
        detail = f"Controlled compounding: base {pct:.2f}% x ${equity:,.2f} = ${raw:,.2f}, after min/max = ${budget:,.2f}. DD brakes: {dd1}% / {dd2}%, pause at {pause_dd}%."
    if buying_power is not None and buying_power > 0:
        detail += f" Current Alpaca buying power: ${buying_power:,.2f}."
    return detail


@app.callback(
    Output("max-trades", "value"),
    Output("direction-mode", "value"),
    Output("min-score", "value"),
    Output("candle-mode", "value"),
    Output("use-news", "value"),
    Output("news-filter", "value"),
    Output("stress-filter", "value"),
    Output("qqq-stress-threshold", "value"),
    Output("macro-filter", "value"),
    Output("kill-switch", "value"),
    Output("enable-mr", "value"),
    Output("enable-or", "value"),
    Output("backtest-decision-mode", "value"),
    Output("live-quality-gate", "value"),
    Output("quality-start-time", "value"),
    Output("quality-end-time", "value"),
    Output("quality-min-rvol", "value"),
    Output("quality-min-daily-atr", "value"),
    Output("quality-min-dir-rs", "value"),
    Output("quality-max-dir-rs", "value"),
    Output("quality-min-dir-open-rs", "value"),
    Output("quality-max-dir-open-rs", "value"),
    Output("quality-min-dir-vwap", "value"),
    Output("quality-max-dir-vwap", "value"),
    Output("quality-max-abs-vwap", "value"),
    Input("settings-preset", "value"),
)
def apply_settings_preset(preset_name):
    if preset_name == "manual":
        return tuple([no_update] * 25)
    if preset_name == "live_raw_optimized_v358":
        # Load defaults only. Users can still modify every control before running.
        # Risk sizing stays separate and is never changed here.
        return (2, "long_short", 2, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v358", "10:00", "11:00", 1.50, 0.0, 0.00, 2.00, 0.00, 2.00, -999.0, 999.0, 999.0)
    if preset_name == "live_hunter_v359":
        # Load defaults only. Users can still modify every control before running.
        # Risk sizing stays separate and is never changed here.
        return (1, "long_short", 2, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v359", "10:00", "11:00", 1.00, 0.0, 0.00, 999.00, -999.00, 999.00, 0.50, 2.00, 1.50)
    if preset_name == "live_longrun_robust_v362":
        # Long-period raw-replay findings from 2021-2026: low-ATR days and weak
        # directional confirmation were the main loss sources.  Defaults remain
        # editable through the UI and risk sizing stays separate.
        return (1, "long_short", 2, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "custom", "10:00", "11:00", 1.20, 4.00, 0.00, 3.00, 0.00, 999.00, 0.50, 2.00, 1.50)
    if preset_name == "live_grid_robust_v363":
        # V36.3: chosen from raw-bar/live-style grid testing on the uploaded raw
        # IEX bars.  Conservative because broader/high-frequency combinations
        # failed long-run validation. Defaults only; all controls remain editable.
        return (1, "long_short", 0, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "custom", "10:00", "11:00", 1.20, 4.00, 0.00, 3.00, 0.00, 5.00, 0.50, 2.00, 1.50)
    if preset_name == "live_professional_momentum_v364":
        # V36.4: GitHub-inspired live-safe professional momentum hybrid.
        # Defaults only; risk sizing and every visible filter remain editable.
        # Uses Top 1, no mean-reversion, OR/retest enabled, and a 10:00-12:00
        # momentum/VWAP/relative-strength gate tested on raw-bar live replay.
        return (1, "long_short", 10, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v364", "10:00", "12:00", 1.00, 4.00, -1.00, 3.00, -999.00, 999.00, 0.50, 2.00, 2.00)
    if preset_name == "live_positive_context_v377":
        # V37.8: Uses mined multi-indicator profitable-pattern matcher. This runs in
        # full raw-bar/live replay mode by default and only accepts candidates
        # whose live indicators match historically profitable contexts.
        return (2, "long_short", 2, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v377", "10:00", "12:00", 0.00, 0.00, -999.00, 999.00, -999.00, 999.00, -999.00, 999.00, 999.00)
    if preset_name == "live_indicator_pattern_v379":
        # V37.9: decision-time indicator pattern scorer. It uses causal signal-bar
        # values only and ranks by pattern score before the regular Top-N selector.
        return (2, "long_short", 0, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v379", "09:35", "12:00", 0.00, 0.00, -999.00, 999.00, -999.00, 999.00, -999.00, 999.00, 999.00)
    # Exact strategy filters from the user's strongest uploaded report 153601.
    if preset_name == "live_active_pattern_v38":
        # V38 active mode: higher-frequency research preset. Uses live-safe
        # signal-time fields only and leaves risk/compounding separate.
        return (5, "long_short", 5, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v38_active", "09:35", "11:00", 0.00, 0.00, -999.00, 999.00, -999.00, 999.00, -999.00, 999.00, 999.00)
    if preset_name == "live_stable_pattern_v38":
        # V38 stable mode: lower frequency than active but more stable across
        # yearly slices in the raw-live candidate test.
        return (15, "long_short", 5, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v38_stable", "09:35", "11:00", 0.00, 0.00, -999.00, 999.00, -999.00, 999.00, -999.00, 999.00, 999.00)
    if preset_name == "live_active_plus_v382":
        # V38.2 Active Plus - more trades: more trades than stable with a stricter 10:00-11:00
        # non-engulfing indicator-state gate. Defaults only; all controls remain editable.
        return (15, "long_short", 0, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v382_active_plus", "10:00", "11:00", 0.00, 0.00, -999.00, 999.00, -999.00, 999.00, -999.00, 999.00, 999.00)
    if preset_name == "live_adaptive_composite_v383":
        # V38.3 Adaptive Composite: combines the strongest regimes found across the
        # uploaded live/raw reports: stable long ORB/VWAP, high-RVOL directional
        # VWAP/RS, and short late-morning weakness.  Defaults are live-safe and
        # use only signal-time fields; risk sizing remains separate.
        return (3, "long_short", 5, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v383_adaptive", "09:35", "12:00", 0.00, 0.00, -999.00, 999.00, -999.00, 999.00, -999.00, 999.00, 999.00)
    if preset_name == "live_failure_reversal_v384":
        return (5, "long_short", 0, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v384_failure_reversal", "09:35", "12:00", 0.00, 0.00, -999.00, 999.00, -999.00, 999.00, -999.00, 999.00, 999.00)
    if preset_name == "live_adaptive_plus_v385":
        return (10, "long_short", 0, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v385_adaptive_plus", "09:35", "11:55", 0.00, 0.00, -999.00, 999.00, -999.00, 999.00, -999.00, 999.00, 999.00)
    if preset_name == "live_more_trades_v382":
        # Research-only higher-activity version. More opportunities, weaker robustness.
        return (3, "long_short", 0, "off", "false", "off", "off", 4.2, "off", "off", "false", "true", "raw_bar_replay", "v382_more_trades", "09:45", "11:00", 0.00, 0.00, -999.00, 999.00, -999.00, 999.00, -999.00, 999.00, 999.00)

    # This deliberately does NOT touch risk sizing or compounding controls.
    return (2, "long_short", 2, "selective", "true", "skip", "skip", 4.2, "off", "off", "false", "false", "end_of_day_top_n", "off", "10:00", "11:00", 1.00, 0.0, 0.00, 999.00, -999.00, 999.00, 0.50, 2.00, 1.50)


@app.callback(
    Output("metrics-row", "children"),
    Output("equity-fig", "figure"),
    Output("drawdown-fig", "figure"),
    Output("symbol-fig", "figure"),
    Output("r-fig", "figure"),
    Output("setup-fig", "figure"),
    Output("daily-fig", "figure"),
    Output("symbol-table", "data"), Output("symbol-table", "columns"),
    Output("setup-table", "data"), Output("setup-table", "columns"),
    Output("daily-table", "data"), Output("daily-table", "columns"),
    Output("exit-table", "data"), Output("exit-table", "columns"),
    Output("mfe-table", "data"), Output("mfe-table", "columns"),
    Output("candle-table", "data"), Output("candle-table", "columns"),
    Output("score-table", "data"), Output("score-table", "columns"),
    Output("time-table", "data"), Output("time-table", "columns"),
    Output("trades-table", "data"), Output("trades-table", "columns"),
    Output("run-status", "children"),
    Input("run-btn", "n_clicks"),
    State("strategy-profile", "value"), State("settings-preset", "value"), State("preset", "value"), State("custom-symbols", "value"),
    State("start-date", "date"), State("end-date", "date"), State("feed", "value"),
    State("backtest-session-mode", "value"), State("backtest-decision-mode", "value"),
    State("live-quality-gate", "value"), State("quality-start-time", "value"), State("quality-end-time", "value"),
    State("quality-min-rvol", "value"), State("quality-min-daily-atr", "value"), State("quality-min-dir-rs", "value"), State("quality-max-dir-rs", "value"),
    State("quality-min-dir-open-rs", "value"), State("quality-max-dir-open-rs", "value"),
    State("quality-min-dir-vwap", "value"), State("quality-max-dir-vwap", "value"), State("quality-max-abs-vwap", "value"),
    State("openai-filter-mode", "value"), State("openai-model", "value"), State("openai-max-candidates", "value"), State("openai-min-confidence", "value"), State("account-value", "value"), State("risk-dollars-v12", "value"),
    State("risk-mode", "value"), State("base-risk-pct", "value"), State("min-risk-dollars", "value"), State("max-risk-dollars", "value"),
    State("dd1-risk-pct", "value"), State("dd2-risk-pct", "value"), State("pause-dd-pct", "value"),
    State("min-score", "value"), State("max-trades", "value"), State("slippage-bps", "value"), State("use-news", "value"), State("candle-mode", "value"),
    State("macro-filter", "value"), State("stress-filter", "value"), State("news-filter", "value"), State("qqq-stress-threshold", "value"), State("kill-switch", "value"),
    State("enable-mr", "value"), State("enable-or", "value"), State("direction-mode", "value"),
    prevent_initial_call=True,
)
def run_backtest_callback(n_clicks, strategy_profile, settings_preset, preset, custom_symbols, start_date, end_date, feed, backtest_session_mode, backtest_decision_mode, live_quality_gate, quality_start_time, quality_end_time, quality_min_rvol, quality_min_daily_atr, quality_min_dir_rs, quality_max_dir_rs, quality_min_dir_open_rs, quality_max_dir_open_rs, quality_min_dir_vwap, quality_max_dir_vwap, quality_max_abs_vwap, openai_filter_mode, openai_model, openai_max_candidates, openai_min_confidence, account_value, risk_dollars, risk_mode, base_risk_pct, min_risk_dollars, max_risk_dollars, dd1_risk_pct, dd2_risk_pct, pause_dd_pct, min_score, max_trades, slippage_bps, use_news, candle_mode, macro_filter, stress_filter, news_filter, qqq_stress_threshold, kill_switch, enable_mr, enable_or, direction_mode):
    blank_tables = ([], []) * 9
    if not n_clicks:
        metrics = [metric_card("Win Rate", "--", "target: 62-75% after validation"), metric_card("Total P&L", "--"), metric_card("Profit Factor", "--"), metric_card("Trades", "--"), metric_card("Avg MFE", "--"), metric_card("Expectancy", "--")]
        empty = empty_figure
        return (metrics, empty("Equity Curve"), empty("Drawdown"), empty("P&L by Symbol"), empty("R-Multiple Distribution"), empty("P&L by Setup Type"), empty("Trades by Day"), *blank_tables, "Ready. Add Alpaca keys to .env, choose symbols, then run a backtest. Note: per-symbol results are portfolio-selected after Top trades/day, not standalone one-symbol results.")

    try:
        custom_symbols_active = bool(str(custom_symbols or "").strip())
        symbols = parse_symbols(custom_symbols, preset=preset)
        risk_value = float(risk_dollars)
        account_value_f = float(account_value or 10000)
        base_risk_pct_f = float(base_risk_pct or 1.0)
        risk_mode_s = str(risk_mode or "controlled_compounding")
        if risk_value <= 0:
            raise ValueError("Fixed Risk $/trade must be greater than zero")
        if base_risk_pct_f <= 0:
            raise ValueError("Base risk percent must be greater than zero")
        params = StrategyParams(
            strategy_profile=strategy_profile or "symbol_playbook_v25",
            direction_mode=str(direction_mode or "long_only"),
            backtest_session_mode=str(backtest_session_mode or "regular_only"),
            backtest_decision_mode=str(backtest_decision_mode or "end_of_day_top_n"),
            v25_allow_generic_symbols=custom_symbols_active,
            openai_trade_filter_enabled=(str(openai_filter_mode or "off").lower() == "on"),
            openai_trade_filter_model=str(openai_model or os.getenv("OPENAI_TRADE_FILTER_MODEL", "gpt-5-mini")),
            openai_trade_filter_max_candidates_per_day=int(openai_max_candidates or 5000),
            openai_trade_filter_batch_mode="full_run",
            openai_trade_filter_min_confidence=float(openai_min_confidence or 0.0),
            initial_account_value=account_value_f,
            risk_per_trade_dollars=risk_value,
            requested_risk_percent=base_risk_pct_f,
            risk_per_trade_pct=base_risk_pct_f / 100.0,
            position_sizing_mode=risk_mode_s,
            compounding_base_risk_pct=base_risk_pct_f,
            compounding_min_risk_dollars=float(min_risk_dollars or 0.0),
            compounding_max_risk_dollars=float(max_risk_dollars or 0.0),
            compounding_dd1_risk_pct=float(dd1_risk_pct or 0.75),
            compounding_dd2_risk_pct=float(dd2_risk_pct or 0.50),
            compounding_pause_dd_pct=float(pause_dd_pct or 15.0),
            max_position_notional_pct=9999.0,
            min_candidate_score=float(min_score or 70),
            max_trades_per_day=int(max_trades or 10),
            slippage_bps=float(slippage_bps or 0),
            candle_pattern_mode=str(candle_mode or "exit_only"),
            enable_mean_reversion=(str(enable_mr).lower() == "true"),
            enable_or_retest=(str(enable_or).lower() == "true"),
            v27_macro_filter_mode=str(macro_filter or "off"),
            v27_market_stress_mode=str(stress_filter or "off"),
            v27_news_filter_mode=str(news_filter or "off"),
            v27_symbol_kill_switch_mode=str(kill_switch or "off"),
            v27_qqq_stress_abs_change_pct=float(qqq_stress_threshold or 1.25),
            enable_v358_live_quality_filter=False,
            enable_v359_live_hunter_filter=False,
            enable_v364_professional_momentum_filter=False,
            enable_v377_positive_context_filter=False,
        )
        # Live raw quality gate is now a user-editable control, not a hidden preset override.
        # It can be combined with any other visible filter and any risk/compounding mode.
        gate_mode = str(live_quality_gate or "off").lower()
        if gate_mode == "v358":
            params.enable_v358_live_quality_filter = True
            params.v358_quality_start_time = str(quality_start_time or "10:00")
            params.v358_quality_end_time = str(quality_end_time or "11:00")
            params.v358_min_rvol = float(quality_min_rvol or 0.0)
            params.v358_min_daily_atr_pct = float(quality_min_daily_atr or 0.0)
            params.v358_min_directional_rs = float(quality_min_dir_rs or -999.0)
            params.v358_max_directional_rs = float(quality_max_dir_rs or 999.0)
            params.v358_min_directional_open_rs = float(quality_min_dir_open_rs or -999.0)
            params.v358_max_directional_open_rs = float(quality_max_dir_open_rs or 999.0)
            params.v358_min_directional_vwap_extension_atr = float(quality_min_dir_vwap or -999.0)
            params.v358_max_directional_vwap_extension_atr = float(quality_max_dir_vwap or 999.0)
            params.v358_max_abs_vwap_extension_atr = float(quality_max_abs_vwap or 999.0)
        elif gate_mode == "v364":
            params.enable_v364_professional_momentum_filter = True
            params.v364_quality_start_time = str(quality_start_time or "10:00")
            params.v364_quality_end_time = str(quality_end_time or "12:00")
            params.v364_min_rvol = float(quality_min_rvol or 0.0)
            params.v364_min_daily_atr_pct = float(quality_min_daily_atr or 0.0)
            params.v364_min_directional_rs = float(quality_min_dir_rs or -999.0)
            params.v364_max_directional_rs = float(quality_max_dir_rs or 999.0)
            params.v364_min_directional_open_rs = float(quality_min_dir_open_rs or -999.0)
            params.v364_max_directional_open_rs = float(quality_max_dir_open_rs or 999.0)
            params.v364_min_directional_vwap_extension_atr = float(quality_min_dir_vwap or -999.0)
            params.v364_max_directional_vwap_extension_atr = float(quality_max_dir_vwap or 999.0)
            params.v364_max_abs_vwap_extension_atr = float(quality_max_abs_vwap or 999.0)
        elif gate_mode == "v377":
            params.enable_v377_positive_context_filter = True
        elif gate_mode == "v379":
            params.enable_v379_decision_pattern_filter = True
            params.v379_pattern_mode = "balanced_vwap_prevhigh"
        elif gate_mode == "v38_active":
            params.enable_v379_decision_pattern_filter = True
            params.v379_pattern_mode = "v38_active"
        elif gate_mode == "v38_stable":
            params.enable_v379_decision_pattern_filter = True
            params.v379_pattern_mode = "v38_stable"
        elif gate_mode == "v382_active_plus":
            params.enable_v379_decision_pattern_filter = True
            params.v379_pattern_mode = "v382_active_plus"
        elif gate_mode == "v383_adaptive":
            params.enable_v379_decision_pattern_filter = True
            params.v379_pattern_mode = "v383_adaptive"
        elif gate_mode == "v384_failure_reversal":
            params.enable_v379_decision_pattern_filter = True
            params.v379_pattern_mode = "v384_failure_reversal"
        elif gate_mode == "v385_adaptive_plus":
            params.enable_v379_decision_pattern_filter = True
            params.v379_pattern_mode = "v385_adaptive_plus"
        elif gate_mode == "v382_more_trades":
            params.enable_v379_decision_pattern_filter = True
            params.v379_pattern_mode = "v382_more_trades"
        elif gate_mode in {"v359", "custom"}:
            params.enable_v359_live_hunter_filter = True
            params.v359_quality_start_time = str(quality_start_time or "10:00")
            params.v359_quality_end_time = str(quality_end_time or "11:00")
            params.v359_min_rvol = float(quality_min_rvol or 0.0)
            params.v359_min_daily_atr_pct = float(quality_min_daily_atr or 0.0)
            params.v359_min_directional_rs = float(quality_min_dir_rs or -999.0)
            params.v359_max_directional_rs = float(quality_max_dir_rs or 999.0)
            params.v359_min_directional_open_rs = float(quality_min_dir_open_rs or -999.0)
            params.v359_max_directional_open_rs = float(quality_max_dir_open_rs or 999.0)
            params.v359_min_directional_vwap_extension_atr = float(quality_min_dir_vwap or -999.0)
            params.v359_max_directional_vwap_extension_atr = float(quality_max_dir_vwap or 999.0)
            params.v359_max_abs_vwap_extension_atr = float(quality_max_abs_vwap or 999.0)

        if str(strategy_profile or "") == "symbol_playbook_v25":
            params.max_open_positions = int(max_trades or 2)
            params.daily_loss_limit_pct = 100.0
            params.max_consecutive_losses = 99
            params.v25_target_r = 0.75
            params.v25_max_hold_bars = 12
            # Presets load defaults through the GUI callback only. Do not override
            # user-edited controls here; every visible field must remain active.

        if custom_symbols_active:
            # Custom-symbol mode must genuinely run the user's watchlist, not the
            # packaged V25 replay universe.  The original V25 liquidity limits were
            # tuned for large/liquid names and would silently eliminate most penny
            # or micro-cap symbols before signals are created.  For custom research
            # we relax tradability gates while preserving the same V25 event/VP
            # structure, target, stop, hold, sizing and portfolio rules.
            params.v25_allow_generic_symbols = True
            params.min_price = 0.01
            params.min_avg_20d_dollar_volume = 0.0
            params.min_current_5m_dollar_volume = 0.0
            params.min_daily_atr_pct = 0.0
            params.max_daily_atr_pct = 999.0
            params.v25_min_rvol = 0.0

        result = run_backtest(symbols=symbols, start_date=start_date, end_date=end_date, params=params, feed=feed, use_cache=True, use_news=(str(use_news).lower() == "true"), export_report=True, session_mode=str(backtest_session_mode or "regular_only"))
        metrics = result["metrics"]
        selected = result.get("selected_trades", pd.DataFrame())
        candidates = result.get("candidates", pd.DataFrame())
        report_paths = result.get("report_paths", {})

        metrics_cards = [
            metric_card("Win Rate", fmt_pct(metrics["win_rate"]), "target range: 62-75%"),
            metric_card("Total P&L", fmt_money(metrics["total_pnl"]), fmt_pct(metrics["total_return_pct"])),
            metric_card("Profit Factor", f"{metrics['profit_factor']:.2f}", ">1.50 preferred"),
            metric_card("Trades", f"{metrics['total_trades']}", f"{len(candidates)} raw candidates"),
            metric_card("Risk Used", fmt_money(metrics.get("risk_per_trade_dollars", metrics.get("avg_risk_budget", 0))), f"actual avg {fmt_money(metrics.get('avg_actual_dollars_at_risk', 0))}"),
            metric_card("Avg Notional", fmt_money(metrics.get("avg_notional", 0)), f"{fmt_pct(metrics.get('avg_notional_pct', 0))} of equity"),
            metric_card("Avg MFE", f"{metrics.get('avg_mfe_r', 0):.2f}R", "favorable excursion"),
            metric_card("Expectancy", f"{metrics['expectancy_r']:.2f}R", "avg R per trade"),
        ]

        equity_fig = make_equity_fig(result["equity_curve"])
        drawdown_fig = make_drawdown_fig(result["drawdown_curve"])
        symbol_fig = make_symbol_fig(result["symbol_summary"])
        r_fig = make_r_distribution_fig(selected)
        setup_fig = make_setup_fig(result.get("setup_summary", pd.DataFrame()))
        daily_fig = make_daily_fig(result.get("daily_summary", pd.DataFrame()))

        sym_data, sym_cols = table_payload(result.get("symbol_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor", "avg_duration"])
        setup_data, setup_cols = table_payload(result.get("setup_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor", "avg_mfe_r", "avg_mae_r"])
        daily_data, daily_cols = table_payload(result.get("daily_summary", pd.DataFrame()), money_cols=["pnl"], pct_cols=["win_rate"], round_cols=["avg_score"])
        exit_data, exit_cols = table_payload(result.get("exit_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "profit_factor", "avg_mfe_r", "avg_mae_r"])
        mfe_data, mfe_cols = table_payload(result.get("mfe_mae_summary", pd.DataFrame()), round_cols=["value"])
        candle_data, candle_cols = table_payload(result.get("candle_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor", "avg_mfe_r", "avg_mae_r"])
        score_data, score_cols = table_payload(result.get("score_band_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor"])
        time_data, time_cols = table_payload(result.get("time_bucket_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor"])

        trade_cols = ["symbol", "entry_time_et", "exit_time_et", "trigger_type", "quality", "candidate_score", "openai_confidence", "openai_reason", "entry_candle_pattern", "entry_price", "stop_price", "shares", "notional", "risk_budget", "actual_dollars_at_risk", "pnl_dollars", "pnl_dollars_from_shares", "risk_application_delta", "r_multiple", "mfe_r", "mae_r", "position_sizing_mode", "low_followthrough_mode", "target1_hit", "target2_hit", "exit_reason"]
        if not selected.empty:
            for col in trade_cols:
                if col not in selected.columns:
                    selected[col] = ""
            trade_data = selected[trade_cols].copy()
        else:
            trade_data = pd.DataFrame(columns=trade_cols)
        trade_data, trade_columns = table_payload(trade_data, money_cols=["entry_price", "stop_price", "pnl_dollars"], round_cols=["candidate_score", "candle_pattern_score", "r_multiple", "mfe_r", "mae_r"])
        custom_note = " custom_symbols_override=ON(raw local Alpaca bars + missing-data fetch, relaxed custom liquidity gates)" if custom_symbols_active else " custom_symbols_override=OFF(preset/replay baseline when regular)"
        live_quality_notes = []
        if bool(getattr(params, "enable_v358_live_quality_filter", False)):
            live_quality_notes.append(
                f"quality_gate=v358({params.v358_quality_start_time}-{params.v358_quality_end_time}, "
                f"rvol>={params.v358_min_rvol}, dailyATR>={getattr(params, 'v358_min_daily_atr_pct', 0.0)}, dirRS={params.v358_min_directional_rs}..{params.v358_max_directional_rs}, "
                f"openRS={params.v358_min_directional_open_rs}..{params.v358_max_directional_open_rs}, "
                f"dirVWAP={params.v358_min_directional_vwap_extension_atr}..{params.v358_max_directional_vwap_extension_atr}, "
                f"absVWAP<={params.v358_max_abs_vwap_extension_atr})"
            )
        if bool(getattr(params, "enable_v359_live_hunter_filter", False)):
            live_quality_notes.append(
                f"quality_gate={gate_mode}({params.v359_quality_start_time}-{params.v359_quality_end_time}, "
                f"rvol>={params.v359_min_rvol}, dailyATR>={getattr(params, 'v359_min_daily_atr_pct', 0.0)}, dirRS={params.v359_min_directional_rs}..{params.v359_max_directional_rs}, "
                f"openRS={params.v359_min_directional_open_rs}..{params.v359_max_directional_open_rs}, "
                f"dirVWAP={params.v359_min_directional_vwap_extension_atr}..{params.v359_max_directional_vwap_extension_atr}, "
                f"absVWAP<={params.v359_max_abs_vwap_extension_atr})"
            )
        if bool(getattr(params, "enable_v377_positive_context_filter", False)):
            live_quality_notes.append("quality_gate=v378_mined_profitable_indicator_pattern(live-safe)")
        if bool(getattr(params, "enable_v379_decision_pattern_filter", False)):
            live_quality_notes.append(f"quality_gate=decision_time_pattern({getattr(params, 'v379_pattern_mode', '')})")
        live_quality_note = " " + "; ".join(live_quality_notes) if live_quality_notes else " quality_gate=OFF"
        diag = result.get("diagnostics", {}) or {}
        skipped_df = result.get("skipped_symbols", pd.DataFrame())
        skipped_preview = ""
        if skipped_df is not None and not skipped_df.empty:
            try:
                skipped_preview = " skipped=" + "; ".join((skipped_df.head(8)["symbol"].astype(str) + ":" + skipped_df.head(8)["reason"].astype(str)).tolist())
            except Exception:
                skipped_preview = f" skipped_symbols={len(skipped_df)}"
        cache_note = ""
        try:
            cs = diag.get("cache_status", [])
            if cs:
                downloaded = sum(int(x.get("downloaded_rows", 0) or 0) for x in cs if isinstance(x, dict))
                cache_note = f" cache_downloaded_rows={downloaded}"
        except Exception:
            cache_note = ""
        prefilter_stats = diag.get("prefilter_stats", {}) if isinstance(diag, dict) else {}
        prefilter_note = ""
        if isinstance(prefilter_stats, dict):
            prefilter_note = (
                f" filters_removed=macro:{prefilter_stats.get('macro_filtered', 0)},"
                f" qqq:{prefilter_stats.get('stress_filtered', 0)},"
                f" catalyst:{prefilter_stats.get('news_filtered', 0)},"
                f" kill:{diag.get('v27_kill_switch_skipped_selected_trades', 0)}"
            )
        user_alerts = _format_user_alerts(result, custom_symbols_active)
        ai_diag = (diag.get("openai_filter") or {}) if isinstance(diag, dict) else {}
        ai_note = ""
        if ai_diag.get("enabled"):
            ai_note = f" OpenAI filter: model={ai_diag.get('model')}, review_mode={ai_diag.get('review_mode')}, batch_mode={ai_diag.get('batch_mode')}, api_calls={ai_diag.get('api_calls')}, prompt_sizes={ai_diag.get('prompt_candidate_counts')}, reviewed={ai_diag.get('reviewed')}, approved={ai_diag.get('approved')}, after_filter={diag.get('openai_candidates_after_filter')}."
        top_note = f" top_requested={max_trades}, top_effective={diag.get('effective_top_trades_per_day', diag.get('top_trades_per_day', max_trades))}, max_selected_per_day={diag.get('actual_max_selected_trades_in_single_day', 'n/a')}, days_hit_requested_top={diag.get('days_at_or_above_requested_top', 'n/a')}, full_candidate_universe={diag.get('full_candidate_universe_available', 'n/a')}, source={diag.get('dynamic_candidate_source_kind', 'n/a')}, fallback_top_limit={diag.get('fallback_top_file_limit', 'n/a')}."
        status = f"Backtest complete: {len(symbols)} symbols ({', '.join(symbols[:12])}{'...' if len(symbols) > 12 else ''}), {start_date} to {end_date}, feed={feed}, session={backtest_session_mode}, decision_mode={diag.get('backtest_decision_mode', backtest_decision_mode)},{custom_note}, execution={result.get('execution_timeframe')}, raw_alerts={diag.get('raw_alerts', 'n/a')}, raw_candidates={diag.get('raw_candidates', len(candidates) if candidates is not None else 'n/a')}, selected_trades={metrics.get('total_trades', 0)},{top_note}{live_quality_note},{prefilter_note},{cache_note}{skipped_preview},{ai_note} UI risk=${float(risk_dollars):.2f}, engine risk=${metrics.get('risk_per_trade_dollars', metrics.get('avg_risk_budget', 0)):.2f}, sizing={risk_mode_s}, fixed risk input=${risk_dollars}, base risk={base_risk_pct_f:.2f}%, max risk=${float(max_risk_dollars or 0):.2f}, min score={min_score}, candle mode={candle_mode}, macro={macro_filter}, qqq stress={stress_filter}, news filter={news_filter}, kill switch={kill_switch}, news/catalyst proxy={use_news}, mean reversion={enable_mr}, OR/retest={enable_or}, direction={direction_mode}. {user_alerts} Report: {report_paths.get('latest_zip', report_paths.get('zip_path', 'not saved'))}"
        return metrics_cards, equity_fig, drawdown_fig, symbol_fig, r_fig, setup_fig, daily_fig, sym_data, sym_cols, setup_data, setup_cols, daily_data, daily_cols, exit_data, exit_cols, mfe_data, mfe_cols, candle_data, candle_cols, score_data, score_cols, time_data, time_cols, trade_data, trade_columns, status
    except Exception as exc:
        metrics_cards = [metric_card("Error", "Backtest failed"), metric_card("Fix", "Check keys/date/feed"), metric_card("Details", str(exc)[:60])]
        empty = empty_figure
        return (metrics_cards, empty("Equity Curve"), empty("Drawdown"), empty("P&L by Symbol"), empty("R-Multiple Distribution"), empty("P&L by Setup Type"), empty("Trades by Day"), *blank_tables, f"Error: {exc}")


def make_equity_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Equity Curve")
    fig = px.line(df, x="exit_time", y="equity", title="Equity Curve")
    fig.update_traces(line_width=3)
    return polish_fig(fig, y_title="Account Equity ($)")


def make_drawdown_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Drawdown")
    fig = px.area(df, x="exit_time", y="drawdown_pct", title="Drawdown %")
    return polish_fig(fig, y_title="Drawdown %")


def make_symbol_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("P&L by Symbol")
    fig = px.bar(df.sort_values("total_pnl"), x="symbol", y="total_pnl", title="P&L by Symbol (Portfolio-Selected Trades)", hover_data=["trades", "win_rate", "avg_r"])
    return polish_fig(fig, y_title="P&L ($)")


def make_r_distribution_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("R-Multiple Distribution")
    fig = px.histogram(df, x="r_multiple", nbins=30, title="R-Multiple Distribution")
    fig.add_vline(x=0, line_dash="dash")
    return polish_fig(fig, y_title="Trades", x_title="R Multiple")


def make_setup_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("P&L by Setup Type")
    fig = px.bar(df.sort_values("total_pnl"), x="trigger_type", y="total_pnl", title="P&L by Setup Type", hover_data=["trades", "win_rate", "avg_r"])
    return polish_fig(fig, y_title="P&L ($)")


def make_daily_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Trades by Day")
    fig = px.bar(df, x="session_date", y="trades", title="Trades by Day", hover_data=["pnl", "win_rate", "avg_score"])
    return polish_fig(fig, y_title="Trades")


def polish_fig(fig: go.Figure, y_title: str = "", x_title: str = "") -> go.Figure:
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=40, r=20, t=55, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Arial", color="#0f172a"), title=dict(font=dict(size=18)))
    fig.update_xaxes(title=x_title, gridcolor="#e5e7eb")
    fig.update_yaxes(title=y_title, gridcolor="#e5e7eb")
    return fig


def table_payload(df: pd.DataFrame, money_cols=None, pct_cols=None, round_cols=None):
    money_cols = money_cols or []
    pct_cols = pct_cols or []
    round_cols = round_cols or []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
    for col in money_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: fmt_money(float(x)) if pd.notna(x) else "")
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: fmt_pct(float(x)) if pd.notna(x) else "")
    for col in round_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")
    columns = [{"name": col.replace("_", " ").title(), "id": col} for col in out.columns]
    return out.to_dict("records"), columns


if __name__ == "__main__":
    debug = os.getenv("DASH_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", "8050"))
    app.run_server(host="0.0.0.0", port=port, debug=debug)

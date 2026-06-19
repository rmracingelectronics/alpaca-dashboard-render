from __future__ import annotations

import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import REPORTS_DIR

REPORT_DIR = REPORTS_DIR


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return value.strip("_")[:120] or "report"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _write_df(path: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        pd.DataFrame().to_csv(path, index=False)
        return
    clean = df.copy()
    for col in clean.columns:
        if pd.api.types.is_datetime64_any_dtype(clean[col]):
            clean[col] = clean[col].astype(str)
    clean.to_csv(path, index=False)


def generate_report_zip(result: dict[str, Any]) -> Path:
    """Write a complete diagnostic report for upload/review.

    The ZIP intentionally includes both selected trades and raw candidates. That
    lets us decide whether entries are bad, exits are bad, or portfolio rules are
    filtering the wrong trades.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = result.get("params", {}) or {}
    metrics = result.get("metrics", {}) or {}
    mode = _safe_name(params.get("strategy_profile", "strategy"))
    start = _safe_name(result.get("start_date", "start"))
    end = _safe_name(result.get("end_date", "end"))
    risk_value = metrics.get("risk_per_trade_dollars", params.get("risk_per_trade_dollars", "risk"))
    try:
        risk_label = f"risk{float(risk_value):g}"
    except Exception:
        risk_label = _safe_name(risk_value)
    folder = REPORT_DIR / f"backtest_{mode}_{start}_{end}_{risk_label}_{ts}"
    folder.mkdir(parents=True, exist_ok=True)

    tables = {
        "selected_trades.csv": result.get("selected_trades", pd.DataFrame()),
        "raw_candidates.csv": result.get("candidates", pd.DataFrame()),
        "portfolio_trades_all.csv": result.get("portfolio_trades", pd.DataFrame()),
        "symbol_summary.csv": result.get("symbol_summary", pd.DataFrame()),
        "setup_summary.csv": result.get("setup_summary", pd.DataFrame()),
        "daily_summary.csv": result.get("daily_summary", pd.DataFrame()),
        "exit_summary.csv": result.get("exit_summary", pd.DataFrame()),
        "score_band_summary.csv": result.get("score_band_summary", pd.DataFrame()),
        "hour_summary.csv": result.get("hour_summary", pd.DataFrame()),
        "direction_summary.csv": result.get("direction_summary", pd.DataFrame()),
        "module_summary.csv": result.get("module_summary", pd.DataFrame()),
        "candle_pattern_summary.csv": result.get("candle_summary", pd.DataFrame()),
        "regime_summary.csv": result.get("regime_summary", pd.DataFrame()),
        "market_context_summary.csv": result.get("market_context_summary", pd.DataFrame()),
        "sizing_scenarios.csv": result.get("sizing_scenarios", pd.DataFrame()),
        "compact_alert_signals.csv": result.get("signals", pd.DataFrame()),
        "skipped_symbols.csv": result.get("skipped_symbols", pd.DataFrame()),
    }
    for filename, df in tables.items():
        _write_df(folder / filename, df)

    manifest = {
        "created_at": ts,
        "symbols": result.get("symbols", []),
        "start_date": result.get("start_date"),
        "end_date": result.get("end_date"),
        "feed": result.get("feed"),
        "use_news": result.get("use_news"),
        "metrics": result.get("metrics", {}),
        "diagnostics": result.get("diagnostics", {}),
        "params": result.get("params", {}),
        "notes": [
            "Upload this ZIP or selected CSV files when asking for strategy debugging.",
            "selected_trades.csv includes MFE/MAE so we can see if entries worked but exits failed.",
            "raw_candidates.csv helps diagnose whether scoring/ranking is predictive.",
            "candle_pattern_summary.csv shows whether candlestick confirmation improves or hurts the strategy.",
            "regime_summary.csv shows whether low-followthrough mode is still hurting or improving the system.",
            "market_context_summary.csv shows which V8/V12 market-condition bucket produced the trades.",
            "sizing_scenarios.csv shows what the same R-multiple sequence would have produced at $25/$50/$100/$200/$500 risk per trade.",
        ],
    }
    with open(folder / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(manifest), f, indent=2)

    zip_path = REPORT_DIR / f"{folder.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in folder.iterdir():
            zf.write(file, arcname=file.name)
    return zip_path


def export_backtest_report(result: dict[str, Any]) -> dict[str, str]:
    zip_path = generate_report_zip(result)
    latest_zip = REPORT_DIR / "latest_backtest_report.zip"
    try:
        latest_zip.write_bytes(zip_path.read_bytes())
    except Exception:
        latest_zip = zip_path
    return {
        "zip_path": str(zip_path),
        "latest_zip": str(latest_zip),
    }

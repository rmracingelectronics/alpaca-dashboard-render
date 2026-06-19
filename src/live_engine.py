from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .alpaca_rest import AlpacaDataClient
from .alpaca_trading import AlpacaTradingClient
from .backtest import _apply_v25_candlestick_filter, _v27_apply_preselection_filters, _v27_select_top_n_with_caps, _v28_calculate_risk_budget
from .config import PROJECT_ROOT, AlpacaSettings, StrategyParams
from .indicators import add_daily_features, add_intraday_features, build_qqq_context, merge_market_context
from .live_store import LiveStore, utc_now_iso
from .strategy import compute_signals
from .symbols import WATCHLISTS, parse_symbols

NY = ZoneInfo("America/New_York")
LIVE_DIR = PROJECT_ROOT / "data" / "live_trading"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
ORDER_PREFIXES = ("rmv33-", "rmv32-", "rm33-")



@dataclass
class LiveRiskConfig:
    account_value_fallback: float = 10_000.0
    position_sizing_mode: str = "fixed_dollar_risk"
    fixed_risk_dollars: float = 100.0
    base_risk_pct: float = 1.0
    min_risk_dollars: float = 10.0
    max_risk_dollars: float = 100.0
    dd1_risk_pct: float = 0.75
    dd2_risk_pct: float = 0.50
    pause_dd_pct: float = 15.0
    allow_fractional: bool = False


@dataclass
class LiveSettings:
    enabled: bool = False
    dry_run: bool = True
    allow_live_trading: bool = False
    feed: str = "iex"
    symbols: list[str] | None = None
    lookback_days: int = 75
    incremental_fetch_days: int = 3
    poll_seconds: int = 60
    max_daily_trades: int = 2
    max_open_positions: int = 2
    max_orders_per_symbol_per_day: int = 1
    max_daily_loss_dollars: float = 500.0
    force_market_open: bool = True
    enable_max_hold_exit: bool = True
    use_news_proxy: bool = True
    selection_mode: str = "seen_so_far_top_n"
    log_path: Path = LIVE_DIR / "paper_trading_log.csv"
    state_path: Path = LIVE_DIR / "paper_trading_state.json"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def load_live_settings_from_env() -> tuple[LiveSettings, LiveRiskConfig]:
    preset = os.getenv("LIVE_WATCHLIST_PRESET", "v25_playbook")
    custom_symbols = os.getenv("LIVE_SYMBOLS", "").strip()
    symbols = parse_symbols(custom_symbols, preset=preset)
    if not symbols:
        symbols = WATCHLISTS.get("v25_playbook", [])
    live = LiveSettings(
        enabled=_env_bool("PAPER_TRADING_ENABLED", False),
        dry_run=_env_bool("PAPER_TRADING_DRY_RUN", True),
        allow_live_trading=_env_bool("ALLOW_LIVE_TRADING", False),
        feed=os.getenv("ALPACA_FEED", "iex"),
        symbols=symbols,
        lookback_days=max(35, _env_int("LIVE_LOOKBACK_DAYS", 75)),
        incremental_fetch_days=max(1, _env_int("LIVE_INCREMENTAL_FETCH_DAYS", 3)),
        poll_seconds=max(30, _env_int("LIVE_POLL_SECONDS", 60)),
        max_daily_trades=max(1, _env_int("LIVE_MAX_DAILY_TRADES", 2)),
        max_open_positions=max(1, _env_int("LIVE_MAX_OPEN_POSITIONS", 2)),
        max_orders_per_symbol_per_day=max(1, _env_int("LIVE_MAX_ORDERS_PER_SYMBOL_PER_DAY", 1)),
        max_daily_loss_dollars=max(0.0, _env_float("LIVE_MAX_DAILY_LOSS_DOLLARS", 500.0)),
        force_market_open=_env_bool("LIVE_REQUIRE_MARKET_OPEN", True),
        enable_max_hold_exit=_env_bool("LIVE_ENABLE_MAX_HOLD_EXIT", True),
        use_news_proxy=_env_bool("LIVE_USE_NEWS_PROXY", True),
        selection_mode=os.getenv("LIVE_SELECTION_MODE", "seen_so_far_top_n"),
    )
    risk = LiveRiskConfig(
        account_value_fallback=_env_float("LIVE_ACCOUNT_VALUE_FALLBACK", 10000.0),
        position_sizing_mode=os.getenv("LIVE_RISK_MODE", "fixed_dollar_risk"),
        fixed_risk_dollars=_env_float("LIVE_FIXED_RISK_DOLLARS", 100.0),
        base_risk_pct=_env_float("LIVE_BASE_RISK_PCT", 1.0),
        min_risk_dollars=_env_float("LIVE_MIN_RISK_DOLLARS", 10.0),
        max_risk_dollars=_env_float("LIVE_MAX_RISK_DOLLARS", 100.0),
        dd1_risk_pct=_env_float("LIVE_DD1_RISK_PCT", 0.75),
        dd2_risk_pct=_env_float("LIVE_DD2_RISK_PCT", 0.50),
        pause_dd_pct=_env_float("LIVE_PAUSE_DD_PCT", 15.0),
        allow_fractional=_env_bool("LIVE_ALLOW_FRACTIONAL_SHARES", False),
    )
    return live, risk


def make_best_report_153601_params(risk: LiveRiskConfig) -> StrategyParams:
    """One paper-trading strategy preset; risk sizing remains separate."""
    return StrategyParams(
        strategy_profile="symbol_playbook_v25",
        direction_mode="long_short",
        initial_account_value=float(risk.account_value_fallback),
        risk_per_trade_dollars=float(risk.fixed_risk_dollars),
        requested_risk_percent=float(risk.base_risk_pct),
        risk_per_trade_pct=float(risk.base_risk_pct) / 100.0,
        position_sizing_mode=str(risk.position_sizing_mode),
        compounding_base_risk_pct=float(risk.base_risk_pct),
        compounding_min_risk_dollars=float(risk.min_risk_dollars),
        compounding_max_risk_dollars=float(risk.max_risk_dollars),
        compounding_dd1_risk_pct=float(risk.dd1_risk_pct),
        compounding_dd2_risk_pct=float(risk.dd2_risk_pct),
        compounding_pause_dd_pct=float(risk.pause_dd_pct),
        max_position_notional_pct=9999.0,
        min_candidate_score=2.0,
        max_trades_per_day=2,
        max_open_positions=2,
        max_alerts_per_symbol_per_day=1,
        daily_loss_limit_pct=100.0,
        max_consecutive_losses=99,
        slippage_bps=3.0,
        candle_pattern_mode="selective",
        enable_mean_reversion=False,
        enable_or_retest=False,
        v27_macro_filter_mode="off",
        v27_market_stress_mode="skip",
        v27_news_filter_mode="skip",
        v27_symbol_kill_switch_mode="off",
        v27_qqq_stress_abs_change_pct=4.2,
        v25_target_r=0.75,
        v25_max_hold_bars=12,
    )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _floor_5min(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return ts - timedelta(minutes=ts.minute % 5)


def latest_closed_5m_start(now: datetime | None = None) -> pd.Timestamp:
    now = now or _now_utc()
    return pd.Timestamp(_floor_5min(now) - timedelta(minutes=5)).tz_convert("UTC")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
    except Exception:
        pass
    return default


def _order_side(strategy_side: str) -> str:
    return "buy" if str(strategy_side).lower() == "long" else "sell"


def _round_price(price: float) -> float:
    return round(float(price), 2) if price >= 1 else round(float(price), 4)


def _load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _append_log(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def _merge_bar_cache(existing: pd.DataFrame | None, new: pd.DataFrame, start_keep: datetime) -> pd.DataFrame:
    frames = []
    if existing is not None and not existing.empty:
        frames.append(existing)
    if new is not None and not new.empty:
        frames.append(new)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"])
    out = out[out["timestamp"] >= pd.Timestamp(start_keep)].copy()
    out = out.drop_duplicates(["symbol", "timestamp"], keep="last")
    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


class LivePaperTradingEngine:
    def __init__(self, live: LiveSettings, risk: LiveRiskConfig):
        self.live = live
        self.risk = risk
        self.params = make_best_report_153601_params(risk)
        self.settings = AlpacaSettings()
        self.data_client = AlpacaDataClient(self.settings)
        self.trading_client = AlpacaTradingClient(self.settings)
        self.store = LiveStore()
        self.state = _load_state(self.live.state_path)
        db_state = self.store.get_state("engine_state", {})
        if isinstance(db_state, dict):
            self.state.update({k: v for k, v in db_state.items() if k not in self.state})
        self._bars_5m_cache: pd.DataFrame = pd.DataFrame()
        self._daily_cache: pd.DataFrame = pd.DataFrame()

    def validate_environment(self) -> None:
        self.store.set_state("settings", {"live": self._serializable_live(), "risk": asdict(self.risk), "alpaca": self.settings.redacted()})
        if not self.settings.is_configured:
            self.store.insert_event("worker_start_blocked", {"message": "Missing Alpaca API keys."}, status="blocked")
            raise RuntimeError("Missing Alpaca API keys. Add ALPACA_API_KEY and ALPACA_SECRET_KEY as Render environment variables.")
        base = str(self.settings.trading_base_url or "").lower()
        is_paper = "paper-api" in base
        if not is_paper and not self.live.allow_live_trading:
            self.store.insert_event("worker_start_blocked", {"message": "Trading URL is not Alpaca paper endpoint."}, status="blocked")
            raise RuntimeError("Refusing to start because ALPACA_TRADING_BASE_URL is not the paper endpoint and ALLOW_LIVE_TRADING is false.")
        if not self.live.enabled:
            self.store.insert_event("worker_start_blocked", {"message": "PAPER_TRADING_ENABLED is false."}, status="blocked")
            raise RuntimeError("PAPER_TRADING_ENABLED is false. Set it to true only when you want the worker to submit paper orders.")
        self.store.insert_event("worker_started", {"message": "Paper worker validated and started.", "dry_run": self.live.dry_run}, status="started")

    def _serializable_live(self) -> dict[str, Any]:
        data = asdict(self.live)
        data["log_path"] = str(self.live.log_path)
        data["state_path"] = str(self.live.state_path)
        return data

    def _heartbeat(self, status: str, message: str = "", extra: dict[str, Any] | None = None) -> None:
        payload = {
            "updated_at_utc": utc_now_iso(),
            "status": status,
            "message": message,
            "dry_run": self.live.dry_run,
            "enabled": self.live.enabled,
            "symbols": len(self.live.symbols or []),
            "feed": self.live.feed,
            "strategy_preset": "Best Report 153601",
        }
        if extra:
            payload.update(extra)
        self.store.set_state("heartbeat", payload)

    def _account_snapshot(self) -> dict[str, Any]:
        try:
            account = self.trading_client.get_account()
            self.store.insert_account_snapshot(account)
            return account
        except Exception as exc:
            self.store.insert_event("account_sync_error", {"message": str(exc)}, status="error")
            return {}

    def _account_equity(self, account: dict[str, Any] | None = None) -> tuple[float, float, float]:
        account = account or self._account_snapshot()
        equity = _safe_float(account.get("equity"), self.risk.account_value_fallback) if account else self.risk.account_value_fallback
        last_equity = _safe_float(account.get("last_equity"), equity) if account else equity
        high_watermark = max(equity, last_equity, self.risk.account_value_fallback)
        daily_pl = equity - last_equity
        return equity, high_watermark, daily_pl

    def _market_is_open(self) -> bool:
        if not self.live.force_market_open:
            return True
        try:
            clock = self.trading_client.get_clock()
            self.store.set_state("market_clock", clock)
            return bool(clock.get("is_open"))
        except Exception:
            ny = datetime.now(NY)
            return ny.weekday() < 5 and ((ny.hour, ny.minute) >= (9, 35)) and ((ny.hour, ny.minute) <= (15, 55))

    def _parse_strategy_client_order_id(self, client_order_id: str) -> dict[str, Any] | None:
        cid = str(client_order_id or "").strip()
        if not cid.startswith(ORDER_PREFIXES):
            return None
        # Expected V33 shape: rmv33-SYMBOL-YYYYMMDDHHMM-l/s. Keep the parser
        # tolerant so older V32/V33 test IDs do not break dashboard/state recovery.
        match = re.match(r"^(rmv\d+|rm\d+)-([A-Z0-9.]+)-(\d{12})-([ls])", cid, flags=re.IGNORECASE)
        if not match:
            return {"client_order_id": cid}
        _, symbol, stamp, side_code = match.groups()
        try:
            ts_et = datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=NY)
            session_date = ts_et.date().isoformat()
            signal_time_et = ts_et.strftime("%Y-%m-%d %H:%M")
        except Exception:
            session_date = ""
            signal_time_et = ""
        return {
            "client_order_id": cid,
            "symbol": symbol.upper(),
            "session_date": session_date,
            "signal_time_et": signal_time_et,
            "strategy_side": "long" if side_code.lower() == "l" else "short",
        }

    def _strategy_order_records_from_orders(self, orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}

        def visit(order: dict[str, Any]) -> None:
            parsed = self._parse_strategy_client_order_id(str(order.get("client_order_id", "")))
            if parsed:
                cid = parsed.get("client_order_id", "")
                if cid:
                    rec = dict(parsed)
                    rec.update({
                        "order_id": order.get("id", ""),
                        "status": order.get("status", ""),
                        "submitted_at": order.get("submitted_at") or order.get("created_at") or "",
                        "filled_at": order.get("filled_at") or "",
                    })
                    if not rec.get("symbol"):
                        rec["symbol"] = str(order.get("symbol", "")).upper()
                    records[str(cid)] = rec
            for leg in order.get("legs", []) or []:
                if isinstance(leg, dict):
                    visit(leg)

        for order in orders or []:
            if isinstance(order, dict):
                visit(order)
        return list(records.values())

    def _merge_strategy_order_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for rec in self.state.get("submitted_strategy_order_records", []) or []:
            cid = str(rec.get("client_order_id", "")) if isinstance(rec, dict) else ""
            if cid:
                merged[cid] = dict(rec)
        for rec in records or []:
            cid = str(rec.get("client_order_id", "")) if isinstance(rec, dict) else ""
            if cid:
                old = merged.get(cid, {})
                old.update({k: v for k, v in rec.items() if v not in (None, "")})
                merged[cid] = old
        out = list(merged.values())
        out.sort(key=lambda r: str(r.get("submitted_at") or r.get("signal_time_et") or ""))
        return out[-1000:]

    def _sync_account_state(self) -> list[dict[str, Any]]:
        positions: list[dict[str, Any]] = []
        try:
            positions = self.trading_client.list_positions()
            current_symbols = {str(p.get("symbol", "")).upper() for p in positions if p.get("symbol")}
            for pos in positions:
                self.store.upsert_position(pos)
            self.store.mark_missing_positions_closed(current_symbols)
        except Exception as exc:
            self.store.insert_event("position_sync_error", {"message": str(exc)}, status="error")
        try:
            after = (_now_utc() - timedelta(days=5)).isoformat()
            orders = self.trading_client.list_orders(status="all", symbols=None, limit=500, nested=True, after=after, direction="desc")
            self.store.upsert_orders_recursive(orders)
            records = self._strategy_order_records_from_orders(orders)
            if records:
                self.state["submitted_strategy_order_records"] = self._merge_strategy_order_records(records)
        except Exception as exc:
            self.store.insert_event("order_sync_error", {"message": str(exc)}, status="error")
        return positions

    def _existing_symbols(self) -> set[str]:
        symbols: set[str] = set()
        try:
            for pos in self.trading_client.list_positions():
                sym = str(pos.get("symbol", "")).upper()
                if sym:
                    symbols.add(sym)
        except Exception:
            pass
        try:
            for order in self.trading_client.list_open_orders(self.live.symbols or []):
                sym = str(order.get("symbol", "")).upper()
                if sym:
                    symbols.add(sym)
        except Exception:
            pass
        return symbols

    def _daily_order_count(self, session_date: str) -> int:
        key_count = len({str(k) for k in self.state.get("submitted_signal_keys", []) if f"|{session_date}|" in str(k)})
        order_count = len({
            str(r.get("client_order_id", ""))
            for r in self.state.get("submitted_strategy_order_records", []) or []
            if isinstance(r, dict) and str(r.get("session_date", "")) == str(session_date) and str(r.get("client_order_id", ""))
        })
        return max(key_count, order_count)

    def _symbol_daily_count(self, symbol: str, session_date: str) -> int:
        prefix = f"{symbol.upper()}|{session_date}|"
        key_count = len({str(k) for k in self.state.get("submitted_signal_keys", []) if str(k).startswith(prefix)})
        order_count = len({
            str(r.get("client_order_id", ""))
            for r in self.state.get("submitted_strategy_order_records", []) or []
            if isinstance(r, dict)
            and str(r.get("symbol", "")).upper() == symbol.upper()
            and str(r.get("session_date", "")) == str(session_date)
            and str(r.get("client_order_id", ""))
        })
        return max(key_count, order_count)

    def _update_bar_cache(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        now = _now_utc()
        end_dt = now + timedelta(minutes=5)
        keep_start = now - timedelta(days=max(35, int(self.live.lookback_days)))
        symbols = list(dict.fromkeys([s.upper() for s in (self.live.symbols or []) if s]))
        all_symbols = list(dict.fromkeys(["QQQ"] + symbols))
        if self._bars_5m_cache.empty:
            fetch_start_5m = keep_start
        else:
            latest = pd.to_datetime(self._bars_5m_cache["timestamp"], utc=True).max().to_pydatetime()
            incremental_start = now - timedelta(days=max(1, int(self.live.incremental_fetch_days)))
            fetch_start_5m = max(incremental_start, latest - timedelta(minutes=30))
        fetch_start_daily = now - timedelta(days=max(45, int(self.live.lookback_days) + 20))
        bars_5m_new = self.data_client.get_stock_bars(all_symbols, "5Min", fetch_start_5m, end_dt, feed=self.live.feed, adjustment="split", use_cache=False)
        daily_new = self.data_client.get_stock_bars(all_symbols, "1Day", fetch_start_daily, end_dt, feed=self.live.feed, adjustment="split", use_cache=False)
        self._bars_5m_cache = _merge_bar_cache(self._bars_5m_cache, bars_5m_new, keep_start)
        self._daily_cache = _merge_bar_cache(self._daily_cache, daily_new, fetch_start_daily)
        return self._bars_5m_cache.copy(), self._daily_cache.copy()

    def _add_live_v25_filter_aliases(self, alerts: pd.DataFrame) -> pd.DataFrame:
        out = alerts.copy()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        out["date"] = out["timestamp"].dt.tz_convert("America/New_York").dt.date.astype(str)
        out["score"] = pd.to_numeric(out.get("candidate_score", 0.0), errors="coerce").fillna(-9999.0)
        out["qqq_chg_open"] = pd.to_numeric(out.get("qqq_change_from_open", out.get("qqq_day_change_percent", 0.0)), errors="coerce").fillna(0.0)
        out["gap_pct"] = pd.to_numeric(out.get("gap_percent", 0.0), errors="coerce").fillna(0.0)
        out["rvol_tod"] = pd.to_numeric(out.get("rvol_time_of_day", 0.0), errors="coerce").fillna(0.0)
        for col in [
            "bullish_continuation_candle",
            "bullish_rejection_candle",
            "bullish_engulfing_candle",
            "bearish_continuation_candle",
            "bearish_rejection_candle",
            "bearish_engulfing_candle",
        ]:
            if col not in out.columns:
                out[col] = False
            out[col] = out[col].fillna(False).astype(bool)
        long_mask = out.get("side", "").astype(str).str.lower().eq("long")
        short_mask = out.get("side", "").astype(str).str.lower().eq("short")
        out["side_continuation_candle"] = False
        out["side_rejection_candle"] = False
        out["side_engulfing_candle"] = False
        out["opposing_candle_warning"] = False
        out.loc[long_mask, "side_continuation_candle"] = out.loc[long_mask, "bullish_continuation_candle"]
        out.loc[long_mask, "side_rejection_candle"] = out.loc[long_mask, "bullish_rejection_candle"]
        out.loc[long_mask, "side_engulfing_candle"] = out.loc[long_mask, "bullish_engulfing_candle"]
        out.loc[long_mask, "opposing_candle_warning"] = (
            out.loc[long_mask, "bearish_continuation_candle"] | out.loc[long_mask, "bearish_rejection_candle"] | out.loc[long_mask, "bearish_engulfing_candle"]
        )
        out.loc[short_mask, "side_continuation_candle"] = out.loc[short_mask, "bearish_continuation_candle"]
        out.loc[short_mask, "side_rejection_candle"] = out.loc[short_mask, "bearish_rejection_candle"]
        out.loc[short_mask, "side_engulfing_candle"] = out.loc[short_mask, "bearish_engulfing_candle"]
        out.loc[short_mask, "opposing_candle_warning"] = (
            out.loc[short_mask, "bullish_continuation_candle"] | out.loc[short_mask, "bullish_rejection_candle"] | out.loc[short_mask, "bullish_engulfing_candle"]
        )
        out["entry_candle_ok"] = out["side_continuation_candle"] | out["side_rejection_candle"] | out["side_engulfing_candle"]
        out["candle_pattern_score"] = (
            out["side_continuation_candle"].astype(int) + out["side_rejection_candle"].astype(int) + out["side_engulfing_candle"].astype(int) - out["opposing_candle_warning"].astype(int)
        )
        return out

    def fetch_recent_signals(self) -> pd.DataFrame:
        now = _now_utc()
        closed_ts = latest_closed_5m_start(now)
        today_ny = closed_ts.tz_convert("America/New_York").date()
        symbols = list(dict.fromkeys([s.upper() for s in (self.live.symbols or []) if s]))
        bars_5m, daily = self._update_bar_cache()
        if bars_5m.empty or daily.empty:
            return pd.DataFrame()
        bars_5m = bars_5m[pd.to_datetime(bars_5m["timestamp"], utc=True) <= closed_ts].copy()
        qqq_5m = bars_5m[bars_5m["symbol"] == "QQQ"].copy()
        qqq_daily = daily[daily["symbol"] == "QQQ"].copy()
        if qqq_5m.empty or qqq_daily.empty:
            return pd.DataFrame()
        qqq_context = build_qqq_context(qqq_5m, qqq_daily)
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            sym_5m = bars_5m[bars_5m["symbol"] == symbol].copy()
            sym_daily = daily[daily["symbol"] == symbol].copy()
            if sym_5m.empty or sym_daily.empty:
                continue
            intraday = add_intraday_features(sym_5m)
            intraday = add_daily_features(intraday, sym_daily)
            merged = merge_market_context(intraday, qqq_context)
            signals = compute_signals(merged, self.params)
            if signals.empty:
                continue
            signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True, errors="coerce")
            same_day = signals["timestamp"].dt.tz_convert("America/New_York").dt.date.eq(today_ny)
            if "buy_alert" not in signals.columns:
                continue
            alert_mask = signals["buy_alert"].fillna(False).astype(bool)
            alerts = signals[same_day & (signals["timestamp"] <= closed_ts) & alert_mask].copy()
            if alerts.empty:
                continue
            frames.append(alerts)
        if not frames:
            return pd.DataFrame()
        alerts_all = pd.concat(frames, ignore_index=True)
        alerts_all = self._add_live_v25_filter_aliases(alerts_all)
        min_score = float(getattr(self.params, "min_candidate_score", 2.0) or 2.0)
        alerts_all = alerts_all[pd.to_numeric(alerts_all["score"], errors="coerce").fillna(-9999.0) >= min_score].copy()
        alerts_all = _apply_v25_candlestick_filter(alerts_all, str(getattr(self.params, "candle_pattern_mode", "selective")))
        alerts_all, stats = _v27_apply_preselection_filters(alerts_all, self.params, use_news=bool(self.live.use_news_proxy))
        self.store.set_state("last_filter_stats", stats)
        if alerts_all.empty:
            return alerts_all
        selected_so_far = _v27_select_top_n_with_caps(alerts_all, int(getattr(self.params, "max_trades_per_day", 2) or 2), self.params)
        if selected_so_far.empty:
            return selected_so_far
        selected_so_far["timestamp"] = pd.to_datetime(selected_so_far["timestamp"], utc=True, errors="coerce")
        if str(self.live.selection_mode).lower() == "latest_bar_only":
            out = alerts_all[alerts_all["timestamp"] == closed_ts].copy()
        else:
            out = selected_so_far[selected_so_far["timestamp"] == closed_ts].copy()
        if out.empty:
            return out
        out["session_date"] = out["timestamp"].dt.tz_convert("America/New_York").dt.date.astype(str)
        return out.sort_values(["score", "timestamp"], ascending=[False, True]).reset_index(drop=True)

    def _latest_reference_prices(self, symbols: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            quotes = self.data_client.latest_quotes(symbols, feed=self.live.feed)
            if not quotes.empty:
                for _, r in quotes.iterrows():
                    sym = str(r.get("symbol", "")).upper()
                    mid = _safe_float(r.get("mid"), 0.0)
                    if sym and mid > 0:
                        out[sym] = mid
        except Exception as exc:
            self.store.insert_event("quote_sync_warning", {"message": str(exc)}, status="warning")
        return out

    def build_order_plan(self, signal: pd.Series, equity: float, high_watermark: float, reference_price: float | None = None) -> dict[str, Any] | None:
        side = str(signal.get("side", "")).lower()
        if side not in {"long", "short"}:
            return None
        signal_close = _safe_float(signal.get("close"), 0.0)
        entry_price = _safe_float(reference_price, 0.0) if reference_price else signal_close
        atr_value = _safe_float(signal.get("atr5m14"), 0.0)
        if entry_price <= 0 or atr_value <= 0:
            return None
        risk_per_share = max(entry_price * float(getattr(self.params, "v25_min_stop_pct", 0.0015)), float(getattr(self.params, "v25_stop_atr_mult", 0.60)) * atr_value)
        if risk_per_share <= 0:
            return None
        target_r = float(getattr(self.params, "v25_target_r", 0.75))
        if side == "long":
            stop_price = entry_price - risk_per_share
            target_price = entry_price + target_r * risk_per_share
        else:
            stop_price = entry_price + risk_per_share
            target_price = entry_price - target_r * risk_per_share
        risk_budget, effective_pct, dd_pct, paused = _v28_calculate_risk_budget(equity, high_watermark, self.params)
        if paused or risk_budget <= 0:
            return None
        raw_qty = risk_budget / risk_per_share
        qty = raw_qty if self.risk.allow_fractional else math.floor(raw_qty)
        if qty <= 0:
            return None
        sig_ts = pd.Timestamp(signal.get("timestamp")).tz_convert("UTC")
        ts_et = sig_ts.tz_convert("America/New_York")
        symbol = str(signal.get("symbol", "")).upper()
        client_order_id = f"rmv33-{symbol}-{ts_et.strftime('%Y%m%d%H%M')}-{side[0]}"[:48]
        max_hold_until = sig_ts + pd.Timedelta(minutes=5 * int(getattr(self.params, "v25_max_hold_bars", 12)))
        return {
            "symbol": symbol,
            "strategy_side": side,
            "alpaca_side": _order_side(side),
            "signal_time_utc": sig_ts.isoformat(),
            "signal_time_et": ts_et.strftime("%Y-%m-%d %H:%M"),
            "session_date": ts_et.date().isoformat(),
            "trigger_type": str(signal.get("trigger_type", "")),
            "candidate_score": _safe_float(signal.get("candidate_score", signal.get("score", 0.0)), 0.0),
            "entry_reference_price": _round_price(entry_price),
            "signal_close": _round_price(signal_close),
            "risk_per_share": risk_per_share,
            "risk_budget": risk_budget,
            "effective_risk_pct": effective_pct,
            "drawdown_before_trade_pct": dd_pct,
            "qty": qty,
            "target_price": _round_price(target_price),
            "stop_price": _round_price(stop_price),
            "max_hold_until_utc": max_hold_until.isoformat(),
            "client_order_id": client_order_id,
            "submitted_at_utc": utc_now_iso(),
        }

    def _daily_loss_reached(self, daily_pl: float) -> bool:
        limit = float(self.live.max_daily_loss_dollars or 0.0)
        return limit > 0 and daily_pl <= -abs(limit)

    def _save_engine_state(self) -> None:
        self.state["last_run_utc"] = utc_now_iso()
        self.state["settings"] = {"live": self._serializable_live(), "risk": asdict(self.risk)}
        self.store.set_state("engine_state", self.state)
        self.store.set_state("settings", {"live": self._serializable_live(), "risk": asdict(self.risk), "alpaca": self.settings.redacted()})
        _save_state(self.live.state_path, self.state)

    def enforce_max_hold_exits(self) -> list[dict[str, Any]]:
        if not self.live.enable_max_hold_exit:
            return []
        rows: list[dict[str, Any]] = []
        open_positions = self.store.open_positions()
        if open_positions.empty or "max_hold_until_utc" not in open_positions.columns:
            return rows
        now_ts = pd.Timestamp(_now_utc())
        submitted = set(self.state.get("max_hold_exit_keys", []))
        for _, pos in open_positions.iterrows():
            symbol = str(pos.get("symbol", "")).upper()
            deadline_raw = pos.get("max_hold_until_utc")
            if not symbol or not deadline_raw:
                continue
            deadline = pd.Timestamp(deadline_raw)
            if deadline.tzinfo is None:
                deadline = deadline.tz_localize("UTC")
            else:
                deadline = deadline.tz_convert("UTC")
            key = f"{symbol}|{deadline.isoformat()}"
            if now_ts < deadline or key in submitted:
                continue
            row = {"timestamp": utc_now_iso(), "event": "max_hold_exit_due", "symbol": symbol, "max_hold_until_utc": deadline.isoformat(), "dry_run": self.live.dry_run}
            if self.live.dry_run:
                row["alpaca_status"] = "dry_run_close_not_submitted"
            else:
                cancelled = self.trading_client.cancel_open_orders_for_symbol(symbol)
                response = self.trading_client.close_position(symbol)
                self.store.upsert_order(response)
                row["alpaca_status"] = str(response.get("status", "close_submitted"))
                row["alpaca_order_id"] = response.get("id", "")
                row["cancelled_open_orders"] = cancelled
            self.store.insert_event("max_hold_exit_due", row, symbol=symbol, status=row.get("alpaca_status"))
            submitted.add(key)
            rows.append(row)
        self.state["max_hold_exit_keys"] = list(submitted)[-1000:]
        return rows

    def run_once(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        timestamp = utc_now_iso()
        self._heartbeat("running", "Worker scan started.")
        account = self._account_snapshot()
        positions = self._sync_account_state()
        if not self._market_is_open():
            row = {"timestamp": timestamp, "event": "market_closed", "message": "No scan/order because market is closed."}
            rows.append(row)
            self.store.insert_event("market_closed", row, status="idle")
            _append_log(self.live.log_path, rows)
            self._heartbeat("idle", "Market is closed.", {"open_positions": len(positions)})
            self._save_engine_state()
            return rows
        rows.extend(self.enforce_max_hold_exits())
        existing = self._existing_symbols()
        equity, high_watermark, daily_pl = self._account_equity(account)
        if self._daily_loss_reached(daily_pl):
            row = {"timestamp": timestamp, "event": "daily_loss_limit", "message": "Daily loss limit reached; no new entries.", "daily_pl": daily_pl}
            rows.append(row)
            self.store.insert_event("daily_loss_limit", row, status="blocked")
            _append_log(self.live.log_path, rows)
            self._heartbeat("blocked", "Daily loss limit reached.", {"daily_pl": daily_pl})
            self._save_engine_state()
            return rows
        signals = self.fetch_recent_signals()
        if signals.empty:
            row = {"timestamp": timestamp, "event": "no_signal", "message": "No Best Report 153601 signal on latest eligible closed 5-minute bar."}
            rows.append(row)
            self.store.insert_event("no_signal", row, status="idle")
            _append_log(self.live.log_path, rows)
            self._heartbeat("idle", "No latest-bar signal.", {"open_positions": len(existing), "equity": equity})
            self._save_engine_state()
            return rows
        submitted = list(self.state.get("submitted_signal_keys", []))
        today = datetime.now(NY).date().isoformat()
        remaining_daily = max(0, int(self.live.max_daily_trades) - self._daily_order_count(today))
        available_slots = max(0, int(self.live.max_open_positions) - len(existing))
        capacity = min(remaining_daily, available_slots)
        if capacity <= 0:
            row = {"timestamp": timestamp, "event": "capacity_full", "message": "Daily/open-position limits reached.", "signals_seen": len(signals)}
            rows.append(row)
            self.store.insert_event("capacity_full", row, status="blocked")
            _append_log(self.live.log_path, rows)
            self._heartbeat("blocked", "Capacity full.", {"signals_seen": len(signals), "open_positions": len(existing)})
            self._save_engine_state()
            return rows
        quote_lookup = self._latest_reference_prices([str(s).upper() for s in signals["symbol"].dropna().unique().tolist()])
        taken = 0
        for _, signal in signals.iterrows():
            sym = str(signal.get("symbol", "")).upper()
            plan = self.build_order_plan(signal, equity, high_watermark, reference_price=quote_lookup.get(sym))
            if not plan:
                continue
            key = f"{plan['symbol']}|{plan['session_date']}|{plan['signal_time_et']}|{plan['strategy_side']}|{plan['trigger_type']}"
            if key in submitted:
                continue
            if plan["symbol"] in existing:
                continue
            if self._symbol_daily_count(plan["symbol"], plan["session_date"]) >= self.live.max_orders_per_symbol_per_day:
                continue
            log_row = {"timestamp": timestamp, "event": "paper_order_plan", **plan, "dry_run": self.live.dry_run}
            if self.live.dry_run:
                log_row["alpaca_status"] = "dry_run_not_submitted"
                self.store.upsert_signal_plan(log_row, status="dry_run_not_submitted")
            else:
                try:
                    result = self.trading_client.submit_market_bracket_order(
                        symbol=plan["symbol"],
                        side=plan["alpaca_side"],
                        qty=float(plan["qty"]),
                        take_profit_price=float(plan["target_price"]),
                        stop_price=float(plan["stop_price"]),
                        client_order_id=plan["client_order_id"],
                        fractional=self.risk.allow_fractional,
                    )
                    log_row["alpaca_status"] = result.status
                    log_row["alpaca_order_id"] = result.response.get("id", "")
                    self.store.upsert_order(result.response)
                    self.store.upsert_signal_plan(log_row, status=result.status)
                    existing.add(plan["symbol"])
                except Exception as exc:
                    log_row["alpaca_status"] = "submit_error"
                    log_row["message"] = str(exc)
                    self.store.upsert_signal_plan(log_row, status="submit_error")
                    self.store.insert_event("paper_order_submit_error", log_row, symbol=plan["symbol"], status="submit_error")
                    rows.append(log_row)
                    continue
            self.store.insert_event("paper_order_plan", log_row, symbol=plan["symbol"], status=log_row.get("alpaca_status"))
            submitted.append(key)
            self.state["submitted_strategy_order_records"] = self._merge_strategy_order_records([
                {
                    "client_order_id": plan["client_order_id"],
                    "symbol": plan["symbol"],
                    "session_date": plan["session_date"],
                    "signal_time_et": plan["signal_time_et"],
                    "strategy_side": plan["strategy_side"],
                    "status": log_row.get("alpaca_status", ""),
                    "submitted_at": log_row.get("submitted_at_utc", ""),
                }
            ])
            rows.append(log_row)
            taken += 1
            if taken >= capacity:
                break
        self.state["submitted_signal_keys"] = submitted[-1000:]
        if not rows:
            rows.append({"timestamp": timestamp, "event": "signals_filtered", "message": "Signals existed but were duplicate, already open, or failed sizing/capacity checks.", "signals_seen": len(signals)})
            self.store.insert_event("signals_filtered", rows[-1], status="filtered")
        try:
            self._sync_account_state()
        except Exception:
            pass
        _append_log(self.live.log_path, rows)
        self._heartbeat("running", f"Scan complete; submitted/planned {taken} order(s).", {"signals_seen": len(signals), "orders_planned": taken, "equity": equity})
        self._save_engine_state()
        return rows

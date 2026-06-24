from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyTuneConfig:
    """Live-safe filter/selection parameters for raw-replay candidate datasets.

    This config never uses future trade outcome fields for filtering. It uses only
    timestamp-available candidate features, then evaluates selected trades with
    their later realized R multiple.
    """

    name: str = "candidate_config"
    direction: str = "long_short"  # long_short, long_only, short_only
    top_trades_per_day: int = 1
    max_symbol_per_day: int = 1
    time_start: str = "10:00"
    time_end: str = "11:00"
    min_score: float = 0.0
    min_rvol: float = 1.0
    min_daily_atr_pct: float = 0.0
    min_directional_rs: float = -999.0
    max_directional_rs: float = 999.0
    min_directional_open_rs: float = -999.0
    max_directional_open_rs: float = 999.0
    min_directional_vwap_atr: float = -999.0
    max_directional_vwap_atr: float = 999.0
    max_abs_vwap_atr: float = 999.0
    max_abs_qqq_change_from_open: float = 999.0
    max_abs_gap_percent: float = 999.0
    min_risk_per_share_pct: float = 0.0
    max_risk_per_share_pct: float = 999.0
    candle_mode: str = "off"  # off, require_ok, reject_opposing, strict
    use_news_proxy: bool = False
    catalyst_filter: str = "off"  # off, skip_catalyst, require_catalyst
    catalyst_gap_abs_threshold: float = 2.5
    catalyst_rvol_threshold: float = 2.5
    include_vwap_pullback: bool = True
    include_gap_continuation: bool = True
    include_sweep_reclaim_reject: bool = True
    include_or_retest: bool = True
    include_late_trend: bool = True
    ranking_score: str = "score_rvol_vwap"  # candidate_score, score_rvol_vwap, score_expected_quality

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StrategyTuneConfig":
        valid = {f.name for f in StrategyTuneConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return StrategyTuneConfig(**{k: v for k, v in d.items() if k in valid})


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def _time_minutes(value: str) -> int:
    parts = str(value).strip().split(":")
    if len(parts) < 2:
        raise ValueError(f"Invalid HH:MM time: {value!r}")
    return int(parts[0]) * 60 + int(parts[1])


def prepare_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized, live-safe feature columns used by the tuner."""
    out = df.copy()
    if "timestamp" in out.columns:
        ts = _to_dt(out["timestamp"])
    elif "signal_time" in out.columns:
        ts = _to_dt(out["signal_time"])
    else:
        raise ValueError("Candidate dataset must contain timestamp or signal_time")
    out["_ts"] = ts
    if "session_date" in out.columns:
        out["_date"] = pd.to_datetime(out["session_date"], errors="coerce").dt.date.astype(str)
    else:
        out["_date"] = ts.dt.tz_convert("America/New_York").dt.date.astype(str)
    ts_et = ts.dt.tz_convert("America/New_York")
    out["_minute_of_day"] = ts_et.dt.hour * 60 + ts_et.dt.minute
    out["_hour_et"] = ts_et.dt.hour
    side = out.get("side", "").astype(str).str.lower()
    sign = np.where(side.eq("short"), -1.0, 1.0)
    # Normalize common numeric columns. Some package versions use qqq_change_from_open
    # while older research files used qqq_day_change_percent. Keep both live-safe
    # names aligned so dashboard/tuner QQQ settings actually affect results.
    if "candidate_score" not in out.columns and "score" in out.columns:
        out["candidate_score"] = out["score"]
    if "qqq_day_change_percent" not in out.columns and "qqq_change_from_open" in out.columns:
        out["qqq_day_change_percent"] = out["qqq_change_from_open"]
    if "qqq_change_from_open" not in out.columns and "qqq_day_change_percent" in out.columns:
        out["qqq_change_from_open"] = out["qqq_day_change_percent"]
    for c in [
        "candidate_score",
        "rvol_time_of_day",
        "daily_atr14_percent",
        "day_relative_strength",
        "open_relative_strength",
        "vwap_extension_atr",
        "qqq_day_change_percent",
        "qqq_change_from_open",
        "gap_percent",
        "entry_price",
        "risk_per_share",
        "r_multiple",
    ]:
        if c not in out.columns:
            out[c] = np.nan
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["_directional_rs"] = sign * out["day_relative_strength"].fillna(0.0)
    out["_directional_open_rs"] = sign * out["open_relative_strength"].fillna(0.0)
    out["_directional_vwap_atr"] = sign * out["vwap_extension_atr"].fillna(0.0)
    out["_abs_vwap_atr"] = out["vwap_extension_atr"].abs().fillna(0.0)
    out["_abs_qqq"] = out["qqq_day_change_percent"].abs().fillna(0.0)
    out["_abs_gap"] = out["gap_percent"].abs().fillna(0.0)
    out["_risk_per_share_pct"] = np.where(
        out["entry_price"].fillna(0) > 0,
        (out["risk_per_share"].abs() / out["entry_price"].abs()) * 100.0,
        np.nan,
    )
    out["_trigger"] = out.get("trigger_type", "").astype(str)
    out["_symbol"] = out.get("symbol", "").astype(str).str.upper()
    out["_side"] = side
    if "entry_candle_ok" not in out.columns:
        out["entry_candle_ok"] = True
    if "opposing_candle_warning_at_entry" not in out.columns and "opposing_candle_warning" in out.columns:
        out["opposing_candle_warning_at_entry"] = out["opposing_candle_warning"]
    if "opposing_candle_warning_at_entry" not in out.columns:
        out["opposing_candle_warning_at_entry"] = False
    out["entry_candle_ok"] = out["entry_candle_ok"].fillna(False).astype(bool)
    out["opposing_candle_warning_at_entry"] = out["opposing_candle_warning_at_entry"].fillna(False).astype(bool)
    out["_catalyst_proxy"] = (
        (out["_abs_gap"] >= 2.5) | (out["rvol_time_of_day"].fillna(0.0) >= 2.5)
    )
    out["_ranking_score_base"] = out["candidate_score"].fillna(0.0)
    return out


def apply_tune_config(df: pd.DataFrame, cfg: StrategyTuneConfig) -> pd.DataFrame:
    if "_ts" not in df.columns:
        df = prepare_candidate_frame(df)
    m = pd.Series(True, index=df.index)
    start_m = _time_minutes(cfg.time_start)
    end_m = _time_minutes(cfg.time_end)
    m &= df["_minute_of_day"].between(start_m, end_m, inclusive="both")
    if cfg.direction == "long_only":
        m &= df["_side"].eq("long")
    elif cfg.direction == "short_only":
        m &= df["_side"].eq("short")
    m &= df["candidate_score"].fillna(-1e9) >= cfg.min_score
    m &= df["rvol_time_of_day"].fillna(-1e9) >= cfg.min_rvol
    m &= df["daily_atr14_percent"].fillna(-1e9) >= cfg.min_daily_atr_pct
    m &= df["_directional_rs"].fillna(-1e9).between(cfg.min_directional_rs, cfg.max_directional_rs, inclusive="both")
    m &= df["_directional_open_rs"].fillna(-1e9).between(cfg.min_directional_open_rs, cfg.max_directional_open_rs, inclusive="both")
    m &= df["_directional_vwap_atr"].fillna(-1e9).between(cfg.min_directional_vwap_atr, cfg.max_directional_vwap_atr, inclusive="both")
    m &= df["_abs_vwap_atr"].fillna(1e9) <= cfg.max_abs_vwap_atr
    m &= df["_abs_qqq"].fillna(1e9) <= cfg.max_abs_qqq_change_from_open
    m &= df["_abs_gap"].fillna(1e9) <= cfg.max_abs_gap_percent
    m &= df["_risk_per_share_pct"].fillna(1e9).between(cfg.min_risk_per_share_pct, cfg.max_risk_per_share_pct, inclusive="both")

    trig = df["_trigger"].str.lower()
    setup_ok = pd.Series(False, index=df.index)
    if cfg.include_vwap_pullback:
        setup_ok |= trig.str.contains("vwap_pullback", na=False)
    if cfg.include_gap_continuation:
        setup_ok |= trig.str.contains("gap_cont", na=False)
    if cfg.include_sweep_reclaim_reject:
        setup_ok |= trig.str.contains("sweep|reclaim|reject", regex=True, na=False)
    if cfg.include_or_retest:
        setup_ok |= trig.str.contains("orb|opening|retest|10_or", regex=True, na=False)
    if cfg.include_late_trend:
        setup_ok |= trig.str.contains("late_trend|trend_follow", regex=True, na=False)
    m &= setup_ok

    if cfg.candle_mode == "require_ok":
        m &= df["entry_candle_ok"]
    elif cfg.candle_mode == "reject_opposing":
        m &= ~df["opposing_candle_warning_at_entry"]
    elif cfg.candle_mode == "strict":
        m &= df["entry_candle_ok"] & ~df["opposing_candle_warning_at_entry"]

    if cfg.use_news_proxy:
        catalyst = (df["_abs_gap"] >= cfg.catalyst_gap_abs_threshold) | (df["rvol_time_of_day"].fillna(0.0) >= cfg.catalyst_rvol_threshold)
        if cfg.catalyst_filter == "skip_catalyst":
            m &= ~catalyst
        elif cfg.catalyst_filter == "require_catalyst":
            m &= catalyst

    return df.loc[m].copy()


def add_ranking_score(df: pd.DataFrame, cfg: StrategyTuneConfig) -> pd.DataFrame:
    out = df.copy()
    score = out["candidate_score"].fillna(0.0).astype(float)
    if cfg.ranking_score == "score_rvol_vwap":
        score = score + 2.0 * np.log1p(out["rvol_time_of_day"].clip(lower=0).fillna(0.0))
        score = score + 1.5 * out["_directional_vwap_atr"].clip(lower=-2, upper=3).fillna(0.0)
        score = score - 1.0 * np.maximum(out["_abs_vwap_atr"].fillna(0.0) - 2.0, 0.0)
    elif cfg.ranking_score == "score_expected_quality":
        score = score + 2.5 * out["_directional_rs"].clip(lower=-2, upper=5).fillna(0.0)
        score = score + 2.0 * np.log1p(out["rvol_time_of_day"].clip(lower=0).fillna(0.0))
        score = score - 1.5 * out["_abs_qqq"].clip(lower=0, upper=5).fillna(0.0)
    out["_tune_rank_score"] = score
    return out


def _fmt_num(value: Any, digits: int = 3) -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "NA"
        v = float(value)
        if math.isnan(v):
            return "NA"
        return f"{v:.{digits}f}"
    except Exception:
        return str(value)


def add_decision_context(trades: pd.DataFrame, cfg: StrategyTuneConfig) -> pd.DataFrame:
    """Add human-readable trade-selection reasons and market-condition fields.

    The returned dataframe intentionally keeps every original candidate column. The
    extra columns make selected_trades.csv/trade_decision_report.csv useful for
    auditing why a trade was allowed at the exact entry timestamp.
    """
    if trades is None or trades.empty:
        return trades.copy() if trades is not None else pd.DataFrame()
    out = trades.copy()
    if "_tune_rank_score" not in out.columns:
        out = add_ranking_score(out, cfg)
    if "_date" in out.columns:
        out["selected_trade_number_for_day"] = out.groupby("_date").cumcount() + 1
    else:
        out["selected_trade_number_for_day"] = range(1, len(out) + 1)
    if "_symbol" in out.columns and "_date" in out.columns:
        out["selected_symbol_trade_number_for_day"] = out.groupby(["_date", "_symbol"]).cumcount() + 1
    else:
        out["selected_symbol_trade_number_for_day"] = 1

    cfg_items = cfg.to_dict()
    for k, v in cfg_items.items():
        out[f"decision_cfg_{k}"] = v

    def _reason(row: pd.Series) -> str:
        parts = [
            f"selected because eligible candidate passed config {cfg.name}",
            f"direction={cfg.direction}",
            f"side={row.get('_side', row.get('side', 'NA'))}",
            f"time={row.get('_minute_of_day', 'NA')} within {cfg.time_start}-{cfg.time_end}",
            f"candidate_score={_fmt_num(row.get('candidate_score'))} >= {cfg.min_score}",
            f"rvol={_fmt_num(row.get('rvol_time_of_day'))} >= {cfg.min_rvol}",
            f"daily_atr_pct={_fmt_num(row.get('daily_atr14_percent'))} >= {cfg.min_daily_atr_pct}",
            f"directional_rs={_fmt_num(row.get('_directional_rs'))} in [{cfg.min_directional_rs},{cfg.max_directional_rs}]",
            f"directional_open_rs={_fmt_num(row.get('_directional_open_rs'))} in [{cfg.min_directional_open_rs},{cfg.max_directional_open_rs}]",
            f"directional_vwap_atr={_fmt_num(row.get('_directional_vwap_atr'))} in [{cfg.min_directional_vwap_atr},{cfg.max_directional_vwap_atr}]",
            f"abs_vwap_atr={_fmt_num(row.get('_abs_vwap_atr'))} <= {cfg.max_abs_vwap_atr}",
            f"abs_qqq_change={_fmt_num(row.get('_abs_qqq'))} <= {cfg.max_abs_qqq_change_from_open}",
            f"abs_gap={_fmt_num(row.get('_abs_gap'))} <= {cfg.max_abs_gap_percent}",
            f"risk_pct={_fmt_num(row.get('_risk_per_share_pct'))} in [{cfg.min_risk_per_share_pct},{cfg.max_risk_per_share_pct}]",
            f"trigger={row.get('_trigger', row.get('trigger_type', 'NA'))}",
            f"candle_ok={row.get('entry_candle_ok', 'NA')}",
            f"opposing_candle_warning={row.get('opposing_candle_warning_at_entry', 'NA')}",
            f"catalyst_proxy={row.get('_catalyst_proxy', 'NA')} filter={cfg.catalyst_filter} use_news_proxy={cfg.use_news_proxy}",
            f"rank_score={_fmt_num(row.get('_tune_rank_score'))} using {cfg.ranking_score}",
            f"selected_trade_number_for_day={row.get('selected_trade_number_for_day', 'NA')} of max {cfg.top_trades_per_day}",
            f"selected_symbol_trade_number_for_day={row.get('selected_symbol_trade_number_for_day', 'NA')} of max {cfg.max_symbol_per_day}",
        ]
        return " | ".join(parts)

    out["decision_reason"] = out.apply(_reason, axis=1)
    out["market_conditions_at_entry"] = out.apply(
        lambda row: " | ".join([
            f"symbol={row.get('_symbol', row.get('symbol', 'NA'))}",
            f"side={row.get('_side', row.get('side', 'NA'))}",
            f"entry_time={row.get('_ts', row.get('timestamp', 'NA'))}",
            f"entry_price={_fmt_num(row.get('entry_price'))}",
            f"rvol={_fmt_num(row.get('rvol_time_of_day'))}",
            f"daily_atr_pct={_fmt_num(row.get('daily_atr14_percent'))}",
            f"gap_pct={_fmt_num(row.get('gap_percent'))}",
            f"day_rs={_fmt_num(row.get('day_relative_strength'))}",
            f"open_rs={_fmt_num(row.get('open_relative_strength'))}",
            f"vwap_ext_atr={_fmt_num(row.get('vwap_extension_atr'))}",
            f"qqq_change_open={_fmt_num(row.get('qqq_change_from_open', row.get('qqq_day_change_percent')))}",
            f"candidate_score={_fmt_num(row.get('candidate_score'))}",
            f"rank_score={_fmt_num(row.get('_tune_rank_score'))}",
            f"trigger={row.get('_trigger', row.get('trigger_type', 'NA'))}",
        ]),
        axis=1,
    )
    preferred = [
        "decision_reason", "market_conditions_at_entry", "_date", "_ts", "timestamp", "session_date",
        "_symbol", "symbol", "_side", "side", "trigger_type", "_trigger", "entry_price",
        "r_multiple", "pnl_dollars", "candidate_score", "_tune_rank_score", "selected_trade_number_for_day",
        "selected_symbol_trade_number_for_day", "_eligible_day_count", "_eligible_order_in_day",
        "_eligible_same_timestamp_rank", "rvol_time_of_day", "daily_atr14_percent", "gap_percent",
        "day_relative_strength", "open_relative_strength", "vwap_extension_atr", "_directional_rs",
        "_directional_open_rs", "_directional_vwap_atr", "_abs_vwap_atr", "qqq_change_from_open",
        "qqq_day_change_percent", "_abs_qqq", "risk_per_share", "_risk_per_share_pct", "entry_candle_ok",
        "opposing_candle_warning_at_entry", "_catalyst_proxy",
    ]
    first = [c for c in preferred if c in out.columns]
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]


def build_trade_decision_report(trades: pd.DataFrame, cfg: StrategyTuneConfig | None = None) -> pd.DataFrame:
    """Return an audit report of selected trades, preserving all indicator columns."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    if cfg is None:
        return trades.copy()
    return add_decision_context(trades, cfg)


def select_live_style(df: pd.DataFrame, cfg: StrategyTuneConfig) -> pd.DataFrame:
    """Live-safe selection: process candidates chronologically; no replacement by future candidates."""
    filt = apply_tune_config(df, cfg)
    if filt.empty:
        return filt.copy()
    filt = add_ranking_score(filt, cfg)
    filt = filt.sort_values(["_date", "_ts", "_tune_rank_score"], ascending=[True, True, False])
    filt["_eligible_day_count"] = filt.groupby("_date")["_date"].transform("size")
    filt["_eligible_order_in_day"] = filt.groupby("_date").cumcount() + 1
    filt["_eligible_same_timestamp_rank"] = filt.groupby(["_date", "_ts"]).cumcount() + 1
    selected_idx: List[int] = []
    daily_count: Dict[str, int] = {}
    symbol_day_count: Dict[Tuple[str, str], int] = {}
    for idx, row in filt.iterrows():
        d = str(row["_date"])
        if daily_count.get(d, 0) >= int(cfg.top_trades_per_day):
            continue
        sym = str(row["_symbol"])
        key = (d, sym)
        if symbol_day_count.get(key, 0) >= int(cfg.max_symbol_per_day):
            continue
        selected_idx.append(idx)
        daily_count[d] = daily_count.get(d, 0) + 1
        symbol_day_count[key] = symbol_day_count.get(key, 0) + 1
    selected = filt.loc[selected_idx].copy()
    return add_decision_context(selected, cfg)


def metrics_from_trades(trades: pd.DataFrame, fixed_risk: float = 100.0) -> Dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "trades": 0,
            "total_r": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "gross_profit_r": 0.0,
            "gross_loss_r": 0.0,
            "max_drawdown_r": 0.0,
            "pnl_dollars": 0.0,
            "trade_days": 0,
        }
    r = pd.to_numeric(trades.get("r_multiple"), errors="coerce").fillna(0.0)
    gp = float(r[r > 0].sum())
    gl = float(r[r < 0].sum())
    total = float(r.sum())
    equity = r.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    pf = gp / abs(gl) if gl < 0 else (999.0 if gp > 0 else 0.0)
    return {
        "trades": int(len(r)),
        "total_r": total,
        "win_rate": float((r > 0).mean() * 100.0) if len(r) else 0.0,
        "profit_factor": float(pf),
        "expectancy_r": float(r.mean()) if len(r) else 0.0,
        "gross_profit_r": gp,
        "gross_loss_r": gl,
        "max_drawdown_r": float(dd.min()) if len(dd) else 0.0,
        "pnl_dollars": total * fixed_risk,
        "trade_days": int(trades.get("_date", trades.get("session_date", pd.Series(dtype=str))).nunique()),
    }


def score_metrics(m: Dict[str, Any], min_trades: int = 10, target_trades_per_month: float = 2.0) -> float:
    """Robustness score. Positive total R matters, but too few trades/large drawdown are penalized."""
    trades = float(m.get("trades", 0))
    total_r = float(m.get("total_r", 0.0))
    pf = float(m.get("profit_factor", 0.0))
    exp_r = float(m.get("expectancy_r", 0.0))
    dd = abs(float(m.get("max_drawdown_r", 0.0)))
    score = total_r
    score += min(max(pf - 1.0, -1.0), 3.0) * 1.0
    score += min(max(exp_r, -1.0), 1.0) * 2.0
    score -= dd * 0.35
    if trades < min_trades:
        score -= (min_trades - trades) * 0.35
    return float(score)


def evaluate_config(df: pd.DataFrame, cfg: StrategyTuneConfig, fixed_risk: float = 100.0) -> Tuple[Dict[str, Any], pd.DataFrame]:
    selected = select_live_style(df, cfg)
    m = metrics_from_trades(selected, fixed_risk=fixed_risk)
    m["fitness_score"] = score_metrics(m)
    return m, selected


def split_by_date(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if "_date" not in df.columns:
        df = prepare_candidate_frame(df)
    date = pd.to_datetime(df["_date"], errors="coerce")
    m = pd.Series(True, index=df.index)
    if start:
        m &= date >= pd.Timestamp(start)
    if end:
        m &= date <= pd.Timestamp(end)
    return df.loc[m].copy()


def random_config(rng: random.Random, idx: int) -> StrategyTuneConfig:
    time_windows = [
        ("09:35", "10:30"), ("09:45", "10:45"), ("10:00", "11:00"),
        ("10:00", "12:00"), ("10:30", "11:30"), ("11:00", "12:30"),
        ("09:35", "12:00"),
    ]
    ts, te = rng.choice(time_windows)
    dmin_choices = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0]
    dmin = rng.choice(dmin_choices)
    dmax = rng.choice([1.5, 2.0, 3.0, 5.0, 999.0])
    if dmax < dmin:
        dmax = 999.0
    odmin = rng.choice([-1.0, -0.5, 0.0, 0.25, 0.5, 1.0])
    odmax = rng.choice([1.5, 2.0, 3.0, 5.0, 999.0])
    if odmax < odmin:
        odmax = 999.0
    vmin = rng.choice([-1.0, -0.5, 0.0, 0.25, 0.5, 0.75])
    vmax = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 999.0])
    if vmax < vmin:
        vmax = 999.0
    return StrategyTuneConfig(
        name=f"trial_{idx:05d}",
        direction=rng.choice(["long_short", "long_only", "short_only"]),
        top_trades_per_day=rng.choice([1, 1, 2, 2, 3, 5, 7, 10, 15]),
        max_symbol_per_day=rng.choice([1, 1, 2]),
        time_start=ts,
        time_end=te,
        min_score=rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40]),
        min_rvol=rng.choice([0.0, 0.8, 1.0, 1.2, 1.5, 2.0]),
        min_daily_atr_pct=rng.choice([0.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        min_directional_rs=dmin,
        max_directional_rs=dmax,
        min_directional_open_rs=odmin,
        max_directional_open_rs=odmax,
        min_directional_vwap_atr=vmin,
        max_directional_vwap_atr=vmax,
        max_abs_vwap_atr=rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 999.0]),
        max_abs_qqq_change_from_open=rng.choice([0.5, 1.0, 1.5, 2.0, 999.0]),
        max_abs_gap_percent=rng.choice([1.0, 2.0, 3.0, 5.0, 999.0]),
        min_risk_per_share_pct=rng.choice([0.0, 0.05, 0.10, 0.15]),
        max_risk_per_share_pct=rng.choice([0.75, 1.0, 1.5, 2.0, 3.0, 999.0]),
        candle_mode=rng.choice(["off", "off", "require_ok", "reject_opposing", "strict"]),
        use_news_proxy=rng.choice([False, False, True]),
        catalyst_filter=rng.choice(["off", "off", "skip_catalyst", "require_catalyst"]),
        catalyst_gap_abs_threshold=rng.choice([1.5, 2.0, 2.5, 3.0]),
        catalyst_rvol_threshold=rng.choice([1.5, 2.0, 2.5, 3.0]),
        include_vwap_pullback=rng.choice([True, True, False]),
        include_gap_continuation=rng.choice([True, False]),
        include_sweep_reclaim_reject=rng.choice([True, False]),
        include_or_retest=rng.choice([True, False]),
        include_late_trend=rng.choice([True, False]),
        ranking_score=rng.choice(["candidate_score", "score_rvol_vwap", "score_expected_quality"]),
    )


def seed_configs() -> List[StrategyTuneConfig]:
    """Include known manually found configurations and broad baselines."""
    return [
        StrategyTuneConfig(name="baseline_v359_live_hunter", time_start="10:00", time_end="11:00", min_rvol=1.0, min_daily_atr_pct=4.0, min_directional_rs=0, max_directional_rs=2.0, min_directional_vwap_atr=0.5, max_directional_vwap_atr=2.0, max_abs_vwap_atr=1.5, top_trades_per_day=1, catalyst_filter="off", use_news_proxy=False),
        StrategyTuneConfig(name="baseline_v363_robust", time_start="10:00", time_end="11:00", min_rvol=1.2, min_daily_atr_pct=4.0, min_directional_rs=0, max_directional_rs=3.0, min_directional_open_rs=0, max_directional_open_rs=5.0, min_directional_vwap_atr=0.5, max_directional_vwap_atr=2.0, max_abs_vwap_atr=1.5, top_trades_per_day=1, catalyst_filter="off", use_news_proxy=False, include_sweep_reclaim_reject=False, include_or_retest=False),
        StrategyTuneConfig(name="broad_10_12", time_start="10:00", time_end="12:00", min_rvol=1.0, min_daily_atr_pct=3.0, min_directional_rs=-0.5, max_directional_rs=3.0, min_directional_vwap_atr=0.0, max_directional_vwap_atr=2.5, max_abs_vwap_atr=2.5, top_trades_per_day=1),
        StrategyTuneConfig(name="early_momentum", time_start="09:45", time_end="10:45", min_rvol=1.5, min_daily_atr_pct=3.0, min_directional_rs=0, max_directional_rs=5.0, min_directional_vwap_atr=0.25, max_directional_vwap_atr=3.0, max_abs_vwap_atr=3.0, top_trades_per_day=1),
        StrategyTuneConfig(name="short_only_10_11", direction="short_only", time_start="10:00", time_end="11:00", min_rvol=1.0, min_daily_atr_pct=4.0, min_directional_rs=0, max_directional_rs=3.0, min_directional_vwap_atr=0.0, max_directional_vwap_atr=2.0, top_trades_per_day=1),
        StrategyTuneConfig(name="long_only_10_11", direction="long_only", time_start="10:00", time_end="11:00", min_rvol=1.0, min_daily_atr_pct=4.0, min_directional_rs=0, max_directional_rs=3.0, min_directional_vwap_atr=0.0, max_directional_vwap_atr=2.0, top_trades_per_day=1),
    ]


def generate_configs(n_random: int, seed: int = 42) -> List[StrategyTuneConfig]:
    rng = random.Random(seed)
    configs = seed_configs()
    for i in range(n_random):
        configs.append(random_config(rng, i))
    # Remove impossible configs with all setup families disabled and deduplicate by parameters except name.
    uniq: Dict[str, StrategyTuneConfig] = {}
    for c in configs:
        if not any([c.include_vwap_pullback, c.include_gap_continuation, c.include_sweep_reclaim_reject, c.include_or_retest, c.include_late_trend]):
            continue
        d = c.to_dict().copy()
        d.pop("name", None)
        key = json.dumps(d, sort_keys=True)
        if key not in uniq:
            uniq[key] = c
    return list(uniq.values())


def _row_from_metrics(prefix: str, m: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{k}": v for k, v in m.items()}


def tune_train_validate_holdout(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    validate_start: str,
    validate_end: str,
    holdout_start: Optional[str],
    holdout_end: Optional[str],
    n_random: int = 500,
    top_train_keep: int = 50,
    seed: int = 42,
    min_validation_trades: int = 5,
    min_validation_pf: float = 1.05,
    fixed_risk: float = 100.0,
) -> Dict[str, Any]:
    dfp = prepare_candidate_frame(df)
    train = split_by_date(dfp, train_start, train_end)
    val = split_by_date(dfp, validate_start, validate_end)
    hold = split_by_date(dfp, holdout_start, holdout_end) if holdout_start or holdout_end else pd.DataFrame()
    configs = generate_configs(n_random, seed)
    train_rows = []
    scored: List[Tuple[float, StrategyTuneConfig, Dict[str, Any]]] = []
    for cfg in configs:
        m, _ = evaluate_config(train, cfg, fixed_risk=fixed_risk)
        row = {"config_name": cfg.name, **cfg.to_dict(), **_row_from_metrics("train", m)}
        train_rows.append(row)
        scored.append((float(m.get("fitness_score", 0.0)), cfg, m))
    train_df = pd.DataFrame(train_rows).sort_values("train_fitness_score", ascending=False)
    top_cfgs = [x[1] for x in sorted(scored, key=lambda z: z[0], reverse=True)[:top_train_keep]]
    val_rows = []
    best_score = -1e18
    best_cfg: Optional[StrategyTuneConfig] = None
    best_bundle: Dict[str, Any] = {}
    for cfg in top_cfgs:
        mt, st = evaluate_config(train, cfg, fixed_risk=fixed_risk)
        mv, sv = evaluate_config(val, cfg, fixed_risk=fixed_risk)
        mh, sh = evaluate_config(hold, cfg, fixed_risk=fixed_risk) if not hold.empty else (metrics_from_trades(pd.DataFrame()), pd.DataFrame())
        stable_bonus = 0.0
        if mv["trades"] >= min_validation_trades and mv["profit_factor"] >= min_validation_pf and mv["total_r"] > 0:
            stable_bonus += 5.0
        validation_score = score_metrics(mv, min_trades=min_validation_trades) + stable_bonus - max(0.0, -mt["total_r"]) * 0.25
        row = {"config_name": cfg.name, **cfg.to_dict(), **_row_from_metrics("train", mt), **_row_from_metrics("validate", mv), **_row_from_metrics("holdout", mh), "selection_score": validation_score}
        val_rows.append(row)
        if validation_score > best_score:
            best_score = validation_score
            best_cfg = cfg
            best_bundle = {"train_metrics": mt, "validate_metrics": mv, "holdout_metrics": mh, "train_trades": st, "validate_trades": sv, "holdout_trades": sh}
    val_df = pd.DataFrame(val_rows).sort_values("selection_score", ascending=False)
    return {
        "train_trials": train_df,
        "validation_results": val_df,
        "best_config": best_cfg,
        "best_bundle": best_bundle,
        "periods": {
            "train_start": train_start,
            "train_end": train_end,
            "validate_start": validate_start,
            "validate_end": validate_end,
            "holdout_start": holdout_start,
            "holdout_end": holdout_end,
        },
    }


def walkforward_windows(df: pd.DataFrame, train_days: int, validate_days: int, test_days: int, lookahead_days: int = 1, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, str]]:
    if "_date" not in df.columns:
        df = prepare_candidate_frame(df)
    dates = pd.Series(pd.to_datetime(df["_date"], errors="coerce").dropna().unique()).sort_values().reset_index(drop=True)
    if start:
        dates = dates[dates >= pd.Timestamp(start)]
    if end:
        dates = dates[dates <= pd.Timestamp(end)]
    dates = dates.reset_index(drop=True)
    wins = []
    i = 0
    while i + train_days + validate_days + test_days + 2 * lookahead_days <= len(dates):
        tr_s = dates.iloc[i]
        tr_e = dates.iloc[i + train_days - 1]
        va_s = dates.iloc[i + train_days + lookahead_days]
        va_e = dates.iloc[i + train_days + lookahead_days + validate_days - 1]
        te_s = dates.iloc[i + train_days + lookahead_days + validate_days + lookahead_days]
        te_e = dates.iloc[i + train_days + lookahead_days + validate_days + lookahead_days + test_days - 1]
        wins.append({
            "train_start": str(tr_s.date()), "train_end": str(tr_e.date()),
            "validate_start": str(va_s.date()), "validate_end": str(va_e.date()),
            "test_start": str(te_s.date()), "test_end": str(te_e.date()),
        })
        i += test_days
    return wins


def run_walkforward_tuning(
    df: pd.DataFrame,
    train_days: int = 504,
    validate_days: int = 126,
    test_days: int = 63,
    lookahead_days: int = 1,
    start: Optional[str] = None,
    end: Optional[str] = None,
    n_random: int = 300,
    top_train_keep: int = 40,
    seed: int = 42,
    fixed_risk: float = 100.0,
) -> Dict[str, Any]:
    dfp = prepare_candidate_frame(df)
    wins = walkforward_windows(dfp, train_days, validate_days, test_days, lookahead_days, start, end)
    window_rows = []
    all_test_trades = []
    chosen_configs = []
    for wi, w in enumerate(wins, start=1):
        res = tune_train_validate_holdout(
            dfp,
            train_start=w["train_start"], train_end=w["train_end"],
            validate_start=w["validate_start"], validate_end=w["validate_end"],
            holdout_start=w["test_start"], holdout_end=w["test_end"],
            n_random=n_random, top_train_keep=top_train_keep,
            seed=seed + wi * 1000, fixed_risk=fixed_risk,
        )
        cfg = res["best_config"]
        bundle = res["best_bundle"]
        if cfg is None:
            continue
        test_trades = bundle["holdout_trades"].copy()
        if not test_trades.empty:
            test_trades["walkforward_window"] = wi
            test_trades["walkforward_config_name"] = cfg.name
            all_test_trades.append(test_trades)
        row = {"window": wi, **w, **cfg.to_dict(), **_row_from_metrics("train", bundle["train_metrics"]), **_row_from_metrics("validate", bundle["validate_metrics"]), **_row_from_metrics("test", bundle["holdout_metrics"])}
        window_rows.append(row)
        chosen_configs.append({"window": wi, **w, **cfg.to_dict()})
    selected = pd.concat(all_test_trades, ignore_index=True) if all_test_trades else pd.DataFrame()
    overall = metrics_from_trades(selected, fixed_risk=fixed_risk)
    windows_df = pd.DataFrame(window_rows)
    configs_df = pd.DataFrame(chosen_configs)
    if not windows_df.empty:
        positive_windows = int((pd.to_numeric(windows_df["test_total_r"], errors="coerce") > 0).sum())
        overall["windows"] = int(len(windows_df))
        overall["positive_windows"] = positive_windows
        overall["positive_window_pct"] = positive_windows / len(windows_df) * 100.0
    return {"window_summary": windows_df, "chosen_configs": configs_df, "selected_trades": selected, "overall": overall, "windows": wins}


def bootstrap_metrics(trades: pd.DataFrame, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    if trades is None or trades.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    r = pd.to_numeric(trades["r_multiple"], errors="coerce").dropna().to_numpy()
    rows = []
    for _ in range(n):
        sample = rng.choice(r, size=len(r), replace=True)
        sdf = pd.DataFrame({"r_multiple": sample})
        rows.append(metrics_from_trades(sdf))
    return pd.DataFrame(rows)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def load_config(path: str | Path) -> StrategyTuneConfig:
    return StrategyTuneConfig.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


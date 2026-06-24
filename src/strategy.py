from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from .config import StrategyParams
from .positive_context_profiles import apply_positive_context_profile_filter
from .decision_pattern_scorer import apply_decision_time_pattern_scorer



# V25 Symbol/Event Playbook derived from direct raw-bar research on the 2022-2026 local dataset.
# Live trading uses the same symbol+event+side candidate universe as the historical
# V25 replay file packaged under data/v25_research/v25_candidates_all.csv.gz.
# The backtest replay file contains the exact historical candidates; live mode
# reconstructs those same event families from current Alpaca bars and ranks them
# with the same combo-level robust score plus a deterministic live bar-quality score.
_FALLBACK_V25_PLAYBOOK_COMBOS = [
    ("AAPL", "L_prev_low_sweep_reclaim", "long", 25.2933, 7.3059),
    ("ADBE", "S_prev_high_sweep_reject", "short", 21.5370, -0.7500),
    ("AMAT", "L_late_trend_follow", "long", 24.0912, 5.6865),
    ("AMD", "S_vwap_pullback_trend", "short", 17.5825, -0.6536),
    ("BA", "L_gap_cont_controlled", "long", 32.1324, 6.7500),
    ("BA", "L_late_trend_follow", "long", 12.8547, -1.7500),
    ("COIN", "L_vwap_pullback_trend", "long", 19.7918, 10.0000),
    ("COIN", "S_vwap_pullback_trend", "short", 30.7200, 3.6852),
    ("COST", "S_RS_accel_breakdown", "short", 12.1544, -0.7500),
    ("CRM", "L_10_ORB_confirmed", "long", 29.3711, 21.2500),
    ("DIS", "S_late_trend_follow", "short", 6.8009, 1.0000),
    ("HD", "L_10_ORB_confirmed", "long", 12.1744, -6.4387),
    ("INTC", "S_late_trend_follow", "short", 10.6937, 20.0432),
    ("MA", "L_vwap_pullback_trend", "long", 47.0530, 1.8851),
    ("MU", "S_prev_high_sweep_reject", "short", 11.6534, 14.5000),
    ("MU", "S_vwap_pullback_trend", "short", 19.7052, -8.5973),
    ("PG", "S_10_ORB_confirmed", "short", 17.8728, 10.0905),
    ("PYPL", "L_early_lowATR_RS_nearVWAP", "long", 32.0015, 1.2500),
    ("PYPL", "L_gap_cont_controlled", "long", 22.3710, -4.0000),
    ("QCOM", "S_gap_cont_controlled", "short", 37.5386, 7.5000),
    ("SMCI", "L_prev_low_sweep_reclaim", "long", 23.3764, 4.5000),
    ("TSLA", "L_prev_low_sweep_reclaim", "long", 20.9123, 7.5000),
    ("V", "L_late_trend_follow", "long", 31.5877, 9.1231),
    ("WMT", "L_vwap_pullback_trend", "long", 21.9106, 16.1785),
    ("XOM", "L_10_ORB_confirmed", "long", 33.3221, 15.4264),
    ("XOM", "L_late_trend_follow", "long", 31.2831, -2.2201),
]


def _float_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    """Return a numeric Series for a column, or a full default Series.

    Raw-bar replay can omit optional feature columns for some symbols/timeframes.
    Returning a scalar default causes .astype/.fillna crashes later, so all live
    quality gates use Series defaults.
    """
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    return pd.Series(float(default), index=df.index, dtype="float64")


def _apply_realtime_quality_gate(df: pd.DataFrame, params: StrategyParams, prefix: str, output_col: str) -> pd.DataFrame:
    """Apply a live-safe quality gate using only signal-bar values.

    prefix maps to params such as v358_min_rvol or v359_min_rvol. Directional
    values normalize long/short logic: for shorts, relative weakness and below-
    VWAP extension are converted into positive trade-direction values.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if "timestamp_ny" in out.columns:
        try:
            time_str = pd.to_datetime(out["timestamp_ny"], errors="coerce").dt.strftime("%H:%M")
        except Exception:
            time_str = out["timestamp_ny"].astype(str).str.slice(11, 16)
    elif "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
        time_str = ts.dt.strftime("%H:%M")
    else:
        time_str = pd.Series("", index=out.index)

    start_time = str(getattr(params, f"{prefix}_quality_start_time", "10:00") or "10:00")
    end_time = str(getattr(params, f"{prefix}_quality_end_time", "11:00") or "11:00")
    side_short = out["side"].astype(str).str.lower().eq("short") if "side" in out.columns else pd.Series(False, index=out.index)

    rs = _float_series(out, "day_relative_strength", 0.0)
    ors = _float_series(out, "open_relative_strength", 0.0)
    vwap = _float_series(out, "vwap_extension_atr", 0.0)
    rvol = _float_series(out, "rvol_time_of_day", 0.0)
    daily_atr = _float_series(out, "daily_atr14_percent", 0.0)
    atr5 = _float_series(out, "atr5m14", 0.0)
    close = _float_series(out, "close", 0.0).replace(0, np.nan)

    dir_rs = pd.Series(np.where(side_short, -rs, rs), index=out.index, dtype="float64")
    dir_ors = pd.Series(np.where(side_short, -ors, ors), index=out.index, dtype="float64")
    dir_vwap = pd.Series(np.where(side_short, -vwap, vwap), index=out.index, dtype="float64")
    abs_vwap = vwap.abs()

    # Approximate V25 stop size at the signal bar.  The exact live entry is the
    # next bar, but this approximation is causal and catches unrealistic oversized
    # ATR-risk setups before ranking.  Defaults are wide for older gates.
    min_stop_pct_default = float(getattr(params, "v25_min_stop_pct", 0.0015) or 0.0015) * 100.0
    stop_atr_mult = float(getattr(params, "v25_stop_atr_mult", 0.60) or 0.60)
    approx_signal_risk_pct = pd.Series(np.maximum(min_stop_pct_default, (stop_atr_mult * atr5 / close * 100.0)), index=out.index).replace([np.inf, -np.inf], np.nan).fillna(999.0)
    min_signal_risk_pct = float(getattr(params, f"{prefix}_min_signal_risk_pct", 0.0) or 0.0)
    max_signal_risk_pct = float(getattr(params, f"{prefix}_max_signal_risk_pct", 999.0) or 999.0)

    qmask = (
        (time_str >= start_time)
        & (time_str <= end_time)
        & (rvol >= float(getattr(params, f"{prefix}_min_rvol", 1.0) or 1.0))
        & (daily_atr >= float(getattr(params, f"{prefix}_min_daily_atr_pct", 0.0) or 0.0))
        & (dir_rs >= float(getattr(params, f"{prefix}_min_directional_rs", -999.0)))
        & (dir_rs <= float(getattr(params, f"{prefix}_max_directional_rs", 999.0)))
        & (dir_ors >= float(getattr(params, f"{prefix}_min_directional_open_rs", -999.0)))
        & (dir_ors <= float(getattr(params, f"{prefix}_max_directional_open_rs", 999.0)))
        & (dir_vwap >= float(getattr(params, f"{prefix}_min_directional_vwap_extension_atr", -999.0)))
        & (dir_vwap <= float(getattr(params, f"{prefix}_max_directional_vwap_extension_atr", 999.0)))
        & (abs_vwap <= float(getattr(params, f"{prefix}_max_abs_vwap_extension_atr", 999.0) or 999.0))
        & (approx_signal_risk_pct >= min_signal_risk_pct)
        & (approx_signal_risk_pct <= max_signal_risk_pct)
    )
    qmask = pd.Series(qmask, index=out.index).fillna(False).astype(bool)
    out[output_col] = qmask
    out[f"{prefix}_directional_rs"] = dir_rs
    out[f"{prefix}_directional_open_rs"] = dir_ors
    out[f"{prefix}_directional_vwap_atr"] = dir_vwap
    out[f"{prefix}_approx_signal_risk_pct"] = approx_signal_risk_pct

    # Optional live-safe score shaping.  This is not a future-looking optimizer;
    # it simply ranks current candidates by live quality fields so the walk-forward
    # selector has a better priority order when several candidates are seen.
    wr = float(getattr(params, f"{prefix}_score_weight_rvol", 0.0) or 0.0)
    wdrs = float(getattr(params, f"{prefix}_score_weight_directional_rs", 0.0) or 0.0)
    wdv = float(getattr(params, f"{prefix}_score_weight_directional_vwap", 0.0) or 0.0)
    wav = float(getattr(params, f"{prefix}_score_penalty_abs_vwap", 0.0) or 0.0)
    if "candidate_score" in out.columns and any(abs(x) > 1e-12 for x in [wr, wdrs, wdv, wav]):
        base_score = pd.to_numeric(out["candidate_score"], errors="coerce").fillna(0.0)
        quality_adjustment = wr * np.log1p(rvol.clip(lower=0.0)) + wdrs * dir_rs + wdv * dir_vwap - wav * abs_vwap
        out[f"{prefix}_base_candidate_score"] = base_score
        out[f"{prefix}_quality_score_adjustment"] = quality_adjustment
        out["candidate_score"] = base_score + quality_adjustment
    if "buy_alert" in out.columns:
        out["buy_alert"] = out["buy_alert"].fillna(False).astype(bool) & qmask
    return out


def _load_v25_playbook_combos() -> list[tuple[str, str, str, float, float]]:
    path = Path(__file__).resolve().parents[1] / "data" / "v25_research" / "v25_candidates_all.csv.gz"
    if not path.exists():
        return _FALLBACK_V25_PLAYBOOK_COMBOS
    try:
        cols = ["symbol", "event", "side", "robust_score", "r075"]
        df = pd.read_csv(path, usecols=cols)
        if df.empty:
            return _FALLBACK_V25_PLAYBOOK_COMBOS
        grouped = (
            df.groupby(["symbol", "event", "side"], dropna=False)
            .agg(robust_score=("robust_score", "first"), gross_r=("r075", "sum"))
            .reset_index()
        )
        combos: list[tuple[str, str, str, float, float]] = []
        for _, row in grouped.iterrows():
            combos.append(
                (
                    str(row["symbol"]).upper(),
                    str(row["event"]),
                    str(row["side"]).lower(),
                    float(row["robust_score"]) * 100.0,
                    float(row["gross_r"]),
                )
            )
        return combos or _FALLBACK_V25_PLAYBOOK_COMBOS
    except Exception:
        return _FALLBACK_V25_PLAYBOOK_COMBOS


EXCLUDED_PRESET_SYMBOLS = {"PYPL", "V", "DIS", "WMT", "HD"}
V25_PLAYBOOK_COMBOS = [c for c in _load_v25_playbook_combos() if c[0] not in EXCLUDED_PRESET_SYMBOLS]
V25_COMBO_SCORE = {(sym, event, side): (score, gross_r) for sym, event, side, score, gross_r in V25_PLAYBOOK_COMBOS}
V25_ALLOWED = set(V25_COMBO_SCORE.keys())
V25_SYMBOLS = {sym for sym, _, _, _, _ in V25_PLAYBOOK_COMBOS}


def _series_false_like(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index)


def _add_v25_developing_profile(g: pd.DataFrame, bins: int = 48) -> pd.DataFrame:
    """Approximate developing volume profile POC/VAH/VAL per session.

    This is intentionally lightweight and deterministic for dashboard backtests. It uses
    regular-session bars up to the current signal bar to build a cumulative volume profile.
    """
    out_parts = []
    for _, day in g.groupby("session_date", sort=False):
        d = day.sort_values("timestamp").copy()
        typ = ((d["high"].astype(float) + d["low"].astype(float) + d["close"].astype(float)) / 3.0).to_numpy()
        vol = d["volume"].fillna(0).astype(float).to_numpy()
        lo = float(np.nanmin(d["low"].astype(float).to_numpy())) if len(d) else np.nan
        hi = float(np.nanmax(d["high"].astype(float).to_numpy())) if len(d) else np.nan
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            d["v25_poc"] = np.nan; d["v25_vah"] = np.nan; d["v25_val"] = np.nan
            out_parts.append(d); continue
        edges = np.linspace(lo, hi, bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        cum = np.zeros(bins, dtype=float)
        poc_vals = []; vah_vals = []; val_vals = []
        for price, v in zip(typ, vol):
            if np.isfinite(price):
                b = int(np.searchsorted(edges, price, side="right") - 1)
                b = max(0, min(bins - 1, b))
                cum[b] += max(0.0, float(v) if np.isfinite(v) else 0.0)
            total = cum.sum()
            if total <= 0:
                poc_vals.append(np.nan); vah_vals.append(np.nan); val_vals.append(np.nan); continue
            poc_i = int(np.argmax(cum))
            target = total * 0.70
            left = right = poc_i
            included = cum[poc_i]
            # Expand around POC by adding the side with more next-bin volume.
            while included < target and (left > 0 or right < bins - 1):
                lv = cum[left - 1] if left > 0 else -1
                rv = cum[right + 1] if right < bins - 1 else -1
                if rv >= lv and right < bins - 1:
                    right += 1; included += cum[right]
                elif left > 0:
                    left -= 1; included += cum[left]
                else:
                    break
            poc_vals.append(float(centers[poc_i])); val_vals.append(float(centers[left])); vah_vals.append(float(centers[right]))
        d["v25_poc"] = poc_vals; d["v25_vah"] = vah_vals; d["v25_val"] = val_vals
        out_parts.append(d)
    return pd.concat(out_parts, ignore_index=True) if out_parts else g




def _v25_live_bar_quality_score(g: pd.DataFrame, event_side: str) -> pd.Series:
    """Approximate the V25 replay fallback_score from live 5-minute bars.

    The historical replay file uses a combo-level robust score plus a bar-quality
    fallback score. In live/raw-bar replay mode some optional candle/context
    columns may be absent depending on the selected session/feed.  Always coerce
    missing columns to full-length Series so pandas scalar defaults never trigger
    errors such as: ``float object has no attribute fillna``.
    """

    def _num_series(col: str, default: float = 0.0) -> pd.Series:
        if col in g.columns:
            src = g[col]
        else:
            src = pd.Series(default, index=g.index)
        return pd.to_numeric(src, errors="coerce").fillna(default)

    def _bool_series(col: str, default: bool = False) -> pd.Series:
        if col in g.columns:
            src = g[col]
        else:
            src = pd.Series(default, index=g.index)
        return src.fillna(default).astype(bool)

    rvol = _num_series("rvol_time_of_day", 0.0).clip(lower=0, upper=6)
    close_pos_raw = _num_series("candle_close_position", 0.5).clip(0, 1)
    side_close = close_pos_raw if event_side == "long" else (1.0 - close_pos_raw)
    rs = _num_series("open_relative_strength", 0.0)
    chg = _num_series("stock_change_from_open", 0.0)
    vwap_ext = _num_series("vwap_extension_atr", 0.0)
    side_rs = rs if event_side == "long" else -rs
    side_chg = chg if event_side == "long" else -chg
    side_vwap = vwap_ext if event_side == "long" else -vwap_ext

    if event_side == "long":
        candle_bonus = (
            _bool_series("bullish_continuation_candle").astype(float) * 0.35
            + _bool_series("bullish_rejection_candle").astype(float) * 0.45
            + _bool_series("bullish_engulfing_candle").astype(float) * 0.55
            - _bool_series("bearish_continuation_candle").astype(float) * 0.60
        )
    else:
        candle_bonus = (
            _bool_series("bearish_continuation_candle").astype(float) * 0.35
            + _bool_series("bearish_rejection_candle").astype(float) * 0.45
            + _bool_series("bearish_engulfing_candle").astype(float) * 0.55
            - _bool_series("bullish_continuation_candle").astype(float) * 0.60
        )
    quality = (
        0.25
        + np.log1p(rvol) * 2.15
        + side_close * 1.20
        + side_rs.clip(lower=-1.0, upper=5.0) * 0.28
        + side_chg.clip(lower=-1.0, upper=5.0) * 0.22
        + side_vwap.clip(lower=-0.5, upper=3.0) * 0.18
        + candle_bonus
    )
    return pd.to_numeric(quality, errors="coerce").fillna(0.0).clip(lower=0.0, upper=7.0)


def _compute_v25_playbook_signals(out: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    if "liquidity_filter" not in out.columns:
        out["liquidity_filter"] = (
            (out["close"] >= params.min_price)
            & (out["avg_20d_dollar_volume"] >= params.min_avg_20d_dollar_volume)
            & (out["current_5m_dollar_volume"] >= params.min_current_5m_dollar_volume)
            & (out["daily_atr14_percent"] >= params.min_daily_atr_pct)
            & (out["daily_atr14_percent"] <= params.max_daily_atr_pct)
            & (out["atr5m14"] > 0)
        )
    frames = []
    generic_symbols_enabled = bool(getattr(params, "v25_allow_generic_symbols", False))
    generic_event_base_scores = {
        "L_vwap_pullback_trend": 22.0,
        "S_vwap_pullback_trend": 22.0,
        "L_10_ORB_confirmed": 26.0,
        "S_10_ORB_confirmed": 26.0,
        "L_prev_low_sweep_reclaim": 24.0,
        "S_prev_high_sweep_reject": 24.0,
        "L_late_trend_follow": 20.0,
        "S_late_trend_follow": 20.0,
        "L_RS_accel_breakout": 18.0,
        "S_RS_accel_breakdown": 18.0,
        "L_gap_cont_controlled": 24.0,
        "S_gap_cont_controlled": 24.0,
        "L_early_lowATR_RS_nearVWAP": 20.0,
    }
    for symbol, g0 in out.groupby("symbol", sort=False):
        g = g0.copy().sort_values("timestamp").reset_index(drop=True)
        if symbol not in V25_SYMBOLS and not generic_symbols_enabled:
            g["buy_alert"] = False
            frames.append(g); continue
        g = _add_v25_developing_profile(g, bins=int(getattr(params, "v25_profile_bins", 48)))
        prev_high = g["high"].shift(1); prev_low = g["low"].shift(1); prev_close = g["close"].shift(1)
        recent_low = g["low"].rolling(4, min_periods=1).min()
        recent_high = g["high"].rolling(4, min_periods=1).max()
        rs_accel = g["open_relative_strength"].diff(3)
        atr = g["atr5m14"].replace(0, np.nan)
        vol_ok = g["rvol_time_of_day"].fillna(0) >= float(getattr(params, "v25_min_rvol", 0.75))
        bull = (g["close"] > g["open"]) & (g["candle_close_position"] >= 0.58)
        bear = (g["close"] < g["open"]) & (g["candle_close_position"] <= 0.42)
        trend_long = (g["close"] > g["session_vwap"]) & (g["ema9"] >= g["ema20"])
        trend_short = (g["close"] < g["session_vwap"]) & (g["ema9"] <= g["ema20"])
        near_vwap_long = recent_low <= (np.minimum(g["session_vwap"], g["ema20"]) + 0.30 * atr)
        near_vwap_short = recent_high >= (np.maximum(g["session_vwap"], g["ema20"]) - 0.30 * atr)

        # Developing volume profile reaction. This was the strongest structure filter in the raw-data tests.
        tol = float(getattr(params, "v25_profile_tolerance_atr", 0.18)) * atr
        touch_poc_long = (g["low"] <= g["v25_poc"] + tol) & (g["close"] >= g["v25_poc"])
        touch_val_long = (g["low"] <= g["v25_val"] + tol) & (g["close"] >= g["v25_val"])
        touch_poc_short = (g["high"] >= g["v25_poc"] - tol) & (g["close"] <= g["v25_poc"])
        touch_vah_short = (g["high"] >= g["v25_vah"] - tol) & (g["close"] <= g["v25_vah"])
        vp_long = (touch_poc_long | touch_val_long) & (g["candle_close_position"] >= 0.52)
        vp_short = (touch_poc_short | touch_vah_short) & (g["candle_close_position"] <= 0.48)

        events: dict[str, pd.Series] = {}
        events["L_vwap_pullback_trend"] = trend_long & near_vwap_long & (g["close"] > prev_high) & bull & vol_ok & (g["vwap_extension_atr"] <= 1.55)
        events["S_vwap_pullback_trend"] = trend_short & near_vwap_short & (g["close"] < prev_low) & bear & vol_ok & (g["vwap_extension_atr"] >= -1.55)
        events["L_10_ORB_confirmed"] = (g["time_str"].between("10:00", "10:59")) & (g["close"] > g["opening_30_high"] + 0.10 * atr) & trend_long & bull & vol_ok
        events["S_10_ORB_confirmed"] = (g["time_str"].between("10:00", "10:59")) & (g["close"] < g["opening_30_low"] - 0.10 * atr) & trend_short & bear & vol_ok
        events["L_prev_low_sweep_reclaim"] = ((g["low"] < g["prev_day_low"]) | (g["low"] < g["intraday_low_so_far"].shift(1))) & (g["close"] > np.minimum(g["prev_day_low"], g["intraday_low_so_far"].shift(1))) & bull
        events["S_prev_high_sweep_reject"] = ((g["high"] > g["prev_day_high"]) | (g["high"] > g["intraday_high_so_far"].shift(1))) & (g["close"] < np.maximum(g["prev_day_high"], g["intraday_high_so_far"].shift(1))) & bear
        events["L_late_trend_follow"] = (g["time_str"].between("11:00", "13:30")) & trend_long & bull & (g["day_relative_strength"] >= 0.35) & (g["vwap_extension_atr"].between(0, 2.20)) & vol_ok
        events["S_late_trend_follow"] = (g["time_str"].between("11:00", "13:30")) & trend_short & bear & (g["day_relative_strength"] <= -0.35) & (g["vwap_extension_atr"].between(-2.20, 0)) & vol_ok
        events["L_RS_accel_breakout"] = (rs_accel >= 0.30) & (g["close"] > prev_high) & trend_long & bull & vol_ok
        events["S_RS_accel_breakdown"] = (rs_accel <= -0.30) & (g["close"] < prev_low) & trend_short & bear & vol_ok
        events["L_gap_cont_controlled"] = (g["gap_percent"] >= 1.50) & (g["stock_change_from_open"] >= 1.0) & trend_long & bull & (g["vwap_extension_atr"].between(0, 4.0))
        events["S_gap_cont_controlled"] = (g["gap_percent"] <= -1.50) & (g["stock_change_from_open"] <= -1.0) & trend_short & bear & (g["vwap_extension_atr"].between(-4.0, 0))
        events["L_early_lowATR_RS_nearVWAP"] = (g["time_str"].between("09:40", "09:55")) & (g["daily_atr14_percent"] < 2.5) & (g["day_relative_strength"] >= 0.30) & (g["vwap_extension_atr"].between(-1.0, 0.85)) & bull

        # Make the dashboard setup toggles active for the V25 raw-bar path.
        # The old raw replay path ignored these controls because V25 returned early
        # from compute_signals().  These switches are intentionally live-safe and
        # operate only on event families visible at the signal candle.
        if not bool(getattr(params, "enable_mean_reversion", False)):
            events["L_prev_low_sweep_reclaim"] = pd.Series(False, index=g.index)
            events["S_prev_high_sweep_reject"] = pd.Series(False, index=g.index)
        if not bool(getattr(params, "enable_or_retest", False)):
            events["L_10_ORB_confirmed"] = pd.Series(False, index=g.index)
            events["S_10_ORB_confirmed"] = pd.Series(False, index=g.index)

        g["buy_alert"] = False
        g["side"] = ""
        g["trigger_type"] = ""
        g["setup_family"] = "v25_symbol_playbook"
        g["opportunity_module"] = "v25_symbol_playbook"
        g["candidate_score"] = np.nan
        g["fallback_score"] = np.nan
        g["quality"] = ""
        g["v25_base_score"] = np.nan
        g["v25_live_bar_quality_score"] = np.nan
        g["v25_historical_gross_r"] = np.nan
        g["v25_profile_filter"] = False
        g["entry_candle_ok"] = False
        g["opposing_candle_warning"] = False
        g["supporting_score"] = 0
        g["trigger_level"] = g["close"]

        dir_mode = str(getattr(params, "direction_mode", "long_short")).lower()
        allow_long = dir_mode in {"long_only", "long_short"}
        allow_short = dir_mode in {"short_only", "long_short"}
        for event, mask in events.items():
            event_side = "long" if event.startswith("L_") else "short"
            if event_side == "long" and not allow_long:
                continue
            if event_side == "short" and not allow_short:
                continue
            key = (symbol, event, event_side)
            combo_allowed = key in V25_ALLOWED
            if not combo_allowed and not generic_symbols_enabled:
                continue
            vp_mask = vp_long if event_side == "long" else vp_short
            m = mask.fillna(False) & vp_mask.fillna(False) & g["liquidity_filter"].fillna(False)
            # If two events hit on the same bar, keep the higher live score.
            if combo_allowed:
                base_score, gross_r = V25_COMBO_SCORE[key]
            else:
                # Custom-symbol exploratory mode. These symbols do not have a
                # researched V25 symbol+event pocket, so use the same event
                # structure and live bar-quality score but mark the historical
                # edge as unknown. This makes custom watchlists backtest/fetch
                # correctly without changing the Best Report 153601 preset path.
                base_score = float(generic_event_base_scores.get(event, 18.0))
                gross_r = np.nan
            fallback_score = _v25_live_bar_quality_score(g, event_side)
            live_score = base_score + fallback_score
            better = m & ((~g["buy_alert"]) | (g["candidate_score"].fillna(-999) < live_score))
            g.loc[better, "buy_alert"] = True
            g.loc[better, "side"] = event_side
            g.loc[better, "trigger_type"] = "v25_" + event
            g.loc[better, "candidate_score"] = live_score[better]
            g.loc[better, "fallback_score"] = fallback_score[better]
            g.loc[better, "quality"] = "v25_profile_reaction" if combo_allowed else "v25_generic_custom_symbol_research"
            g.loc[better, "v25_base_score"] = base_score
            g.loc[better, "v25_live_bar_quality_score"] = fallback_score[better]
            g.loc[better, "v25_historical_gross_r"] = gross_r
            g.loc[better, "v25_profile_filter"] = True
            g.loc[better, "supporting_score"] = 1

        # Side-aware candle columns used by the dashboard Candlestick mode.
        # V36.2 fix: previous V25 raw replay marked every V25 alert as
        # entry_candle_ok=True, so changing the Candlestick dropdown did not change
        # results.  These fields are now derived from the actual signal candle.
        side_short_final = g["side"].astype(str).str.lower().eq("short")
        g["side_continuation_candle"] = np.where(side_short_final, g.get("bearish_continuation_candle", False), g.get("bullish_continuation_candle", False))
        g["side_rejection_candle"] = np.where(side_short_final, g.get("bearish_rejection_candle", False), g.get("bullish_rejection_candle", False))
        g["side_engulfing_candle"] = np.where(side_short_final, g.get("bearish_engulfing_candle", False), g.get("bullish_engulfing_candle", False))
        g["entry_candle_ok"] = np.where(side_short_final, g.get("short_entry_candle_ok", False), g.get("long_entry_candle_ok", False))
        g["opposing_candle_warning"] = np.where(side_short_final, g.get("short_exit_warning_candle", False), g.get("long_exit_warning_candle", False))
        frames.append(g)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # V35.8/V35.9 live/raw-bar quality gates for the V25 playbook path.
    if not result.empty and bool(getattr(params, "enable_v358_live_quality_filter", False)):
        result = _apply_realtime_quality_gate(result, params, "v358", "v358_live_quality_ok")
    if not result.empty and bool(getattr(params, "enable_v359_live_hunter_filter", False)):
        result = _apply_realtime_quality_gate(result, params, "v359", "v359_live_hunter_ok")
    if not result.empty and bool(getattr(params, "enable_v364_professional_momentum_filter", False)):
        result = _apply_realtime_quality_gate(result, params, "v364", "v364_professional_momentum_ok")
    if not result.empty and bool(getattr(params, "enable_v377_positive_context_filter", False)):
        result = apply_positive_context_profile_filter(result, params)
    if not result.empty and bool(getattr(params, "enable_v379_decision_pattern_filter", False)):
        result = apply_decision_time_pattern_scorer(result, params)
    return result

def _to_bool_int(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool).astype(int)


def compute_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Create candidate alerts.

    V4 is not a single all-or-nothing signal. It classifies market context, then
    creates several explicit setup types. The dashboard/report then tells us which
    setup types deserve to stay.
    """
    out = df.copy().sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume", "atr5m14"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["stock_day_change_percent"] = (out["close"] - out["prev_close"]) / out["prev_close"] * 100
    out["stock_change_from_open"] = (out["close"] - out["session_open"]) / out["session_open"] * 100
    out["gap_percent"] = (out["session_open"] - out["prev_close"]) / out["prev_close"] * 100
    out["day_relative_strength"] = out["stock_day_change_percent"] - out["qqq_day_change_percent"]
    out["open_relative_strength"] = out["stock_change_from_open"] - out["qqq_change_from_open"]
    out["vwap_extension_atr"] = (out["close"] - out["session_vwap"]) / out["atr5m14"].replace(0, np.nan)
    out["candle_range_atr"] = out["candle_range"] / out["atr5m14"].replace(0, np.nan)
    if "news_count_last_3d" not in out.columns:
        out["news_count_last_3d"] = 0

    if str(getattr(params, "strategy_profile", "")).lower() == "symbol_playbook_v25":
        return _compute_v25_playbook_signals(out, params)

    return _compute_adaptive_v4_signals(out, params)


def _compute_adaptive_v4_signals(out: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    primary_window = (out["time_str"] >= params.primary_start) & (out["time_str"] <= params.primary_end)
    midday_window = (out["time_str"] > params.midday_start) & (out["time_str"] < params.midday_end)
    afternoon_window = (out["time_str"] >= params.afternoon_start) & (out["time_str"] <= params.afternoon_end)
    # V12 default is morning-core only. The long reports showed 09:xx, 13:xx, and 15:xx
    # were persistent drag, while 10:00-11:30 contained the robust edge.
    if bool(getattr(params, "v12_morning_only", False)):
        out["time_filter"] = primary_window
    else:
        out["time_filter"] = primary_window | afternoon_window

    out["liquidity_filter"] = (
        (out["close"] >= params.min_price)
        & (out["avg_20d_dollar_volume"] >= params.min_avg_20d_dollar_volume)
        & (out["current_5m_dollar_volume"] >= params.min_current_5m_dollar_volume)
        & (out["daily_atr14_percent"] >= params.min_daily_atr_pct)
        & (out["daily_atr14_percent"] <= params.max_daily_atr_pct)
        & (out["atr5m14"] > 0)
    )

    qqq_bull = (
        (out["qqq_close"] > out["qqq_session_vwap"])
        & (out["qqq_ema9"] >= out["qqq_ema20"])
        & (out["qqq_15min_change_percent"].fillna(0) >= params.qqq_15min_change_min_pct)
    )
    qqq_bear = (
        (out["qqq_close"] < out["qqq_session_vwap"])
        & (out["qqq_ema9"] <= out["qqq_ema20"])
        & (out["qqq_15min_change_percent"].fillna(0) <= abs(params.qqq_15min_change_min_pct))
    )
    qqq_not_panic = (
        (out["qqq_daily_atr14_percent"] <= params.qqq_atr14_daily_percent_max)
        & (out["qqq_day_change_percent"] >= params.qqq_max_intraday_loss_pct)
    )
    out["market_bull"] = qqq_bull & qqq_not_panic
    out["market_bear"] = qqq_bear & (out["qqq_daily_atr14_percent"] <= params.qqq_atr14_daily_percent_max)
    out["market_neutral"] = (~out["market_bull"]) & (~out["market_bear"]) & qqq_not_panic
    out["market_not_broken"] = out["market_bull"] | out["market_neutral"] | out["market_bear"]

    out["reason_for_move"] = (
        (out["gap_percent"].abs() >= params.min_gap_percent)
        | (out["news_count_last_3d"].fillna(0) >= 1)
        | (out["rvol_time_of_day"] >= params.min_rvol_reason)
        | (out["day_relative_strength"].abs() >= params.min_day_relative_strength)
        | (out["open_relative_strength"].abs() >= params.min_open_relative_strength)
    )

    frames = []
    for _, g0 in out.groupby("symbol", sort=False):
        g = g0.copy().sort_values("timestamp")
        prev_high = g["high"].shift(1)
        prev_low = g["low"].shift(1)
        prev_close = g["close"].shift(1)
        prior_high = g.groupby("session_date")["high"].cummax().shift(1)
        prior_low = g.groupby("session_date")["low"].cummin().shift(1)
        new_session = g["session_date"] != g["session_date"].shift(1)
        prior_high = prior_high.mask(new_session)
        prior_low = prior_low.mask(new_session)
        g["prior_intraday_high"] = prior_high
        g["prior_intraday_low"] = prior_low

        recent_low = g["low"].rolling(params.pullback_lookback_bars, min_periods=1).min()
        recent_high = g["high"].rolling(params.pullback_lookback_bars, min_periods=1).max()
        recent_vol = g["volume"].rolling(3, min_periods=1).mean()
        g["recent_low"] = recent_low
        g["recent_high"] = recent_high

        run_range_long = (prior_high - g["intraday_low_so_far"]).replace(0, np.nan)
        retrace_long = (prior_high - recent_low) / run_range_long
        g["fib_retrace_long"] = retrace_long
        g["fib_zone_long"] = retrace_long.between(params.fib_zone_min, params.fib_zone_max)
        g["fib_golden_long"] = retrace_long.between(params.fib_golden_min, params.fib_golden_max)

        run_range_short = (g["intraday_high_so_far"] - prior_low).replace(0, np.nan)
        retrace_short = (recent_high - prior_low) / run_range_short
        g["fib_retrace_short"] = retrace_short
        g["fib_zone_short"] = retrace_short.between(params.fib_zone_min, params.fib_zone_max)
        g["fib_golden_short"] = retrace_short.between(params.fib_golden_min, params.fib_golden_max)

        volume_ok = (g["rvol_time_of_day"] >= params.rvol_min) | (g["volume"] >= 1.15 * g["median_volume_last_20_5m"])
        pullback_volume_ok = recent_vol <= 1.25 * g["median_volume_last_20_5m"]
        strong_long_candle = (
            (g["close"] > g["open"])
            & (g["candle_close_position"] >= params.candle_close_position_min)
        )
        strong_short_candle = (
            (g["close"] < g["open"])
            & (g["candle_close_position"] <= params.candle_close_position_short_max)
        )
        not_chasing_long = (g["vwap_extension_atr"] <= params.max_vwap_extension_atr) & (g["candle_range_atr"] <= params.max_candle_range_atr)
        not_chasing_short = (g["vwap_extension_atr"] >= -params.max_vwap_extension_atr) & (g["candle_range_atr"] <= params.max_candle_range_atr)

        touched_vwap_or_ema_long = recent_low <= (np.minimum(g["session_vwap"], g["ema20"]) + params.ema_pullback_buffer_atr * g["atr5m14"])
        touched_vwap_or_ema_short = recent_high >= (np.maximum(g["session_vwap"], g["ema20"]) - params.ema_pullback_buffer_atr * g["atr5m14"])
        enough_pullback_long = (recent_high - recent_low) >= params.min_pullback_depth_atr * g["atr5m14"]
        enough_pullback_short = enough_pullback_long

        # Trend continuation: pullback into VWAP/EMA/Fibonacci zone, then reclaim prior bar high.
        g["trigger_v4_trend_pullback_long"] = (
            bool(params.enable_trend_pullback)
            & (g["close"] > g["session_vwap"])
            & (g["ema9"] >= g["ema20"])
            & (g["close"] > prev_high)
            & (touched_vwap_or_ema_long | g["fib_zone_long"])
            & enough_pullback_long
            & volume_ok
            & pullback_volume_ok
            & strong_long_candle
            & not_chasing_long
        )

        # V13 micro-pullback continuation: in strong trends, many valid day-trading
        # entries only retest EMA9 / shallow VWAP area, not EMA20/VWAP deeply enough
        # for the older V12 trigger. Keep this strict: must be near VWAP, have RVOL,
        # close through the prior bar high, and avoid chasing extension.
        recent_depth = (recent_high - recent_low) / g["atr5m14"].replace(0, np.nan)
        touched_ema9_long = recent_low <= (g["ema9"] + float(getattr(params, "micro_pullback_ema9_buffer_atr", 0.12)) * g["atr5m14"])
        shallow_pullback_depth_ok = recent_depth >= float(getattr(params, "micro_pullback_min_depth_atr", 0.16))
        g["trigger_v13_micro_pullback_long"] = (
            bool(getattr(params, "enable_micro_pullback", True))
            & (g["close"] > g["session_vwap"])
            & (g["ema9"] >= g["ema20"])
            & (g["close"] > prev_high)
            & touched_ema9_long
            & shallow_pullback_depth_ok
            & (g["rvol_time_of_day"] >= float(getattr(params, "micro_pullback_min_rvol", 1.25)))
            & strong_long_candle
            & pullback_volume_ok
            & (g["vwap_extension_atr"] <= float(getattr(params, "micro_pullback_max_vwap_extension_atr", 0.90)))
            & (g["day_relative_strength"] >= float(getattr(params, "micro_pullback_min_day_rs", 0.50)))
            & (g["open_relative_strength"] >= float(getattr(params, "micro_pullback_min_open_rs", 0.00)))
        )

        # Opening range retest: breakout already happened; entry after retest and continuation.
        g["trigger_v4_or_retest_long"] = (
            bool(params.enable_or_retest)
            & (g["close"] > g["opening_range_high"] + params.breakout_buffer_atr * g["atr5m14"])
            & (recent_low <= g["opening_range_high"] + params.retest_buffer_atr * g["atr5m14"])
            & (prev_close >= g["session_vwap"].shift(1))
            & (g["close"] > prev_high)
            & volume_ok
            & strong_long_candle
            & not_chasing_long
        )

        # Mean reversion long: capitulation candle under VWAP, then first strength confirmation.
        g["trigger_v4_mean_reversion_long"] = (
            bool(params.enable_mean_reversion)
            & (g["vwap_extension_atr"] <= -params.mr_min_vwap_extension_atr)
            & (g["rsi2"] <= params.mr_rsi2_long_max)
            & (g["lower_wick"] >= 0.35 * g["candle_range"])
            & (g["close"] > prev_high)
            & (g["qqq_15min_change_percent"].fillna(0) > -0.45)
            & volume_ok
        )

        # V10 additional setup: VWAP reclaim after a controlled flush.
        # This is designed for the older/less explosive periods where pure momentum
        # produced too few trades. It requires an actual flush below VWAP, oversold
        # short-term pressure, and a reclaim/strength candle. It is not allowed to
        # chase far above VWAP.
        recent_rsi2_min = g["rsi2"].rolling(4, min_periods=1).min()
        recent_below_vwap = recent_low <= (g["session_vwap"] - 0.55 * g["atr5m14"])
        reclaim_vwap = (prev_close < g["session_vwap"].shift(1)) & (g["close"] > g["session_vwap"] + params.vwap_reclaim_buffer_atr * g["atr5m14"])
        reclaim_strength = (g["close"] > prev_high) & (g["close"] > g["ema9"])
        bullish_reclaim_candle = (
            g.get("bullish_rejection_candle", False)
            | g.get("bullish_continuation_candle", False)
            | ((g["lower_wick"] >= 0.25 * g["candle_range"]) & strong_long_candle)
        )
        g["trigger_v10_vwap_reclaim_reversal_long"] = (
            bool(getattr(params, "enable_vwap_reclaim_reversal", True))
            & recent_below_vwap
            & (recent_rsi2_min <= 22.0)
            & (reclaim_vwap | reclaim_strength)
            & bullish_reclaim_candle
            & volume_ok
            & (g["vwap_extension_atr"] <= 0.85)
            & (g["qqq_15min_change_percent"].fillna(0) > -0.55)
            & (g["close"] >= g["ema20"] - 0.25 * g["atr5m14"])
        )

        # Short-side versions. These are disabled unless direction_mode is long_short.
        allow_short = str(params.direction_mode).lower() in {"long_short", "short_only"}
        g["trigger_v4_trend_pullback_short"] = (
            bool(params.enable_trend_pullback)
            & allow_short
            & (g["close"] < g["session_vwap"])
            & (g["ema9"] <= g["ema20"])
            & (g["close"] < prev_low)
            & (touched_vwap_or_ema_short | g["fib_zone_short"])
            & enough_pullback_short
            & volume_ok
            & pullback_volume_ok
            & strong_short_candle
            & not_chasing_short
        )
        g["trigger_v4_or_retest_short"] = (
            bool(params.enable_or_retest)
            & allow_short
            & (g["close"] < g["opening_range_low"] - params.breakout_buffer_atr * g["atr5m14"])
            & (recent_high >= g["opening_range_low"] - params.retest_buffer_atr * g["atr5m14"])
            & (prev_close <= g["session_vwap"].shift(1))
            & (g["close"] < prev_low)
            & volume_ok
            & strong_short_candle
            & not_chasing_short
        )
        g["trigger_v4_mean_reversion_short"] = (
            allow_short
            & bool(params.enable_mean_reversion)
            & (g["vwap_extension_atr"] >= params.mr_min_vwap_extension_atr)
            & (g["rsi2"] >= params.mr_rsi2_short_min)
            & (g["upper_wick"] >= 0.35 * g["candle_range"])
            & (g["close"] < prev_low)
            & (g["qqq_15min_change_percent"].fillna(0) < 0.45)
            & volume_ok
        )

        # V18 Opportunity Rules. These modules are based on the independent
        # opportunity dataset, not on the old strategy candidate set.
        v18_enabled = bool(getattr(params, "enable_opportunity_v18", False)) and str(getattr(params, "strategy_profile", "")).lower().startswith("opportunity_v18")
        time_early = (g["time_str"] >= "09:40") & (g["time_str"] <= "09:55")
        time_10am = (g["time_str"] >= "10:00") & (g["time_str"] <= "10:55")
        time_11am = (g["time_str"] >= "11:00") & (g["time_str"] <= "11:55")
        daily_atr_pct = pd.to_numeric(g["daily_atr14_percent"], errors="coerce")
        low_atr_day = daily_atr_pct < 2.50
        moderate_atr_day = daily_atr_pct.between(1.50, 4.00)
        day_range = (g["intraday_high_so_far"] - g["intraday_low_so_far"]).replace(0, np.nan)
        range_pos = (g["close"] - g["intraday_low_so_far"]) / day_range
        g["range_position_day"] = range_pos
        near_vwap_long = g["vwap_extension_atr"].between(-1.00, 0.75)
        near_vwap_short = g["vwap_extension_atr"].between(-0.75, 1.00)
        ema_stack_long = (g["ema9"] >= g["ema20"]) & (g["close"] >= g["ema9"])
        ema_stack_short = (g["ema9"] <= g["ema20"]) & (g["close"] <= g["ema9"])
        qqq_same_long = (g["qqq_change_from_open"].fillna(0) >= 0) | (g["qqq_15min_change_percent"].fillna(0) >= 0)
        qqq_same_short = (g["qqq_change_from_open"].fillna(0) <= 0) | (g["qqq_15min_change_percent"].fillna(0) <= 0)
        not_extreme_candle = g["candle_range_atr"].fillna(0) <= 2.20
        entry_quality_long = strong_long_candle | g["bullish_rejection_candle"].fillna(False) | g["bullish_engulfing_candle"].fillna(False) | (g["candle_close_position"] >= 0.52)
        entry_quality_short = strong_short_candle | g["bearish_rejection_candle"].fillna(False) | g["bearish_engulfing_candle"].fillna(False) | (g["candle_close_position"] <= 0.48)

        # M1: Early low-ATR long opportunity family.
        g["trigger_v18_m1_early_low_atr_long"] = (
            v18_enabled
            & bool(getattr(params, "enable_v18_m1_early_low_atr_long", True))
            & time_early
            & low_atr_day
            & entry_quality_long
            & not_extreme_candle
            & (
                near_vwap_long
                | ema_stack_long
                | (g["day_relative_strength"] >= 0.50)
                | (g["open_relative_strength"] <= -0.50)
            )
        )

        # M2: Early low-ATR short opportunity family.
        g["trigger_v18_m2_early_low_atr_short"] = (
            v18_enabled
            & bool(getattr(params, "enable_v18_m2_early_low_atr_short", True))
            & allow_short
            & time_early
            & low_atr_day
            & entry_quality_short
            & not_extreme_candle
            & (
                (g["open_relative_strength"] <= -0.50)
                | (near_vwap_short & (g["day_relative_strength"] <= -0.50))
                | ema_stack_short
            )
        )

        # M3: Controlled high-gap continuation. Direction follows the gap and
        # requires post-open confirmation, avoiding blind gap chasing.
        g["trigger_v18_m3_gap_long"] = (
            v18_enabled
            & bool(getattr(params, "enable_v18_m3_controlled_gap", True))
            & (g["gap_percent"] >= 8.0)
            & (g["stock_change_from_open"] >= 1.50)
            & (g["vwap_extension_atr"] >= 0)
            & (g["vwap_extension_atr"] <= 4.0)
            & ((g["day_relative_strength"] >= 0.50) | (g["rvol_time_of_day"] >= 1.30))
            & entry_quality_long
            & not_extreme_candle
            & (g["time_str"] >= "09:45")
            & (g["time_str"] <= "11:15")
        )
        g["trigger_v18_m3_gap_short"] = (
            v18_enabled
            & bool(getattr(params, "enable_v18_m3_controlled_gap", True))
            & allow_short
            & (g["gap_percent"] <= -8.0)
            & (g["stock_change_from_open"] <= -1.50)
            & (g["vwap_extension_atr"] <= 0)
            & (g["vwap_extension_atr"] >= -4.0)
            & ((g["day_relative_strength"] <= -0.50) | (g["rvol_time_of_day"] >= 1.30))
            & entry_quality_short
            & not_extreme_candle
            & (g["time_str"] >= "09:45")
            & (g["time_str"] <= "11:15")
        )

        # M4: 10:00 moderate-ATR range continuation.
        g["trigger_v18_m4_10am_continuation_long"] = (
            v18_enabled
            & bool(getattr(params, "enable_v18_m4_10am_continuation", True))
            & time_10am
            & moderate_atr_day
            & (range_pos >= 0.75)
            & g["vwap_extension_atr"].between(0.0, 2.0)
            & (qqq_same_long | (g["day_relative_strength"] >= 0.50))
            & entry_quality_long
            & (g["rvol_time_of_day"] >= 0.95)
        )
        g["trigger_v18_m4_10am_continuation_short"] = (
            v18_enabled
            & bool(getattr(params, "enable_v18_m4_10am_continuation", True))
            & allow_short
            & time_10am
            & moderate_atr_day
            & (range_pos <= 0.25)
            & g["vwap_extension_atr"].between(-2.0, 0.0)
            & (qqq_same_short | (g["day_relative_strength"] <= -0.50))
            & entry_quality_short
            & (g["rvol_time_of_day"] >= 0.95)
        )

        # M5: 11:00 opening-range rejection. This is a smaller module and should
        # use lower risk. It only triggers on a rejection candle after a real OR break.
        g["trigger_v18_m5_or_rejection_long"] = (
            v18_enabled
            & bool(getattr(params, "enable_v18_m5_11am_or_rejection", True))
            & time_11am
            & (g["close"] >= g["opening_range_high"] + 1.0 * g["atr5m14"])
            & g["bullish_rejection_candle"].fillna(False)
            & (g["rvol_time_of_day"] >= 0.90)
        )
        g["trigger_v18_m5_or_rejection_short"] = (
            v18_enabled
            & bool(getattr(params, "enable_v18_m5_11am_or_rejection", True))
            & allow_short
            & time_11am
            & (g["close"] <= g["opening_range_low"] - 1.0 * g["atr5m14"])
            & g["bearish_rejection_candle"].fillna(False)
            & (g["rvol_time_of_day"] >= 0.90)
        )

        frames.append(g)

    out = pd.concat(frames, ignore_index=True) if frames else out

    trigger_cols_long = [
        "trigger_v18_m1_early_low_atr_long",
        "trigger_v18_m3_gap_long",
        "trigger_v18_m4_10am_continuation_long",
        "trigger_v18_m5_or_rejection_long",
        "trigger_v4_trend_pullback_long",
        "trigger_v13_micro_pullback_long",
        "trigger_v4_or_retest_long",
        "trigger_v4_mean_reversion_long",
        "trigger_v10_vwap_reclaim_reversal_long",
    ]
    trigger_cols_short = [
        "trigger_v18_m2_early_low_atr_short",
        "trigger_v18_m3_gap_short",
        "trigger_v18_m4_10am_continuation_short",
        "trigger_v18_m5_or_rejection_short",
        "trigger_v4_trend_pullback_short",
        "trigger_v4_or_retest_short",
        "trigger_v4_mean_reversion_short",
        "trigger_v10_vwap_reclaim_reversal_short",
    ]
    for col in trigger_cols_long + trigger_cols_short:
        if col not in out.columns:
            out[col] = False

    out["long_trigger"] = False
    for col in trigger_cols_long:
        out["long_trigger"] = out["long_trigger"] | out[col].fillna(False)
    out["short_trigger"] = False
    for col in trigger_cols_short:
        out["short_trigger"] = out["short_trigger"] | out[col].fillna(False)
    out["key_level_trigger"] = out["long_trigger"] | out["short_trigger"]
    out["side"] = np.where(out["short_trigger"], "short", np.where(out["long_trigger"], "long", ""))

    conditions = [
        out["trigger_v18_m1_early_low_atr_long"].fillna(False),
        out["trigger_v18_m3_gap_long"].fillna(False),
        out["trigger_v18_m4_10am_continuation_long"].fillna(False),
        out["trigger_v18_m5_or_rejection_long"].fillna(False),
        out["trigger_v13_micro_pullback_long"].fillna(False),
        out["trigger_v4_trend_pullback_long"].fillna(False),
        out["trigger_v4_or_retest_long"].fillna(False),
        out["trigger_v4_mean_reversion_long"].fillna(False),
        out["trigger_v10_vwap_reclaim_reversal_long"].fillna(False),
        out["trigger_v18_m2_early_low_atr_short"].fillna(False),
        out["trigger_v18_m3_gap_short"].fillna(False),
        out["trigger_v18_m4_10am_continuation_short"].fillna(False),
        out["trigger_v18_m5_or_rejection_short"].fillna(False),
        out["trigger_v4_trend_pullback_short"].fillna(False),
        out["trigger_v4_or_retest_short"].fillna(False),
        out["trigger_v4_mean_reversion_short"].fillna(False),
        out["trigger_v10_vwap_reclaim_reversal_short"].fillna(False),
    ]
    out["trigger_type"] = np.select(
        conditions,
        [
            "v18_m1_early_low_atr_long",
            "v18_m3_gap_long",
            "v18_m4_10am_continuation_long",
            "v18_m5_or_rejection_long",
            "v13_micro_pullback_long",
            "v4_trend_pullback_long",
            "v4_or_retest_long",
            "v4_mean_reversion_long",
            "v10_vwap_reclaim_reversal_long",
            "v18_m2_early_low_atr_short",
            "v18_m3_gap_short",
            "v18_m4_10am_continuation_short",
            "v18_m5_or_rejection_short",
            "v4_trend_pullback_short",
            "v4_or_retest_short",
            "v4_mean_reversion_short",
            "v10_vwap_reclaim_reversal_short",
        ],
        default="",
    )
    out["trigger_level"] = np.select(
        conditions,
        [
            out["low"],
            out["recent_low"],
            out["recent_low"],
            out["opening_range_high"],
            out["recent_low"],
            out["recent_low"],
            out["opening_range_high"],
            out["session_vwap"],
            out["session_vwap"],
            out["high"],
            out["recent_high"],
            out["recent_high"],
            out["opening_range_low"],
            out["recent_high"],
            out["opening_range_low"],
            out["session_vwap"],
            out["session_vwap"],
        ],
        default=np.nan,
    )
    # If long and short triggers happen on the same candle, align side with the
    # selected trigger_type rather than the broad aggregate masks.
    out["side"] = np.select(
        [out["trigger_type"].str.contains("short", na=False), out["trigger_type"].str.contains("long", na=False)],
        ["short", "long"],
        default="",
    )

    # Setup family lets exits and diagnostics treat trend vs mean-reversion differently.
    out["setup_family"] = np.select(
        [
            out["trigger_type"].str.contains("mean_reversion", na=False),
            out["trigger_type"].str.contains("vwap_reclaim_reversal", na=False),
            out["trigger_type"].str.contains("or_retest", na=False),
        ],
        ["mean_reversion", "vwap_reclaim_reversal", "opening_range_retest"],
        default="trend_pullback",
    )

    # Scores. These are deliberately component-based so reports can show why a setup passed.
    market_score_long = np.select(
        [out["market_bull"], out["market_neutral"] & (out["day_relative_strength"] >= 1.25), out["market_bear"] & (out["day_relative_strength"] >= params.weak_market_min_relative_strength)],
        [18.0, 10.0, 5.0],
        default=0.0,
    )
    market_score_short = np.select(
        [out["market_bear"], out["market_neutral"] & (out["day_relative_strength"] <= -1.25), out["market_bull"] & (out["day_relative_strength"] <= -params.weak_market_min_relative_strength)],
        [18.0, 10.0, 5.0],
        default=0.0,
    )
    out["market_score"] = np.where(out["side"] == "short", market_score_short, market_score_long)

    out["time_score"] = np.select(
        [primary_window, afternoon_window, midday_window],
        [12.0, 6.0, -8.0],
        default=0.0,
    )

    out["volume_score"] = np.select(
        [out["rvol_time_of_day"] >= 2.2, out["rvol_time_of_day"] >= 1.5, out["volume"] >= 1.5 * out["median_volume_last_20_5m"], out["volume"] >= 1.15 * out["median_volume_last_20_5m"]],
        [16.0, 12.0, 10.0, 6.0],
        default=0.0,
    )
    rs_abs = np.where(out["side"] == "short", -out["day_relative_strength"], out["day_relative_strength"])
    ors_abs = np.where(out["side"] == "short", -out["open_relative_strength"], out["open_relative_strength"])
    out["relative_strength_score"] = np.select(
        [rs_abs >= 3.0, rs_abs >= 1.8, ors_abs >= 1.0, ors_abs >= 0.4],
        [16.0, 12.0, 9.0, 5.0],
        default=0.0,
    )
    out["trigger_score"] = np.select(
        [
            (out["trigger_type"].str.contains("trend_pullback", na=False) | out["trigger_type"].str.contains("micro_pullback", na=False)),
            out["trigger_type"].str.contains("vwap_reclaim_reversal", na=False),
            out["trigger_type"].str.contains("or_retest", na=False),
            out["trigger_type"].str.contains("mean_reversion", na=False),
        ],
        [22.0, 18.0, 18.0, 16.0],
        default=0.0,
    )
    trend_score_long = (
        5.0 * (out["close"] > out["session_vwap"]).astype(float)
        + 5.0 * (out["close"] > out["ema9"]).astype(float)
        + 5.0 * (out["ema9"] >= out["ema20"]).astype(float)
    )
    trend_score_short = (
        5.0 * (out["close"] < out["session_vwap"]).astype(float)
        + 5.0 * (out["close"] < out["ema9"]).astype(float)
        + 5.0 * (out["ema9"] <= out["ema20"]).astype(float)
    )
    out["trend_score"] = np.where(out["side"] == "short", trend_score_short, trend_score_long)
    candle_long = out["candle_close_position"] >= 0.70
    candle_short = out["candle_close_position"] <= 0.30
    out["candle_score"] = np.where(
        out["side"] == "short",
        np.where(candle_short, 8.0, np.where(out["candle_close_position"] <= params.candle_close_position_short_max, 5.0, 0.0)),
        np.where(candle_long, 8.0, np.where(out["candle_close_position"] >= params.candle_close_position_min, 5.0, 0.0)),
    )

    # Candlestick pattern module. This is separate from the simple candle-close score.
    # It can be disabled from the dashboard. In score mode it helps ranking; in
    # confirm mode it also becomes an entry requirement. Opposing reversal candles
    # penalize the setup before entry and can trigger a protective exit later.
    candle_mode = str(getattr(params, "candle_pattern_mode", "off")).lower()
    if candle_mode not in {"off", "score", "confirm", "selective", "exit_only"}:
        candle_mode = "off"
    long_entry_candle_ok = out.get("long_entry_candle_ok", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    short_entry_candle_ok = out.get("short_entry_candle_ok", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    long_exit_warning = out.get("long_exit_warning_candle", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    short_exit_warning = out.get("short_exit_warning_candle", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    out["entry_candle_ok"] = np.where(out["side"] == "short", short_entry_candle_ok, long_entry_candle_ok)
    out["opposing_candle_warning"] = np.where(out["side"] == "short", short_exit_warning, long_exit_warning)

    pattern = out.get("candle_pattern_primary", pd.Series("neutral", index=out.index)).fillna("neutral").astype(str)
    out["bullish_rejection_entry"] = pattern.eq("bullish_rejection")
    out["bearish_rejection_entry"] = pattern.eq("bearish_rejection")
    out["rejection_entry_for_side"] = np.where(out["side"].eq("short"), out["bearish_rejection_entry"], out["bullish_rejection_entry"])
    out["continuation_entry_for_side"] = np.where(out["side"].eq("short"), pattern.eq("bearish_continuation"), pattern.eq("bullish_continuation"))
    out["weak_entry_pattern_for_side"] = np.where(
        out["side"].eq("short"),
        pattern.isin(["bearish_engulfing", "bearish_inside_breakout"]),
        pattern.isin(["bullish_engulfing", "bullish_inside_breakout"]),
    )

    # V7 selective candle logic is based on the actual long-period reports:
    # rejection candles were the only clearly profitable pattern; broad continuation,
    # engulfing, and inside-breakout labels were not predictive by themselves.
    if candle_mode in {"off", "exit_only"}:
        out["candle_pattern_score"] = 0.0
    elif candle_mode == "selective":
        out["candle_pattern_score"] = np.select(
            [out["rejection_entry_for_side"], out["continuation_entry_for_side"], out["weak_entry_pattern_for_side"], out["opposing_candle_warning"]],
            [float(getattr(params, "selective_rejection_bonus", 12.0)), float(getattr(params, "selective_continuation_bonus", 2.0)), -float(getattr(params, "selective_weak_pattern_penalty", 12.0)), -float(getattr(params, "candle_opposing_penalty", 10.0))],
            default=0.0,
        )
    elif candle_mode == "confirm":
        # Confirmation mode requires a favorable candle but does not add a score boost,
        # otherwise it becomes equivalent to broad score mode.
        out["candle_pattern_score"] = 0.0
    else:
        out["candle_pattern_score"] = np.where(
            out["entry_candle_ok"],
            float(getattr(params, "candle_entry_bonus", 8.0)),
            np.where(out["opposing_candle_warning"], -float(getattr(params, "candle_opposing_penalty", 10.0)), 0.0),
        )

    fib_score_long = np.where(out["fib_golden_long"].fillna(False), 8.0, np.where(out["fib_zone_long"].fillna(False), 5.0, 0.0))
    fib_score_short = np.where(out["fib_golden_short"].fillna(False), 8.0, np.where(out["fib_zone_short"].fillna(False), 5.0, 0.0))
    out["structure_score"] = np.where(out["side"] == "short", fib_score_short, fib_score_long)
    out["reversion_score"] = np.select(
        [
            out["trigger_type"].str.contains("mean_reversion", na=False) & (out["rsi2"] <= params.mr_rsi2_long_max) & (out["side"] == "long"),
            out["trigger_type"].str.contains("mean_reversion", na=False) & (out["rsi2"] >= params.mr_rsi2_short_min) & (out["side"] == "short"),
            out["trigger_type"].str.contains("vwap_reclaim_reversal", na=False),
        ],
        [12.0, 12.0, 8.0],
        default=0.0,
    )

    # Penalize extension and lunch. We do not block all midday setups, but a midday setup
    # must be much stronger to survive the score threshold.
    out["stretch_penalty"] = np.select(
        [out["candle_range_atr"] > 2.0, out["candle_range_atr"] > 1.6, midday_window],
        [10.0, 5.0, 8.0],
        default=0.0,
    )
    out["candidate_score"] = (
        out["market_score"]
        + out["time_score"]
        + out["volume_score"]
        + out["relative_strength_score"]
        + out["trigger_score"]
        + out["trend_score"]
        + out["candle_score"]
        + out["candle_pattern_score"]
        + out["structure_score"]
        + out["reversion_score"]
        - out["stretch_penalty"]
    ).clip(lower=0, upper=100)

    # V18 module score. The old score was optimized for the previous trend-pullback
    # strategy. For opportunity modules, use module base scores plus only simple
    # features that were common in the discovery analysis: relative strength, RVOL,
    # controlled VWAP extension, range position, and candle context.
    v18_mask = out["trigger_type"].str.startswith("v18_", na=False)
    out["opportunity_module"] = np.select(
        [
            out["trigger_type"].str.contains("m1_", na=False),
            out["trigger_type"].str.contains("m2_", na=False),
            out["trigger_type"].str.contains("m3_", na=False),
            out["trigger_type"].str.contains("m4_", na=False),
            out["trigger_type"].str.contains("m5_", na=False),
        ],
        ["M1_early_low_atr_long", "M2_early_low_atr_short", "M3_controlled_gap", "M4_10am_continuation", "M5_11am_or_rejection"],
        default="legacy",
    )
    v18_base = np.select(
        [
            out["trigger_type"].str.contains("m1_", na=False),
            out["trigger_type"].str.contains("m2_", na=False),
            out["trigger_type"].str.contains("m3_", na=False),
            out["trigger_type"].str.contains("m4_", na=False),
            out["trigger_type"].str.contains("m5_", na=False),
        ],
        [90.0, 80.0, 85.0, 72.0, 65.0],
        default=0.0,
    )
    directional_day_rs = np.where(out["side"].eq("short"), -out["day_relative_strength"], out["day_relative_strength"])
    directional_open_rs = np.where(out["side"].eq("short"), -out["open_relative_strength"], out["open_relative_strength"])
    directional_vwap_ext = np.where(out["side"].eq("short"), -out["vwap_extension_atr"], out["vwap_extension_atr"])
    range_pos_series = pd.to_numeric(out.get("range_position_day", pd.Series(np.nan, index=out.index)), errors="coerce")
    directional_range_pos = np.where(out["side"].eq("short"), 1.0 - range_pos_series, range_pos_series)
    v18_bonus = (
        np.where(pd.to_numeric(out["rvol_time_of_day"], errors="coerce") >= 1.50, 5.0, 0.0)
        + np.where(directional_day_rs >= 1.00, 5.0, np.where(directional_day_rs >= 0.50, 2.5, 0.0))
        + np.where(directional_open_rs >= 0.75, 3.0, 0.0)
        + np.where((directional_vwap_ext >= -0.20) & (directional_vwap_ext <= 1.25), 4.0, 0.0)
        + np.where(directional_range_pos >= 0.75, 3.0, 0.0)
        + np.where(out["rejection_entry_for_side"], 4.0, 0.0)
        + np.where(out["continuation_entry_for_side"], 2.0, 0.0)
    )
    v18_penalty = (
        np.where(np.abs(pd.to_numeric(out["gap_percent"], errors="coerce")) > 14.0, 6.0, 0.0)
        + np.where(np.abs(pd.to_numeric(out["vwap_extension_atr"], errors="coerce")) > 3.5, 8.0, 0.0)
        + np.where(pd.to_numeric(out["candle_range_atr"], errors="coerce") > 2.2, 5.0, 0.0)
    )
    out.loc[v18_mask, "candidate_score"] = pd.Series(v18_base + v18_bonus - v18_penalty, index=out.index).clip(lower=0, upper=100).loc[v18_mask]

    out["quality"] = np.select(
        [out["candidate_score"] >= 86, out["candidate_score"] >= params.high_quality_score, out["candidate_score"] >= params.min_candidate_score],
        ["A", "B", "C"],
        default="",
    )

    # Component flags for diagnostics.
    out["support_volume"] = out["volume_score"] >= 10
    out["support_relative_strength"] = out["relative_strength_score"] >= 9
    out["support_trend"] = out["trend_score"] >= 10
    out["support_strong_close"] = out["candle_score"] >= 5
    out["support_candle_pattern"] = out["candle_pattern_score"] > 0
    out["support_structure"] = out["structure_score"] >= 5
    out["supporting_score"] = sum(_to_bool_int(out[c]) for c in ["support_volume", "support_relative_strength", "support_trend", "support_strong_close", "support_candle_pattern", "support_structure"])

    # Direction-specific market rules. Long-only mode blocks short signals.
    direction_mode_value = str(params.direction_mode).lower()
    direction_ok = ((out["side"] == "long") & (direction_mode_value in {"long_only", "long_short"})) | ((out["side"] == "short") & (direction_mode_value in {"short_only", "long_short"}))
    market_ok_long = (out["side"] == "long") & (out["market_bull"] | out["market_neutral"] | (out["day_relative_strength"] >= params.weak_market_min_relative_strength))
    market_ok_short = (out["side"] == "short") & (out["market_bear"] | out["market_neutral"] | (out["day_relative_strength"] <= -params.weak_market_min_relative_strength))
    out["market_direction_ok"] = market_ok_long | market_ok_short

    candle_mode_value = str(getattr(params, "candle_pattern_mode", "off")).lower()
    candle_mode_confirm = candle_mode_value == "confirm"
    candle_mode_selective = candle_mode_value == "selective"
    candle_entry_filter = (~candle_mode_confirm) | out["entry_candle_ok"].fillna(False).astype(bool)

    out["low_followthrough_context"] = pd.to_numeric(out["daily_atr14_percent"], errors="coerce") < float(getattr(params, "low_followthrough_atr_pct", 4.0))
    is_or_retest = out["setup_family"].eq("opening_range_retest")
    is_mean_reversion = out["setup_family"].eq("mean_reversion")
    is_momentum = ~is_mean_reversion

    # Selective candle gate:
    # - OR/retest is only kept when it is a true rejection/retest candle.
    # - In low-followthrough regimes, continuation candles are not enough; require rejection.
    # - In normal regimes, trend pullback may use rejection, continuation, or neutral candles.
    selective_pattern_ok = pd.Series(True, index=out.index)
    if candle_mode_selective:
        neutral_pattern = pattern.eq("neutral")
        trend_normal_ok = (~out["low_followthrough_context"]) & out["setup_family"].eq("trend_pullback") & (out["rejection_entry_for_side"] | out["continuation_entry_for_side"] | neutral_pattern)
        trend_lowft_ok = out["low_followthrough_context"] & out["setup_family"].eq("trend_pullback") & out["rejection_entry_for_side"] & (out["rvol_time_of_day"] >= 1.4)
        or_ok = is_or_retest & out["rejection_entry_for_side"] & (out["rvol_time_of_day"] >= 1.3)
        mr_ok = is_mean_reversion & out["rejection_entry_for_side"]
        selective_pattern_ok = trend_normal_ok | trend_lowft_ok | or_ok | mr_ok

    if bool(getattr(params, "enable_or_retest_only_rejection", True)):
        # V10: OR/retest was the least stable module. Keep it only when it is
        # genuinely high quality: a rejection retest, or a continuation retest
        # with strong RVOL/relative strength and no VWAP chase. This is a generic
        # quality rule, not a symbol-specific post-filter.
        or_score_floor = float(getattr(params, "or_retest_min_score", 92.0))
        or_rvol_floor = float(getattr(params, "or_retest_min_rvol", 1.35))
        or_retest_quality_ok = (
            (~is_or_retest)
            | (
                is_or_retest
                & (out["candidate_score"] >= or_score_floor)
                & (out["rvol_time_of_day"] >= or_rvol_floor)
                & (out["vwap_extension_atr"].abs() <= 1.10)
                & (
                    out["rejection_entry_for_side"]
                    | (
                        out["continuation_entry_for_side"]
                        & (out["open_relative_strength"] >= 1.75)
                        & (out["day_relative_strength"] >= 2.25)
                    )
                )
            )
        )
    else:
        or_retest_quality_ok = pd.Series(True, index=out.index)

    # Regime gate. Low-followthrough momentum trades were the largest loss source
    # in V5/V6. We do not ban all low-volatility trades; we only allow them when
    # the candle shows real rejection at the level, or when using mean reversion.
    if bool(getattr(params, "avoid_low_followthrough_momentum", True)):
        lowft_momentum_ok = (~(out["low_followthrough_context"] & is_momentum)) | out["rejection_entry_for_side"]
    else:
        lowft_momentum_ok = pd.Series(True, index=out.index)

    # V8 robust market/candle filters. These are intentionally broad, evidence-led
    # rules from the long reports:
    # - Daily ATR below ~2.5% did not provide enough intraday range for momentum.
    # - VWAP extension above ~1.25 ATR often meant chasing.
    # - Open-relative-strength in the 0..1% dead zone was weak; strong open RS
    #   around 2..4% was much better.
    # - Inside-bar breakout labels were consistently bad in the long reports.
    # - High-volatility trades need real RVOL and relative strength.
    pattern_str = pattern.fillna("neutral").astype(str)
    is_bull_inside = pattern_str.eq("bullish_inside_breakout")
    is_bear_inside = pattern_str.eq("bearish_inside_breakout")
    inside_breakout_for_side = np.where(out["side"].eq("short"), is_bear_inside, is_bull_inside)
    is_neutral_pattern = pattern_str.eq("neutral")
    is_engulfing_for_side = np.where(out["side"].eq("short"), pattern_str.eq("bearish_engulfing"), pattern_str.eq("bullish_engulfing"))
    strong_engulfing_context = (out["rvol_time_of_day"] >= 1.5) & ((rs_abs >= 2.0) | (ors_abs >= 1.5))

    daily_atr = pd.to_numeric(out["daily_atr14_percent"], errors="coerce")
    abs_vwap_ext = pd.to_numeric(out["vwap_extension_atr"], errors="coerce").abs()
    high_vol_context = daily_atr >= float(getattr(params, "high_vol_daily_atr_pct", 4.5))
    tradable_momentum_range = daily_atr >= float(getattr(params, "min_momentum_daily_atr_pct", 2.5))
    clean_extension = abs_vwap_ext <= float(getattr(params, "max_clean_vwap_extension_atr", 1.25))
    open_rs_dead_zone = (ors_abs >= 0.0) & (ors_abs < float(getattr(params, "weak_open_rs_upper", 1.0)))
    dead_zone_escape = (out["rvol_time_of_day"] >= 2.0) | (rs_abs >= 3.0) | out["rejection_entry_for_side"]
    high_vol_quality = (~high_vol_context) | (
        (ors_abs >= float(getattr(params, "min_open_rs_high_vol", 1.2)))
        & (rs_abs >= float(getattr(params, "min_day_rs_high_vol", 2.0)))
        & (out["rvol_time_of_day"] >= float(getattr(params, "min_rvol_high_vol", 1.45)))
        & (abs_vwap_ext <= float(getattr(params, "max_clean_vwap_extension_atr", 1.25)))
    ) | out["rejection_entry_for_side"]

    candle_quality_ok = pd.Series(True, index=out.index)
    if bool(getattr(params, "block_inside_breakout_entries", True)):
        candle_quality_ok = candle_quality_ok & (~inside_breakout_for_side)
    if bool(getattr(params, "block_neutral_confirm_entries", True)) and candle_mode in {"confirm", "selective"}:
        candle_quality_ok = candle_quality_ok & (~is_neutral_pattern)
    if bool(getattr(params, "allow_engulfing_only_with_volume_rs", True)):
        candle_quality_ok = candle_quality_ok & ((~is_engulfing_for_side) | strong_engulfing_context)

    if bool(getattr(params, "enable_v8_regime_filters", True)):
        momentum_range_ok = (~is_momentum) | tradable_momentum_range
        if bool(getattr(params, "strict_overextended_block", True)):
            # The long reports showed overextended chase entries were the most
            # consistent remaining loser. Rejection candles are not enough to
            # justify chasing above the clean VWAP-extension zone; let mean
            # reversion handle stretched moves instead.
            momentum_extension_ok = (~is_momentum) | clean_extension
        else:
            momentum_extension_ok = (~is_momentum) | clean_extension | out["rejection_entry_for_side"]
        momentum_open_rs_ok = (~is_momentum) | (~open_rs_dead_zone) | dead_zone_escape
        qqq_followthrough_ok = (~is_momentum) | (out["qqq_15min_change_percent"].fillna(0) >= -0.10) | (rs_abs >= 2.25) | (ors_abs >= 1.75)
        v8_regime_ok = momentum_range_ok & momentum_extension_ok & momentum_open_rs_ok & high_vol_quality & qqq_followthrough_ok & candle_quality_ok
    else:
        v8_regime_ok = candle_quality_ok

    out["v8_trade_context"] = np.select(
        [
            is_mean_reversion,
            daily_atr < float(getattr(params, "min_momentum_daily_atr_pct", 2.5)),
            high_vol_context & (~high_vol_quality),
            open_rs_dead_zone & (~dead_zone_escape),
            abs_vwap_ext > float(getattr(params, "max_clean_vwap_extension_atr", 1.25)),
            out["low_followthrough_context"],
            high_vol_context,
        ],
        [
            "mean_reversion",
            "too_low_range",
            "high_vol_weak_confirmation",
            "open_rs_dead_zone",
            "overextended_chase",
            "low_followthrough",
            "high_vol_clean",
        ],
        default="clean_momentum",
    )
    out["v8_regime_ok"] = v8_regime_ok
    out["v8_candle_quality_ok"] = candle_quality_ok

    # V12 core setup gate. This is based on the uploaded long-period reports:
    #   * trend_pullback is the core edge;
    #   * generic VWAP-reclaim reversal was negative;
    #   * the profitable VWAP subsets were: 10:00 reversal-window continuation,
    #     bullish engulfing reclaim, and controlled low-followthrough rejection.
    # This keeps trade frequency without letting the weak VWAP reclaims back in.
    is_trend_pullback_setup = out["setup_family"].eq("trend_pullback")
    is_vwap_reversal_setup = out["setup_family"].eq("vwap_reclaim_reversal")
    neutral_entry_pattern = pattern_str.eq("neutral")
    v12_vwap_window = (
        (out["time_str"] >= str(getattr(params, "v12_vwap_window_start", "10:00")))
        & (out["time_str"] <= str(getattr(params, "v12_vwap_window_end", "10:59")))
    )
    v12_vwap_engulfing_ok = (
        is_vwap_reversal_setup
        & is_engulfing_for_side
        & (out["candidate_score"] >= float(getattr(params, "v12_vwap_engulfing_min_score", 94.0)))
        & (out["rvol_time_of_day"] >= float(getattr(params, "v12_vwap_min_rvol", 1.45)))
    )
    v12_vwap_lowft_rejection_ok = (
        is_vwap_reversal_setup
        & out["rejection_entry_for_side"]
        & out["low_followthrough_context"]
        & v12_vwap_window
        & (out["rvol_time_of_day"] >= float(getattr(params, "v12_vwap_lowft_rejection_min_rvol", 1.10)))
    )
    v12_vwap_morning_continuation_ok = (
        is_vwap_reversal_setup
        & out["continuation_entry_for_side"]
        & v12_vwap_window
        & (out["candidate_score"] >= float(getattr(params, "v12_vwap_reversal_min_score", 96.0)))
        & (out["rvol_time_of_day"] >= float(getattr(params, "v12_vwap_min_rvol", 1.45)))
    )
    v12_vwap_reversal_ok = (
        v12_vwap_engulfing_ok | v12_vwap_lowft_rejection_ok | v12_vwap_morning_continuation_ok
    ) & ((~neutral_entry_pattern) | (not bool(getattr(params, "v12_vwap_reversal_block_neutral", True))))
    out["v12_vwap_quality_ok"] = (~is_vwap_reversal_setup) | v12_vwap_reversal_ok
    if bool(getattr(params, "enable_v12_core_filter", False)):
        v12_core_setup_ok = is_trend_pullback_setup | v12_vwap_reversal_ok
    else:
        v12_core_setup_ok = pd.Series(True, index=out.index)
    out["v12_core_setup_ok"] = v12_core_setup_ok

    # V13 quality gate for the new micro-pullback setup. It increases frequency
    # but only when score/RVOL/extension are still clean.
    is_micro_pullback_setup = out["trigger_type"].str.contains("micro_pullback", na=False)
    micro_quality_ok = (
        (~is_micro_pullback_setup)
        | (
            is_micro_pullback_setup
            & (out["candidate_score"] >= float(getattr(params, "micro_pullback_min_score", 88.0)))
            & (out["rvol_time_of_day"] >= float(getattr(params, "micro_pullback_min_rvol", 1.25)))
            & (out["vwap_extension_atr"] <= float(getattr(params, "micro_pullback_max_vwap_extension_atr", 0.90)))
            & (~out["low_followthrough_context"] | out["rejection_entry_for_side"] | (out["day_relative_strength"] >= 2.0))
        )
    )
    out["v13_micro_quality_ok"] = micro_quality_ok

    # V14 long-period robustness filter. The 2022-2026 report showed that the
    # remaining bad windows were mostly high-gap/high-volatility momentum chases
    # and low-followthrough entries with weak relative strength. These rules are
    # deliberately market-condition based, not symbol-name based.
    bullish_continuation_entry = pattern_str.eq("bullish_continuation")
    v14_filter_ok = pd.Series(True, index=out.index)
    if bool(getattr(params, "enable_v14_long_period_filters", True)):
        max_gap = float(getattr(params, "max_momentum_gap_percent", 8.0))
        max_stock_day_change = float(getattr(params, "max_momentum_stock_day_change_percent", 12.0))
        lowft_min_day_rs = float(getattr(params, "block_lowft_if_day_rs_below", 1.0))
        lowft_min_open_rs = float(getattr(params, "block_lowft_if_open_rs_below", 1.0))
        min_cont_rvol = float(getattr(params, "min_bullish_continuation_rvol", 1.35))

        extreme_gap_chase = (out["gap_percent"] > max_gap) & (~out["rejection_entry_for_side"])
        extreme_intraday_chase = (out["stock_day_change_percent"] > max_stock_day_change) & (~out["rejection_entry_for_side"])
        weak_lowft = out["low_followthrough_context"] & (
            (out["day_relative_strength"] < lowft_min_day_rs)
            | (out["open_relative_strength"] < lowft_min_open_rs)
        )
        weak_continuation_candle = bullish_continuation_entry & (out["rvol_time_of_day"] < min_cont_rvol)
        v14_filter_ok = ~(extreme_gap_chase | extreme_intraday_chase | weak_lowft | weak_continuation_candle)
    out["v14_filter_ok"] = v14_filter_ok

    # V15 quality layer from the 2022-2026 report. These are condition-based,
    # not symbol-specific. They target the remaining common failure modes:
    # late/noon entries, gap continuation attempts that immediately fail, and
    # weak bullish-engulfing entries without enough volume confirmation.
    v15_quality_ok = pd.Series(True, index=out.index)
    if str(getattr(params, "strategy_profile", "")).lower().startswith(("adaptive_v15", "adaptive_v16", "adaptive_v19")):
        noon_entry = out["timestamp_ny"].dt.strftime("%H:%M") >= "11:55"
        big_gap_continuation = (out["gap_percent"].abs() >= 4.0) & bullish_continuation_entry & (~out["rejection_entry_for_side"])
        weak_engulfing = pattern_str.eq("bullish_engulfing") & (out["rvol_time_of_day"] < 2.2)
        long_momentum = out["side"].eq("long") & out["setup_family"].eq("trend_pullback")
        v15_quality_ok = ~(long_momentum & (noon_entry | big_gap_continuation | weak_engulfing))
    out["v15_quality_ok"] = v15_quality_ok

    opportunity_profile = str(getattr(params, "strategy_profile", "")).lower().startswith("opportunity_v18")
    v18_module_mask = out["trigger_type"].str.startswith("v18_", na=False)
    if opportunity_profile:
        # Opportunity Rules mode intentionally bypasses the old V8/V12/V13 gates.
        # Those gates were designed around the narrow trend-pullback strategy and
        # would block the early low-ATR and short-side opportunities discovered in
        # the independent dataset. Liquidity, direction, exact module trigger, and
        # score threshold still apply.
        out["buy_alert"] = (
            direction_ok
            & out["liquidity_filter"]
            & v18_module_mask
            & candle_entry_filter
            & (out["candidate_score"] >= params.min_candidate_score)
        )
        if bool(getattr(params, "use_legacy_core_when_v18", False)):
            legacy_alert = (
                direction_ok
                & out["market_not_broken"]
                & out["market_direction_ok"]
                & out["time_filter"]
                & out["liquidity_filter"]
                & out["reason_for_move"]
                & out["key_level_trigger"]
                & candle_entry_filter
                & selective_pattern_ok
                & or_retest_quality_ok
                & lowft_momentum_ok
                & v8_regime_ok
                & out["v12_core_setup_ok"]
                & out["v13_micro_quality_ok"]
                & out["v14_filter_ok"]
                & out["v15_quality_ok"]
                & (out["candidate_score"] >= params.min_candidate_score)
                & (out["support_volume"] | out["support_relative_strength"] | out["support_candle_pattern"] | is_mean_reversion)
            )
            out["buy_alert"] = out["buy_alert"] | legacy_alert
    else:
        out["buy_alert"] = (
            direction_ok
            & out["market_not_broken"]
            & out["market_direction_ok"]
            & out["time_filter"]
            & out["liquidity_filter"]
            & out["reason_for_move"]
            & out["key_level_trigger"]
            & candle_entry_filter
            & selective_pattern_ok
            & or_retest_quality_ok
            & lowft_momentum_ok
            & v8_regime_ok
            & out["v12_core_setup_ok"]
            & out["v13_micro_quality_ok"]
            & out["v14_filter_ok"]
            & out["v15_quality_ok"]
            & (out["candidate_score"] >= params.min_candidate_score)
            & (out["support_volume"] | out["support_relative_strength"] | out["support_candle_pattern"] | is_mean_reversion)
        )

    # V35.8/V35.9 live/raw-bar quality gates.
    # These intentionally use only signal-bar values that are known at decision time.
    if bool(getattr(params, "enable_v358_live_quality_filter", False)):
        out = _apply_realtime_quality_gate(out, params, "v358", "v358_live_quality_ok")
    if bool(getattr(params, "enable_v359_live_hunter_filter", False)):
        out = _apply_realtime_quality_gate(out, params, "v359", "v359_live_hunter_ok")
    if bool(getattr(params, "enable_v364_professional_momentum_filter", False)):
        out = _apply_realtime_quality_gate(out, params, "v364", "v364_professional_momentum_ok")
    if bool(getattr(params, "enable_v377_positive_context_filter", False)):
        out = apply_positive_context_profile_filter(out, params)
    if bool(getattr(params, "enable_v379_decision_pattern_filter", False)):
        out = apply_decision_time_pattern_scorer(out, params)

    return out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def simulate_candidates(signal_df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    trades: list[dict[str, Any]] = []
    for symbol, group in signal_df.groupby("symbol", sort=False):
        g = group.sort_values("timestamp").reset_index(drop=True)
        alerts_taken_by_date: dict[Any, int] = {}
        for i, row in g.iterrows():
            if not bool(row.get("buy_alert", False)):
                continue
            session_date = row["session_date"]
            if alerts_taken_by_date.get(session_date, 0) >= params.max_alerts_per_symbol_per_day:
                continue
            trade = _simulate_trade(g, i, params)
            if trade is not None:
                trades.append(trade)
                alerts_taken_by_date[session_date] = alerts_taken_by_date.get(session_date, 0) + 1
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades).sort_values("entry_time").reset_index(drop=True)



def _simulate_v25_trade(g: pd.DataFrame, signal_i: int, params: StrategyParams) -> Optional[dict[str, Any]]:
    if signal_i + 1 >= len(g):
        return None
    signal = g.iloc[signal_i]
    next_bar = g.iloc[signal_i + 1]
    if next_bar["session_date"] != signal["session_date"]:
        return None
    side = str(signal.get("side", "long")).lower()
    if side not in {"long", "short"}:
        return None
    atr_value = float(signal.get("atr5m14", np.nan))
    raw_entry = float(next_bar["open"])
    entry_price = _buy_slippage(raw_entry, params.slippage_bps) if side == "long" else _sell_slippage(raw_entry, params.slippage_bps)
    if not np.isfinite(atr_value) or atr_value <= 0 or not np.isfinite(entry_price):
        return None
    risk_per_share = max(entry_price * float(getattr(params, "v25_min_stop_pct", 0.0015)), float(getattr(params, "v25_stop_atr_mult", 0.60)) * atr_value)
    if risk_per_share <= 0:
        return None
    target_r = float(getattr(params, "v25_target_r", 0.75))
    max_hold_bars = int(getattr(params, "v25_max_hold_bars", 12))
    if side == "long":
        stop_price = entry_price - risk_per_share
        target1 = entry_price + target_r * risk_per_share
    else:
        stop_price = entry_price + risk_per_share
        target1 = entry_price - target_r * risk_per_share
    target2 = target1
    entry_session = next_bar["session_date"]
    entry_index = signal_i + 1
    exit_time = next_bar["timestamp"]
    exit_reason = "end_of_data"
    high_since_entry = entry_price
    low_since_entry = entry_price
    max_favorable_price = entry_price
    max_adverse_price = entry_price
    pnl_per_share = 0.0
    target1_hit = False
    target2_hit = False

    last_close = float(next_bar["close"])
    for j in range(entry_index, min(len(g), entry_index + max_hold_bars)):
        bar = g.iloc[j]
        if bar["session_date"] != entry_session:
            prior = g.iloc[j - 1]
            exit_price = _exit_price(side, float(prior["close"]), params.slippage_bps)
            pnl_per_share = _pnl(side, entry_price, exit_price)
            exit_time = prior["timestamp"]
            exit_reason = "session_end"
            break
        high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])
        last_close = close
        high_since_entry = max(high_since_entry, high)
        low_since_entry = min(low_since_entry, low)
        if side == "long":
            max_favorable_price = max(max_favorable_price, high)
            max_adverse_price = min(max_adverse_price, low)
        else:
            max_favorable_price = min(max_favorable_price, low)
            max_adverse_price = max(max_adverse_price, high)
        # Conservative first-touch sequencing: stop before target if both touch in the same bar.
        if _stop_hit(side, high, low, stop_price):
            exit_price = _exit_price(side, stop_price, params.slippage_bps)
            pnl_per_share = _pnl(side, entry_price, exit_price)
            exit_time = bar["timestamp"]
            exit_reason = "v25_stop_first"
            break
        if _target_hit(side, high, low, target1):
            exit_price = _exit_price(side, target1, params.slippage_bps)
            pnl_per_share = _pnl(side, entry_price, exit_price)
            exit_time = bar["timestamp"]
            exit_reason = "v25_target_0_75r"
            target1_hit = True
            target2_hit = True
            break
    else:
        exit_price = _exit_price(side, last_close, params.slippage_bps)
        pnl_per_share = _pnl(side, entry_price, exit_price)
        last_i = min(len(g) - 1, entry_index + max_hold_bars - 1)
        exit_time = g.iloc[last_i]["timestamp"]
        exit_reason = "v25_time_exit"

    r_multiple = pnl_per_share / risk_per_share if risk_per_share > 0 else np.nan
    if side == "long":
        mfe_r = (max_favorable_price - entry_price) / risk_per_share
        mae_r = (max_adverse_price - entry_price) / risk_per_share
    else:
        mfe_r = (entry_price - max_favorable_price) / risk_per_share
        mae_r = (entry_price - max_adverse_price) / risk_per_share
    duration_minutes = (pd.Timestamp(exit_time) - pd.Timestamp(next_bar["timestamp"])).total_seconds() / 60
    return {
        "symbol": signal["symbol"],
        "side": side,
        "signal_time": signal["timestamp"],
        "signal_time_et": pd.Timestamp(signal["timestamp"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
        "entry_time": next_bar["timestamp"],
        "entry_time_et": pd.Timestamp(next_bar["timestamp"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
        "entry_hour_et": pd.Timestamp(next_bar["timestamp"]).tz_convert("America/New_York").strftime("%H:00"),
        "exit_time": exit_time,
        "exit_time_et": pd.Timestamp(exit_time).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
        "session_date": entry_session,
        "trigger_type": signal.get("trigger_type", ""),
        "setup_family": signal.get("setup_family", "v25_symbol_playbook"),
        "opportunity_module": signal.get("opportunity_module", "v25_symbol_playbook"),
        "module_risk_multiplier": 1.0,
        "module_max_hold_bars": max_hold_bars,
        "candidate_score": float(signal.get("candidate_score", np.nan)),
        "quality": signal.get("quality", "v25_profile_reaction"),
        "supporting_score": int(signal.get("supporting_score", 0)),
        "entry_candle_pattern": signal.get("candle_pattern_primary", ""),
        "entry_candle_ok": bool(signal.get("entry_candle_ok", True)),
        "opposing_candle_warning_at_entry": bool(signal.get("opposing_candle_warning", False)),
        "candle_pattern_score": float(signal.get("candle_pattern_score", 0.0)),
        "candle_pattern_mode": str(getattr(params, "candle_pattern_mode", "off")),
        "entry_trigger_price": entry_price,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target1": target1,
        "target2": target2,
        "risk_per_share": risk_per_share,
        "low_followthrough_mode": False,
        "pnl_per_share": pnl_per_share,
        "r_multiple": r_multiple,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "mfe_to_mae_ratio": abs(mfe_r / mae_r) if mae_r < 0 else np.nan,
        "target1_hit": target1_hit,
        "target2_hit": target2_hit,
        "exit_reason": exit_reason,
        "duration_minutes": duration_minutes,
        "gap_percent": signal.get("gap_percent"),
        "rvol_time_of_day": signal.get("rvol_time_of_day"),
        "day_relative_strength": signal.get("day_relative_strength"),
        "open_relative_strength": signal.get("open_relative_strength"),
        "stock_day_change_percent": signal.get("stock_day_change_percent"),
        "qqq_day_change_percent": signal.get("qqq_day_change_percent"),
        "vwap_extension_atr": signal.get("vwap_extension_atr"),
        "daily_atr14_percent": signal.get("daily_atr14_percent"),
        "v8_trade_context": "v25_symbol_playbook",
        "v8_regime_ok": True,
        "v8_candle_quality_ok": True,
        "v25_profile_filter": bool(signal.get("v25_profile_filter", False)),
        "v25_historical_gross_r": signal.get("v25_historical_gross_r", np.nan),
        "positive_context_profile_match": bool(signal.get("positive_context_profile_match", False)),
        "positive_context_profile_name": signal.get("positive_context_profile_name", ""),
        "positive_context_profile_reason": signal.get("positive_context_profile_reason", ""),
        "positive_context_dir_rs": signal.get("positive_context_dir_rs", np.nan),
        "positive_context_dir_open_rs": signal.get("positive_context_dir_open_rs", np.nan),
        "positive_context_dir_vwap": signal.get("positive_context_dir_vwap", np.nan),
        "positive_context_rvol_time_of_day": signal.get("positive_context_rvol_time_of_day", np.nan),
        "positive_context_daily_atr14_percent": signal.get("positive_context_daily_atr14_percent", np.nan),
        "positive_context_abs_gap": signal.get("positive_context_abs_gap", np.nan),
        "positive_context_abs_qqq": signal.get("positive_context_abs_qqq", np.nan),
        "positive_context_active_profile_count": signal.get("positive_context_active_profile_count", np.nan),
        "v379_pattern_mode": signal.get("v379_pattern_mode", ""),
        "v379_pattern_match": bool(signal.get("v379_pattern_match", False)),
        "v379_reason": signal.get("v379_reason", ""),
        "v379_pattern_score": signal.get("v379_pattern_score", np.nan),
        "v379_rank_score": signal.get("v379_rank_score", np.nan),
        "v379_original_score": signal.get("v379_original_score", np.nan),
        "v379_candle_component": signal.get("v379_candle_component", np.nan),
        "v379_rvol_component": signal.get("v379_rvol_component", np.nan),
        "v379_rs_component": signal.get("v379_rs_component", np.nan),
        "v379_vwap_component": signal.get("v379_vwap_component", np.nan),
        "v379_directional_rs": signal.get("v379_directional_rs", np.nan),
        "v379_directional_open_rs": signal.get("v379_directional_open_rs", np.nan),
        "v379_directional_vwap_atr": signal.get("v379_directional_vwap_atr", np.nan),
    }

def _simulate_trade(g: pd.DataFrame, signal_i: int, params: StrategyParams) -> Optional[dict[str, Any]]:
    if signal_i + 1 >= len(g):
        return None
    signal = g.iloc[signal_i]
    next_bar = g.iloc[signal_i + 1]
    atr_value = float(signal.get("atr5m14", np.nan))
    if not np.isfinite(atr_value) or atr_value <= 0:
        return None
    if next_bar["session_date"] != signal["session_date"]:
        return None

    if str(signal.get("trigger_type", "")).startswith("v25_"):
        return _simulate_v25_trade(g, signal_i, params)

    side = str(signal.get("side", "long")).lower()
    if side not in {"long", "short"}:
        return None
    setup_family = str(signal.get("setup_family", "trend_pullback"))
    trigger_type_value = str(signal.get("trigger_type", ""))
    candle_mode = str(getattr(params, "candle_pattern_mode", "off")).lower()
    candle_exit_enabled = candle_mode in {"score", "confirm", "selective", "exit_only"}

    signal_close = float(signal["close"])
    signal_high = float(signal["high"])
    signal_low = float(signal["low"])
    next_open = float(next_bar["open"])

    if side == "long":
        entry_trigger = max(signal_close, signal_high + params.entry_breakout_buffer_atr * atr_value)
        entry_zone_high = entry_trigger + params.max_entry_chase_atr * atr_value
        if next_open > entry_zone_high:
            return None
        if next_open >= entry_trigger:
            raw_entry = next_open
        elif float(next_bar["high"]) >= entry_trigger:
            raw_entry = entry_trigger
        else:
            return None
        entry_price = _buy_slippage(raw_entry, params.slippage_bps)
    else:
        entry_trigger = min(signal_close, signal_low - params.entry_breakout_buffer_atr * atr_value)
        entry_zone_low = entry_trigger - params.max_entry_chase_atr * atr_value
        if next_open < entry_zone_low:
            return None
        if next_open <= entry_trigger:
            raw_entry = next_open
        elif float(next_bar["low"]) <= entry_trigger:
            raw_entry = entry_trigger
        else:
            return None
        # For a short entry, slippage means we sell slightly lower than expected.
        entry_price = _sell_slippage(raw_entry, params.slippage_bps)

    trigger_level = float(signal.get("trigger_level", np.nan))
    if not np.isfinite(trigger_level):
        return None

    if side == "long":
        technical_stop = min(signal_low, trigger_level - params.stop_trigger_buffer_atr * atr_value)
        technical_risk = entry_price - technical_stop
    else:
        technical_stop = max(signal_high, trigger_level + params.stop_trigger_buffer_atr * atr_value)
        technical_risk = technical_stop - entry_price
    if technical_risk <= 0:
        return None
    risk_per_share = max(technical_risk, params.min_risk_atr * atr_value)
    if risk_per_share > params.max_risk_atr * atr_value:
        return None

    daily_atr_pct = float(signal.get("daily_atr14_percent", np.nan))
    low_followthrough_mode = (
        np.isfinite(daily_atr_pct)
        and daily_atr_pct < float(getattr(params, "low_followthrough_atr_pct", 4.0))
        and setup_family != "mean_reversion"
    )

    module_max_hold_bars = None
    module_risk_multiplier = 1.0
    if trigger_type_value.startswith("v18_m1"):
        t1_r = float(getattr(params, "m1_target1_r", 0.75)); t2_r = float(getattr(params, "m1_target2_r", 1.25))
        target1_size_param = float(getattr(params, "m1_target1_sell_pct", 0.70)); target2_size_param = float(getattr(params, "m1_target2_sell_pct", 0.30))
        breakeven_after_r = 0.40; module_max_hold_bars = int(getattr(params, "m1_max_hold_bars", 12)); module_risk_multiplier = float(getattr(params, "m1_risk_multiplier", 1.0))
    elif trigger_type_value.startswith("v18_m2"):
        t1_r = float(getattr(params, "m2_target1_r", 0.60)); t2_r = float(getattr(params, "m2_target2_r", 1.00))
        target1_size_param = float(getattr(params, "m2_target1_sell_pct", 0.80)); target2_size_param = float(getattr(params, "m2_target2_sell_pct", 0.20))
        breakeven_after_r = 0.35; module_max_hold_bars = int(getattr(params, "m2_max_hold_bars", 12)); module_risk_multiplier = float(getattr(params, "m2_risk_multiplier", 0.75))
    elif trigger_type_value.startswith("v18_m3"):
        t1_r = float(getattr(params, "m3_target1_r", 0.75)); t2_r = float(getattr(params, "m3_target2_r", 1.50))
        target1_size_param = float(getattr(params, "m3_target1_sell_pct", 0.60)); target2_size_param = float(getattr(params, "m3_target2_sell_pct", 0.40))
        breakeven_after_r = 0.40; module_max_hold_bars = int(getattr(params, "m3_max_hold_bars", 12)); module_risk_multiplier = float(getattr(params, "m3_risk_multiplier", 1.0))
    elif trigger_type_value.startswith("v18_m4"):
        t1_r = float(getattr(params, "m4_target1_r", 0.55)); t2_r = float(getattr(params, "m4_target2_r", 1.00))
        target1_size_param = float(getattr(params, "m4_target1_sell_pct", 0.80)); target2_size_param = float(getattr(params, "m4_target2_sell_pct", 0.20))
        breakeven_after_r = 0.30; module_max_hold_bars = int(getattr(params, "m4_max_hold_bars", 10)); module_risk_multiplier = float(getattr(params, "m4_risk_multiplier", 0.65))
    elif trigger_type_value.startswith("v18_m5"):
        t1_r = float(getattr(params, "m5_target1_r", 0.50)); t2_r = float(getattr(params, "m5_target2_r", 0.80))
        target1_size_param = float(getattr(params, "m5_target1_sell_pct", 1.00)); target2_size_param = float(getattr(params, "m5_target2_sell_pct", 0.00))
        breakeven_after_r = 0.25; module_max_hold_bars = int(getattr(params, "m5_max_hold_bars", 8)); module_risk_multiplier = float(getattr(params, "m5_risk_multiplier", 0.50))
    elif setup_family == "mean_reversion":
        t1_r = params.mr_target1_r
        t2_r = params.mr_target2_r
        target1_size_param = params.target1_sell_pct
        target2_size_param = params.target2_sell_pct
        breakeven_after_r = params.breakeven_after_r
    elif low_followthrough_mode:
        # V5 showed that simply shrinking targets in low-followthrough mode made
        # winners too small while losses stayed large. In V7 low-followthrough
        # momentum trades must pass the selective rejection gate first, then use
        # the normal momentum exit model.
        t1_r = params.target1_r
        t2_r = params.target2_r
        target1_size_param = params.target1_sell_pct
        target2_size_param = params.target2_sell_pct
        breakeven_after_r = params.breakeven_after_r
    else:
        t1_r = params.target1_r
        t2_r = params.target2_r
        target1_size_param = params.target1_sell_pct
        target2_size_param = params.target2_sell_pct
        breakeven_after_r = params.breakeven_after_r

    if side == "long":
        stop_price = entry_price - risk_per_share
        target1 = entry_price + t1_r * risk_per_share
        target2 = entry_price + t2_r * risk_per_share
    else:
        stop_price = entry_price + risk_per_share
        target1 = entry_price - t1_r * risk_per_share
        target2 = entry_price - t2_r * risk_per_share

    remaining = 1.0
    pnl_per_share = 0.0
    target1_hit = False
    target2_hit = False
    exit_reason = "end_of_data"
    exit_candle_pattern = ""
    exit_time = next_bar["timestamp"]
    high_since_entry = entry_price
    low_since_entry = entry_price
    max_favorable_price = entry_price
    max_adverse_price = entry_price
    stop_active = stop_price
    entry_session = next_bar["session_date"]
    entry_index = signal_i + 1
    target1_size = min(max(target1_size_param, 0.0), 1.0)
    target2_size = min(max(target2_size_param, 0.0), 1.0 - target1_size)
    first_exit_index = entry_index + 1 if params.skip_entry_bar_exits else entry_index

    for j in range(first_exit_index, len(g)):
        bar = g.iloc[j]
        if bar["session_date"] != entry_session:
            prior = g.iloc[j - 1]
            exit_price = _exit_price(side, float(prior["close"]), params.slippage_bps)
            pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
            remaining = 0.0
            exit_time = prior["timestamp"]
            exit_reason = "session_end"
            break

        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        high_since_entry = max(high_since_entry, high)
        low_since_entry = min(low_since_entry, low)
        if side == "long":
            max_favorable_price = max(max_favorable_price, high)
            max_adverse_price = min(max_adverse_price, low)
        else:
            max_favorable_price = min(max_favorable_price, low)
            max_adverse_price = max(max_adverse_price, high)
        bars_elapsed = j - entry_index + 1

        # Conservative intrabar assumption: stops are processed before targets when both occur in same 5-min bar.
        if remaining > 0 and _stop_hit(side, high, low, stop_active):
            exit_price = _exit_price(side, stop_active, params.slippage_bps)
            pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
            remaining = 0.0
            exit_time = bar["timestamp"]
            exit_reason = "stop_loss" if not target1_hit else "breakeven_or_trailing_stop"
            break

        if remaining > 0 and (not target1_hit):
            ema9_now = float(bar.get("ema9", np.nan))
            soft_r = float(getattr(params, "soft_failure_stop_r", 0.65))
            if side == "long":
                soft_fail = (close <= entry_price - soft_r * risk_per_share) and (close < trigger_level or (np.isfinite(ema9_now) and close < ema9_now))
            else:
                soft_fail = (close >= entry_price + soft_r * risk_per_share) and (close > trigger_level or (np.isfinite(ema9_now) and close > ema9_now))
            if soft_fail:
                exit_price = _exit_price(side, close, params.slippage_bps)
                pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
                remaining = 0.0
                exit_time = bar["timestamp"]
                exit_reason = "soft_failure_stop"
                break

        if remaining > 0 and (not target1_hit) and _target_hit(side, high, low, target1):
            exit_price = _exit_price(side, target1, params.slippage_bps)
            pnl_per_share += target1_size * _pnl(side, entry_price, exit_price)
            remaining -= target1_size
            target1_hit = True
            stop_active = _breakeven_stop(side, entry_price, params.slippage_bps, current_stop=stop_active)

        if remaining > 0 and (not target1_hit) and _favorable_reached(side, high_since_entry, low_since_entry, entry_price, float(getattr(params, "protective_stop_after_r", 0.35)) * risk_per_share):
            # Once the trade proves a small favorable move, do not allow it to fall all the way
            # back to the original full stop. This specifically targets the long-report failure
            # mode: trades first move +0.3R..+0.5R, then reverse into a large loss.
            protective_r = float(getattr(params, "protective_stop_r", -0.12))
            if side == "long":
                protective_stop = entry_price + protective_r * risk_per_share
                stop_active = max(stop_active, protective_stop)
            else:
                protective_stop = entry_price - protective_r * risk_per_share
                stop_active = min(stop_active, protective_stop)

        if remaining > 0 and (not target1_hit) and _favorable_reached(side, high_since_entry, low_since_entry, entry_price, breakeven_after_r * risk_per_share):
            stop_active = _breakeven_stop(side, entry_price, params.slippage_bps, current_stop=stop_active)

        if remaining > 0 and target1_hit and (not target2_hit) and _target_hit(side, high, low, target2):
            exit_price = _exit_price(side, target2, params.slippage_bps)
            pnl_per_share += target2_size * _pnl(side, entry_price, exit_price)
            remaining -= target2_size
            target2_hit = True

        if remaining > 0 and target1_hit:
            atr_now = float(bar.get("atr5m14", atr_value)) if np.isfinite(bar.get("atr5m14", atr_value)) else atr_value
            if side == "long":
                trailing_stop = high_since_entry - params.trailing_atr_multiple * atr_now
                stop_active = max(stop_active, trailing_stop)
            else:
                trailing_stop = low_since_entry + params.trailing_atr_multiple * atr_now
                stop_active = min(stop_active, trailing_stop)
            if _stop_hit(side, high, low, stop_active):
                exit_price = _exit_price(side, stop_active, params.slippage_bps)
                pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
                remaining = 0.0
                exit_time = bar["timestamp"]
                exit_reason = "atr_trailing_stop" if target2_hit else "breakeven_or_trailing_stop"
                break

        # Candlestick-pattern exit. We do not exit just because a pattern has a
        # classic name. We only use warning candles after the trade has made a
        # minimum favorable move or after target 1 is already hit. This helps lock
        # intraday scalps when a rejection/engulfing candle says momentum is fading.
        if remaining > 0 and candle_exit_enabled:
            warning_col = "long_exit_warning_candle" if side == "long" else "short_exit_warning_candle"
            candle_warning = bool(bar.get(warning_col, False))
            favorable_for_exit = _favorable_reached(
                side,
                high_since_entry,
                low_since_entry,
                entry_price,
                float(getattr(params, "candle_exit_min_mfe_r", 0.20)) * risk_per_share,
            )
            if candle_warning and (target1_hit or favorable_for_exit):
                exit_price = _exit_price(side, close, params.slippage_bps)
                pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
                remaining = 0.0
                exit_time = bar["timestamp"]
                exit_reason = "candlestick_reversal_exit"
                exit_candle_pattern = str(bar.get("candle_pattern_primary", "warning_candle"))
                break

        if remaining > 0 and (not target1_hit) and bars_elapsed >= params.early_breakdown_candles:
            ema9 = float(bar.get("ema9", np.nan))
            if side == "long" and close < trigger_level and np.isfinite(ema9) and close < ema9:
                exit_price = _exit_price(side, close, params.slippage_bps)
                pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
                remaining = 0.0
                exit_time = bar["timestamp"]
                exit_reason = "early_breakdown"
                break
            if side == "short" and close > trigger_level and np.isfinite(ema9) and close > ema9:
                exit_price = _exit_price(side, close, params.slippage_bps)
                pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
                remaining = 0.0
                exit_time = bar["timestamp"]
                exit_reason = "early_breakdown"
                break

        if remaining > 0 and (not target1_hit) and bars_elapsed >= params.failure_candles:
            if not _favorable_reached(side, high_since_entry, low_since_entry, entry_price, params.failure_min_r * risk_per_share):
                exit_price = _exit_price(side, close, params.slippage_bps)
                pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
                remaining = 0.0
                exit_time = bar["timestamp"]
                exit_reason = "momentum_failure"
                break

        if remaining > 0 and setup_family == "mean_reversion" and bars_elapsed >= params.mr_max_hold_bars:
            exit_price = _exit_price(side, close, params.slippage_bps)
            pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
            remaining = 0.0
            exit_time = bar["timestamp"]
            exit_reason = "mean_reversion_time_exit"
            break

        if remaining > 0 and module_max_hold_bars is not None and bars_elapsed >= int(module_max_hold_bars):
            exit_price = _exit_price(side, close, params.slippage_bps)
            pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
            remaining = 0.0
            exit_time = bar["timestamp"]
            exit_reason = "module_time_exit"
            break

        if remaining > 0 and str(bar["time_str"]) >= params.exit_all_time:
            exit_price = _exit_price(side, close, params.slippage_bps)
            pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
            remaining = 0.0
            exit_time = bar["timestamp"]
            exit_reason = "time_exit"
            break

    if remaining > 0:
        last = g[g["session_date"] == entry_session].iloc[-1]
        exit_price = _exit_price(side, float(last["close"]), params.slippage_bps)
        pnl_per_share += remaining * _pnl(side, entry_price, exit_price)
        exit_time = last["timestamp"]
        exit_reason = "end_of_session_fallback"

    r_multiple = pnl_per_share / risk_per_share if risk_per_share > 0 else np.nan
    if side == "long":
        mfe_r = (max_favorable_price - entry_price) / risk_per_share
        mae_r = (max_adverse_price - entry_price) / risk_per_share
    else:
        mfe_r = (entry_price - max_favorable_price) / risk_per_share
        mae_r = (entry_price - max_adverse_price) / risk_per_share
    duration_minutes = (pd.Timestamp(exit_time) - pd.Timestamp(next_bar["timestamp"])).total_seconds() / 60
    return {
        "symbol": signal["symbol"],
        "side": side,
        "signal_time": signal["timestamp"],
        "signal_time_et": pd.Timestamp(signal["timestamp"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
        "entry_time": next_bar["timestamp"],
        "entry_time_et": pd.Timestamp(next_bar["timestamp"]).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
        "entry_hour_et": pd.Timestamp(next_bar["timestamp"]).tz_convert("America/New_York").strftime("%H:00"),
        "exit_time": exit_time,
        "exit_time_et": pd.Timestamp(exit_time).tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M"),
        "session_date": entry_session,
        "trigger_type": signal.get("trigger_type", ""),
        "setup_family": signal.get("setup_family", ""),
        "opportunity_module": signal.get("opportunity_module", "legacy"),
        "module_risk_multiplier": float(module_risk_multiplier),
        "module_max_hold_bars": int(module_max_hold_bars) if module_max_hold_bars is not None else None,
        "candidate_score": float(signal.get("candidate_score", np.nan)),
        "quality": signal.get("quality", ""),
        "supporting_score": int(signal.get("supporting_score", 0)),
        "entry_candle_pattern": signal.get("candle_pattern_primary", ""),
        "entry_candle_ok": bool(signal.get("entry_candle_ok", False)),
        "opposing_candle_warning_at_entry": bool(signal.get("opposing_candle_warning", False)),
        "candle_pattern_score": float(signal.get("candle_pattern_score", 0.0)),
        "candle_pattern_mode": candle_mode,
        "exit_candle_pattern": exit_candle_pattern,
        "entry_trigger_price": entry_trigger,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target1": target1,
        "target2": target2,
        "risk_per_share": risk_per_share,
        "low_followthrough_mode": bool(low_followthrough_mode),
        "pnl_per_share": pnl_per_share,
        "r_multiple": r_multiple,
        "mfe_r": mfe_r,
        "mae_r": mae_r,
        "mfe_to_mae_ratio": abs(mfe_r / mae_r) if mae_r < 0 else np.nan,
        "target1_hit": target1_hit,
        "target2_hit": target2_hit,
        "exit_reason": exit_reason,
        "duration_minutes": duration_minutes,
        "gap_percent": signal.get("gap_percent"),
        "rvol_time_of_day": signal.get("rvol_time_of_day"),
        "day_relative_strength": signal.get("day_relative_strength"),
        "open_relative_strength": signal.get("open_relative_strength"),
        "stock_day_change_percent": signal.get("stock_day_change_percent"),
        "qqq_day_change_percent": signal.get("qqq_day_change_percent"),
        "vwap_extension_atr": signal.get("vwap_extension_atr"),
        "daily_atr14_percent": signal.get("daily_atr14_percent"),
        "v8_trade_context": signal.get("v8_trade_context", ""),
        "v8_regime_ok": bool(signal.get("v8_regime_ok", True)),
        "v8_candle_quality_ok": bool(signal.get("v8_candle_quality_ok", True)),
        "positive_context_profile_match": bool(signal.get("positive_context_profile_match", False)),
        "positive_context_profile_name": signal.get("positive_context_profile_name", ""),
        "positive_context_profile_reason": signal.get("positive_context_profile_reason", ""),
        "positive_context_dir_rs": signal.get("positive_context_dir_rs", np.nan),
        "positive_context_dir_open_rs": signal.get("positive_context_dir_open_rs", np.nan),
        "positive_context_dir_vwap": signal.get("positive_context_dir_vwap", np.nan),
        "positive_context_rvol_time_of_day": signal.get("positive_context_rvol_time_of_day", np.nan),
        "positive_context_daily_atr14_percent": signal.get("positive_context_daily_atr14_percent", np.nan),
        "positive_context_abs_gap": signal.get("positive_context_abs_gap", np.nan),
        "positive_context_abs_qqq": signal.get("positive_context_abs_qqq", np.nan),
        "positive_context_active_profile_count": signal.get("positive_context_active_profile_count", np.nan),
        "v379_pattern_mode": signal.get("v379_pattern_mode", ""),
        "v379_pattern_match": bool(signal.get("v379_pattern_match", False)),
        "v379_reason": signal.get("v379_reason", ""),
        "v379_pattern_score": signal.get("v379_pattern_score", np.nan),
        "v379_rank_score": signal.get("v379_rank_score", np.nan),
        "v379_original_score": signal.get("v379_original_score", np.nan),
        "v379_candle_component": signal.get("v379_candle_component", np.nan),
        "v379_rvol_component": signal.get("v379_rvol_component", np.nan),
        "v379_rs_component": signal.get("v379_rs_component", np.nan),
        "v379_vwap_component": signal.get("v379_vwap_component", np.nan),
        "v379_directional_rs": signal.get("v379_directional_rs", np.nan),
        "v379_directional_open_rs": signal.get("v379_directional_open_rs", np.nan),
        "v379_directional_vwap_atr": signal.get("v379_directional_vwap_atr", np.nan),
    }



def _strategy_calculate_risk_budget(equity: float, high_watermark: float, params: StrategyParams) -> tuple[float, float, float, bool, str]:
    """Universal position sizing for every strategy/backtest path.

    Returns (risk_budget, effective_risk_pct, drawdown_pct, paused, sizing_mode).
    This intentionally mirrors the dashboard sizing modes so presets never hide
    risk/compounding behavior.
    """
    equity = float(equity or 0.0)
    high_watermark = max(float(high_watermark or equity or 0.0), equity)
    mode = str(getattr(params, "position_sizing_mode", "fixed_dollar_risk") or "fixed_dollar_risk").lower()
    if equity <= 0:
        return 0.0, 0.0, 100.0, True, mode

    drawdown_pct = max(0.0, (high_watermark - equity) / high_watermark * 100.0) if high_watermark > 0 else 0.0

    if mode in {"fixed", "fixed_dollar", "fixed_dollar_risk", "fixed_dollar_risk_exact"}:
        risk = float(getattr(params, "risk_per_trade_dollars", 100.0) or 100.0)
        risk = max(0.0, risk)
        pct = risk / equity * 100.0 if equity > 0 else 0.0
        return risk, pct, drawdown_pct, False, "fixed_dollar_risk"

    if mode in {"percent", "percent_equity", "full_compounding", "percent_equity_full_compounding"}:
        # The dashboard passes Base risk % into both requested_risk_percent and
        # risk_per_trade_pct. Prefer the decimal field but fall back safely.
        pct_decimal = float(getattr(params, "risk_per_trade_pct", 0.01) or 0.01)
        if pct_decimal > 1.0:
            pct_decimal = pct_decimal / 100.0
        risk = max(0.0, equity * pct_decimal)
        pct = risk / equity * 100.0 if equity > 0 else 0.0
        return risk, pct, drawdown_pct, False, "percent_equity"

    # Controlled compounding: current equity * base risk %, then optional drawdown
    # brakes, dollar min/max caps, and maximum % of equity cap.
    pause_dd = float(getattr(params, "compounding_pause_dd_pct", 15.0) or 15.0)
    if drawdown_pct >= pause_dd:
        return 0.0, 0.0, drawdown_pct, True, "controlled_compounding"

    pct = float(getattr(params, "compounding_base_risk_pct", 1.0) or 1.0)
    dd1 = float(getattr(params, "compounding_dd1_pct", 5.0) or 5.0)
    dd2 = float(getattr(params, "compounding_dd2_pct", 10.0) or 10.0)
    if drawdown_pct >= dd2:
        pct = float(getattr(params, "compounding_dd2_risk_pct", 0.50) or 0.50)
    elif drawdown_pct >= dd1:
        pct = float(getattr(params, "compounding_dd1_risk_pct", 0.75) or 0.75)

    risk = equity * pct / 100.0
    min_risk = float(getattr(params, "compounding_min_risk_dollars", 10.0) or 0.0)
    max_risk = float(getattr(params, "compounding_max_risk_dollars", 300.0) or 0.0)
    max_pct = float(getattr(params, "compounding_max_risk_pct_of_equity", 1.25) or 0.0)
    if min_risk > 0:
        risk = max(risk, min_risk)
    if max_risk > 0:
        risk = min(risk, max_risk)
    if max_pct > 0:
        risk = min(risk, equity * max_pct / 100.0)
    risk = max(0.0, risk)
    effective_pct = risk / equity * 100.0 if equity > 0 else 0.0
    return risk, effective_pct, drawdown_pct, False, "controlled_compounding"


def _strategy_high_watermark_before_trade(selected: list[dict[str, Any]], initial_equity: float, entry_time: pd.Timestamp) -> float:
    """Return the maximum closed-equity value known before this entry."""
    high = float(initial_equity or 0.0)
    closed = []
    for t in selected:
        try:
            if pd.Timestamp(t.get("exit_time")) <= entry_time:
                closed.append(t)
        except Exception:
            continue
    if not closed:
        return high
    closed = sorted(closed, key=lambda x: pd.Timestamp(x.get("exit_time")))
    equity = float(initial_equity or 0.0)
    for t in closed:
        equity += float(t.get("pnl_dollars", 0.0) or 0.0)
        high = max(high, equity)
    return high

def apply_portfolio_rules(candidates: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    decision_mode = str(getattr(params, "backtest_decision_mode", "end_of_day_top_n") or "end_of_day_top_n").lower()
    if str(getattr(params, "strategy_profile", "")).lower() == "symbol_playbook_v25" and decision_mode not in {"live_simulated", "live", "walk_forward", "seen_so_far_top_n", "raw_bar_replay", "full_raw_bar_replay", "historical_raw_replay", "raw_replay"}:
        # V25 research ranking selected top N candidates by score per session, not the earliest chronological candidates.
        # Live-simulation mode preselects chronologically in backtest.py before this function.
        candidates = (
            candidates.sort_values(["session_date", "candidate_score", "entry_time"], ascending=[True, False, True])
            .groupby("session_date", group_keys=False)
            .head(int(getattr(params, "max_trades_per_day", 2)))
            .sort_values("entry_time")
            .reset_index(drop=True)
        )

    selected: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []

    for _, cand in candidates.sort_values(["entry_time", "candidate_score"], ascending=[True, False]).iterrows():
        entry_time = pd.Timestamp(cand["entry_time"])
        entry_date = cand["session_date"]
        closed_before = [t for t in selected if pd.Timestamp(t["exit_time"]) <= entry_time]
        equity_at_entry = params.initial_account_value + sum(float(t["pnl_dollars"]) for t in closed_before)
        realized_today = sum(
            float(t["pnl_dollars"])
            for t in closed_before
            if pd.Timestamp(t["exit_time"]).tz_convert("America/New_York").date() == entry_date
        )
        taken_today = sum(1 for t in selected if t["session_date"] == entry_date)
        open_now = [t for t in selected if pd.Timestamp(t["entry_time"]) <= entry_time < pd.Timestamp(t["exit_time"])]
        closed_today = sorted(
            [
                t for t in closed_before
                if pd.Timestamp(t["exit_time"]).tz_convert("America/New_York").date() == entry_date
            ],
            key=lambda x: pd.Timestamp(x["exit_time"]),
        )
        latest_closed = closed_today[-params.max_consecutive_losses :]
        consecutive_losses = len(latest_closed) >= params.max_consecutive_losses and all(float(t["pnl_dollars"]) < 0 for t in latest_closed)

        skip_reason = None
        if equity_at_entry <= 0:
            skip_reason = "equity_depleted"
        elif taken_today >= params.max_trades_per_day:
            skip_reason = "max_trades_per_day"
        elif len(open_now) >= params.max_open_positions:
            skip_reason = "max_open_positions"
        elif realized_today <= -(params.initial_account_value * params.daily_loss_limit_pct):
            skip_reason = "daily_loss_limit"
        elif consecutive_losses:
            skip_reason = "consecutive_loss_limit_today"

        if skip_reason:
            row = cand.to_dict()
            row["selected"] = False
            row["skip_reason"] = skip_reason
            skipped_rows.append(row)
            continue

        high_watermark_before_trade = _strategy_high_watermark_before_trade(selected, float(params.initial_account_value), entry_time)
        base_risk_budget, effective_risk_pct, drawdown_before_trade_pct, sizing_paused, sizing_mode = _strategy_calculate_risk_budget(
            equity_at_entry, high_watermark_before_trade, params
        )
        module_risk_multiplier = float(cand.get("module_risk_multiplier", 1.0) if hasattr(cand, "get") else 1.0)
        risk_budget = base_risk_budget * module_risk_multiplier
        entry_price = float(cand["entry_price"])
        risk_per_share = float(cand["risk_per_share"])

        if sizing_paused or risk_budget <= 0:
            row = cand.to_dict()
            row["selected"] = False
            row["skip_reason"] = "sizing_paused_or_zero_risk"
            row["equity_at_entry"] = equity_at_entry
            row["risk_budget"] = risk_budget
            row["base_risk_per_trade_dollars"] = base_risk_budget
            row["module_risk_multiplier"] = module_risk_multiplier
            row["risk_per_trade_dollars"] = risk_budget
            row["requested_risk_percent"] = float(getattr(params, "requested_risk_percent", effective_risk_pct) or effective_risk_pct)
            row["engine_risk_per_trade_pct"] = effective_risk_pct / 100.0
            row["risk_per_trade_percent"] = effective_risk_pct
            row["effective_risk_pct"] = effective_risk_pct
            row["drawdown_before_trade_pct"] = drawdown_before_trade_pct
            row["compounding_high_watermark_before_trade"] = high_watermark_before_trade
            row["position_sizing_mode"] = sizing_mode
            skipped_rows.append(row)
            continue

        # Risk is applied to the trade itself, not only to reporting.
        # Fractional shares are used in the backtest so all sizing modes scale
        # P&L exactly according to the trade's R multiple.
        requested_risk_shares = (risk_budget / risk_per_share) if risk_per_share > 0 else 0.0
        cash_shares = 0.0
        shares = requested_risk_shares
        sizing_cap_applied = False

        if shares <= 0 or shares < int(getattr(params, "min_shares", 1)):
            row = cand.to_dict()
            row["selected"] = False
            row["skip_reason"] = "position_size_zero"
            row["equity_at_entry"] = equity_at_entry
            row["risk_budget"] = risk_budget
            row["base_risk_per_trade_dollars"] = base_risk_budget
            row["module_risk_multiplier"] = module_risk_multiplier
            row["risk_per_trade_dollars"] = risk_budget
            row["requested_risk_percent"] = float(getattr(params, "requested_risk_percent", effective_risk_pct) or effective_risk_pct)
            row["engine_risk_per_trade_pct"] = effective_risk_pct / 100.0
            row["risk_per_trade_percent"] = effective_risk_pct
            row["effective_risk_pct"] = effective_risk_pct
            row["drawdown_before_trade_pct"] = drawdown_before_trade_pct
            row["compounding_high_watermark_before_trade"] = high_watermark_before_trade
            row["position_sizing_mode"] = sizing_mode
            row["max_position_notional_pct"] = float(getattr(params, "max_position_notional_pct", 999.0))
            row["requested_risk_shares"] = requested_risk_shares
            row["cash_cap_shares"] = cash_shares
            row["sizing_cap_applied"] = sizing_cap_applied
            skipped_rows.append(row)
            continue

        row = cand.to_dict()
        row["selected"] = True
        row["skip_reason"] = ""
        row["equity_at_entry"] = equity_at_entry
        row["risk_budget"] = risk_budget
        row["base_risk_per_trade_dollars"] = base_risk_budget
        row["module_risk_multiplier"] = module_risk_multiplier
        row["risk_per_trade_dollars"] = risk_budget
        row["requested_risk_percent"] = float(getattr(params, "requested_risk_percent", effective_risk_pct) or effective_risk_pct)
        row["engine_risk_per_trade_pct"] = effective_risk_pct / 100.0
        row["risk_per_trade_percent"] = effective_risk_pct
        row["effective_risk_pct"] = effective_risk_pct
        row["drawdown_before_trade_pct"] = drawdown_before_trade_pct
        row["compounding_high_watermark_before_trade"] = high_watermark_before_trade
        row["position_sizing_mode"] = sizing_mode
        row["max_position_notional_pct"] = float(getattr(params, "max_position_notional_pct", 999.0))
        row["requested_risk_shares"] = requested_risk_shares
        row["cash_cap_shares"] = cash_shares
        row["sizing_cap_applied"] = sizing_cap_applied
        row["shares"] = shares
        row["notional"] = shares * entry_price
        row["notional_pct_of_equity"] = (row["notional"] / equity_at_entry * 100) if equity_at_entry > 0 else np.nan
        row["actual_dollars_at_risk"] = shares * risk_per_share
        row["actual_risk_pct_of_equity"] = (row["actual_dollars_at_risk"] / equity_at_entry * 100) if equity_at_entry > 0 else np.nan
        # Force the dollar result to equal R-multiple * requested risk.
        # This makes the risk control auditable and avoids hidden rounding/cap effects.
        row["pnl_dollars"] = float(row["r_multiple"]) * risk_budget
        row["pnl_dollars_from_shares"] = shares * float(row["pnl_per_share"])
        row["risk_application_delta"] = float(row["pnl_dollars"]) - float(row["pnl_dollars_from_shares"])
        row["return_on_risk_pct"] = float(row["pnl_dollars"]) / risk_budget * 100 if risk_budget > 0 else np.nan
        row["return_on_actual_risk_pct"] = float(row["pnl_dollars"]) / row["actual_dollars_at_risk"] * 100 if row["actual_dollars_at_risk"] > 0 else np.nan
        selected.append(row)

    combined = selected + skipped_rows
    if not combined:
        return pd.DataFrame()
    out = pd.DataFrame(combined).sort_values("entry_time").reset_index(drop=True)
    return out




def _build_sizing_scenarios(selected: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Show how the exact same selected trades scale at different fixed risk amounts.

    This is intentionally independent of the Dash input. If the UI ever fails to
    pass a new risk value, this table still proves what $25/$50/$100/$200/$500
    would have produced from the same R-multiple series.
    """
    if selected is None or selected.empty or "r_multiple" not in selected.columns:
        return pd.DataFrame(columns=["risk_dollars", "total_pnl", "total_return_pct", "max_drawdown_pct", "ending_equity", "avg_pnl_per_trade"])
    r = pd.to_numeric(selected["r_multiple"], errors="coerce").fillna(0.0).reset_index(drop=True)
    base_risk = float(getattr(params, "risk_per_trade_dollars", 100.0))
    scenario_values = [25.0, 50.0, 100.0, 200.0, 500.0]
    if base_risk not in scenario_values:
        scenario_values.append(base_risk)
    scenario_values = sorted(set(scenario_values))
    rows = []
    for risk in scenario_values:
        pnl = r * risk
        equity = float(params.initial_account_value) + pnl.cumsum()
        peak = equity.cummax()
        dd_pct = ((equity - peak) / peak * 100.0).min() if len(equity) else 0.0
        rows.append({
            "risk_dollars": risk,
            "total_trades": int(len(r)),
            "total_pnl": float(pnl.sum()),
            "total_return_pct": float((equity.iloc[-1] / float(params.initial_account_value) - 1.0) * 100.0) if len(equity) else 0.0,
            "max_drawdown_pct": float(dd_pct),
            "ending_equity": float(equity.iloc[-1]) if len(equity) else float(params.initial_account_value),
            "avg_pnl_per_trade": float(pnl.mean()) if len(pnl) else 0.0,
        })
    return pd.DataFrame(rows)

def summarize_results(trades: pd.DataFrame, params: StrategyParams) -> dict[str, Any]:
    selected = trades[trades.get("selected", False) == True].copy() if not trades.empty else pd.DataFrame()
    if selected.empty:
        return {
            "metrics": _empty_metrics(params),
            "selected_trades": selected,
            "symbol_summary": pd.DataFrame(),
            "setup_summary": pd.DataFrame(),
            "daily_summary": pd.DataFrame(),
            "exit_summary": pd.DataFrame(),
            "score_band_summary": pd.DataFrame(),
            "hour_summary": pd.DataFrame(),
            "direction_summary": pd.DataFrame(),
            "module_summary": pd.DataFrame(),
            "candle_summary": pd.DataFrame(),
            "regime_summary": pd.DataFrame(),
            "market_context_summary": pd.DataFrame(),
            "sizing_scenarios": _build_sizing_scenarios(selected, params),
            "equity_curve": pd.DataFrame(),
            "drawdown_curve": pd.DataFrame(),
        }

    selected = selected.sort_values("exit_time").reset_index(drop=True)
    selected["pnl_dollars"] = pd.to_numeric(selected["pnl_dollars"], errors="coerce").fillna(0.0)
    selected["equity"] = params.initial_account_value + selected["pnl_dollars"].cumsum()
    selected["running_peak"] = selected["equity"].cummax()
    selected["drawdown"] = selected["equity"] - selected["running_peak"]
    selected["drawdown_pct"] = selected["drawdown"] / selected["running_peak"] * 100
    selected["score_band"] = pd.cut(
        selected["candidate_score"],
        bins=[0, 60, 70, 80, 90, 101],
        labels=["<60", "60-69", "70-79", "80-89", "90+"],
        right=False,
    ).astype(str)

    wins = selected[selected["pnl_dollars"] > 0]
    losses = selected[selected["pnl_dollars"] < 0]
    total_profit = wins["pnl_dollars"].sum()
    total_loss = losses["pnl_dollars"].sum()
    profit_factor = (total_profit / abs(total_loss)) if total_loss < 0 else np.inf
    win_rate = len(wins) / len(selected) * 100 if len(selected) else 0
    total_return_pct = (selected["equity"].iloc[-1] / params.initial_account_value - 1) * 100
    expectancy_r = selected["r_multiple"].mean()
    avg_win_r = wins["r_multiple"].mean() if len(wins) else 0
    avg_loss_r = losses["r_multiple"].mean() if len(losses) else 0
    max_drawdown_pct = selected["drawdown_pct"].min()
    trading_days_with_trades = selected["session_date"].nunique()
    trades_per_trade_day = len(selected) / trading_days_with_trades if trading_days_with_trades else 0

    metrics = {
        "total_trades": int(len(selected)),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else 999.0,
        "total_pnl": float(selected["pnl_dollars"].sum()),
        "total_return_pct": float(total_return_pct),
        "expectancy_r": float(expectancy_r),
        "avg_win_r": float(avg_win_r),
        "avg_loss_r": float(avg_loss_r),
        "max_drawdown_pct": float(max_drawdown_pct),
        "avg_duration_minutes": float(selected["duration_minutes"].mean()),
        "target1_hit_rate": float(selected["target1_hit"].mean() * 100),
        "target2_hit_rate": float(selected["target2_hit"].mean() * 100),
        "avg_mfe_r": float(selected["mfe_r"].mean()),
        "avg_mae_r": float(selected["mae_r"].mean()),
        "initial_account_value": params.initial_account_value,
        "ending_equity": float(selected["equity"].iloc[-1]),
        "trading_days_with_trades": int(trading_days_with_trades),
        "trade_days": int(trading_days_with_trades),
        "trades_per_trade_day": float(trades_per_trade_day),
        "avg_candidate_score": float(selected["candidate_score"].mean()) if "candidate_score" in selected.columns else 0.0,
        # Average applied risk is the real engine risk for the selected trades.
        # In compounding modes this changes trade-by-trade, so do not report the
        # fixed-dollar UI input as the active engine risk.
        "risk_per_trade_dollars": float(selected["risk_budget"].mean()) if "risk_budget" in selected.columns else float(getattr(params, "risk_per_trade_dollars", params.initial_account_value * params.risk_per_trade_pct)),
        "requested_risk_dollars": float(getattr(params, "risk_per_trade_dollars", 0.0)),
        "requested_risk_percent": float(getattr(params, "requested_risk_percent", params.risk_per_trade_pct * 100)),
        "risk_per_trade_pct": float(params.risk_per_trade_pct),
        "risk_per_trade_percent": float(selected["risk_per_trade_percent"].mean()) if "risk_per_trade_percent" in selected.columns else float(params.risk_per_trade_pct * 100),
        "position_sizing_mode": str(getattr(params, "position_sizing_mode", "fixed_dollar_risk")),
        "max_position_notional_pct": float(getattr(params, "max_position_notional_pct", 999.0)),
        "avg_risk_budget": float(selected["risk_budget"].mean()) if "risk_budget" in selected.columns else 0.0,
        "avg_actual_dollars_at_risk": float(selected["actual_dollars_at_risk"].mean()) if "actual_dollars_at_risk" in selected.columns else 0.0,
        "avg_actual_risk_pct_of_equity": float(selected["actual_risk_pct_of_equity"].mean()) if "actual_risk_pct_of_equity" in selected.columns else 0.0,
        "avg_notional": float(selected["notional"].mean()) if "notional" in selected.columns else 0.0,
        "median_notional": float(selected["notional"].median()) if "notional" in selected.columns else 0.0,
        "avg_notional_pct": float(selected["notional_pct_of_equity"].mean()) if "notional_pct_of_equity" in selected.columns else 0.0,
        "sizing_cap_applied_trades": int(selected["sizing_cap_applied"].fillna(False).astype(bool).sum()) if "sizing_cap_applied" in selected.columns else 0,
    }

    symbol_summary = _group_summary(selected, "symbol", include_duration=True).sort_values(["total_pnl", "win_rate"], ascending=[False, False])
    setup_summary = _group_summary(selected, "trigger_type").sort_values(["total_pnl", "win_rate"], ascending=[False, False])
    direction_summary = _group_summary(selected, "side").sort_values(["total_pnl", "win_rate"], ascending=[False, False])
    module_summary = (
        _group_summary(selected, "opportunity_module").sort_values(["total_pnl", "win_rate"], ascending=[False, False])
        if "opportunity_module" in selected.columns
        else pd.DataFrame()
    )
    candle_summary = (
        _group_summary(selected, "entry_candle_pattern").sort_values(["total_pnl", "win_rate"], ascending=[False, False])
        if "entry_candle_pattern" in selected.columns
        else pd.DataFrame()
    )
    regime_summary = (
        _group_summary(selected, "low_followthrough_mode").sort_values(["total_pnl", "win_rate"], ascending=[False, False])
        if "low_followthrough_mode" in selected.columns
        else pd.DataFrame()
    )
    market_context_summary = (
        _group_summary(selected, "v8_trade_context").sort_values(["total_pnl", "win_rate"], ascending=[False, False])
        if "v8_trade_context" in selected.columns
        else pd.DataFrame()
    )
    exit_summary = _group_summary(selected, "exit_reason").sort_values(["total_pnl", "win_rate"], ascending=[False, False])
    score_band_summary = _group_summary(selected, "score_band").sort_values("score_band")
    hour_summary = _group_summary(selected, "entry_hour_et").sort_values("entry_hour_et")

    daily_summary = (
        selected.groupby("session_date")
        .agg(
            trades=("symbol", "size"),
            pnl=("pnl_dollars", "sum"),
            win_rate=("pnl_dollars", lambda s: (s > 0).mean() * 100),
            avg_score=("candidate_score", "mean"),
        )
        .reset_index()
        .sort_values("session_date")
    )

    equity_curve = selected[["exit_time", "equity", "pnl_dollars", "symbol", "side", "trigger_type"]].copy()
    drawdown_curve = selected[["exit_time", "drawdown_pct"]].copy()
    sizing_scenarios = _build_sizing_scenarios(selected, params)
    return {
        "metrics": metrics,
        "selected_trades": selected,
        "symbol_summary": symbol_summary,
        "setup_summary": setup_summary,
        "daily_summary": daily_summary,
        "exit_summary": exit_summary,
        "score_band_summary": score_band_summary,
        "hour_summary": hour_summary,
        "direction_summary": direction_summary,
        "module_summary": module_summary,
        "candle_summary": candle_summary,
        "regime_summary": regime_summary,
        "market_context_summary": market_context_summary,
        "sizing_scenarios": sizing_scenarios,
        "equity_curve": equity_curve,
        "drawdown_curve": drawdown_curve,
    }


def _group_summary(df: pd.DataFrame, group_col: str, include_duration: bool = False) -> pd.DataFrame:
    aggs = dict(
        trades=(group_col, "size"),
        win_rate=("pnl_dollars", lambda s: (s > 0).mean() * 100),
        total_pnl=("pnl_dollars", "sum"),
        avg_r=("r_multiple", "mean"),
        avg_win_r=("r_multiple", lambda s: s[s > 0].mean() if (s > 0).any() else 0),
        avg_loss_r=("r_multiple", lambda s: s[s < 0].mean() if (s < 0).any() else 0),
        avg_mfe_r=("mfe_r", "mean"),
        avg_mae_r=("mae_r", "mean"),
        avg_score=("candidate_score", "mean"),
        profit_factor=("pnl_dollars", _profit_factor_from_series),
    )
    if include_duration:
        aggs["avg_duration"] = ("duration_minutes", "mean")
    return df.groupby(group_col).agg(**aggs).reset_index()


def _empty_metrics(params: StrategyParams) -> dict[str, Any]:
    return {
        "total_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "total_pnl": 0.0,
        "total_return_pct": 0.0,
        "expectancy_r": 0.0,
        "avg_win_r": 0.0,
        "avg_loss_r": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_duration_minutes": 0.0,
        "target1_hit_rate": 0.0,
        "target2_hit_rate": 0.0,
        "avg_mfe_r": 0.0,
        "avg_mae_r": 0.0,
        "initial_account_value": params.initial_account_value,
        "ending_equity": params.initial_account_value,
        "trading_days_with_trades": 0,
        "trade_days": 0,
        "trades_per_trade_day": 0.0,
        "avg_candidate_score": 0.0,
        # Average applied risk is the real engine risk for the selected trades.
        # In compounding modes this changes trade-by-trade, so do not report the
        # fixed-dollar UI input as the active engine risk.
        "risk_per_trade_dollars": float(selected["risk_budget"].mean()) if "risk_budget" in selected.columns else float(getattr(params, "risk_per_trade_dollars", params.initial_account_value * params.risk_per_trade_pct)),
        "requested_risk_dollars": float(getattr(params, "risk_per_trade_dollars", 0.0)),
        "requested_risk_percent": float(getattr(params, "requested_risk_percent", params.risk_per_trade_pct * 100)),
        "risk_per_trade_pct": float(params.risk_per_trade_pct),
        "risk_per_trade_percent": float(params.risk_per_trade_pct * 100),
        "position_sizing_mode": str(getattr(params, "position_sizing_mode", "fixed_dollar_risk")),
        "max_position_notional_pct": float(getattr(params, "max_position_notional_pct", 999.0)),
        "avg_risk_budget": 0.0,
        "avg_actual_dollars_at_risk": 0.0,
        "avg_actual_risk_pct_of_equity": 0.0,
        "avg_notional": 0.0,
        "median_notional": 0.0,
        "avg_notional_pct": 0.0,
        "sizing_cap_applied_trades": 0,
    }


def _profit_factor_from_series(s: pd.Series) -> float:
    wins = s[s > 0].sum()
    losses = s[s < 0].sum()
    if losses < 0:
        return float(wins / abs(losses))
    return 999.0 if wins > 0 else 0.0


def _buy_slippage(price: float, slippage_bps: float) -> float:
    return price * (1 + slippage_bps / 10_000.0)


def _sell_slippage(price: float, slippage_bps: float) -> float:
    return price * (1 - slippage_bps / 10_000.0)


def _exit_price(side: str, raw_price: float, slippage_bps: float) -> float:
    # Long exit is a sell, short exit is a buy-to-cover.
    return _sell_slippage(raw_price, slippage_bps) if side == "long" else _buy_slippage(raw_price, slippage_bps)


def _pnl(side: str, entry_price: float, exit_price: float) -> float:
    return exit_price - entry_price if side == "long" else entry_price - exit_price


def _stop_hit(side: str, high: float, low: float, stop_price: float) -> bool:
    return low <= stop_price if side == "long" else high >= stop_price


def _target_hit(side: str, high: float, low: float, target_price: float) -> bool:
    return high >= target_price if side == "long" else low <= target_price


def _favorable_reached(side: str, high_since_entry: float, low_since_entry: float, entry_price: float, distance: float) -> bool:
    return (high_since_entry >= entry_price + distance) if side == "long" else (low_since_entry <= entry_price - distance)


def _breakeven_stop(side: str, entry_price: float, slippage_bps: float, current_stop: float) -> float:
    slip = slippage_bps / 10_000.0
    if side == "long":
        breakeven = entry_price / max(1.0 - slip, 0.0001)
        return max(current_stop, breakeven)
    breakeven = entry_price / (1.0 + slip)
    return min(current_stop, breakeven)

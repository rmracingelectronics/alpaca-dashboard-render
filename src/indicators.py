from __future__ import annotations

import numpy as np
import pandas as pd

NY_TZ = "America/New_York"


def ensure_ny_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["timestamp_ny"] = out["timestamp"].dt.tz_convert(NY_TZ)
    out["session_date"] = out["timestamp_ny"].dt.date
    out["time_str"] = out["timestamp_ny"].dt.strftime("%H:%M")
    return out


def regular_session_only(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_ny_timestamp(df)
    mask = (out["time_str"] >= "09:30") & (out["time_str"] <= "16:00")
    return out.loc[mask].copy()


def ema(series: pd.Series, span: int) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    close = pd.to_numeric(series, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate intraday VWAP while avoiding pandas object-dtype fillna warnings."""
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    typical = (high + low + close) / 3.0
    pv = typical * volume

    cumulative_pv = pv.groupby(df["session_date"]).cumsum()
    cumulative_vol = volume.groupby(df["session_date"]).cumsum()

    return cumulative_pv / cumulative_vol.replace(0, np.nan)


def add_daily_features(intraday: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Merge previous-day and rolling daily features onto intraday bars."""
    daily_clean = daily.copy()
    daily_clean["timestamp"] = pd.to_datetime(daily_clean["timestamp"], utc=True)
    daily_clean["session_date"] = daily_clean["timestamp"].dt.tz_convert(NY_TZ).dt.date
    daily_clean = daily_clean.sort_values(["symbol", "session_date"])
    for col in ["open", "high", "low", "close", "volume"]:
        daily_clean[col] = pd.to_numeric(daily_clean[col], errors="coerce")
    daily_clean["prev_day_high"] = daily_clean.groupby("symbol")["high"].shift(1)
    daily_clean["prev_day_low"] = daily_clean.groupby("symbol")["low"].shift(1)
    daily_clean["prev_close"] = daily_clean.groupby("symbol")["close"].shift(1)
    daily_clean["daily_dollar_volume"] = daily_clean["volume"] * daily_clean["close"]
    daily_clean["avg_20d_dollar_volume"] = (
        daily_clean.groupby("symbol")["daily_dollar_volume"]
        .transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    )
    prev_close = daily_clean.groupby("symbol")["close"].shift(1)
    daily_tr = pd.concat(
        [
            daily_clean["high"] - daily_clean["low"],
            (daily_clean["high"] - prev_close).abs(),
            (daily_clean["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily_clean["daily_atr14"] = daily_tr.groupby(daily_clean["symbol"]).transform(
        lambda s: s.rolling(14, min_periods=14).mean()
    )
    daily_clean["daily_atr14_percent"] = daily_clean["daily_atr14"] / daily_clean["close"] * 100
    merge_cols = [
        "symbol",
        "session_date",
        "prev_day_high",
        "prev_day_low",
        "prev_close",
        "avg_20d_dollar_volume",
        "daily_atr14_percent",
    ]
    return intraday.merge(daily_clean[merge_cols], on=["symbol", "session_date"], how="left")


def add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    out = regular_session_only(df).sort_values(["symbol", "timestamp"]).copy()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    frames = []
    for symbol, group in out.groupby("symbol", sort=False):
        g = group.sort_values("timestamp").copy()
        g["ema9"] = ema(g["close"], 9)
        g["ema20"] = ema(g["close"], 20)
        g["ema50"] = ema(g["close"], 50)
        g["rsi2"] = rsi(g["close"], 2)
        g["rsi14"] = rsi(g["close"], 14)
        g["atr5m14"] = atr(g, 14)
        g["session_vwap"] = session_vwap(g)
        g["current_5m_dollar_volume"] = g["volume"] * g["close"]
        g["median_volume_last_20_5m"] = g["volume"].rolling(20, min_periods=10).median().shift(1)
        g["candle_range"] = g["high"] - g["low"]
        g["candle_body"] = (g["close"] - g["open"]).abs()
        g["upper_wick"] = g["high"] - g[["open", "close"]].max(axis=1)
        g["lower_wick"] = g[["open", "close"]].min(axis=1) - g["low"]
        g["candle_close_position"] = np.where(
            g["candle_range"] > 0,
            (g["close"] - g["low"]) / g["candle_range"],
            np.nan,
        )

        # Candlestick feature engineering. Alpaca supplies OHLCV; we classify the
        # candles ourselves in a numeric, backtestable way. These features are
        # intentionally simple and explainable: body %, wick %, engulfing,
        # rejection/pin-bar, inside-bar breakout, continuation, and exhaustion.
        cr = g["candle_range"].replace(0, np.nan)
        g["body_pct"] = g["candle_body"] / cr
        g["upper_wick_pct"] = g["upper_wick"] / cr
        g["lower_wick_pct"] = g["lower_wick"] / cr
        g["is_green_candle"] = g["close"] > g["open"]
        g["is_red_candle"] = g["close"] < g["open"]
        g["is_doji"] = (g["body_pct"] <= 0.16) & (g["upper_wick_pct"] >= 0.20) & (g["lower_wick_pct"] >= 0.20)

        prev_open = g["open"].shift(1)
        prev_close = g["close"].shift(1)
        prev_high = g["high"].shift(1)
        prev_low = g["low"].shift(1)
        prev_green = prev_close > prev_open
        prev_red = prev_close < prev_open
        prev_inside = (prev_high < g["high"].shift(2)) & (prev_low > g["low"].shift(2))

        g["bullish_continuation_candle"] = (
            g["is_green_candle"]
            & (g["body_pct"] >= 0.50)
            & (g["candle_close_position"] >= 0.70)
            & (g["upper_wick_pct"] <= 0.35)
        )
        g["bearish_continuation_candle"] = (
            g["is_red_candle"]
            & (g["body_pct"] >= 0.50)
            & (g["candle_close_position"] <= 0.30)
            & (g["lower_wick_pct"] <= 0.35)
        )
        g["bullish_rejection_candle"] = (
            (g["lower_wick_pct"] >= 0.45)
            & (g["lower_wick"] >= 1.35 * g["candle_body"].replace(0, np.nan))
            & (g["candle_close_position"] >= 0.62)
        )
        g["bearish_rejection_candle"] = (
            (g["upper_wick_pct"] >= 0.45)
            & (g["upper_wick"] >= 1.35 * g["candle_body"].replace(0, np.nan))
            & (g["candle_close_position"] <= 0.38)
        )
        g["bullish_engulfing_candle"] = (
            g["is_green_candle"]
            & prev_red.fillna(False)
            & (g["open"] <= prev_close)
            & (g["close"] >= prev_open)
            & (g["body_pct"] >= 0.35)
        )
        g["bearish_engulfing_candle"] = (
            g["is_red_candle"]
            & prev_green.fillna(False)
            & (g["open"] >= prev_close)
            & (g["close"] <= prev_open)
            & (g["body_pct"] >= 0.35)
        )
        g["inside_bar"] = (g["high"] < prev_high) & (g["low"] > prev_low)
        g["bullish_inside_breakout"] = prev_inside.fillna(False) & (g["close"] > prev_high) & g["is_green_candle"]
        g["bearish_inside_breakout"] = prev_inside.fillna(False) & (g["close"] < prev_low) & g["is_red_candle"]
        g["long_entry_candle_ok"] = (
            g["bullish_continuation_candle"]
            | g["bullish_rejection_candle"]
            | g["bullish_engulfing_candle"]
            | g["bullish_inside_breakout"]
        )
        g["short_entry_candle_ok"] = (
            g["bearish_continuation_candle"]
            | g["bearish_rejection_candle"]
            | g["bearish_engulfing_candle"]
            | g["bearish_inside_breakout"]
        )
        g["long_exit_warning_candle"] = (
            g["bearish_rejection_candle"]
            | g["bearish_engulfing_candle"]
            | ((g["is_doji"]) & (g["candle_close_position"] <= 0.45))
            | ((g["upper_wick_pct"] >= 0.52) & (g["candle_close_position"] <= 0.55))
        )
        g["short_exit_warning_candle"] = (
            g["bullish_rejection_candle"]
            | g["bullish_engulfing_candle"]
            | ((g["is_doji"]) & (g["candle_close_position"] >= 0.55))
            | ((g["lower_wick_pct"] >= 0.52) & (g["candle_close_position"] >= 0.45))
        )
        g["candle_pattern_primary"] = np.select(
            [
                g["bullish_engulfing_candle"],
                g["bullish_rejection_candle"],
                g["bullish_inside_breakout"],
                g["bullish_continuation_candle"],
                g["bearish_engulfing_candle"],
                g["bearish_rejection_candle"],
                g["bearish_inside_breakout"],
                g["bearish_continuation_candle"],
                g["is_doji"],
            ],
            [
                "bullish_engulfing",
                "bullish_rejection",
                "bullish_inside_breakout",
                "bullish_continuation",
                "bearish_engulfing",
                "bearish_rejection",
                "bearish_inside_breakout",
                "bearish_continuation",
                "doji_indecision",
            ],
            default="neutral",
        )
        g["session_open"] = g.groupby("session_date")["open"].transform("first")
        opening_mask_15 = (g["time_str"] >= "09:30") & (g["time_str"] < "09:45")
        opening_high = g.loc[opening_mask_15].groupby("session_date")["high"].max()
        opening_low = g.loc[opening_mask_15].groupby("session_date")["low"].min()
        g["opening_range_high"] = g["session_date"].map(opening_high)
        g["opening_range_low"] = g["session_date"].map(opening_low)
        opening_mask_30 = (g["time_str"] >= "09:30") & (g["time_str"] < "10:00")
        opening30_high = g.loc[opening_mask_30].groupby("session_date")["high"].max()
        opening30_low = g.loc[opening_mask_30].groupby("session_date")["low"].min()
        g["opening_30_high"] = g["session_date"].map(opening30_high)
        g["opening_30_low"] = g["session_date"].map(opening30_low)
        g["intraday_high_so_far"] = g.groupby("session_date")["high"].cummax()
        g["intraday_low_so_far"] = g.groupby("session_date")["low"].cummin()

        # Time-of-day RVOL: compare each time slot's volume with the median of that same slot over prior 20 sessions.
        g["time_slot"] = g["time_str"]
        slot_frames = []
        for slot, sg in g.groupby("time_slot", sort=False):
            sg = sg.sort_values("session_date").copy()
            sg["slot_median_volume_20d"] = sg["volume"].rolling(20, min_periods=5).median().shift(1)
            slot_frames.append(sg[["timestamp", "slot_median_volume_20d"]])
        if slot_frames:
            slot_ref = pd.concat(slot_frames, ignore_index=True)
            g = g.merge(slot_ref, on="timestamp", how="left")
        else:
            g["slot_median_volume_20d"] = np.nan
        fallback_volume = g["median_volume_last_20_5m"]
        base_volume = g["slot_median_volume_20d"].fillna(fallback_volume)
        g["rvol_time_of_day"] = g["volume"] / base_volume.replace(0, np.nan)
        g["vwap_extension_atr"] = (g["close"] - g["session_vwap"]) / g["atr5m14"].replace(0, np.nan)
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_qqq_context(qqq_5m: pd.DataFrame, qqq_daily: pd.DataFrame) -> pd.DataFrame:
    q = add_intraday_features(qqq_5m)
    q = add_daily_features(q, qqq_daily)
    qdaily = qqq_daily.copy()
    qdaily["timestamp"] = pd.to_datetime(qdaily["timestamp"], utc=True)
    qdaily["session_date"] = qdaily["timestamp"].dt.tz_convert(NY_TZ).dt.date
    qdaily = qdaily.sort_values("session_date")
    for col in ["open", "high", "low", "close", "volume"]:
        qdaily[col] = pd.to_numeric(qdaily[col], errors="coerce")
    qdaily_prev_close = qdaily["close"].shift(1)
    qdaily_tr = pd.concat(
        [
            qdaily["high"] - qdaily["low"],
            (qdaily["high"] - qdaily_prev_close).abs(),
            (qdaily["low"] - qdaily_prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    qdaily["daily_atr14"] = qdaily_tr.rolling(14, min_periods=14).mean()
    qdaily["qqq_daily_atr14_percent"] = qdaily["daily_atr14"] / qdaily["close"] * 100
    qdaily_small = qdaily[["session_date", "qqq_daily_atr14_percent"]]

    # Resample 5-min QQQ to 15-min bars per session for EMA50 regime.
    q_resample = q.set_index("timestamp_ny").copy()
    fifteen_frames = []
    for session_date, group in q_resample.groupby("session_date"):
        ohlcv = group.resample("15min", label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna(subset=["close"])
        if not ohlcv.empty:
            ohlcv["session_date"] = session_date
            fifteen_frames.append(ohlcv)
    if fifteen_frames:
        q15 = pd.concat(fifteen_frames).reset_index().rename(columns={"timestamp_ny": "timestamp_ny_15m"})
        q15["timestamp"] = q15["timestamp_ny_15m"].dt.tz_convert("UTC")
        q15 = q15.sort_values("timestamp")
        q15["qqq_15m_close"] = q15["close"]
        q15["qqq_15m_ema50"] = ema(q15["close"], 50)
        q15 = q15[["timestamp", "qqq_15m_close", "qqq_15m_ema50"]]
    else:
        q15 = pd.DataFrame(columns=["timestamp", "qqq_15m_close", "qqq_15m_ema50"])

    q5 = q[
        [
            "timestamp",
            "session_date",
            "close",
            "session_vwap",
            "prev_close",
            "session_open",
            "ema9",
            "ema20",
            "rsi2",
            "atr5m14",
        ]
    ].copy()
    q5 = q5.rename(
        columns={
            "close": "qqq_close",
            "session_vwap": "qqq_session_vwap",
            "prev_close": "qqq_prev_close",
            "session_open": "qqq_session_open",
            "ema9": "qqq_ema9",
            "ema20": "qqq_ema20",
            "rsi2": "qqq_rsi2",
            "atr5m14": "qqq_atr5m14",
        }
    )
    q5["qqq_day_change_percent"] = (q5["qqq_close"] - q5["qqq_prev_close"]) / q5["qqq_prev_close"] * 100
    q5["qqq_change_from_open"] = (q5["qqq_close"] - q5["qqq_session_open"]) / q5["qqq_session_open"] * 100
    q5["qqq_15min_change_percent"] = q5.groupby("session_date")["qqq_close"].transform(lambda x: (x - x.shift(3)) / x.shift(3) * 100)
    q5["qqq_15min_change_percent"] = q5.groupby("session_date")["qqq_close"].transform(lambda x: (x - x.shift(3)) / x.shift(3) * 100)
    q5 = q5.merge(qdaily_small, on="session_date", how="left")
    if not q15.empty:
        q5 = pd.merge_asof(q5.sort_values("timestamp"), q15.sort_values("timestamp"), on="timestamp", direction="backward")
    else:
        q5["qqq_15m_close"] = np.nan
        q5["qqq_15m_ema50"] = np.nan
    q5["market_filter_pass"] = (
        (q5["qqq_15m_close"] > q5["qqq_15m_ema50"])
        & (q5["qqq_close"] > q5["qqq_session_vwap"])
        & (q5["qqq_daily_atr14_percent"] <= 3.2)
    )
    return q5.sort_values("timestamp")


def merge_market_context(symbol_df: pd.DataFrame, qqq_context: pd.DataFrame) -> pd.DataFrame:
    left = symbol_df.sort_values("timestamp").copy()
    right = qqq_context.sort_values("timestamp").copy()
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward", suffixes=("", "_qqq"))
    return merged

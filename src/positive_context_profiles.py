from __future__ import annotations

from typing import Any
import json
import numpy as np
import pandas as pd

# V37.8: mined from the profitable Top-N report, but converted into stricter
# multi-indicator, live-safe patterns. The rules below use only values available
# at signal/entry time: symbol, side, trigger, timestamp, score, RVOL, ATR,
# relative strength, VWAP extension, gap and QQQ context. They do not read the
# current trade outcome, future bars, MFE/MAE, exit reason or R multiple.
DEFAULT_POSITIVE_CONTEXT_PROFILES = json.loads(r'''
{
  "version": "v378_indicator_pattern_matcher",
  "source": "latest_backtest_report(1).zip",
  "profiles": [
    {
      "name": "QCOM_S_gap_cont_controlled_1",
      "symbol": "QCOM",
      "side": "short",
      "trigger_type": "v25_S_gap_cont_controlled",
      "rules": {
        "candidate_score": [
          39.556923,
          42.354777
        ],
        "abs_qqq": [
          0.162491,
          1.734698
        ]
      },
      "source_group_metrics": {
        "n": 22,
        "wr": 68.18181818181817,
        "total_r": 4.25,
        "avg_r": 0.19318181818181818,
        "pf": 1.6071428571428572,
        "losses": 7,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 11,
        "wr": 100.0,
        "total_r": 8.25,
        "avg_r": 0.75,
        "pf": 999,
        "losses": 0,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 37.05,
      "rule_count": 2,
      "event": "S_gap_cont_controlled"
    },
    {
      "name": "QCOM_S_gap_cont_controlled_2",
      "symbol": "QCOM",
      "side": "short",
      "trigger_type": "v25_S_gap_cont_controlled",
      "rules": {
        "rvol_time_of_day": [
          0.91314,
          2.353743
        ],
        "abs_qqq": [
          0.162491,
          1.734698
        ]
      },
      "source_group_metrics": {
        "n": 22,
        "wr": 68.18181818181817,
        "total_r": 4.25,
        "avg_r": 0.19318181818181818,
        "pf": 1.6071428571428572,
        "losses": 7,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 11,
        "wr": 100.0,
        "total_r": 8.25,
        "avg_r": 0.75,
        "pf": 999,
        "losses": 0,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 37.05,
      "rule_count": 2,
      "event": "S_gap_cont_controlled"
    },
    {
      "name": "BA_L_gap_cont_controlled_1",
      "symbol": "BA",
      "side": "long",
      "trigger_type": "v25_L_gap_cont_controlled",
      "rules": {
        "minute_et": [
          595.0,
          663.75
        ],
        "daily_atr14_percent": [
          1.886598,
          4.677545
        ]
      },
      "source_group_metrics": {
        "n": 18,
        "wr": 66.66666666666666,
        "total_r": 3.0,
        "avg_r": 0.16666666666666666,
        "pf": 1.5,
        "losses": 6,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 9,
        "wr": 100.0,
        "total_r": 6.75,
        "avg_r": 0.75,
        "pf": 999,
        "losses": 0,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 33.95,
      "rule_count": 3,
      "event": "L_gap_cont_controlled"
    },
    {
      "name": "BA_L_gap_cont_controlled_3",
      "symbol": "BA",
      "side": "long",
      "trigger_type": "v25_L_gap_cont_controlled",
      "rules": {
        "abs_gap": [
          1.538889,
          3.735173
        ],
        "minute_et": [
          600.625,
          663.75
        ]
      },
      "source_group_metrics": {
        "n": 18,
        "wr": 66.66666666666666,
        "total_r": 3.0,
        "avg_r": 0.16666666666666666,
        "pf": 1.5,
        "losses": 6,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 8,
        "wr": 100.0,
        "total_r": 6.0,
        "avg_r": 0.75,
        "pf": 999,
        "losses": 0,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 33.9,
      "rule_count": 2,
      "event": "L_gap_cont_controlled"
    },
    {
      "name": "XOM_L_late_trend_follow_1",
      "symbol": "XOM",
      "side": "long",
      "trigger_type": "v25_L_late_trend_follow",
      "rules": {
        "abs_gap": [
          0.236034,
          1.658523
        ],
        "dir_vwap": [
          0.748675,
          1.856362
        ]
      },
      "source_group_metrics": {
        "n": 27,
        "wr": 55.55555555555556,
        "total_r": 1.0299495254467077,
        "avg_r": 0.038146278720248436,
        "pf": 1.10077734234397,
        "losses": 12,
        "years": 5,
        "years_pos": 3
      },
      "profile_metrics": {
        "n": 10,
        "wr": 90.0,
        "total_r": 6.558766629874183,
        "avg_r": 0.6558766629874183,
        "pf": 35.29718686419127,
        "losses": 1,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 35.058766629874185,
      "rule_count": 2,
      "event": "L_late_trend_follow"
    },
    {
      "name": "BA_L_late_trend_follow_2",
      "symbol": "BA",
      "side": "long",
      "trigger_type": "v25_L_late_trend_follow",
      "rules": {
        "minute_et": [
          736.25,
          785.0
        ],
        "candidate_score": [
          14.126036,
          19.734214
        ]
      },
      "source_group_metrics": {
        "n": 38,
        "wr": 71.05263157894737,
        "total_r": 9.25,
        "avg_r": 0.24342105263157895,
        "pf": 1.8409090909090908,
        "losses": 11,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 19,
        "wr": 94.73684210526315,
        "total_r": 12.5,
        "avg_r": 0.6578947368421053,
        "pf": 13.5,
        "losses": 1,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 43.7,
      "rule_count": 2,
      "event": "L_late_trend_follow"
    },
    {
      "name": "BA_L_late_trend_follow_3",
      "symbol": "BA",
      "side": "long",
      "trigger_type": "v25_L_late_trend_follow",
      "rules": {
        "minute_et": [
          736.25,
          785.0
        ],
        "rvol_time_of_day": [
          0.274587,
          2.95198
        ]
      },
      "source_group_metrics": {
        "n": 38,
        "wr": 71.05263157894737,
        "total_r": 9.25,
        "avg_r": 0.24342105263157895,
        "pf": 1.8409090909090908,
        "losses": 11,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 19,
        "wr": 94.73684210526315,
        "total_r": 12.5,
        "avg_r": 0.6578947368421053,
        "pf": 13.5,
        "losses": 1,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 43.7,
      "rule_count": 2,
      "event": "L_late_trend_follow"
    },
    {
      "name": "TSLA_L_prev_low_sweep_reclaim_1",
      "symbol": "TSLA",
      "side": "long",
      "trigger_type": "v25_L_prev_low_sweep_reclaim",
      "rules": {
        "dir_rs": [
          -2.168405,
          -0.060585
        ],
        "candidate_score": [
          24.258113,
          27.570384
        ]
      },
      "source_group_metrics": {
        "n": 47,
        "wr": 59.57446808510638,
        "total_r": 2.0,
        "avg_r": 0.0425531914893617,
        "pf": 1.105263157894737,
        "losses": 19,
        "years": 5,
        "years_pos": 3
      },
      "profile_metrics": {
        "n": 18,
        "wr": 94.44444444444444,
        "total_r": 11.75,
        "avg_r": 0.6527777777777778,
        "pf": 12.75,
        "losses": 1,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 42.65,
      "rule_count": 2,
      "event": "L_prev_low_sweep_reclaim"
    },
    {
      "name": "TSLA_L_prev_low_sweep_reclaim_2",
      "symbol": "TSLA",
      "side": "long",
      "trigger_type": "v25_L_prev_low_sweep_reclaim",
      "rules": {
        "dir_rs": [
          -2.168405,
          -0.060585
        ],
        "rvol_time_of_day": [
          1.234489,
          2.933901
        ]
      },
      "source_group_metrics": {
        "n": 47,
        "wr": 59.57446808510638,
        "total_r": 2.0,
        "avg_r": 0.0425531914893617,
        "pf": 1.105263157894737,
        "losses": 19,
        "years": 5,
        "years_pos": 3
      },
      "profile_metrics": {
        "n": 18,
        "wr": 94.44444444444444,
        "total_r": 11.75,
        "avg_r": 0.6527777777777778,
        "pf": 12.75,
        "losses": 1,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 42.65,
      "rule_count": 2,
      "event": "L_prev_low_sweep_reclaim"
    },
    {
      "name": "MU_S_prev_high_sweep_reject_2",
      "symbol": "MU",
      "side": "short",
      "trigger_type": "v25_S_prev_high_sweep_reject",
      "rules": {
        "abs_vwap": [
          0.351189,
          2.147393
        ],
        "dir_rs": [
          -6.00158,
          -0.073844
        ],
        "abs_qqq": [
          0.013912,
          0.711817
        ]
      },
      "source_group_metrics": {
        "n": 42,
        "wr": 73.80952380952381,
        "total_r": 12.25,
        "avg_r": 0.2916666666666667,
        "pf": 2.1136363636363638,
        "losses": 11,
        "years": 5,
        "years_pos": 5
      },
      "profile_metrics": {
        "n": 15,
        "wr": 93.33333333333333,
        "total_r": 9.5,
        "avg_r": 0.6333333333333333,
        "pf": 10.5,
        "losses": 1,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 38.5,
      "rule_count": 3,
      "event": "S_prev_high_sweep_reject"
    },
    {
      "name": "MU_S_prev_high_sweep_reject_3",
      "symbol": "MU",
      "side": "short",
      "trigger_type": "v25_S_prev_high_sweep_reject",
      "rules": {
        "abs_vwap": [
          0.351189,
          2.147393
        ],
        "dir_open_rs": [
          -6.00158,
          -0.073844
        ],
        "abs_qqq": [
          0.013912,
          0.711817
        ]
      },
      "source_group_metrics": {
        "n": 42,
        "wr": 73.80952380952381,
        "total_r": 12.25,
        "avg_r": 0.2916666666666667,
        "pf": 2.1136363636363638,
        "losses": 11,
        "years": 5,
        "years_pos": 5
      },
      "profile_metrics": {
        "n": 15,
        "wr": 93.33333333333333,
        "total_r": 9.5,
        "avg_r": 0.6333333333333333,
        "pf": 10.5,
        "losses": 1,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 38.5,
      "rule_count": 3,
      "event": "S_prev_high_sweep_reject"
    },
    {
      "name": "INTC_S_late_trend_follow_refined_v378",
      "symbol": "INTC",
      "side": "short",
      "event": "S_late_trend_follow",
      "trigger_type": "v25_S_late_trend_follow",
      "rules": {
        "minute_et": [
          720,
          840
        ],
        "daily_atr14_percent": [
          2.3,
          5.6
        ],
        "dir_vwap": [
          1.5,
          2.6
        ],
        "abs_qqq": [
          0.0,
          0.8
        ]
      },
      "source_group_metrics": {
        "n": 79,
        "wr": 68.35,
        "total_r": 15.7301,
        "pf": 1.635,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 30,
        "wr": 80.0,
        "total_r": 12.2301,
        "avg_r": 0.4077,
        "pf": 3.1196,
        "losses": 6,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 60.0,
      "rule_count": 4
    },
    {
      "name": "XOM_L_10_ORB_confirmed_2",
      "symbol": "XOM",
      "side": "long",
      "trigger_type": "v25_L_10_ORB_confirmed",
      "rules": {
        "abs_gap": [
          0.070012,
          1.338518
        ],
        "rvol_time_of_day": [
          0.64859,
          2.3954
        ]
      },
      "source_group_metrics": {
        "n": 83,
        "wr": 66.26506024096386,
        "total_r": 12.676359664275248,
        "avg_r": 0.15272722487078613,
        "pf": 1.4527271308669731,
        "losses": 28,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 52,
        "wr": 78.84615384615384,
        "total_r": 19.176359664275246,
        "avg_r": 0.3687761473899086,
        "pf": 2.7433054240250225,
        "losses": 11,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 45.762970512325296,
      "rule_count": 2,
      "event": "L_10_ORB_confirmed"
    },
    {
      "name": "MU_S_vwap_pullback_trend_3",
      "symbol": "MU",
      "side": "short",
      "trigger_type": "v25_S_vwap_pullback_trend",
      "rules": {
        "dir_rs": [
          -1.985931,
          1.75941
        ],
        "daily_atr14_percent": [
          3.006883,
          10.106559
        ]
      },
      "source_group_metrics": {
        "n": 137,
        "wr": 64.96350364963503,
        "total_r": 18.152733486620306,
        "avg_r": 0.13250170428190006,
        "pf": 1.3781819476379231,
        "losses": 48,
        "years": 5,
        "years_pos": 5
      },
      "profile_metrics": {
        "n": 103,
        "wr": 72.81553398058253,
        "total_r": 28.25,
        "avg_r": 0.27427184466019416,
        "pf": 2.0089285714285716,
        "losses": 28,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 68.66785714285714,
      "rule_count": 2,
      "event": "S_vwap_pullback_trend"
    },
    {
      "name": "MA_L_vwap_pullback_trend_3",
      "symbol": "MA",
      "side": "long",
      "trigger_type": "v25_L_vwap_pullback_trend",
      "rules": {
        "dir_vwap": [
          0.190767,
          0.872702
        ],
        "abs_vwap": [
          0.190767,
          0.872702
        ]
      },
      "source_group_metrics": {
        "n": 108,
        "wr": 60.18518518518518,
        "total_r": 6.355410753716535,
        "avg_r": 0.0588463958677457,
        "pf": 1.1513193036599174,
        "losses": 42,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 80,
        "wr": 70.0,
        "total_r": 18.605410753716534,
        "avg_r": 0.23256763442145667,
        "pf": 1.8089309023355014,
        "losses": 23,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 51.723272558387535,
      "rule_count": 2,
      "event": "L_vwap_pullback_trend"
    },
    {
      "name": "COIN_S_vwap_pullback_trend_2",
      "symbol": "COIN",
      "side": "short",
      "trigger_type": "v25_S_vwap_pullback_trend",
      "rules": {
        "minute_et": [
          615.0,
          715.0
        ],
        "candidate_score": [
          32.104806,
          37.01097
        ]
      },
      "source_group_metrics": {
        "n": 118,
        "wr": 61.016949152542374,
        "total_r": 8.0,
        "avg_r": 0.06779661016949153,
        "pf": 1.173913043478261,
        "losses": 46,
        "years": 5,
        "years_pos": 4
      },
      "profile_metrics": {
        "n": 91,
        "wr": 65.93406593406593,
        "total_r": 14.0,
        "avg_r": 0.15384615384615385,
        "pf": 1.4516129032258065,
        "losses": 31,
        "years": 5,
        "years_pos": 5
      },
      "pattern_score": 49.70322580645161,
      "rule_count": 2,
      "event": "S_vwap_pullback_trend"
    },
    {
      "name": "PG_S_10_ORB_confirmed_1",
      "symbol": "PG",
      "side": "short",
      "trigger_type": "v25_S_10_ORB_confirmed",
      "rules": {
        "abs_gap": [
          0.042731,
          0.295419
        ],
        "daily_atr14_percent": [
          1.031684,
          1.289214
        ]
      },
      "source_group_metrics": {
        "n": 44,
        "wr": 59.09090909090909,
        "total_r": 2.0904603508959045,
        "avg_r": 0.047510462520361466,
        "pf": 1.1200755673630622,
        "losses": 18,
        "years": 5,
        "years_pos": 2
      },
      "profile_metrics": {
        "n": 9,
        "wr": 100.0,
        "total_r": 6.75,
        "avg_r": 0.75,
        "pf": 999,
        "losses": 0,
        "years": 4,
        "years_pos": 4
      },
      "pattern_score": 33.45,
      "rule_count": 2,
      "event": "S_10_ORB_confirmed"
    },
    {
      "name": "PG_S_10_ORB_confirmed_2",
      "symbol": "PG",
      "side": "short",
      "trigger_type": "v25_S_10_ORB_confirmed",
      "rules": {
        "abs_gap": [
          0.042731,
          0.233827
        ],
        "abs_qqq": [
          0.106686,
          0.290681
        ]
      },
      "source_group_metrics": {
        "n": 44,
        "wr": 59.09090909090909,
        "total_r": 2.0904603508959045,
        "avg_r": 0.047510462520361466,
        "pf": 1.1200755673630622,
        "losses": 18,
        "years": 5,
        "years_pos": 2
      },
      "profile_metrics": {
        "n": 9,
        "wr": 100.0,
        "total_r": 6.75,
        "avg_r": 0.75,
        "pf": 999,
        "losses": 0,
        "years": 4,
        "years_pos": 4
      },
      "pattern_score": 33.45,
      "rule_count": 2,
      "event": "S_10_ORB_confirmed"
    }
  ]
}''')


def _float_series(df: pd.DataFrame, names: list[str], default: float = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _event_series(df: pd.DataFrame) -> pd.Series:
    if "trigger_type" in df.columns:
        s = df["trigger_type"].fillna("").astype(str)
    elif "event" in df.columns:
        s = df["event"].fillna("").astype(str)
    else:
        s = pd.Series("", index=df.index)
    return s.str.replace("v25_", "", regex=False)


def _minute_et(df: pd.DataFrame) -> pd.Series:
    for col in ["entry_time_et", "signal_time_et", "timestamp_ny"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce")
            try:
                out = (ts.dt.hour * 60 + ts.dt.minute).astype(float)
                if out.notna().any():
                    return out
            except Exception:
                pass
    for col in ["timestamp", "signal_time", "entry_time"]:
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce")
            try:
                ts = ts.dt.tz_convert("America/New_York")
                return (ts.dt.hour * 60 + ts.dt.minute).astype(float)
            except Exception:
                pass
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    side_short = df.get("side", pd.Series("", index=df.index)).fillna("").astype(str).str.lower().eq("short")
    mult = pd.Series(np.where(side_short, -1.0, 1.0), index=df.index, dtype="float64")
    rs = _float_series(df, ["day_relative_strength", "rs_open"], np.nan)
    ors = _float_series(df, ["open_relative_strength", "rs_open"], np.nan)
    vwap = _float_series(df, ["vwap_extension_atr", "vwap_ext_atr"], np.nan)
    gap = _float_series(df, ["gap_percent", "gap_pct"], np.nan)
    qqq = _float_series(df, ["qqq_day_change_percent", "qqq_change_from_open", "qqq_chg_open"], np.nan)
    out["minute_et"] = _minute_et(df)
    out["rvol_time_of_day"] = _float_series(df, ["rvol_time_of_day", "rvol_tod"], np.nan)
    out["daily_atr14_percent"] = _float_series(df, ["daily_atr14_percent", "daily_atr_pct"], np.nan)
    out["abs_gap"] = gap.abs()
    out["gap_percent"] = gap
    out["dir_rs"] = mult * rs
    out["dir_open_rs"] = mult * ors
    out["dir_vwap"] = mult * vwap
    out["abs_vwap"] = vwap.abs()
    out["abs_qqq"] = qqq.abs()
    out["qqq_day_change_percent"] = qqq
    out["candidate_score"] = _float_series(df, ["candidate_score", "score"], np.nan)
    out["fallback_score"] = _float_series(df, ["fallback_score", "v25_live_bar_quality_score"], np.nan)
    out["candle_body_pct"] = _float_series(df, ["candle_body_pct"], np.nan)
    out["close_pos_bar"] = _float_series(df, ["close_pos_bar", "close_pos_bar_calc"], np.nan)
    return out


def _profile_allowed(profile: dict[str, Any], min_source_total_r: float, min_profile_total_r: float, min_profile_pf: float) -> bool:
    source = profile.get("source_group_metrics", {}) or {}
    metrics = profile.get("profile_metrics", {}) or {}
    return (
        float(source.get("total_r", profile.get("source_total_r", 0.0)) or 0.0) >= min_source_total_r
        and float(metrics.get("total_r", profile.get("profile_backtest_total_r", 0.0)) or 0.0) >= min_profile_total_r
        and float(metrics.get("pf", profile.get("profile_backtest_profit_factor", 0.0)) or 0.0) >= min_profile_pf
    )


def apply_positive_context_profile_filter(df: pd.DataFrame, params: Any) -> pd.DataFrame:
    """Apply the V37.8 mined positive-indicator pattern matcher.

    The matcher is designed for live/raw replay. It only evaluates current
    candidate fields available at the signal time. It was learned from the
    profitable report by comparing repeated profitable indicator bands against
    losing selected trades, then keeping multi-indicator patterns with year
    stability. A candidate must match a symbol/side/trigger profile.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    profiles = DEFAULT_POSITIVE_CONTEXT_PROFILES.get("profiles", [])
    if not profiles:
        out["positive_context_profile_match"] = False
        return out

    min_source_total_r = float(getattr(params, "v377_min_source_total_r", 0.0) or 0.0)
    min_profile_total_r = float(getattr(params, "v377_min_profile_total_r", 0.0) or 0.0)
    min_profile_pf = float(getattr(params, "v377_min_profile_profit_factor", 1.0) or 1.0)
    active_profiles = [p for p in profiles if _profile_allowed(p, min_source_total_r, min_profile_total_r, min_profile_pf)]

    feats = _feature_frame(out)
    symbol = out.get("symbol", pd.Series("", index=out.index)).fillna("").astype(str).str.upper()
    side = out.get("side", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    event = _event_series(out)

    match = pd.Series(False, index=out.index)
    names = pd.Series("", index=out.index, dtype="object")
    reasons = pd.Series("", index=out.index, dtype="object")
    pattern_score = pd.Series(np.nan, index=out.index, dtype="float64")
    pattern_wr = pd.Series(np.nan, index=out.index, dtype="float64")
    pattern_pf = pd.Series(np.nan, index=out.index, dtype="float64")
    pattern_total_r = pd.Series(np.nan, index=out.index, dtype="float64")
    pattern_n = pd.Series(np.nan, index=out.index, dtype="float64")

    # Apply more specific profiles first: symbol-specific, then more rules, then stronger PF/R.
    def sort_key(p: dict[str, Any]) -> tuple:
        metrics = p.get("profile_metrics", {}) or {}
        sym_specific = 0 if str(p.get("symbol", "*")).strip() == "*" else 1
        return (
            sym_specific,
            len(p.get("rules", {}) or {}),
            float(metrics.get("pf", 0.0) or 0.0),
            float(metrics.get("total_r", 0.0) or 0.0),
        )

    for profile in sorted(active_profiles, key=sort_key, reverse=True):
        prof_symbol = str(profile.get("symbol", "")).upper().strip()
        prof_side = str(profile.get("side", "")).lower().strip()
        prof_event = str(profile.get("event", profile.get("trigger_type", ""))).replace("v25_", "")
        base = side.eq(prof_side) & event.eq(prof_event)
        if prof_symbol and prof_symbol != "*":
            base &= symbol.eq(prof_symbol)
        if not bool(base.any()):
            continue
        m = base.copy()
        parts: list[str] = []
        missing_required = False
        for feat, bounds in (profile.get("rules", {}) or {}).items():
            if feat not in feats.columns:
                missing_required = True
                break
            lo, hi = float(bounds[0]), float(bounds[1])
            vals = feats[feat]
            # Missing values should fail, not silently default to zero.
            m &= vals.notna() & vals.ge(lo) & vals.le(hi)
            parts.append(f"{feat}={lo:g}..{hi:g}")
        if missing_required:
            continue
        new = m & ~match
        if bool(new.any()):
            metrics = profile.get("profile_metrics", {}) or {}
            source = profile.get("source_group_metrics", {}) or {}
            match |= m
            names.loc[new] = str(profile.get("name", "positive_indicator_pattern"))
            pattern_score.loc[new] = float(profile.get("pattern_score", np.nan) or np.nan)
            pattern_wr.loc[new] = float(metrics.get("wr", np.nan) or np.nan)
            pattern_pf.loc[new] = float(metrics.get("pf", np.nan) or np.nan)
            pattern_total_r.loc[new] = float(metrics.get("total_r", np.nan) or np.nan)
            pattern_n.loc[new] = float(metrics.get("n", np.nan) or np.nan)
            reasons.loc[new] = (
                "matched V37.8 mined profitable indicator pattern | "
                + str(profile.get("name", ""))
                + " | profile_n=" + str(metrics.get("n", ""))
                + " | profile_WR=" + str(metrics.get("wr", ""))
                + " | profile_R=" + str(metrics.get("total_r", ""))
                + " | profile_PF=" + str(metrics.get("pf", ""))
                + " | source_R=" + str(source.get("total_r", ""))
                + " | " + "; ".join(parts)
            )

    out["positive_context_profile_match"] = match.fillna(False).astype(bool)
    out["positive_context_profile_name"] = names
    out["positive_context_profile_reason"] = reasons
    out["positive_context_pattern_score"] = pattern_score
    out["positive_context_pattern_win_rate"] = pattern_wr
    out["positive_context_pattern_profit_factor"] = pattern_pf
    out["positive_context_pattern_total_r"] = pattern_total_r
    out["positive_context_pattern_n"] = pattern_n
    for col in feats.columns:
        out[f"positive_context_{col}"] = feats[col]
    out["positive_context_active_profile_count"] = len(active_profiles)
    if "buy_alert" in out.columns:
        out["buy_alert"] = out["buy_alert"].fillna(False).astype(bool) & out["positive_context_profile_match"]
    return out

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ACTIONS = ("skip", "trade")
NY_TZ = "America/New_York"


@dataclass
class QLearningPolicyConfig:
    """Tabular Q-learning trade/no-trade policy for live-safe candidate rows.

    This is intentionally discrete/tabular rather than a black-box neural net so
    the user can audit every state bucket and see exactly why a candidate was
    approved or rejected.  The action space is deliberately simple:

    - skip: do nothing for this algorithm-generated candidate.
    - trade: let the candidate continue into the normal live/backtest selector.

    Rewards are in R-multiple units and include a mean-variance / utility-style
    risk penalty inspired by Ritter's paper:

        reward = r_multiple - 0.5 * kappa * (r_multiple - reward_mu)^2

    In the first pass reward_mu is 0.0, matching the paper's suggested initial
    biased estimator.  You can later use --reward-mu to experiment.
    """

    alpha: float = 0.05
    gamma: float = 0.0
    epsilon: float = 0.10
    kappa: float = 0.25
    reward_mu: float = 0.0
    train_epochs: int = 25
    min_state_count: int = 8
    min_edge: float = 0.0
    use_hierarchical_fallback: bool = True
    random_seed: int = 42
    max_abs_reward: float = 3.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "QLearningPolicyConfig":
        data = data or {}
        allowed = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _as_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        val = row.get(col, default)
        if pd.isna(val):
            return float(default)
        out = float(val)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _as_str(row: pd.Series, col: str, default: str = "") -> str:
    try:
        val = row.get(col, default)
        if pd.isna(val):
            return str(default)
        return str(val)
    except Exception:
        return str(default)


def _timestamp_ny(row: pd.Series) -> pd.Timestamp | None:
    for col in ["timestamp_ny", "entry_time_et", "entry_time", "timestamp"]:
        if col in row.index:
            try:
                ts = pd.Timestamp(row.get(col))
                if pd.isna(ts):
                    continue
                if ts.tzinfo is None:
                    # timestamp_ny and entry_time_et are already local-looking.
                    if col in {"timestamp_ny", "entry_time_et"}:
                        return ts.tz_localize(NY_TZ)
                    return ts.tz_localize("UTC").tz_convert(NY_TZ)
                return ts.tz_convert(NY_TZ)
            except Exception:
                continue
    return None


def _bucket_num(x: float, bins: list[float], labels: list[str]) -> str:
    try:
        x = float(x)
    except Exception:
        return "na"
    if not math.isfinite(x):
        return "na"
    for edge, label in zip(bins, labels):
        if x < edge:
            return label
    return labels[-1] if labels else "hi"


def _time_bucket(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return "t_na"
    minutes = int(ts.hour) * 60 + int(ts.minute)
    if minutes < 9 * 60 + 45:
        return "t_0930_0945"
    if minutes < 10 * 60:
        return "t_0945_1000"
    if minutes < 10 * 60 + 30:
        return "t_1000_1030"
    if minutes < 11 * 60:
        return "t_1030_1100"
    if minutes < 12 * 60:
        return "t_1100_1200"
    if minutes < 13 * 60 + 30:
        return "t_1200_1330"
    if minutes < 15 * 60:
        return "t_1330_1500"
    return "t_late"


def _setup_family(trigger: str) -> str:
    t = str(trigger or "").lower()
    if "gap_cont" in t:
        return "gap_cont"
    if "vwap_pullback" in t:
        return "vwap_pullback"
    if "prev_low" in t or "prev_high" in t or "sweep" in t or "reclaim" in t or "reject" in t:
        return "sweep_reclaim_reject"
    if "late_trend" in t or "trend_follow" in t:
        return "trend_follow"
    if "orb" in t or "opening" in t or "10_or" in t:
        return "or_retest"
    return t[:40] or "unknown_setup"


def row_to_feature_dict(row: pd.Series) -> dict[str, Any]:
    side = _as_str(row, "side", "").lower()
    short = side == "short"
    rs = _as_float(row, "day_relative_strength", 0.0)
    ors = _as_float(row, "open_relative_strength", 0.0)
    vwap = _as_float(row, "vwap_extension_atr", 0.0)
    directional_rs = -rs if short else rs
    directional_ors = -ors if short else ors
    directional_vwap = -vwap if short else vwap
    score = _as_float(row, "candidate_score", _as_float(row, "score", 0.0))
    ts = _timestamp_ny(row)
    return {
        "side": side or "na",
        "setup": _setup_family(_as_str(row, "trigger_type", "")),
        "time_bucket": _time_bucket(ts),
        "score_bucket": _bucket_num(score, [10, 20, 30, 40, 50], ["sc_lt10", "sc_10_20", "sc_20_30", "sc_30_40", "sc_40_50", "sc_ge50"]),
        "rvol_bucket": _bucket_num(_as_float(row, "rvol_time_of_day", 0.0), [0.75, 1.0, 1.5, 2.5, 4.0], ["rv_lt075", "rv_075_1", "rv_1_15", "rv_15_25", "rv_25_4", "rv_ge4"]),
        "atr_bucket": _bucket_num(_as_float(row, "daily_atr14_percent", 0.0), [1.5, 2.5, 4.0, 6.5, 9.0], ["atr_lt15", "atr_15_25", "atr_25_4", "atr_4_65", "atr_65_9", "atr_ge9"]),
        "dir_rs_bucket": _bucket_num(directional_rs, [-2.0, -1.0, 0.0, 1.0, 2.5, 5.0], ["drs_lt_m2", "drs_m2_m1", "drs_m1_0", "drs_0_1", "drs_1_25", "drs_25_5", "drs_ge5"]),
        "dir_ors_bucket": _bucket_num(directional_ors, [-2.0, -1.0, 0.0, 1.0, 2.5, 5.0], ["dors_lt_m2", "dors_m2_m1", "dors_m1_0", "dors_0_1", "dors_1_25", "dors_25_5", "dors_ge5"]),
        "dir_vwap_bucket": _bucket_num(directional_vwap, [-1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 4.0], ["dv_lt_m15", "dv_m15_m05", "dv_m05_0", "dv_0_05", "dv_05_15", "dv_15_25", "dv_25_4", "dv_ge4"]),
        "abs_vwap_bucket": _bucket_num(abs(vwap), [0.5, 1.0, 1.5, 2.5, 4.0], ["av_lt05", "av_05_1", "av_1_15", "av_15_25", "av_25_4", "av_ge4"]),
        "qqq_bucket": _bucket_num(_as_float(row, "qqq_change_from_open", 0.0), [-1.5, -0.75, -0.25, 0.25, 0.75, 1.5], ["qqq_lt_m15", "qqq_m15_m075", "qqq_m075_m025", "qqq_m025_p025", "qqq_p025_p075", "qqq_p075_p15", "qqq_ge15"]),
        "candle_ok": "candle_ok" if bool(row.get("entry_candle_ok", False)) else "candle_na",
        "opposing": "opp_warn" if bool(row.get("opposing_candle_warning", False)) else "opp_none",
    }


STATE_LEVELS = {
    "full": ["side", "setup", "time_bucket", "score_bucket", "rvol_bucket", "atr_bucket", "dir_rs_bucket", "dir_ors_bucket", "dir_vwap_bucket", "abs_vwap_bucket", "qqq_bucket", "candle_ok", "opposing"],
    "core": ["side", "setup", "time_bucket", "rvol_bucket", "atr_bucket", "dir_rs_bucket", "dir_vwap_bucket"],
    "setup_time": ["side", "setup", "time_bucket"],
    "setup": ["side", "setup"],
    "side": ["side"],
    "global": [],
}


def make_state_key(row: pd.Series, level: str = "full") -> str:
    feats = row_to_feature_dict(row)
    cols = STATE_LEVELS.get(level, STATE_LEVELS["full"])
    if not cols:
        return f"{level}:GLOBAL"
    return level + ":" + "|".join(str(feats.get(c, "na")) for c in cols)


def _session_date(row: pd.Series) -> str:
    for col in ["session_date", "date"]:
        if col in row.index:
            val = row.get(col)
            if not pd.isna(val):
                return str(val)[:10]
    ts = _timestamp_ny(row)
    return ts.date().isoformat() if ts is not None else ""


def _sort_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "timestamp" in out.columns:
        out["_sort_ts"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    elif "entry_time" in out.columns:
        out["_sort_ts"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    else:
        out["_sort_ts"] = pd.NaT
    out["_session_date"] = out.apply(_session_date, axis=1)
    out["_score_sort"] = pd.to_numeric(out.get("candidate_score", out.get("score", 0.0)), errors="coerce").fillna(0.0)
    return out.sort_values(["_sort_ts", "_score_sort", "symbol"], ascending=[True, False, True]).reset_index(drop=True)


def _trade_reward(r_multiple: float, cfg: QLearningPolicyConfig) -> float:
    try:
        r = float(r_multiple)
    except Exception:
        r = 0.0
    if not math.isfinite(r):
        r = 0.0
    if cfg.max_abs_reward > 0:
        r = max(-cfg.max_abs_reward, min(cfg.max_abs_reward, r))
    reward = r - 0.5 * float(cfg.kappa) * ((r - float(cfg.reward_mu)) ** 2)
    return float(reward)


def _new_q_row() -> dict[str, float]:
    return {"skip": 0.0, "trade": 0.0}


def train_q_learning_policy(candidates: pd.DataFrame, cfg: QLearningPolicyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or QLearningPolicyConfig()
    df = _sort_candidates(candidates)
    if df.empty:
        raise ValueError("No candidates supplied to Q-learning trainer.")
    if "r_multiple" not in df.columns:
        raise ValueError("Dataset must contain r_multiple outcomes for Q-learning training.")
    # Store counts for hierarchical fallback and diagnostics.
    counts: dict[str, int] = {}
    rewards_by_state: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        reward = _trade_reward(_as_float(row, "r_multiple", 0.0), cfg)
        for level in STATE_LEVELS:
            key = make_state_key(row, level)
            counts[key] = counts.get(key, 0) + 1
            rewards_by_state.setdefault(key, []).append(reward)

    q: dict[str, dict[str, float]] = {key: _new_q_row() for key in counts}
    rng = np.random.default_rng(int(cfg.random_seed))
    rows = list(df.iterrows())
    n = len(rows)
    for _epoch in range(int(cfg.train_epochs)):
        for i, (_, row) in enumerate(rows):
            next_row = rows[i + 1][1] if i + 1 < n else None
            # If the next row is a new session, treat it as terminal so the policy
            # does not overvalue skipping because of unrelated future days.
            same_day_next = next_row is not None and _session_date(row) == _session_date(next_row)
            reward_trade = _trade_reward(_as_float(row, "r_multiple", 0.0), cfg)
            for level in STATE_LEVELS:
                s = make_state_key(row, level)
                if s not in q:
                    q[s] = _new_q_row()
                if same_day_next:
                    ns = make_state_key(next_row, level)  # type: ignore[arg-type]
                    next_val = max(q.get(ns, _new_q_row()).values())
                else:
                    next_val = 0.0
                # Update both actions because this offline candidate environment has
                # complete information: skip has zero immediate reward; trade has the
                # realized R-multiple utility reward.
                q[s]["trade"] += float(cfg.alpha) * (reward_trade + float(cfg.gamma) * next_val - q[s]["trade"])
                q[s]["skip"] += float(cfg.alpha) * (0.0 + float(cfg.gamma) * next_val - q[s]["skip"])
                # A tiny epsilon smoothing term keeps unseen/exploration semantics
                # represented in the table without making training stochastic.
                if float(cfg.epsilon) > 0 and rng.random() < float(cfg.epsilon) / max(1, n):
                    q[s]["skip"] *= 0.999
                    q[s]["trade"] *= 0.999

    state_stats = []
    for key, cnt in counts.items():
        rewards = rewards_by_state.get(key, [])
        qt = q.get(key, _new_q_row())
        state_stats.append({
            "state_key": key,
            "count": cnt,
            "q_skip": qt.get("skip", 0.0),
            "q_trade": qt.get("trade", 0.0),
            "q_edge": qt.get("trade", 0.0) - qt.get("skip", 0.0),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "win_rate_reward_positive": float(np.mean(np.array(rewards) > 0)) if rewards else 0.0,
        })
    model = {
        "model_type": "tabular_q_learning_trade_filter",
        "version": "v37_ritter_q_learning_policy",
        "actions": list(ACTIONS),
        "config": asdict(cfg),
        "state_levels": STATE_LEVELS,
        "q_table": q,
        "counts": counts,
        "state_stats": state_stats,
        "trained_rows": int(len(df)),
        "trained_start": str(df["_sort_ts"].min()),
        "trained_end": str(df["_sort_ts"].max()),
    }
    return model


def save_q_model(model: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # state_stats can be large but still JSON-friendly.
    with path.open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, sort_keys=True, default=str)
    return path


def load_q_model(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_state_for_row(row: pd.Series, model: dict[str, Any], min_state_count: int | None = None) -> tuple[str, str, int]:
    cfg = QLearningPolicyConfig.from_dict(model.get("config", {}))
    min_count = int(min_state_count if min_state_count is not None else cfg.min_state_count)
    counts = model.get("counts", {}) or {}
    q_table = model.get("q_table", {}) or {}
    levels = ["full", "core", "setup_time", "setup", "side", "global"] if cfg.use_hierarchical_fallback else ["full", "global"]
    chosen_key = "global:GLOBAL"
    chosen_level = "global"
    chosen_count = int(counts.get(chosen_key, 0) or 0)
    for level in levels:
        key = make_state_key(row, level)
        cnt = int(counts.get(key, 0) or 0)
        if key in q_table and cnt >= min_count:
            return key, level, cnt
        if key in q_table and cnt > chosen_count:
            chosen_key, chosen_level, chosen_count = key, level, cnt
    return chosen_key, chosen_level, chosen_count


def apply_q_policy(candidates: pd.DataFrame, model: dict[str, Any], min_edge: float | None = None, min_state_count: int | None = None) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    cfg = QLearningPolicyConfig.from_dict(model.get("config", {}))
    threshold = float(min_edge if min_edge is not None else cfg.min_edge)
    q_table = model.get("q_table", {}) or {}
    rows = []
    for _, row in candidates.iterrows():
        key, level, cnt = choose_state_for_row(row, model, min_state_count=min_state_count)
        qrow = q_table.get(key, {"skip": 0.0, "trade": 0.0})
        q_skip = float(qrow.get("skip", 0.0) or 0.0)
        q_trade = float(qrow.get("trade", 0.0) or 0.0)
        edge = q_trade - q_skip
        out = row.to_dict()
        out.update({
            "q_policy_state_key": key,
            "q_policy_state_level": level,
            "q_policy_state_count": cnt,
            "q_policy_skip_value": q_skip,
            "q_policy_trade_value": q_trade,
            "q_policy_edge": edge,
            "q_policy_approved": bool(edge >= threshold),
        })
        rows.append(out)
    return pd.DataFrame(rows)


def live_style_select(df: pd.DataFrame, top_trades_per_day: int = 1, max_symbol_per_day: int = 1, score_col: str = "q_policy_edge") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = _sort_candidates(df)
    if "q_policy_approved" in work.columns:
        work = work[work["q_policy_approved"].fillna(False).astype(bool)].copy()
    if work.empty:
        return work
    selected = []
    for day, g in work.groupby("_session_date", sort=True):
        taken = 0
        per_symbol: dict[str, int] = {}
        for ts, now in g.groupby("_sort_ts", sort=True):
            if taken >= int(top_trades_per_day):
                break
            now = now.copy()
            now["_model_score"] = pd.to_numeric(now.get(score_col, now.get("candidate_score", 0.0)), errors="coerce").fillna(0.0)
            now["_candidate_score"] = pd.to_numeric(now.get("candidate_score", 0.0), errors="coerce").fillna(0.0)
            for _, row in now.sort_values(["_model_score", "_candidate_score"], ascending=[False, False]).iterrows():
                if taken >= int(top_trades_per_day):
                    break
                sym = str(row.get("symbol", "")).upper()
                if not sym:
                    continue
                if per_symbol.get(sym, 0) >= int(max_symbol_per_day):
                    continue
                selected.append(row.to_dict())
                taken += 1
                per_symbol[sym] = per_symbol.get(sym, 0) + 1
    return pd.DataFrame(selected).drop(columns=[c for c in ["_sort_ts", "_score_sort", "_session_date", "_model_score", "_candidate_score"] if c in pd.DataFrame(selected).columns], errors="ignore") if selected else pd.DataFrame()


def summarize_trades(trades: pd.DataFrame, fixed_risk_dollars: float = 100.0) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "trades": 0, "total_r": 0.0, "win_rate": 0.0, "profit_factor": 0.0,
            "expectancy_r": 0.0, "gross_profit_r": 0.0, "gross_loss_r": 0.0,
            "pnl_dollars": 0.0, "max_drawdown_r": 0.0, "trade_days": 0,
        }
    r = pd.to_numeric(trades.get("r_multiple", 0.0), errors="coerce").fillna(0.0)
    equity = r.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    gp = float(r[r > 0].sum())
    gl = float(r[r < 0].sum())
    if "session_date" in trades.columns:
        trade_days = int(trades["session_date"].astype(str).nunique())
    else:
        trade_days = 0
    return {
        "trades": int(len(trades)),
        "total_r": float(r.sum()),
        "win_rate": float((r > 0).mean() * 100.0),
        "profit_factor": float(gp / abs(gl)) if gl < 0 else (999.0 if gp > 0 else 0.0),
        "expectancy_r": float(r.mean()),
        "gross_profit_r": gp,
        "gross_loss_r": gl,
        "pnl_dollars": float(r.sum() * float(fixed_risk_dollars)),
        "max_drawdown_r": float(dd.min()) if len(dd) else 0.0,
        "trade_days": trade_days,
    }


def backtest_q_policy(candidates: pd.DataFrame, model: dict[str, Any], top_trades_per_day: int = 1, max_symbol_per_day: int = 1, min_edge: float | None = None, min_state_count: int | None = None, fixed_risk_dollars: float = 100.0) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    reviewed = apply_q_policy(candidates, model, min_edge=min_edge, min_state_count=min_state_count)
    selected = live_style_select(reviewed, top_trades_per_day=top_trades_per_day, max_symbol_per_day=max_symbol_per_day, score_col="q_policy_edge")
    summary = summarize_trades(selected, fixed_risk_dollars=fixed_risk_dollars)
    summary.update({
        "reviewed_candidates": int(len(reviewed)),
        "approved_candidates": int(reviewed.get("q_policy_approved", pd.Series([], dtype=bool)).fillna(False).astype(bool).sum()) if not reviewed.empty else 0,
        "top_trades_per_day": int(top_trades_per_day),
        "max_symbol_per_day": int(max_symbol_per_day),
        "fixed_risk_dollars": float(fixed_risk_dollars),
    })
    return selected, summary, reviewed


def split_by_dates(df: pd.DataFrame, train_end: str, validate_end: str) -> pd.DataFrame:
    out = df.copy()
    if "session_date" not in out.columns:
        out["session_date"] = out.apply(_session_date, axis=1)
    dates = pd.to_datetime(out["session_date"], errors="coerce")
    train_cut = pd.Timestamp(train_end)
    val_cut = pd.Timestamp(validate_end)
    out["split"] = np.select([dates <= train_cut, dates <= val_cut], ["train", "validate"], default="test")
    return out

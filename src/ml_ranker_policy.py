from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
except Exception:  # pragma: no cover
    ExtraTreesRegressor = RandomForestRegressor = HistGradientBoostingRegressor = None  # type: ignore
    ExtraTreesClassifier = RandomForestClassifier = None  # type: ignore

NY_TZ = "America/New_York"


@dataclass
class MLRankerConfig:
    """Walk-forward candidate ranker configuration.

    The model predicts the expected R-multiple / utility of a candidate generated
    by the deterministic live-safe engine. It is deliberately a rank/filter model,
    not a signal generator: features are limited to data available at the candidate
    timestamp.
    """

    model_type: str = "extra_trees_regressor"  # extra_trees_regressor, random_forest_regressor, hist_gbdt_regressor
    target: str = "utility_r"  # utility_r, r_multiple, win
    kappa: float = 0.25
    n_estimators: int = 300
    max_depth: int = 6
    min_samples_leaf: int = 8
    random_seed: int = 42
    min_pred_r: float = 0.05
    min_pred_win_prob: float = 0.55
    min_train_rows: int = 250
    top_trades_per_day: int = 1
    max_symbol_per_day: int = 1
    bootstrap_samples: int = 1000
    use_symbol_feature: bool = True
    use_setup_feature: bool = True
    use_time_feature: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MLRankerConfig":
        data = data or {}
        allowed = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
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


def timestamp_ny_from_row(row: pd.Series) -> pd.Timestamp | None:
    for col in ["timestamp_ny", "entry_time_et", "entry_time", "timestamp"]:
        if col in row.index:
            try:
                ts = pd.Timestamp(row.get(col))
                if pd.isna(ts):
                    continue
                if ts.tzinfo is None:
                    if col in {"timestamp_ny", "entry_time_et"}:
                        return ts.tz_localize(NY_TZ)
                    return ts.tz_localize("UTC").tz_convert(NY_TZ)
                return ts.tz_convert(NY_TZ)
            except Exception:
                continue
    return None


def session_date_from_row(row: pd.Series) -> str:
    for col in ["session_date", "date"]:
        if col in row.index:
            val = row.get(col)
            if not pd.isna(val):
                return str(val)[:10]
    ts = timestamp_ny_from_row(row)
    return ts.date().isoformat() if ts is not None else ""


def setup_family_from_trigger(trigger: str) -> str:
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


def _minute_of_day(ts: pd.Timestamp | None) -> float:
    if ts is None:
        return float("nan")
    return float(int(ts.hour) * 60 + int(ts.minute))


def ensure_candidate_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "session_date" not in out.columns:
        out["session_date"] = out.apply(session_date_from_row, axis=1)
    if "_sort_ts" not in out.columns:
        if "timestamp" in out.columns:
            out["_sort_ts"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
        elif "entry_time" in out.columns:
            out["_sort_ts"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
        else:
            out["_sort_ts"] = pd.NaT
    if "candidate_score" not in out.columns and "score" in out.columns:
        out["candidate_score"] = pd.to_numeric(out["score"], errors="coerce")
    if "candidate_score" not in out.columns:
        out["candidate_score"] = 0.0
    if "side" not in out.columns and "strategy_side" in out.columns:
        out["side"] = out["strategy_side"]
    if "trigger_type" not in out.columns and "setup" in out.columns:
        out["trigger_type"] = out["setup"]
    return out


def add_live_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add no-lookahead features to candidate rows.

    These features are computed from candidate-row fields created by the raw-bar
    replay/live engine. No outcome columns are used here.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    out = ensure_candidate_columns(df).copy()
    sides = out.get("side", pd.Series("", index=out.index)).astype(str).str.lower()
    short_mult = np.where(sides.eq("short"), -1.0, 1.0)
    def num(col: str, default: float = 0.0) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce").fillna(default).astype(float)
        return pd.Series(default, index=out.index, dtype=float)

    out["ml_direction_mult"] = short_mult
    out["ml_dir_rs"] = num("day_relative_strength") * short_mult
    out["ml_dir_open_rs"] = num("open_relative_strength") * short_mult
    out["ml_dir_vwap_atr"] = num("vwap_extension_atr") * short_mult
    out["ml_abs_vwap_atr"] = num("vwap_extension_atr").abs()
    out["ml_abs_qqq_change"] = num("qqq_change_from_open").abs()
    out["ml_rvol_log"] = np.log1p(num("rvol_time_of_day").clip(lower=0))
    out["ml_atr_pct"] = num("daily_atr14_percent")
    out["ml_signal_risk_pct"] = num("signal_risk_pct", num("risk_pct", 0.0))
    out["ml_score"] = num("candidate_score", num("score", 0.0))
    out["ml_stock_change_open"] = num("stock_change_from_open")
    out["ml_dir_stock_change_open"] = num("stock_change_from_open") * short_mult
    out["ml_news_count_last_3d"] = num("news_count_last_3d")
    out["ml_entry_candle_ok"] = out.get("entry_candle_ok", pd.Series(False, index=out.index)).fillna(False).astype(bool).astype(int)
    out["ml_opposing_candle_warning"] = out.get("opposing_candle_warning", pd.Series(False, index=out.index)).fillna(False).astype(bool).astype(int)
    # Timestamp-derived features.
    ts_list = []
    buckets = []
    for _, row in out.iterrows():
        ts = timestamp_ny_from_row(row)
        ts_list.append(_minute_of_day(ts))
        buckets.append(_time_bucket(ts))
    out["ml_minute_of_day"] = pd.Series(ts_list, index=out.index, dtype=float)
    out["ml_time_bucket"] = buckets
    out["ml_setup_family"] = out.get("trigger_type", pd.Series("", index=out.index)).astype(str).map(setup_family_from_trigger)
    out["ml_symbol"] = out.get("symbol", pd.Series("", index=out.index)).astype(str).str.upper()
    out["ml_side"] = sides.where(sides.ne(""), "na")
    # Intraday benchmark / outperformance-inspired live-safe features.
    out["ml_stock_minus_qqq"] = num("stock_change_from_open") - num("qqq_change_from_open")
    out["ml_dir_stock_minus_qqq"] = out["ml_stock_minus_qqq"] * short_mult
    return out


NUMERIC_FEATURES = [
    "ml_score", "ml_rvol_log", "rvol_time_of_day", "ml_atr_pct", "ml_dir_rs", "ml_dir_open_rs",
    "ml_dir_vwap_atr", "ml_abs_vwap_atr", "ml_abs_qqq_change", "qqq_change_from_open",
    "ml_signal_risk_pct", "ml_stock_change_open", "ml_dir_stock_change_open", "ml_news_count_last_3d",
    "ml_entry_candle_ok", "ml_opposing_candle_warning", "ml_minute_of_day", "ml_stock_minus_qqq",
    "ml_dir_stock_minus_qqq",
]

CATEGORICAL_FEATURES = ["ml_side", "ml_setup_family", "ml_time_bucket", "ml_symbol"]


def make_feature_matrix(df: pd.DataFrame, columns: list[str] | None = None, cfg: MLRankerConfig | None = None) -> tuple[pd.DataFrame, list[str]]:
    cfg = cfg or MLRankerConfig()
    work = add_live_safe_features(df)
    if work.empty:
        return pd.DataFrame(), columns or []
    numeric = [c for c in NUMERIC_FEATURES if c in work.columns]
    cat = []
    for c in CATEGORICAL_FEATURES:
        if c == "ml_symbol" and not cfg.use_symbol_feature:
            continue
        if c == "ml_setup_family" and not cfg.use_setup_feature:
            continue
        if c == "ml_time_bucket" and not cfg.use_time_feature:
            continue
        if c in work.columns:
            cat.append(c)
    X_num = work[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_cat = pd.get_dummies(work[cat].astype(str), columns=cat, dummy_na=True) if cat else pd.DataFrame(index=work.index)
    X = pd.concat([X_num, X_cat], axis=1)
    X.columns = [str(c) for c in X.columns]
    if columns is not None:
        for col in columns:
            if col not in X.columns:
                X[col] = 0.0
        X = X[columns]
        return X, columns
    cols = sorted(X.columns)
    return X[cols], cols


def target_series(df: pd.DataFrame, cfg: MLRankerConfig) -> pd.Series:
    r = pd.to_numeric(df.get("r_multiple", 0.0), errors="coerce").fillna(0.0).clip(-5, 5)
    if str(cfg.target).lower() == "win":
        return (r > 0).astype(int)
    if str(cfg.target).lower() == "utility_r":
        return r - 0.5 * float(cfg.kappa) * (r ** 2)
    return r


def _make_model(cfg: MLRankerConfig):
    mtype = str(cfg.model_type).lower()
    if mtype == "random_forest_regressor":
        if RandomForestRegressor is None:
            raise RuntimeError("scikit-learn is required for random_forest_regressor")
        return RandomForestRegressor(
            n_estimators=int(cfg.n_estimators), max_depth=None if int(cfg.max_depth) <= 0 else int(cfg.max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf), random_state=int(cfg.random_seed), n_jobs=-1,
        )
    if mtype == "hist_gbdt_regressor":
        if HistGradientBoostingRegressor is None:
            raise RuntimeError("scikit-learn is required for hist_gbdt_regressor")
        return HistGradientBoostingRegressor(
            max_iter=max(50, int(cfg.n_estimators)), max_leaf_nodes=31,
            max_depth=None if int(cfg.max_depth) <= 0 else int(cfg.max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf), random_state=int(cfg.random_seed),
        )
    if mtype == "extra_trees_classifier":
        if ExtraTreesClassifier is None:
            raise RuntimeError("scikit-learn is required for extra_trees_classifier")
        return ExtraTreesClassifier(
            n_estimators=int(cfg.n_estimators), max_depth=None if int(cfg.max_depth) <= 0 else int(cfg.max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf), random_state=int(cfg.random_seed), n_jobs=-1,
        )
    if mtype == "random_forest_classifier":
        if RandomForestClassifier is None:
            raise RuntimeError("scikit-learn is required for random_forest_classifier")
        return RandomForestClassifier(
            n_estimators=int(cfg.n_estimators), max_depth=None if int(cfg.max_depth) <= 0 else int(cfg.max_depth),
            min_samples_leaf=int(cfg.min_samples_leaf), random_state=int(cfg.random_seed), n_jobs=-1,
        )
    if ExtraTreesRegressor is None:
        raise RuntimeError("scikit-learn is required for extra_trees_regressor")
    return ExtraTreesRegressor(
        n_estimators=int(cfg.n_estimators), max_depth=None if int(cfg.max_depth) <= 0 else int(cfg.max_depth),
        min_samples_leaf=int(cfg.min_samples_leaf), random_state=int(cfg.random_seed), n_jobs=-1,
    )


def train_ranker_model(train_df: pd.DataFrame, cfg: MLRankerConfig | None = None) -> dict[str, Any]:
    cfg = cfg or MLRankerConfig()
    if train_df is None or train_df.empty:
        raise ValueError("No training rows supplied to ML ranker.")
    if "r_multiple" not in train_df.columns:
        raise ValueError("Training data must contain r_multiple outcomes.")
    X, columns = make_feature_matrix(train_df, cfg=cfg)
    y = target_series(train_df, cfg)
    model = _make_model(cfg)
    model.fit(X, y)
    feature_importance = []
    if hasattr(model, "feature_importances_"):
        vals = getattr(model, "feature_importances_", None)
        if vals is not None:
            feature_importance = [
                {"feature": c, "importance": float(v)} for c, v in sorted(zip(columns, vals), key=lambda x: float(x[1]), reverse=True)
            ]
    return {
        "model_type": "walkforward_ml_trade_ranker",
        "version": "v37_1_pybroker_style_walkforward_ranker",
        "config": asdict(cfg),
        "feature_columns": columns,
        "model": model,
        "trained_rows": int(len(train_df)),
        "trained_start": str(pd.to_datetime(train_df.get("session_date", pd.Series(dtype=str)), errors="coerce").min()),
        "trained_end": str(pd.to_datetime(train_df.get("session_date", pd.Series(dtype=str)), errors="coerce").max()),
        "feature_importance": feature_importance,
    }


def score_candidates(candidates: pd.DataFrame, trained: dict[str, Any]) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame()
    cfg = MLRankerConfig.from_dict(trained.get("config", {}))
    columns = list(trained.get("feature_columns", []))
    model = trained.get("model")
    if model is None:
        raise ValueError("Trained model payload has no model object.")
    out = add_live_safe_features(candidates)
    X, _ = make_feature_matrix(out, columns=columns, cfg=cfg)
    pred = model.predict(X)
    out["ml_pred_r"] = pd.Series(pred, index=out.index, dtype=float)
    # Classifier models expose probability; map positive-class probability into both columns.
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            if probs.shape[1] >= 2:
                out["ml_pred_win_prob"] = probs[:, 1]
                out["ml_pred_r"] = out["ml_pred_win_prob"] - 0.5
        except Exception:
            out["ml_pred_win_prob"] = np.nan
    else:
        # A smooth proxy, useful for threshold scans and display.
        out["ml_pred_win_prob"] = 1.0 / (1.0 + np.exp(-out["ml_pred_r"].clip(-10, 10)))
    return out


def save_ranker_model(trained: dict[str, Any], folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "model.joblib"
    metadata_path = folder / "model.json"
    joblib.dump(trained.get("model"), model_path)
    metadata = {k: v for k, v in trained.items() if k != "model"}
    metadata["model_joblib"] = str(model_path.name)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    fi = pd.DataFrame(metadata.get("feature_importance", []))
    if not fi.empty:
        fi.to_csv(folder / "feature_importance.csv", index=False)
    return metadata_path


def load_ranker_model(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.is_dir():
        metadata_path = path / "model.json"
    else:
        metadata_path = path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_file = metadata.get("model_joblib", "model.joblib")
    model_path = metadata_path.parent / str(model_file)
    metadata["model"] = joblib.load(model_path)
    return metadata


def live_style_select_by_score(df: pd.DataFrame, score_col: str = "ml_pred_r", threshold: float = 0.0, top_trades_per_day: int = 1, max_symbol_per_day: int = 1, min_win_prob: float | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = ensure_candidate_columns(df).copy()
    work["_score"] = pd.to_numeric(work.get(score_col, 0.0), errors="coerce").fillna(-9999.0)
    if min_win_prob is not None and "ml_pred_win_prob" in work.columns:
        work = work[pd.to_numeric(work["ml_pred_win_prob"], errors="coerce").fillna(0.0) >= float(min_win_prob)].copy()
    work = work[work["_score"] >= float(threshold)].copy()
    if work.empty:
        return work.drop(columns=["_score"], errors="ignore")
    work["_session_date"] = work.apply(session_date_from_row, axis=1)
    work = work.sort_values(["_sort_ts", "_score", "candidate_score", "symbol"], ascending=[True, False, False, True])
    selected: list[dict[str, Any]] = []
    for day, g in work.groupby("_session_date", sort=True):
        taken = 0
        per_symbol: dict[str, int] = {}
        for ts, now in g.groupby("_sort_ts", sort=True):
            if taken >= int(top_trades_per_day):
                break
            for _, row in now.sort_values(["_score", "candidate_score"], ascending=[False, False]).iterrows():
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
    if not selected:
        return pd.DataFrame()
    return pd.DataFrame(selected).drop(columns=["_score", "_session_date"], errors="ignore")


def summarize_trades(trades: pd.DataFrame, fixed_risk_dollars: float = 100.0) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "trades": 0, "total_r": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy_r": 0.0,
            "gross_profit_r": 0.0, "gross_loss_r": 0.0, "max_drawdown_r": 0.0, "pnl_dollars": 0.0,
            "trade_days": 0,
        }
    r = pd.to_numeric(trades.get("r_multiple", 0.0), errors="coerce").fillna(0.0)
    equity = r.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    gp = float(r[r > 0].sum())
    gl = float(r[r < 0].sum())
    trade_days = int(trades.get("session_date", pd.Series(dtype=str)).astype(str).nunique()) if "session_date" in trades.columns else 0
    return {
        "trades": int(len(trades)),
        "total_r": float(r.sum()),
        "win_rate": float((r > 0).mean() * 100.0),
        "profit_factor": float(gp / abs(gl)) if gl < 0 else (999.0 if gp > 0 else 0.0),
        "expectancy_r": float(r.mean()),
        "gross_profit_r": gp,
        "gross_loss_r": gl,
        "max_drawdown_r": float(dd.min()) if len(dd) else 0.0,
        "pnl_dollars": float(r.sum() * float(fixed_risk_dollars)),
        "trade_days": trade_days,
    }


def bootstrap_summary(trades: pd.DataFrame, samples: int = 1000, seed: int = 42) -> dict[str, Any]:
    if trades is None or trades.empty or int(samples) <= 0:
        return {"bootstrap_samples": int(samples), "total_r_p05": 0.0, "total_r_p50": 0.0, "total_r_p95": 0.0, "expectancy_r_p05": 0.0, "expectancy_r_p95": 0.0}
    r = pd.to_numeric(trades.get("r_multiple", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    totals = []
    means = []
    n = len(r)
    for _ in range(int(samples)):
        sample = rng.choice(r, size=n, replace=True)
        totals.append(float(np.sum(sample)))
        means.append(float(np.mean(sample)))
    return {
        "bootstrap_samples": int(samples),
        "total_r_p05": float(np.percentile(totals, 5)),
        "total_r_p50": float(np.percentile(totals, 50)),
        "total_r_p95": float(np.percentile(totals, 95)),
        "expectancy_r_p05": float(np.percentile(means, 5)),
        "expectancy_r_p50": float(np.percentile(means, 50)),
        "expectancy_r_p95": float(np.percentile(means, 95)),
    }


def scan_thresholds(scored: pd.DataFrame, thresholds: Iterable[float], top_trades_per_day: int = 1, max_symbol_per_day: int = 1, fixed_risk_dollars: float = 100.0, min_win_prob: float | None = None) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        selected = live_style_select_by_score(scored, threshold=float(thr), top_trades_per_day=top_trades_per_day, max_symbol_per_day=max_symbol_per_day, min_win_prob=min_win_prob)
        summary = summarize_trades(selected, fixed_risk_dollars=fixed_risk_dollars)
        summary.update({"threshold": float(thr), "top_trades_per_day": int(top_trades_per_day), "max_symbol_per_day": int(max_symbol_per_day)})
        # Robustness objective: positive R, PF, and low drawdown; enough trades.
        summary["objective"] = float(summary["total_r"]) + 0.25 * float(summary["profit_factor"]) + 0.15 * min(50, int(summary["trades"])) + float(summary["max_drawdown_r"])
        rows.append(summary)
    return pd.DataFrame(rows).sort_values(["objective", "total_r", "trades"], ascending=[False, False, False]).reset_index(drop=True)

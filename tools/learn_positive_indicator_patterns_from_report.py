from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FEATURES = [
    "minute_et",
    "candidate_score",
    "rvol_time_of_day",
    "daily_atr14_percent",
    "dir_rs",
    "dir_open_rs",
    "dir_vwap",
    "abs_gap",
    "abs_qqq",
    "abs_vwap",
]


def _read_report(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            name = "selected_trade_market_conditions.csv"
            if name not in z.namelist():
                name = "selected_trades.csv"
            return pd.read_csv(io.BytesIO(z.read(name)))
    return pd.read_csv(path)


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time_col = "entry_time_et" if "entry_time_et" in out.columns else "signal_time_et" if "signal_time_et" in out.columns else None
    if time_col:
        ts = pd.to_datetime(out[time_col], errors="coerce")
    elif "timestamp" in out.columns:
        ts = pd.to_datetime(out["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    else:
        ts = pd.Series(pd.NaT, index=out.index)
    out["minute_et"] = (ts.dt.hour * 60 + ts.dt.minute).astype(float)
    out["year"] = ts.dt.year
    aliases = {
        "rvol_time_of_day": "rvol_tod",
        "daily_atr14_percent": "daily_atr_pct",
        "day_relative_strength": "rs_open",
        "open_relative_strength": "rs_open",
        "vwap_extension_atr": "vwap_ext_atr",
        "gap_percent": "gap_pct",
        "qqq_day_change_percent": "qqq_chg_open",
        "candidate_score": "score",
    }
    for col, alt in aliases.items():
        if col not in out.columns:
            out[col] = out[alt] if alt in out.columns else np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    side_short = out.get("side", pd.Series("", index=out.index)).fillna("").astype(str).str.lower().eq("short")
    mult = np.where(side_short, -1.0, 1.0)
    out["dir_rs"] = mult * out["day_relative_strength"]
    out["dir_open_rs"] = mult * out["open_relative_strength"]
    out["dir_vwap"] = mult * out["vwap_extension_atr"]
    out["abs_vwap"] = out["vwap_extension_atr"].abs()
    out["abs_gap"] = out["gap_percent"].abs()
    out["abs_qqq"] = out["qqq_day_change_percent"].abs()
    if "trigger_type" not in out.columns:
        out["trigger_type"] = out.get("event", "").astype(str).map(lambda x: f"v25_{x}" if not str(x).startswith("v25_") else str(x))
    out["event"] = out["trigger_type"].astype(str).str.replace("v25_", "", regex=False)
    out["r_multiple"] = pd.to_numeric(out["r_multiple"], errors="coerce")
    return out.dropna(subset=["r_multiple", "symbol", "side", "trigger_type"])


def _metrics(g: pd.DataFrame) -> dict[str, Any]:
    n = int(len(g))
    if n == 0:
        return {"n": 0, "wr": 0.0, "total_r": 0.0, "avg_r": 0.0, "pf": 0.0, "losses": 0, "years": 0, "years_pos": 0}
    wins = g["r_multiple"] > 0
    gp = float(g.loc[wins, "r_multiple"].sum())
    gl = float(-g.loc[~wins, "r_multiple"].sum())
    years = g.groupby("year", dropna=True)["r_multiple"].sum()
    return {
        "n": n,
        "wr": float(wins.mean() * 100.0),
        "total_r": float(g["r_multiple"].sum()),
        "avg_r": float(g["r_multiple"].mean()),
        "pf": float(gp / gl) if gl > 0 else (999.0 if gp > 0 else 0.0),
        "losses": int((~wins).sum()),
        "years": int(years.shape[0]),
        "years_pos": int((years > 0).sum()),
    }


def _intervals(g: pd.DataFrame, feat: str, min_n: int) -> list[dict[str, Any]]:
    vals = pd.to_numeric(g[feat], errors="coerce").dropna()
    if len(vals) < min_n:
        return []
    qs = np.unique(np.nanquantile(vals, np.linspace(0, 1, 9)))
    out: list[dict[str, Any]] = []
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            lo, hi = float(qs[i]), float(qs[j])
            sub = g[g[feat].between(lo, hi, inclusive="both")]
            m = _metrics(sub)
            if m["n"] >= min_n and m["total_r"] > 0 and m["pf"] >= 1.25 and m["wr"] >= 60:
                out.append({"feature": feat, "lo": lo, "hi": hi, **m})
    return sorted(out, key=lambda x: (x["total_r"], x["pf"], x["n"]), reverse=True)[:4]


def mine(df: pd.DataFrame, min_group_n: int, max_profiles: int) -> dict[str, Any]:
    profiles: list[dict[str, Any]] = []
    for keys, g in df.groupby(["symbol", "side", "trigger_type"]):
        base = _metrics(g)
        if base["n"] < min_group_n or base["total_r"] <= 0 or base["pf"] < 1.05:
            continue
        min_n = max(8, int(round(base["n"] * 0.20)))
        intervals = []
        for feat in FEATURES:
            if feat in g.columns:
                intervals.extend(_intervals(g, feat, min_n))
        intervals = sorted(intervals, key=lambda x: (x["total_r"], x["pf"], x["n"]), reverse=True)[:16]
        candidates = []
        for a_i, a in enumerate(intervals):
            for b in intervals[a_i + 1 :]:
                if a["feature"] != b["feature"]:
                    candidates.append([a, b])
        for a_i, a in enumerate(intervals[:8]):
            for b_i, b in enumerate(intervals[a_i + 1 : 8], a_i + 1):
                for c in intervals[b_i + 1 : 8]:
                    if len({a["feature"], b["feature"], c["feature"]}) == 3:
                        candidates.append([a, b, c])
        best = []
        for combo in candidates:
            mask = pd.Series(True, index=g.index)
            for it in combo:
                mask &= g[it["feature"]].between(float(it["lo"]), float(it["hi"]), inclusive="both")
            sub = g[mask]
            m = _metrics(sub)
            if m["n"] < min_n or m["total_r"] <= 0 or m["pf"] < 1.40 or m["wr"] < 63:
                continue
            if base["years"] >= 4 and m["years_pos"] < min(4, m["years"]):
                continue
            score = m["total_r"] + m["n"] * 0.25 + min(m["pf"], 10.0) * 2.0 + m["years_pos"] * 2.0 - len(combo)
            best.append((score, combo, m))
        for rank, (score, combo, m) in enumerate(sorted(best, key=lambda x: x[0], reverse=True)[:2], 1):
            symbol, side, trigger = keys
            event = str(trigger).replace("v25_", "")
            profiles.append(
                {
                    "name": f"{symbol}_{event}_{rank}",
                    "symbol": str(symbol).upper(),
                    "side": str(side).lower(),
                    "trigger_type": str(trigger),
                    "event": event,
                    "rules": {it["feature"]: [round(float(it["lo"]), 6), round(float(it["hi"]), 6)] for it in combo},
                    "source_group_metrics": base,
                    "profile_metrics": m,
                    "pattern_score": float(score),
                    "rule_count": len(combo),
                }
            )
    profiles = sorted(profiles, key=lambda p: (p["profile_metrics"]["years_pos"], p["profile_metrics"]["total_r"], p["profile_metrics"]["pf"]), reverse=True)[:max_profiles]
    return {
        "version": "custom_mined_positive_indicator_patterns",
        "description": "Mined multi-indicator signal-time patterns from a backtest report. No future fields are used at decision time.",
        "profiles": profiles,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-group-n", type=int, default=18)
    ap.add_argument("--max-profiles", type=int, default=80)
    args = ap.parse_args()
    df = _prep(_read_report(Path(args.report)))
    result = mine(df, args.min_group_n, args.max_profiles)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {len(result['profiles'])} profiles to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

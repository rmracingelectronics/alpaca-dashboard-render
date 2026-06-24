from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from requests import exceptions as req_exc


OPENAI_RESPONSES_URL = os.getenv("OPENAI_RESPONSES_URL", "https://api.openai.com/v1/responses")


@dataclass
class OpenAITradeFilterResult:
    candidates: pd.DataFrame
    decisions: pd.DataFrame
    diagnostics: dict[str, Any]


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return str(value)


def _extract_response_text(payload: dict[str, Any]) -> str:
    text = payload.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content, dict):
                if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                    parts.append(content.get("text", ""))
    return "\n".join(parts).strip()


def _candidate_payload(row: pd.Series) -> dict[str, Any]:
    """Create a no-lookahead payload for OpenAI.

    The backtest candidate row contains future outcome fields after simulation.  Do not
    send P&L, exit, MFE/MAE, target hit or result columns to the model.  The model gets
    only signal/entry-time information that would be known at decision time.
    """
    entry_time = pd.Timestamp(row.get("entry_time")) if row.get("entry_time") is not None else None
    entry_time_ny = ""
    if entry_time is not None and not pd.isna(entry_time):
        try:
            if entry_time.tzinfo is None:
                entry_time = entry_time.tz_localize("UTC")
            entry_time_ny = entry_time.tz_convert("America/New_York").isoformat()
        except Exception:
            entry_time_ny = str(row.get("entry_time"))
    data = {
        "candidate_id": int(row.get("openai_candidate_id")),
        "symbol": _safe_str(row.get("symbol")).upper(),
        "side": _safe_str(row.get("side")).lower(),
        "entry_time_ny": entry_time_ny,
        "session_date": _safe_str(row.get("session_date")),
        "trigger_type": _safe_str(row.get("trigger_type")),
        "setup_family": _safe_str(row.get("setup_family")),
        "quality": _safe_str(row.get("quality")),
        "candidate_score": _safe_float(row.get("candidate_score")),
        "fallback_score": _safe_float(row.get("fallback_score")),
        "supporting_score": _safe_float(row.get("supporting_score")),
        "entry_price": _safe_float(row.get("entry_price")),
        "risk_per_share": _safe_float(row.get("risk_per_share")),
        "risk_percent_of_price": _safe_float(row.get("risk_percent_of_price")),
        "rvol_time_of_day": _safe_float(row.get("rvol_time_of_day")),
        "current_5m_dollar_volume": _safe_float(row.get("current_5m_dollar_volume")),
        "avg_20d_dollar_volume": _safe_float(row.get("avg_20d_dollar_volume")),
        "gap_percent": _safe_float(row.get("gap_percent")),
        "stock_change_from_open": _safe_float(row.get("stock_change_from_open")),
        "day_relative_strength": _safe_float(row.get("day_relative_strength")),
        "open_relative_strength": _safe_float(row.get("open_relative_strength")),
        "vwap_extension_atr": _safe_float(row.get("vwap_extension_atr")),
        "candle_close_position": _safe_float(row.get("candle_close_position")),
        "candle_range_atr": _safe_float(row.get("candle_range_atr")),
        "daily_atr14_percent": _safe_float(row.get("daily_atr14_percent")),
        "atr5m14": _safe_float(row.get("atr5m14")),
        "qqq_change_from_open": _safe_float(row.get("qqq_change_from_open")),
        "qqq_15min_change_pct": _safe_float(row.get("qqq_15min_change_pct")),
        "news_count_last_3d": _safe_float(row.get("news_count_last_3d"), 0.0),
    }
    # Keep the request small and JSON-clean.
    return {k: v for k, v in data.items() if v is not None and v != ""}


_SYSTEM_PROMPT = """You are a real-time intraday trade-quality reviewer for a paper-trading research dashboard.

CRITICAL LIVE-TRADING RULES:
- Each candidate is an independent moment-in-time decision.
- For each candidate, pretend the current time is exactly that candidate's entry_time_ny.
- Use only the fields provided for that candidate. Do not infer, assume, or wait for anything that happens after that timestamp.
- Candidates may be batched together only to reduce API cost. Do NOT rank them against each other.
- Do NOT choose the best trade of the day, best trade of the batch, or best later opportunity. In live mode, the future candidates would not be known.
- Do NOT reject a valid candidate because a later candidate in the batch looks better.
- Do NOT approve a candidate because the system needs trades. No trade is acceptable.

Your task is to decide whether each algorithm-generated candidate is objectively strong enough to enter at its own timestamp.
Act as a veto/confirmation layer, not as a portfolio optimizer. The algorithm has already found and scored candidates; reject only when there are clear current-time red flags or the setup is not strong enough on its own.

Review live-available quality: setup/side consistency, volume confirmation, relative strength/weakness, VWAP alignment for that setup type, candle quality if provided, QQQ stress, liquidity, spread/risk feasibility if provided, and time-of-day quality.

Keep each reason concise, maximum one sentence.
Return JSON only following the schema. Do not include prose outside JSON."""


def _schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "candidate_id": {"type": "integer"},
                        "trade": {"type": "boolean"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "reason": {"type": "string"},
                    },
                    "required": ["candidate_id", "trade", "confidence", "reason"],
                },
            }
        },
        "required": ["decisions"],
    }


def _call_openai(payload: dict[str, Any], model: str, timeout: int | None = None) -> dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env before enabling the OpenAI trade filter.")

    read_timeout = int(float(os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", str(timeout or 180))))
    connect_timeout = int(float(os.getenv("OPENAI_CONNECT_TIMEOUT_SECONDS", "10")))
    retries = max(0, int(os.getenv("OPENAI_REQUEST_RETRIES", "3")))
    retry_sleep_base = float(os.getenv("OPENAI_RETRY_SLEEP_SECONDS", "2"))
    retryable_statuses = {408, 409, 429, 500, 502, 503, 504}

    request_body = {
        "model": model,
        "input": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"), default=str)},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "trade_filter_decisions",
                "schema": _schema(),
                "strict": True,
            }
        },
    }
    max_output_tokens = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "").strip()
    if max_output_tokens:
        try:
            request_body["max_output_tokens"] = int(max_output_tokens)
        except Exception:
            pass

    last_error: Exception | None = None
    last_response_text = ""
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                OPENAI_RESPONSES_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=request_body,
                timeout=(connect_timeout, read_timeout),
            )
            if resp.status_code >= 400:
                last_response_text = resp.text[:1000]
                if resp.status_code in retryable_statuses and attempt < retries:
                    time.sleep(retry_sleep_base * (2 ** attempt))
                    continue
                raise RuntimeError(f"OpenAI API request failed {resp.status_code}: {last_response_text}")
            text = _extract_response_text(resp.json())
            if not text:
                raise RuntimeError("OpenAI API returned no parseable text output.")
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"OpenAI API returned non-JSON output: {text[:700]}") from exc
        except (req_exc.Timeout, req_exc.ConnectionError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep_base * (2 ** attempt))
                continue
            break

    msg = (
        "OpenAI API request timed out or failed after "
        f"{retries + 1} attempt(s). Current timeouts: connect={connect_timeout}s, read={read_timeout}s. "
        "For large batches, either increase OPENAI_REQUEST_TIMEOUT_SECONDS or lower "
        "'AI max independent decisions per API call'."
    )
    if last_response_text:
        msg += f" Last response: {last_response_text[:700]}"
    if last_error is not None:
        msg += f" Last error: {last_error}"
    raise RuntimeError(msg)


def _normalize_openai_decisions(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize one OpenAI JSON response into internal decision rows."""
    rows: list[dict[str, Any]] = []
    for dec in parsed.get("decisions", []) or []:
        try:
            cid = int(dec.get("candidate_id"))
        except Exception:
            continue
        rows.append({
            "openai_candidate_id": cid,
            "openai_trade": bool(dec.get("trade", False)),
            "openai_confidence": _safe_float(dec.get("confidence"), 0.0) or 0.0,
            "openai_reason": _safe_str(dec.get("reason"))[:600],
        })
    return rows


def _iter_batches(df: pd.DataFrame, batch_size: int):
    batch_size = max(1, int(batch_size or 200))
    for start in range(0, len(df), batch_size):
        yield start // batch_size + 1, df.iloc[start:start + batch_size].copy()


def review_candidates_with_openai(
    candidates: pd.DataFrame,
    max_trades_per_day: int,
    model: str = "gpt-5-mini",
    max_candidates_per_day: int = 25,
    min_confidence: float = 0.0,
    batch_mode: str = "full_run",
) -> OpenAITradeFilterResult:
    """Review algorithm-created candidates with OpenAI.

    V35.3 keeps API batching for cost/speed, but the prompt now treats every
    candidate as an independent real-time decision at its own entry timestamp.
    This avoids the unrealistic "best of the day" / "best of the batch" logic.

    The existing dashboard field is kept for compatibility but is now interpreted
    as "AI max independent decisions per API call".  If a very large backtest
    creates more candidates than that value, the code sends multiple large chunks,
    never one candidate at a time.
    """
    if candidates is None or candidates.empty:
        return OpenAITradeFilterResult(candidates=pd.DataFrame(), decisions=pd.DataFrame(), diagnostics={"enabled": True, "reviewed": 0, "approved": 0, "rejected": 0})

    work = candidates.copy().reset_index(drop=True)
    work["openai_candidate_id"] = range(1, len(work) + 1)
    decisions: list[dict[str, Any]] = []
    api_calls = 0
    prompt_candidate_counts: list[int] = []
    batch_mode_s = str(batch_mode or "full_run").lower().strip()
    prompt_limit = max(1, int(max_candidates_per_day or 200))

    # Keep API chunks in chronological order to mirror a live replay.
    # Candidate decisions are still independent; ordering is for audit/readability.
    sort_cols = [c for c in ["entry_time", "session_date", "candidate_score"] if c in work.columns]
    if sort_cols:
        ascending = [True if c != "candidate_score" else False for c in sort_cols]
        ordered = work.sort_values(sort_cols, ascending=ascending).copy()
    else:
        ordered = work.copy()

    if batch_mode_s in {"per_day", "daily"}:
        # Compatibility mode: one request per day, with up to N candidates/day.
        if "session_date" not in ordered.columns:
            ordered["session_date"] = "all"
        for session_date, group in ordered.groupby("session_date", sort=True):
            daily = group.head(prompt_limit).copy()
            if daily.empty:
                continue
            payload = {
                "research_mode": True,
                "batch_mode": "per_day",
                "session_date": str(session_date),
                "max_trades_allowed_for_day": int(max_trades_per_day or 2),
                "review_mode": "realtime_independent_decisions",
                "instruction": "Return one independent approve/reject decision per candidate_id. Treat each candidate as if reviewed live at its own entry_time_ny. This daily batch exists only to reduce API calls; do not rank candidates or choose a best trade of the day. Approve only if the candidate is strong enough to trade at that exact timestamp using only the provided live-available data. You may approve zero, one, or many candidates. No future outcome is provided.",
                "candidates": [_candidate_payload(row) for _, row in daily.iterrows()],
            }
            parsed = _call_openai(payload, model=model)
            api_calls += 1
            prompt_candidate_counts.append(len(daily))
            decisions.extend(_normalize_openai_decisions(parsed))
    else:
        # Preferred V35.3 mode: send large API batches for cost/speed, but
        # every candidate must be reviewed as an independent real-time decision.
        total_batches = (len(ordered) + prompt_limit - 1) // prompt_limit
        for batch_idx, batch in _iter_batches(ordered, prompt_limit):
            if batch.empty:
                continue
            dates = []
            if "session_date" in batch.columns:
                dates = [str(x) for x in sorted(batch["session_date"].dropna().unique().tolist())]
            payload = {
                "research_mode": True,
                "review_mode": "realtime_independent_decisions",
                "batch_mode": "api_cost_batch_not_ranking_batch",
                "batch_index": int(batch_idx),
                "batch_count": int(total_batches),
                "date_range_in_batch": dates[:200],
                "max_trades_per_day_is_applied_by_software_after_review": int(max_trades_per_day or 2),
                "instruction": "Return one independent approve/reject decision per candidate_id. Treat each candidate as if reviewed live at its own entry_time_ny. The batch exists only to reduce API calls; do not compare candidates against later candidates, do not choose the best of the day, and do not rank the batch. Approve a candidate only if it is strong enough to trade at that exact timestamp using only the provided live-available data. You may approve zero, one, or many candidates. No future outcome is provided.",
                "candidates": [_candidate_payload(row) for _, row in batch.iterrows()],
            }
            parsed = _call_openai(payload, model=model)
            api_calls += 1
            prompt_candidate_counts.append(len(batch))
            decisions.extend(_normalize_openai_decisions(parsed))

    dec_df = pd.DataFrame(decisions)
    if dec_df.empty:
        work["openai_trade"] = False
        work["openai_confidence"] = 0.0
        work["openai_reason"] = "OpenAI returned no decision for this candidate."
    else:
        dec_df = dec_df.drop_duplicates(subset=["openai_candidate_id"], keep="last")
        work = work.merge(dec_df, on="openai_candidate_id", how="left")
        work["openai_trade"] = work["openai_trade"].fillna(False).astype(bool)
        work["openai_confidence"] = pd.to_numeric(work["openai_confidence"], errors="coerce").fillna(0.0)
        work["openai_reason"] = work["openai_reason"].fillna("Not reviewed or not approved by OpenAI.")

    approved_mask = work["openai_trade"] & (work["openai_confidence"] >= float(min_confidence or 0.0))
    reviewed_ids = set(dec_df["openai_candidate_id"].astype(int).tolist()) if not dec_df.empty else set()
    work["openai_reviewed"] = work["openai_candidate_id"].isin(reviewed_ids)
    filtered = work.loc[approved_mask].copy().reset_index(drop=True)
    diagnostics = {
        "enabled": True,
        "model": model,
        "batch_mode": "per_day" if batch_mode_s in {"per_day", "daily"} else "realtime_independent_batch",
        "review_mode": "realtime_independent_decisions",
        "api_calls": api_calls,
        "prompt_candidate_counts": prompt_candidate_counts,
        "largest_prompt_candidates": int(max(prompt_candidate_counts) if prompt_candidate_counts else 0),
        "input_candidates": int(len(candidates)),
        "reviewed": int(work["openai_reviewed"].sum()),
        "approved": int(len(filtered)),
        "rejected": int(len(work) - len(filtered)),
        "max_candidates_per_prompt": int(prompt_limit),
        "min_confidence": float(min_confidence or 0.0),
    }
    return OpenAITradeFilterResult(candidates=filtered, decisions=work, diagnostics=diagnostics)

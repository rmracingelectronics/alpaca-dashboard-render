from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
import requests

from .config import AlpacaSettings, CACHE_DIR, LOCAL_BARS_DIR


class AlpacaAPIError(RuntimeError):
    pass


class AlpacaDataClient:
    """REST client with a persistent local OHLCV store.

    V16/V15 local-store design:
    - Bars are stored locally by symbol/timeframe/feed/adjustment, independent of
      the exact backtest date range.
    - A long backtest first checks the local store and only pulls missing yearly
      chunks from Alpaca.
    - Bulk prefetch downloads missing symbols for a whole year/timeframe in one
      paginated request instead of one symbol at a time.

    This means the first 4-year test still needs to download missing data, but
    every later test over overlapping periods should run almost completely from
    disk.
    """

    def __init__(self, settings: AlpacaSettings | None = None, timeout: int = 30):
        self.settings = settings or AlpacaSettings()
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def headers(self) -> Dict[str, str]:
        if not self.settings.api_key or not self.settings.secret_key:
            raise AlpacaAPIError(
                "Missing ALPACA_API_KEY or ALPACA_SECRET_KEY. Copy .env.example to .env and fill in your keys."
            )
        return {
            "APCA-API-KEY-ID": self.settings.api_key,
            "APCA-API-SECRET-KEY": self.settings.secret_key,
            "Accept": "application/json",
        }

    def _get(self, path: str, params: Dict[str, Any], max_retries: int = 6) -> Dict[str, Any]:
        url = f"{self.settings.data_base_url.rstrip('/')}{path}"
        attempt = 0
        while True:
            response = self.session.get(url, headers=self.headers, params=params, timeout=self.timeout)
            if response.status_code == 429 and attempt < max_retries:
                wait = min(2 ** attempt, 20)
                time.sleep(wait)
                attempt += 1
                continue
            if response.status_code >= 400:
                raise AlpacaAPIError(f"Alpaca request failed {response.status_code}: {response.text[:500]}")
            return response.json()

    @staticmethod
    def _cache_path(prefix: str, params: Dict[str, Any]) -> Path:
        key = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
        return CACHE_DIR / f"{prefix}_{digest}.csv"

    @staticmethod
    def _parse_bars_payload(payload: Dict[str, Any], symbols: Iterable[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        bars = payload.get("bars", {}) or {}
        for symbol in symbols:
            for item in bars.get(symbol, []) or []:
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": item.get("t"),
                        "open": item.get("o"),
                        "high": item.get("h"),
                        "low": item.get("l"),
                        "close": item.get("c"),
                        "volume": item.get("v"),
                        "trade_count": item.get("n"),
                        "vwap": item.get("vw"),
                    }
                )
        if not rows:
            return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="ISO8601")
        return AlpacaDataClient._normalize_bars_df(df)

    @staticmethod
    def _normalize_bars_df(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
        out = df.copy()
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce", format="ISO8601")
        out = out.dropna(subset=["timestamp"])
        for col in ["open", "high", "low", "close", "vwap"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce", downcast="float")
        for col in ["volume", "trade_count"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
                out[col] = pd.to_numeric(out[col], downcast="integer")
        out = out.drop_duplicates(subset=["symbol", "timestamp"]).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        return out

    @staticmethod
    def _safe_part(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))

    def _local_store_path(self, symbol: str, timeframe: str, feed: str, adjustment: str) -> Path:
        return (
            LOCAL_BARS_DIR
            / self._safe_part(feed.lower())
            / self._safe_part(timeframe)
            / self._safe_part(adjustment)
            / f"{symbol.upper()}.pkl.gz"
        )

    def _read_local_symbol(self, symbol: str, timeframe: str, feed: str, adjustment: str) -> pd.DataFrame:
        path = self._local_store_path(symbol, timeframe, feed, adjustment)
        # V16 uses gzip-compressed pickle files by default. For backward
        # compatibility, also read the older uncompressed .pkl local-store files.
        candidates = [path]
        if path.name.endswith(".pkl.gz"):
            candidates.append(path.with_suffix(""))  # old .pkl path
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                df = pd.read_pickle(candidate)
                return self._normalize_bars_df(df)
            except Exception:
                continue
        return pd.DataFrame()

    def _write_local_symbol(self, symbol: str, timeframe: str, feed: str, adjustment: str, df: pd.DataFrame) -> None:
        path = self._local_store_path(symbol, timeframe, feed, adjustment)
        path.parent.mkdir(parents=True, exist_ok=True)
        clean = self._normalize_bars_df(df)
        if not clean.empty:
            clean = clean[clean["symbol"].astype(str).str.upper() == symbol.upper()].copy()
        # gzip-compressed pickle keeps the local cache much smaller than the
        # previous raw pickle files and requires no extra dependencies.
        clean.to_pickle(path, compression="gzip")

    @classmethod
    def _to_utc_timestamp(cls, value: str | datetime) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    @classmethod
    def _range_days(cls, start: str | datetime, end: str | datetime) -> int:
        start_ts = cls._to_utc_timestamp(start)
        end_ts = cls._to_utc_timestamp(end)
        return max(0, int((end_ts - start_ts).days))

    @classmethod
    def _year_chunks(cls, start: str | datetime, end: str | datetime) -> list[tuple[str, str]]:
        start_ts = cls._to_utc_timestamp(start).normalize()
        end_ts = cls._to_utc_timestamp(end).normalize()
        chunks: list[tuple[str, str]] = []
        cursor = start_ts
        while cursor < end_ts:
            next_year = pd.Timestamp(year=cursor.year + 1, month=1, day=1, tz="UTC")
            chunk_end = min(next_year, end_ts)
            chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            cursor = chunk_end
        return chunks

    @classmethod
    def _month_chunks(cls, start: str | datetime, end: str | datetime) -> list[tuple[str, str]]:
        """Monthly chunks for session-aware extended-hours gap checks.

        The original yearly chunks are preserved for regular backtests.  Extended
        research needs finer granularity so one existing premarket bar in a year
        does not incorrectly mark the entire year as locally complete.
        """
        start_ts = cls._to_utc_timestamp(start).normalize()
        end_ts = cls._to_utc_timestamp(end).normalize()
        chunks: list[tuple[str, str]] = []
        cursor = start_ts
        while cursor < end_ts:
            next_month = (cursor + pd.offsets.MonthBegin(1)).normalize()
            chunk_end = min(next_month, end_ts)
            chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
            cursor = chunk_end
        return chunks

    def _fetch_bars_api(
        self,
        symbols: list[str],
        timeframe: str,
        start: str | datetime,
        end: str | datetime,
        feed: str,
        adjustment: str,
    ) -> pd.DataFrame:
        symbols = [s.upper() for s in symbols if s]
        if not symbols:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        page_token: Optional[str] = None
        params_base = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": self._format_dt(start),
            "end": self._format_dt(end),
            "feed": feed,
            "adjustment": adjustment,
            "limit": 10000,
            "sort": "asc",
        }
        while True:
            params = dict(params_base)
            if page_token:
                params["page_token"] = page_token
            payload = self._get("/v2/stocks/bars", params)
            frame = self._parse_bars_payload(payload, symbols)
            if not frame.empty:
                frames.append(frame)
            page_token = payload.get("next_page_token")
            if not page_token:
                break
        if not frames:
            return pd.DataFrame()
        return self._normalize_bars_df(pd.concat(frames, ignore_index=True))

    def _fetch_bars_api_tolerant(
        self,
        symbols: list[str],
        timeframe: str,
        start: str | datetime,
        end: str | datetime,
        feed: str,
        adjustment: str,
    ) -> pd.DataFrame:
        """Fetch bars without letting one unsupported/invalid symbol kill a custom run.

        Alpaca can reject a multi-symbol request if one symbol is not available on
        the selected feed.  That is common with custom penny/micro-cap lists.  The
        original preset universe did not hit this much, but custom watchlists need
        graceful degradation: fetch what Alpaca has, skip what it does not.
        """
        symbols = [s.upper() for s in symbols if s]
        if not symbols:
            return pd.DataFrame()
        try:
            return self._fetch_bars_api(symbols, timeframe, start, end, feed, adjustment)
        except AlpacaAPIError:
            if len(symbols) <= 1:
                return pd.DataFrame()
            frames: list[pd.DataFrame] = []
            for symbol in symbols:
                try:
                    frame = self._fetch_bars_api([symbol], timeframe, start, end, feed, adjustment)
                    if frame is not None and not frame.empty:
                        frames.append(frame)
                except AlpacaAPIError:
                    continue
            if not frames:
                return pd.DataFrame()
            return self._normalize_bars_df(pd.concat(frames, ignore_index=True))

    def _symbol_has_chunk(self, symbol: str, timeframe: str, chunk_start: str, chunk_end: str, feed: str, adjustment: str) -> bool:
        df = self._read_local_symbol(symbol, timeframe, feed, adjustment)
        if df.empty:
            return False
        start_ts = self._to_utc_timestamp(chunk_start)
        end_ts = self._to_utc_timestamp(chunk_end)
        mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
        return bool(mask.any())

    def _symbol_has_session_chunk(
        self,
        symbol: str,
        timeframe: str,
        chunk_start: str,
        chunk_end: str,
        feed: str,
        adjustment: str,
        session_mode: str = "regular_only",
    ) -> bool:
        """Return True when the local store already contains bars for the requested session.

        This preserves the original yearly-chunk cache behavior while adding a
        stricter check for extended-hours research.  For extended-hours mode, a
        symbol/year is considered present only if it has at least one bar outside
        the regular 09:30-16:00 ET session.  That prevents an existing
        regular-only cache file from blocking an extended-hours download.
        """
        df = self._read_local_symbol(symbol, timeframe, feed, adjustment)
        if df.empty:
            return False
        start_ts = self._to_utc_timestamp(chunk_start)
        end_ts = self._to_utc_timestamp(chunk_end)
        work = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy()
        if work.empty:
            return False
        mode = str(session_mode or "regular_only").lower()
        if str(timeframe).lower() not in {"5min", "1min", "1m", "5m"}:
            return True
        ts_ny = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
        time_str = ts_ny.dt.strftime("%H:%M")
        regular = (time_str >= "09:30") & (time_str <= "16:00")
        if mode in {"extended_hours", "extended", "prepost"}:
            ext = ((time_str >= "04:00") & (time_str < "09:30")) | ((time_str > "16:00") & (time_str <= "20:00"))
            return bool(ext.any())
        if mode in {"twenty_four_five", "24_5", "all", "all_available"}:
            return bool((~regular).any()) or bool(regular.any())
        return bool(regular.any())

    def prefetch_stock_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: str | datetime,
        end: str | datetime,
        feed: str = "iex",
        adjustment: str = "split",
        use_cache: bool = True,
        session_mode: str = "regular_only",
    ) -> dict[str, Any]:
        """Download missing local bars for symbols/date range and return a status dict."""
        symbols = list(dict.fromkeys([s.upper() for s in symbols if s]))
        if not use_cache:
            return {"mode": "cache_disabled", "symbols": len(symbols), "downloaded_rows": 0}
        downloaded_rows = 0
        downloaded_chunks = 0
        skipped_chunks = 0
        mode_l = str(session_mode or "regular_only").lower()
        chunk_source = self._month_chunks(start, end) if (str(timeframe).lower() in {"5min", "1min", "1m", "5m"} and mode_l in {"extended_hours", "extended", "prepost", "twenty_four_five", "24_5", "all", "all_available"}) else self._year_chunks(start, end)
        for chunk_start, chunk_end in chunk_source:
            if mode_l in {"extended_hours", "extended", "prepost", "twenty_four_five", "24_5", "all", "all_available"}:
                missing = [s for s in symbols if not self._symbol_has_session_chunk(s, timeframe, chunk_start, chunk_end, feed, adjustment, session_mode=session_mode)]
            else:
                missing = [s for s in symbols if not self._symbol_has_chunk(s, timeframe, chunk_start, chunk_end, feed, adjustment)]
            if not missing:
                skipped_chunks += len(symbols)
                continue
            fetched = self._fetch_bars_api_tolerant(missing, timeframe, chunk_start, chunk_end, feed, adjustment)
            downloaded_rows += int(len(fetched))
            downloaded_chunks += len(missing)
            for symbol in missing:
                existing = self._read_local_symbol(symbol, timeframe, feed, adjustment)
                sym_new = fetched[fetched["symbol"] == symbol].copy() if not fetched.empty else pd.DataFrame()
                frames_to_merge = []
                if existing is not None and not existing.empty:
                    frames_to_merge.append(existing)
                if sym_new is not None and not sym_new.empty:
                    frames_to_merge.append(sym_new)
                merged = pd.concat(frames_to_merge, ignore_index=True) if frames_to_merge else pd.DataFrame()
                self._write_local_symbol(symbol, timeframe, feed, adjustment, merged)
        return {
            "mode": "local_bar_store",
            "symbols": len(symbols),
            "timeframe": timeframe,
            "downloaded_rows": downloaded_rows,
            "downloaded_symbol_year_chunks": downloaded_chunks,
            "cached_symbol_year_chunks": skipped_chunks,
        }

    def get_stock_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: str | datetime,
        end: str | datetime,
        feed: str = "iex",
        adjustment: str = "split",
        use_cache: bool = True,
        session_mode: str = "regular_only",
    ) -> pd.DataFrame:
        symbols = list(dict.fromkeys([s.upper() for s in symbols if s]))
        if not symbols:
            return pd.DataFrame()

        if use_cache:
            self.prefetch_stock_bars(symbols, timeframe, start, end, feed=feed, adjustment=adjustment, use_cache=True, session_mode=session_mode)
            start_ts = self._to_utc_timestamp(start)
            end_ts = self._to_utc_timestamp(end)
            frames: list[pd.DataFrame] = []
            for symbol in symbols:
                df = self._read_local_symbol(symbol, timeframe, feed, adjustment)
                if df.empty:
                    continue
                mask = (df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)
                part = df.loc[mask].copy()
                if not part.empty:
                    frames.append(part)
            if not frames:
                return pd.DataFrame()
            return self._normalize_bars_df(pd.concat(frames, ignore_index=True))

        # Non-cache path: direct API request for exactly the requested symbols/range.
        return self._fetch_bars_api_tolerant(symbols, timeframe, start, end, feed, adjustment)

    def get_news_counts_by_day(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        params_base = {
            "symbols": ",".join(symbols),
            "start": self._format_dt(start),
            "end": self._format_dt(end),
            "limit": 50,
            "sort": "asc",
            "include_content": "false",
            "exclude_contentless": "false",
        }
        cache_path = self._cache_path("news", params_base)
        if use_cache and cache_path.exists():
            return pd.read_csv(cache_path, parse_dates=["session_date"])

        rows: list[dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            params = dict(params_base)
            if page_token:
                params["page_token"] = page_token
            payload = self._get("/v1beta1/news", params)
            for article in payload.get("news", []) or []:
                ts = pd.to_datetime(article.get("created_at") or article.get("updated_at"), utc=True, errors="coerce", format="ISO8601")
                if pd.isna(ts):
                    continue
                symbols_for_article = [s.upper() for s in article.get("symbols", []) or []]
                for symbol in symbols_for_article:
                    if symbol in symbols:
                        rows.append({"symbol": symbol, "session_date": ts.date(), "daily_news_count": 1})
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        if not rows:
            out = pd.DataFrame(columns=["symbol", "session_date", "news_count_last_3d"])
        else:
            counts = pd.DataFrame(rows).groupby(["symbol", "session_date"], as_index=False)["daily_news_count"].sum()
            out_frames = []
            for symbol, group in counts.groupby("symbol"):
                group = group.sort_values("session_date").copy()
                date_range = pd.date_range(group["session_date"].min(), group["session_date"].max(), freq="D").date
                dense = pd.DataFrame({"session_date": date_range})
                dense["symbol"] = symbol
                dense = dense.merge(group, on=["symbol", "session_date"], how="left").fillna({"daily_news_count": 0})
                dense["news_count_last_3d"] = dense["daily_news_count"].rolling(3, min_periods=1).sum()
                out_frames.append(dense[["symbol", "session_date", "news_count_last_3d"]])
            out = pd.concat(out_frames, ignore_index=True)
            out["session_date"] = pd.to_datetime(out["session_date"])

        if use_cache:
            out.to_csv(cache_path, index=False)
        return out

    def latest_quotes(self, symbols: list[str], feed: str = "iex") -> pd.DataFrame:
        payload = self._get(
            "/v2/stocks/quotes/latest",
            {"symbols": ",".join(symbols), "feed": feed},
        )
        rows = []
        quotes = payload.get("quotes", {}) or {}
        for symbol, quote in quotes.items():
            bid = quote.get("bp")
            ask = quote.get("ap")
            mid = (bid + ask) / 2 if bid and ask else None
            spread_pct = ((ask - bid) / mid * 100) if mid else None
            rows.append({"symbol": symbol, "bid": bid, "ask": ask, "mid": mid, "spread_percent": spread_pct})
        return pd.DataFrame(rows)

    @staticmethod
    def _format_dt(value: str | datetime) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        return value


def pad_start_for_indicators(start_date: str, days: int = 45) -> str:
    dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    return (dt - timedelta(days=days)).date().isoformat()

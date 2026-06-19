from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import requests

from .config import AlpacaSettings


class AlpacaTradingAPIError(RuntimeError):
    pass


@dataclass
class OrderResult:
    ok: bool
    status: str
    symbol: str
    side: str
    qty: float
    response: dict[str, Any]


class AlpacaTradingClient:
    """Small REST wrapper for Alpaca paper/live trading endpoints.

    The app is deliberately configured for paper trading by default. The worker
    refuses to run against a non-paper base URL unless ALLOW_LIVE_TRADING=true.
    """

    def __init__(self, settings: AlpacaSettings | None = None, timeout: int = 30):
        self.settings = settings or AlpacaSettings()
        self.timeout = timeout
        self.session = requests.Session()

    @property
    def base_url(self) -> str:
        return str(self.settings.trading_base_url or "https://paper-api.alpaca.markets/v2").rstrip("/")

    @property
    def headers(self) -> Dict[str, str]:
        if not self.settings.api_key or not self.settings.secret_key:
            raise AlpacaTradingAPIError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY.")
        return {
            "APCA-API-KEY-ID": self.settings.api_key,
            "APCA-API-SECRET-KEY": self.settings.secret_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any] | list[dict[str, Any]]:
        url = f"{self.base_url}{path}"
        for attempt in range(5):
            response = self.session.request(method, url, headers=self.headers, timeout=self.timeout, **kwargs)
            if response.status_code == 429 and attempt < 4:
                time.sleep(min(2 ** attempt, 15))
                continue
            if response.status_code >= 400:
                raise AlpacaTradingAPIError(f"Alpaca trading request failed {response.status_code}: {response.text[:800]}")
            if not response.text.strip():
                return {}
            return response.json()
        raise AlpacaTradingAPIError("Alpaca trading request failed after retries.")

    def get_account(self) -> dict[str, Any]:
        return dict(self._request("GET", "/account"))

    def get_clock(self) -> dict[str, Any]:
        return dict(self._request("GET", "/clock"))

    def list_positions(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/positions")
        return list(payload) if isinstance(payload, list) else []

    def list_open_orders(self, symbols: Optional[Iterable[str]] = None) -> list[dict[str, Any]]:
        return self.list_orders(status="open", limit=500, nested=True, symbols=symbols)

    def list_orders(
        self,
        status: str = "all",
        limit: int = 200,
        nested: bool = True,
        symbols: Optional[Iterable[str]] = None,
        direction: str = "desc",
        after: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "status": status,
            "limit": max(1, min(int(limit), 500)),
            "nested": "true" if nested else "false",
            "direction": direction,
        }
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        payload = self._request("GET", "/orders", params=params)
        orders = list(payload) if isinstance(payload, list) else []
        if symbols:
            wanted = {s.upper() for s in symbols}
            orders = [o for o in orders if str(o.get("symbol", "")).upper() in wanted]
        return orders


    def list_account_activities(
        self,
        activity_type: str | None = None,
        after: str | None = None,
        until: str | None = None,
        direction: str = "desc",
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """Return Alpaca account activities, commonly FILL records for closed/opened trade fills."""
        path = "/account/activities"
        if activity_type:
            path += f"/{activity_type}"
        params: dict[str, Any] = {
            "direction": direction,
            "page_size": max(1, min(int(page_size), 100)),
        }
        if after:
            params["after"] = after
        if until:
            params["until"] = until
        payload = self._request("GET", path, params=params)
        return list(payload) if isinstance(payload, list) else []

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"/orders/{order_id}")

    def cancel_open_orders_for_symbol(self, symbol: str) -> list[str]:
        symbol = symbol.upper()
        cancelled: list[str] = []
        for order in self.list_open_orders([symbol]):
            oid = str(order.get("id") or "")
            if not oid:
                continue
            try:
                self.cancel_order(oid)
                cancelled.append(oid)
            except Exception:
                continue
        return cancelled

    def close_position(self, symbol: str) -> dict[str, Any]:
        return dict(self._request("DELETE", f"/positions/{symbol.upper()}"))

    @staticmethod
    def _round_price(price: float) -> float:
        if price >= 1:
            return round(float(price), 2)
        return round(float(price), 4)

    @staticmethod
    def _round_qty(qty: float, fractional: bool) -> str:
        if fractional:
            qty = math.floor(float(qty) * 1000.0) / 1000.0
            return f"{max(qty, 0.0):.3f}"
        return str(max(int(math.floor(float(qty))), 0))

    def submit_market_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        take_profit_price: float,
        stop_price: float,
        client_order_id: str,
        fractional: bool = False,
    ) -> OrderResult:
        side = str(side).lower().strip()
        if side not in {"buy", "sell"}:
            raise ValueError("Alpaca order side must be buy or sell.")
        qty_text = self._round_qty(qty, fractional=fractional)
        if float(qty_text) <= 0:
            raise ValueError("Calculated quantity is zero. Increase risk budget or use a symbol with lower risk/share.")
        payload = {
            "symbol": symbol.upper(),
            "qty": qty_text,
            "side": side,
            "type": "market",
            "time_in_force": "day",
            "order_class": "bracket",
            "take_profit": {"limit_price": self._round_price(take_profit_price)},
            "stop_loss": {"stop_price": self._round_price(stop_price)},
            "client_order_id": client_order_id[:48],
        }
        response = dict(self._request("POST", "/orders", json=payload))
        return OrderResult(
            ok=True,
            status=str(response.get("status", "submitted")),
            symbol=symbol.upper(),
            side=side,
            qty=float(qty_text),
            response=response,
        )

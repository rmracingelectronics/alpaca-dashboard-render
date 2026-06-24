from __future__ import annotations

# QQQ is always fetched internally for market regime and relative strength.
MARKET_SYMBOLS = ["QQQ"]
EXCLUDED_PRESET_SYMBOLS = {"PYPL", "V", "DIS", "WMT", "HD", "BRK.B", "BRK-B"}

STARTER_WATCHLIST = [
    "NVDA", "TSLA", "AMD", "PLTR", "SOFI",
    "COIN", "HOOD", "RIVN", "AAPL", "META",
]

# Balanced 20-name list. Good for quick testing.
DEFAULT_WATCHLIST = [
    "NVDA", "TSLA", "AMD", "PLTR", "SOFI",
    "COIN", "MARA", "HOOD", "RIVN", "SMCI",
    "AAPL", "MSFT", "META", "AMZN", "GOOGL",
    "NFLX", "AVGO", "INTC", "MU", "UBER",
]

# Higher-beta list. More alerts, bigger swings. Use small risk/trade.
HIGH_BETA_WATCHLIST = [
    "TSLA", "AMD", "PLTR", "SOFI", "COIN",
    "MARA", "RIOT", "HOOD", "RIVN", "SMCI",
    "IONQ", "AI", "UPST", "DKNG", "CVNA",
    "RBLX", "AFRM", "MSTR", "SNAP", "LCID",
]

# Higher-liquidity mega-cap / large-cap names with cleaner fills.
LIQUID_LARGE_CAP_WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "META", "AMZN",
    "GOOGL", "NFLX", "AVGO", "AMD", "TSLA",
    "COST", "JPM", "XOM", "LLY", "UNH",
]

# Best default for a day-trading scanner: enough names to find trades every 1-2 days.
DAY_TRADING_50_WATCHLIST = [
    "NVDA", "TSLA", "AMD", "PLTR", "SOFI", "COIN", "MARA", "RIOT", "HOOD", "RIVN",
    "SMCI", "MSTR", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "NFLX", "AVGO", "INTC",
    "MU", "UBER", "ARM", "CRWD", "DDOG", "SNOW", "SHOP", "SQ", "AFRM", "UPST",
    "DKNG", "RBLX", "NIO", "XPEV", "LCID", "F", "GM", "BAC", "JPM", "XOM",
    "CVX", "OXY", "TSM", "QCOM", "ORCL", "CRM", "BA", "DIS", "PYPL", "SNAP",
]

# Slower but very liquid names. Lower alert count but cleaner fills.
MEGA_CAP_PLUS_WATCHLIST = [
    "NVDA", "AAPL", "MSFT", "META", "AMZN", "GOOGL", "NFLX", "AVGO", "AMD", "TSLA",
    "COST", "JPM", "XOM", "LLY", "UNH", "MA", "PG", "JNJ",
    "ORCL", "CRM", "ADBE", "QCOM", "TXN", "AMAT", "MU", "INTC",
]

# V13 curated preset built from the uploaded V12 long-period reports.
# Goal: keep the high-quality mega-cap behaviour while adding the few
# high-beta names that actually contributed under the strict trend-pullback core.
EDGE_CORE_40_WATCHLIST = [
    # Strong / frequent in Mega Cap Plus V12
    "ORCL", "AVGO", "UNH", "MU", "NVDA", "GOOGL", "CRM", "TSLA", "LLY", "AMD",
    "AMAT", "AAPL", "V", "TXN", "INTC",
    # Mega-cap liquidity / clean fills
    "MSFT", "META", "NFLX", "COST", "JPM", "XOM", "HD", "WMT", "PG", "JNJ",
    "MA", "QCOM", "ADBE",
    # V12 extra winners from broader/high-beta tests
    "PLTR", "HOOD", "TSM",
    # Additional liquid candidates kept for opportunity count but still filtered by V13 core
    "UBER", "ARM", "CRWD", "SHOP", "BA", "DIS", "PYPL", "COIN", "MSTR", "SMCI",
]

# V18 preset from the independent opportunity dataset. This is not just the old
# strategy's winners; it combines symbols that appeared across the early low-ATR,
# controlled-gap, 10am continuation, and OR-rejection opportunity families.
OPPORTUNITY_CORE_35_WATCHLIST = [
    "AAPL", "MSFT", "AMAT", "UNH", "JNJ", "HD", "JPM", "XOM", "V", "MA",
    "COST", "WMT", "PG", "LLY", "CRM", "TSM", "BA", "TXN", "ORCL", "ADBE",
    "NVDA", "TSLA", "ARM", "PYPL", "SHOP", "HOOD", "MSTR", "SMCI", "COIN",
    "PLTR", "AMD", "AVGO", "MU", "NFLX", "META",
]


# V25 symbol-personality playbook universe: symbols that had at least one
# positive train/validate/test symbol+event+side pocket with the developing
# volume-profile reaction filter in the raw 5-minute research.
V25_PLAYBOOK_WATCHLIST = [
    # Exact symbols present in data/v25_research/v25_candidates_all.csv.gz,
    # which is the historical candidate universe used by Best Report 153601.
    "AAPL", "ADBE", "AMAT", "AMD", "BA", "COIN", "COST", "CRM", "DIS",
    "HD", "INTC", "MA", "MU", "PG", "PYPL", "QCOM", "SMCI", "TSLA",
    "V", "WMT", "XOM",
]

def _filter_excluded(symbols: list[str]) -> list[str]:
    return [s for s in symbols if str(s).upper() not in EXCLUDED_PRESET_SYMBOLS]

STARTER_WATCHLIST = _filter_excluded(STARTER_WATCHLIST)
DEFAULT_WATCHLIST = _filter_excluded(DEFAULT_WATCHLIST)
HIGH_BETA_WATCHLIST = _filter_excluded(HIGH_BETA_WATCHLIST)
LIQUID_LARGE_CAP_WATCHLIST = _filter_excluded(LIQUID_LARGE_CAP_WATCHLIST)
DAY_TRADING_50_WATCHLIST = _filter_excluded(DAY_TRADING_50_WATCHLIST)
MEGA_CAP_PLUS_WATCHLIST = _filter_excluded(MEGA_CAP_PLUS_WATCHLIST)
EDGE_CORE_40_WATCHLIST = _filter_excluded(EDGE_CORE_40_WATCHLIST)
OPPORTUNITY_CORE_35_WATCHLIST = _filter_excluded(OPPORTUNITY_CORE_35_WATCHLIST)
V25_PLAYBOOK_WATCHLIST = _filter_excluded(V25_PLAYBOOK_WATCHLIST)

WATCHLISTS = {
    "starter": STARTER_WATCHLIST,
    "balanced": DEFAULT_WATCHLIST,
    "day_trading_50": DAY_TRADING_50_WATCHLIST,
    "high_beta": HIGH_BETA_WATCHLIST,
    "large_cap": LIQUID_LARGE_CAP_WATCHLIST,
    "mega_cap_plus": MEGA_CAP_PLUS_WATCHLIST,
    "edge_core_40": EDGE_CORE_40_WATCHLIST,
    "opportunity_core_35": OPPORTUNITY_CORE_35_WATCHLIST,
    "v25_playbook": V25_PLAYBOOK_WATCHLIST,
}


def parse_symbols(raw: str | None, preset: str = "starter") -> list[str]:
    """Parse dashboard symbols.

    Important behavior:
    - If the custom-symbols box is populated, it is a true override and uses exactly
      the user's symbols (deduped, QQQ removed because QQQ is fetched internally).
    - EXCLUDED_PRESET_SYMBOLS only removes symbols from presets. It does not block
      a manually typed custom symbol, because the user may intentionally test it.
    """
    custom_mode = bool(raw and raw.strip())
    if custom_mode:
        symbols = [s.strip().upper() for s in raw.replace("\n", ",").replace(";", ",").split(",") if s.strip()]
    else:
        symbols = _filter_excluded(list(WATCHLISTS.get(preset, STARTER_WATCHLIST)))
    # Preserve order, remove duplicates and QQQ from tradable set.
    seen = set()
    cleaned = []
    for sym in symbols:
        sym = str(sym).strip().upper()
        if not sym or sym == "QQQ":
            continue
        if sym not in seen:
            seen.add(sym)
            cleaned.append(sym)
    return cleaned

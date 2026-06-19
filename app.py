from __future__ import annotations

import os
from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update

from src.backtest import run_backtest
from src.config import AlpacaSettings, StrategyParams
from src.live_dashboard import load_live_paper_snapshot
from src.symbols import WATCHLISTS, parse_symbols

app = Dash(__name__, suppress_callback_exceptions=True, title="Alpaca Symbol Playbook V33")
server = app.server

DEFAULT_END = date.today()
DEFAULT_START = DEFAULT_END - timedelta(days=90)


def metric_card(label: str, value: str, help_text: str = "") -> html.Div:
    return html.Div(
        className="metric-card",
        children=[html.Div(label, className="metric-label"), html.Div(value, className="metric-value"), html.Div(help_text, className="metric-help") if help_text else None],
    )


def empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=330,
        margin=dict(l=40, r=20, t=55, b=40),
        annotations=[dict(text="Run a backtest to populate this chart", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")],
    )
    return fig


def fmt_money(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "$0.00"
    return f"${x:,.2f}"


def fmt_pct(x: float | int | None) -> str:
    if x is None or pd.isna(x):
        return "0.00%"
    return f"{x:,.2f}%"


def table_card(title: str, subtitle: str, table_id: str, page_size: int = 10, filterable: bool = False) -> html.Div:
    return html.Div(
        className="card",
        children=[
            html.Div(className="section-head", children=[html.H3(title), html.Span(subtitle)]),
            dash_table.DataTable(
                id=table_id,
                page_size=page_size,
                sort_action="native",
                filter_action="native" if filterable else "none",
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "left", "padding": "10px", "fontFamily": "Inter, Arial", "fontSize": "12px"},
                style_header={"fontWeight": "700", "backgroundColor": "#f8fafc"},
            ),
        ],
    )


settings = AlpacaSettings()
status_text = "Configured" if settings.is_configured else "Missing API keys"
status_class = "status-pill ok" if settings.is_configured else "status-pill warn"

app.layout = html.Div(
    className="page",
    children=[
        html.Div(
            className="hero",
            children=[
                html.Div(children=[html.Div("ALPACA MOMENTUM LAB", className="eyebrow"), html.H1("Day-Trading Symbol Playbook V33"), html.P("Render-ready paper trading dashboard: backtests stay separate from risk sizing, while the live panel reads positions, signal plans, orders, and closed-position events from the shared paper-trading worker database.")]),
                html.Div(className=status_class, children=status_text),
            ],
        ),
        html.Div(
            className="layout",
            children=[
                html.Div(
                    className="sidebar card",
                    children=[
                        html.H3("Backtest Controls"),
                        html.Div(title="Uses the Symbol/Event Playbook engine for the current best strategy work.", children=[
                            html.Label("Strategy mode"),
                            dcc.Dropdown(
                                id="strategy-profile",
                                options=[
                                    {"label": "Symbol/Event Playbook - Best Report 153601 strategy + separate sizing", "value": "symbol_playbook_v25"},
                                ],
                                value="symbol_playbook_v25",
                                clearable=False,
                            ),
                        ]),
                        html.Div(title="Applies only the winning strategy filters. It does not change account value, fixed risk, compounding, or risk sizing settings.", children=[
                            html.Label("Strategy preset"),
                            dcc.Dropdown(
                                id="settings-preset",
                                options=[
                                    {"label": "Best Report 153601 - Top 2, L+S, selective candles, QQQ skip 4.2, news skip", "value": "best_qqq_news"},
                                ],
                                value="best_qqq_news",
                                clearable=False,
                            ),
                        ]),
                        html.Div(title="Selects the stock universe to test. Best Report 153601 uses the V25 playbook universe.", children=[html.Label("Watchlist preset"), dcc.Dropdown(id="preset", options=[{"label": k.replace("_", " ").title(), "value": k} for k in WATCHLISTS.keys()], value="v25_playbook", clearable=False)]),
                        html.Div(title="Optional: enter a comma-separated custom symbol list. Leave blank to use the watchlist preset above.", children=[html.Label("Custom symbols, comma separated"), dcc.Textarea(id="custom-symbols", placeholder="Example: NVDA, TSLA, AMD, PLTR, SOFI", value="", className="textarea")]),
                        html.Div(className="two-col", children=[html.Div(title="Backtest start date.", children=[html.Label("Start"), dcc.DatePickerSingle(id="start-date", date=DEFAULT_START.isoformat())]), html.Div(title="Backtest end date.", children=[html.Label("End"), dcc.DatePickerSingle(id="end-date", date=DEFAULT_END.isoformat())])]),
                        html.Div(title="IEX is the free Alpaca feed. SIP is the paid/unlimited feed if your account has access.", children=[html.Label("Alpaca feed"), dcc.Dropdown(id="feed", options=[{"label": "IEX - free plan", "value": "iex"}, {"label": "SIP - paid/unlimited", "value": "sip"}], value=os.getenv("ALPACA_FEED", "iex"), clearable=False)]),
                        html.Div(className="two-col", children=[html.Div(title="Starting account size used for the equity curve and risk calculations.", children=[html.Label("Account value"), dcc.Input(id="account-value", value=10000, type="number", min=500, step=100)]), html.Div(title="Used only when Risk mode is Fixed dollar risk. This disables compounding and risks the same dollars on every trade.", children=[html.Label("Fixed risk $/trade"), dcc.RadioItems(id="risk-dollars-v12", options=[{"label": "$10", "value": 10}, {"label": "$25", "value": 25}, {"label": "$50", "value": 50}, {"label": "$100", "value": 100}, {"label": "$200", "value": 200}, {"label": "$500", "value": 500}], value=100, inline=True)])]),
                        html.Hr(),
                        html.H3("Risk sizing / compounding", className="section-title"),
                        html.Div(className="hint", id="risk-mode-help", children="Risk sizing is independent from the strategy preset. Change it to compare fixed risk vs compounding without changing the trade signals."),
                        html.Div(title="Fixed dollar risk = no compounding. Percent of equity = full compounding. Controlled compounding = compounding with cap and drawdown brakes.", children=[
                            html.Label("Risk / compounding mode"),
                            dcc.Dropdown(id="risk-mode", options=[
                                {"label": "Fixed dollar risk - no compounding", "value": "fixed_dollar_risk"},
                                {"label": "Percent of equity - full compounding", "value": "percent_equity"},
                                {"label": "Controlled compounding - small-account friendly", "value": "controlled_compounding"},
                            ], value="percent_equity", clearable=False),
                        ]),
                        html.Div(id="risk-percent-panel", className="three-col", children=[
                            html.Div(title="Used by Percent of equity and Controlled compounding. 1.0 means risk 1% of current equity per trade.", children=[html.Label("Base risk %"), dcc.Input(id="base-risk-pct", value=1.0, type="number", min=0.1, max=5.0, step=0.05)]),
                        ]),
                        html.Div(id="controlled-compounding-panel", children=[
                            html.Div(className="three-col", children=[
                                html.Div(title="Controlled compounding only. Minimum dollar risk per trade. Use a small value like $5 or $10 for small accounts.", children=[html.Label("Min risk $"), dcc.Input(id="min-risk-dollars", value=10, type="number", min=0, max=1000, step=5)]),
                                html.Div(title="Controlled compounding only. Maximum dollar risk per trade. This prevents profits from increasing position size too quickly.", children=[html.Label("Max risk $"), dcc.Input(id="max-risk-dollars", value=300, type="number", min=0, max=5000, step=25)]),
                            ]),
                            html.Div(className="three-col", children=[
                                html.Div(title="Controlled compounding only. When account drawdown reaches about 5%, use this reduced risk percentage.", children=[html.Label("DD 5% risk %"), dcc.Input(id="dd1-risk-pct", value=0.75, type="number", min=0.0, max=5.0, step=0.05)]),
                                html.Div(title="Controlled compounding only. When account drawdown reaches about 10%, use this smaller risk percentage.", children=[html.Label("DD 10% risk %"), dcc.Input(id="dd2-risk-pct", value=0.50, type="number", min=0.0, max=5.0, step=0.05)]),
                                html.Div(title="Controlled compounding only. If drawdown reaches this level, new trades are skipped until equity recovers.", children=[html.Label("Pause DD %"), dcc.Input(id="pause-dd-pct", value=15.0, type="number", min=1.0, max=50.0, step=0.5)]),
                            ]),
                        ]),
                        html.Div(className="hint", children="To reproduce the high-equity uploaded report, select Risk mode = Percent of equity - full compounding and Base risk = 1.0%. To disable compounding, select Fixed dollar risk. To use safer compounding for smaller accounts, select Controlled compounding.") ,
                        html.Div(className="two-col", children=[html.Div(title="Minimum playbook score. 0 disables the score filter. The winning report used 2.", children=[html.Label("Min score (V25 raw score, 0=off)"), dcc.Input(id="min-score", value=2, type="number", min=0, max=60, step=1)]), html.Div(title="Limits how many selected trades can be taken per day. Winning report used Top 2.", children=[html.Label("Top trades/day"), dcc.RadioItems(id="max-trades", options=[{"label": "Top 1", "value": 1}, {"label": "Top 2", "value": 2}, {"label": "Top 3", "value": 3}], value=2, inline=True)])]),
                        html.Div(className="two-col", children=[html.Div(title="Estimated execution slippage in basis points. 3 bps means about 0.03% per execution adjustment.", children=[html.Label("Slippage bps"), dcc.Input(id="slippage-bps", value=3, type="number", min=0, max=50, step=1)]), html.Div(title="Turns on the historical news/catalyst proxy used by the catalyst filter. Winning report used Yes.", children=[html.Label("Use news / catalyst proxy?"), dcc.Dropdown(id="use-news", options=[{"label": "No - ignore news proxy", "value": "false"}, {"label": "Yes - activate news/catalyst flags", "value": "true"}], value="true", clearable=False)])]),
                        html.Div(title="Optional macro calendar filter. Winning report left this Off.", children=[
                            html.Label("Macro/news risk filters - optional"),
                            dcc.Dropdown(
                                id="macro-filter",
                                options=[
                                    {"label": "Off - do not filter macro calendar days", "value": "off"},
                                    {"label": "Top 1 only on macro calendar days", "value": "top1"},
                                    {"label": "Skip macro calendar days", "value": "skip"},
                                ],
                                value="off",
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="two-col", children=[
                            html.Div(title="Controls what happens on high QQQ stress days. Winning report skipped QQQ stress days.", children=[html.Label("QQQ stress filter"), dcc.Dropdown(id="stress-filter", options=[{"label": "Off", "value": "off"}, {"label": "Top 1 only on QQQ stress days", "value": "top1"}, {"label": "Skip QQQ stress days", "value": "skip"}], value="skip", clearable=False)]),
                            html.Div(title="Controls what happens when the news/catalyst proxy flags a candidate. Winning report skipped catalyst candidates.", children=[html.Label("News/catalyst filter"), dcc.Dropdown(id="news-filter", options=[{"label": "Off", "value": "off"}, {"label": "Top 1 only on catalyst proxy days", "value": "top1"}, {"label": "Skip catalyst proxy candidates", "value": "skip"}], value="skip", clearable=False)]),
                        ]),
                        html.Div(className="two-col", children=[
                            html.Div(title="QQQ stress threshold used by the stress filter. Winning report used 4.2.", children=[html.Label("QQQ stress threshold %"), dcc.Input(id="qqq-stress-threshold", value=4.2, type="number", min=0.25, max=5.0, step=0.05)]),
                            html.Div(title="Optional rolling symbol/side pause after losses. Winning report left this Off.", children=[html.Label("Symbol/side kill switch"), dcc.Dropdown(id="kill-switch", options=[{"label": "Off", "value": "off"}, {"label": "Moderate: pause after -3R / 20 trades", "value": "moderate"}, {"label": "Strict: pause after -2R / 10 trades", "value": "strict"}], value="off", clearable=False)]),
                        ]),
                        html.Div(title="Entry/exit candlestick filter. Winning report used Selective; Broad score gave the same result in your test.", children=[
                            html.Label("Candlestick patterns"),
                            dcc.Dropdown(
                                id="candle-mode",
                                options=[
                                    {"label": "Exit-only - candle reversal exits only", "value": "exit_only"},
                                    {"label": "Off - ignore candle patterns", "value": "off"},
                                    {"label": "Selective - rejection filter + reversal exits", "value": "selective"},
                                    {"label": "Broad confirm + exits - comparison", "value": "confirm"},
                                    {"label": "Broad score + exits - comparison", "value": "score"},
                                ],
                                value="selective",
                                clearable=False,
                            ),
                        ]),
                        html.Div(className="two-col", children=[
                            html.Div(children=[html.Label("Mean reversion"), dcc.Dropdown(id="enable-mr", options=[{"label": "On", "value": "true"}, {"label": "Off", "value": "false"}], value="false", clearable=False)]),
                            html.Div(children=[html.Label("OR/retest setup"), dcc.Dropdown(id="enable-or", options=[{"label": "Off", "value": "false"}, {"label": "Selective only", "value": "true"}], value="false", clearable=False)]),
                        ]),
                        html.Label("Direction mode"),
                        dcc.Dropdown(id="direction-mode", options=[{"label": "Long only", "value": "long_only"}, {"label": "Short only", "value": "short_only"}, {"label": "Long + short", "value": "long_short"}], value="long_short", clearable=False),
                        html.Button("Run Backtest", id="run-btn", className="primary-btn", n_clicks=0),
                        html.Div(id="run-status", className="run-status"),
                        html.Hr(),
                        html.H4("Version 33 Notes"),
                        html.Ul(
                            className="rule-list",
                            children=[
                                html.Li("V33 uses the V25/V27 symbol-event playbook, keeps one winning strategy preset, and separates strategy settings from risk sizing/compounding."),
                                html.Li("Entry uses the next 5-minute bar open, with conservative first-touch sequencing: stop before target if both touch in the same bar."),
                                html.Li("V27 uses the full approved candidate universe, then applies symbol, date, direction, min-score, candlestick, macro/news, QQQ stress, and kill-switch filters before selecting Top 1/2/3 per day."),
                                html.Li("V25 uses 0.75R target, 1.0R stop, and 12-bar maximum hold to match the raw-data test assumptions."),
                                html.Li("The single baseline preset is Best Report 153601; live trading uses the V25 playbook watchlist and the same score/candle/news/QQQ-stress filter stack."),
                                html.Li("Candlestick options now affect V25/V27 entries: Off and Exit-only reproduce baseline; Selective/Confirm/Score apply entry candle filters."),
                                html.Li("Short-side trading is enabled by default because the expanded raw-data tests showed short-only was more stable than long-only."),
                                html.Li("Live paper orders use market bracket orders for the 0.75R target and 1.0R stop, while the worker separately enforces the 12-bar max-hold exit."),
                                html.Li("Risk can now be fixed-dollar, full percent-equity compounding, or controlled compounding with drawdown brakes."),
                                html.Li("Risk sizing is independent: Fixed dollar risk disables compounding; Controlled compounding uses 1% risk, $10 minimum, $300 cap, 0.75% risk after 5% drawdown, 0.50% after 10%, and pause after 15% drawdown."),
                                html.Li("Reports are saved to reports/latest_backtest_report.zip after every run; live paper signal plans, opened positions, closed positions, orders, and worker events are read from the shared Render database."),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="main",
                    children=[
                        dcc.Interval(id="live-refresh-interval", interval=30_000, n_intervals=0),
                        html.Div(className="card", children=[
                            html.Div(className="section-head", children=[html.H3("Live Alpaca Paper Monitor"), html.Span("Auto-refreshes every 30 seconds from the shared worker database")]),
                            html.Div(id="live-status", className="run-status"),
                            html.Div(id="live-metrics-row", className="metrics-grid"),
                        ]),
                        html.Div(className="grid-2", children=[
                            table_card("Open Live/Paper Positions", "Positions currently open in Alpaca paper and synced by the worker", "live-positions-table", page_size=10),
                            table_card("Recent Signal Plans", "Algorithm decisions, sizing, stop, target, and paper-order submission status", "live-plans-table", page_size=12, filterable=True),
                        ]),
                        html.Div(className="grid-2", children=[
                            table_card("Recent Alpaca Paper Orders", "Parent bracket orders and exit legs synced from Alpaca", "live-orders-table", page_size=12, filterable=True),
                            table_card("Closed / Recently Closed Positions", "Positions that the worker saw open and later missing from Alpaca /positions", "live-closed-positions-table", page_size=12, filterable=True),
                        ]),
                        table_card("Worker Events", "Scans, blocked entries, submitted paper orders, sync errors, and max-hold exits", "live-events-table", page_size=12, filterable=True),
                        html.Div(className="section-head", children=[html.H3("Historical Backtest Lab"), html.Span("Same baseline strategy preset; risk sizing remains separate")]),
                        html.Div(id="metrics-row", className="metrics-grid"),
                        html.Div(className="grid-2", children=[html.Div(className="card", children=[dcc.Graph(id="equity-fig", figure=empty_figure("Equity Curve"))]), html.Div(className="card", children=[dcc.Graph(id="drawdown-fig", figure=empty_figure("Drawdown"))])]),
                        html.Div(className="grid-2", children=[html.Div(className="card", children=[dcc.Graph(id="symbol-fig", figure=empty_figure("P&L by Symbol"))]), html.Div(className="card", children=[dcc.Graph(id="r-fig", figure=empty_figure("R-Multiple Distribution"))])]),
                        html.Div(className="grid-2", children=[html.Div(className="card", children=[dcc.Graph(id="setup-fig", figure=empty_figure("P&L by Setup Type"))]), html.Div(className="card", children=[dcc.Graph(id="daily-fig", figure=empty_figure("Trades by Day"))])]),
                        table_card("Symbol Summary", "Which symbols fit this strategy best?", "symbol-table", page_size=12),
                        html.Div(className="grid-2", children=[table_card("Setup Summary", "Which trigger type works?", "setup-table", page_size=10), table_card("Daily Summary", "Frequency and daily consistency", "daily-table", page_size=10)]),
                        html.Div(className="grid-2", children=[table_card("Exit Reason Summary", "Are exits helping or hurting?", "exit-table", page_size=10), table_card("MFE / MAE Diagnosis", "Entry vs exit failure clues", "mfe-table", page_size=10)]),
                        html.Div(className="grid-2", children=[table_card("Candlestick Pattern Summary", "Do candle patterns improve entries/exits?", "candle-table", page_size=10), table_card("Score Band Summary", "Is score predictive?", "score-table", page_size=10)]),
                        table_card("Time Bucket Summary", "Best time of day", "time-table", page_size=10),
                        table_card("Trades", "Selected trades after portfolio/risk/adaptive rules", "trades-table", page_size=15, filterable=True),
                    ],
                ),
            ],
        ),
    ],
)



@app.callback(
    Output("live-metrics-row", "children"),
    Output("live-positions-table", "data"), Output("live-positions-table", "columns"),
    Output("live-plans-table", "data"), Output("live-plans-table", "columns"),
    Output("live-orders-table", "data"), Output("live-orders-table", "columns"),
    Output("live-closed-positions-table", "data"), Output("live-closed-positions-table", "columns"),
    Output("live-events-table", "data"), Output("live-events-table", "columns"),
    Output("live-status", "children"),
    Input("live-refresh-interval", "n_intervals"),
)
def refresh_live_paper_monitor(n_intervals):
    try:
        snapshot = load_live_paper_snapshot(days=7)
        metrics = [metric_card(m.get("label", "Metric"), m.get("value", "--"), m.get("help", "")) for m in snapshot.get("metrics", [])]
        if not metrics:
            metrics = [
                metric_card("Paper Monitor", "No data", "Start the Render worker to populate the shared database"),
                metric_card("Open Positions", "--"),
                metric_card("Signal Plans", "--"),
                metric_card("Worker", "--"),
            ]
        pos_data, pos_cols = table_payload(snapshot.get("open_positions", pd.DataFrame()))
        plan_data, plan_cols = table_payload(snapshot.get("plans", pd.DataFrame()))
        order_data, order_cols = table_payload(snapshot.get("orders", pd.DataFrame()))
        closed_data, closed_cols = table_payload(snapshot.get("closed_positions", pd.DataFrame()))
        event_data, event_cols = table_payload(snapshot.get("events", pd.DataFrame()))
        return metrics, pos_data, pos_cols, plan_data, plan_cols, order_data, order_cols, closed_data, closed_cols, event_data, event_cols, snapshot.get("status", "Live paper monitor refreshed.")
    except Exception as exc:
        metrics = [metric_card("Live Monitor", "Error", str(exc)[:80]), metric_card("Open Positions", "--"), metric_card("Signal Plans", "--"), metric_card("Worker", "--")]
        blank = ([], [])
        return metrics, *blank, *blank, *blank, *blank, *blank, f"Live paper monitor error: {exc}"


@app.callback(
    Output("risk-percent-panel", "style"),
    Output("controlled-compounding-panel", "style"),
    Output("risk-mode-help", "children"),
    Input("risk-mode", "value"),
)
def update_risk_mode_visibility(risk_mode):
    mode = str(risk_mode or "fixed_dollar_risk")
    hidden = {"display": "none"}
    shown = {}
    if mode == "fixed_dollar_risk":
        return hidden, hidden, "Fixed dollar risk is selected: compounding is OFF. The Fixed risk $/trade radio buttons set the same dollar risk for every trade."
    if mode == "percent_equity":
        return shown, hidden, "Percent of equity is selected: full compounding is ON. Base risk % is applied to current equity on every trade, with no cap or drawdown brake. Use 1.0% to reproduce the high-equity report."
    return shown, shown, "Controlled compounding is selected: compounding is ON, but the minimum/maximum risk and drawdown brakes below control how much profits can increase risk."


@app.callback(
    Output("max-trades", "value"),
    Output("direction-mode", "value"),
    Output("min-score", "value"),
    Output("candle-mode", "value"),
    Output("use-news", "value"),
    Output("news-filter", "value"),
    Output("stress-filter", "value"),
    Output("qqq-stress-threshold", "value"),
    Output("macro-filter", "value"),
    Output("kill-switch", "value"),
    Input("settings-preset", "value"),
)
def apply_settings_preset(preset_name):
    if preset_name == "manual":
        return tuple([no_update] * 10)
    # Exact strategy filters from the user's strongest uploaded report 153601.
    # This deliberately does NOT touch risk sizing or compounding controls.
    return (2, "long_short", 2, "selective", "true", "skip", "skip", 4.2, "off", "off")


@app.callback(
    Output("metrics-row", "children"),
    Output("equity-fig", "figure"),
    Output("drawdown-fig", "figure"),
    Output("symbol-fig", "figure"),
    Output("r-fig", "figure"),
    Output("setup-fig", "figure"),
    Output("daily-fig", "figure"),
    Output("symbol-table", "data"), Output("symbol-table", "columns"),
    Output("setup-table", "data"), Output("setup-table", "columns"),
    Output("daily-table", "data"), Output("daily-table", "columns"),
    Output("exit-table", "data"), Output("exit-table", "columns"),
    Output("mfe-table", "data"), Output("mfe-table", "columns"),
    Output("candle-table", "data"), Output("candle-table", "columns"),
    Output("score-table", "data"), Output("score-table", "columns"),
    Output("time-table", "data"), Output("time-table", "columns"),
    Output("trades-table", "data"), Output("trades-table", "columns"),
    Output("run-status", "children"),
    Input("run-btn", "n_clicks"),
    State("strategy-profile", "value"), State("preset", "value"), State("custom-symbols", "value"),
    State("start-date", "date"), State("end-date", "date"), State("feed", "value"),
    State("account-value", "value"), State("risk-dollars-v12", "value"),
    State("risk-mode", "value"), State("base-risk-pct", "value"), State("min-risk-dollars", "value"), State("max-risk-dollars", "value"),
    State("dd1-risk-pct", "value"), State("dd2-risk-pct", "value"), State("pause-dd-pct", "value"),
    State("min-score", "value"), State("max-trades", "value"), State("slippage-bps", "value"), State("use-news", "value"), State("candle-mode", "value"),
    State("macro-filter", "value"), State("stress-filter", "value"), State("news-filter", "value"), State("qqq-stress-threshold", "value"), State("kill-switch", "value"),
    State("enable-mr", "value"), State("enable-or", "value"), State("direction-mode", "value"),
    prevent_initial_call=True,
)
def run_backtest_callback(n_clicks, strategy_profile, preset, custom_symbols, start_date, end_date, feed, account_value, risk_dollars, risk_mode, base_risk_pct, min_risk_dollars, max_risk_dollars, dd1_risk_pct, dd2_risk_pct, pause_dd_pct, min_score, max_trades, slippage_bps, use_news, candle_mode, macro_filter, stress_filter, news_filter, qqq_stress_threshold, kill_switch, enable_mr, enable_or, direction_mode):
    blank_tables = ([], []) * 9
    if not n_clicks:
        metrics = [metric_card("Win Rate", "--", "target: 62-75% after validation"), metric_card("Total P&L", "--"), metric_card("Profit Factor", "--"), metric_card("Trades", "--"), metric_card("Avg MFE", "--"), metric_card("Expectancy", "--")]
        empty = empty_figure
        return (metrics, empty("Equity Curve"), empty("Drawdown"), empty("P&L by Symbol"), empty("R-Multiple Distribution"), empty("P&L by Setup Type"), empty("Trades by Day"), *blank_tables, "Ready. Add Alpaca keys to .env, choose symbols, then run a backtest.")

    try:
        symbols = parse_symbols(custom_symbols, preset=preset)
        risk_value = float(risk_dollars)
        account_value_f = float(account_value or 10000)
        base_risk_pct_f = float(base_risk_pct or 1.0)
        risk_mode_s = str(risk_mode or "controlled_compounding")
        if risk_value <= 0:
            raise ValueError("Fixed Risk $/trade must be greater than zero")
        if base_risk_pct_f <= 0:
            raise ValueError("Base risk percent must be greater than zero")
        params = StrategyParams(
            strategy_profile=strategy_profile or "symbol_playbook_v25",
            direction_mode=str(direction_mode or "long_only"),
            initial_account_value=account_value_f,
            risk_per_trade_dollars=risk_value,
            requested_risk_percent=base_risk_pct_f,
            risk_per_trade_pct=base_risk_pct_f / 100.0,
            position_sizing_mode=risk_mode_s,
            compounding_base_risk_pct=base_risk_pct_f,
            compounding_min_risk_dollars=float(min_risk_dollars or 0.0),
            compounding_max_risk_dollars=float(max_risk_dollars or 0.0),
            compounding_dd1_risk_pct=float(dd1_risk_pct or 0.75),
            compounding_dd2_risk_pct=float(dd2_risk_pct or 0.50),
            compounding_pause_dd_pct=float(pause_dd_pct or 15.0),
            max_position_notional_pct=9999.0,
            min_candidate_score=float(min_score or 70),
            max_trades_per_day=int(max_trades or 10),
            slippage_bps=float(slippage_bps or 0),
            candle_pattern_mode=str(candle_mode or "exit_only"),
            enable_mean_reversion=(str(enable_mr).lower() == "true"),
            enable_or_retest=(str(enable_or).lower() == "true"),
            v27_macro_filter_mode=str(macro_filter or "off"),
            v27_market_stress_mode=str(stress_filter or "off"),
            v27_news_filter_mode=str(news_filter or "off"),
            v27_symbol_kill_switch_mode=str(kill_switch or "off"),
            v27_qqq_stress_abs_change_pct=float(qqq_stress_threshold or 1.25),
        )
        if str(strategy_profile or "") == "symbol_playbook_v25":
            params.max_open_positions = int(max_trades or 2)
            params.daily_loss_limit_pct = 100.0
            params.max_consecutive_losses = 99
            params.v25_target_r = 0.75
            params.v25_max_hold_bars = 12
        result = run_backtest(symbols=symbols, start_date=start_date, end_date=end_date, params=params, feed=feed, use_cache=True, use_news=(str(use_news).lower() == "true"), export_report=True)
        metrics = result["metrics"]
        selected = result["selected_trades"]
        candidates = result["candidates"]
        report_paths = result.get("report_paths", {})

        metrics_cards = [
            metric_card("Win Rate", fmt_pct(metrics["win_rate"]), "target range: 62-75%"),
            metric_card("Total P&L", fmt_money(metrics["total_pnl"]), fmt_pct(metrics["total_return_pct"])),
            metric_card("Profit Factor", f"{metrics['profit_factor']:.2f}", ">1.50 preferred"),
            metric_card("Trades", f"{metrics['total_trades']}", f"{len(candidates)} raw candidates"),
            metric_card("Risk Used", fmt_money(metrics.get("risk_per_trade_dollars", metrics.get("avg_risk_budget", 0))), f"actual avg {fmt_money(metrics.get('avg_actual_dollars_at_risk', 0))}"),
            metric_card("Avg Notional", fmt_money(metrics.get("avg_notional", 0)), f"{fmt_pct(metrics.get('avg_notional_pct', 0))} of equity"),
            metric_card("Avg MFE", f"{metrics.get('avg_mfe_r', 0):.2f}R", "favorable excursion"),
            metric_card("Expectancy", f"{metrics['expectancy_r']:.2f}R", "avg R per trade"),
        ]

        equity_fig = make_equity_fig(result["equity_curve"])
        drawdown_fig = make_drawdown_fig(result["drawdown_curve"])
        symbol_fig = make_symbol_fig(result["symbol_summary"])
        r_fig = make_r_distribution_fig(selected)
        setup_fig = make_setup_fig(result.get("setup_summary", pd.DataFrame()))
        daily_fig = make_daily_fig(result.get("daily_summary", pd.DataFrame()))

        sym_data, sym_cols = table_payload(result.get("symbol_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor", "avg_duration"])
        setup_data, setup_cols = table_payload(result.get("setup_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor", "avg_mfe_r", "avg_mae_r"])
        daily_data, daily_cols = table_payload(result.get("daily_summary", pd.DataFrame()), money_cols=["pnl"], pct_cols=["win_rate"], round_cols=["avg_score"])
        exit_data, exit_cols = table_payload(result.get("exit_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "profit_factor", "avg_mfe_r", "avg_mae_r"])
        mfe_data, mfe_cols = table_payload(result.get("mfe_mae_summary", pd.DataFrame()), round_cols=["value"])
        candle_data, candle_cols = table_payload(result.get("candle_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor", "avg_mfe_r", "avg_mae_r"])
        score_data, score_cols = table_payload(result.get("score_band_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor"])
        time_data, time_cols = table_payload(result.get("time_bucket_summary", pd.DataFrame()), money_cols=["total_pnl"], pct_cols=["win_rate"], round_cols=["avg_r", "avg_score", "profit_factor"])

        trade_cols = ["symbol", "entry_time_et", "exit_time_et", "trigger_type", "quality", "candidate_score", "entry_candle_pattern", "entry_price", "stop_price", "shares", "notional", "risk_budget", "actual_dollars_at_risk", "pnl_dollars", "pnl_dollars_from_shares", "risk_application_delta", "r_multiple", "mfe_r", "mae_r", "position_sizing_mode", "low_followthrough_mode", "target1_hit", "target2_hit", "exit_reason"]
        if not selected.empty:
            for col in trade_cols:
                if col not in selected.columns:
                    selected[col] = ""
            trade_data = selected[trade_cols].copy()
        else:
            trade_data = pd.DataFrame(columns=trade_cols)
        trade_data, trade_columns = table_payload(trade_data, money_cols=["entry_price", "stop_price", "pnl_dollars"], round_cols=["candidate_score", "candle_pattern_score", "r_multiple", "mfe_r", "mae_r"])
        status = f"Backtest complete: {len(symbols)} symbols, {start_date} to {end_date}, feed={feed}, execution={result.get('execution_timeframe')}, UI risk=${float(risk_dollars):.2f}, engine risk=${metrics.get('risk_per_trade_dollars', metrics.get('avg_risk_budget', 0)):.2f}, sizing={risk_mode_s}, fixed risk input=${risk_dollars}, base risk={base_risk_pct_f:.2f}%, max risk=${float(max_risk_dollars or 0):.2f}, min score={min_score}, candle mode={candle_mode}, macro={macro_filter}, qqq stress={stress_filter}, news filter={news_filter}, kill switch={kill_switch}, news/catalyst proxy={use_news}, mean reversion={enable_mr}, OR/retest={enable_or}, direction={direction_mode}. Report: {report_paths.get('latest_zip', report_paths.get('zip_path', 'not saved'))}"
        return metrics_cards, equity_fig, drawdown_fig, symbol_fig, r_fig, setup_fig, daily_fig, sym_data, sym_cols, setup_data, setup_cols, daily_data, daily_cols, exit_data, exit_cols, mfe_data, mfe_cols, candle_data, candle_cols, score_data, score_cols, time_data, time_cols, trade_data, trade_columns, status
    except Exception as exc:
        metrics_cards = [metric_card("Error", "Backtest failed"), metric_card("Fix", "Check keys/date/feed"), metric_card("Details", str(exc)[:60])]
        empty = empty_figure
        return (metrics_cards, empty("Equity Curve"), empty("Drawdown"), empty("P&L by Symbol"), empty("R-Multiple Distribution"), empty("P&L by Setup Type"), empty("Trades by Day"), *blank_tables, f"Error: {exc}")


def make_equity_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Equity Curve")
    fig = px.line(df, x="exit_time", y="equity", title="Equity Curve")
    fig.update_traces(line_width=3)
    return polish_fig(fig, y_title="Account Equity ($)")


def make_drawdown_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Drawdown")
    fig = px.area(df, x="exit_time", y="drawdown_pct", title="Drawdown %")
    return polish_fig(fig, y_title="Drawdown %")


def make_symbol_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("P&L by Symbol")
    fig = px.bar(df.sort_values("total_pnl"), x="symbol", y="total_pnl", title="P&L by Symbol", hover_data=["trades", "win_rate", "avg_r"])
    return polish_fig(fig, y_title="P&L ($)")


def make_r_distribution_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("R-Multiple Distribution")
    fig = px.histogram(df, x="r_multiple", nbins=30, title="R-Multiple Distribution")
    fig.add_vline(x=0, line_dash="dash")
    return polish_fig(fig, y_title="Trades", x_title="R Multiple")


def make_setup_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("P&L by Setup Type")
    fig = px.bar(df.sort_values("total_pnl"), x="trigger_type", y="total_pnl", title="P&L by Setup Type", hover_data=["trades", "win_rate", "avg_r"])
    return polish_fig(fig, y_title="P&L ($)")


def make_daily_fig(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure("Trades by Day")
    fig = px.bar(df, x="session_date", y="trades", title="Trades by Day", hover_data=["pnl", "win_rate", "avg_score"])
    return polish_fig(fig, y_title="Trades")


def polish_fig(fig: go.Figure, y_title: str = "", x_title: str = "") -> go.Figure:
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=40, r=20, t=55, b=40), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Arial", color="#0f172a"), title=dict(font=dict(size=18)))
    fig.update_xaxes(title=x_title, gridcolor="#e5e7eb")
    fig.update_yaxes(title=y_title, gridcolor="#e5e7eb")
    return fig


def table_payload(df: pd.DataFrame, money_cols=None, pct_cols=None, round_cols=None):
    money_cols = money_cols or []
    pct_cols = pct_cols or []
    round_cols = round_cols or []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
    for col in money_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: fmt_money(float(x)) if pd.notna(x) else "")
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: fmt_pct(float(x)) if pd.notna(x) else "")
    for col in round_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")
    columns = [{"name": col.replace("_", " ").title(), "id": col} for col in out.columns]
    return out.to_dict("records"), columns


if __name__ == "__main__":
    debug = os.getenv("DASH_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", "8050"))
    app.run_server(host="0.0.0.0", port=port, debug=debug)

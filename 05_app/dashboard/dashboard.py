"""
dashboard.py - Streamlit dashboard for Finnhub stock market analytics across all timeframes.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)


@st.cache_resource
def _get_engine():
    host = os.environ["DB_HOST"]
    port = os.environ.get("DB_PORT", "5432")
    dbname = os.environ["DB_NAME"]
    user = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


PRICE_COLUMNS = [
    "symbol",
    "commodity_name",
    "commodity_category",
    "timeframe",
    "time_index",
    "open_price",
    "high_price",
    "low_price",
    "close_price",
    "latest_price",
    "price_change",
    "price_change_pct",
    "price_trend",
    "intraday_range",
    "intraday_range_pct",
    "volatility_level",
    "category_avg_close",
    "category_count",
]


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def _table_exists(conn, table_name: str) -> bool:
    return conn.execute(
        text("SELECT to_regclass(:table_name) IS NOT NULL"),
        {"table_name": table_name},
    ).scalar_one()


@st.cache_data(ttl=5)
def _load_prices() -> pd.DataFrame:
    engine = _get_engine()
    with engine.connect() as conn:
        if not (
            _table_exists(conn, "gold.dim_commodity")
            and _table_exists(conn, "gold.fact_commodity_prices")
        ):
            return _empty_prices()

        return pd.read_sql(text("""
            SELECT d.symbol,
                   d.commodity_name,
                   d.commodity_category,
                   f.timeframe,
                   f.time_index,
                   f.open_price,
                   f.high_price,
                   f.low_price,
                   f.close_price,
                   f.latest_price,
                   f.price_change,
                   f.price_change_pct,
                   f.price_trend,
                   f.intraday_range,
                   f.intraday_range_pct,
                   f.volatility_level,
                   f.category_avg_close,
                   f.category_count
            FROM   gold.fact_commodity_prices f
            JOIN   gold.dim_commodity         d ON d.commodity_id = f.commodity_id
            ORDER  BY d.symbol, f.timeframe, f.time_index
        """), conn)


def _latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return (
        df.sort_values(["symbol", "time_index"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("close_price", ascending=False)
        .reset_index(drop=True)
    )


def _colour_change(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: #6b9e7a; font-weight: 600"
    if val < 0:
        return "color: #b85c5c; font-weight: 600"
    return "color: #71717a; font-weight: 600"


def _tick_colour(val):
    if str(val).lower() == "up":
        return "color: #6b9e7a; font-weight: 600"
    if str(val).lower() == "down":
        return "color: #b85c5c; font-weight: 600"
    return "color: #71717a; font-weight: 600"


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #111114;
            --panel: #19191f;
            --line: #27272e;
            --text: #d4d4d8;
            --muted: #71717a;
            --accent: #a1a1aa;
            --up: #6b9e7a;
            --down: #b85c5c;
        }
        html, body, [data-testid="stAppViewContainer"] { background: var(--bg); }
        [data-testid="stAppViewContainer"] {
            background: var(--bg);
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 2.8rem;
        }
        [data-testid="stSidebar"] {
            background: #141417;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
            color: var(--text);
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: var(--text) !important;
            background-color: var(--panel) !important;
            border: 1px solid var(--line) !important;
        }
        [data-testid="stSidebar"] input::placeholder {
            color: #52525b !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: var(--text);
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            box-shadow: none;
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
            font-size: 0.82rem;
        }
        div[data-testid="stMetricValue"] {
            color: var(--text);
            font-weight: 600;
        }
        .commodity-header {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 24px 28px;
            margin-bottom: 18px;
            box-shadow: none;
        }
        .commodity-header h1 {
            color: var(--text);
            font-size: 1.6rem;
            margin: 0 0 4px 0;
            font-weight: 600;
            letter-spacing: -0.01em;
        }
        .commodity-header p {
            margin: 0;
            color: var(--muted);
        }
        .side-title {
            color: var(--text);
            font-size: 1.1rem;
            font-weight: 600;
            margin: .3rem 0 .35rem 0;
        }
        .side-subtitle {
            color: var(--muted);
            font-size: .76rem;
            margin-bottom: 1.05rem;
        }
        .side-active {
            background: rgba(161, 161, 170, 0.08);
            border-left: 3px solid var(--accent);
            border-radius: 6px;
            color: var(--text);
            font-size: .95rem;
            font-weight: 600;
            margin: .5rem 0 1rem 0;
            padding: .78rem .9rem;
        }
        .side-section {
            border-top: 1px solid var(--line);
            color: var(--muted);
            font-size: .72rem;
            font-weight: 600;
            letter-spacing: .08em;
            margin-top: 1rem;
            padding-top: 1rem;
            text-transform: uppercase;
        }
        .stat-card {
            border: 1px solid var(--line);
            border-radius: 10px;
            min-height: 126px;
            padding: 20px 22px;
            box-shadow: none;
        }
        .stat-card.purple {
            background: var(--panel);
        }
        .stat-card.cyan {
            background: var(--panel);
        }
        .stat-card.dark {
            background: var(--panel);
        }
        .stat-label {
            color: var(--muted);
            font-size: .82rem;
            font-weight: 500;
            margin-bottom: 18px;
        }
        .stat-value {
            color: var(--text);
            font-size: 2rem;
            font-weight: 600;
            line-height: 1;
        }
        .stat-sub {
            color: var(--muted);
            font-size: .82rem;
            margin-left: .3rem;
        }
        .mini-card {
            align-items: center;
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            display: flex;
            justify-content: space-between;
            min-height: 58px;
            padding: 12px 16px;
        }
        .mini-card strong { color: var(--text); font-size: .86rem; }
        .mini-badge {
            border-radius: 999px;
            font-size: .74rem;
            font-weight: 600;
            padding: 4px 10px;
        }
        .mini-badge.up { background: rgba(107, 158, 122, 0.15); color: var(--up); }
        .mini-badge.down { background: rgba(184, 92, 92, 0.15); color: var(--down); }
        .block-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 600;
            margin: 18px 0 10px;
        }
        .help-line {
            color: var(--muted);
            font-size: .84rem;
            margin: -4px 0 12px 0;
        }
        .legend-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 4px 0 14px;
        }
        .legend-pill {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--text);
            font-size: .8rem;
            font-weight: 500;
            padding: 7px 11px;
        }
        .legend-pill.up { border-color: rgba(107, 158, 122, 0.4); color: var(--up); }
        .legend-pill.down { border-color: rgba(184, 92, 92, 0.4); color: var(--down); }
        .stPlotlyChart {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 10px;
            box-shadow: none;
            padding: 10px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid var(--line);
            border-radius: 10px;
            overflow: hidden;
        }
        div[data-testid="stSelectbox"],
        div[data-testid="stSlider"],
        div[data-testid="stRadio"],
        div[data-testid="stTextInput"],
        div[data-testid="stMultiSelect"] {
            color: var(--text);
        }
        /* Multiselect tag pills */
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
            background-color: var(--panel) !important;
            border: 1px solid var(--line) !important;
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] span[data-baseweb="tag"] span {
            color: var(--text) !important;
        }
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] span[data-baseweb="tag"] svg {
            fill: var(--muted) !important;
        }
        /* Multiselect dropdown background */
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div {
            background-color: var(--panel) !important;
            border-color: var(--line) !important;
        }
        /* Fix overlapping input box inside multiselect/selectbox */
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] input,
        [data-testid="stSidebar"] [data-testid="stSelectbox"] input {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin: 0 !important;
            box-shadow: none !important;
        }

        /* Pulse indicator styling */
        .live-indicator-container {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin: 18px 0 10px;
        }
        .live-dot {
            width: 7px;
            height: 7px;
            background-color: var(--accent);
            border-radius: 50%;
            display: inline-block;
            animation: livePulseDot 2.5s infinite ease-in-out;
        }
        .live-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 600;
            margin: 0;
        }
        @keyframes livePulseDot {
            0% {
                transform: scale(0.9);
                box-shadow: 0 0 0 0 rgba(161, 161, 170, 0.4);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 5px rgba(161, 161, 170, 0);
            }
            100% {
                transform: scale(0.9);
                box-shadow: 0 0 0 0 rgba(161, 161, 170, 0);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _inject_dynamic_animations()


def _inject_dynamic_animations() -> None:
    anim_id = int(time.time() * 1000)
    st.markdown(
        f"""
        <style>
        /* -- Subtle entry animations ------------------------------------------------ */
        @keyframes fadeUp_{anim_id} {{
            0% {{
                opacity: 0;
                transform: translateY(8px);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        @keyframes fadeIn_{anim_id} {{
            0% {{ opacity: 0.75; }}
            100% {{ opacity: 1; }}
        }}
        @keyframes slideUp_{anim_id} {{
            0% {{
                opacity: 0.8;
                transform: translateY(4px);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        /* -- Apply animations ------------------------------------------------------- */
        .commodity-header {{
            animation: fadeUp_{anim_id} 0.4s ease-out both;
        }}
        .stat-card {{
            animation: fadeUp_{anim_id} 0.42s ease-out both;
            animation-delay: 0.03s;
        }}
        .mini-card {{
            animation: fadeUp_{anim_id} 0.42s ease-out both;
        }}
        .mini-card:nth-of-type(1) {{ animation-delay: 0.06s; }}
        .mini-card:nth-of-type(2) {{ animation-delay: 0.09s; }}

        div[data-testid="stSelectbox"],
        div[data-testid="stSlider"],
        div[data-testid="stRadio"],
        div[data-testid="stTextInput"],
        div[data-testid="stMultiSelect"] {{
            animation: fadeUp_{anim_id} 0.45s ease-out both;
            animation-delay: 0.1s;
        }}

        div[data-testid="stMetric"] {{
            animation: fadeIn_{anim_id} 0.35s ease-out both;
        }}
        div[data-testid="stMetric"]:nth-of-type(1) {{ animation-delay: 0s; }}
        div[data-testid="stMetric"]:nth-of-type(2) {{ animation-delay: 0.03s; }}
        div[data-testid="stMetric"]:nth-of-type(3) {{ animation-delay: 0.06s; }}
        div[data-testid="stMetric"]:nth-of-type(4) {{ animation-delay: 0.09s; }}

        .stPlotlyChart {{
            animation: fadeIn_{anim_id} 0.45s ease-out both;
        }}
        [data-testid="stDataFrame"] {{
            animation: slideUp_{anim_id} 0.45s ease-out both;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _stat_card(label: str, value: str, suffix: str = "", tone: str = "dark") -> None:
    st.markdown(
        f"""
        <div class="stat-card {tone}">
            <div class="stat-label">{label}</div>
            <div><span class="stat-value">{value}</span><span class="stat-sub">{suffix}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _mini_card(label: str, value: str, up: bool = True) -> None:
    badge = "up" if up else "down"
    st.markdown(
        f"""
        <div class="mini-card">
            <strong>{label}</strong>
            <span class="mini-badge {badge}">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _block_title(title: str) -> None:
    st.markdown(f'<div class="block-title">{title}</div>', unsafe_allow_html=True)


def _help_line(text: str) -> None:
    st.markdown(f'<div class="help-line">{text}</div>', unsafe_allow_html=True)


def _dark_layout(fig, height: int, extra_right: int = 0, extra_bottom: int = 0):
    fig.update_layout(
        height=height,
        paper_bgcolor="#19191f",
        plot_bgcolor="#19191f",
        font=dict(color="#d4d4d8"),
        margin=dict(l=10, r=10 + extra_right, t=44, b=10 + extra_bottom),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#71717a"),
            itemwidth=30,
        ),
        title_font=dict(color="#d4d4d8", size=15),
        xaxis=dict(gridcolor="#27272e", zerolinecolor="#27272e"),
        yaxis=dict(gridcolor="#27272e", zerolinecolor="#27272e"),
    )
    return fig


def _fallback_quote(row: pd.Series) -> dict[str, float]:
    return {
        "live_price": float(row.get("latest_price") or row.get("close_price") or 0),
        "open": float(row.get("open_price") or 0),
        "high": float(row.get("high_price") or 0),
        "low": float(row.get("low_price") or 0),
        "previous_close": float(row.get("close_price") or 0),
    }


@st.fragment(run_every="30s")
def _live_market_stream(latest_df: pd.DataFrame) -> None:
    # Inject dynamic update style inside fragment to trigger animation on update
    update_id = int(time.time() * 1000)
    st.markdown(
        f"""
        <style>
        @keyframes fadeIn_{update_id} {{
            0% {{ opacity: 0.75; }}
            100% {{ opacity: 1; }}
        }}
        @keyframes slideUp_{update_id} {{
            0% {{
                opacity: 0.8;
                transform: translateY(4px);
            }}
            100% {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        div[data-testid="stMetric"] {{
            animation: fadeIn_{update_id} 0.35s ease-out both;
        }}
        div[data-testid="stMetric"]:nth-of-type(1) {{ animation-delay: 0s; }}
        div[data-testid="stMetric"]:nth-of-type(2) {{ animation-delay: 0.03s; }}
        div[data-testid="stMetric"]:nth-of-type(3) {{ animation-delay: 0.06s; }}
        div[data-testid="stMetric"]:nth-of-type(4) {{ animation-delay: 0.09s; }}

        .stPlotlyChart {{
            animation: fadeIn_{update_id} 0.45s ease-out both;
        }}
        [data-testid="stDataFrame"] {{
            animation: slideUp_{update_id} 0.45s ease-out both;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="live-indicator-container">
            <span class="live-dot"></span>
            <span class="live-title">Live Prices</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _help_line("Prices update every 30 seconds from Finnhub. Green = price went up, Red = price went down.")

    previous_prices = st.session_state.setdefault("live_previous_prices", {})
    history = st.session_state.setdefault("live_stream_history", [])
    # Phnom Penh is UTC+7
    timestamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=7)))
    rows: list[dict[str, object]] = []

    for _, row in latest_df.iterrows():
        symbol = str(row["symbol"]).upper()
        quote = _fallback_quote(row)
        source = "Warehouse DWH"

        live_price = quote["live_price"]
        prev_tick = previous_prices.get(symbol)
        tick_change = live_price - prev_tick if prev_tick is not None else 0.0
        previous_prices[symbol] = live_price

        previous_close = quote["previous_close"]
        day_change = live_price - previous_close if previous_close else 0.0
        day_change_pct = (day_change / previous_close * 100) if previous_close else 0.0
        direction = "up" if day_change > 0 else "down" if day_change < 0 else "flat"
        tick_status = "up" if tick_change > 0 else "down" if tick_change < 0 else "flat"

        rows.append({
            "Ticker": symbol,
            "Company": row["commodity_name"],
            "Price (USD)": live_price,
            "Prev Close": previous_close,
            "vs Close": direction.upper(),
            "Day Change": day_change,
            "Day %": day_change_pct,
            "Since Last Update": tick_change,
            "Trend": tick_status.upper(),
            "Source": source,
            "Last Updated": timestamp.strftime("%H:%M:%S"),
        })
        history.append({
            "time": timestamp,
            "symbol": symbol,
            "price": live_price,
            "day_change": day_change,
            "day_change_pct": day_change_pct,
        })

    st.session_state["live_stream_history"] = history[-240:]
    live_df = pd.DataFrame(rows)

    up_count = int((live_df["vs Close"] == "UP").sum())
    down_count = int((live_df["vs Close"] == "DOWN").sum())
    flat_count = int((live_df["vs Close"] == "FLAT").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks Tracked", len(live_df))
    c2.metric("Going Up", up_count)
    c3.metric("Going Down", down_count)
    c4.metric("Unchanged", flat_count)

    # -- Live price chart --
    chart_df = pd.DataFrame(st.session_state["live_stream_history"])
    if not chart_df.empty:
        fig = px.line(
            chart_df,
            x="time",
            y="price",
            color="symbol",
            markers=True,
            title="Live Stock Prices (updates every 30s)",
            labels={
                "time": "Time",
                "price": "Price (USD)",
                "symbol": "Ticker",
                "day_change": "Day Change",
                "day_change_pct": "Day Change %",
            },
            hover_data={
                "price": ":,.2f",
                "day_change": ":+,.2f",
                "day_change_pct": ":+,.2f",
            },
            color_discrete_sequence=["#8eaadc", "#a78bba", "#9ec5a0", "#d4a06a", "#c48888", "#7cb8b2"],
        )
        st.plotly_chart(_dark_layout(fig, 310), width="stretch")

    # -- Live price table --
    display_cols = [
        "Ticker", "Company", "Price (USD)", "vs Close", "Day Change", "Day %",
        "Since Last Update", "Trend", "Prev Close", "Source", "Last Updated",
    ]
    display_df = live_df[display_cols].copy()
    display_df.index = range(1, len(display_df) + 1)
    st.dataframe(
        display_df.style
        .map(_colour_change, subset=["Day Change", "Day %", "Since Last Update"])
        .map(_tick_colour, subset=["vs Close", "Trend"])
        .format({
            "Price (USD)": "${:,.2f}",
            "Prev Close": "${:,.2f}",
            "Day Change": "{:+,.2f}",
            "Day %": "{:+,.2f}%",
            "Since Last Update": "{:+,.4f}",
        }),
        width="stretch",
        height=360,
    )

    st.caption(f"Last updated at {timestamp.strftime('%H:%M:%S')} \u00b7 Only this section refreshes automatically.")


@st.cache_data(ttl=300)
def _load_predictions() -> pd.DataFrame:
    engine = _get_engine()
    with engine.connect() as conn:
        if not (
            _table_exists(conn, "gold.dim_commodity")
            and _table_exists(conn, "gold.fact_predictions")
        ):
            return pd.DataFrame()

        return pd.read_sql(text("""
            SELECT d.symbol,
                   d.commodity_name,
                   d.commodity_category,
                   p.timeframe,
                   p.indicator,
                   p.model_name,
                   p.predicted_time,
                   p.predicted_value,
                   p.confidence_low,
                   p.confidence_high,
                   p.run_at
            FROM   gold.fact_predictions p
            JOIN   gold.dim_commodity    d ON d.commodity_id = p.commodity_id
            ORDER  BY d.symbol, p.timeframe, p.indicator, p.model_name, p.predicted_time
        """), conn)


def _create_forecast_chart(hist_df: pd.DataFrame, pred_df: pd.DataFrame, ticker: str, indicator: str, model_name: str, timeframe: str) -> go.Figure:
    h_sub = hist_df[(hist_df["symbol"] == ticker) & (hist_df["timeframe"] == timeframe)].sort_values("time_index")
    p_sub = pred_df[(pred_df["symbol"] == ticker) & (pred_df["timeframe"] == timeframe) & (pred_df["indicator"] == indicator) & (pred_df["model_name"] == model_name)].sort_values("predicted_time")

    fig = go.Figure()

    if h_sub.empty and p_sub.empty:
        return fig

    ind_label = {
        "close_price": "Close Price",
        "open_price": "Open Price",
        "high_price": "High Price",
        "low_price": "Low Price",
        "latest_price": "Latest Price"
    }.get(indicator, indicator)

    # 1. Plot history
    if not h_sub.empty:
        fig.add_trace(go.Scatter(
            x=h_sub["time_index"],
            y=h_sub[indicator],
            mode="lines+markers",
            name=f"Historical {ind_label}",
            line=dict(color="#a78bba", width=2, shape="spline"),
            marker=dict(size=5),
            hovertemplate="Date: %{x}<br>Price: $%{y:,.2f}<extra>History</extra>"
        ))

    # 2. Plot predictions
    if not p_sub.empty:
        # Prepend last history row to prediction rows to seamlessly bridge lines
        if not h_sub.empty:
            last_hist = h_sub.iloc[-1]
            bridge_df = pd.DataFrame([{
                "predicted_time": pd.to_datetime(last_hist["time_index"]).date(),
                "predicted_value": float(last_hist[indicator]),
                "confidence_low": float(last_hist[indicator]),
                "confidence_high": float(last_hist[indicator])
            }])
            p_sub_plot = pd.concat([bridge_df, p_sub], ignore_index=True)
        else:
            p_sub_plot = p_sub

        # Transparent lower bound line
        fig.add_trace(go.Scatter(
            x=p_sub_plot["predicted_time"],
            y=p_sub_plot["confidence_low"],
            mode="lines",
            line=dict(width=0, shape="spline"),
            showlegend=False,
            hovertemplate="Lower Bound: $%{y:,.2f}<extra></extra>"
        ))

        # Upper bound line with fill to previous trace
        fig.add_trace(go.Scatter(
            x=p_sub_plot["predicted_time"],
            y=p_sub_plot["confidence_high"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(142, 170, 220, 0.08)",
            line=dict(width=0, shape="spline"),
            name="95% Confidence Interval",
            hovertemplate="Upper Bound: $%{y:,.2f}<extra></extra>"
        ))

        # Main forecast line
        fig.add_trace(go.Scatter(
            x=p_sub_plot["predicted_time"],
            y=p_sub_plot["predicted_value"],
            mode="lines+markers",
            name=f"Forecast ({model_name})",
            line=dict(color="#8eaadc", width=2, dash="dash", shape="spline"),
            marker=dict(size=6, symbol="diamond"),
            hovertemplate="Date: %{x}<br>Forecast: $%{y:,.2f}<extra>Prediction</extra>"
        ))

    model_title = "Linear Trend" if model_name == "linear_trend" else "Holt's Smoothing"
    fig.update_layout(
        title=f"{ticker} {ind_label} Forecast \u2014 {model_title}",
        xaxis=dict(
            title="Time",
            gridcolor="#27272e",
            type="date",
        ),
        yaxis=dict(
            title="Price (USD)",
            tickformat="$",
            gridcolor="#27272e",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=40, t=70, b=20)
    )

    return fig


def main() -> None:
    st.set_page_config(
        page_title="Finnhub Trading Dashboard",
        page_icon="$",
        layout="wide",
    )
    _inject_style()

    try:
        df = _load_prices()
    except Exception as exc:
        st.error(f"Cannot connect to the database: {exc}")
        st.info("Run `python main.py` first to populate the warehouse, then reload.")
        return

    try:
        pred_df = _load_predictions()
    except Exception:
        pred_df = pd.DataFrame()

    if df.empty:
        st.warning("Warehouse is empty. Run `python main.py` to ingest Finnhub stock prices.")
        return

    # Extract all sectors
    all_categories = sorted(df["commodity_category"].dropna().unique().tolist())

    with st.sidebar:
        st.markdown('<div class="side-title">Stock Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-subtitle">Real-time prices from Finnhub API</div>', unsafe_allow_html=True)

        # Navigation toggle
        st.markdown('<div class="side-section">Navigation</div>', unsafe_allow_html=True)
        selected_view = st.radio(
            "Select View",
            ["Live & History", "ML Forecasting"],
            label_visibility="collapsed",
            key="side_view_selector",
        )

        # Timeframe Selector (Focus on Weeks by default)
        st.markdown('<div class="side-section">Granularity</div>', unsafe_allow_html=True)
        selected_tf_label = st.selectbox(
            "Timeframe Granularity",
            ["Days", "Weeks", "Months", "Years"],
            index=1,  # Default to Weeks (per instruction)
            help="Select granularity of the historical and forecast views"
        )
        tf_map = {
            "Days": "day",
            "Weeks": "week",
            "Months": "month",
            "Years": "year"
        }
        selected_tf = tf_map[selected_tf_label]

        # Filter database by selected timeframe granularity
        df_tf = df[df["timeframe"] == selected_tf].copy()

        if df_tf.empty:
            st.error(f"No database records found for timeframe: {selected_tf_label}. Ingesting...")
            return

        all_dates = sorted(df_tf["time_index"].dropna().unique().tolist())

        if selected_view == "Live & History":
            st.markdown('<div class="side-section">Filter Stocks</div>', unsafe_allow_html=True)
            selected_categories = st.multiselect(
                "Filter by sector",
                all_categories,
                default=all_categories,
                help="Choose which market sectors to display",
            )

            # Stretches range slider over date domain
            min_date = pd.to_datetime(all_dates[0]).date()
            max_date = pd.to_datetime(all_dates[-1]).date()
            if min_date == max_date:
                max_date = max_date + dt.timedelta(days=1)

            date_range = st.slider(
                "Date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                help="Filter the historical price charts below",
            )

            search_text = st.text_input("Search by ticker or company", placeholder="e.g. AAPL or Apple")
            if search_text:
                match_count = df_tf[
                    df_tf["symbol"].str.contains(search_text, case=False, na=False)
                    | df_tf["commodity_name"].str.contains(search_text, case=False, na=False)
                ]["symbol"].nunique()
                st.caption(f"Found {match_count} matching stock{'s' if match_count != 1 else ''}.")

            st.markdown('<div class="side-section">How it works</div>', unsafe_allow_html=True)
            st.caption("The Live Prices section updates automatically every 30 seconds without refreshing the full page.")
        else:
            st.markdown('<div class="side-section">ML Predictions</div>', unsafe_allow_html=True)
            st.caption("Forecasting models are automatically fit, evaluated, and logged to MLflow via the containerized ML pipeline.")

    if selected_view == "Live & History":
        filtered_df = df_tf[
            df_tf["commodity_category"].isin(selected_categories)
            & pd.to_datetime(df_tf["time_index"]).dt.date.between(date_range[0], date_range[1])
        ].copy()

        if search_text:
            filtered_df = filtered_df[
                filtered_df["symbol"].str.contains(search_text, case=False, na=False)
                | filtered_df["commodity_name"].str.contains(search_text, case=False, na=False)
            ].copy()

        latest_df = _latest_snapshot(filtered_df)
        if latest_df.empty:
            st.warning("No tickers match the current filters.")
            return

        rising = int((latest_df["price_change_pct"] > 0).sum())
        falling = int((latest_df["price_change_pct"] < 0).sum())
        high_vol = int((latest_df["volatility_level"] == "high").sum())
        avg_close = latest_df["close_price"].mean()
        avg_change = latest_df["price_change_pct"].mean()

        st.markdown(
            f"""
            <div class="commodity-header">
                <h1>Stock Market Dashboard ({selected_tf_label})</h1>
                <p>Track live stock prices, compare trends, and monitor your watchlist — powered by Finnhub.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        top1, top2, top3, top4 = st.columns([1.1, 1.1, 1, 1])
        with top1:
            _stat_card("Avg Close Price", f"${avg_close:,.2f}", "", "purple")
        with top2:
            _stat_card("Avg Change", f"{avg_change:,.2f}", "%", "cyan")
        with top3:
            _mini_card("Stocks Going Up", f"+{rising}", True)
            _mini_card("Stocks Going Down", f"-{falling}", False)
        with top4:
            _mini_card("Stocks in Watchlist", str(len(latest_df)), True)
            _mini_card("High Volatility", str(high_vol), False)

        _live_market_stream(latest_df)

        bottom_left, bottom_right = st.columns([1.55, 1])
        with bottom_left:
            _block_title(f"Price History ({selected_tf_label})")
            _help_line("How stock prices have changed over time. Smooth spline curves make trends easy to follow.")

            all_symbols_in_filters = sorted(filtered_df["symbol"].unique())

            c_sel1, c_sel2 = st.columns([2.2, 1])
            with c_sel1:
                selected_symbols_history = st.multiselect(
                    "Compare stocks",
                    options=all_symbols_in_filters,
                    default=all_symbols_in_filters[:4],
                    key="history_symbols_multiselect",
                    help="Select which stocks to compare on the historical price chart below to avoid visual clutter"
                )
            with c_sel2:
                selected_metric = st.selectbox(
                    "Price Metric",
                    options=["close_price", "open_price", "high_price", "low_price"],
                    format_func=lambda x: {
                        "close_price": "Close Price",
                        "open_price": "Open Price",
                        "high_price": "High Price",
                        "low_price": "Low Price"
                    }.get(x, x),
                    help="Select which price point to plot"
                )

            history_plot_df = filtered_df[filtered_df["symbol"].isin(selected_symbols_history)].copy()

            if not history_plot_df.empty:
                fig = go.Figure()

                is_single = len(selected_symbols_history) == 1
                metric_label = {
                    "close_price": "Close Price",
                    "open_price": "Open Price",
                    "high_price": "High Price",
                    "low_price": "Low Price"
                }.get(selected_metric, selected_metric)

                CHART_THEMES = [
                    {"line": "#8eaadc", "fill": "rgba(142, 170, 220, 0.06)"},
                    {"line": "#a78bba", "fill": "rgba(167, 139, 186, 0.06)"},
                    {"line": "#9ec5a0", "fill": "rgba(158, 197, 160, 0.06)"},
                    {"line": "#d4a06a", "fill": "rgba(212, 160, 106, 0.06)"},
                    {"line": "#c48888", "fill": "rgba(196, 136, 136, 0.06)"},
                    {"line": "#7cb8b2", "fill": "rgba(124, 184, 178, 0.06)"},
                    {"line": "#b8b88e", "fill": "rgba(184, 184, 142, 0.06)"},
                    {"line": "#a1a1aa", "fill": "rgba(161, 161, 170, 0.06)"},
                ]

                for idx, sym in enumerate(selected_symbols_history):
                    sym_df = history_plot_df[history_plot_df["symbol"] == sym].sort_values("time_index")
                    if sym_df.empty:
                        continue

                    theme = CHART_THEMES[idx % len(CHART_THEMES)]
                    color = theme["line"]
                    fill_color = theme["fill"]

                    if is_single:
                        fig.add_trace(go.Scatter(
                            x=sym_df["time_index"],
                            y=sym_df[selected_metric],
                            mode="lines+markers",
                            name=f"{sym} ({metric_label})",
                            line=dict(color=color, width=2, shape="spline"),
                            marker=dict(size=5, symbol="circle", line=dict(color="#19191f", width=1)),
                            fill="tozeroy",
                            fillcolor=fill_color,
                            hovertemplate=f"<b>{sym}</b><br>Date: %{{x}}<br>{metric_label}: $%{{y:,.2f}}<extra></extra>"
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=sym_df["time_index"],
                            y=sym_df[selected_metric],
                            mode="lines+markers",
                            name=sym,
                            line=dict(color=color, width=2, shape="spline"),
                            marker=dict(size=5, symbol="circle", line=dict(color="#19191f", width=1)),
                            hovertemplate=f"<b>{sym}</b><br>Date: %{{x}}<br>{metric_label}: $%{{y:,.2f}}<extra></extra>"
                        ))

                fig.update_layout(
                    title=dict(
                        text=f"Historical {metric_label} Trends ({selected_tf_label})",
                        font=dict(color="#d4d4d8", size=15, weight="bold")
                    ),
                    xaxis=dict(
                        title=dict(text="Date", font=dict(color="#71717a")),
                        type="date",
                        gridcolor="#27272e",
                        tickfont=dict(color="#71717a")
                    ),
                    yaxis=dict(
                        title=dict(text="Price (USD)", font=dict(color="#71717a")),
                        tickformat="$",
                        gridcolor="#27272e",
                        tickfont=dict(color="#71717a")
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.04,
                        xanchor="right",
                        x=1,
                        font=dict(color="#71717a", size=11)
                    ),
                    margin=dict(l=15, r=15, t=65, b=15),
                    hovermode="x unified"
                )

                st.plotly_chart(_dark_layout(fig, 330, extra_right=0), width="stretch")
            else:
                st.info("Select at least one stock to plot the price history.")

        with bottom_right:
            _block_title("Sector Breakdown")
            _help_line("How many stocks belong to each market sector.")
            sector_counts = latest_df.groupby("commodity_category", as_index=False).agg(count=("symbol", "count"))
            sector_counts = sector_counts.rename(columns={"commodity_category": "Sector"})
            pie = px.pie(
                sector_counts,
                names="Sector",
                values="count",
                hole=0.62,
                title="Stocks by Sector",
                color_discrete_sequence=["#8eaadc", "#a78bba", "#9ec5a0", "#d4a06a", "#c48888"],
            )
            pie.update_traces(textfont_color="#d4d4d8")
            st.plotly_chart(_dark_layout(pie, 360, extra_right=30, extra_bottom=30), width="stretch")
    else:
        # ML forecasting page
        if pred_df.empty:
            st.warning("No prediction data found in database. Run the pipeline to generate forecasts.")
            return

        # Check if predictions exist for selected timeframe
        pred_df_tf = pred_df[pred_df["timeframe"] == selected_tf].copy()
        if pred_df_tf.empty:
            st.warning(f"No forecasting predictions generated for {selected_tf_label} yet. Execute train.py to build models.")
            return

        st.markdown(
            f"""
            <div class="commodity-header">
                <h1>ML Price Forecasting ({selected_tf_label})</h1>
                <p>Track advanced predictive analytics driven by Linear Trend and Holt's Exponential Smoothing models — logged via MLflow.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Forecasting controls
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            all_symbols = sorted(pred_df_tf["symbol"].unique())
            selected_ticker = st.selectbox(
                "Select Stock Ticker",
                options=all_symbols,
                help="Select which stock symbol to analyze forecasts for"
            )
        with ctrl2:
            model_options = sorted(pred_df_tf["model_name"].unique())
            model_name_map = {
                "linear_trend": "Linear Trend",
                "holt_smoothing": "Holt's Exponential Smoothing"
            }
            selected_model_raw = st.selectbox(
                "Select Forecasting Model",
                options=model_options,
                format_func=lambda x: model_name_map.get(x, x),
                help="Choose which algorithm's forecast to display"
            )
        with ctrl3:
            indicator_options = sorted(pred_df_tf["indicator"].unique())
            indicator_name_map = {
                "close_price": "Close Price",
                "open_price": "Open Price",
                "high_price": "High Price",
                "low_price": "Low Price",
                "latest_price": "Latest Price"
            }
            selected_indicator = st.selectbox(
                "Select Target Indicator",
                options=indicator_options,
                format_func=lambda x: indicator_name_map.get(x, x),
                help="Select which price field to forecast"
            )

        # Load run metadata
        ticker_preds = pred_df_tf[
            (pred_df_tf["symbol"] == selected_ticker) &
            (pred_df_tf["indicator"] == selected_indicator) &
            (pred_df_tf["model_name"] == selected_model_raw)
        ]

        f_top1, f_top2 = st.columns(2)
        if not ticker_preds.empty:
            run_at_val = pd.to_datetime(ticker_preds["run_at"].iloc[0]).tz_convert("Asia/Phnom_Penh").strftime("%Y-%m-%d %H:%M:%S")
            with f_top1:
                _stat_card("Active Forecast Model", model_name_map.get(selected_model_raw, selected_model_raw), f" ({selected_ticker})", "purple")
            with f_top2:
                _stat_card("Last Pipeline Run Time", run_at_val, " (Phnom Penh)", "cyan")

        # Joined visual timeline chart
        st.markdown('<div class="block-title">Forecast Projections Timeline</div>', unsafe_allow_html=True)
        st.markdown('<p class="help-line">History line (solid) joined seamlessly with forecasting line (dashed) surrounded by a 95% confidence interval shaded region.</p>', unsafe_allow_html=True)

        fig = _create_forecast_chart(df, pred_df_tf, selected_ticker, selected_indicator, selected_model_raw, selected_tf)
        st.plotly_chart(_dark_layout(fig, 380, extra_right=30), width="stretch")

        # Table of forecast values
        st.markdown('<div class="block-title">Detailed Forecast Values</div>', unsafe_allow_html=True)
        st.markdown('<p class="help-line">Exact values and confidence boundaries for the projected 3-period horizon. Spread % represents bound uncertainty.</p>', unsafe_allow_html=True)

        if not ticker_preds.empty:
            display_preds = ticker_preds[["predicted_time", "predicted_value", "confidence_low", "confidence_high"]].copy()
            # Convert values to float
            display_preds["predicted_value"] = display_preds["predicted_value"].astype(float)
            display_preds["confidence_low"] = display_preds["confidence_low"].astype(float)
            display_preds["confidence_high"] = display_preds["confidence_high"].astype(float)
            display_preds["Confidence Spread %"] = (
                (display_preds["confidence_high"] - display_preds["confidence_low"]) / display_preds["predicted_value"] * 100
            )

            # Format columns
            display_preds = display_preds.rename(columns={
                "predicted_time": "Projected Date",
                "predicted_value": "Forecasted Price (USD)",
                "confidence_low": "Lower Confidence Bound",
                "confidence_high": "Upper Confidence Bound",
            })

            display_preds.index = range(1, len(display_preds) + 1)

            st.dataframe(
                display_preds.style.format({
                    "Forecasted Price (USD)": "${:,.2f}",
                    "Lower Confidence Bound": "${:,.2f}",
                    "Upper Confidence Bound": "${:,.2f}",
                    "Confidence Spread %": "{:,.2f}%",
                }),
                width="stretch",
            )
        else:
            st.info("No forecast values available for selected combination.")


if __name__ == "__main__":
    main()

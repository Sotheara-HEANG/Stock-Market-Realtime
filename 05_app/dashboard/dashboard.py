"""
dashboard.py - Streamlit dashboard for Finnhub stock market analytics.
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
    "year",
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
                   f.year,
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
            ORDER  BY d.symbol, f.year
        """), conn)


def _latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return (
        df.sort_values(["symbol", "year"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("close_price", ascending=False)
        .reset_index(drop=True)
    )


def _colour_change(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color: #15803d; font-weight: bold"
    if val < 0:
        return "color: #b91c1c; font-weight: bold"
    return "color: #475569; font-weight: bold"


def _tick_colour(val):
    if str(val).lower() == "up":
        return "color: #21d4fd; font-weight: bold"
    if str(val).lower() == "down":
        return "color: #ff5bc8; font-weight: bold"
    return "color: #a8b3cf; font-weight: bold"


def _inject_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0c1020;
            --panel: #151a2e;
            --panel-2: #1a2038;
            --line: #29304b;
            --text: #eef3ff;
            --muted: #9aa6c3;
            --cyan: #21d4fd;
            --purple: #c84cff;
            --pink: #ff5bc8;
        }
        html, body, [data-testid="stAppViewContainer"] { background: var(--bg); }
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 9%, rgba(200,76,255,.26), transparent 23rem),
                radial-gradient(circle at 95% 88%, rgba(33,212,253,.18), transparent 22rem),
                #0c1020;
        }
        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 2.8rem;
        }
        [data-testid="stSidebar"] {
            background: #14182b;
            border-right: 1px solid #272e48;
        }
        [data-testid="stSidebar"] * {
            color: #e8edff;
        }
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            color: #f0f4ff !important;
            background-color: #1e2440 !important;
            border: 1px solid #3a4268 !important;
        }
        [data-testid="stSidebar"] input::placeholder {
            color: #6b7a9e !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: #a8b3cf;
        }
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: var(--text);
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(145deg, #171d33, #111629);
            border: 1px solid #29304b;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 20px 44px rgba(0,0,0,.28);
        }
        div[data-testid="stMetricLabel"] p {
            color: #a8b3cf;
            font-size: 0.82rem;
        }
        div[data-testid="stMetricValue"] {
            color: #f8fbff;
            font-weight: 700;
        }
        .commodity-header {
            background: linear-gradient(145deg, rgba(21,26,46,.96), rgba(11,15,29,.96));
            border: 1px solid #2a3150;
            border-radius: 24px;
            padding: 26px 30px;
            margin-bottom: 18px;
            box-shadow: 0 28px 80px rgba(0,0,0,.35);
        }
        .commodity-header h1 {
            color: #f8fbff;
            font-size: 2rem;
            margin: 0 0 4px 0;
            letter-spacing: 0;
        }
        .commodity-header p {
            margin: 0;
            color: #9aa6c3;
        }
        .side-title {
            color: #f8fbff;
            font-size: 1.15rem;
            font-weight: 800;
            margin: .3rem 0 .35rem 0;
        }
        .side-subtitle {
            color: #8f9bb8;
            font-size: .76rem;
            margin-bottom: 1.05rem;
        }
        .side-active {
            background: linear-gradient(90deg, rgba(200,76,255,.45), rgba(33,212,253,.08));
            border-left: 4px solid var(--purple);
            border-radius: 8px;
            color: #fff;
            font-size: .95rem;
            font-weight: 800;
            margin: .5rem 0 1rem 0;
            padding: .78rem .9rem;
        }
        .side-section {
            border-top: 1px solid #282f48;
            color: #aeb9d8;
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .06em;
            margin-top: 1rem;
            padding-top: 1rem;
            text-transform: uppercase;
        }
        .stat-card {
            border: 1px solid rgba(255,255,255,.12);
            border-radius: 14px;
            min-height: 126px;
            padding: 20px 22px;
            box-shadow: 0 18px 42px rgba(0,0,0,.22);
        }
        .stat-card.purple {
            background: linear-gradient(135deg, #df58ff 0%, #8d62ff 100%);
        }
        .stat-card.cyan {
            background: linear-gradient(135deg, #19d5d2 0%, #4387ff 100%);
        }
        .stat-card.dark {
            background: linear-gradient(145deg, #171d33, #111629);
        }
        .stat-label {
            color: rgba(255,255,255,.78);
            font-size: .84rem;
            font-weight: 650;
            margin-bottom: 18px;
        }
        .stat-value {
            color: #ffffff;
            font-size: 2.1rem;
            font-weight: 800;
            line-height: 1;
        }
        .stat-sub {
            color: rgba(255,255,255,.74);
            font-size: .82rem;
            margin-left: .3rem;
        }
        .mini-card {
            align-items: center;
            background: linear-gradient(145deg, #171d33, #111629);
            border: 1px solid #29304b;
            border-radius: 13px;
            display: flex;
            justify-content: space-between;
            min-height: 58px;
            padding: 12px 16px;
        }
        .mini-card strong { color: #f8fbff; font-size: .86rem; }
        .mini-badge {
            border-radius: 999px;
            color: #fff;
            font-size: .74rem;
            font-weight: 800;
            padding: 4px 10px;
        }
        .mini-badge.up { background: rgba(33,212,253,.38); }
        .mini-badge.down { background: rgba(255,91,200,.38); }
        .block-title {
            color: #f8fbff;
            font-size: 1rem;
            font-weight: 800;
            margin: 18px 0 10px;
        }
        .help-line {
            color: #a8b3cf;
            font-size: .86rem;
            margin: -4px 0 12px 0;
        }
        .legend-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 4px 0 14px;
        }
        .legend-pill {
            background: #151a2e;
            border: 1px solid #29304b;
            border-radius: 999px;
            color: #dbe5ff;
            font-size: .8rem;
            font-weight: 700;
            padding: 7px 11px;
        }
        .legend-pill.up { border-color: rgba(33,212,253,.55); color: #21d4fd; }
        .legend-pill.down { border-color: rgba(255,91,200,.55); color: #ff5bc8; }
        .stPlotlyChart {
            background: #151a2e;
            border: 1px solid #29304b;
            border-radius: 14px;
            box-shadow: 0 18px 42px rgba(0,0,0,.18);
            padding: 10px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #29304b;
            border-radius: 14px;
            overflow: hidden;
        }
        div[data-testid="stSelectbox"],
        div[data-testid="stSlider"],
        div[data-testid="stRadio"],
        div[data-testid="stTextInput"],
        div[data-testid="stMultiSelect"] {
            color: #f8fbff;
        }
        /* Multiselect tag pills */
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
            background-color: #2a3158 !important;
            border: 1px solid #4a5280 !important;
            color: #dbe5ff !important;
        }
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] span[data-baseweb="tag"] span {
            color: #dbe5ff !important;
        }
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] span[data-baseweb="tag"] svg {
            fill: #8f9bb8 !important;
        }
        /* Multiselect dropdown background */
        [data-testid="stSidebar"] [data-testid="stMultiSelect"] > div > div {
            background-color: #1e2440 !important;
            border-color: #3a4268 !important;
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

        /* ── Premium UI Micro-Animations ────────────────────────────────── */
        @keyframes premiumPopUp {
            0% {
                opacity: 0;
                transform: translateY(14px) scale(0.97);
                filter: brightness(1.2) blur(1px);
            }
            100% {
                opacity: 1;
                transform: translateY(0) scale(1);
                filter: brightness(1) blur(0);
            }
        }
        @keyframes liveMetricUpdate {
            0% {
                opacity: 0.7;
                transform: scale(0.96);
                border-color: var(--cyan);
                box-shadow: 0 0 18px rgba(33, 212, 253, 0.4);
            }
            40% {
                transform: scale(1.025);
                border-color: var(--purple);
                box-shadow: 0 0 22px rgba(200, 76, 255, 0.35);
            }
            100% {
                opacity: 1;
                transform: scale(1);
                border-color: #29304b;
                box-shadow: 0 20px 44px rgba(0,0,0,.28);
            }
        }
        @keyframes chartUpdate {
            0% {
                opacity: 0.8;
                transform: scale(0.99);
                border-color: var(--cyan);
                box-shadow: 0 0 15px rgba(33, 212, 253, 0.22);
            }
            100% {
                opacity: 1;
                transform: scale(1);
                border-color: #29304b;
                box-shadow: 0 18px 42px rgba(0,0,0,.18);
            }
        }
        @keyframes tableUpdate {
            0% {
                opacity: 0.8;
                transform: translateY(6px);
                border-color: var(--purple);
                box-shadow: 0 0 15px rgba(200, 76, 255, 0.22);
            }
            100% {
                opacity: 1;
                transform: translateY(0);
                border-color: #29304b;
                box-shadow: none;
            }
        }
        @keyframes livePulseDot {
            0% {
                transform: scale(0.9);
                box-shadow: 0 0 0 0 rgba(33, 212, 253, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 8px rgba(33, 212, 253, 0);
            }
            100% {
                transform: scale(0.9);
                box-shadow: 0 0 0 0 rgba(33, 212, 253, 0);
            }
        }

        /* ── Applying Animations ── */
        .commodity-header {
            animation: premiumPopUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .stat-card.purple {
            animation: premiumPopUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
            animation-delay: 0.04s;
        }
        .stat-card.cyan {
            animation: premiumPopUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
            animation-delay: 0.08s;
        }
        .stat-card.dark {
            animation: premiumPopUp 0.55s cubic-bezier(0.16, 1, 0.3, 1) both;
            animation-delay: 0.12s;
        }
        .mini-card {
            animation: premiumPopUp 0.58s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        .mini-card:nth-of-type(1) { animation-delay: 0.12s; }
        .mini-card:nth-of-type(2) { animation-delay: 0.16s; }

        /* Form Controls staggered load */
        div[data-testid="stSelectbox"],
        div[data-testid="stSlider"],
        div[data-testid="stRadio"],
        div[data-testid="stTextInput"],
        div[data-testid="stMultiSelect"] {
            animation: premiumPopUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
            animation-delay: 0.18s;
        }

        /* Live fragments entry & update animations */
        div[data-testid="stMetric"] {
            animation: liveMetricUpdate 0.55s cubic-bezier(0.34, 1.56, 0.64, 1) both;
        }
        /* Stagger metric cards updates slightly */
        div[data-testid="stMetric"]:nth-of-type(1) { animation-delay: 0s; }
        div[data-testid="stMetric"]:nth-of-type(2) { animation-delay: 0.04s; }
        div[data-testid="stMetric"]:nth-of-type(3) { animation-delay: 0.08s; }
        div[data-testid="stMetric"]:nth-of-type(4) { animation-delay: 0.12s; }

        .stPlotlyChart {
            animation: chartUpdate 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        }
        [data-testid="stDataFrame"] {
            animation: tableUpdate 0.62s cubic-bezier(0.16, 1, 0.3, 1) both;
        }

        /* Pulse indicator styling */
        .live-indicator-container {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            margin: 18px 0 10px;
        }
        .live-dot {
            width: 8px;
            height: 8px;
            background-color: var(--cyan);
            border-radius: 50%;
            display: inline-block;
            animation: livePulseDot 2s infinite ease-in-out;
        }
        .live-title {
            color: #f8fbff;
            font-size: 1rem;
            font-weight: 800;
            margin: 0;
        }
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
        paper_bgcolor="#151a2e",
        plot_bgcolor="#151a2e",
        font=dict(color="#dbe5ff"),
        margin=dict(l=10, r=10 + extra_right, t=44, b=10 + extra_bottom),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aeb9d8"),
            itemwidth=30,
        ),
        title_font=dict(color="#f8fbff", size=15),
        xaxis=dict(gridcolor="#2c3450", zerolinecolor="#2c3450"),
        yaxis=dict(gridcolor="#2c3450", zerolinecolor="#2c3450"),
    )
    return fig


# ── Batched quote cache ─────────────────────────────────────────────────────
# Fetches all symbols in one burst, caches them for _QUOTE_CACHE_TTL seconds
# so the 10-30s fragment re-runs never duplicate HTTP calls.
_QUOTE_CACHE_TTL = 25  # seconds
_quote_cache: dict[str, dict] = {}      # symbol -> quote dict
_quote_cache_ts: float = 0.0            # last fetch epoch
_RATE_LIMIT_DELAY = 0.35                # seconds between Finnhub calls


def _finnhub_quote_single(symbol: str, token: str) -> dict[str, float]:
    """Fetch a single quote from Finnhub (internal, use _finnhub_quotes_batch)."""
    response = requests.get(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": symbol, "token": token},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict) or not data.get("c"):
        raise RuntimeError(f"No live quote for {symbol}")
    return {
        "live_price": float(data.get("c") or 0),
        "open": float(data.get("o") or 0),
        "high": float(data.get("h") or 0),
        "low": float(data.get("l") or 0),
        "previous_close": float(data.get("pc") or 0),
    }


def _finnhub_quotes_batch(symbols: list[str]) -> dict[str, dict[str, float]]:
    """Bypassed live API calls to load instantly from warehouse database."""
    return {}


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
    st.markdown(
        """
        <div class="live-indicator-container">
            <span class="live-dot"></span>
            <span class="live-title">📡 Live Prices</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _help_line("Prices update every 30 seconds from Finnhub. Green = price went up · Red = price went down.")

    previous_prices = st.session_state.setdefault("live_previous_prices", {})
    history = st.session_state.setdefault("live_stream_history", [])
    # Phnom Penh is UTC+7 (no daylight saving time)
    timestamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=7)))
    rows: list[dict[str, object]] = []

    # Batch-fetch all quotes at once (cached for 25s)
    all_symbols = [str(r["symbol"]).upper() for _, r in latest_df.iterrows()]
    batch_quotes = _finnhub_quotes_batch(all_symbols)

    for _, row in latest_df.iterrows():
        symbol = str(row["symbol"]).upper()
        if symbol in batch_quotes:
            quote = batch_quotes[symbol]
            source = "Finnhub live"
        else:
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
    c2.metric("📈 Going Up", up_count)
    c3.metric("📉 Going Down", down_count)
    c4.metric("➡️ Unchanged", flat_count)

    # ── Live price chart ──
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
            color_discrete_sequence=["#21d4fd", "#c84cff", "#ff5bc8", "#7c8cff", "#22e6a8", "#f6b44b"],
        )
        st.plotly_chart(_dark_layout(fig, 310), width="stretch")

    # ── Live price table ──
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

    st.caption(f"⏱ Last updated at {timestamp.strftime('%H:%M:%S')} · Only this section refreshes automatically.")


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
                   p.indicator,
                   p.model_name,
                   p.predicted_year,
                   p.predicted_value,
                   p.confidence_low,
                   p.confidence_high,
                   p.run_at
            FROM   gold.fact_predictions p
            JOIN   gold.dim_commodity    d ON d.commodity_id = p.commodity_id
            ORDER  BY d.symbol, p.indicator, p.model_name, p.predicted_year
        """), conn)


def _create_forecast_chart(hist_df: pd.DataFrame, pred_df: pd.DataFrame, ticker: str, indicator: str, model_name: str) -> go.Figure:
    h_sub = hist_df[hist_df["symbol"] == ticker].sort_values("year")
    p_sub = pred_df[(pred_df["symbol"] == ticker) & (pred_df["indicator"] == indicator) & (pred_df["model_name"] == model_name)].sort_values("predicted_year")

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
            x=h_sub["year"],
            y=h_sub[indicator],
            mode="lines+markers",
            name=f"Historical {ind_label}",
            line=dict(color="#c84cff", width=3, shape="spline"),
            marker=dict(size=6),
            hovertemplate="Year %{x}<br>Price: $%{y:,.2f}<extra>History</extra>"
        ))

    # 2. Plot predictions
    if not p_sub.empty:
        # Transparent lower bound line
        fig.add_trace(go.Scatter(
            x=p_sub["predicted_year"],
            y=p_sub["confidence_low"],
            mode="lines",
            line=dict(width=0, shape="spline"),
            showlegend=False,
            hovertemplate="Lower Bound: $%{y:,.2f}<extra></extra>"
        ))
        
        # Upper bound line with fill to previous trace
        fig.add_trace(go.Scatter(
            x=p_sub["predicted_year"],
            y=p_sub["confidence_high"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(33, 212, 253, 0.12)",
            line=dict(width=0, shape="spline"),
            name="95% Confidence Interval",
            hovertemplate="Upper Bound: $%{y:,.2f}<extra></extra>"
        ))

        # Main forecast line
        fig.add_trace(go.Scatter(
            x=p_sub["predicted_year"],
            y=p_sub["predicted_value"],
            mode="lines+markers",
            name=f"Forecast ({model_name})",
            line=dict(color="#21d4fd", width=3, dash="dash", shape="spline"),
            marker=dict(size=7, symbol="diamond"),
            hovertemplate="Year %{x}<br>Forecast: $%{y:,.2f}<extra>Prediction</extra>"
        ))

    model_title = "Linear Trend" if model_name == "linear_trend" else "Holt's Smoothing"
    fig.update_layout(
        title=f"🔮 {ticker} {ind_label} Forecast using {model_title}",
        xaxis=dict(
            title="Year",
            tickmode="linear",
            dtick=1,
            gridcolor="#222b45",
        ),
        yaxis=dict(
            title="Price (USD)",
            tickformat="$",
            gridcolor="#222b45",
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

    all_years = sorted(df["year"].dropna().astype(int).unique().tolist())
    all_categories = sorted(df["commodity_category"].dropna().unique().tolist())

    with st.sidebar:
        st.markdown('<div class="side-title">📊 Stock Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-subtitle">Real-time prices from Finnhub API</div>', unsafe_allow_html=True)
        
        # Navigation toggle
        st.markdown('<div class="side-section">Navigation</div>', unsafe_allow_html=True)
        selected_view = st.radio(
            "Select View",
            ["📡 Live & History", "🔮 ML Forecasting"],
            label_visibility="collapsed",
            key="side_view_selector",
        )

        if selected_view == "📡 Live & History":
            st.markdown('<div class="side-section">Filter Stocks</div>', unsafe_allow_html=True)
            selected_categories = st.multiselect(
                "Filter by sector",
                all_categories,
                default=all_categories,
                help="Choose which market sectors to display",
            )
            year_range = st.slider(
                "Date range (year)",
                min_value=min(all_years),
                max_value=max(all_years),
                value=(min(all_years), max(all_years)),
                help="Filter the historical price charts below",
            )
            search_text = st.text_input("Search by ticker or company", placeholder="e.g. AAPL or Apple")
            if search_text:
                match_count = df[
                    df["symbol"].str.contains(search_text, case=False, na=False)
                    | df["commodity_name"].str.contains(search_text, case=False, na=False)
                ]["symbol"].nunique()
                st.caption(f"🔍 Found {match_count} matching stock{'s' if match_count != 1 else ''}.")

            st.markdown('<div class="side-section">How it works</div>', unsafe_allow_html=True)
            st.caption("The Live Prices section updates automatically every 30 seconds without refreshing the full page.")
        else:
            st.markdown('<div class="side-section">ML Predictions</div>', unsafe_allow_html=True)
            st.caption("Forecasting models are automatically fit, evaluated, and logged to MLflow via the containerized ML pipeline.")

    if selected_view == "📡 Live & History":
        filtered_df = df[
            df["commodity_category"].isin(selected_categories)
            & df["year"].between(year_range[0], year_range[1])
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
            """
            <div class="commodity-header">
                <h1>📈 Stock Market Dashboard</h1>
                <p>Track live stock prices, compare trends, and monitor your watchlist — powered by Finnhub.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        top1, top2, top3, top4 = st.columns([1.1, 1.1, 1, 1])
        with top1:
            _stat_card("Avg Close Price", f"${avg_close:,.2f}", "", "purple")
        with top2:
            _stat_card("Avg Daily Change", f"{avg_change:,.2f}", "%", "cyan")
        with top3:
            _mini_card("Stocks Going Up", f"+{rising}", True)
            _mini_card("Stocks Going Down", f"-{falling}", False)
        with top4:
            _mini_card("Stocks in Watchlist", str(len(latest_df)), True)
            _mini_card("High Volatility", str(high_vol), False)

        _live_market_stream(latest_df)

        bottom_left, bottom_right = st.columns([1.55, 1])
        with bottom_left:
            _block_title("📊 Price History")
            _help_line("How stock prices have changed over time. Smooth spline curves make trends easy to follow.")
            
            # Select tickers to display (prevents horizontal clutter)
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

                # Vibrant high-contrast colors matching dark mode
                CHART_THEMES = [
                    {"line": "#21d4fd", "fill": "rgba(33, 212, 253, 0.12)"},  # Cyan glow
                    {"line": "#c84cff", "fill": "rgba(200, 76, 255, 0.12)"},  # Purple glow
                    {"line": "#ff5bc8", "fill": "rgba(255, 91, 200, 0.12)"},  # Pink glow
                    {"line": "#7c8cff", "fill": "rgba(124, 140, 255, 0.12)"}, # Blue glow
                    {"line": "#22e6a8", "fill": "rgba(34, 230, 168, 0.12)"},  # Green glow
                    {"line": "#f6b44b", "fill": "rgba(246, 180, 75, 0.12)"},  # Orange glow
                    {"line": "#ff4b4b", "fill": "rgba(255, 75, 75, 0.12)"},   # Red glow
                    {"line": "#1cdb6d", "fill": "rgba(28, 219, 109, 0.12)"},  # Emerald glow
                ]
                
                for idx, sym in enumerate(selected_symbols_history):
                    sym_df = history_plot_df[history_plot_df["symbol"] == sym].sort_values("year")
                    if sym_df.empty:
                        continue
                    
                    theme = CHART_THEMES[idx % len(CHART_THEMES)]
                    color = theme["line"]
                    fill_color = theme["fill"]
                    
                    if is_single:
                        fig.add_trace(go.Scatter(
                            x=sym_df["year"],
                            y=sym_df[selected_metric],
                            mode="lines+markers",
                            name=f"{sym} ({metric_label})",
                            line=dict(color=color, width=3.5, shape="spline"),
                            marker=dict(size=8, symbol="circle", line=dict(color="#151a2e", width=1.5)),
                            fill="tozeroy",
                            fillcolor=fill_color,
                            hovertemplate=f"<b>{sym}</b><br>Year: %{{x}}<br>{metric_label}: $%{{y:,.2f}}<extra></extra>"
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=sym_df["year"],
                            y=sym_df[selected_metric],
                            mode="lines+markers",
                            name=sym,
                            line=dict(color=color, width=3, shape="spline"),
                            marker=dict(size=7, symbol="circle", line=dict(color="#151a2e", width=1)),
                            hovertemplate=f"<b>{sym}</b><br>Year: %{{x}}<br>{metric_label}: $%{{y:,.2f}}<extra></extra>"
                        ))
                
                fig.update_layout(
                    title=dict(
                        text=f"📈 Historical {metric_label} Trends",
                        font=dict(color="#f8fbff", size=16, weight="bold")
                    ),
                    xaxis=dict(
                        title=dict(text="Year", font=dict(color="#aeb9d8")),
                        tickmode="linear",
                        dtick=1,
                        gridcolor="#222b45",
                        tickfont=dict(color="#aeb9d8")
                    ),
                    yaxis=dict(
                        title=dict(text="Price (USD)", font=dict(color="#aeb9d8")),
                        tickformat="$",
                        gridcolor="#222b45",
                        tickfont=dict(color="#aeb9d8")
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.04,
                        xanchor="right",
                        x=1,
                        font=dict(color="#aeb9d8", size=11)
                    ),
                    margin=dict(l=15, r=15, t=65, b=15),
                    hovermode="x unified"
                )
                
                st.plotly_chart(_dark_layout(fig, 330, extra_right=0), width="stretch")
            else:
                st.info("Select at least one stock to plot the price history.")

        with bottom_right:
            _block_title("🏷️ Sector Breakdown")
            _help_line("How many stocks belong to each market sector.")
            sector_counts = latest_df.groupby("commodity_category", as_index=False).agg(count=("symbol", "count"))
            sector_counts = sector_counts.rename(columns={"commodity_category": "Sector"})
            pie = px.pie(
                sector_counts,
                names="Sector",
                values="count",
                hole=0.62,
                title="Stocks by Sector",
                color_discrete_sequence=["#21d4fd", "#c84cff", "#ff5bc8", "#376dff", "#22e6a8"],
            )
            pie.update_traces(textfont_color="#f8fbff")
            st.plotly_chart(_dark_layout(pie, 360, extra_right=30, extra_bottom=30), width="stretch")
    else:
        # ML forecasting page
        if pred_df.empty:
            st.warning("No prediction data found in database. Run the pipeline to generate forecasts.")
            return

        st.markdown(
            """
            <div class="commodity-header">
                <h1>🔮 Machine Learning Price Forecasting</h1>
                <p>Track advanced predictive analytics driven by Linear Trend and Holt's Exponential Smoothing models — logged via MLflow.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Forecasting controls
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            all_symbols = sorted(pred_df["symbol"].unique())
            selected_ticker = st.selectbox(
                "Select Stock Ticker",
                options=all_symbols,
                help="Select which stock symbol to analyze forecasts for"
            )
        with ctrl2:
            model_options = sorted(pred_df["model_name"].unique())
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
            indicator_options = sorted(pred_df["indicator"].unique())
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
        ticker_preds = pred_df[
            (pred_df["symbol"] == selected_ticker) & 
            (pred_df["indicator"] == selected_indicator) & 
            (pred_df["model_name"] == selected_model_raw)
        ]
        
        f_top1, f_top2 = st.columns(2)
        if not ticker_preds.empty:
            run_at_val = pd.to_datetime(ticker_preds["run_at"].iloc[0]).tz_convert("Asia/Phnom_Penh").strftime("%Y-%m-%d %H:%M:%S")
            with f_top1:
                _stat_card("Active Forecast Model", model_name_map.get(selected_model_raw, selected_model_raw), f" ({selected_ticker})", "purple")
            with f_top2:
                _stat_card("Last Pipeline Run Time", run_at_val, " (Phnom Penh)", "cyan")
        
        # Joined visual timeline chart
        st.markdown('<div class="block-title">📊 Forecast Projections Timeline</div>', unsafe_allow_html=True)
        st.markdown('<p class="help-line">History line (solid purple) joined seamlessly with forecasting line (dashed cyan) surrounded by a 95% confidence interval shaded glow.</p>', unsafe_allow_html=True)
        
        fig = _create_forecast_chart(df, pred_df, selected_ticker, selected_indicator, selected_model_raw)
        st.plotly_chart(_dark_layout(fig, 380, extra_right=30), width="stretch")

        # Table of forecast values
        st.markdown('<div class="block-title">📋 Detailed Forecast Values</div>', unsafe_allow_html=True)
        st.markdown('<p class="help-line">Exact values and confidence boundaries for the projected 3-year horizon. Spread % represents bound uncertainty.</p>', unsafe_allow_html=True)
        
        if not ticker_preds.empty:
            display_preds = ticker_preds[["predicted_year", "predicted_value", "confidence_low", "confidence_high"]].copy()
            # Convert values to float
            display_preds["predicted_value"] = display_preds["predicted_value"].astype(float)
            display_preds["confidence_low"] = display_preds["confidence_low"].astype(float)
            display_preds["confidence_high"] = display_preds["confidence_high"].astype(float)
            display_preds["Confidence Spread %"] = (
                (display_preds["confidence_high"] - display_preds["confidence_low"]) / display_preds["predicted_value"] * 100
            )
            
            # Format columns
            display_preds = display_preds.rename(columns={
                "predicted_year": "Projected Year",
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

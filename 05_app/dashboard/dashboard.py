"""
dashboard.py - Streamlit dashboard for Finnhub stock market analytics.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
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


@st.cache_data(ttl=60)
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
            color: #111827;
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


def _dark_layout(fig, height: int):
    fig.update_layout(
        height=height,
        paper_bgcolor="#151a2e",
        plot_bgcolor="#151a2e",
        font=dict(color="#dbe5ff"),
        margin=dict(l=10, r=10, t=44, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#aeb9d8")),
        title_font=dict(color="#f8fbff", size=15),
        xaxis=dict(gridcolor="#2c3450", zerolinecolor="#2c3450"),
        yaxis=dict(gridcolor="#2c3450", zerolinecolor="#2c3450"),
    )
    return fig


def _finnhub_quote(symbol: str) -> dict[str, float]:
    token = os.getenv("FINNHUB_API_KEY") or os.getenv("FINNHUB_TOKEN") or ""
    if not token:
        raise RuntimeError("FINNHUB_API_KEY is not set")

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


def _fallback_quote(row: pd.Series) -> dict[str, float]:
    return {
        "live_price": float(row.get("latest_price") or row.get("close_price") or 0),
        "open": float(row.get("open_price") or 0),
        "high": float(row.get("high_price") or 0),
        "low": float(row.get("low_price") or 0),
        "previous_close": float(row.get("close_price") or 0),
    }


@st.fragment(run_every="10s")
def _live_market_stream(latest_df: pd.DataFrame) -> None:
    _block_title("Live Market Stream")
    _help_line("Up/down is measured against previous close. The small tick shows movement since the last stream update.")
    st.markdown(
        """
        <div class="legend-row">
            <span class="legend-pill up">UP = live price above previous close</span>
            <span class="legend-pill down">DOWN = live price below previous close</span>
            <span class="legend-pill">Small Tick = change since last poll</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    previous_prices = st.session_state.setdefault("live_previous_prices", {})
    history = st.session_state.setdefault("live_stream_history", [])
    timestamp = dt.datetime.now()
    rows: list[dict[str, object]] = []

    for _, row in latest_df.iterrows():
        symbol = str(row["symbol"]).upper()
        try:
            quote = _finnhub_quote(symbol)
            source = "Finnhub live"
        except Exception:
            quote = _fallback_quote(row)
            source = "warehouse fallback"

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
            "Symbol": symbol,
            "Name": row["commodity_name"],
            "Live Price": live_price,
            "Previous Close": previous_close,
            "Status": direction.upper(),
            "Change": day_change,
            "Change %": day_change_pct,
            "Small Tick": tick_change,
            "Tick": tick_status.upper(),
            "High": quote["high"],
            "Low": quote["low"],
            "Data": source,
            "Updated": timestamp.strftime("%H:%M:%S"),
        })
        history.append({
            "time": timestamp,
            "symbol": symbol,
            "live_price": live_price,
            "live_move": day_change,
            "live_move_pct": day_change_pct,
            "small_tick_move": tick_change,
            "tick": direction,
        })

    st.session_state["live_stream_history"] = history[-240:]
    live_df = pd.DataFrame(rows)

    up_count = int((live_df["Status"] == "UP").sum())
    down_count = int((live_df["Status"] == "DOWN").sum())
    flat_count = int((live_df["Status"] == "FLAT").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Streaming Symbols", len(live_df))
    c2.metric("Tick Up", up_count)
    c3.metric("Tick Down", down_count)
    c4.metric("Flat", flat_count)

    chart_df = pd.DataFrame(st.session_state["live_stream_history"])
    if not chart_df.empty:
        fig = px.line(
            chart_df,
            x="time",
            y="live_price",
            color="symbol",
            markers=True,
            title="All Stocks: Live Price",
            labels={
                "time": "Time",
                "live_price": "Live Price",
                "symbol": "Ticker",
                "live_move": "Change",
                "live_move_pct": "Change %",
                "small_tick_move": "Poll Change",
            },
            hover_data={
                "live_price": ":,.2f",
                "live_move": ":+,.4f",
                "live_move_pct": ":+,.4f",
                "small_tick_move": ":+,.6f",
            },
            color_discrete_sequence=["#21d4fd", "#c84cff", "#ff5bc8", "#7c8cff", "#22e6a8", "#f6b44b"],
        )
        fig.update_yaxes(range=[200, 700], dtick=100)
        st.plotly_chart(_dark_layout(fig, 310), width="stretch")

    display_cols = [
        "Symbol", "Name", "Live Price", "Status", "Change", "Change %",
        "Small Tick", "Tick", "Previous Close", "High", "Low", "Updated", "Data",
    ]
    st.dataframe(
        live_df[display_cols].style
        .map(_colour_change, subset=["Change", "Change %", "Small Tick"])
        .map(_tick_colour, subset=["Status", "Tick"])
        .format({
            "Live Price": "{:,.2f}",
            "Previous Close": "{:,.2f}",
            "Change": "{:+,.4f}",
            "Change %": "{:+,.4f}%",
            "Small Tick": "{:+,.6f}",
            "High": "{:,.2f}",
            "Low": "{:,.2f}",
        }),
        width="stretch",
        height=360,
    )

    st.caption(f"Updated: {timestamp.strftime('%H:%M:%S')}. Only this live stream block updates.")


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

    if df.empty:
        st.warning("Warehouse is empty. Run `python main.py` to ingest Finnhub stock prices.")
        return

    all_years = sorted(df["year"].dropna().astype(int).unique().tolist())
    all_categories = sorted(df["commodity_category"].dropna().unique().tolist())

    with st.sidebar:
        st.markdown('<div class="side-title">Stock Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-subtitle">Live quotes from Finnhub</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-active">Live Market Stream</div>', unsafe_allow_html=True)
        st.markdown('<div class="side-section">Filters</div>', unsafe_allow_html=True)
        selected_categories = st.multiselect(
            "Sectors",
            all_categories,
            default=all_categories,
        )
        year_range = st.slider(
            "History window",
            min_value=min(all_years),
            max_value=max(all_years),
            value=(min(all_years), max(all_years)),
        )
        search_text = st.text_input("Ticker search", placeholder="AAPL")

        st.markdown('<div class="side-section">Status</div>', unsafe_allow_html=True)
        st.caption("The live market table updates automatically without refreshing the full page.")

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
            <h1>Stock Market Dashboard</h1>
            <p>Live Finnhub quote stream with tick-by-tick increase and decrease tracking.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top1, top2, top3, top4 = st.columns([1.1, 1.1, 1, 1])
    with top1:
        _stat_card("Average Close", f"{avg_close:,.2f}", " USD", "purple")
    with top2:
        _stat_card("Average Change", f"{avg_change:,.2f}", "%", "cyan")
    with top3:
        _mini_card("Rising Tickers", f"+{rising}", True)
        _mini_card("Falling Tickers", f"-{falling}", False)
    with top4:
        _mini_card("Watchlist", str(len(latest_df)), True)
        _mini_card("High Volatility", str(high_vol), False)

    _live_market_stream(latest_df)

    bottom_left, bottom_right = st.columns([1.55, 1])
    with bottom_left:
        _block_title("Stored Price Movement")
        line_df = filtered_df.melt(
            id_vars=["year", "symbol"],
            value_vars=["close_price", "latest_price"],
            var_name="metric",
            value_name="price",
        )
        fig = px.line(
            line_df,
            x="year",
            y="price",
            color="symbol",
            line_dash="metric",
            markers=True,
            title="Warehouse Close and Latest Price by Ticker",
            labels={"year": "Year", "price": "Price", "symbol": "Ticker"},
            color_discrete_sequence=["#21d4fd", "#c84cff", "#ff5bc8", "#7c8cff", "#22e6a8", "#f6b44b"],
        )
        st.plotly_chart(_dark_layout(fig, 330), width="stretch")

    with bottom_right:
        _block_title("Sector Mix")
        sector_counts = latest_df.groupby("commodity_category", as_index=False).agg(count=("symbol", "count"))
        pie = px.pie(
            sector_counts,
            names="commodity_category",
            values="count",
            hole=0.62,
            title="Tickers by Sector",
            color_discrete_sequence=["#21d4fd", "#c84cff", "#ff5bc8", "#376dff", "#22e6a8"],
        )
        pie.update_traces(textfont_color="#f8fbff")
        st.plotly_chart(_dark_layout(pie, 330), width="stretch")


if __name__ == "__main__":
    main()

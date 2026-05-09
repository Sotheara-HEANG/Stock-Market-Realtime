"""
dashboard.py — Streamlit dashboard for the economic/governance pipeline.

Sections:
  1. Overview map — latest governance composite score by country
  2. Country deep-dive — time-series for any indicator + country
  3. Forecasts — historical + 3-year predictions for chosen country/indicator
  4. Compare — overlay multiple countries on one chart
  5. Top N ranking — bar chart of top/bottom countries for an indicator + year

Usage:
    streamlit run 05_app/dashboard/dashboard.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")

DB_URL = (
    f"postgresql+psycopg2://{os.environ.get('DB_USER','kongsattha')}:"
    f"{os.environ.get('DB_PASSWORD','')}@"
    f"{os.environ.get('DB_HOST','localhost')}:"
    f"{os.environ.get('DB_PORT','5432')}/"
    f"{os.environ.get('DB_NAME','econ_pipeline')}"
)


@st.cache_resource
def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


@st.cache_data(ttl=300)
def query(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Econ & Governance Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌍 Economic & Governance Pipeline Dashboard")
st.caption("Data: World Bank WGI · IMF WEO · UNDP HDI · Polity5 · V-Dem")

# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.header("Controls")

all_indicators = query(
    "SELECT DISTINCT indicator FROM indicators ORDER BY indicator"
)["indicator"].tolist()

all_countries = query(
    "SELECT iso_code, name FROM countries ORDER BY name"
)
country_map = dict(zip(all_countries["name"], all_countries["iso_code"]))

selected_section = st.sidebar.radio(
    "Section",
    ["Overview Map", "Country Deep-Dive", "Forecasts", "Compare Countries", "Top N Ranking"],
)

# ── 1. Overview Map ───────────────────────────────────────────────────────────

if selected_section == "Overview Map":
    st.subheader("Latest Governance Composite Score by Country")

    map_indicator = st.selectbox("Indicator", all_indicators, index=all_indicators.index("governance_composite") if "governance_composite" in all_indicators else 0)

    df_map = query(
        """
        SELECT c.iso_code, c.name, i.year, i.value
        FROM   indicators i
        JOIN   countries  c ON c.id = i.country_id
        WHERE  i.indicator = :ind
          AND  i.year = (
              SELECT MAX(year) FROM indicators WHERE indicator = :ind
          )
        """,
        {"ind": map_indicator},
    )

    if df_map.empty:
        st.warning("No data available for this indicator.")
    else:
        fig = px.choropleth(
            df_map,
            locations="iso_code",
            color="value",
            hover_name="name",
            hover_data={"year": True, "value": ":.3f"},
            color_continuous_scale="RdYlGn",
            title=f"{map_indicator} — {df_map['year'].iloc[0]}",
        )
        fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df_map[["name", "iso_code", "year", "value"]]
            .sort_values("value", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )

# ── 2. Country Deep-Dive ──────────────────────────────────────────────────────

elif selected_section == "Country Deep-Dive":
    st.subheader("Country Time-Series")

    col1, col2 = st.columns(2)
    with col1:
        country_name = st.selectbox("Country", list(country_map.keys()))
    with col2:
        indicator = st.selectbox("Indicator", all_indicators)

    iso = country_map[country_name]

    df_ts = query(
        """
        SELECT i.year, i.value, i.source, i.unit
        FROM   indicators i
        JOIN   countries  c ON c.id = i.country_id
        WHERE  UPPER(c.iso_code) = UPPER(:iso)
          AND  i.indicator = :ind
        ORDER  BY i.year
        """,
        {"iso": iso, "ind": indicator},
    )

    if df_ts.empty:
        st.warning(f"No data for {country_name} — {indicator}")
    else:
        unit = df_ts["unit"].iloc[0] if "unit" in df_ts.columns else ""
        fig = px.line(
            df_ts, x="year", y="value",
            title=f"{country_name} — {indicator}",
            labels={"value": unit, "year": "Year"},
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_ts, use_container_width=True)

# ── 3. Forecasts ──────────────────────────────────────────────────────────────

elif selected_section == "Forecasts":
    st.subheader("Historical Data + 3-Year Forecast")

    col1, col2, col3 = st.columns(3)
    with col1:
        country_name = st.selectbox("Country", list(country_map.keys()))
    with col2:
        indicator = st.selectbox("Indicator", all_indicators)
    with col3:
        model_options = query(
            "SELECT DISTINCT model_name FROM predictions ORDER BY model_name"
        )["model_name"].tolist()
        model = st.selectbox("Model", model_options) if model_options else None

    iso = country_map[country_name]

    df_hist = query(
        """
        SELECT i.year, i.value
        FROM   indicators i
        JOIN   countries  c ON c.id = i.country_id
        WHERE  UPPER(c.iso_code) = UPPER(:iso)
          AND  i.indicator = :ind
        ORDER  BY i.year
        """,
        {"iso": iso, "ind": indicator},
    )

    df_pred = pd.DataFrame()
    if model:
        df_pred = query(
            """
            SELECT p.predicted_year AS year, p.predicted_value AS value,
                   p.confidence_low, p.confidence_high
            FROM   predictions p
            JOIN   countries   c ON c.id = p.country_id
            WHERE  UPPER(c.iso_code) = UPPER(:iso)
              AND  p.indicator = :ind
              AND  p.model_name = :model
            ORDER  BY p.predicted_year
            """,
            {"iso": iso, "ind": indicator, "model": model},
        )

    if df_hist.empty:
        st.warning(f"No historical data for {country_name} — {indicator}")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_hist["year"], y=df_hist["value"],
            name="Historical", mode="lines+markers",
            line=dict(color="#1f77b4"),
        ))

        if not df_pred.empty:
            fig.add_trace(go.Scatter(
                x=df_pred["year"], y=df_pred["value"],
                name=f"Forecast ({model})", mode="lines+markers",
                line=dict(color="#ff7f0e", dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([df_pred["year"], df_pred["year"][::-1]]),
                y=pd.concat([df_pred["confidence_high"], df_pred["confidence_low"][::-1]]),
                fill="toself",
                fillcolor="rgba(255,127,14,0.15)",
                line=dict(color="rgba(255,127,14,0)"),
                name="95% CI",
            ))

        fig.update_layout(
            title=f"{country_name} — {indicator}",
            xaxis_title="Year", yaxis_title="Value",
            legend=dict(orientation="h"),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── 4. Compare Countries ──────────────────────────────────────────────────────

elif selected_section == "Compare Countries":
    st.subheader("Compare Countries")

    indicator = st.selectbox("Indicator", all_indicators)
    selected_names = st.multiselect(
        "Countries", list(country_map.keys()), default=list(country_map.keys())[:5]
    )
    iso_list = [country_map[n] for n in selected_names]

    if iso_list:
        placeholders = ", ".join(f"'{c}'" for c in iso_list)
        df_cmp = query(
            f"""
            SELECT c.iso_code, c.name, i.year, i.value
            FROM   indicators i
            JOIN   countries  c ON c.id = i.country_id
            WHERE  UPPER(c.iso_code) IN ({placeholders})
              AND  i.indicator = :ind
            ORDER  BY c.iso_code, i.year
            """,
            {"ind": indicator},
        )

        if df_cmp.empty:
            st.warning("No data for selected countries / indicator.")
        else:
            fig = px.line(
                df_cmp, x="year", y="value", color="name",
                title=f"{indicator} — Country Comparison",
                labels={"value": indicator, "year": "Year", "name": "Country"},
                markers=True,
            )
            st.plotly_chart(fig, use_container_width=True)

# ── 5. Top N Ranking ──────────────────────────────────────────────────────────

elif selected_section == "Top N Ranking":
    st.subheader("Top N Countries Ranking")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        indicator = st.selectbox("Indicator", all_indicators)
    with col2:
        available_years = query(
            "SELECT DISTINCT year FROM indicators WHERE indicator = :ind ORDER BY year DESC",
            {"ind": indicator},
        )["year"].tolist()
        year = st.selectbox("Year", available_years) if available_years else 2022
    with col3:
        n = st.slider("Top N", 5, 30, 15)
    with col4:
        ascending = st.checkbox("Lowest first", value=False)

    order = "ASC" if ascending else "DESC"
    df_top = query(
        f"""
        SELECT c.iso_code, c.name, c.region, i.value
        FROM   indicators i
        JOIN   countries  c ON c.id = i.country_id
        WHERE  i.indicator = :ind AND i.year = :yr
        ORDER  BY i.value {order} NULLS LAST
        LIMIT  :n
        """,
        {"ind": indicator, "yr": int(year), "n": n},
    )

    if df_top.empty:
        st.warning("No data.")
    else:
        fig = px.bar(
            df_top, x="value", y="name", orientation="h",
            color="value", color_continuous_scale="RdYlGn",
            title=f"{'Bottom' if ascending else 'Top'} {n} — {indicator} ({year})",
            labels={"value": indicator, "name": "Country"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=max(400, n * 28))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_top.reset_index(drop=True), use_container_width=True)

"""
dashboard.py — Streamlit dashboard for the real-time finance pipeline.

Reads directly from the Gold warehouse layer (PostgreSQL).
Five tabs:
  1. Overview   — market-wide price change bar chart + live price table
  2. Asset      — deep-dive metrics + OHLC bar + sector context
  3. Sectors    — sector-level bar charts and comparison table
  4. Rankings   — top / bottom N assets by any indicator
  5. Forecasts  — MLflow-generated price predictions with confidence bands

Usage:
    streamlit run 05_app/dashboard/dashboard.py
    # or from the project root:
    python -m streamlit run 05_app/dashboard/dashboard.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

# ---------------------------------------------------------------------------
# DB helpers (cached so reconnection only happens when session resets)
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_engine():
    host     = os.environ["DB_HOST"]
    port     = os.environ.get("DB_PORT", "5432")
    dbname   = os.environ["DB_NAME"]
    user     = os.environ["DB_USER"]
    password = os.environ.get("DB_PASSWORD", "")
    url      = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=60)
def _load_prices() -> pd.DataFrame:
    return pd.read_sql("""
        SELECT d.company_name,
               d.sector,
               f.current_price_usd,
               f.open_price_usd,
               f.day_high_usd,
               f.day_low_usd,
               f.previous_close_usd,
               f.price_change_usd,
               f.price_change_pct,
               f.trading_volume,
               f.intraday_range_pct,
               f.price_momentum,
               f.sector_avg_price,
               f.sector_avg_change_pct
        FROM   gold.fact_prices f
        JOIN   gold.dim_asset   d ON d.asset_id = f.asset_id
        ORDER  BY f.price_change_pct DESC NULLS LAST
    """, _get_engine())


@st.cache_data(ttl=60)
def _load_predictions() -> pd.DataFrame:
    try:
        return pd.read_sql("""
            SELECT d.company_name,
                   p.indicator,
                   p.model_name,
                   p.predicted_year,
                   p.predicted_value,
                   p.confidence_low,
                   p.confidence_high
            FROM   gold.fact_predictions p
            JOIN   gold.dim_asset        d ON d.asset_id = p.asset_id
            ORDER  BY p.predicted_year
        """, _get_engine())
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helper: colour a cell red/green by sign
# ---------------------------------------------------------------------------

def _colour_pct(val):
    if pd.isna(val):
        return ""
    return "color: green; font-weight: bold" if val > 0 else "color: red; font-weight: bold"


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Finance Pipeline Dashboard",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Real-Time Finance Dashboard")
    st.caption("Source: RapidAPI · Warehouse: Bronze / Silver / Gold · ML: MLflow")

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        df = _load_prices()
    except Exception as exc:
        st.error(f"Cannot connect to the database: {exc}")
        st.info("Run `python main.py` first to populate the warehouse, then reload.")
        return

    if df.empty:
        st.warning("Warehouse is empty. Run `python main.py` to ingest data.")
        return

    pred_df = _load_predictions()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Overview", "🔍 Asset Detail", "🏢 Sectors", "🏆 Rankings", "🔮 Forecasts"]
    )

    # ── Tab 1: Overview ──────────────────────────────────────────────────────
    with tab1:
        st.subheader("Market Snapshot")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Assets Tracked",   len(df))
        c2.metric("Avg Price (USD)",   f"${df['current_price_usd'].mean():.2f}")
        gainers = int((df["price_change_pct"] > 0).sum())
        losers  = int((df["price_change_pct"] < 0).sum())
        c3.metric("Gainers 📈", gainers, delta=f"+{gainers}")
        c4.metric("Losers  📉", losers,  delta=f"-{losers}", delta_color="inverse")

        st.markdown("---")

        fig = px.bar(
            df.sort_values("price_change_pct"),
            x="price_change_pct",
            y="company_name",
            orientation="h",
            color="price_change_pct",
            color_continuous_scale=["#d62728", "#aec7e8", "#2ca02c"],
            color_continuous_midpoint=0,
            title="Price Change % — All Assets",
            labels={"price_change_pct": "Change %", "company_name": ""},
        )
        fig.update_layout(height=620, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Live Price Table")
        display_cols = {
            "company_name":      "Company",
            "sector":            "Sector",
            "current_price_usd": "Price ($)",
            "price_change_pct":  "Change %",
            "day_high_usd":      "High ($)",
            "day_low_usd":       "Low ($)",
            "trading_volume":    "Volume",
            "intraday_range_pct":"Range %",
            "price_momentum":    "Momentum",
        }
        tbl = df[list(display_cols.keys())].rename(columns=display_cols)
        st.dataframe(
            tbl.style
               .applymap(_colour_pct, subset=["Change %", "Range %"])
               .format({"Price ($)": "${:.2f}", "High ($)": "${:.2f}",
                        "Low ($)": "${:.2f}", "Change %": "{:.2f}%",
                        "Range %": "{:.2f}%", "Volume": "{:,.0f}"}),
            use_container_width=True,
            height=400,
        )

    # ── Tab 2: Asset Detail ──────────────────────────────────────────────────
    with tab2:
        st.subheader("Asset Deep-Dive")

        asset = st.selectbox("Select Asset", df["company_name"].tolist(), key="asset_sel")
        row   = df[df["company_name"] == asset].iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price",   f"${row['current_price_usd']:.2f}",
                  delta=f"{row['price_change_pct']:.2f}%")
        c2.metric("Day High",        f"${row['day_high_usd']:.2f}")
        c3.metric("Day Low",         f"${row['day_low_usd']:.2f}")
        c4.metric("Volume",          f"{row['trading_volume']:,.0f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Open",            f"${row['open_price_usd']:.2f}")
        c6.metric("Prev Close",      f"${row['previous_close_usd']:.2f}")
        c7.metric("Intraday Range",  f"{row['intraday_range_pct']:.2f}%")
        c8.metric("Momentum",        row["price_momentum"] if pd.notna(row["price_momentum"]) else "—")

        # OHLC price range chart
        labels = ["Open", "Low", "Current", "High"]
        values = [
            row["open_price_usd"], row["day_low_usd"],
            row["current_price_usd"], row["day_high_usd"],
        ]
        colours = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
        fig2 = go.Figure(go.Bar(x=labels, y=values, marker_color=colours))
        fig2.update_layout(
            title=f"{asset} — Today's Price Range",
            yaxis_title="Price (USD)",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        if pd.notna(row.get("sector")):
            st.info(
                f"**Sector:** {row['sector']}  |  "
                f"Sector avg price: **${row['sector_avg_price']:.2f}**  |  "
                f"Sector avg change: **{row['sector_avg_change_pct']:.2f}%**"
            )

    # ── Tab 3: Sectors ───────────────────────────────────────────────────────
    with tab3:
        st.subheader("Sector Analysis")

        sec_df = (
            df.groupby("sector")
            .agg(
                n_assets   = ("company_name",      "count"),
                avg_price  = ("current_price_usd", "mean"),
                avg_change = ("price_change_pct",  "mean"),
                avg_volume = ("trading_volume",     "mean"),
            )
            .reset_index()
            .sort_values("avg_change", ascending=False)
        )

        col_a, col_b = st.columns(2)

        with col_a:
            fig3 = px.bar(
                sec_df, x="sector", y="avg_change",
                color="avg_change",
                color_continuous_scale=["#d62728", "#aec7e8", "#2ca02c"],
                color_continuous_midpoint=0,
                title="Average Price Change % by Sector",
                labels={"avg_change": "Avg Change %", "sector": ""},
            )
            fig3.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            fig4 = px.bar(
                sec_df, x="sector", y="avg_price",
                color="sector",
                title="Average Price by Sector",
                labels={"avg_price": "Avg Price ($)", "sector": ""},
            )
            st.plotly_chart(fig4, use_container_width=True)

        st.dataframe(
            sec_df.rename(columns={
                "sector":     "Sector",
                "n_assets":   "# Assets",
                "avg_price":  "Avg Price ($)",
                "avg_change": "Avg Change %",
                "avg_volume": "Avg Volume",
            }).style
              .applymap(_colour_pct, subset=["Avg Change %"])
              .format({
                  "Avg Price ($)": "${:.2f}",
                  "Avg Change %":  "{:.2f}%",
                  "Avg Volume":    "{:,.0f}",
              }),
            use_container_width=True,
        )

        # Asset breakdown for selected sector
        st.markdown("---")
        selected_sector = st.selectbox("Drill into sector", sec_df["sector"].tolist())
        sector_assets   = df[df["sector"] == selected_sector].sort_values(
            "price_change_pct", ascending=False
        )
        fig5 = px.bar(
            sector_assets,
            x="company_name", y="current_price_usd",
            color="price_change_pct",
            color_continuous_scale=["#d62728", "#aec7e8", "#2ca02c"],
            color_continuous_midpoint=0,
            title=f"{selected_sector} — Prices",
            labels={"current_price_usd": "Price ($)", "company_name": "", "price_change_pct": "Change %"},
        )
        st.plotly_chart(fig5, use_container_width=True)

    # ── Tab 4: Rankings ──────────────────────────────────────────────────────
    with tab4:
        st.subheader("Asset Rankings")

        col1, col2, col3 = st.columns(3)
        rank_by = col1.selectbox(
            "Rank by",
            ["price_change_pct", "current_price_usd", "trading_volume", "intraday_range_pct"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        top_n   = col2.slider("Show top N", 3, 20, 10)
        order   = col3.radio("Order", ["Top (highest)", "Bottom (lowest)"])

        ascending = order.startswith("Bottom")
        ranked = (
            df[["company_name", "sector", rank_by, "current_price_usd", "price_change_pct"]]
            .dropna(subset=[rank_by])
            .sort_values(rank_by, ascending=ascending)
            .head(top_n)
        )

        fig6 = px.bar(
            ranked,
            x=rank_by, y="company_name",
            orientation="h",
            color=rank_by,
            color_continuous_scale=["#d62728", "#aec7e8", "#2ca02c"] if not ascending
                                   else ["#2ca02c", "#aec7e8", "#d62728"],
            color_continuous_midpoint=0 if "change" in rank_by else None,
            title=f"{'Top' if not ascending else 'Bottom'} {top_n} by {rank_by.replace('_', ' ').title()}",
            labels={rank_by: rank_by.replace("_", " ").title(), "company_name": ""},
        )
        fig6.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig6, use_container_width=True)
        st.dataframe(ranked.reset_index(drop=True), use_container_width=True)

    # ── Tab 5: Forecasts ─────────────────────────────────────────────────────
    with tab5:
        st.subheader("Price Forecasts (MLflow Models)")

        if pred_df.empty:
            st.info(
                "No predictions available yet.  \n"
                "Predictions appear after **≥3 pipeline runs** have accumulated "
                "historical data in `gold.fact_prices`.  \n"
                "Run `python main.py` repeatedly (or on a schedule) to build history."
            )
        else:
            col1, col2 = st.columns(2)
            sel_asset = col1.selectbox(
                "Asset", pred_df["company_name"].unique().tolist(), key="fc_asset"
            )
            sel_indicator = col2.selectbox(
                "Indicator", pred_df["indicator"].unique().tolist(), key="fc_ind"
            )

            filtered = pred_df[
                (pred_df["company_name"] == sel_asset) &
                (pred_df["indicator"]    == sel_indicator)
            ]

            if filtered.empty:
                st.warning("No predictions for this combination.")
            else:
                # Add current price as anchor point
                current_row = df[df["company_name"] == sel_asset]
                if not current_row.empty and sel_indicator == "current_price_usd":
                    anchor = pd.DataFrame([{
                        "company_name":    sel_asset,
                        "indicator":       sel_indicator,
                        "model_name":      "actual",
                        "predicted_year":  int(current_row["current_price_usd"].index[0]),
                        "predicted_value": float(current_row["current_price_usd"].values[0]),
                        "confidence_low":  float(current_row["current_price_usd"].values[0]),
                        "confidence_high": float(current_row["current_price_usd"].values[0]),
                    }])
                    filtered = pd.concat([anchor, filtered], ignore_index=True)

                fig7 = px.line(
                    filtered,
                    x="predicted_year",
                    y="predicted_value",
                    color="model_name",
                    markers=True,
                    title=f"{sel_asset} — {sel_indicator.replace('_', ' ').title()} Forecast",
                    labels={
                        "predicted_year":  "Year",
                        "predicted_value": sel_indicator.replace("_", " ").title(),
                        "model_name":      "Model",
                    },
                )

                # Confidence bands per model
                for model in filtered["model_name"].unique():
                    m = filtered[filtered["model_name"] == model]
                    fig7.add_trace(go.Scatter(
                        x=pd.concat([m["predicted_year"], m["predicted_year"][::-1]]),
                        y=pd.concat([m["confidence_high"], m["confidence_low"][::-1]]),
                        fill="toself",
                        fillcolor="rgba(99,110,250,0.1)" if "linear" in model else "rgba(239,85,59,0.1)",
                        line=dict(color="rgba(255,255,255,0)"),
                        showlegend=False,
                        name=f"{model} 95% CI",
                    ))

                fig7.update_layout(height=450)
                st.plotly_chart(fig7, use_container_width=True)

                st.dataframe(
                    filtered[["model_name", "predicted_year", "predicted_value",
                               "confidence_low", "confidence_high"]]
                    .rename(columns={
                        "model_name":      "Model",
                        "predicted_year":  "Year",
                        "predicted_value": "Forecast",
                        "confidence_low":  "Low (95%)",
                        "confidence_high": "High (95%)",
                    }),
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()

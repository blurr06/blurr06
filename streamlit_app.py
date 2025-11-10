"""Interactive demand forecasting dashboard for convenience store owners."""

from __future__ import annotations

from io import BytesIO
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_sample_data, preprocess_sales_data
from src.forecasting import (
    apply_scenario_adjustments,
    generate_forecast,
    prepare_time_series,
    summarize_forecast,
)

st.set_page_config(
    page_title="Convenience Store Demand Forecaster",
    page_icon="🛒",
    layout="wide",
)


def load_data_source(option: str, uploaded_file) -> pd.DataFrame:
    if option == "Sample dataset":
        return load_sample_data()

    if uploaded_file is None:
        st.warning("Upload a CSV file with the required columns to use your own data.")
        return load_sample_data()

    try:
        return preprocess_sales_data(pd.read_csv(uploaded_file))
    except Exception as exc:  # noqa: BLE001 - user provided data, show friendly message
        st.error(f"Unable to read the uploaded file: {exc}")
        return load_sample_data()


@st.cache_data(show_spinner=False)
def cached_prepare_time_series(df: pd.DataFrame, store: str, categories: tuple[str, ...], freq: str):
    return prepare_time_series(df, store, list(categories), freq)


@st.cache_data(show_spinner=False)
def cached_forecast(series: pd.Series, periods: int, freq: str):
    return generate_forecast(series, periods, freq)


def render_header():
    st.title("🛒 Demand Forecasting Cockpit")
    st.markdown(
        """
        Empower your category managers with scenario-based forecasts across stores, categories,
        and promotional plans. Upload your own data or explore the sample dataset to see how
        the tool reacts to price changes, promotions, and external factors.
        """
    )


sidebar = st.sidebar
sidebar.header("Configuration")

with sidebar:
    data_source = st.radio("Data source", ["Sample dataset", "Upload my own CSV"], index=0)
    uploaded = sidebar.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

    forecast_horizon = st.slider("Forecast horizon (weeks)", min_value=4, max_value=24, value=12, step=1)
    frequency = st.selectbox("Aggregation frequency", options=["W", "M"], format_func=lambda x: "Weekly" if x == "W" else "Monthly")

    price_change = st.slider("Price change %", min_value=-20, max_value=20, value=0, step=1)
    promo_lift = st.slider("Promotion lift %", min_value=0, max_value=50, value=10, step=5)
    external_shift = st.slider("External demand shift %", min_value=-20, max_value=20, value=0, step=1)
    elasticity = st.slider("Price elasticity", min_value=0.5, max_value=2.5, value=1.2, step=0.1)

    st.markdown("""---
    **Columns required for custom data**
    - `date`
    - `store`
    - `product_category`
    - `sales_units`
    - `avg_selling_price`
    - `promotion_flag`
    """)

render_header()

raw_data = load_data_source(data_source, uploaded)
stores = ["All Stores"] + sorted(raw_data["store"].unique())
selected_store = st.selectbox("Select store", stores)

categories = sorted(raw_data["product_category"].unique())
selected_categories = st.multiselect("Product categories", categories, default=categories)

if not selected_categories:
    st.warning("Please choose at least one product category to generate a forecast.")
    st.stop()

try:
    aggregated = cached_prepare_time_series(raw_data, selected_store, tuple(selected_categories), frequency)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

forecast_result = cached_forecast(aggregated["sales_units"], forecast_horizon, frequency)
scenario_forecast = apply_scenario_adjustments(
    forecast_result.forecast,
    price_change_pct=price_change,
    promo_lift_pct=promo_lift,
    external_shift_pct=external_shift,
    price_elasticity=elasticity,
)
summary_metrics = summarize_forecast(aggregated, scenario_forecast)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Trailing 4-period avg", f"{summary_metrics['trailing_actuals']:.0f} units")
col2.metric("Scenario forecast", f"{summary_metrics['forecast_units']:.0f} units")
col3.metric(
    "Change vs baseline",
    f"{summary_metrics['change_vs_baseline_pct']:.1f}%",
    delta=f"{summary_metrics['forecast_units'] - summary_metrics['baseline_units']:.0f} units",
)
col4.metric("Projected revenue", f"${summary_metrics['forecast_revenue']:,.0f}")

history_df = aggregated.reset_index()
forecast_plot_df = scenario_forecast.copy()
forecast_plot_df = forecast_plot_df.merge(
    forecast_result.forecast[["date", "baseline_forecast"]], on="date", how="left"
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history_df["date"],
        y=history_df["sales_units"],
        mode="lines+markers",
        name="Actual sales",
        line=dict(color="#1f77b4"),
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast_plot_df["date"],
        y=forecast_plot_df["baseline_forecast"],
        mode="lines",
        name="Baseline forecast",
        line=dict(color="#ff7f0e", dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast_plot_df["date"],
        y=forecast_plot_df["adjusted_forecast"],
        mode="lines+markers",
        name="Scenario forecast",
        line=dict(color="#2ca02c"),
    )
)
fig.add_trace(
    go.Scatter(
        x=list(forecast_plot_df["date"]) + list(forecast_plot_df["date"][::-1]),
        y=list(forecast_plot_df["adjusted_upper_ci"]) + list(forecast_plot_df["adjusted_lower_ci"][::-1]),
        fill="toself",
        fillcolor="rgba(44, 160, 44, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="Scenario 95% CI",
    )
)

fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Units",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=30, b=40),
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("View data tables"):
    st.subheader("Historical demand")
    st.dataframe(history_df.tail(52))

    st.subheader("Forecast details")
    display_cols = [
        "date",
        "baseline_forecast",
        "adjusted_forecast",
        "adjusted_lower_ci",
        "adjusted_upper_ci",
        "scenario_factor",
    ]
    st.dataframe(scenario_forecast[display_cols])


def _create_download(sc_df: pd.DataFrame) -> BytesIO:
    buffer = BytesIO()
    sc_df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


download_buffer = _create_download(scenario_forecast)
st.download_button(
    label="Download forecast as CSV",
    data=download_buffer,
    file_name="forecast_results.csv",
    mime="text/csv",
)

with st.expander("Model diagnostics"):
    st.text(forecast_result.model_summary)


st.caption(
    "Tip: Adjust price elasticity if you expect different sensitivity to pricing changes "
    "for seasonal items versus staples."
)

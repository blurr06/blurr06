"""Forecasting utilities for the demand dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ForecastResult:
    """Container for forecast results used by the app."""

    forecast: pd.DataFrame
    model_summary: str


FREQ_TO_SEASONAL_PERIODS = {
    "W": 52,
    "M": 12,
    "D": 7,
}


def prepare_time_series(
    df: pd.DataFrame,
    store: Optional[str],
    categories: list[str],
    freq: str = "W",
) -> pd.DataFrame:
    """Aggregate the raw data into a time series grouped by the provided filters."""

    if store and store != "All Stores":
        filtered = df[df["store"] == store]
    else:
        filtered = df.copy()

    if categories:
        filtered = filtered[filtered["product_category"].isin(categories)]

    if filtered.empty:
        raise ValueError("The selected filters did not return any rows.")

    ts = filtered.set_index("date").sort_index()
    aggregated = ts.resample(freq).agg({
        "sales_units": "sum",
        "revenue": "sum",
    })

    aggregated["avg_price"] = np.where(
        aggregated["sales_units"] > 0,
        aggregated["revenue"] / aggregated["sales_units"],
        np.nan,
    )
    aggregated["avg_price"].interpolate(method="linear", inplace=True, limit_direction="both")

    aggregated["sales_units"].replace(0, np.nan, inplace=True)
    aggregated["sales_units"].interpolate(method="linear", inplace=True, limit_direction="both")
    aggregated["sales_units"].fillna(method="bfill", inplace=True)
    aggregated["sales_units"].fillna(method="ffill", inplace=True)

    return aggregated


def generate_forecast(
    series: pd.Series,
    forecast_periods: int,
    freq: str = "W",
) -> ForecastResult:
    """Produce a baseline forecast using Holt-Winters exponential smoothing."""

    if forecast_periods <= 0:
        raise ValueError("forecast_periods must be a positive integer")

    series = series.asfreq(freq)
    seasonal_periods = FREQ_TO_SEASONAL_PERIODS.get(freq)
    use_seasonal = seasonal_periods and len(series.dropna()) >= seasonal_periods * 2

    if use_seasonal:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
    else:
        model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal=None,
            initialization_method="estimated",
        )

    fit = model.fit(optimized=True)
    forecast = fit.forecast(forecast_periods)

    residuals = getattr(fit, "resid", None)
    if residuals is not None and len(residuals) > 1:
        resid_std = float(np.std(residuals, ddof=1))
    else:
        resid_std = 0.0

    ci_buffer = 1.96 * resid_std if resid_std else 0.0
    forecast_df = pd.DataFrame(
        {
            "date": forecast.index,
            "baseline_forecast": forecast.values,
            "lower_ci": forecast.values - ci_buffer,
            "upper_ci": forecast.values + ci_buffer,
        }
    )
    forecast_df["lower_ci"] = forecast_df["lower_ci"].clip(lower=0)

    return ForecastResult(forecast=forecast_df, model_summary=str(fit.summary()))


def apply_scenario_adjustments(
    forecast_df: pd.DataFrame,
    price_change_pct: float,
    promo_lift_pct: float,
    external_shift_pct: float,
    price_elasticity: float = 1.2,
) -> pd.DataFrame:
    """Return a forecast adjusted for scenario planning inputs."""

    adj_factor = 1 + (-price_elasticity * price_change_pct / 100.0)
    adj_factor += promo_lift_pct / 100.0
    adj_factor += external_shift_pct / 100.0
    adj_factor = max(0, adj_factor)

    adjusted = forecast_df.copy()
    adjusted["adjusted_forecast"] = adjusted["baseline_forecast"] * adj_factor
    adjusted["adjusted_lower_ci"] = adjusted["lower_ci"] * adj_factor
    adjusted["adjusted_upper_ci"] = adjusted["upper_ci"] * adj_factor

    adjusted[["adjusted_lower_ci", "adjusted_forecast", "adjusted_upper_ci"]] = adjusted[
        ["adjusted_lower_ci", "adjusted_forecast", "adjusted_upper_ci"]
    ].clip(lower=0)

    adjusted["scenario_factor"] = adj_factor
    return adjusted


def summarize_forecast(
    historical: pd.DataFrame,
    scenario_forecast: pd.DataFrame,
) -> dict[str, float]:
    """Calculate high-level KPIs for the dashboard."""

    trailing_actuals = historical["sales_units"].tail(4).mean()
    latest_price = historical["avg_price"].tail(4).mean()

    forecast_units = float(scenario_forecast["adjusted_forecast"].sum())
    baseline_units = float(scenario_forecast["baseline_forecast"].sum())

    revenue = forecast_units * float(latest_price if not np.isnan(latest_price) else 0)

    change_vs_baseline = (
        (forecast_units - baseline_units) / baseline_units * 100 if baseline_units else 0
    )

    return {
        "trailing_actuals": float(trailing_actuals),
        "avg_price": float(latest_price),
        "forecast_units": forecast_units,
        "baseline_units": baseline_units,
        "forecast_revenue": revenue,
        "change_vs_baseline_pct": change_vs_baseline,
    }

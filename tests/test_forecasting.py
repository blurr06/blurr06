"""Tests for forecasting utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from src.forecasting import apply_scenario_adjustments, generate_forecast, prepare_time_series


def _create_mock_data() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=40, freq="W")
    data = pd.DataFrame(
        {
            "date": list(dates) * 2,
            "store": ["Downtown"] * 40 + ["Suburban"] * 40,
            "product_category": ["Snacks"] * 80,
            "sales_units": [200 + i for i in range(80)],
            "avg_selling_price": [2.5] * 80,
            "promotion_flag": [False] * 80,
            "revenue": [(200 + i) * 2.5 for i in range(80)],
        }
    )
    return data


def test_prepare_time_series_filters_correctly():
    df = _create_mock_data()
    ts = prepare_time_series(df, "Downtown", ["Snacks"], "W")
    assert not ts.empty
    assert ts.index.freqstr == "W-SUN"


def test_generate_forecast_produces_expected_periods():
    df = _create_mock_data()
    ts = prepare_time_series(df, "Downtown", ["Snacks"], "W")
    result = generate_forecast(ts["sales_units"], forecast_periods=6)
    assert len(result.forecast) == 6
    assert {"date", "baseline_forecast", "lower_ci", "upper_ci"}.issubset(result.forecast.columns)


def test_apply_scenario_adjustments_changes_forecast():
    df = _create_mock_data()
    ts = prepare_time_series(df, "Downtown", ["Snacks"], "W")
    result = generate_forecast(ts["sales_units"], forecast_periods=4)
    base = result.forecast["baseline_forecast"].sum()
    adjusted = apply_scenario_adjustments(result.forecast, price_change_pct=-10, promo_lift_pct=5, external_shift_pct=0)
    assert "adjusted_forecast" in adjusted
    assert adjusted["adjusted_forecast"].sum() > base

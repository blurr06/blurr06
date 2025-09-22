"""Data loading and preprocessing helpers for the demand forecasting dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_convenience_store_sales.csv"
EXPECTED_COLUMNS = {
    "date",
    "store",
    "product_category",
    "sales_units",
    "avg_selling_price",
    "promotion_flag",
}


def load_sample_data() -> pd.DataFrame:
    """Return the bundled sample dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "The packaged sample dataset could not be found. Expected at " f"{DATA_PATH}"
        )

    return preprocess_sales_data(pd.read_csv(DATA_PATH))


def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate column names and compute helper columns used by the app.

    Parameters
    ----------
    df:
        Raw sales data that matches the expected schema.

    Returns
    -------
    pandas.DataFrame
        A cleaned dataframe with datetime index and a revenue column.
    """

    missing = EXPECTED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    clean_df = df.copy()
    clean_df["date"] = pd.to_datetime(clean_df["date"])
    clean_df = clean_df.sort_values("date")
    clean_df["sales_units"] = clean_df["sales_units"].astype(float)
    clean_df["avg_selling_price"] = clean_df["avg_selling_price"].astype(float)
    clean_df["promotion_flag"] = clean_df["promotion_flag"].astype(bool)
    clean_df["revenue"] = clean_df["sales_units"] * clean_df["avg_selling_price"]

    return clean_df.reset_index(drop=True)


def filter_data(
    df: pd.DataFrame,
    stores: Iterable[str] | None = None,
    categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filter the dataset by store and product category."""

    filtered = df
    if stores:
        filtered = filtered[filtered["store"].isin(list(stores))]
    if categories:
        filtered = filtered[filtered["product_category"].isin(list(categories))]

    return filtered

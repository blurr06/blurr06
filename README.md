# Convenience Store Demand Forecasting Dashboard

An interactive Streamlit application that helps convenience store operators explore and forecast demand across stores and product categories. The dashboard combines Holt-Winters forecasting with scenario planning levers for price, promotions, and external events.


## Features

- 📊 **Interactive exploration** — filter by store and product categories and switch between weekly and monthly aggregation.
- 🔮 **Baseline forecasts** — Holt-Winters exponential smoothing generates a demand outlook with confidence intervals.
- 🧮 **Scenario planning** — tune price changes, promotional lift, and external factors with configurable price elasticity.
- 💾 **Data portability** — upload your own CSV data or start with the bundled sample dataset; export forecast results back to CSV.
- 🛠️ **Diagnostics** — review model summaries and inspect the underlying time series powering the forecast.

## Getting started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Streamlit app**

   ```bash
   streamlit run streamlit_app.py
   ```

3. **Open the dashboard** in your browser (Streamlit prints the local URL). Load the sample dataset or upload your own sales history.

## Data requirements

To use your own data, provide a CSV with the following columns:

| Column              | Description                                                  |
|---------------------|--------------------------------------------------------------|
| `date`              | Transaction date (daily, weekly, or monthly granularity).    |
| `store`             | Store identifier or name.                                   |
| `product_category`  | Product grouping to forecast.                               |
| `sales_units`       | Units sold during the period.                               |
| `avg_selling_price` | Average selling price for the period.                       |
| `promotion_flag`    | Boolean flag indicating if a promotion occurred.            |

The sample dataset lives at `data/sample_convenience_store_sales.csv` and contains two years of weekly data for three store archetypes.

## Running tests

The repository includes lightweight tests for the forecasting utilities:

```bash
pytest
```

## Project structure

```
.
├── data/
│   └── sample_convenience_store_sales.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   └── forecasting.py
├── streamlit_app.py
├── tests/
│   └── test_forecasting.py
├── requirements.txt
└── README.md
```


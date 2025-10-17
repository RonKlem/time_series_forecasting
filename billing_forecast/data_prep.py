import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


"""
data_prep.py
-----------
This file contains data preprocessing helpers used by the forecasting pipeline.

Goals and assumptions (high level, non-technical):
- Convert the raw billing records into one row per customer per month.
- Create simple "features" that help models predict future monthly volume:
  - Lag features: previous months' volumes
  - Rolling averages: smoothed recent activity
  - Time features: month and year to capture seasonality
  - Encoded categorical identifiers (customer, product area, product code)

We intentionally drop rows where lag features are missing (early months) to
keep the model training simple. In production you could impute or use a
different approach for initial timesteps.
"""


def preprocess(df, lags=[1, 2, 3], rolling_windows=[3, 6]):
    """
    Convert raw billing rows into aggregated customer-month features.

    Args:
        df (pd.DataFrame): Input data with columns: billing_month, customer_id,
            product_area, unique_product_code, volume, revenue.
        lags (list): which lag months to create (e.g. [1,2,3] creates vol_lag_1..3).
        rolling_windows (list): rolling window sizes for creating moving averages.

    Returns:
        agg (pd.DataFrame): aggregated and feature-enriched dataframe, one row
            per customer per billing_month. Contains label column `volume`.
        encoders (dict): label encoders used for customer/product encodings.

    Notes for non-technical readers:
        - A "lag" feature is simply the value of volume from a previous month.
          For example, vol_lag_1 is last month's volume. These help models learn
          how recent history relates to next month's bill.
        - A rolling average smooths short-term fluctuations, giving a sense of
          recent typical behavior (e.g., a 3-month rolling mean).
    """
    df = df.copy()
    # Ensure billing month is understood as a date/time
    df['billing_month'] = pd.to_datetime(df['billing_month'])

    # Aggregate to one row per customer-month: sum volumes and revenue for
    # that customer in that month. If your raw data already has one row per
    # customer-month you can skip this step.
    agg = df.groupby(['customer_id', 'billing_month']).agg({
        'volume': 'sum',
        'revenue': 'sum'
    }).reset_index()

    # Sort so that operations that look at history work correctly
    agg = agg.sort_values(['customer_id', 'billing_month'])

    # Create lag features: volume from previous months for the same customer.
    # We create columns like vol_lag_1 (1 month ago), vol_lag_2 (2 months ago),
    # etc. These are shifted per-customer.
    for lag in lags:
        agg[f'vol_lag_{lag}'] = agg.groupby('customer_id')['volume'].shift(lag)

    # Create rolling mean features from recent history. We shift by 1 so the
    # rolling value does not include the current month's volume (avoid leakage).
    for w in rolling_windows:
        agg[f'vol_roll_mean_{w}'] = (
            agg.groupby('customer_id')['volume']
            .shift(1)
            .rolling(w)
            .mean()
            .reset_index(0, drop=True)
        )

    # Simple time features: month and year. These can help the model learn
    # seasonality (e.g., higher volumes in December).
    agg['month'] = agg['billing_month'].dt.month
    agg['year'] = agg['billing_month'].dt.year

    # For categorical fields (product area and product code) the raw input may
    # contain multiple rows per customer-month. We choose the most frequent
    # value in the raw data for each customer-month.
    prod = df.groupby(['customer_id', 'billing_month']).agg({
        'product_area': lambda x: x.value_counts().index[0],
        'unique_product_code': lambda x: x.value_counts().index[0]
    }).reset_index()
    agg = agg.merge(prod, on=['customer_id', 'billing_month'], how='left')

    # Encode categorical variables as integers so models (and embeddings) can
    # work with them. LabelEncoder turns each unique value into a number.
    le_cust = LabelEncoder()
    agg['customer_id_enc'] = le_cust.fit_transform(agg['customer_id'])
    le_pa = LabelEncoder()
    agg['product_area_enc'] = le_pa.fit_transform(agg['product_area'].astype(str))
    le_pc = LabelEncoder()
    agg['product_code_enc'] = le_pc.fit_transform(agg['unique_product_code'].astype(str))

    # Drop rows that have missing values for lag/rolling features (these occur
    # at the start of each customer's history). We keep this simple approach
    # here; a more advanced pipeline could impute or use sequence models that
    # tolerate shorter histories.
    agg = agg.dropna().reset_index(drop=True)

    encoders = {'le_customer': le_cust, 'le_pa': le_pa, 'le_pc': le_pc}
    return agg, encoders


def time_train_test_split(df, date_col='billing_month', test_months=6):
    """
    Split the dataset into training and testing sets using the most recent
    `test_months` as the test period.

    This is a simple and common approach for time series forecasting: we train
    on older months and evaluate on the latest months to see how the model
    performs into the future.

    Args:
        df (pd.DataFrame): dataframe containing a datetime column named by
            `date_col` (default 'billing_month').
        test_months (int): how many months at the end to hold out as test.

    Returns:
        train (pd.DataFrame), test (pd.DataFrame)
    """
    max_date = df[date_col].max()
    split_date = max_date - pd.DateOffset(months=test_months)
    train = df[df[date_col] <= split_date].reset_index(drop=True)
    test = df[df[date_col] > split_date].reset_index(drop=True)
    return train, test

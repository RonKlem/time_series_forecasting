import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


def fit_and_save_scaler(df, numeric_features, scaler_path):
    """
    Fit a StandardScaler on the provided dataframe columns and save it to disk.

    Args:
        df (pd.DataFrame): dataframe containing the numeric columns to fit on.
        numeric_features (list): list of column names to use for scaler fitting.
        scaler_path (str): file path to save the fitted scaler (joblib file).

    Returns:
        scaler (StandardScaler): the fitted scaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(df[numeric_features].values)
    joblib.dump({'scaler': scaler, 'features': list(numeric_features)}, scaler_path)
    return scaler


def load_scaler(scaler_path):
    """
    Load a previously saved scaler and its feature order.

    Returns:
        dict with keys: 'scaler' (StandardScaler) and 'features' (list)
    """
    loaded = joblib.load(scaler_path)
    return loaded


def transform_df_with_scaler(df, scaler_path, inplace=False):
    """
    Transform numeric columns of a dataframe using a saved scaler.

    The scaler file must contain a dict {'scaler': scaler, 'features': [..]}.

    Args:
        df (pd.DataFrame): dataframe containing the numeric columns to transform.
        scaler_path (str): path to saved scaler (joblib file).
        inplace (bool): if True, replace columns in-place on the passed df;
                        otherwise a copy is returned.

    Returns:
        transformed_df (pd.DataFrame)
    """
    loaded = load_scaler(scaler_path)
    scaler = loaded['scaler']
    features = loaded['features']
    target = df if inplace else df.copy()
    # Ensure all features exist in dataframe
    missing = [f for f in features if f not in target.columns]
    if missing:
        raise KeyError(f"Missing features for scaler transform: {missing}")
    transformed = scaler.transform(target[features].values)
    # Overwrite the original columns with scaled values
    target.loc[:, features] = transformed
    return target


def inverse_transform_array(scaled_array, scaler_path, feature_name=None):
    """
    Inverse transform a scaled array (1D or 2D) back to original scale.

    If feature_name is provided and scaled_array is 1D, the function will
    inverse-transform only that feature by placing the values into a temporary
    array with the correct column order expected by the scaler.

    Args:
        scaled_array (np.ndarray): scaled values (n_samples,) or (n_samples, n_features)
        scaler_path (str): path to saved scaler
        feature_name (str|None): column name if scaled_array is a single feature

    Returns:
        inv (np.ndarray): values in the original scale
    """
    loaded = load_scaler(scaler_path)
    scaler = loaded['scaler']
    features = loaded['features']

    arr = np.array(scaled_array)
    # If 1D and feature_name provided, create a full shaped array
    if arr.ndim == 1 and feature_name is not None:
        if feature_name not in features:
            raise KeyError(f"feature_name {feature_name} not found in scaler features")
        idx = features.index(feature_name)
        full = np.zeros((arr.shape[0], len(features)))
        full[:, idx] = arr
        inv_full = scaler.inverse_transform(full)
        return inv_full[:, idx]

    # If 2D and matches scaler features, directly inverse transform
    if arr.ndim == 2 and arr.shape[1] == len(features):
        return scaler.inverse_transform(arr)

    # If arr is 2D but has single column and no feature_name, assume single-feature
    if arr.ndim == 2 and arr.shape[1] == 1 and feature_name is not None:
        return inverse_transform_array(arr.ravel(), scaler_path, feature_name)

    raise ValueError("Cannot inverse transform array: shape/feature mismatch. Provide feature_name for 1D arrays.")

import os
import numpy as np
import pandas as pd
from data_generator import generate_synthetic_data
from src.data_prep import preprocess, time_train_test_split
from src.models import BaselineModels, train_lightgbm, predict_lightgbm, train_xgboost, predict_xgboost, rmse, mape
from src.models import tune_ridge, tune_random_forest, tune_lightgbm_sklearn, tune_xgboost_sklearn

try:
    from src.pytorch_model import LSTMModel, train_torch_model, TimeSeriesDataset
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

import matplotlib.pyplot as plt
import seaborn as sns


"""
train.py
--------
This script ties together the data generation, preprocessing, model training,
hyperparameter tuning, evaluation, and simple plotting.

High-level flow (non-technical description):
1. Generate or load billing data (one or more rows per customer per month).
2. Preprocess the data: aggregate to customer-month, create lag features and
   encode categories.
3. Split the processed data into training and test (by time)
4. Train baseline models and perform hyperparameter tuning where applicable.
5. Train an optional PyTorch LSTM model and (optionally) run Optuna tuning.
6. Evaluate models using RMSE and MAPE and save simple plots for a few users.

The script is intentionally conservative: optional packages (LightGBM, XGBoost,
PyTorch, Optuna) are used only if they are installed; otherwise we skip those
steps and still produce baseline results.
"""


def prepare_features(df):
    """
    Convenience wrapper that runs preprocessing and returns features and labels.

    Returns:
        df_proc: the processed dataframe
        X: feature dataframe (used for model training)
        y: target array
        encs: encoders dictionary for mapping categories back if needed
    """
    df_proc, encs = preprocess(df)
    feature_cols = [c for c in df_proc.columns if c not in ['billing_month', 'customer_id', 'volume', 'revenue', 'product_area', 'unique_product_code']]
    X = df_proc[feature_cols].copy()
    y = df_proc['volume'].values
    return df_proc, X, y, encs


def main():
    # 1) Load or generate data
    print('Generating synthetic data')
    df = generate_synthetic_data(n_customers=80, n_months=36)

    # 2) Preprocess into features
    df_proc, X, y, encs = prepare_features(df)

    # 3) Split by time: hold out the last N months for testing
    train_df, test_df = time_train_test_split(df_proc, test_months=6)

    feature_cols = [c for c in df_proc.columns if c not in ['billing_month', 'customer_id', 'volume', 'revenue', 'product_area', 'unique_product_code']]
    X_train = train_df[feature_cols]
    y_train = train_df['volume'].values
    X_test = test_df[feature_cols]
    y_test = test_df['volume'].values

    # 4) Baseline models with hyperparameter tuning
    print('Tuning and training baseline models')
    # Tune Ridge (a regularized linear model)
    print(' - Tuning Ridge (time-aware grid search)')
    try:
        best_ridge, best_ridge_params = tune_ridge(X_train, y_train)
        preds_ridge = best_ridge.predict(X_test)
    except Exception as e:
        # If tuning fails, fall back to a default model
        print('Ridge tuning failed, falling back to default Ridge:', e)
        baselines = BaselineModels()
        baselines.lr.fit(X_train, y_train)
        preds_ridge = baselines.lr.predict(X_test)
        best_ridge_params = {}

    # Tune RandomForest with randomized search
    print(' - Tuning RandomForest (time-aware randomized search)')
    try:
        best_rf, best_rf_params = tune_random_forest(X_train, y_train)
        preds_rf = best_rf.predict(X_test)
    except Exception as e:
        print('RandomForest tuning failed, falling back to default:', e)
        baselines = BaselineModels()
        baselines.rf.fit(X_train, y_train)
        preds_rf = baselines.rf.predict(X_test)
        best_rf_params = {}

    # Collect baseline results
    results = {}
    predictions_by_model = {}
    results['ridge'] = {'rmse': rmse(y_test, preds_ridge), 'mape': mape(y_test, preds_ridge)}
    predictions_by_model['ridge'] = preds_ridge
    results['random_forest'] = {'rmse': rmse(y_test, preds_rf), 'mape': mape(y_test, preds_rf)}
    predictions_by_model['random_forest'] = preds_rf
    best_params_summary = {'ridge': best_ridge_params, 'random_forest': best_rf_params}

    # 5) Advanced models: LightGBM and XGBoost (only if installed)
    print('Tuning and training LightGBM (if installed)')
    try:
        best_lgbm, best_lgbm_params = tune_lightgbm_sklearn(X_train, y_train)
        if best_lgbm is not None:
            p = best_lgbm.predict(X_test)
            results['lightgbm'] = {'rmse': rmse(y_test, p), 'mape': mape(y_test, p)}
            predictions_by_model['lightgbm'] = p
            best_params_summary['lightgbm'] = best_lgbm_params
    except Exception as e:
        print('LightGBM tuning skipped or failed:', e)

    print('Tuning and training XGBoost (if installed)')
    try:
        best_xgb, best_xgb_params = tune_xgboost_sklearn(X_train, y_train)
        if best_xgb is not None:
            p = best_xgb.predict(X_test)
            results['xgboost'] = {'rmse': rmse(y_test, p), 'mape': mape(y_test, p)}
            predictions_by_model['xgboost'] = p
            best_params_summary['xgboost'] = best_xgb_params
    except Exception as e:
        print('XGBoost tuning skipped or failed:', e)

    # 6) Deep learning model (PyTorch) with optional Optuna tuning
    print('Training PyTorch LSTM (if torch installed)')
    # Prepare input matrix for the PyTorch model: numeric features followed
    # by three encoded categorical columns (customer, product area, product code)
    cat_cols = ['customer_id_enc', 'product_area_enc', 'product_code_enc']
    numeric_cols = [c for c in feature_cols if c not in cat_cols]
    X_all = df_proc[numeric_cols + cat_cols].values
    y_all = df_proc['volume'].values

    train_idx = train_df.index
    test_idx = test_df.index
    X_train_torch = X_all[train_idx]
    y_train_torch = y_all[train_idx]
    X_test_torch = X_all[test_idx]
    y_test_torch = y_all[test_idx]

    preds_torch = None
    if HAVE_TORCH:
        try:
            # create a small validation split from the training data
            from sklearn.model_selection import train_test_split
            X_tr, X_val, y_tr, y_val = train_test_split(X_train_torch, y_train_torch, test_size=0.2, random_state=0)
            train_ds = TimeSeriesDataset(X_tr, y_tr)
            val_ds = TimeSeriesDataset(X_val, y_val)

            # basic model configuration
            num_numeric = len(numeric_cols)
            num_customers = int(df_proc['customer_id_enc'].max() + 1)
            num_pa = int(df_proc['product_area_enc'].max() + 1)
            num_pc = int(df_proc['product_code_enc'].max() + 1)

            model = LSTMModel(num_numeric=num_numeric, cust_emb_size=8, num_customers=num_customers, pa_emb_size=4, num_pa=num_pa, pc_emb_size=6, num_pc=num_pc)
            model = train_torch_model(train_ds, val_ds, model, epochs=30, patience=5)

            # Predict on the test set
            test_ds = TimeSeriesDataset(X_test_torch, y_test_torch)
            from torch.utils.data import DataLoader
            import torch
            loader = DataLoader(test_ds, batch_size=128)
            preds_torch_list = []
            model.eval()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            with torch.no_grad():
                for Xb, yb in loader:
                    Xb = torch.tensor(Xb).to(device)
                    cust_idx = Xb[:, -3].long()
                    pa_idx = Xb[:, -2].long()
                    pc_idx = Xb[:, -1].long()
                    numeric = Xb[:, :-3]
                    p = model(numeric, cust_idx, pa_idx, pc_idx).cpu().numpy()
                    preds_torch_list.append(p)
            preds_torch = np.concatenate(preds_torch_list)
            results['pytorch_lstm'] = {'rmse': rmse(y_test_torch, preds_torch), 'mape': mape(y_test_torch, preds_torch)}
            predictions_by_model['pytorch_lstm'] = preds_torch

            # Optional: run a short Optuna search to show best hyperparameters
            try:
                from src.pytorch_model import optuna_tune_pytorch
                print('Running short Optuna tuning for PyTorch model (may take time)')
                X_tr_split, X_val_split, y_tr_split, y_val_split = train_test_split(X_train_torch, y_train_torch, test_size=0.2, random_state=0)
                study = optuna_tune_pytorch(X_tr_split, y_tr_split, X_val_split, y_val_split, num_numeric=len(numeric_cols), num_customers=num_customers, num_pa=num_pa, num_pc=num_pc, timeout=120)
                if study is not None:
                    best_params_summary['pytorch_lstm'] = study.best_params
            except Exception as e:
                print('Optuna tuning for PyTorch failed or Optuna not installed:', e)
        except Exception as e:
            print('PyTorch model skipped or failed:', e)

    # 7) Print and summarize results
    print('\nEvaluation results:')
    for k, v in results.items():
        print(f"{k}: RMSE={v['rmse']:.4f}, MAPE={v['mape']:.2f}%")

    print('\nBest hyperparameters found:')
    for model_name, params in best_params_summary.items():
        print(f"{model_name}: {params}")

    # 8) Plot a few sample customers to visualize predictions vs actuals
    print('Plotting sample customers (plots/)')
    os.makedirs('plots', exist_ok=True)
    sample_customers = list(df_proc['customer_id'].unique()[:3])
    for cust in sample_customers:
        sub = df_proc[df_proc['customer_id'] == cust].sort_values('billing_month')
        test_mask = sub['billing_month'] > (df_proc['billing_month'].max() - pd.DateOffset(months=6))
        test_idxs = sub[test_mask].index.tolist()
        plt.figure(figsize=(8, 4))
        plt.plot(sub['billing_month'], sub['volume'], label='actual')
        # Overlay predictions from each model for the test period
        for name, preds_arr in predictions_by_model.items():
            positions = [list(test_df.index).index(i) for i in test_idxs if i in list(test_df.index)]
            if len(positions) == 0:
                continue
            model_preds = preds_arr[positions]
            plt.scatter(sub[test_mask]['billing_month'], model_preds, label=name, alpha=0.8)
        plt.title(f'Customer {cust}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'plots/{cust}.png')

    print('Done. Plots saved to plots/')


if __name__ == '__main__':
    main()
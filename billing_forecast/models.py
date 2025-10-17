import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint, uniform as sp_uniform


"""
models.py
---------
This module contains simple model training utilities and hyperparameter tuning helpers
for classical machine learning models and boosting libraries.

The goal is to make the pipeline easy to understand for non-technical readers:
- We provide simple baseline models (Ridge and Random Forest).
- We provide tuning helpers that search for good hyperparameters while respecting
  the time-ordering of the data (so we don't accidentally "peek" into the future).
- We include safe wrappers for LightGBM and XGBoost; these are optional and will
  be skipped if the libraries are not installed.

All tuning methods use TimeSeriesSplit: this is like normal cross-validation but
it preserves time order by creating training/validation folds that don't mix
future data into the training set. This prevents data leakage for time series.
"""


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error (RMSE).

    Args:
        y_true: array-like of actual target values.
        y_pred: array-like of predicted target values.

    Returns:
        A float - the RMSE value. Lower is better. This metric measures the
        typical size of the prediction error in the same units as the target.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error (MAPE).

    This expresses the average prediction error as a percentage of the true
    values. Small values are better. We clip very small true values to avoid
    division-by-zero issues.

    Args:
        y_true: array-like of actual target values.
        y_pred: array-like of predicted target values.

    Returns:
        A float - the MAPE percentage (e.g., 12.5 means 12.5%).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-8, None))) * 100


class BaselineModels:
    """
    Simple container for baseline models.

    This class trains two simple models that are easy to interpret:
    - Ridge (a regularized linear regression)
    - Random Forest (an ensemble of decision trees)

    Usage:
        baselines = BaselineModels()
        preds = baselines.fit_predict(X_train, y_train, X_test)

    The `fit_predict` method fits both models on the training data and returns
    a dictionary of predictions for the test set.
    """

    def __init__(self):
        # Ridge provides a stable linear baseline and is less likely to overfit
        # than ordinary least squares when features are correlated.
        self.lr = Ridge()
        # RandomForest is a strong off-the-shelf model for tabular data.
        self.rf = RandomForestRegressor(n_estimators=50, random_state=0)

    def fit_predict(self, X_train, y_train, X_test):
        """
        Fit both baseline models and return predictions for X_test.

        Returns:
            dict: {'lr': preds_from_ridge, 'rf': preds_from_random_forest}
        """
        results = {}
        # Fit Ridge and predict
        self.lr.fit(X_train, y_train)
        preds_lr = self.lr.predict(X_test)
        results['lr'] = preds_lr

        # Fit Random Forest and predict
        self.rf.fit(X_train, y_train)
        preds_rf = self.rf.predict(X_test)
        results['rf'] = preds_rf
        return results


def tune_ridge(X, y, cv_splits=3, alphas=None):
    """
    Tune the Ridge (linear) model using a grid search over the regularization
    strength (alpha). We use TimeSeriesSplit for cross-validation so that each
    validation fold only uses past data for training.

    Args:
        X, y: features and target for tuning (typically training set only).
        cv_splits: number of time-based folds to use.
        alphas: list of alpha values to try (if None, a default list is used).

    Returns:
        best_estimator, best_params: the fitted estimator with best params and
        the parameter dictionary found by grid search.
    """
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    # TimeSeriesSplit ensures we don't train on future data when validating.
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    ridge = Ridge()
    # GridSearchCV tries each alpha and evaluates with time-aware folds.
    gs = GridSearchCV(ridge, param_grid={'alpha': alphas}, cv=tscv, scoring='neg_root_mean_squared_error')
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def tune_random_forest(X, y, cv_splits=3, n_iter=20, random_state=0):
    """
    Tune a RandomForest using randomized search with time-aware cross-validation.

    RandomizedSearchCV samples hyperparameters from distributions. This is
    usually faster than exhaustive grid search and often finds good values.

    Args:
        X, y: features and target for tuning (training set only).
        cv_splits: number of TimeSeriesSplit folds.
        n_iter: number of random samples to try.
        random_state: seed for reproducibility.

    Returns:
        best_estimator, best_params
    """
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    # Define search ranges for some important RandomForest hyperparameters.
    param_dist = {
        'n_estimators': sp_randint(50, 300),
        'max_depth': sp_randint(3, 30),
        'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.5]
    }
    rscv = RandomizedSearchCV(RandomForestRegressor(random_state=random_state), param_distributions=param_dist, n_iter=n_iter, cv=tscv, scoring='neg_root_mean_squared_error', random_state=random_state)
    rscv.fit(X, y)
    return rscv.best_estimator_, rscv.best_params_


def tune_lightgbm_sklearn(X, y, cv_splits=3, n_iter=20, random_state=0):
    """
    Tune a LightGBM model using the scikit-learn style LGBMRegressor and
    RandomizedSearchCV. If LightGBM is not installed, the function returns
    (None, None) and the caller may skip LightGBM.

    Returns:
        best_estimator, best_params or (None, None) if lightgbm not installed.
    """
    try:
        from lightgbm import LGBMRegressor
    except Exception:
        # LightGBM is optional; return None so calling code can skip it.
        return None, None
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    param_dist = {
        'n_estimators': sp_randint(50, 500),
        'num_leaves': sp_randint(16, 128),
        'learning_rate': sp_uniform(0.001, 0.2),
        'max_depth': sp_randint(3, 20)
    }
    rscv = RandomizedSearchCV(LGBMRegressor(random_state=random_state), param_distributions=param_dist, n_iter=n_iter, cv=tscv, scoring='neg_root_mean_squared_error', random_state=random_state)
    rscv.fit(X, y)
    return rscv.best_estimator_, rscv.best_params_


def tune_xgboost_sklearn(X, y, cv_splits=3, n_iter=20, random_state=0):
    """
    Tune an XGBoost model using scikit-learn's XGBRegressor and RandomizedSearchCV.
    If XGBoost is not installed, returns (None, None).
    """
    try:
        from xgboost import XGBRegressor
    except Exception:
        return None, None
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    param_dist = {
        'n_estimators': sp_randint(50, 500),
        'max_depth': sp_randint(3, 12),
        'learning_rate': sp_uniform(0.001, 0.2),
        'subsample': sp_uniform(0.5, 0.5)
    }
    rscv = RandomizedSearchCV(XGBRegressor(objective='reg:squarederror', random_state=random_state), param_distributions=param_dist, n_iter=n_iter, cv=tscv, scoring='neg_root_mean_squared_error', random_state=random_state)
    rscv.fit(X, y)
    return rscv.best_estimator_, rscv.best_params_


def train_lightgbm(X_train, y_train, X_valid=None, y_valid=None):
    """
    Train a LightGBM booster using the native LightGBM API. This function is
    a simple helper: it creates datasets and calls lightgbm.train. If the
    library is not installed, the function returns None.
    """
    try:
        import lightgbm as lgb
    except Exception:
        return None
    # Prepare LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    if X_valid is not None:
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        # Use early stopping to avoid overfitting when a validation set is present
        booster = lgb.train({'objective':'regression','metric':'rmse'}, train_data, 100, valid_sets=[valid_data], early_stopping_rounds=10, verbose_eval=False)
    else:
        booster = lgb.train({'objective':'regression','metric':'rmse'}, train_data, 100, verbose_eval=False)
    return booster


def predict_lightgbm(model, X):
    """
    Predict using a LightGBM booster object returned by train_lightgbm.
    Returns None if the model is None.
    """
    if model is None:
        return None
    return model.predict(X)


def train_xgboost(X_train, y_train, X_valid=None, y_valid=None):
    """
    Train an XGBoost model using the native xgboost.train API. Returns the
    trained booster or None if xgboost is not available.
    """
    try:
        import xgboost as xgb
    except Exception:
        return None
    dtrain = xgb.DMatrix(X_train, label=y_train)
    params = {'objective':'reg:squarederror','eval_metric':'rmse'}
    if X_valid is not None:
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        evallist = [(dtrain,'train'),(dvalid,'eval')]
        bst = xgb.train(params, dtrain, num_boost_round=100, evals=evallist, early_stopping_rounds=10, verbose_eval=False)
    else:
        bst = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
    return bst


def predict_xgboost(model, X):
    """
    Predict using an XGBoost booster object returned by train_xgboost.
    Returns None if model is None.
    """
    if model is None:
        return None
    import xgboost as xgb
    d = xgb.DMatrix(X)
    return model.predict(d)


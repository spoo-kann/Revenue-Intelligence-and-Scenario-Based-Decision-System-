"""
Model Training — Linear Regression, Random Forest, Gradient Boosting, SVR, Time Series
Enhancement: K-Fold Cross-Validation (k=5) added to all ML models
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Features used for ML models (exclude target and non-numeric / leakage columns)
FEATURE_COLS = [
    'Month', 'Quarter', 'Year', 'DayOfYear',
    'Month_Sin', 'Month_Cos',
    'Lag_1', 'Lag_2', 'Lag_3',
    'Rolling_Mean_3', 'Rolling_Std_3', 'Rolling_Mean_6',
    'Units_Sold', 'Price',
]

# K-Fold config — 5 splits, no shuffle (preserves time order)
CV_SPLITS = 5


def _get_features(df: pd.DataFrame) -> list[str]:
    available = [c for c in FEATURE_COLS if c in df.columns]
    return available


def _split(df: pd.DataFrame, test_ratio: float = 0.2):
    n = len(df)
    split_idx = max(int(n * (1 - test_ratio)), n - 6)  # at least 6 test points
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _eval(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def _cross_validate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Run 5-fold cross-validation and return mean ± std for RMSE, MAE, R².
    Uses shuffle=False to preserve temporal ordering across folds.
    If dataset is too small for 5 folds, falls back to 3.
    """
    n_splits = CV_SPLITS if len(X) >= CV_SPLITS * 4 else 3
    kf = KFold(n_splits=n_splits, shuffle=False)

    rmse_scores = -cross_val_score(model, X, y, cv=kf,
                                   scoring='neg_root_mean_squared_error')
    mae_scores  = -cross_val_score(model, X, y, cv=kf,
                                   scoring='neg_mean_absolute_error')
    r2_scores   =  cross_val_score(model, X, y, cv=kf,
                                   scoring='r2')
    return {
        'cv_rmse_mean': float(rmse_scores.mean()),
        'cv_rmse_std':  float(rmse_scores.std()),
        'cv_mae_mean':  float(mae_scores.mean()),
        'cv_mae_std':   float(mae_scores.std()),
        'cv_r2_mean':   float(r2_scores.mean()),
        'cv_r2_std':    float(r2_scores.std()),
        'cv_splits':    n_splits,
    }


def _exp_smoothing(train_y, test_len, alpha=0.3):
    """Simple exponential smoothing as a lightweight time-series baseline."""
    s = train_y.iloc[0]
    for v in train_y:
        s = alpha * v + (1 - alpha) * s
    return np.array([s] * test_len)


def train_all_models(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """
    Train multiple models on the feature-engineered DataFrame.
    Each ML model is evaluated with:
      - Hold-out test set metrics (RMSE, MAE, R²)
      - 5-fold cross-validation metrics (CV RMSE mean ± std)
    Returns (list_of_model_dicts, best_model_dict).
    Best model selected by CV RMSE mean (more reliable than single split).
    """
    feat_cols = _get_features(df)
    train_df, test_df = _split(df)

    X_all   = df[feat_cols].values          # full dataset for CV
    y_all   = df['Revenue'].values

    X_train = train_df[feat_cols].values
    y_train = train_df['Revenue'].values
    X_test  = test_df[feat_cols].values
    y_test  = test_df['Revenue'].values

    models_config = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression",  Ridge(alpha=1.0)),
        ("Random Forest",     RandomForestRegressor(n_estimators=150, max_depth=6,
                                                    min_samples_leaf=2, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                                         learning_rate=0.08, random_state=42)),
        ("SVR",               Pipeline([
                                  ('scaler', StandardScaler()),
                                  ('svr',    SVR(kernel='rbf', C=500, epsilon=0.1))
                              ])),
    ]

    results = []

    for name, model in models_config:
        # ── K-Fold CV on full dataset (before final fit) ──────────────────────
        cv_metrics = _cross_validate(model, X_all, y_all)

        # ── Final fit on train split, evaluate on hold-out test ──────────────
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        metrics = _eval(y_test, y_pred)

        pred_df = pd.DataFrame({
            'Actual':    y_test,
            'Predicted': y_pred
        }, index=test_df.index)

        results.append({
            'name':      name,
            'model':     model,
            # Hold-out metrics
            'rmse':      metrics['rmse'],
            'mae':       metrics['mae'],
            'r2':        metrics['r2'],
            # Cross-validation metrics
            **cv_metrics,
            'pred_df':   pred_df,
            'feat_cols': feat_cols,
            'X_train':   X_train,
            'y_train':   y_train,
        })

    # ── Time Series (Exponential Smoothing) — no CV (not sklearn compatible) ─
    ts_pred    = _exp_smoothing(train_df['Revenue'], len(test_df))
    ts_metrics = _eval(y_test, ts_pred)
    ts_pred_df = pd.DataFrame({
        'Actual':    y_test,
        'Predicted': ts_pred
    }, index=test_df.index)
    results.append({
        'name':         'Exp. Smoothing (TS)',
        'model':        None,
        'rmse':         ts_metrics['rmse'],
        'mae':          ts_metrics['mae'],
        'r2':           ts_metrics['r2'],
        # CV not applicable for exponential smoothing
        'cv_rmse_mean': ts_metrics['rmse'],
        'cv_rmse_std':  0.0,
        'cv_mae_mean':  ts_metrics['mae'],
        'cv_mae_std':   0.0,
        'cv_r2_mean':   ts_metrics['r2'],
        'cv_r2_std':    0.0,
        'cv_splits':    0,
        'pred_df':      ts_pred_df,
        'feat_cols':    feat_cols,
        'ts':           True,
        'last_smooth':  float(train_df['Revenue'].ewm(alpha=0.3, adjust=False).mean().iloc[-1]),
    })

    # Sort by CV RMSE mean (more reliable than single hold-out RMSE)
    results.sort(key=lambda x: x['cv_rmse_mean'])
    best = results[0]

    return results, best


def get_best_model(models: list[dict]) -> dict:
    return min(models, key=lambda m: m['cv_rmse_mean'])
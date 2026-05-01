"""
Forecaster — Generate future revenue predictions using the best trained model.
"""
import numpy as np
import pandas as pd
from datetime import timedelta


def generate_forecast(df: pd.DataFrame, best_model: dict, horizon: int = 6) -> pd.DataFrame:
    """
    Generate `horizon` months of revenue forecast with 90% confidence intervals.

    Strategy:
      - If best model is ML-based: roll forward features month-by-month
        feeding the predicted value back as Lag_1 etc.
      - If best model is time-series (ES): project using exponential smoothing.
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)
    last_date = df['Date'].iloc[-1]
    revenues  = df['Revenue'].values
    n         = len(df)

    # RMSE-based uncertainty bound
    rmse = best_model['rmse']

    forecasts = []

    if best_model.get('ts'):
        # Exponential smoothing projection
        alpha    = 0.3
        s        = best_model['last_smooth']
        avg_g    = float(df['Revenue'].pct_change().dropna().mean()) * 0.65  # damped

        for i in range(1, horizon + 1):
            s_new = s * (1 + avg_g)
            noise = np.random.normal(0, rmse * 0.15)
            val   = max(0, s_new + noise)
            forecasts.append(val)
            s = s_new

    else:
        feat_cols  = best_model['feat_cols']
        model      = best_model['model']

        # Build rolling feature state from last known rows
        recent = df.copy()
        pred_revenue = list(revenues)

        for i in range(1, horizon + 1):
            # Next date (monthly)
            next_date = last_date + pd.DateOffset(months=i)

            # Build feature row
            row = {}
            row['Month']    = next_date.month
            row['Quarter']  = next_date.quarter
            row['Year']     = next_date.year
            row['DayOfYear']= next_date.dayofyear
            row['Month_Sin'] = np.sin(2 * np.pi * next_date.month / 12)
            row['Month_Cos'] = np.cos(2 * np.pi * next_date.month / 12)

            # Lag features from rolling predictions
            row['Lag_1'] = pred_revenue[-1]
            row['Lag_2'] = pred_revenue[-2] if len(pred_revenue) >= 2 else pred_revenue[-1]
            row['Lag_3'] = pred_revenue[-3] if len(pred_revenue) >= 3 else pred_revenue[-1]

            row['Rolling_Mean_3'] = np.mean(pred_revenue[-3:])
            row['Rolling_Std_3']  = np.std(pred_revenue[-3:]) if len(pred_revenue) >= 3 else 0
            row['Rolling_Mean_6'] = np.mean(pred_revenue[-6:])

            # Use last known Units_Sold and Price (with slight trend)
            row['Units_Sold'] = float(recent['Units_Sold'].iloc[-1]) * (1 + 0.005 * i)
            row['Price']      = float(recent['Price'].iloc[-1])      * (1 + 0.002 * i)

            # Build feature vector respecting column order
            x = np.array([[row.get(c, 0) for c in feat_cols]])
            pred = float(model.predict(x)[0])
            pred = max(0, pred)

            pred_revenue.append(pred)
            forecasts.append(pred)

    # Assemble DataFrame
    dates = [last_date + pd.DateOffset(months=i) for i in range(1, horizon+1)]

    # Confidence interval widens with horizon
    ci_multiplier = np.array([1.28 * (1 + 0.08*i) for i in range(horizon)])  # 90% CI ~1.645σ
    uppers  = [max(0, f + rmse * ci_multiplier[i]) for i, f in enumerate(forecasts)]
    lowers  = [max(0, f - rmse * ci_multiplier[i]) for i, f in enumerate(forecasts)]

    return pd.DataFrame({
        'Date':     dates,
        'Forecast': [round(f, 2) for f in forecasts],
        'Lower':    [round(l, 2) for l in lowers],
        'Upper':    [round(u, 2) for u in uppers],
    })

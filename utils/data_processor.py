"""
Data Processor — Validation, Cleaning, Feature Engineering
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


# ── Column auto-detection ─────────────────────────────────────────────────────

def _detect_col(df: pd.DataFrame, patterns: list[str]) -> str | None:
    for col in df.columns:
        for pat in patterns:
            if pat.lower() in col.lower():
                return col
    return None


def _detect_columns(df: pd.DataFrame) -> dict:
    return {
        'date':    _detect_col(df, ['date','time','month','period','week','day']),
        'revenue': _detect_col(df, ['revenue','sales','income','amount','total','turnover']),
        'units':   _detect_col(df, ['unit','qty','quantity','sold','volume','count']),
        'price':   _detect_col(df, ['price','rate','cost','avg_price','unit_price']),
    }


# ── Sample data generator ─────────────────────────────────────────────────────

def generate_sample_data(n_months: int = 36) -> pd.DataFrame:
    """Generate synthetic retail revenue data for demo purposes."""
    random.seed(42)
    np.random.seed(42)
    rows = []
    base_price = 48.0
    base_units = 1600

    start = datetime(2022, 1, 1)
    for i in range(n_months):
        d = start + timedelta(days=30 * i)
        month = d.month

        # Seasonality: peak in Nov/Dec, dip in Jan/Feb
        seasonal = 1 + 0.35 * np.sin((month - 3) * np.pi / 6)
        trend    = 1 + i * 0.012
        noise    = 0.92 + np.random.random() * 0.16

        price = round(base_price + np.random.normal(0, 2.5), 2)
        units = int(base_units * seasonal * trend * noise)
        revenue = round(price * units, 2)

        rows.append({
            'Date': d.strftime('%Y-%m-%d'),
            'Revenue': revenue,
            'Units_Sold': units,
            'Price': price
        })
    return pd.DataFrame(rows)


# ── Validation & Cleaning ─────────────────────────────────────────────────────

def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    log = []
    df = df.copy()

    log.append({'type': 'info', 'msg': f'Loaded {len(df)} rows and {len(df.columns)} columns.'})

    # Detect columns
    cols = _detect_columns(df)

    # ── Date column ──────────────────────────────────────────────────────────
    date_col = cols['date']
    if date_col is None:
        # Try first column
        date_col = df.columns[0]
        log.append({'type': 'warning', 'msg': f'No date column detected. Using "{date_col}" as date.'})
    try:
        df['Date'] = pd.to_datetime(df[date_col], infer_datetime_format=True)
        log.append({'type': 'success', 'msg': f'Date column "{date_col}" parsed successfully.'})
    except Exception:
        log.append({'type': 'error', 'msg': f'Could not parse "{date_col}" as dates. Please check format.'})
        df['Date'] = pd.date_range(start='2022-01-01', periods=len(df), freq='MS')

    # ── Revenue column ───────────────────────────────────────────────────────
    rev_col = cols['revenue']
    if rev_col is None:
        # Use first numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        rev_col  = num_cols[0] if num_cols else None
        log.append({'type': 'warning', 'msg': f'No revenue column detected. Using "{rev_col}".'})
    if rev_col:
        df['Revenue'] = pd.to_numeric(df[rev_col], errors='coerce')
    else:
        log.append({'type': 'error', 'msg': 'No numeric revenue column found.'})
        df['Revenue'] = np.nan

    # ── Units column ─────────────────────────────────────────────────────────
    units_col = cols['units']
    if units_col:
        df['Units_Sold'] = pd.to_numeric(df[units_col], errors='coerce')
        log.append({'type': 'success', 'msg': f'Units column: "{units_col}".'})
    else:
        df['Units_Sold'] = np.nan
        log.append({'type': 'warning', 'msg': 'No units column found. Setting to NaN.'})

    # ── Price column ─────────────────────────────────────────────────────────
    price_col = cols['price']
    if price_col:
        df['Price'] = pd.to_numeric(df[price_col], errors='coerce')
        log.append({'type': 'success', 'msg': f'Price column: "{price_col}".'})
    else:
        # Derive price from revenue / units if possible
        if 'Units_Sold' in df.columns:
            df['Price'] = np.where(df['Units_Sold'] > 0, df['Revenue'] / df['Units_Sold'], np.nan)
            log.append({'type': 'info', 'msg': 'Price derived from Revenue / Units_Sold.'})
        else:
            df['Price'] = np.nan

    # Keep only derived columns
    clean = df[['Date', 'Revenue', 'Units_Sold', 'Price']].copy()

    # ── Missing values ───────────────────────────────────────────────────────
    before = len(clean)
    clean = clean.dropna(subset=['Date', 'Revenue'])
    removed = before - len(clean)
    if removed:
        log.append({'type': 'warning', 'msg': f'Removed {removed} rows with missing Date or Revenue.'})

    # Fill remaining NaNs with forward fill then median
    for col in ['Units_Sold', 'Price']:
        if clean[col].isna().sum() > 0:
            clean[col] = clean[col].ffill().fillna(clean[col].median())

    # ── Outlier capping (IQR 1.5) ────────────────────────────────────────────
    q1, q3 = clean['Revenue'].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    lower_bound = max(0, q1 - 3 * iqr)
    outliers = ((clean['Revenue'] < lower_bound) | (clean['Revenue'] > upper)).sum()
    if outliers:
        clean['Revenue'] = clean['Revenue'].clip(lower_bound, upper)
        log.append({'type': 'warning', 'msg': f'Capped {outliers} revenue outliers using 3×IQR rule.'})

    # ── Sort by date ─────────────────────────────────────────────────────────
    clean = clean.sort_values('Date').reset_index(drop=True)

    log.append({'type': 'success',
                'msg': f'Cleaning complete. {len(clean)} clean rows ready.'})
    return clean, log


# ── Feature Engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time features
    df['Year']        = df['Date'].dt.year
    df['Month']       = df['Date'].dt.month
    df['Quarter']     = df['Date'].dt.quarter
    df['DayOfYear']   = df['Date'].dt.dayofyear

    # Cyclic encoding for month (helps models learn seasonality)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    # Lag features
    for lag in [1, 2, 3]:
        df[f'Lag_{lag}'] = df['Revenue'].shift(lag)

    # Rolling statistics
    df['Rolling_Mean_3'] = df['Revenue'].shift(1).rolling(3, min_periods=1).mean()
    df['Rolling_Std_3']  = df['Revenue'].shift(1).rolling(3, min_periods=2).std().fillna(0)
    df['Rolling_Mean_6'] = df['Revenue'].shift(1).rolling(6, min_periods=1).mean()

    # Growth rate
    df['Growth_Rate'] = df['Revenue'].pct_change() * 100

    # Revenue / Price interaction
    if df['Price'].notna().any() and (df['Price'] > 0).any():
        df['Price_Revenue_Ratio'] = df['Revenue'] / (df['Price'] + 1e-9)

    # Fill NaN lags with forward-fill then 0
    df = df.bfill().fillna(0)

    return df

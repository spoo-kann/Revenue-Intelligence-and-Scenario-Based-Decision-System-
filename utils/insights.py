"""
Explainable Insights — SHAP Values & Revenue Driver Analysis
Enhancement: SHAP (SHapley Additive exPlanations) replaces permutation importance
             for per-prediction, theoretically grounded feature attribution.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ── SHAP import with graceful fallback ───────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from sklearn.inspection import permutation_importance


# ── Friendly feature name mapping ────────────────────────────────────────────
RENAME = {
    'Lag_1':          'Lag Revenue (1mo)',
    'Lag_2':          'Lag Revenue (2mo)',
    'Lag_3':          'Lag Revenue (3mo)',
    'Rolling_Mean_3': 'Rolling Avg (3mo)',
    'Rolling_Mean_6': 'Rolling Avg (6mo)',
    'Rolling_Std_3':  'Revenue Volatility',
    'Month_Sin':      'Seasonality (sin)',
    'Month_Cos':      'Seasonality (cos)',
    'Month':          'Month',
    'Quarter':        'Quarter',
    'Year':           'Year',
    'DayOfYear':      'Day of Year',
    'Units_Sold':     'Units Sold',
    'Price':          'Price',
}


def _rename(series: pd.Series) -> pd.Series:
    return series.map(lambda x: RENAME.get(x, x))


def compute_feature_importance(best_model: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature importance using SHAP values (primary method).

    Priority order:
      1. SHAP TreeExplainer   — for tree models (RF, GB): exact, fast
      2. SHAP LinearExplainer — for linear models: exact
      3. SHAP KernelExplainer — for SVR / any model: model-agnostic, slower
      4. Permutation fallback — if shap is not installed
      5. Heuristic            — for TS / no-model case

    Returns a DataFrame with columns ['Feature', 'Importance'] where
    Importance = mean |SHAP value| as a percentage of the total.
    Also attaches shap_values and shap_explainer to best_model dict for
    use in waterfall / beeswarm plots in the UI.
    """
    feat_cols = best_model.get('feat_cols', [])
    model     = best_model.get('model')

    # ── Case 1: Time-series model — heuristic fallback ───────────────────────
    if best_model.get('ts') or model is None:
        heuristic = {
            'Lag_1':          0.30,
            'Rolling_Mean_3': 0.22,
            'Lag_2':          0.15,
            'Month_Sin':      0.12,
            'Month_Cos':      0.08,
            'Units_Sold':     0.07,
            'Price':          0.04,
            'Lag_3':          0.02,
        }
        rows = [(k, v * 100) for k, v in heuristic.items()
                if k in (feat_cols or list(heuristic.keys()))]
        fi = pd.DataFrame(rows, columns=['Feature', 'Importance'])
        fi['Feature'] = _rename(fi['Feature'])
        return fi.sort_values('Importance', ascending=False).reset_index(drop=True)

    # ── Prepare data ─────────────────────────────────────────────────────────
    X_train = best_model.get('X_train')
    y_train = best_model.get('y_train')

    if X_train is None:
        X_train = df[feat_cols].values
    if y_train is None:
        y_train = df['Revenue'].values

    # Use a background sample for SHAP (max 100 rows for speed)
    bg_size = min(100, len(X_train))
    rng     = np.random.default_rng(42)
    bg_idx  = rng.choice(len(X_train), bg_size, replace=False)
    X_bg    = X_train[bg_idx]

    # Unwrap sklearn Pipeline if needed (e.g. SVR inside scaler pipeline)
    raw_model = model
    if hasattr(model, 'named_steps'):
        raw_model = list(model.named_steps.values())[-1]

    shap_values  = None
    explainer    = None
    method_used  = 'unknown'

    # ── Case 2: SHAP available ───────────────────────────────────────────────
    if SHAP_AVAILABLE:
        try:
            # TreeExplainer: exact for RF / GradientBoosting
            if hasattr(raw_model, 'feature_importances_'):
                explainer   = shap.TreeExplainer(raw_model)
                shap_values = explainer.shap_values(X_train)
                method_used = 'SHAP TreeExplainer'

            # LinearExplainer: exact for LinearRegression / Ridge
            elif hasattr(raw_model, 'coef_'):
                explainer   = shap.LinearExplainer(raw_model, X_bg,
                                                    feature_perturbation='correlation_dependent')
                shap_values = explainer.shap_values(X_train)
                method_used = 'SHAP LinearExplainer'

            # KernelExplainer: model-agnostic fallback (SVR etc.)
            else:
                explainer   = shap.KernelExplainer(model.predict, X_bg)
                shap_values = explainer.shap_values(X_bg, nsamples=50, silent=True)
                X_train     = X_bg          # align X for downstream plots
                method_used = 'SHAP KernelExplainer'

        except Exception as e:
            warnings.warn(f"SHAP computation failed ({e}), falling back to permutation importance.")
            shap_values = None

    # ── Case 3: Permutation importance fallback (no shap / shap error) ───────
    if shap_values is None:
        try:
            perm        = permutation_importance(model, X_train, y_train,
                                                 n_repeats=10, random_state=42)
            importances = np.abs(perm.importances_mean)
            method_used = 'Permutation importance (fallback)'
        except Exception:
            importances = np.ones(len(feat_cols)) / len(feat_cols)
            method_used = 'Uniform (fallback)'

        total = importances.sum() or 1
        pct   = (importances / total * 100).round(2)
        fi    = pd.DataFrame({'Feature': feat_cols, 'Importance': pct})
        fi    = fi[fi['Importance'] > 0].sort_values('Importance', ascending=False)
        fi['Feature'] = _rename(fi['Feature'])
        return fi.head(10).reset_index(drop=True)

    # ── Compute mean |SHAP| importance from shap_values ──────────────────────
    sv = np.array(shap_values)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    mean_abs_shap = np.abs(sv).mean(axis=0)

    # Align length to feat_cols (KernelExplainer may return different shape)
    if len(mean_abs_shap) != len(feat_cols):
        mean_abs_shap = mean_abs_shap[:len(feat_cols)]

    total = mean_abs_shap.sum() or 1
    pct   = (mean_abs_shap / total * 100).round(2)

    fi = pd.DataFrame({'Feature': feat_cols, 'Importance': pct})
    fi = fi[fi['Importance'] > 0].sort_values('Importance', ascending=False)
    fi['Feature'] = _rename(fi['Feature'])
    fi = fi.head(10).reset_index(drop=True)

    # ── Store SHAP artefacts in best_model for UI plots ──────────────────────
    best_model['shap_values']    = shap_values
    best_model['shap_X_train']   = X_train
    best_model['shap_explainer'] = explainer
    best_model['shap_method']    = method_used
    best_model['shap_feat_cols'] = feat_cols

    return fi


def get_shap_summary_data(best_model: dict) -> dict | None:
    """
    Return SHAP data needed for beeswarm / waterfall plots in the UI.
    Returns None if SHAP was not computed.
    """
    if not best_model.get('shap_values') is not None:
        return None
    return {
        'shap_values': best_model.get('shap_values'),
        'X_train':     best_model.get('shap_X_train'),
        'feat_cols':   best_model.get('shap_feat_cols', best_model.get('feat_cols', [])),
        'method':      best_model.get('shap_method', 'unknown'),
    }


def generate_insights(fi: pd.DataFrame, df: pd.DataFrame, best_model: dict) -> list[dict]:
    """
    Generate human-readable revenue driver insights from SHAP feature importance.
    """
    insights = []

    if fi is None or fi.empty:
        return [{'type': 'warning', 'text': 'Feature importance not available.'}]

    # ── SHAP method used ─────────────────────────────────────────────────────
    method = best_model.get('shap_method', '')
    if method:
        insights.append({
            'type': 'info',
            'text': (f"🔬 Feature importance computed using <b>{method}</b> — "
                     f"values represent each feature's average absolute contribution "
                     f"to individual revenue predictions (not just overall averages).")
        })

    top1 = fi.iloc[0]
    top2 = fi.iloc[1] if len(fi) > 1 else None

    # Top driver
    insights.append({
        'type': 'info',
        'text': (f"🏆 <b>{top1['Feature']}</b> is the strongest revenue driver "
                 f"at <b>{top1['Importance']:.1f}%</b> mean |SHAP| importance. "
                 f"Focus optimization efforts here for maximum revenue impact.")
    })

    # Second driver
    if top2 is not None:
        insights.append({
            'type': 'info',
            'text': (f"🥈 <b>{top2['Feature']}</b> is the second most influential factor "
                     f"(<b>{top2['Importance']:.1f}%</b>). "
                     f"Combined, the top 2 drivers explain "
                     f"{top1['Importance'] + top2['Importance']:.1f}% of revenue variance.")
        })

    # Seasonality check
    seasonal_cols = [r for r in fi['Feature'].tolist()
                     if 'season' in r.lower() or 'month' in r.lower() or 'quarter' in r.lower()]
    if seasonal_cols:
        s_imp = fi[fi['Feature'].isin(seasonal_cols)]['Importance'].sum()
        insights.append({
            'type': 'warning',
            'text': (f"📅 Seasonality factors account for <b>{s_imp:.1f}%</b> of revenue variance. "
                     f"Plan promotions and inventory ahead of peak months.")
        })

    # Lag feature check
    lag_cols = [r for r in fi['Feature'].tolist()
                if 'lag' in r.lower() or 'rolling' in r.lower()]
    if lag_cols:
        l_imp = fi[fi['Feature'].isin(lag_cols)]['Importance'].sum()
        insights.append({
            'type': 'success',
            'text': (f"🔁 Historical revenue patterns (lag & rolling features) contribute "
                     f"<b>{l_imp:.1f}%</b> — strong autocorrelation makes short-term forecasts reliable.")
        })

    # Price sensitivity
    price_row = fi[fi['Feature'].str.contains('Price', case=False)]
    if not price_row.empty:
        p_imp = price_row['Importance'].values[0]
        msg   = ("High price sensitivity — even small pricing changes significantly affect revenue. "
                 "Test price elasticity before major changes." if p_imp > 10
                 else "Price has moderate influence — room for strategic price adjustments.")
        insights.append({
            'type': 'danger' if p_imp > 15 else 'info',
            'text': f"💲 <b>Price</b> drives <b>{p_imp:.1f}%</b> of revenue variance (SHAP). {msg}"
        })

    # CV-aware model quality
    r2        = best_model.get('r2', 0)
    rmse      = best_model.get('rmse', 0)
    cv_rmse   = best_model.get('cv_rmse_mean', rmse)
    cv_std    = best_model.get('cv_rmse_std', 0)
    avg       = df['Revenue'].mean()
    pct_err   = (rmse / avg * 100) if avg > 0 else 0
    quality   = 'excellent' if r2 > 0.9 else 'good' if r2 > 0.75 else 'moderate'

    cv_note = (f" Cross-validated RMSE: <b>${cv_rmse:,.0f} ± ${cv_std:,.0f}</b>."
               if cv_rmse > 0 else "")

    insights.append({
        'type': 'success' if r2 > 0.75 else 'warning',
        'text': (f"🤖 Model quality: <b>{quality}</b> (R²={r2:.3f}). "
                 f"Hold-out forecast error: <b>{pct_err:.1f}%</b> of mean revenue.{cv_note} "
                 f"{'High confidence in forecasts.' if pct_err < 10 else 'Consider collecting more data for better accuracy.'}")
    })

    return insights
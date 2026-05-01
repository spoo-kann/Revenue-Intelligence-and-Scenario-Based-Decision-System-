import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from utils.data_processor import validate_and_clean, engineer_features, generate_sample_data
from utils.models import train_all_models, get_best_model
from utils.forecaster import generate_forecast
from utils.insights import compute_feature_importance, generate_insights
from utils.report import generate_report_csv, generate_report_pdf
from auth import (show_login_page, show_user_card, is_logged_in,
                  get_current_user, get_permissions, get_role, can_access_step,
                  show_access_denied, logout, show_activity_log, ROLE_PERMISSIONS)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RevIQ — Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design System CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* ── Base (overridden by dynamic theme injection) ─────── */
:root {
    --bg:#080b12; --surface:#0d1117; --card:#111827;
    --border:#1f2937; --border2:#374151; --text1:#f9fafb;
    --text2:#9ca3af; --text3:#4b5563; --accent:#6366f1;
    --accent2:#8b5cf6; --green:#10b981; --amber:#f59e0b;
    --red:#ef4444; --cyan:#06b6d4;
}

.stApp { background: var(--bg); }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1400px; }

h1,h2,h3,h4,h5,h6 { color: var(--text1) !important; font-family: 'Space Grotesk', sans-serif !important; }
p, li, span, div, label { color: var(--text2); }
.stMarkdown p { color: var(--text2); }

/* ── Sidebar ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div { padding: 0; }

/* ── Sidebar nav radio ────────────────────────────────── */
div[data-testid="stSidebar"] .stRadio > label { display: none; }
div[data-testid="stSidebar"] .stRadio > div { display: flex; flex-direction: column; gap: 1px; }
div[data-testid="stSidebar"] .stRadio > div > label > div:first-child { display: none !important; }
div[data-testid="stSidebar"] .stRadio input[type="radio"] { display: none !important; }
div[data-testid="stSidebar"] .stRadio > div > label {
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 9px 16px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    margin: 0 !important;
}
div[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(99,102,241,0.08) !important;
    color: #d1d5db !important;
}
div[data-testid="stSidebar"] .stRadio > div [data-testid="stMarkdownContainer"] p {
    font-size: 13px !important; color: inherit !important; margin: 0 !important;
}

/* ── Metrics ──────────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
    position: relative; overflow: hidden;
}
div[data-testid="stMetric"]::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--cyan));
}
div[data-testid="stMetric"] label {
    color: #6b7280 !important; font-size: 10px !important;
    font-weight: 600 !important; letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stMetricValue"] > div {
    color: var(--text1) !important; font-size: 26px !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricDelta"] > div { font-size: 12px !important; }

/* ── Buttons ──────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 8px !important; padding: 10px 20px !important;
    font-weight: 600 !important; font-size: 13px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    width: 100% !important; transition: all 0.2s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { opacity: 0.85 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Dataframes ───────────────────────────────────────── */
.stDataFrame { border-radius: 10px !important; overflow: hidden !important; }

/* ── File uploader ────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1.5px dashed var(--border2) !important;
    border-radius: 12px !important;
}

/* ── Progress ─────────────────────────────────────────── */
.stProgress > div > div {
    background: linear-gradient(90deg, #6366f1, #06b6d4) !important;
    border-radius: 4px !important;
}

/* ── Select / number inputs ───────────────────────────── */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text1) !important;
}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* ── Custom components ────────────────────────────────── */
.reviq-header {
    display: flex; align-items: center; gap: 14px;
    padding: 18px 0 14px; border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.reviq-logo {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #6366f1, #06b6d4);
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 18px; flex-shrink: 0;
}
.reviq-title { font-size: 21px; font-weight: 700; color: var(--text1); letter-spacing: -0.02em; }
.reviq-sub { font-size: 12px; color: var(--text3); margin-top: 2px; font-family: 'JetBrains Mono', monospace; }

.step-badge {
    display: inline-flex; align-items: center;
    background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.25);
    color: #818cf8; font-size: 10px; font-weight: 700;
    padding: 3px 10px; border-radius: 999px;
    letter-spacing: 0.07em; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace; margin-bottom: 6px;
}

.section-title { font-size: 18px; font-weight: 700; color: var(--text1); letter-spacing: -0.01em; margin-bottom: 4px; }
.section-sub   { font-size: 13px; color: var(--text3); margin-bottom: 20px; }

.insight-card {
    background: var(--card); border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 10px; padding: 14px 16px;
    margin-bottom: 10px; font-size: 13.5px; line-height: 1.7; color: var(--text2);
}
.insight-card.success { border-left-color: var(--green); background: rgba(16,185,129,0.04); }
.insight-card.warning { border-left-color: var(--amber); background: rgba(245,158,11,0.04); }
.insight-card.info    { border-left-color: var(--accent); background: rgba(99,102,241,0.04); }
.insight-card.danger  { border-left-color: var(--red);   background: rgba(239,68,68,0.04); }
.insight-card b { color: var(--text1); }

.model-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px 20px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 16px;
}
.model-card.best { border-color: var(--green); background: rgba(16,185,129,0.04); }
.model-name { font-size: 14px; font-weight: 600; color: var(--text1); flex: 1; }
.model-metric { text-align: center; min-width: 90px; }
.model-metric-label { font-size: 10px; color: var(--text3); font-family: 'JetBrains Mono',monospace; text-transform: uppercase; letter-spacing: 0.06em; }
.model-metric-value { font-size: 15px; font-weight: 600; color: var(--text1); margin-top: 2px; }
.best-badge {
    background: rgba(16,185,129,0.15); color: var(--green);
    border: 1px solid rgba(16,185,129,0.3); font-size: 10px; font-weight: 700;
    padding: 2px 8px; border-radius: 999px; letter-spacing: 0.06em;
    font-family: 'JetBrains Mono', monospace;
}

.nav-label {
    font-size: 9px; font-weight: 700; color: #374151;
    text-transform: uppercase; letter-spacing: 0.14em;
    padding: 16px 16px 6px;
    font-family: 'JetBrains Mono', monospace;
}
.sidebar-top {
    padding: 20px 16px 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 8px;
}

.empty-state {
    text-align: center; padding: 64px 20px;
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; margin: 20px 0;
}
.empty-icon  { font-size: 42px; margin-bottom: 14px; }
.empty-title { font-size: 16px; font-weight: 600; color: var(--text1); margin-bottom: 6px; }
.empty-sub   { font-size: 13px; color: var(--text3); }

.tag { display: inline-block; background: rgba(99,102,241,0.12); color: #818cf8;
    font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 4px;
    font-family: 'JetBrains Mono',monospace; letter-spacing: 0.05em; margin-right: 4px; }
.tag-green { background: rgba(16,185,129,0.12); color: var(--green); }
.tag-amber { background: rgba(245,158,11,0.12); color: var(--amber); }

.info-bar {
    padding: 10px 16px; background: var(--card);
    border: 1px solid var(--border); border-radius: 8px;
    font-size: 12.5px; color: #6b7280; margin-top: 4px;
}

/* ── Animated project title ───────────────────────────── */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes fadeSlideDown {
    0%   { opacity: 0; transform: translateY(-12px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0); }
    50%       { box-shadow: 0 0 24px 4px rgba(99,102,241,0.18); }
}
.project-title-banner {
    animation: fadeSlideDown 0.7s ease forwards, pulse-glow 3s ease-in-out 0.7s infinite;
    background: linear-gradient(135deg,
        rgba(99,102,241,0.15) 0%,
        rgba(139,92,246,0.12) 40%,
        rgba(6,182,212,0.10) 100%);
    background-size: 200% 200%;
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 18px 28px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.project-title-banner::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4, #6366f1);
    background-size: 200% auto;
    animation: gradientShift 3s linear infinite;
}
.project-title-text {
    font-size: 19px;
    font-weight: 700;
    letter-spacing: -0.02em;
    font-family: 'Space Grotesk', sans-serif;
    background: linear-gradient(90deg, #e0e7ff, #c4b5fd, #67e8f9, #e0e7ff);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
}
.project-title-sub {
    font-size: 11px;
    color: #4b5563;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _chart(title='', height=320):
    return dict(
        title=dict(text=title, font=dict(color='#f9fafb', size=14,
                   family='Space Grotesk'), x=0),
        paper_bgcolor='#111827', plot_bgcolor='#111827',
        font=dict(color='#9ca3af', family='Space Grotesk'),
        height=height,
        margin=dict(l=12, r=12, t=44 if title else 12, b=12),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')),
        xaxis=dict(gridcolor='#1f2937', zeroline=False, linecolor='#1f2937'),
        yaxis=dict(gridcolor='#1f2937', zeroline=False, linecolor='#1f2937'),
    )

def _insight(text, kind='info'):
    cls = {'success':'success','warning':'warning','info':'info','danger':'danger'}.get(kind,'info')
    st.markdown(f"<div class='insight-card {cls}'>{text}</div>", unsafe_allow_html=True)

def _empty(icon, title, sub):
    st.markdown(f"<div class='empty-state'><div class='empty-icon'>{icon}</div>"
                f"<div class='empty-title'>{title}</div>"
                f"<div class='empty-sub'>{sub}</div></div>", unsafe_allow_html=True)

def _model_card_html(m, is_best):
    cls   = 'best' if is_best else ''
    badge = "<span class='best-badge'>BEST</span>" if is_best else ""
    cv    = f"${m.get('cv_rmse_mean', m['rmse']):,.0f} ±{m.get('cv_rmse_std',0):,.0f}"
    return (f"<div class='model-card {cls}'>"
            f"<div class='model-name'>{m['name']} {badge}</div>"
            f"<div class='model-metric'><div class='model-metric-label'>RMSE</div>"
            f"<div class='model-metric-value'>${m['rmse']:,.0f}</div></div>"
            f"<div class='model-metric'><div class='model-metric-label'>CV RMSE</div>"
            f"<div class='model-metric-value'>{cv}</div></div>"
            f"<div class='model-metric'><div class='model-metric-label'>R²</div>"
            f"<div class='model-metric-value'>{m['r2']:.3f}</div></div>"
            f"</div>")

# ── Session state ─────────────────────────────────────────────────────────────
for k in ['raw_df','cleaned_df','featured_df','models','best_model',
          'forecast_df','feature_importance','validation_log','saved_scenarios']:
    if k not in st.session_state:
        st.session_state[k] = None

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

# ── Auth gate — stop here if not logged in ────────────────────────────────────
if not is_logged_in():
    show_login_page()
    st.stop()

# ── Dynamic theme injection ───────────────────────────────────────────────────
if st.session_state.dark_mode:
    theme_css = """
    :root {
        --bg:      #080b12;  --surface: #0d1117; --card:    #111827;
        --border:  #1f2937;  --border2: #374151; --text1:   #f9fafb;
        --text2:   #9ca3af;  --text3:   #4b5563; --accent:  #6366f1;
        --accent2: #8b5cf6;  --green:   #10b981; --amber:   #f59e0b;
        --red:     #ef4444;  --cyan:    #06b6d4;
    }
    [data-testid="stSidebar"] { background: #0d1117 !important; }
    """
else:
    theme_css = """
    :root {
        --bg:      #f8fafc;  --surface: #f1f5f9; --card:    #ffffff;
        --border:  #e2e8f0;  --border2: #cbd5e1; --text1:   #0f172a;
        --text2:   #475569;  --text3:   #94a3b8; --accent:  #6366f1;
        --accent2: #8b5cf6;  --green:   #10b981; --amber:   #f59e0b;
        --red:     #ef4444;  --cyan:    #0891b2;
    }
    .stApp { background: #f8fafc !important; }
    [data-testid="stSidebar"] { background: #f1f5f9 !important; border-right: 1px solid #e2e8f0 !important; }
    div[data-testid="stMetric"] { background: #ffffff !important; border-color: #e2e8f0 !important; }
    div[data-testid="stMetric"] label { color: #64748b !important; }
    div[data-testid="stMetricValue"] > div { color: #0f172a !important; }
    .stButton > button { background: linear-gradient(135deg,#6366f1,#8b5cf6) !important; color:#fff !important; }
    .insight-card { background: #ffffff !important; border-color: #e2e8f0 !important; color: #475569 !important; }
    .insight-card b { color: #0f172a !important; }
    .model-card { background: #ffffff !important; border-color: #e2e8f0 !important; }
    .model-name, .model-metric-value { color: #0f172a !important; }
    .model-metric-label { color: #94a3b8 !important; }
    .reviq-title { color: #0f172a !important; }
    .reviq-sub, .nav-label { color: #94a3b8 !important; }
    .info-bar { background: #ffffff !important; border-color: #e2e8f0 !important; color: #64748b !important; }
    .empty-state { background: #ffffff !important; border-color: #e2e8f0 !important; }
    .empty-title { color: #0f172a !important; }
    h1,h2,h3,h4 { color: #0f172a !important; }
    p, div, span, label { color: #475569; }
    .stMarkdown p { color: #475569; }
    div[data-testid="stSidebar"] .stRadio > div > label { color: #64748b !important; }
    div[data-testid="stSidebar"] .stRadio > div > label:hover { background: rgba(99,102,241,0.08) !important; color: #0f172a !important; }
    """
st.markdown(f"<style>{theme_css}</style>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── User card ───────────────────────────────────────────────────────────
    show_user_card()
    # ── Logo + toggle row ──────────────────────────────────────────────────
    col_logo, col_toggle = st.columns([3, 1])
    with col_logo:
        st.markdown("""
<div style='padding:18px 0 10px 16px;'>
  <div style='display:flex;align-items:center;gap:10px;'>
    <div style='width:34px;height:34px;background:linear-gradient(135deg,#6366f1,#06b6d4);
                border-radius:8px;display:flex;align-items:center;
                justify-content:center;font-size:16px;flex-shrink:0;'>📊</div>
    <div>
      <div style='font-size:16px;font-weight:700;color:var(--text1);letter-spacing:-0.02em;'>RevIQ</div>
      <div style='font-size:11px;color:var(--text3);margin-top:1px;'>Revenue Intelligence</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    with col_toggle:
        st.markdown("<div style='padding-top:22px;'>", unsafe_allow_html=True)
        toggle_label = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(toggle_label, key="theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Override button CSS just for toggle (small, no full-width)
    st.markdown("""
<style>
div[data-testid="stSidebar"] div[data-testid="column"]:last-child .stButton > button {
    width: auto !important; padding: 6px 10px !important;
    font-size: 16px !important; background: transparent !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    line-height: 1 !important; min-height: 0 !important;
}
</style>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:var(--border);margin:0 0 8px;'></div>",
                unsafe_allow_html=True)

    # ── Navigation ─────────────────────────────────────────────────────────
    st.markdown("<div class='nav-label'>Navigation</div>", unsafe_allow_html=True)

    def _nav(icon, name, done):
        return f"{icon}  {name}{'  ✓' if done else ''}"

    nav_items = [
        _nav("📂", "Upload & Clean",  st.session_state.cleaned_df is not None),
        _nav("📊", "EDA",             st.session_state.cleaned_df is not None),
        _nav("🤖", "Train Models",    st.session_state.models is not None),
        _nav("🔮", "Forecast",        st.session_state.forecast_df is not None),
        _nav("🎯", "Target Check",    st.session_state.forecast_df is not None),
        _nav("⚙️", "Scenarios",       st.session_state.forecast_df is not None),
        _nav("💡", "Insights",        st.session_state.feature_importance is not None),
        _nav("📥", "Report",          st.session_state.forecast_df is not None),
        _nav("📋", "Activity Log",    True),
    ]

    active_nav  = st.radio("nav", nav_items, index=0, label_visibility="collapsed")
    active_step = nav_items.index(active_nav)

    st.markdown("<div style='height:1px;background:var(--border);margin:10px 0 0;'></div>",
                unsafe_allow_html=True)

    # ── Mini sparkline ─────────────────────────────────────────────────────
    if st.session_state.cleaned_df is not None:
        cl  = st.session_state.cleaned_df
        rev = cl['Revenue'].values
        dates = cl['Date'].values

        st.markdown("<div class='nav-label' style='margin-top:10px;'>Revenue Trend</div>",
                    unsafe_allow_html=True)

        spark_bg   = '#0d1117' if st.session_state.dark_mode else '#f1f5f9'
        spark_line = '#6366f1'
        spark_fill = 'rgba(99,102,241,0.12)'
        spark_text = '#9ca3af' if st.session_state.dark_mode else '#64748b'

        fig_spark = go.Figure()
        fig_spark.add_trace(go.Scatter(
            x=list(range(len(rev))), y=rev,
            mode='lines',
            line=dict(color=spark_line, width=1.8),
            fill='tozeroy', fillcolor=spark_fill,
            hovertemplate='$%{y:,.0f}<extra></extra>'
        ))
        # Highlight last point
        fig_spark.add_trace(go.Scatter(
            x=[len(rev)-1], y=[rev[-1]],
            mode='markers',
            marker=dict(color='#10b981', size=6),
            hovertemplate=f'Latest: ${rev[-1]:,.0f}<extra></extra>'
        ))
        fig_spark.update_layout(
            height=90, margin=dict(l=8, r=8, t=4, b=4),
            paper_bgcolor=spark_bg, plot_bgcolor=spark_bg,
            showlegend=False,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(fig_spark, use_container_width=True, config={'displayModeBar': False})

        # Mini stats below sparkline
        pct_chg = ((rev[-1] - rev[0]) / rev[0] * 100) if rev[0] != 0 else 0
        arrow   = "↑" if pct_chg >= 0 else "↓"
        color   = "#10b981" if pct_chg >= 0 else "#ef4444"
        st.markdown(f"""
<div style='display:flex;justify-content:space-between;padding:0 8px 10px;font-size:11px;
            font-family:"JetBrains Mono",monospace;'>
  <span style='color:{spark_text};'>${rev.min()/1000:.1f}k – ${rev.max()/1000:.1f}k</span>
  <span style='color:{color};font-weight:700;'>{arrow} {abs(pct_chg):.1f}%</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:1px;background:var(--border);margin:0 0 0;'></div>",
                    unsafe_allow_html=True)

    # ── Data health indicator ───────────────────────────────────────────────
    if st.session_state.cleaned_df is not None and st.session_state.validation_log is not None:
        cl  = st.session_state.cleaned_df
        raw = st.session_state.raw_df

        st.markdown("<div class='nav-label' style='margin-top:10px;'>Data Health</div>",
                    unsafe_allow_html=True)

        # Compute score
        raw_rows     = len(raw) if raw is not None else len(cl)
        clean_rows   = len(cl)
        null_rev     = cl['Revenue'].isna().sum()
        null_units   = cl['Units_Sold'].isna().sum()
        warnings_log = sum(1 for e in st.session_state.validation_log if e['type'] == 'warning')
        errors_log   = sum(1 for e in st.session_state.validation_log if e['type'] == 'error')

        completeness  = min(100, (clean_rows / max(raw_rows, 1)) * 100)
        null_penalty  = (null_rev + null_units) * 5
        warn_penalty  = warnings_log * 4
        err_penalty   = errors_log * 15
        score = max(0, min(100, round(completeness - null_penalty - warn_penalty - err_penalty)))

        if score >= 85:
            h_color, h_label, h_bg = '#10b981', 'Excellent', 'rgba(16,185,129,0.1)'
        elif score >= 65:
            h_color, h_label, h_bg = '#f59e0b', 'Good',      'rgba(245,158,11,0.1)'
        else:
            h_color, h_label, h_bg = '#ef4444', 'Needs Work', 'rgba(239,68,68,0.1)'

        bar_pct = score
        st.markdown(f"""
<div style='margin:0 8px 10px;padding:12px 14px;background:{h_bg};
            border:1px solid {h_color}33;border-radius:10px;'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
    <span style='font-size:11px;font-weight:700;color:{h_color};
                 font-family:"JetBrains Mono",monospace;text-transform:uppercase;
                 letter-spacing:0.07em;'>{h_label}</span>
    <span style='font-size:18px;font-weight:700;color:{h_color};'>{score}</span>
  </div>
  <div style='height:5px;background:var(--border);border-radius:999px;overflow:hidden;'>
    <div style='height:100%;width:{bar_pct}%;background:{h_color};
                border-radius:999px;transition:width 0.4s;'></div>
  </div>
  <div style='margin-top:10px;display:grid;grid-template-columns:1fr 1fr;gap:6px;
              font-size:10.5px;font-family:"JetBrains Mono",monospace;'>
    <div style='color:var(--text3);'>Rows kept</div>
    <div style='color:var(--text1);text-align:right;'>{clean_rows}/{raw_rows}</div>
    <div style='color:var(--text3);'>Warnings</div>
    <div style='color:{"#f59e0b" if warnings_log else "var(--text1)"};text-align:right;'>{warnings_log}</div>
    <div style='color:var(--text3);'>Errors</div>
    <div style='color:{"#ef4444" if errors_log else "var(--text1)"};text-align:right;'>{errors_log}</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height:1px;background:var(--border);margin:0;'></div>",
                    unsafe_allow_html=True)

    # ── CSV format hint ─────────────────────────────────────────────────────
    st.markdown("<div class='nav-label' style='margin-top:10px;'>CSV Format</div>",
                unsafe_allow_html=True)
    st.markdown("""
<div style='padding:4px 16px 14px;font-size:11px;color:var(--text3);
            line-height:2.2;font-family:"JetBrains Mono",monospace;'>
  Date &nbsp;&nbsp;&nbsp;&nbsp;→ YYYY-MM-DD<br>
  Revenue &nbsp;→ numeric<br>
  Units &nbsp;&nbsp;&nbsp;→ quantity<br>
  Price &nbsp;&nbsp;&nbsp;→ unit price
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:1px;background:var(--border);margin:0 0 10px;'></div>",
                unsafe_allow_html=True)

    # ── Role permissions summary ────────────────────────────────────────────
    perms = get_permissions()
    role  = get_role()
    accessible = len(perms["steps"])
    st.markdown(f"""
<div style='padding:8px 16px;font-size:11px;color:var(--text3);
            font-family:"JetBrains Mono",monospace;'>
  <span style='color:{ROLE_PERMISSIONS[role]["color"]};font-weight:700;'>
    {role.upper()}
  </span>
  &nbsp;·&nbsp; {accessible}/9 steps accessible
</div>
""", unsafe_allow_html=True)

    col_r, col_l = st.columns(2)
    with col_r:
        if perms["can_reset"]:
            if st.button("↺ Reset"):
                for k in ['raw_df','cleaned_df','featured_df','models','best_model',
                          'forecast_df','feature_importance','validation_log','saved_scenarios']:
                    st.session_state[k] = None
                st.rerun()
    with col_l:
        if st.button("⎋ Logout"):
            logout()
            st.rerun()

# ── Page header ───────────────────────────────────────────────────────────────
_headers = [
    ("Upload & Clean Data",        "Ingest, validate and prepare your revenue dataset"),
    ("Exploratory Data Analysis",  "Understand patterns, trends and seasonality in your data"),
    ("ML Model Training",          "Train 6 forecasting models with 5-fold cross-validation"),
    ("Revenue Forecasting",        "Forward-looking predictions with 90% confidence intervals"),
    ("Target Feasibility Check",   "Evaluate whether your revenue targets are achievable"),
    ("Scenario Simulation",        "Model what-if scenarios using price, volume & marketing levers"),
    ("Explainable AI Insights",    "SHAP-based feature attribution and revenue driver analysis"),
    ("Report & Export",            "Download full analysis as PDF or CSV"),
    ("Activity Log",               "All user login, logout and registration events"),
]
_t, _s = _headers[active_step]

# ── Animated project title banner ────────────────────────────────────────────
st.markdown("""
<div class='project-title-banner'>
  <div>
    <div class='project-title-text'>
      Revenue Intelligence &amp; Scenario Based Decision System
    </div>
    <div class='project-title-sub'>
      ML &nbsp;·&nbsp; Forecasting &nbsp;·&nbsp; XAI &nbsp;·&nbsp; Scenario Simulation &nbsp;·&nbsp; Real-time Analytics
    </div>
  </div>
  <div style='font-size:28px;opacity:0.5;animation:fadeSlideDown 1s ease forwards;'>📈</div>
</div>
""", unsafe_allow_html=True)

# ── Step header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='reviq-header'>
  <div class='reviq-logo'>📊</div>
  <div>
    <div class='reviq-title'>{_t}</div>
    <div class='reviq-sub'>{_s}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 1 — Upload & Clean
# ════════════════════════════════════════════════════════════════════════════════
if active_step == 0:
    if not can_access_step(0):
        show_access_denied("Upload & Clean")
    else:
        col_up, col_s = st.columns([3, 1])
        with col_up:
            uploaded = st.file_uploader("Drop your CSV file here", type=["csv"],
                                        help="Columns auto-detected: Date, Revenue, Units_Sold, Price")
        with col_s:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📦 Load Sample Data"):
                st.session_state.raw_df = generate_sample_data()
                st.success("36-month synthetic retail dataset loaded.")

        if uploaded:
            try:
                st.session_state.raw_df = pd.read_csv(uploaded)
                st.success(f"✓ **{uploaded.name}** — {len(st.session_state.raw_df):,} rows, "
                           f"{len(st.session_state.raw_df.columns)} columns")
            except Exception as e:
                st.error(f"Read error: {e}")

        if st.session_state.raw_df is not None:
            df_raw = st.session_state.raw_df
            with st.expander("📋 Raw data preview", expanded=True):
                st.dataframe(df_raw.head(10), use_container_width=True, hide_index=True)

            st.markdown("<div style='height:1px;background:#1f2937;margin:20px 0;'></div>",
                        unsafe_allow_html=True)

            col_b, col_i = st.columns([1, 3])
            with col_b:
                run_clean = st.button("🧹 Validate & Clean")
            with col_i:
                st.markdown("<div class='info-bar'>Caps outliers (3×IQR) · Forward-fills gaps · "
                            "Sorts by date · Engineers 14 ML features</div>", unsafe_allow_html=True)

            if run_clean:
                with st.spinner("Cleaning…"):
                    cleaned, log = validate_and_clean(df_raw)
                    st.session_state.cleaned_df     = cleaned
                    st.session_state.validation_log = log
                with st.spinner("Engineering features…"):
                    st.session_state.featured_df = engineer_features(cleaned)
                st.success("✅ Data pipeline complete.")
                st.rerun()

            if st.session_state.validation_log:
                with st.expander("🔍 Validation log", expanded=False):
                    for e in st.session_state.validation_log:
                        k = {'success':'success','warning':'warning','error':'danger'}.get(e['type'],'info')
                        _insight(e['msg'], k)

            if st.session_state.cleaned_df is not None:
                cl = st.session_state.cleaned_df
                st.markdown("<div style='height:1px;background:#1f2937;margin:20px 0;'></div>",
                            unsafe_allow_html=True)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Clean Rows",    f"{len(cl):,}")
                c2.metric("Total Revenue", f"${cl['Revenue'].sum()/1000:.1f}k")
                c3.metric("Avg / Period",  f"${cl['Revenue'].mean():,.0f}")
                c4.metric("Date Span",     f"{len(cl)} periods")
                col_l, col_r = st.columns(2)
                with col_l:
                    with st.expander("📊 Cleaned data", expanded=False):
                        st.dataframe(cl, use_container_width=True, hide_index=True)
                with col_r:
                    with st.expander("⚙️ Feature matrix", expanded=False):
                        st.dataframe(st.session_state.featured_df.head(12),
                                     use_container_width=True, hide_index=True)
        else:
            _empty("📂", "No data loaded yet",
                   "Upload a CSV or click 'Load Sample Data' to begin")

# ════════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ════════════════════════════════════════════════════════════════════════════════
elif active_step == 1:
    if not can_access_step(1):
        show_access_denied("EDA")
    elif st.session_state.cleaned_df is None:
        _empty("📊", "Complete Step 1 first", "Upload and clean your data to unlock EDA")
    else:
        df  = st.session_state.cleaned_df
        rev = df['Revenue'].values
        avg_g = df['Growth_Rate'].mean() if 'Growth_Rate' in df.columns else 0
        cv    = rev.std() / rev.mean() * 100

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Peak Revenue",    f"${rev.max():,.0f}")
        c2.metric("Min Revenue",     f"${rev.min():,.0f}")
        c3.metric("Avg Growth/Mo",   f"{avg_g:+.1f}%")
        c4.metric("Volatility (CV)", f"{cv:.1f}%")

        st.markdown("<div style='height:1px;background:#1f2937;margin:20px 0;'></div>",
                    unsafe_allow_html=True)

        # Trend
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(
            x=df['Date'], y=df['Revenue'],
            fill='tozeroy', fillcolor='rgba(99,102,241,0.07)',
            line=dict(color='#6366f1', width=2.5),
            mode='lines+markers', marker=dict(size=4, color='#6366f1'), name='Revenue'
        ))
        if len(df) >= 3:
            ma = df['Revenue'].rolling(3, min_periods=1).mean()
            fig_t.add_trace(go.Scatter(
                x=df['Date'], y=ma, mode='lines',
                line=dict(color='#10b981', width=2, dash='dot'), name='3-mo MA'
            ))
        fig_t.update_layout(**_chart('Revenue over time', 300))
        st.plotly_chart(fig_t, use_container_width=True)

        col_l, col_r = st.columns(2)
        monthly = None
        with col_l:
            if 'Month' in df.columns:
                monthly = df.groupby('Month')['Revenue'].mean().reset_index()
                monthly['mn'] = pd.to_datetime(monthly['Month'], format='%m').dt.strftime('%b')
                fig_sea = go.Figure(go.Bar(
                    x=monthly['mn'], y=monthly['Revenue'],
                    marker=dict(color=monthly['Revenue'],
                                colorscale='Viridis', showscale=False)
                ))
                fig_sea.update_layout(**_chart('Seasonality — avg revenue by month', 270))
                st.plotly_chart(fig_sea, use_container_width=True)

        with col_r:
            fig_sc = px.scatter(df, x='Units_Sold', y='Revenue',
                                trendline='ols', color_discrete_sequence=['#06b6d4'])
            fig_sc.update_layout(**_chart('Revenue vs Units Sold', 270))
            st.plotly_chart(fig_sc, use_container_width=True)

        # Correlation
        num_cols = [c for c in ['Revenue','Units_Sold','Price','Growth_Rate','Lag_1']
                    if c in df.columns]
        fig_corr = px.imshow(df[num_cols].corr(), text_auto='.2f',
                             color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        fig_corr.update_layout(**_chart('Correlation matrix', 300))
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("#### Key observations")
        peak = (monthly['mn'].iloc[monthly['Revenue'].values.argmax()]
                if monthly is not None else "N/A")
        _insight(f"📅 Peak revenue month: <b>{peak}</b>. Front-load inventory and campaigns "
                 f"4–6 weeks prior.", 'info')
        _insight(f"📈 Average monthly growth: <b>{avg_g:+.1f}%</b> — trend is "
                 f"{'<b style=\"color:#10b981\">positive</b>' if avg_g > 0 else '<b style=\"color:#ef4444\">declining</b>'}.",
                 'success' if avg_g > 0 else 'danger')
        _insight(f"📊 Volatility CV = <b>{cv:.1f}%</b>. "
                 f"{'High variance — consider demand-smoothing.' if cv > 20 else 'Stable revenue stream.'}",
                 'warning' if cv > 20 else 'success')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 3 — Train Models
# ════════════════════════════════════════════════════════════════════════════════
elif active_step == 2:
    if not can_access_step(2):
        show_access_denied("Train Models")
    elif st.session_state.featured_df is None:
        _empty("🤖", "Complete Step 1 first", "Data must be cleaned before training")
    else:
        col_b, col_i = st.columns([1, 3])
        with col_b:
            do_train = st.button("🚀 Train All Models")
        with col_i:
            st.markdown("<div class='info-bar'>Linear Reg · Ridge · Random Forest · "
                        "Gradient Boosting · SVR · Exp. Smoothing &nbsp;·&nbsp; 5-fold CV</div>",
                        unsafe_allow_html=True)

        if do_train:
            prog  = st.progress(0, text="Starting…")
            steps = ["Preparing features","Train / test split (80/20)",
                     "Linear Regression","Ridge Regression","Random Forest",
                     "Gradient Boosting","SVR (RBF kernel)","Exp. Smoothing",
                     "5-Fold Cross-Validation","Selecting best model"]
            box  = st.empty()
            log  = ""
            for i, s in enumerate(steps):
                import time; time.sleep(0.2)
                bar = ('█' * (i+1)).ljust(10)
                log += f"  [{bar}]  {s}\n"
                box.code(log, language=None)
                prog.progress((i+1)/len(steps), text=s)
            models, best = train_all_models(st.session_state.featured_df)
            st.session_state.models     = models
            st.session_state.best_model = best
            prog.progress(1.0, text="Complete")
            st.success(f"✅ Best: **{best['name']}** — CV RMSE "
                       f"${best.get('cv_rmse_mean', best['rmse']):,.0f}")
            st.rerun()

        if st.session_state.models:
            models = st.session_state.models
            best   = st.session_state.best_model

            st.markdown("#### Model leaderboard")
            st.markdown("<span class='tag'>Ranked by CV RMSE</span> "
                        "<span class='tag tag-green'>Lower = better</span><br><br>",
                        unsafe_allow_html=True)
            for m in models:
                st.markdown(_model_card_html(m, m['name'] == best['name']),
                            unsafe_allow_html=True)

            st.markdown("<div style='height:1px;background:#1f2937;margin:20px 0;'></div>",
                        unsafe_allow_html=True)

            col_l, col_r = st.columns(2)
            with col_l:
                fig_r = go.Figure(go.Bar(
                    x=[m['name'] for m in models],
                    y=[m['rmse'] for m in models],
                    error_y=dict(type='data',
                                 array=[m.get('cv_rmse_std', 0) for m in models],
                                 color='rgba(156,163,175,0.5)', thickness=1.5, width=4),
                    marker=dict(
                        color=['#10b981' if m['name']==best['name'] else '#374151'
                               for m in models],
                        line=dict(width=0)
                    )
                ))
                fig_r.update_layout(**_chart('RMSE with CV std dev', 300))
                st.plotly_chart(fig_r, use_container_width=True)

            with col_r:
                pdf = best.get('pred_df')
                if pdf is not None:
                    fig_ap = go.Figure()
                    idx = list(range(len(pdf)))
                    fig_ap.add_trace(go.Scatter(x=idx, y=pdf['Actual'],
                                               mode='lines+markers', name='Actual',
                                               line=dict(color='#6366f1', width=2),
                                               marker=dict(size=5)))
                    fig_ap.add_trace(go.Scatter(x=idx, y=pdf['Predicted'],
                                               mode='lines+markers', name='Predicted',
                                               line=dict(color='#10b981', width=2, dash='dash'),
                                               marker=dict(size=5)))
                    fig_ap.update_layout(**_chart(f'Actual vs Predicted — {best["name"]}', 300))
                    st.plotly_chart(fig_ap, use_container_width=True)

            _insight(f"⭐ Best model: <b>{best['name']}</b> — RMSE <b>${best['rmse']:,.0f}</b>"
                     f" · CV RMSE <b>${best.get('cv_rmse_mean',best['rmse']):,.0f}"
                     f" ±{best.get('cv_rmse_std',0):,.0f}</b> · R² <b>{best['r2']:.4f}</b>",
                     'success')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 4 — Forecast
# ════════════════════════════════════════════════════════════════════════════════
elif active_step == 3:
    if not can_access_step(3):
        show_access_denied("Forecast")
    elif st.session_state.best_model is None:
        _empty("🔮", "Train models first", "Complete Step 3 to enable forecasting")
    else:
        col_ctrl, col_chart = st.columns([1, 3])
        with col_ctrl:
            st.markdown("<div style='background:#111827;border:1px solid #1f2937;"
                        "border-radius:12px;padding:20px;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px;color:#4b5563;font-weight:700;"
                        "text-transform:uppercase;letter-spacing:0.09em;margin-bottom:14px;"
                        "font-family:\"JetBrains Mono\",monospace;'>Forecast settings</div>",
                        unsafe_allow_html=True)
            horizon    = st.slider("Horizon (months)", 1, 12, 6)
            do_forecast = st.button("🔮 Generate Forecast")
            st.markdown("</div>", unsafe_allow_html=True)

        if do_forecast:
            with st.spinner("Generating…"):
                fdf = generate_forecast(st.session_state.featured_df,
                                        st.session_state.best_model, horizon)
                st.session_state.forecast_df = fdf
            st.success("Forecast ready!")
            st.rerun()

        if st.session_state.forecast_df is not None:
            fdf  = st.session_state.forecast_df
            hist = st.session_state.cleaned_df

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=hist['Date'], y=hist['Revenue'], mode='lines',
                name='Historical', line=dict(color='#6366f1', width=2)
            ))
            fig_f.add_trace(go.Scatter(
                x=pd.concat([fdf['Date'], fdf['Date'][::-1]]),
                y=pd.concat([fdf['Upper'], fdf['Lower'][::-1]]),
                fill='toself', fillcolor='rgba(16,185,129,0.08)',
                line=dict(color='rgba(0,0,0,0)'), name='90% CI'
            ))
            fig_f.add_trace(go.Scatter(
                x=fdf['Date'], y=fdf['Forecast'], mode='lines+markers',
                name='Forecast',
                line=dict(color='#10b981', width=2.5, dash='dash'),
                marker=dict(size=6, color='#10b981')
            ))
            fig_f.add_vline(x=str(hist['Date'].iloc[-1]),
                            line=dict(color='#f59e0b', dash='dot', width=1.5))
            fig_f.update_layout(**_chart('Revenue forecast with 90% CI', 400))
            st.plotly_chart(fig_f, use_container_width=True)

            last  = hist['Revenue'].iloc[-1]
            growth = ((fdf['Forecast'].iloc[-1] - last) / last) * 100
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Forecast", f"${fdf['Forecast'].sum()/1000:.1f}k")
            c2.metric("Next Month",     f"${fdf['Forecast'].iloc[0]:,.0f}")
            c3.metric("Final Month",    f"${fdf['Forecast'].iloc[-1]:,.0f}")
            c4.metric("Period Growth",  f"{growth:+.1f}%")

            st.markdown("<div style='height:1px;background:#1f2937;margin:20px 0;'></div>",
                        unsafe_allow_html=True)
            tbl = fdf.copy()
            tbl['Date']     = tbl['Date'].dt.strftime('%b %Y')
            tbl['Forecast'] = tbl['Forecast'].apply(lambda v: f"${v:,.0f}")
            tbl['Lower']    = tbl['Lower'].apply(lambda v: f"${v:,.0f}")
            tbl['Upper']    = tbl['Upper'].apply(lambda v: f"${v:,.0f}")
            tbl.columns     = ['Month','Forecast','Lower (90% CI)','Upper (90% CI)']
            st.dataframe(tbl, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 5 — Target Feasibility
# ════════════════════════════════════════════════════════════════════════════════
elif active_step == 4:
    if not can_access_step(4):
        show_access_denied("Target Check")
    elif st.session_state.forecast_df is None:
        _empty("🎯", "Generate forecast first", "Complete Step 4 to check target feasibility")
    else:
        fdf  = st.session_state.forecast_df
        hist = st.session_state.cleaned_df

        col_in, col_out = st.columns([1, 2])
        with col_in:
            st.markdown("<div style='background:#111827;border:1px solid #1f2937;"
                        "border-radius:12px;padding:20px;'>", unsafe_allow_html=True)
            target = st.number_input("Monthly target ($)", min_value=0.0,
                                     value=float(fdf['Forecast'].mean()*1.1),
                                     step=1000.0, format="%.0f")
            period = st.selectbox("Check month",
                                  [f"Month {i+1}" for i in range(len(fdf))],
                                  index=len(fdf)-1)
            idx = int(period.split(" ")[1]) - 1
            st.markdown("</div>", unsafe_allow_html=True)

        with col_out:
            fv   = fdf['Forecast'].iloc[idx]
            uv   = fdf['Upper'].iloc[idx]
            lv   = fdf['Lower'].iloc[idx]
            last = hist['Revenue'].iloc[-1]
            req  = ((target - last) / last) * 100

            if target <= fv:
                label, kind = "✅ Easily Achievable", 'success'
                note = f"Forecast <b>${fv:,.0f}</b> exceeds target by <b>{((fv-target)/target*100):.1f}%</b>."
            elif target <= uv:
                label, kind = "⚠️ Achievable — Stretch", 'warning'
                note = f"Within 90% CI upper bound (<b>${uv:,.0f}</b>). Needs strong execution."
            else:
                label, kind = "❌ Beyond Forecast Range", 'danger'
                note = f"Exceeds upper bound by <b>${target-uv:,.0f}</b>. Major intervention required."

            _insight(f"<b>{label}</b><br>{note}", kind)

            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=fv,
                delta={'reference': target, 'valueformat': ',.0f'},
                title={'text': f"Forecast vs ${target:,.0f} target",
                       'font': {'color': '#9ca3af', 'size': 13}},
                number={'prefix': '$', 'valueformat': ',.0f',
                        'font': {'color': '#f9fafb', 'size': 28}},
                gauge={
                    'axis': {'range': [0, max(target, uv)*1.15],
                             'tickcolor': '#4b5563',
                             'tickfont': {'color': '#6b7280','size': 11}},
                    'bar': {'color': '#6366f1', 'thickness': 0.25},
                    'bgcolor': '#1f2937', 'borderwidth': 0,
                    'steps': [
                        {'range': [lv, fv], 'color': 'rgba(99,102,241,0.15)'},
                        {'range': [fv, uv], 'color': 'rgba(16,185,129,0.10)'},
                    ],
                    'threshold': {'line': {'color': '#f59e0b', 'width': 3}, 'value': target}
                }
            ))
            fig_g.update_layout(paper_bgcolor='#111827', font=dict(color='#9ca3af'),
                                height=260, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

        st.markdown(f"<span class='tag'>Required growth from last period</span>&nbsp;"
                    f"<span style='font-size:18px;font-weight:700;color:#f9fafb;'>"
                    f"{req:+.1f}%</span>", unsafe_allow_html=True)
        st.markdown("<div style='height:1px;background:#1f2937;margin:20px 0;'></div>",
                    unsafe_allow_html=True)

        colors = ['#10b981' if v >= target else '#ef4444' for v in fdf['Forecast']]
        fig_c  = go.Figure(go.Bar(x=fdf['Date'].dt.strftime('%b %Y'),
                                  y=fdf['Forecast'], marker_color=colors))
        fig_c.add_hline(y=target, line=dict(color='#f59e0b', dash='dot', width=2),
                        annotation_text=f"Target ${target:,.0f}",
                        annotation_font_color='#f59e0b')
        fig_c.update_layout(**_chart('Forecast vs target — all periods', 300))
        st.plotly_chart(fig_c, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 6 — Scenario Simulation
# ════════════════════════════════════════════════════════════════════════════════
elif active_step == 5:
    if not can_access_step(5):
        show_access_denied("Scenarios")
    elif st.session_state.forecast_df is None:
        _empty("⚙️", "Generate forecast first", "Complete Step 4 to run scenarios")
    else:
        fdf  = st.session_state.forecast_df
        base = fdf['Forecast'].values

        col_ctrl, col_out = st.columns([1, 2])
        with col_ctrl:
            st.markdown("<div style='background:#111827;border:1px solid #1f2937;"
                        "border-radius:12px;padding:20px;'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:11px;color:#4b5563;font-weight:700;"
                        "text-transform:uppercase;letter-spacing:0.09em;margin-bottom:16px;"
                        "font-family:\"JetBrains Mono\",monospace;'>Business levers</div>",
                        unsafe_allow_html=True)
            price_d = st.slider("💲 Price change (%)",    -50, 50,  0, 1)
            units_d = st.slider("📦 Volume change (%)",   -50, 50,  0, 1)
            mkt_d   = st.slider("📣 Marketing spend (%)", -30, 100, 0, 5)
            st.markdown("</div>", unsafe_allow_html=True)
            _insight("Price × volume × marketing multiplier (35% ROMI). "
                     "Instant response — no model re-run needed.", 'info')

        combined = (1 + price_d/100) * (1 + units_d/100) * (1 + mkt_d/100 * 0.35)
        scenario = np.round(base * combined).astype(int)
        diff     = scenario.astype(float) - base
        total_d  = diff.sum()
        pct_d    = (total_d / base.sum()) * 100

        with col_out:
            c1, c2 = st.columns(2)
            c1.metric("Baseline Total", f"${base.sum()/1000:.1f}k")
            c2.metric("Scenario Total", f"${scenario.sum()/1000:.1f}k",
                      delta=f"{pct_d:+.1f}%")

        dates  = fdf['Date'].dt.strftime('%b %Y').tolist()
        colors = ['#10b981' if s >= b else '#ef4444'
                  for s, b in zip(scenario, base)]
        fig_s  = go.Figure()
        fig_s.add_trace(go.Bar(x=dates, y=base.tolist(), name='Baseline',
                               marker_color='#374151',
                               text=[f'${v/1000:.1f}k' for v in base],
                               textposition='outside',
                               textfont=dict(color='#6b7280', size=10)))
        fig_s.add_trace(go.Bar(x=dates, y=scenario.tolist(), name='Scenario',
                               marker_color=colors,
                               text=[f'${v/1000:.1f}k' for v in scenario],
                               textposition='outside',
                               textfont=dict(color='#9ca3af', size=10)))
        fig_s.update_layout(**_chart('Baseline vs Scenario', 360), barmode='group')
        st.plotly_chart(fig_s, use_container_width=True)

        if abs(price_d) > 5 and abs(units_d) > 5:
            msg = (f"Combined price ({price_d:+d}%) + volume ({units_d:+d}%) "
                   f"→ net impact: <b>{pct_d:+.1f}%</b>")
        elif mkt_d > 20:
            msg = f"Marketing +{mkt_d}% → estimated lift: <b>{pct_d:+.1f}%</b> (35% ROMI)"
        elif abs(price_d) > 5:
            msg = (f"Price {price_d:+d}% → revenue impact: <b>{pct_d:+.1f}%</b>. "
                   f"Monitor demand elasticity.")
        else:
            msg = f"Scenario impact: <b>{pct_d:+.1f}%</b> vs baseline."
        _insight(msg, 'success' if total_d >= 0 else 'danger')

        st.markdown("<div style='height:1px;background:#1f2937;margin:20px 0;'></div>",
                    unsafe_allow_html=True)
        if st.button("💾 Save Scenario"):
            if not st.session_state.saved_scenarios:
                st.session_state.saved_scenarios = []
            st.session_state.saved_scenarios.append({
                'Price Δ': f"{price_d:+d}%", 'Volume Δ': f"{units_d:+d}%",
                'Marketing Δ': f"{mkt_d:+d}%",
                'Revenue Impact': f"${total_d/1000:+.1f}k",
                'Impact %': f"{pct_d:+.1f}%"
            })
            st.success("Scenario saved!")

        if st.session_state.saved_scenarios:
            st.markdown("#### Saved scenarios")
            st.dataframe(pd.DataFrame(st.session_state.saved_scenarios),
                         use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 7 — Insights
# ════════════════════════════════════════════════════════════════════════════════
elif active_step == 6:
    if not can_access_step(6):
        show_access_denied("Insights")
    elif st.session_state.best_model is None:
        _empty("💡", "Train models first", "Complete Step 3 to unlock AI insights")
    else:
        col_b, col_i = st.columns([1, 3])
        with col_b:
            do_insights = st.button("💡 Compute SHAP Insights")
        with col_i:
            method = st.session_state.best_model.get('shap_method', '')
            label  = f"Method: <b style='color:#f9fafb;'>{method}</b> · Mean |SHAP| attribution" \
                     if method else "Click to compute SHAP feature importance"
            st.markdown(f"<div class='info-bar'>{label}</div>", unsafe_allow_html=True)

        if do_insights:
            with st.spinner("Computing SHAP values…"):
                fi = compute_feature_importance(st.session_state.best_model,
                                                st.session_state.featured_df)
                st.session_state.feature_importance = fi
            m = st.session_state.best_model.get('shap_method', '')
            st.success(f"✅ SHAP analysis complete — {m}")
            st.rerun()

        if st.session_state.feature_importance is not None:
            fi = st.session_state.feature_importance

            col_l, col_r = st.columns(2)
            with col_l:
                fi_s = fi.sort_values('Importance', ascending=True)
                fig_fi = go.Figure(go.Bar(
                    x=fi_s['Importance'], y=fi_s['Feature'], orientation='h',
                    marker=dict(
                        color=fi_s['Importance'],
                        colorscale=[[0,'#374151'],[0.5,'#6366f1'],[1,'#06b6d4']],
                        showscale=False
                    ),
                    text=[f"{v:.1f}%" for v in fi_s['Importance']],
                    textposition='outside',
                    textfont=dict(color='#9ca3af', size=11)
                ))
                fig_fi.update_layout(**_chart('Mean |SHAP| feature importance', 340))
                fig_fi.update_xaxes(ticksuffix='%')
                st.plotly_chart(fig_fi, use_container_width=True)

            with col_r:
                fig_pie = go.Figure(go.Pie(
                    labels=fi['Feature'], values=fi['Importance'], hole=0.5,
                    marker=dict(colors=['#6366f1','#10b981','#f59e0b','#ef4444',
                                        '#06b6d4','#8b5cf6','#ec4899','#14b8a6',
                                        '#f97316','#84cc16']),
                    textfont=dict(color='#9ca3af', size=11)
                ))
                fig_pie.update_layout(**_chart('Revenue driver breakdown', 340))
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("#### Revenue driver insights")
            for ins in generate_insights(fi, st.session_state.cleaned_df,
                                         st.session_state.best_model):
                kind = ins.get('type','info')
                _insight(ins['text'], kind)

# ════════════════════════════════════════════════════════════════════════════════
# STEP 8 — Report
# ════════════════════════════════════════════════════════════════════════════════
elif active_step == 7:
    if not can_access_step(7):
        show_access_denied("Report")
    elif st.session_state.cleaned_df is None or st.session_state.forecast_df is None:
        _empty("📥", "Complete the pipeline first",
               "Steps 1–4 are required before generating a report")
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("""
<div style='background:#111827;border:1px solid #1f2937;border-radius:14px;padding:24px;'>
  <div style='font-size:15px;font-weight:700;color:#f9fafb;margin-bottom:6px;'>📄 PDF Report</div>
  <div style='font-size:12.5px;color:#6b7280;margin-bottom:20px;'>
    Summary stats · Model comparison · Forecast table · Feature importance
  </div>
""", unsafe_allow_html=True)
            pdf = generate_report_pdf(
                st.session_state.cleaned_df, st.session_state.models,
                st.session_state.forecast_df, st.session_state.feature_importance
            )
            st.download_button("⬇️ Download PDF Report", pdf,
                               "reviq_report.pdf", "application/pdf")
            st.markdown("</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("""
<div style='background:#111827;border:1px solid #1f2937;border-radius:14px;padding:24px;'>
  <div style='font-size:15px;font-weight:700;color:#f9fafb;margin-bottom:6px;'>📊 CSV Export</div>
  <div style='font-size:12.5px;color:#6b7280;margin-bottom:20px;'>
    Cleaned data · Forecast table · Feature importance as structured CSV
  </div>
""", unsafe_allow_html=True)
            csv = generate_report_csv(
                st.session_state.cleaned_df, st.session_state.models,
                st.session_state.forecast_df, st.session_state.feature_importance
            )
            st.download_button("⬇️ Download CSV Export", csv,
                               "reviq_data.csv", "text/csv")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:1px;background:#1f2937;margin:24px 0;'></div>",
                    unsafe_allow_html=True)

        st.markdown("#### Pipeline summary")
        cl   = st.session_state.cleaned_df
        fdf  = st.session_state.forecast_df
        best = st.session_state.best_model

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Data Points",    f"{len(cl):,}")
        c2.metric("Best Model",     best['name'].split()[0])
        c3.metric("Model R²",       f"{best['r2']:.3f}")
        c4.metric("Forecast Total", f"${fdf['Forecast'].sum()/1000:.1f}k")

        if st.session_state.feature_importance is not None:
            st.markdown("#### Top revenue drivers")
            for _, row in st.session_state.feature_importance.head(3).iterrows():
                _insight(f"<b>{row['Feature']}</b> — "
                         f"{row['Importance']:.1f}% mean |SHAP| importance", 'info')

# ════════════════════════════════════════════════════════════════════════════════
# STEP 9 — Activity Log
# ════════════════════════════════════════════════════════════════════════════════
elif active_step == 8:
    show_activity_log()
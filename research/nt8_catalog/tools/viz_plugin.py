import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import glob
import json

st.set_page_config(
    page_title="Catalog Event Viz",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

IS_DARK = st.session_state.theme == "dark"

CSS = f"""
<style>
:root {{
    --bg: {'#09090b' if IS_DARK else '#ffffff'};
    --bg-subtle: {'#0c0c0f' if IS_DARK else '#f9fafb'};
    --card: {'#0c0c0f' if IS_DARK else '#ffffff'};
    --border: {'#1e1e24' if IS_DARK else '#e4e4e7'};
    --text: {'#fafafa' if IS_DARK else '#09090b'};
    --text-muted: #71717a;
    --green: {'#22c55e' if IS_DARK else '#16a34a'};
    --radius: 10px;
}}
header[data-testid="stHeader"], [data-testid="stToolbar"] {{ display: none !important; }}
html, body, [data-testid="stAppViewContainer"], .main, .block-container {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}}
.block-container {{ padding: 2rem 2.5rem 3rem !important; max-width: 1400px !important; }}
.chart-wrap {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.2rem; }}
.chart-title {{ font-size: 0.82rem; font-weight: 600; color: var(--text); }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

head_left, head_right = st.columns([8, 1])
with head_left:
    st.markdown("<h3 style='margin:0'>Catalog Events vs Golden Labels</h3>", unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light" if IS_DARK else "🌙 Dark"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

@st.cache_data
def load_data():
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    REPORTS = os.path.join(ROOT, "research", "nt8_catalog", "reports")
    HORIZONS = os.path.join(REPORTS, "fps_horizons.parquet")
    LABELS_DIR = os.path.join(ROOT, "DATA", "ai_cusp_picks")

    # Load Catalog Events
    df = pd.read_parquet(HORIZONS)
    df.loc[df['doss'] == 'ORB-02', 'entry_ts'] += 1800
    df = df[~df['doss'].isin(['SEASON-12', 'RENKO-24'])]
    df['datetime'] = pd.to_datetime(df['entry_ts'], unit='s', utc=True)
    
    # Load Labels
    files = glob.glob(os.path.join(LABELS_DIR, "ai_picks_*_multi.json"))
    trades = []
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
            if 'trades' in data:
                trades.extend(data['trades'])
    df_labels = pd.DataFrame(trades)
    df_labels['doss'] = '★ GOLDEN_LABEL'
    df_labels['datetime'] = pd.to_datetime(df_labels['entry_ts'], unit='s', utc=True)
    
    return df, df_labels

with st.spinner("Loading millions of rows... just kidding, only entries!"):
    df_cat, df_lab = load_data()

min_date = df_cat['datetime'].min().date()
max_date = df_cat['datetime'].max().date()

st.sidebar.header("Controls")
date_range = st.sidebar.date_input("Select Date Range (Portion)", value=(min_date, min_date + pd.Timedelta(days=7)), min_value=min_date, max_value=max_date)

dossiers = sorted(df_cat['doss'].unique())
selected_dossiers = st.sidebar.multiselect("Select Strategies", options=dossiers, default=dossiers)

if len(date_range) == 2:
    start_date, end_date = date_range
    start_ts = pd.to_datetime(start_date).tz_localize('UTC')
    end_ts = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(days=1)
    
    mask_cat = (df_cat['datetime'] >= start_ts) & (df_cat['datetime'] < end_ts) & (df_cat['doss'].isin(selected_dossiers))
    mask_lab = (df_lab['datetime'] >= start_ts) & (df_lab['datetime'] < end_ts)
    
    df_cat_sub = df_cat[mask_cat]
    df_lab_sub = df_lab[mask_lab]
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Catalog Entries", len(df_cat_sub))
    with c2:
        st.metric("Golden Labels", len(df_lab_sub))
        
    st.markdown('<div class="chart-wrap"><div class="chart-title">Event Timeline (Scatter)</div>', unsafe_allow_html=True)
    
    fig = go.Figure()
    
    # Add Golden Labels on Top
    if not df_lab_sub.empty:
        fig.add_trace(go.Scatter(
            x=df_lab_sub['datetime'],
            y=[y.split('_')[0] if '_' in y else y for y in df_lab_sub['doss']],
            mode='markers',
            marker=dict(symbol='star', size=12, color='gold', line=dict(width=1, color='darkorange')),
            name='Golden Labels'
        ))
    
    # Add Catalog Entries
    for d in selected_dossiers:
        d_df = df_cat_sub[df_cat_sub['doss'] == d]
        if not d_df.empty:
            fig.add_trace(go.Scatter(
                x=d_df['datetime'],
                y=d_df['doss'],
                mode='markers',
                marker=dict(size=5, opacity=0.6),
                name=d
            ))
            
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=11),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)", categoryorder='category descending'),
        height=700,
        showlegend=False,
        hovermode="closest"
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("Please select a full date range in the sidebar.")

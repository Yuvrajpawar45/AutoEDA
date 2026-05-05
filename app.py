import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from utils.eda import run_eda
from utils.charts import plot_distributions, plot_correlation_heatmap, plot_outliers
from utils.insights import generate_insights_stream

st.set_page_config(
    page_title="DataLens · EDA Agent",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', system-ui, sans-serif !important;
    background: #EEF4FF !important;
    color: #0F2340 !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #C8DCFF !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #1E3A5F !important; }

.brand-bar {
    background: linear-gradient(135deg, #1A6BFF 0%, #0A4FCC 100%);
    padding: 22px 24px 18px;
    margin: 0 0 24px 0;
}
.brand-title { font-size: 19px; font-weight: 600; color: #FFFFFF; letter-spacing: -0.3px; margin: 0; }
.brand-sub { font-size: 10px; color: rgba(255,255,255,0.65); margin: 3px 0 0; letter-spacing: 0.8px; text-transform: uppercase; }

.sidebar-label {
    font-size: 10px; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #5A7BA0;
    margin: 18px 0 6px; padding: 0 2px;
}
.sidebar-footer { font-size: 11px; color: #8AAAC8; text-align: center; line-height: 1.7; margin-top: 8px; }

/* Main layout */
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* Top bar */
.top-nav {
    background: #FFFFFF;
    border-bottom: 1px solid #C8DCFF;
    padding: 0 32px;
    display: flex; align-items: center; justify-content: space-between;
    height: 58px;
}
.nav-left { font-size: 14px; font-weight: 600; color: #0F2340; letter-spacing: -0.2px; }
.nav-right {
    display: inline-flex; align-items: center; gap: 6px;
    background: #EBF4FF; border: 1px solid #B8D4FF;
    border-radius: 20px; padding: 4px 14px;
    font-size: 12px; font-weight: 500; color: #1A6BFF;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #16A34A; display: inline-block; }

/* Page header */
.page-header { padding: 26px 32px 0; }
.page-title { font-size: 24px; font-weight: 600; color: #0F2340; letter-spacing: -0.4px; margin: 0 0 3px; }
.page-sub { font-size: 13px; color: #5A7BA0; margin: 0; }
.dataset-chip {
    display: inline-flex; align-items: center;
    background: #EBF4FF; border: 1px solid #B8D4FF;
    border-radius: 6px; padding: 4px 12px; margin-top: 12px;
    font-size: 12px; font-weight: 500; color: #1A5FCC;
    font-family: 'DM Mono', monospace; letter-spacing: 0.2px;
}

/* Metric strip */
.metrics-row {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 12px; padding: 20px 32px;
}
.mc {
    background: #FFFFFF; border: 1px solid #C8DCFF;
    border-radius: 10px; padding: 15px 18px;
    position: relative; overflow: hidden;
}
.mc::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: linear-gradient(90deg, #1A6BFF, #60A5FA);
}
.mc.warn::before { background: linear-gradient(90deg, #D97706, #FBBF24); }
.mc-label { font-size: 10px; font-weight: 600; letter-spacing: 0.7px; text-transform: uppercase; color: #5A7BA0; margin-bottom: 6px; }
.mc-value { font-size: 26px; font-weight: 700; color: #0F2340; letter-spacing: -1px; line-height: 1; }
.mc.warn .mc-value { color: #B45309; }

.section-divider { height: 1px; background: #C8DCFF; margin: 0 32px; }

/* Tabs */
div[data-testid="stTabs"] > div:first-child {
    padding: 0 32px !important;
    background: #FFFFFF;
    border-bottom: 1px solid #C8DCFF;
    gap: 0 !important;
}
button[data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    color: #5A7BA0 !important;
    padding: 14px 22px !important; border-radius: 0 !important;
    border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #1A6BFF !important;
    border-bottom-color: #1A6BFF !important;
    background: transparent !important;
    font-weight: 600 !important;
}
div[data-testid="stTabs"] > div:last-child { padding: 24px 32px !important; }

/* Cards */
.card { background: #FFFFFF; border: 1px solid #C8DCFF; border-radius: 10px; padding: 20px 22px; margin-bottom: 16px; }
.card-title { font-size: 13px; font-weight: 600; color: #0F2340; margin-bottom: 14px; }

/* DataFrames */
[data-testid="stDataFrame"] { border: 1px solid #C8DCFF !important; border-radius: 8px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] th { background: #F0F6FF !important; color: #0F2340 !important; font-weight: 600 !important; }
[data-testid="stDataFrame"] td { color: #1E3A5F !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1A6BFF 0%, #0A4FCC 100%) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 10px 22px !important; letter-spacing: 0.1px !important;
    box-shadow: 0 2px 8px rgba(26,107,255,0.20) !important;
}
.stButton > button:hover { opacity: 0.87 !important; }

/* Insight box */
.insight-output {
    background: #FFFFFF; border: 1px solid #C8DCFF; border-radius: 10px;
    padding: 28px 32px; font-size: 14px; line-height: 1.85; color: #1E3A5F;
}
.insight-output h2 {
    font-size: 15px; font-weight: 700; color: #0F2340;
    margin: 22px 0 10px; padding-bottom: 7px;
    border-bottom: 1px solid #EBF4FF;
}
.insight-output h2:first-child { margin-top: 0; }
.insight-output strong { color: #0F2340; }
.insight-output code {
    background: #EBF4FF; color: #1A5FCC;
    padding: 1px 6px; border-radius: 4px;
    font-family: 'DM Mono', monospace; font-size: 12px;
}
.insight-output ul { padding-left: 18px; }
.insight-output li { margin-bottom: 5px; color: #1E3A5F; }
.insight-output p { color: #1E3A5F; }

/* Insight placeholder */
.insight-placeholder {
    background: #F7FBFF; border: 1.5px dashed #B8D4FF;
    border-radius: 10px; padding: 48px 32px;
    text-align: center;
}
.insight-placeholder .ph-title { font-size: 15px; font-weight: 600; color: #1E3A5F; margin: 0 0 6px; }
.insight-placeholder .ph-sub { font-size: 13px; color: #5A7BA0; margin: 0; }

/* Outlier info box */
.outlier-info {
    background: #F7FBFF; border: 1px solid #C8DCFF;
    border-radius: 8px; padding: 16px;
}
.oi-row { display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #EBF4FF; font-size: 13px; }
.oi-row:last-child { border-bottom: none; }
.oi-label { color: #5A7BA0; }
.oi-value { font-weight: 600; color: #0F2340; }
.oi-danger { color: #B91C1C !important; }

/* Warning banner */
.warn-banner {
    background: #FFFBEB; border: 1px solid #FDE68A;
    border-radius: 8px; padding: 11px 16px; margin-bottom: 18px;
    font-size: 13px; color: #92400E; font-weight: 500;
}

/* Select, slider, expander */
[data-testid="stSelectbox"] > div > div { border-color: #C8DCFF !important; border-radius: 7px !important; color: #1E3A5F !important; }
[data-testid="stExpander"] { border: 1px solid #C8DCFF !important; border-radius: 8px !important; background: #FFFFFF !important; }
[data-testid="stExpander"] summary { color: #0F2340 !important; font-weight: 500 !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 13px !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #EEF4FF; }
::-webkit-scrollbar-thumb { background: #B8D4FF; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-bar">
        <p class="brand-title">DataLens</p>
        <p class="brand-sub">Automated EDA Agent</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    st.markdown('<div class="sidebar-label">Analysis Options</div>', unsafe_allow_html=True)
    outlier_method = st.selectbox("Outlier detection", ["IQR (1.5x)", "IQR (3x)", "Z-score (>3σ)"])
    max_cat_unique = st.slider("Max categorical unique values", 2, 50, 20)

    st.markdown('<div class="sidebar-label">Export</div>', unsafe_allow_html=True)
    export_btn = st.button("Download Report", use_container_width=True)

    st.markdown("""
    <div class="sidebar-footer" style="margin-top:32px">
        Streamlit · Pandas · Matplotlib<br>Groq · LLaMA 3.1
    </div>
    """, unsafe_allow_html=True)

# ── Demo data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_demo():
    np.random.seed(42)
    n = 300
    depts = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    dept = np.random.choice(depts, n)
    age = np.random.randint(22, 60, n)
    yexp = np.clip(age - 22 - np.random.randint(0, 4, n), 0, None)
    salary = (25000 + yexp * 3000 + np.random.randint(-5000, 8000, n) + (dept == "Engineering") * 15000)
    perf = np.clip(np.random.normal(6, 1.5, n), 1, 10).round(1)
    satisfaction = np.random.randint(1, 11, n)
    attrition = np.where(np.random.rand(n) < 0.15, "Yes", "No")
    salary[0] = 500000; age[1] = 95
    return pd.DataFrame({"age": age, "salary": salary.astype(int), "department": dept,
        "years_exp": yexp, "performance_score": perf, "satisfaction": satisfaction, "attrition": attrition})

# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-nav">
    <span class="nav-left">Automated Data Analysis Agent</span>
    <span class="nav-right"><span class="status-dot"></span> Ready</span>
</div>
""", unsafe_allow_html=True)

# ── Empty state ───────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown('<div style="padding:32px">', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("""
        <div style="background:#FFFFFF;border:1px solid #C8DCFF;border-radius:12px;
             padding:48px;text-align:center;margin-top:32px">
            <p style="font-size:42px;margin:0 0 14px;line-height:1">&#9635;</p>
            <p style="font-size:17px;font-weight:600;color:#0F2340;margin:0 0 8px">Upload a CSV to begin</p>
            <p style="font-size:13px;color:#5A7BA0;margin:0;line-height:1.6">
                The agent runs full EDA automatically — distributions, correlations,<br>
                outlier detection, and AI-written insights powered by LLaMA 3.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='margin-top:32px'>", unsafe_allow_html=True)
        if st.button("Try Demo Dataset", use_container_width=True):
            st.session_state.use_demo = True
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    if not st.session_state.get("use_demo"):
        st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    dataset_name = uploaded_file.name
    st.session_state.use_demo = False
else:
    df = load_demo()
    dataset_name = "employee_demo.csv"

eda = run_eda(df, outlier_method=outlier_method, max_cat_unique=max_cat_unique)
out_count = eda['total_outliers']

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <p class="page-title">Dataset Analysis</p>
    <p class="page-sub">Exploratory data analysis — all statistics computed automatically</p>
    <span class="dataset-chip">{dataset_name}</span>
</div>
""", unsafe_allow_html=True)

# ── Metrics ───────────────────────────────────────────────────────────────────
warn_cls = "mc warn" if out_count > 0 else "mc"
st.markdown(f"""
<div class="metrics-row">
    <div class="mc"><div class="mc-label">Total Rows</div><div class="mc-value">{eda['n_rows']:,}</div></div>
    <div class="mc"><div class="mc-label">Columns</div><div class="mc-value">{eda['n_cols']}</div></div>
    <div class="mc"><div class="mc-label">Numeric</div><div class="mc-value">{len(eda['num_cols'])}</div></div>
    <div class="mc"><div class="mc-label">Categorical</div><div class="mc-value">{len(eda['cat_cols'])}</div></div>
    <div class="{warn_cls}"><div class="mc-label">Outliers</div><div class="mc-value">{out_count}</div></div>
</div>
<div class="section-divider"></div>
""", unsafe_allow_html=True)

# ── Export ────────────────────────────────────────────────────────────────────
if export_btn:
    buf = io.StringIO()
    buf.write(f"DataLens EDA Report — {dataset_name}\n{'='*60}\n\n")
    buf.write(f"Rows: {eda['n_rows']:,}  |  Columns: {eda['n_cols']}  |  Outliers: {out_count}\n\n")
    buf.write("COLUMN SUMMARY\n" + "-"*40 + "\n")
    buf.write(eda['summary_df'].to_string(index=False))
    buf.write("\n\nMISSING VALUES\n" + "-"*40 + "\n")
    buf.write(eda['missing'].to_string(index=False) if not eda['missing'].empty else "None")
    st.download_button("Download Report (.txt)", buf.getvalue(),
        file_name=f"eda_{dataset_name}.txt", mime="text/plain")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Overview  ", "  Distributions  ", "  Correlations  ", "  Outliers  ", "  AI Insights  "
])

# ══ OVERVIEW ══
with tab1:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown('<div class="card"><div class="card-title">Column Summary</div>', unsafe_allow_html=True)
        st.dataframe(eda['summary_df'], use_container_width=True, height=340, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card"><div class="card-title">Missing Values</div>', unsafe_allow_html=True)
        if eda['missing'].empty:
            st.success("No missing values detected in any column.")
        else:
            st.dataframe(eda['missing'], use_container_width=True, hide_index=True)
            fig, ax = plt.subplots(figsize=(6, max(2, len(eda['missing']) * 0.5)))
            eda['missing'].set_index("Column")["Missing %"].plot(kind="barh", ax=ax, color="#1A6BFF", alpha=0.85)
            ax.set_xlabel("Missing %", fontsize=10); ax.set_title("Missing values per column", fontsize=11)
            ax.spines[['top','right']].set_visible(False); ax.grid(axis='x', alpha=0.3)
            fig.patch.set_facecolor('white'); plt.tight_layout()
            st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">Data Preview — first 10 rows</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══ DISTRIBUTIONS ══
with tab2:
    if eda['num_cols']:
        st.markdown('<p style="font-size:14px;font-weight:600;color:#0F2340;margin-bottom:16px">Numeric Distributions</p>', unsafe_allow_html=True)
        figs = plot_distributions(df, eda['num_cols'])
        cols = st.columns(2, gap="large")
        for i, fig in enumerate(figs):
            with cols[i % 2]: st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.info("No numeric columns found.")

    if eda['cat_cols']:
        st.markdown("---")
        st.markdown('<p style="font-size:14px;font-weight:600;color:#0F2340;margin-bottom:10px">Categorical Distributions</p>', unsafe_allow_html=True)
        cat_sel = st.multiselect("Columns", eda['cat_cols'], default=eda['cat_cols'][:3], label_visibility="collapsed")
        if cat_sel:
            ccols = st.columns(min(len(cat_sel), 2), gap="large")
            for i, col in enumerate(cat_sel):
                with ccols[i % 2]:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    vc = df[col].value_counts().head(10)
                    labels = [str(l)[:26] + ("..." if len(str(l)) > 26 else "") for l in vc.index]
                    ax.bar(range(len(vc)), vc.values, color="#1A6BFF", alpha=0.82, width=0.62)
                    ax.set_xticks(range(len(vc))); ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=9)
                    ax.set_title(col, fontsize=12, fontweight='600', pad=10, color="#0F2340")
                    ax.set_ylabel("Count", fontsize=10, color="#5A7BA0")
                    ax.tick_params(colors='#1E3A5F')
                    ax.spines[['top','right']].set_visible(False); ax.grid(axis='y', alpha=0.3)
                    fig.patch.set_facecolor('white'); plt.tight_layout()
                    st.pyplot(fig, use_container_width=True, dpi=90); plt.close(fig)

# ══ CORRELATIONS ══
with tab3:
    if len(eda['num_cols']) < 2:
        st.info("At least 2 numeric columns are needed for correlation analysis.")
    else:
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.markdown('<p style="font-size:14px;font-weight:600;color:#0F2340;margin-bottom:12px">Correlation Heatmap</p>', unsafe_allow_html=True)
            fig = plot_correlation_heatmap(df, eda['num_cols'])
            st.pyplot(fig, use_container_width=True); plt.close(fig)
        with c2:
            st.markdown('<p style="font-size:14px;font-weight:600;color:#0F2340;margin-bottom:12px">Top Correlations</p>', unsafe_allow_html=True)
            if not eda['top_correlations'].empty:
                st.dataframe(
                    eda['top_correlations'].style.background_gradient(subset=['Correlation'], cmap='RdYlBu_r', vmin=-1, vmax=1),
                    use_container_width=True, hide_index=True, height=400)

# ══ OUTLIERS ══
with tab4:
    outlier_data = eda['outliers']
    if not outlier_data:
        st.success("No outliers detected using the selected method.")
    else:
        st.markdown(f"""
        <div class="warn-banner">
            Found outliers in <strong>{len(outlier_data)}</strong> column(s) using <strong>{outlier_method}</strong>
        </div>""", unsafe_allow_html=True)
        for col, info in outlier_data.items():
            with st.expander(f"{col}  —  {info['count']} outliers  ({info['pct']:.1f}% of rows)", expanded=True):
                c1, c2 = st.columns([1, 2], gap="large")
                with c1:
                    st.markdown(f"""
                    <div class="outlier-info">
                        <div class="oi-row"><span class="oi-label">Count</span><span class="oi-value">{info['count']}</span></div>
                        <div class="oi-row"><span class="oi-label">% of rows</span><span class="oi-value">{info['pct']:.1f}%</span></div>
                        <div class="oi-row"><span class="oi-label">Lower bound</span><span class="oi-value">{info['lower']:.2f}</span></div>
                        <div class="oi-row"><span class="oi-label">Upper bound</span><span class="oi-value">{info['upper']:.2f}</span></div>
                        <div class="oi-row"><span class="oi-label">Min outlier</span><span class="oi-value oi-danger">{info['min_out']:.2f}</span></div>
                        <div class="oi-row"><span class="oi-label">Max outlier</span><span class="oi-value oi-danger">{info['max_out']:.2f}</span></div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    fig = plot_outliers(df, col)
                    st.pyplot(fig, use_container_width=True); plt.close(fig)

# ══ AI INSIGHTS ══
with tab5:
    st.markdown("""
    <div style="margin-bottom:20px">
        <p style="font-size:17px;font-weight:600;color:#0F2340;margin:0 0 4px">AI-Powered Insights</p>
        <p style="font-size:13px;color:#5A7BA0;margin:0">LLaMA 3.1 via Groq &nbsp;·&nbsp; Free &nbsp;·&nbsp; Instant streaming</p>
    </div>
    """, unsafe_allow_html=True)

    c_btn, c_export, c_info = st.columns([1, 1, 3])
    with c_btn:
        run_insights = st.button("Generate Insights", use_container_width=True)
    with c_export:
        export_insights = st.button("Export Report", use_container_width=True)
    with c_info:
        st.markdown('<p style="font-size:12px;color:#8AAAC8;margin-top:10px;padding-left:4px">Analyzes statistics · quality · patterns · recommendations</p>', unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    result_area = st.empty()

    if "insights_text" not in st.session_state:
        st.session_state.insights_text = ""

    if run_insights:
        with st.spinner("Analyzing with LLaMA 3.1..."):
            full_text = ""
            try:
                for chunk in generate_insights_stream(df, eda):
                    full_text += chunk
                    result_area.markdown(f'<div class="insight-output">{full_text}</div>', unsafe_allow_html=True)
                st.session_state.insights_text = full_text
            except Exception as e:
                st.error(f"Error: {str(e)}")
    elif export_insights and st.session_state.insights_text:
        st.download_button("Download Insights (.txt)", st.session_state.insights_text,
            file_name=f"insights_{dataset_name}.txt", mime="text/plain")
    elif st.session_state.insights_text:
        result_area.markdown(f'<div class="insight-output">{st.session_state.insights_text}</div>', unsafe_allow_html=True)
    else:
        result_area.markdown("""
        <div class="insight-placeholder">
            <p class="ph-title">Ready to analyze</p>
            <p class="ph-sub">Click Generate Insights — covers statistics, data quality, patterns, and modeling recommendations</p>
        </div>""", unsafe_allow_html=True)
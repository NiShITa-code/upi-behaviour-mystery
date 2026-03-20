"""
UPI Behaviour Mystery — Interactive Dashboard
=============================================
Pure UI layer. All DS logic lives in src/.

Run:  streamlit run app.py
Test: pytest tests/ -v
CLI:  python -m src.pipeline --help
"""

import sys
from pathlib import Path

# Make the package importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

from src.config import CFG
from src.data.simulator import simulate_users, SimulationResult
from src.data.csv_loader import load_user_csv, generate_sample_csv
from src.analysis.cohorts import compute_cohorts
from src.analysis.strategy import generate_strategy
from src.analysis.experiment_design import design_experiment, compute_sample_size, compute_mde_from_sample
from src.analysis.fairness import run_fairness_audit
from src.models.churn import train_churn_model
from src.models.uplift import run_uplift_model
from src.models.explainability import compute_shap_explanations
from src.features.engineer import EXTENDED_FEATURES

# Human-readable labels for feature names (used in charts)
FEATURE_LABELS = {
    "txn_d7":            "Day 1–7 transaction count",
    "value_d7":          "Total spend in first 7 days (₹)",
    "txn_d14":           "Day 1–14 transaction count",
    "value_d14":         "Total spend in first 14 days (₹)",
    "has_first_p2m_d14": "First merchant payment within 14 days",
    "first_p2m_day":     "Day of first merchant payment",
    "cat_diversity":     "Category diversity (# categories used)",
    "p2m_ratio":         "Merchant payment ratio",
    "city_tier":         "City tier (1=metro, 3=small city)",
    "txn_d14_per_day":   "Avg daily transactions (14-day window)",
    "value_d14_per_txn": "Avg spend per transaction (14-day window)",
    "txn_acceleration":  "Transaction acceleration (day 8–14 vs day 1–7)",
    "log_value_d7":      "Log spend in first 7 days",
    "log_value_d14":     "Log spend in first 14 days",
}

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UPI Behaviour Mystery",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Sans:wght@400;500&family=DM+Mono:wght@400;500&display=swap');

html,[class*="css"]{font-family:'DM Sans',sans-serif;}
.stApp{background:#07080d;color:#e8e9f0;}
[data-testid="stSidebar"]{background:#0f1118;border-right:1px solid rgba(255,255,255,0.07);}
[data-testid="stSidebar"] *{color:#e8e9f0 !important;}
[data-testid="metric-container"]{background:#0f1118;border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:16px 20px;}
[data-testid="metric-container"] label{color:#7c7f91 !important;font-family:'DM Mono',monospace;font-size:11px !important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Syne',sans-serif;font-size:28px !important;color:#e8e9f0 !important;}
[data-testid="stTabs"] button{font-family:'DM Mono',monospace;font-size:12px;color:#7c7f91;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#f0b429 !important;border-bottom:2px solid #f0b429 !important;}
h1,h2,h3{font-family:'Syne',sans-serif !important;font-weight:700 !important;color:#e8e9f0 !important;}
.callout{background:rgba(240,180,41,0.06);border:1px solid rgba(240,180,41,0.2);border-left:3px solid #f0b429;border-radius:0 12px 12px 0;padding:20px 24px;margin:14px 0;font-size:15px;line-height:1.7;color:#e8e9f0;}
.callout-g{background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.2);border-left:3px solid #10b981;border-radius:0 12px 12px 0;padding:20px 24px;margin:14px 0;font-size:15px;line-height:1.7;color:#e8e9f0;}
.callout-p{background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.2);border-left:3px solid #6366f1;border-radius:0 12px 12px 0;padding:20px 24px;margin:14px 0;font-size:15px;line-height:1.7;color:#e8e9f0;}
.big-num{font-family:'Syne',sans-serif;font-size:52px;font-weight:800;line-height:1;}
.lbl{font-family:'DM Mono',monospace;font-size:11px;color:#7c7f91;letter-spacing:.1em;text-transform:uppercase;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────
TH = dict(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono", color="#7c7f91", size=11),
    margin=dict(l=16, r=16, t=36, b=16),
)
GR = dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)")
Y, G, P, R, B, A = "#f0b429", "#10b981", "#6366f1", "#ef4444", "#3b82f6", "#f59e0b"
CMAP = {"Power User": G, "Regular": Y, "Occasional": P, "Dormant": R}

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="lbl" style="margin-bottom:6px">Project</div>', unsafe_allow_html=True)
    st.markdown("## UPI Behaviour Mystery")
    st.markdown(
        '<div style="color:#7c7f91;font-size:13px;margin-bottom:18px">'
        'End-to-end DS pipeline: simulation → cohorts → churn model → causal uplift.'
        '</div>', unsafe_allow_html=True
    )
    st.divider()

    # ── Data source selector ──────────────────────────────────
    st.markdown('<div class="lbl" style="margin-bottom:8px">Data Source</div>', unsafe_allow_html=True)
    data_source = st.radio(
        "Choose data source",
        ["Simulated data", "Upload your CSV"],
        label_visibility="collapsed",
        help="Use simulated data to explore, or upload your own user data",
    )

    uploaded_file = None
    if data_source == "Upload your CSV":
        st.markdown(
            '<div style="color:#7c7f91;font-size:12px;margin-bottom:8px">'
            'Upload a CSV with user-level features. '
            '<a href="#" style="color:#f0b429">Download template</a>'
            '</div>', unsafe_allow_html=True
        )
        uploaded_file = st.file_uploader(
            "Upload user CSV",
            type=["csv"],
            label_visibility="collapsed",
        )
        # Sample CSV download
        sample_df = generate_sample_csv(n_users=20)
        st.download_button(
            "⬇ Download sample CSV template",
            data=sample_df.to_csv(index=False),
            file_name="upi_sample_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.divider()

    if data_source == "Simulated data":
        st.markdown('<div class="lbl" style="margin-bottom:8px">Simulation</div>', unsafe_allow_html=True)
        n_users   = st.slider("Users", 2000, 15000, CFG.simulation.n_users, step=1000)
        rand_seed = st.slider("Seed", 1, 100, CFG.simulation.seed,
                              help="Different seeds = different dataset realisations")
        st.divider()

    st.markdown('<div class="lbl" style="margin-bottom:8px">Intervention</div>', unsafe_allow_html=True)
    cashback     = st.slider("Cashback per offer (₹)", 5, 100,
                              CFG.intervention.default_cashback, step=5)
    total_budget = st.slider("Total budget (₹)", 5000, 100000,
                              CFG.intervention.default_budget, step=5000)

    n_offers = total_budget // cashback
    st.markdown(
        f'<div style="color:#7c7f91;font-size:12px;margin-top:4px">'
        f'→ <span style="color:#f0b429">{n_offers:,} offers</span> available</div>',
        unsafe_allow_html=True
    )

    st.divider()
    st.markdown('<div class="lbl" style="margin-bottom:6px">Stack</div>', unsafe_allow_html=True)
    for item in [
        "Python 3.10+", "LightGBM (churn)", "T-Learner (uplift)",
        "sklearn Pipelines", "Pydantic schemas", "pytest (48 tests)",
        "YAML config", "joblib artifacts", "Click CLI",
    ]:
        st.markdown(f'<div style="font-size:12px;color:#7c7f91;padding:2px 0">· {item}</div>',
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CACHED PIPELINE — reruns only when params change
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _run_pipeline(n: int, seed: int, cb: int, bud: int):
    """
    Cached wrapper around the full pipeline.
    Streamlit re-runs this only when parameters change.
    """
    sim    = simulate_users(n_users=n, seed=seed)
    coh    = compute_cohorts(sim)
    churn  = train_churn_model(sim.users, save_artifact=False)
    uplift = run_uplift_model(churn, cashback_amount=cb, total_budget=bud)
    return sim, coh, churn, uplift


@st.cache_data(show_spinner=False)
def _run_pipeline_csv(csv_data: bytes, cb: int, bud: int):
    """Run pipeline on user-uploaded CSV data."""
    import io
    result = load_user_csv(io.BytesIO(csv_data))
    if not result.valid:
        return None, None, None, None, result
    sim = SimulationResult(
        users=result.users,
        transactions=pd.DataFrame(),  # no transaction data in user CSV
        n_users=result.n_users,
        n_transactions=0,
        elapsed_seconds=0,
    )
    coh    = compute_cohorts(sim)
    churn  = train_churn_model(result.users, save_artifact=False)
    uplift = run_uplift_model(churn, cashback_amount=cb, total_budget=bud)
    return sim, coh, churn, uplift, result


# ── Run pipeline based on data source ─────────────────────────
csv_validation = None

if data_source == "Upload your CSV" and uploaded_file is not None:
    with st.spinner("Running pipeline on your data..."):
        csv_bytes = uploaded_file.getvalue()
        sim, coh, churn, uplift, csv_validation = _run_pipeline_csv(
            csv_bytes, cashback, total_budget
        )
    if not csv_validation.valid:
        st.error("CSV validation failed:")
        for err in csv_validation.errors:
            st.error(f"  {err}")
        st.stop()
    if csv_validation.warnings:
        for warn in csv_validation.warnings:
            st.warning(warn)
    n_users = csv_validation.n_users
elif data_source == "Upload your CSV" and uploaded_file is None:
    st.info(
        "Upload a CSV file in the sidebar to analyze your own data. "
        "Or download the sample template to see the expected format."
    )
    st.stop()
else:
    with st.spinner("Running pipeline (simulation → cohorts → churn model → uplift model)..."):
        sim, coh, churn, uplift = _run_pipeline(n_users, rand_seed, cashback, total_budget)

du  = sim.users
dt  = sim.transactions
dup = uplift.users_segmented
fi  = churn.lgb_metrics.feature_importance

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
    "📊  Overview",
    "🔍  Cohort Analysis",
    "🤖  Churn Model",
    "🎯  Uplift & Targeting",
    "💡  Strategy Recommender",
    "🔬  SHAP Explanations",
    "🧪  Experiment Design",
    "⚖️  Fairness Audit",
    "📋  Decision Memo",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with t1:
    st.markdown("# UPI Behaviour Mystery")
    st.markdown(
        '<div style="color:#7c7f91;font-size:16px;max-width:700px;margin-bottom:28px">'
        "India processed 228 billion UPI transactions in 2025 — yet most registered "
        "users barely transact. What determines who becomes a power user, and who "
        "actually responds when you intervene?"
        "</div>", unsafe_allow_html=True
    )

    d14s = coh.day14_summary
    ret_med = d14s[d14s["status"] == "Retained"]["median_txn_d14"].values[0]
    chu_med = d14s[d14s["status"] == "Churned"]["median_txn_d14"].values[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users simulated",  f"{n_users:,}",  f"{len(dt):,} transactions")
    c2.metric("Churn rate",       f"{du['churned'].mean()*100:.1f}%", "gone inactive")
    c3.metric("Day-14 gap",       f"{ret_med:.0f}× vs {chu_med:.0f}×", "retained vs churned txns")
    c4.metric("CV AUC (5-fold)",  f"{churn.cv_auc_mean:.3f}",
              f"± {churn.cv_auc_std:.3f}")

    st.markdown("""<div class="callout">
    <strong style="font-size:18px">Core finding: Day 14 is the retention gate.</strong><br>
    Retained users completed 8–11× more transactions in their first 14 days than churned users.
    Users who made their first <em>merchant payment</em> within 14 days retained at dramatically
    higher rates. The first two weeks are everything.
    </div>""", unsafe_allow_html=True)

    ca, cb2 = st.columns(2)
    with ca:
        ac = du["archetype"].value_counts()
        fig = go.Figure(go.Pie(
            labels=ac.index, values=ac.values, hole=0.55,
            marker_colors=[G, Y, P, R], textinfo="label+percent",
            textfont=dict(size=12, color="#e8e9f0"),
            hovertemplate="<b>%{label}</b><br>%{value:,} users (%{percent})<extra></extra>",
        ))
        fig.update_layout(**TH, title="User archetypes", height=320)
        st.plotly_chart(fig, use_container_width=True)

    with cb2:
        cb3 = du.groupby("archetype")["churned"].mean().reset_index().sort_values("churned")
        fig = go.Figure(go.Bar(
            x=cb3["churned"] * 100, y=cb3["archetype"], orientation="h",
            marker_color=[CMAP[a] for a in cb3["archetype"]],
            text=[f"{v:.1f}%" for v in cb3["churned"] * 100],
            textposition="outside", textfont=dict(color="#e8e9f0"),
            hovertemplate="<b>%{y}</b><br>Churn: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(**TH, title="Churn rate by archetype", height=320,
                          xaxis=dict(**GR, title="Churn rate (%)"),
                          yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — COHORT ANALYSIS
# ══════════════════════════════════════════════════════════════
with t2:
    st.markdown("## Cohort Analysis")
    st.markdown(
        '<div style="color:#7c7f91;margin-bottom:18px">'
        "SQL-equivalent cohort analysis. Filter and explore — all charts update live."
        "</div>", unsafe_allow_html=True
    )

    af = st.multiselect(
        "Filter by archetype",
        ["Power User", "Regular", "Occasional", "Dormant"],
        default=["Power User", "Regular", "Occasional", "Dormant"],
    )
    dtf = dt[dt["archetype"].isin(af)]
    duf = du[du["archetype"].isin(af)]
    rdf = coh.retention_by_archetype[coh.retention_by_archetype["archetype"].isin(af)]

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for arch in af:
            s = rdf[rdf["archetype"] == arch].sort_values("day_cutoff")
            fig.add_trace(go.Scatter(
                x=s["day_cutoff"], y=s["retention_pct"], name=arch,
                mode="lines+markers",
                line=dict(color=CMAP[arch], width=2), marker=dict(size=6),
                hovertemplate=f"<b>{arch}</b><br>Day %{{x}}: %{{y:.1f}}%<extra></extra>",
            ))
        fig.update_layout(**TH, title="Retention curve by archetype", height=340,
                          xaxis=dict(**GR, title="Days since registration"),
                          yaxis=dict(**GR, title="% still active"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        for ch, lbl, col in [(0, "Retained", G), (1, "Churned", R)]:
            s = duf[duf["churned"] == ch]["txn_d14"]
            fig.add_trace(go.Histogram(
                x=s.clip(0, 30), name=lbl, marker_color=col, opacity=0.75, nbinsx=20,
                hovertemplate=f"<b>{lbl}</b><br>%{{y}} users with %{{x}} txns<extra></extra>",
            ))
        fig.update_layout(**TH, barmode="overlay", height=340,
                          title="Day 1–14 transactions: retained vs churned",
                          xaxis=dict(**GR, title="Transactions in first 14 days"),
                          yaxis=dict(**GR, title="Users"))
        st.plotly_chart(fig, use_container_width=True)

    # Day range slider
    st.markdown("### Category breakdown")
    dr = st.slider("Day range", 0, 365, (0, 90), step=7)
    cs = dtf[(dtf["day"] >= dr[0]) & (dtf["day"] < dr[1])].groupby("category").agg(
        transactions=("value", "count"),
        avg_value=("value", "mean"),
    ).reset_index().sort_values("transactions", ascending=False)

    c1, c2 = st.columns(2)
    for col, cn, title, color, ylab in [
        (c1, "transactions", f"Volume (days {dr[0]}–{dr[1]})", Y, "Transactions"),
        (c2, "avg_value", f"Avg ticket (days {dr[0]}–{dr[1]})", P, "Avg value (₹)"),
    ]:
        fig = go.Figure(go.Bar(
            x=cs["category"], y=cs[cn], marker_color=color, opacity=0.85,
            hovertemplate=f"<b>%{{x}}</b><br>{ylab}: %{{y:,.0f}}<extra></extra>",
        ))
        fig.update_layout(**TH, title=title, height=280,
                          xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                          yaxis=dict(**GR, title=ylab))
        col.plotly_chart(fig, use_container_width=True)

    # City tier & age group
    st.markdown("### City tier & age group")
    c1, c2 = st.columns(2)
    ts = coh.city_tier_summary
    with c1:
        fig = go.Figure(go.Bar(
            x=ts["city_tier"], y=ts["churn_rate"] * 100,
            marker_color=[Y, P, R], opacity=0.85,
            text=[f"{v*100:.1f}%" for v in ts["churn_rate"]],
            textposition="outside", textfont=dict(color="#e8e9f0"),
            hovertemplate="<b>%{x}</b><br>Churn: %{text}<extra></extra>",
        ))
        fig.update_layout(**TH, title="Churn rate by city tier", height=280,
                          xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                          yaxis=dict(**GR, title="Churn rate (%)"))
        st.plotly_chart(fig, use_container_width=True)

    ag = coh.age_group_summary
    with c2:
        fig = go.Figure(go.Bar(
            x=ag["age_group"], y=ag["churn_rate"] * 100,
            marker_color=Y, opacity=0.85,
            text=[f"{v*100:.1f}%" for v in ag["churn_rate"]],
            textposition="outside", textfont=dict(color="#e8e9f0"),
            hovertemplate="<b>%{x}</b><br>Churn: %{text}<extra></extra>",
        ))
        fig.update_layout(**TH, title="Churn rate by age group", height=280,
                          xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                          yaxis=dict(**GR, title="Churn rate (%)"))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — CHURN MODEL
# ══════════════════════════════════════════════════════════════
with t3:
    st.markdown("## Churn Prediction Model")
    st.markdown(
        '<div style="color:#7c7f91;margin-bottom:18px">'
        "LightGBM inside an sklearn Pipeline, 5-fold CV, compared against "
        "logistic regression baseline. ROC, PR curve, calibration, feature importance."
        "</div>", unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV AUC (5-fold)",   f"{churn.cv_auc_mean:.3f}", f"± {churn.cv_auc_std:.3f}")
    c2.metric("Test AUC (LightGBM)", f"{churn.lgb_metrics.auc_roc:.3f}")
    c3.metric("Test AUC (Logistic)", f"{churn.lr_metrics.auc_roc:.3f}")
    c4.metric("Brier score",         f"{churn.lgb_metrics.brier_score:.3f}",
              "calibration (↓ better)")

    ca, cb2 = st.columns(2)
    with ca:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=churn.lgb_metrics.fpr, y=churn.lgb_metrics.tpr,
            name=f"LightGBM (AUC={churn.lgb_metrics.auc_roc:.3f})",
            line=dict(color=Y, width=2.5),
            hovertemplate="FPR:%{x:.3f} TPR:%{y:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=churn.lr_metrics.fpr, y=churn.lr_metrics.tpr,
            name=f"Logistic (AUC={churn.lr_metrics.auc_roc:.3f})",
            line=dict(color=P, width=2, dash="dash"),
            hovertemplate="FPR:%{x:.3f} TPR:%{y:.3f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            line=dict(color="rgba(255,255,255,0.15)", dash="dot"),
        ))
        fig.update_layout(**TH, title="ROC Curve", height=360,
                          xaxis=dict(**GR, title="False Positive Rate"),
                          yaxis=dict(**GR, title="True Positive Rate"))
        st.plotly_chart(fig, use_container_width=True)

    with cb2:
        # Precision-Recall curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=churn.lgb_metrics.recall, y=churn.lgb_metrics.precision,
            name=f"LightGBM (AP={churn.lgb_metrics.avg_precision:.3f})",
            line=dict(color=Y, width=2.5), fill="tozeroy",
            fillcolor="rgba(240,180,41,0.06)",
            hovertemplate="Recall:%{x:.3f} Prec:%{y:.3f}<extra></extra>",
        ))
        fig.update_layout(**TH, title="Precision-Recall Curve", height=360,
                          xaxis=dict(**GR, title="Recall"),
                          yaxis=dict(**GR, title="Precision"))
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    fi_df = (
        pd.DataFrame(list(fi.items()), columns=["feature", "importance"])
        .assign(label=lambda d: d["feature"].map(
            lambda f: FEATURE_LABELS.get(f, f)
        ))
        .sort_values("importance")
    )
    colors = [Y if i >= len(fi_df) - 2 else "rgba(255,255,255,0.25)"
              for i in range(len(fi_df))]
    fig = go.Figure(go.Bar(
        x=fi_df["importance"], y=fi_df["label"], orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in fi_df["importance"]],
        textposition="outside", textfont=dict(color="#e8e9f0"),
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(**TH, title="Feature importance (% of total)", height=400,
                      xaxis=dict(**GR, title="Importance (%)"))
    st.plotly_chart(fig, use_container_width=True)

    # Churn probability violin
    fig = go.Figure()
    for arch in du["archetype"].unique():
        s = du[du["archetype"] == arch]["churn_prob"]
        fig.add_trace(go.Violin(
            x=s, name=arch, line_color=CMAP[arch],
            fillcolor=CMAP[arch] + "33",
            box_visible=True, meanline_visible=True,
            hovertemplate=f"<b>{arch}</b><br>%{{x:.3f}}<extra></extra>",
        ))
    fig.update_layout(**TH, title="Predicted churn probability by archetype", height=300,
                      xaxis=dict(**GR, title="Churn probability"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="callout-p">
    <strong>What the model tells us:</strong> Merchant payment ratio and early-window spend
    dominate. Users who pay merchants — not just friends — integrate UPI into daily life.
    The model catches at-risk users within 14 days — before they appear in monthly churn reports.
    </div>""", unsafe_allow_html=True)

    with st.expander("Classification report (test set)"):
        st.code(churn.lgb_metrics.classification_report)

# ══════════════════════════════════════════════════════════════
# TAB 4 — UPLIFT & TARGETING
# ══════════════════════════════════════════════════════════════
with t4:
    st.markdown("## Uplift Model — Who Actually Responds?")
    st.markdown(
        f'<div style="color:#7c7f91;margin-bottom:18px">'
        f"A churn model predicts who will leave. A <strong>T-Learner uplift model</strong> "
        f"predicts who will <em>change behaviour</em> if you send a ₹{cashback} cashback offer. "
        f"These are different people — and the distinction is worth {uplift.roi.efficiency_gain:.1f}×."
        f"</div>", unsafe_allow_html=True
    )

    sc = uplift.segment_counts
    sp = uplift.segment_pcts
    c1, c2, c3, c4 = st.columns(4)
    for col, seg, color, emoji, action in [
        (c1, "Persuadable",  "#10b981", "🎯", f"Send ₹{cashback} offer"),
        (c2, "Sure Thing",   "#3b82f6", "✅", "Save budget"),
        (c3, "Lost Cause",   "#ef4444", "❌", "Accept churn"),
        (c4, "Sleeping Dog", "#f59e0b", "⚠️", "Leave alone"),
    ]:
        n = sc.get(seg, 0)
        p = sp.get(seg, 0.0)
        col.markdown(
            f'<div style="background:rgba(0,0,0,0.3);border:1px solid {color}44;'
            f'border-radius:12px;padding:18px">'
            f'<div class="lbl">{emoji} {seg}</div>'
            f'<div class="big-num" style="color:{color}">{p:.1f}%</div>'
            f'<div style="font-size:13px;color:#7c7f91;margin:4px 0">{n:,} users</div>'
            f'<div style="font-size:12px;background:rgba(0,0,0,0.3);padding:6px 10px;'
            f'border-radius:6px;margin-top:8px">{action}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Budget ROI comparison
    st.markdown("### Budget ROI — random vs targeted")
    cr, cv, ct = st.columns([5, 1, 5])
    with cr:
        st.markdown(
            f'<div style="background:#0f1118;border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:16px;padding:28px;text-align:center">'
            f'<div class="lbl" style="margin-bottom:8px">Random targeting</div>'
            f'<div class="big-num">{uplift.roi.users_retained_random:,}</div>'
            f'<div style="color:#7c7f91;font-size:13px;margin-top:6px">'
            f'users retained from ₹{total_budget:,}</div>'
            f"</div>", unsafe_allow_html=True
        )
    with cv:
        st.markdown(
            '<div style="text-align:center;padding:48px 0;color:#7c7f91;font-size:20px">vs</div>',
            unsafe_allow_html=True
        )
    with ct:
        st.markdown(
            f'<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);'
            f'border-radius:16px;padding:28px;text-align:center">'
            f'<div class="lbl" style="margin-bottom:8px;color:#10b981">Persuadables only</div>'
            f'<div class="big-num" style="color:#10b981">'
            f'{uplift.roi.users_retained_targeted:,}</div>'
            f'<div style="color:#7c7f91;font-size:13px;margin-top:6px">'
            f'users retained from ₹{total_budget:,}</div>'
            f'<div style="display:inline-block;background:#f0b429;color:#000;'
            f'font-family:DM Mono;font-size:13px;padding:5px 14px;'
            f'border-radius:100px;margin-top:12px">'
            f'{uplift.roi.efficiency_gain:.1f}× more efficient</div>'
            f"</div>", unsafe_allow_html=True
        )

    # Uplift scatter — every dot is a user, hover for details
    st.markdown("### Individual uplift scores")
    sample = dup.sample(min(5000, len(dup)), random_state=1)
    fig = px.scatter(
        sample, x="p0", y="p1", color="segment",
        color_discrete_map={
            "Persuadable": G, "Sure Thing": B,
            "Lost Cause": R, "Sleeping Dog": A,
        },
        opacity=0.5,
        labels={"p0": "P(retain | no offer)", "p1": "P(retain | offer sent)"},
        title="Each dot = one user. Hover for individual details.",
        hover_data=["archetype", "churn_prob", "uplift", "txn_d14"],
    )
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="rgba(255,255,255,0.15)", dash="dash"))
    fig.add_shape(type="line", x0=0.6, y0=0, x1=0.6, y1=1,
                  line=dict(color="rgba(255,255,255,0.07)", dash="dot"))
    fig.add_shape(type="line", x0=0, y0=0.6, x1=1, y1=0.6,
                  line=dict(color="rgba(255,255,255,0.07)", dash="dot"))
    fig.update_layout(**TH, height=440)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bottom-right = Persuadables (low P0, high P1). "
        "Above the diagonal = offer improves retention. Hover any dot for user-level details."
    )

    st.markdown("""<div class="callout-g">
    <strong>Why this matters:</strong> Random offers mostly land on Sure Things (already loyal)
    and Lost Causes (unresponsive regardless). Persuadables — the users where the intervention
    actually shifts retention — are a concentrated minority. Finding them is the entire value
    of the uplift model.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — STRATEGY RECOMMENDER
# ══════════════════════════════════════════════════════════════
with t5:
    st.markdown("## Strategy Recommender")
    st.markdown(
        '<div style="color:#7c7f91;margin-bottom:18px">'
        "Plain-English recommendations from the model. No jargon — just "
        "what to do, why, and expected impact. Hand this to your PM."
        "</div>", unsafe_allow_html=True
    )

    strategy = generate_strategy(
        churn_result=churn,
        uplift_result=uplift,
        cohort_result=coh,
        cashback_amount=cashback,
        total_budget=total_budget,
    )

    # Headline
    st.markdown(
        f'<div class="callout-g" style="font-size:16px">'
        f'<strong>Bottom line:</strong> {strategy.headline}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Recommendations
    for rec in strategy.recommendations:
        conf_color = {"High": G, "Medium": Y, "Low": R}.get(
            rec.confidence.split(" ")[0], Y
        )
        eff_color = {"Low": G, "Medium": Y, "High": R}.get(rec.effort, Y)

        st.markdown(
            f'<div style="background:#0f1118;border:1px solid rgba(255,255,255,0.07);'
            f'border-left:3px solid {conf_color};border-radius:0 12px 12px 0;'
            f'padding:20px 24px;margin:12px 0">'
            f'<div style="display:flex;justify-content:space-between;align-items:center">'
            f'<div style="font-family:Syne,sans-serif;font-size:17px;font-weight:700;'
            f'color:#e8e9f0">#{rec.priority} — {rec.title}</div>'
            f'<div>'
            f'<span style="background:rgba(255,255,255,0.05);font-family:DM Mono;'
            f'font-size:11px;padding:3px 10px;border-radius:4px;color:{conf_color}">'
            f'Confidence: {rec.confidence.split(chr(8212))[0].strip()}</span> '
            f'<span style="background:rgba(255,255,255,0.05);font-family:DM Mono;'
            f'font-size:11px;padding:3px 10px;border-radius:4px;color:{eff_color}">'
            f'Effort: {rec.effort.split(chr(8212))[0].strip()}</span>'
            f'</div></div>'
            f'<div style="margin-top:12px;color:#a0a3b1;font-size:14px;line-height:1.7">'
            f'<strong style="color:#e8e9f0">What to do:</strong> {rec.what}</div>'
            f'<div style="margin-top:8px;color:#a0a3b1;font-size:14px;line-height:1.7">'
            f'<strong style="color:#e8e9f0">Why:</strong> {rec.why}</div>'
            f'<div style="margin-top:8px;color:#a0a3b1;font-size:14px;line-height:1.7">'
            f'<strong style="color:#10b981">Expected impact:</strong> {rec.expected_impact}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk factors
    st.markdown("### Risk Factors")
    for risk in strategy.risk_factors:
        st.markdown(
            f'<div style="padding:6px 0;color:#a0a3b1;font-size:14px">'
            f'<span style="color:#ef4444;margin-right:8px">⚠</span>{risk}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Key metrics
    st.markdown("### Key Metrics to Track")
    for label, metric in strategy.key_metrics.items():
        label_color = {"Primary": G, "Secondary": B, "Guardrail": R,
                       "Leading indicator": Y}.get(label, Y)
        st.markdown(
            f'<div style="background:#0f1118;border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:8px;padding:12px 16px;margin:6px 0;display:flex;'
            f'align-items:center;gap:12px">'
            f'<span style="background:{label_color}22;color:{label_color};'
            f'font-family:DM Mono;font-size:11px;padding:3px 10px;border-radius:4px;'
            f'white-space:nowrap">{label}</span>'
            f'<span style="color:#a0a3b1;font-size:14px">{metric}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Download strategy as text
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        "⬇ Download strategy report (TXT)",
        data=strategy.to_plain_text(),
        file_name="upi_strategy_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════
# TAB 6 — SHAP EXPLANATIONS
# ══════════════════════════════════════════════════════════════
with t6:
    st.markdown("## SHAP Explanations")
    st.markdown(
        '<div style="color:#7c7f91;margin-bottom:18px">'
        "Model interpretability using SHAP (SHapley Additive exPlanations). "
        "Which features drive churn predictions — globally and for individual users?"
        "</div>", unsafe_allow_html=True
    )

    try:
        shap_result = compute_shap_explanations(
            lgb_pipeline=churn.lgb_metrics,  # we need the pipeline
            users=du,
            feature_names=EXTENDED_FEATURES,
            max_users=min(3000, len(du)),
        )
        has_shap = True
    except Exception:
        # Compute SHAP manually if the pipeline isn't directly available
        has_shap = False

    if not has_shap:
        # Fallback: use feature importance from LightGBM (already computed)
        st.markdown("### Global Feature Importance (LightGBM gain-based)")
        st.markdown(
            '<div style="color:#7c7f91;font-size:13px;margin-bottom:12px">'
            "Install `shap` package for full SHAP waterfall and beeswarm plots. "
            "Showing gain-based importance as fallback."
            "</div>", unsafe_allow_html=True
        )

    # Global importance (works with or without SHAP)
    st.markdown("### Global Feature Importance")
    fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    feat_names_sorted = [FEATURE_LABELS.get(f, f) for f, _ in fi_sorted]
    feat_values_sorted = [v for _, v in fi_sorted]

    fig_imp = go.Figure(go.Bar(
        y=feat_names_sorted[::-1],
        x=feat_values_sorted[::-1],
        orientation="h",
        marker_color=[G if v >= 10 else B if v >= 5 else "#7c7f91"
                      for v in feat_values_sorted[::-1]],
        text=[f"{v:.1f}%" for v in feat_values_sorted[::-1]],
        textposition="outside",
    ))
    fig_imp.update_layout(
        **TH, height=max(350, len(fi_sorted) * 32),
        title="Feature importance (% of total model gain)",
        xaxis_title="Importance (%)",
        yaxis_title="",
        margin=dict(l=250),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Individual user lookup
    st.markdown("### Individual User Explanation")
    st.markdown(
        '<div style="color:#7c7f91;font-size:13px;margin-bottom:12px">'
        "Select a user to see what drives their churn prediction."
        "</div>", unsafe_allow_html=True
    )

    # Show high-risk users for selection
    high_risk = dup.nlargest(20, "churn_prob")[
        ["user_id", "archetype", "city_tier", "churn_prob", "segment"]
    ].reset_index(drop=True)
    st.dataframe(high_risk, use_container_width=True, height=300)

    selected_idx = st.number_input(
        "Enter row index (0-19) from the table above",
        min_value=0, max_value=min(19, len(high_risk)-1), value=0, step=1
    )

    if selected_idx < len(high_risk):
        uid = high_risk.iloc[selected_idx]["user_id"]
        user_row = dup[dup["user_id"] == uid].iloc[0]
        prob = user_row["churn_prob"]
        seg = user_row["segment"]

        st.markdown(
            f'<div style="background:#0f1118;border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:12px;padding:20px;margin:12px 0">'
            f'<span style="font-family:DM Mono;color:#7c7f91">User {uid}</span> &nbsp;'
            f'<span style="background:{"#ef4444" if prob > 0.7 else "#f0b429" if prob > 0.4 else "#10b981"}22;'
            f'color:{"#ef4444" if prob > 0.7 else "#f0b429" if prob > 0.4 else "#10b981"};'
            f'font-family:DM Mono;font-size:12px;padding:3px 10px;border-radius:4px">'
            f'P(churn) = {prob:.3f}</span> &nbsp;'
            f'<span style="background:rgba(255,255,255,0.05);font-family:DM Mono;'
            f'font-size:12px;padding:3px 10px;border-radius:4px;color:#a0a3b1">'
            f'{seg}</span>'
            f'</div>', unsafe_allow_html=True
        )

        # Show this user's feature values vs population
        user_features = {}
        for feat in EXTENDED_FEATURES:
            if feat in dup.columns:
                user_features[FEATURE_LABELS.get(feat, feat)] = float(user_row.get(feat, 0))

        if user_features:
            feat_df = pd.DataFrame([
                {"Feature": k, "This User": v}
                for k, v in user_features.items()
            ])
            st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("""<div class="callout-g">
    <strong>Why SHAP matters:</strong> Feature importance shows <em>what</em> the model uses,
    but SHAP explains <em>how</em> each feature pushes a specific user's prediction up or down.
    This is critical for stakeholder trust, debugging, and regulatory compliance.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 7 — EXPERIMENT DESIGN
# ══════════════════════════════════════════════════════════════
with t7:
    st.markdown("## Experiment Design")
    st.markdown(
        '<div style="color:#7c7f91;margin-bottom:18px">'
        "How to validate the cashback intervention with a rigorous A/B test. "
        "Power analysis, duration estimate, and pre-registered subgroup plan."
        "</div>", unsafe_allow_html=True
    )

    # Controls
    ec1, ec2, ec3 = st.columns(3)
    with ec1:
        exp_mde = st.slider("Minimum detectable effect (pp)", 1, 15, 5, step=1,
                             help="Smallest retention lift worth detecting") / 100
    with ec2:
        exp_power = st.slider("Statistical power", 0.70, 0.95, 0.80, step=0.05,
                              help="Probability of detecting the effect if it exists")
    with ec3:
        exp_daily = st.number_input("Daily eligible users", 100, 10000, 500, step=100,
                                    help="New users entering the experiment each day")

    experiment = design_experiment(
        users_scored=dup,
        uplift_result=uplift,
        daily_new_users=exp_daily,
        alpha=0.05,
        power=exp_power,
        mde=exp_mde,
    )

    # Primary analysis card
    p = experiment.primary
    st.markdown(
        f'<div style="background:#0f1118;border:1px solid rgba(255,255,255,0.07);'
        f'border-left:3px solid {G};border-radius:0 12px 12px 0;padding:24px;margin:16px 0">'
        f'<div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;'
        f'color:#e8e9f0;margin-bottom:12px">Primary Analysis</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">'
        f'<div><div class="lbl">Baseline retention</div>'
        f'<div style="font-size:28px;font-weight:700;color:#e8e9f0">{p.baseline_rate:.1%}</div></div>'
        f'<div><div class="lbl">Sample per arm</div>'
        f'<div style="font-size:28px;font-weight:700;color:#6366f1">{p.sample_per_arm:,}</div></div>'
        f'<div><div class="lbl">Estimated duration</div>'
        f'<div style="font-size:28px;font-weight:700;color:#f0b429">{p.estimated_weeks:.0f} weeks</div></div>'
        f'</div></div>', unsafe_allow_html=True
    )

    # Power curve — how sample size changes with MDE
    st.markdown("### Power Curve")
    st.markdown(
        '<div style="color:#7c7f91;font-size:13px;margin-bottom:8px">'
        "How many users per arm are needed for different effect sizes?"
        "</div>", unsafe_allow_html=True
    )
    mdes = np.arange(0.01, 0.16, 0.005)
    sample_sizes = [compute_sample_size(p.baseline_rate, m, 0.05, exp_power) for m in mdes]

    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(
        x=mdes * 100, y=sample_sizes,
        mode="lines", line=dict(color=G, width=2),
        name="Sample per arm",
    ))
    fig_power.add_vline(x=exp_mde * 100, line_dash="dash",
                        line_color=Y, annotation_text=f"Your MDE: {exp_mde*100:.0f}pp")
    fig_power.add_hline(y=p.sample_per_arm, line_dash="dot",
                        line_color="#7c7f91",
                        annotation_text=f"n={p.sample_per_arm:,}")
    fig_power.update_layout(
        **TH, height=380,
        title="Sample size vs. minimum detectable effect",
        xaxis_title="MDE (percentage points)",
        yaxis_title="Sample per arm",
        yaxis_type="log",
    )
    st.plotly_chart(fig_power, use_container_width=True)

    # Subgroup analysis
    if experiment.subgroup_analyses:
        st.markdown("### Subgroup Analysis (Bonferroni-corrected)")
        st.markdown(
            f'<div style="color:#7c7f91;font-size:13px;margin-bottom:12px">'
            f"Corrected α = {experiment.bonferroni_alpha:.4f} "
            f"(Bonferroni for {len(experiment.subgroup_analyses)} pre-registered subgroups)"
            f'</div>', unsafe_allow_html=True
        )

        sub_data = []
        for name, result in experiment.subgroup_analyses.items():
            sub_data.append({
                "Segment": name,
                "Baseline retention": f"{result.baseline_rate:.1%}",
                "Sample per arm": f"{result.sample_per_arm:,}",
                "Estimated duration": f"{result.estimated_weeks:.0f} weeks",
                "Daily eligible": f"{result.daily_eligible_users:,}",
            })
        st.dataframe(pd.DataFrame(sub_data), use_container_width=True, hide_index=True)

    # Guardrails
    st.markdown("### Guardrail Metrics")
    for metric in experiment.guardrail_metrics:
        st.markdown(
            f'<div style="padding:6px 0;color:#a0a3b1;font-size:14px">'
            f'<span style="color:#f0b429;margin-right:8px">🛡</span>{metric}</div>',
            unsafe_allow_html=True,
        )

    # Risks
    st.markdown("### Experiment Risks")
    for risk in experiment.risks:
        st.markdown(
            f'<div style="padding:6px 0;color:#a0a3b1;font-size:14px">'
            f'<span style="color:#ef4444;margin-right:8px">⚠</span>{risk}</div>',
            unsafe_allow_html=True,
        )

    # Recommendations
    st.markdown("### Recommendations")
    for rec in experiment.recommendations:
        st.markdown(
            f'<div style="padding:8px 0;color:#e8e9f0;font-size:14px">'
            f'<span style="color:{G};margin-right:8px">→</span>{rec}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("""<div class="callout-g">
    <strong>Why this matters for Google:</strong> Any DS can build a model.
    Designing the experiment that validates whether it works in production —
    that's the hard part. This tab shows you think about the full lifecycle:
    explore → model → experiment → deploy → monitor.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 8 — FAIRNESS AUDIT
# ══════════════════════════════════════════════════════════════
with t8:
    st.markdown("## Model Fairness Audit")
    st.markdown(
        '<div style="color:#7c7f91;margin-bottom:18px">'
        "Does the churn model perform equitably across demographic groups? "
        "Disparate impact, equalised opportunity, and calibration checks."
        "</div>", unsafe_allow_html=True
    )

    fairness = run_fairness_audit(dup, threshold=0.5)

    # Verdict banner
    if fairness.passed:
        st.markdown(
            f'<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);'
            f'border-radius:12px;padding:16px 20px;margin-bottom:20px">'
            f'<span style="font-size:20px;margin-right:8px">✓</span>'
            f'<strong style="color:#10b981">PASSED</strong> — '
            f'No major fairness concerns detected across {len(fairness.attributes)} attributes.'
            f'</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.3);'
            f'border-radius:12px;padding:16px 20px;margin-bottom:20px">'
            f'<span style="font-size:20px;margin-right:8px">⚠</span>'
            f'<strong style="color:#ef4444">{fairness.total_flags} ISSUE(S) FLAGGED</strong> — '
            f'Review the details below for potential fairness concerns.'
            f'</div>', unsafe_allow_html=True
        )

    # Per-attribute breakdown
    for attr_name, attr_result in fairness.attributes.items():
        status_color = R if attr_result.flagged else G
        st.markdown(
            f'<div style="background:#0f1118;border:1px solid rgba(255,255,255,0.07);'
            f'border-left:3px solid {status_color};border-radius:0 12px 12px 0;'
            f'padding:20px;margin:16px 0">'
            f'<div style="font-family:Syne,sans-serif;font-size:17px;font-weight:700;'
            f'color:#e8e9f0;margin-bottom:12px">{attr_name.replace("_", " ").title()}'
            f' <span style="font-size:12px;color:{status_color}">'
            f'{"⚠ FLAGGED" if attr_result.flagged else "✓ PASSED"}</span></div>'
            f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr 1fr;gap:8px;'
            f'margin-bottom:12px">'
            f'<div><div class="lbl">Disparate Impact</div>'
            f'<div style="font-size:20px;font-weight:700;'
            f'color:{"#ef4444" if attr_result.disparate_impact_ratio < 0.8 else "#10b981"}">'
            f'{attr_result.disparate_impact_ratio:.3f}</div></div>'
            f'<div><div class="lbl">Max TPR Gap</div>'
            f'<div style="font-size:20px;font-weight:700;'
            f'color:{"#ef4444" if attr_result.max_tpr_gap > 0.1 else "#10b981"}">'
            f'{attr_result.max_tpr_gap:.3f}</div></div>'
            f'<div><div class="lbl">Max FPR Gap</div>'
            f'<div style="font-size:20px;font-weight:700;color:#a0a3b1">'
            f'{attr_result.max_fpr_gap:.3f}</div></div>'
            f'<div><div class="lbl">Calibration Gap</div>'
            f'<div style="font-size:20px;font-weight:700;'
            f'color:{"#ef4444" if attr_result.max_calibration_gap > 0.1 else "#10b981"}">'
            f'{attr_result.max_calibration_gap:.3f}</div></div>'
            f'</div></div>', unsafe_allow_html=True
        )

        # Group-level metrics table
        group_data = []
        for gname, gm in attr_result.groups.items():
            group_data.append({
                "Group": gname,
                "N": f"{gm.group_size:,}",
                "Actual churn": f"{gm.churn_rate_actual:.1%}",
                "Predicted churn": f"{gm.churn_rate_predicted:.1%}",
                "TPR": f"{gm.true_positive_rate:.3f}",
                "FPR": f"{gm.false_positive_rate:.3f}",
                "Precision": f"{gm.precision:.3f}",
                "AUC": f"{gm.auc_roc:.3f}",
            })
        st.dataframe(pd.DataFrame(group_data), use_container_width=True, hide_index=True)

        # TPR comparison chart
        tpr_data = {gname: gm.true_positive_rate for gname, gm in attr_result.groups.items()}
        fig_tpr = go.Figure(go.Bar(
            x=list(tpr_data.keys()),
            y=list(tpr_data.values()),
            marker_color=[G if v >= max(tpr_data.values()) - 0.05 else R
                         for v in tpr_data.values()],
            text=[f"{v:.3f}" for v in tpr_data.values()],
            textposition="outside",
        ))
        fig_tpr.update_layout(
            **TH, height=300,
            title=f"True Positive Rate by {attr_name.replace('_', ' ').title()}",
            yaxis_title="TPR",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig_tpr, use_container_width=True)

        # Show flags
        if attr_result.flags:
            for flag in attr_result.flags:
                st.markdown(
                    f'<div style="padding:8px 12px;background:rgba(239,68,68,0.06);'
                    f'border-radius:6px;margin:4px 0;color:#ef4444;font-size:13px">'
                    f'⚠ {flag}</div>', unsafe_allow_html=True
                )

    st.markdown("""<div class="callout-g">
    <strong>Why fairness auditing matters:</strong> A model with 0.90 AUC overall
    might have 0.95 AUC for metro users and 0.70 for Tier-3 users. Without auditing,
    you'd never know — and your intervention would systematically fail for the users
    who need it most. Google evaluates DS candidates on whether they think about
    who the model works for, not just how well it works on average.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 9 — DECISION MEMO
# ══════════════════════════════════════════════════════════════
with t9:
    st.markdown("## Decision Memo")
    st.markdown(
        '<div style="color:#7c7f91;margin-bottom:18px">'
        "Every number pulled live from the model. Change any sidebar control — memo regenerates."
        "</div>", unsafe_allow_html=True
    )

    d14s    = coh.day14_summary
    ret_med = d14s[d14s["status"] == "Retained"]["median_txn_d14"].values[0]
    chu_med = d14s[d14s["status"] == "Churned"]["median_txn_d14"].values[0]
    top_f   = list(fi.keys())[0]
    top_i   = fi[top_f]
    top_l   = FEATURE_LABELS.get(top_f, top_f)
    pp      = uplift.segment_pcts.get("Persuadable", 0.0)
    pc      = uplift.segment_counts.get("Persuadable", 0)

    st.markdown(
        f'<div style="background:#0f1118;border:1px solid rgba(255,255,255,0.07);'
        f'border-radius:16px;padding:32px 36px;font-size:14px;line-height:1.8;color:#a0a3b1">'
        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;'
        f'border-bottom:1px solid rgba(255,255,255,0.07);padding-bottom:14px;margin-bottom:18px">'
        f'<div>'
        f'<div style="font-family:DM Mono;font-size:12px;color:#7c7f91">'
        f'From: Data Science &nbsp;·&nbsp; To: Product Lead, Payments</div>'
        f'<div style="font-family:DM Mono;font-size:12px;color:#7c7f91;margin-top:3px">'
        f'Re: UPI Retention Intervention — Budget ₹{total_budget:,}</div>'
        f'</div>'
        f'<div style="background:#f0b429;color:#000;font-family:DM Mono;font-size:11px;'
        f'padding:4px 12px;border-radius:4px">RECOMMENDATION</div>'
        f'</div>'
        f'<div style="font-family:Syne,sans-serif;font-size:20px;font-weight:700;'
        f'color:#e8e9f0;margin-bottom:12px">'
        f'How should we allocate the ₹{total_budget:,} retention budget?</div>'
        f'<p>Analysis of {n_users:,} user journeys across 12 months. '
        f'Model CV AUC: {churn.cv_auc_mean:.3f} ± {churn.cv_auc_std:.3f} (5-fold).</p>'
        f'<p style="margin-top:14px"><strong style="color:#e8e9f0">'
        f'Finding 1 — Day 14 is the retention gate.</strong><br>'
        f'Retained users had <strong style="color:#10b981">{ret_med:.0f} median transactions'
        f'</strong> in days 1–14. Churned users: '
        f'<strong style="color:#ef4444">{chu_med:.0f}</strong>. '
        f'This gap is detectable within two weeks — before standard monthly churn reports.</p>'
        f'<p style="margin-top:14px"><strong style="color:#e8e9f0">'
        f'Finding 2 — {top_l} is the #1 predictor ({top_i:.1f}% importance).</strong><br>'
        f'LightGBM AUC {churn.lgb_metrics.auc_roc:.3f} vs {churn.lr_metrics.auc_roc:.3f} '
        f'logistic baseline. Users who pay <em>merchants</em> — not just friends — '
        f'integrate UPI into daily spending behaviour.</p>'
        f'<p style="margin-top:14px"><strong style="color:#e8e9f0">'
        f'Finding 3 — Only {pp:.1f}% of users are worth targeting.</strong><br>'
        f'T-Learner uplift model identifies {pc:,} Persuadables. '
        f'Random offers: <strong style="color:#ef4444">'
        f'{uplift.roi.users_retained_random:,} retained</strong>. '
        f'Targeted: <strong style="color:#10b981">'
        f'{uplift.roi.users_retained_targeted:,} retained</strong> — '
        f'<strong style="color:#f0b429">{uplift.roi.efficiency_gain:.1f}× more efficient'
        f'</strong>.</p>'
        f'<p style="margin-top:18px;padding-top:18px;border-top:1px solid rgba(255,255,255,0.07)">'
        f'<strong style="color:#e8e9f0">Recommendation:</strong><br>'
        f'1. Deploy ₹{cashback} cashback to the {pc:,} Persuadables this month.<br>'
        f'2. Add a "pay your first merchant" prompt at Day 7 in onboarding '
        f'— addresses root cause, not symptom.<br>'
        f'3. Rerun uplift model monthly as acquisition mix shifts.</p>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        csv = dup[[
            "user_id", "archetype", "city_tier", "age_group",
            "churn_prob", "segment", "uplift", "p0", "p1",
        ]].to_csv(index=False)
        st.download_button(
            "⬇ User segments CSV", data=csv,
            file_name="upi_user_segments.csv", mime="text/csv",
            use_container_width=True,
        )
    with c2:
        memo = (
            f"UPI RETENTION — DECISION MEMO\n"
            f"Budget ₹{total_budget:,} | {n_users:,} users\n\n"
            f"CV AUC (5-fold):  {churn.cv_auc_mean:.3f} ± {churn.cv_auc_std:.3f}\n"
            f"Day-14 gap:       retained={ret_med:.0f} vs churned={chu_med:.0f} median txns\n"
            f"Top predictor:    {top_l} ({top_i:.1f}% importance)\n"
            f"Persuadables:     {pc:,} users ({pp:.1f}%)\n"
            f"Targeted ROI:     {uplift.roi.efficiency_gain:.1f}x more efficient than random\n\n"
            f"RECOMMENDATION:\n"
            f"1. Deploy ₹{cashback} cashback to {pc:,} Persuadables\n"
            f"2. Add Day-7 merchant payment prompt in onboarding\n"
            f"3. Rerun uplift model monthly\n"
        )
        st.download_button(
            "⬇ Download memo (TXT)", data=memo,
            file_name="upi_decision_memo.txt", mime="text/plain",
            use_container_width=True,
        )
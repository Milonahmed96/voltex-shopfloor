"""
Voltex ShopFloor Analyst
app.py — Streamlit Dashboard

Store manager dashboard showing daily AI-generated action briefs
alongside performance metrics, trends, and network overview.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
from data_pipeline import load_data, compute_network_benchmarks, build_store_snapshot
from analyst import ShopFloorAnalyst, StoreActionBrief

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Voltex ShopFloor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background: #0F1923; }
    .main .block-container { padding-top: 0; max-width: 1400px; }

    .hero {
        background: linear-gradient(135deg, #0F6E56 0%, #1D9E75 50%, #0F6E56 100%);
        padding: 2rem 2.5rem 1.5rem;
        margin: -1rem -1rem 1.5rem -1rem;
        border-bottom: 3px solid #5DCAA5;
    }
    .hero h1 { color: white; font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem 0; }
    .hero p  { color: #E1F5EE; font-size: 0.9rem; margin: 0; opacity: 0.85; }

    .metric-card {
        background: #1A2535;
        border: 1px solid #253347;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .metric-label { font-size: 0.72rem; color: #5DCAA5; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 0.3rem; }
    .metric-value { font-size: 1.6rem; font-weight: 600; color: #E1F5EE; }
    .metric-delta { font-size: 0.82rem; margin-top: 0.2rem; }
    .delta-pos { color: #5DCAA5; }
    .delta-neg { color: #F7C1C1; }
    .delta-neu { color: #888; }

    .brief-card {
        background: #1A2535;
        border: 1px solid #253347;
        border-radius: 10px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
    }

    .health-green { background:#0D3D2E; color:#5DCAA5; padding:4px 14px;
                    border-radius:20px; font-size:0.82rem; font-weight:600;
                    border:1px solid #1D9E75; display:inline-block; }
    .health-amber { background:#2D1E0A; color:#FAC775; padding:4px 14px;
                    border-radius:20px; font-size:0.82rem; font-weight:600;
                    border:1px solid #BA7517; display:inline-block; }
    .health-red   { background:#2D1515; color:#F7C1C1; padding:4px 14px;
                    border-radius:20px; font-size:0.82rem; font-weight:600;
                    border:1px solid #E24B4A; display:inline-block; }

    .action-card {
        background: #141E2B;
        border-left: 3px solid #1D9E75;
        border-radius: 0 8px 8px 0;
        padding: 0.85rem 1rem;
        margin-bottom: 0.75rem;
    }
    .action-card.amber { border-left-color: #BA7517; }
    .action-card.red   { border-left-color: #E24B4A; }
    .action-rank  { font-size: 0.72rem; color: #5DCAA5; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 0.06em; }
    .action-title { font-size: 1rem; font-weight: 600; color: #E1F5EE; margin: 0.2rem 0 0.5rem; }
    .action-label { font-size: 0.72rem; color: #888; text-transform: uppercase;
                    letter-spacing: 0.06em; margin-top: 0.4rem; }
    .action-text  { font-size: 0.88rem; color: #B4B2A9; line-height: 1.5; margin: 0.1rem 0; }

    .watch-item { background:#1A1A2E; border:1px solid #2A2A4A; border-radius:6px;
                  padding:0.5rem 0.75rem; font-size:0.85rem; color:#FAC775;
                  margin-bottom:0.4rem; }

    .peer-row { display:flex; justify-content:space-between; padding:0.4rem 0;
                border-bottom:1px solid #253347; font-size:0.85rem; }
    .peer-name { color:#B4B2A9; }
    .peer-val  { color:#5DCAA5; font-weight:500; }

    .section-label { font-size:0.72rem; color:#5DCAA5; text-transform:uppercase;
                     letter-spacing:0.08em; margin-bottom:0.5rem; margin-top:1rem; }

    div[data-testid="stSelectbox"] > div { background: #1A2535 !important; }
    .stSelectbox label { color: #5DCAA5 !important; font-size: 0.8rem !important; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_all_data():
    df, stores, promotions = load_data()
    benchmarks = compute_network_benchmarks(df, stores)
    return df, stores, promotions, benchmarks

@st.cache_resource
def get_analyst():
    return ShopFloorAnalyst()

df, stores, promotions, benchmarks = load_all_data()
analyst = get_analyst()

# Available dates and stores
all_dates  = sorted(df["date_str"].unique().tolist())
store_opts = {f"{s['name']} ({s['id']})": s["id"] for s in stores}

# ─────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>⚡ Voltex ShopFloor Analyst</h1>
    <p>AI-powered daily action briefs for store managers · 20 stores · 12-week dataset</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — CONTROLS
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚡ ShopFloor Analyst")
    st.divider()

    selected_store_label = st.selectbox(
        "Select store",
        options=list(store_opts.keys()),
        index=5,  # S06 Reading by default
    )
    selected_store_id = store_opts[selected_store_label]

    selected_date = st.selectbox(
        "Select date",
        options=all_dates,
        index=all_dates.index("2026-02-16") if "2026-02-16" in all_dates else len(all_dates) // 2,
    )

    generate = st.button("Generate brief ↗", type="primary", use_container_width=True)

    st.divider()
    st.markdown("""
    **About**
    ShopFloor Analyst uses Claude Sonnet to reason over store performance data and generate
    daily priority action briefs.

    **Embedded problems:**
    - S06 Reading: VoltCare attach collapse (wk 5+)
    - S03 Croydon: Browse-not-buy pattern
    - S11 Coventry: Outperforming small store
    - S08 vs S09: Promo execution gap
    - S15 Newcastle: Weekend NPS/staffing
    - S17 Sheffield: Recovery store
    """)
    st.divider()
    st.caption("Voltex Retail is a fictional business. All data is synthetic.")

# ─────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📋 Store Brief", "📊 Network Overview", "📈 Trends"])

# ─────────────────────────────────────────────
# TAB 1 — STORE BRIEF
# ─────────────────────────────────────────────

with tab1:
    store_info = next(s for s in stores if s["id"] == selected_store_id)

    # Build snapshot
    try:
        snapshot = build_store_snapshot(
            selected_store_id, selected_date, df, stores, promotions, benchmarks
        )
    except Exception as e:
        st.error(f"No data available for {selected_store_label} on {selected_date}: {e}")
        st.stop()

    # ── Store header
    col_info, col_health = st.columns([3, 1])
    with col_info:
        st.markdown(f"### {store_info['name']}")
        st.markdown(
            f"<span style='color:#888;font-size:0.9rem;'>"
            f"{store_info['city']} · {store_info['region']} · "
            f"{store_info['tier'].title()} store · Week {snapshot.week_num}</span>",
            unsafe_allow_html=True,
        )
    with col_health:
        if snapshot.anomalies:
            st.markdown(
                f"<div style='text-align:right;margin-top:0.5rem;'>"
                f"<span style='color:#FAC775;font-size:0.85rem;'>"
                f"⚠ {len(snapshot.anomalies)} anomaly{'s' if len(snapshot.anomalies) > 1 else ''} detected"
                f"</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Metric cards row
    m1, m2, m3, m4, m5 = st.columns(5)

    def delta_html(val: float, is_pct: bool = True, invert: bool = False) -> str:
        display = f"{val:+.1%}" if is_pct else f"{val:+.0f}"
        good    = val > 0 if not invert else val < 0
        cls     = "delta-pos" if good else "delta-neg" if val != 0 else "delta-neu"
        arrow   = "↑" if val > 0 else "↓" if val < 0 else "→"
        return f'<span class="{cls}">{arrow} {display} vs network</span>'

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Revenue</div>
            <div class="metric-value">£{snapshot.revenue_today/1000:.0f}k</div>
            <div class="metric-delta">{delta_html(snapshot.revenue_vs_network)}</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Footfall</div>
            <div class="metric-value">{snapshot.footfall_today:,}</div>
            <div class="metric-delta">{delta_html(snapshot.footfall_vs_network)}</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Conversion</div>
            <div class="metric-value">{snapshot.conversion_today:.1%}</div>
            <div class="metric-delta">{delta_html(snapshot.conversion_vs_network)}</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">NPS</div>
            <div class="metric-value">{snapshot.nps_today}</div>
            <div class="metric-delta">{delta_html(snapshot.nps_vs_baseline)}</div>
        </div>""", unsafe_allow_html=True)

    with m5:
        staff_color = "delta-neg" if snapshot.staffing_ratio_today < 0.85 else "delta-pos"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Staffing</div>
            <div class="metric-value">{snapshot.staffing_ratio_today:.0%}</div>
            <div class="metric-delta">
                <span class="{staff_color}">of scheduled hours</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Category VoltCare attach chart
    st.markdown('<div class="section-label">VoltCare attach rate by category vs network</div>',
                unsafe_allow_html=True)

    cat_names    = [c.category for c in snapshot.categories]
    cat_attach   = [c.voltcare_attach * 100 for c in snapshot.categories]
    cat_net_avg  = [(c.voltcare_attach - c.voltcare_vs_network) * 100 for c in snapshot.categories]
    cat_colors   = ["#E24B4A" if c.voltcare_vs_network < -0.05
                    else "#FAC775" if c.voltcare_vs_network < 0
                    else "#1D9E75" for c in snapshot.categories]

    fig_attach = go.Figure()
    fig_attach.add_trace(go.Bar(
        name="Network avg", x=cat_names, y=cat_net_avg,
        marker_color="#253347", width=0.35,
        offsetgroup=0,
    ))
    fig_attach.add_trace(go.Bar(
        name="This store", x=cat_names, y=cat_attach,
        marker_color=cat_colors, width=0.35,
        offsetgroup=1,
    ))
    fig_attach.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B4B2A9", size=12),
        barmode="group", height=220,
        margin=dict(l=0, r=0, t=10, b=30),
        legend=dict(orientation="h", y=1.15, font=dict(size=11)),
        yaxis=dict(ticksuffix="%", gridcolor="#253347", tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_attach, use_container_width=True)

    # ── AI Brief section
    st.markdown('<div class="section-label" style="margin-top:1.5rem;">AI Action Brief</div>',
                unsafe_allow_html=True)

    if generate or "brief" not in st.session_state or \
       st.session_state.get("brief_key") != f"{selected_store_id}_{selected_date}":

        if generate:
            with st.spinner("Analysing store performance..."):
                brief = analyst.generate_brief(snapshot)
                st.session_state.brief = brief
                st.session_state.brief_key = f"{selected_store_id}_{selected_date}"
        else:
            brief = None

    else:
        brief = st.session_state.brief

    if brief is None:
        st.markdown("""
        <div style="color:#3A4A5A;font-size:0.9rem;padding:2rem;text-align:center;
                    border:1px dashed #253347;border-radius:10px;">
            ⚡ Select a store and date, then click <strong>Generate brief</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        health_cls = {"GREEN": "health-green", "AMBER": "health-amber", "RED": "health-red"}.get(
            brief.overall_health, "health-amber"
        )
        health_icon = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(brief.overall_health, "⚪")

        st.markdown(
            f'<span class="{health_cls}">{health_icon} {brief.overall_health}</span>'
            f'&nbsp;&nbsp;<span style="color:#B4B2A9;font-size:0.95rem;">{brief.one_line_summary}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:#888;font-size:0.85rem;margin:0.5rem 0 1rem;">{brief.trading_context}</div>',
            unsafe_allow_html=True,
        )

        # Priority actions
        left_brief, right_brief = st.columns([1.2, 1])

        with left_brief:
            st.markdown('<div class="section-label">Priority actions</div>', unsafe_allow_html=True)
            for action in brief.priority_actions:
                conf_color = {"HIGH": "", "MEDIUM": "amber", "LOW": "red"}.get(action.confidence, "")
                st.markdown(f"""
                <div class="action-card {conf_color}">
                    <div class="action-rank">Priority {action.rank} · {action.confidence} confidence</div>
                    <div class="action-title">{action.title}</div>
                    <div class="action-label">What</div>
                    <div class="action-text">{action.what_is_happening}</div>
                    <div class="action-label">Why it matters</div>
                    <div class="action-text">{action.why_it_matters}</div>
                    <div class="action-label">Action</div>
                    <div class="action-text">{action.what_to_do}</div>
                    <div class="action-label">Peer benchmark</div>
                    <div class="action-text">{action.peer_benchmark}</div>
                </div>
                """, unsafe_allow_html=True)

        with right_brief:
            # Going well
            st.markdown('<div class="section-label">Going well</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div style="background:#0D3D2E;border-left:3px solid #1D9E75;'
                f'border-radius:0 8px 8px 0;padding:0.75rem 1rem;'
                f'font-size:0.88rem;color:#9FE1CB;line-height:1.5;">'
                f'✓ {brief.one_thing_going_well}</div>',
                unsafe_allow_html=True,
            )

            # Watch list
            if brief.watch_list:
                st.markdown('<div class="section-label">Watch list</div>', unsafe_allow_html=True)
                for item in brief.watch_list:
                    st.markdown(f'<div class="watch-item">⚠ {item}</div>', unsafe_allow_html=True)

            # Active promotions
            if snapshot.active_promotions:
                st.markdown('<div class="section-label">Active promotions</div>', unsafe_allow_html=True)
                for promo in snapshot.active_promotions:
                    st.markdown(
                        f'<div style="background:#1A1A2E;border:1px solid #2A2A4A;'
                        f'border-radius:6px;padding:0.4rem 0.75rem;font-size:0.82rem;'
                        f'color:#AFA9EC;margin-bottom:0.3rem;">🏷 {promo}</div>',
                        unsafe_allow_html=True,
                    )

            # Peer stores
            if snapshot.peer_stores:
                st.markdown('<div class="section-label">Peer stores today</div>', unsafe_allow_html=True)
                for peer in snapshot.peer_stores:
                    st.markdown(f"""
                    <div class="peer-row">
                        <span class="peer-name">{peer['store_name']}</span>
                        <span class="peer-val">£{peer['revenue']/1000:.0f}k · {peer['conversion_rate']:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TAB 2 — NETWORK OVERVIEW
# ─────────────────────────────────────────────

with tab2:
    st.markdown(f"### Network overview — {selected_date}")

    # Build daily snapshot for all stores on selected date
    network_rows = []
    for store in stores:
        day_data = df[(df["store_id"] == store["id"]) & (df["date_str"] == selected_date)]
        if not day_data.empty:
            row = day_data.iloc[0]
            net_avg = benchmarks[store["tier"]]
            conv_vs = (row["conversion_rate"] - net_avg["conversion_rate"]) / net_avg["conversion_rate"]
            rev_vs  = (row["total_revenue"]   - net_avg["total_revenue"])   / net_avg["total_revenue"]
            vc_vs   = row["vc_computing"] - net_avg["vc_computing"]

            network_rows.append({
                "Store"      : store["name"],
                "ID"         : store["id"],
                "Region"     : store["region"],
                "Tier"       : store["tier"].title(),
                "Revenue"    : round(row["total_revenue"], 0),
                "Footfall"   : int(row["footfall"]),
                "Conversion" : round(row["conversion_rate"] * 100, 1),
                "Conv vs net": round(conv_vs * 100, 1),
                "NPS"        : int(row["nps"]),
                "VC Comp"    : round(row["vc_computing"] * 100, 1),
                "Staffing"   : round(row["staffing_ratio"] * 100, 0),
            })

    if network_rows:
        net_df = pd.DataFrame(network_rows).sort_values("Revenue", ascending=False)

        # Revenue chart
        fig_net = px.bar(
            net_df, x="Store", y="Revenue",
            color="Conv vs net",
            color_continuous_scale=["#E24B4A", "#FAC775", "#1D9E75"],
            color_continuous_midpoint=0,
            title="Store revenue (colour = conversion vs network average)",
        )
        fig_net.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B4B2A9", size=11),
            height=320, margin=dict(l=0, r=0, t=40, b=80),
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#253347"),
            coloraxis_colorbar=dict(title="Conv %", tickfont=dict(size=10)),
            title_font=dict(size=13, color="#B4B2A9"),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # Network table
        display_df = net_df[[
            "Store", "Tier", "Region", "Revenue", "Footfall",
            "Conversion", "NPS", "VC Comp", "Staffing"
        ]].copy()
        display_df["Revenue"] = display_df["Revenue"].apply(lambda x: f"£{x:,.0f}")
        display_df["Conversion"] = display_df["Conversion"].apply(lambda x: f"{x:.1f}%")
        display_df["VC Comp"] = display_df["VC Comp"].apply(lambda x: f"{x:.1f}%")
        display_df["Staffing"] = display_df["Staffing"].apply(lambda x: f"{x:.0f}%")
        display_df["Footfall"] = display_df["Footfall"].apply(lambda x: f"{x:,}")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

# ─────────────────────────────────────────────
# TAB 3 — TRENDS
# ─────────────────────────────────────────────

with tab3:
    st.markdown(f"### {store_info['name']} — 12-week trends")

    store_history = df[df["store_id"] == selected_store_id].copy()
    store_history = store_history.sort_values("date")

    # Weekly aggregation
    store_history["week"] = store_history["date"].dt.isocalendar().week
    weekly = store_history.groupby("week_num").agg(
        revenue        = ("total_revenue", "mean"),
        conversion     = ("conversion_rate", "mean"),
        nps            = ("nps", "mean"),
        vc_computing   = ("vc_computing", "mean"),
        voltmobile     = ("voltmobile_attach", "mean"),
        staffing       = ("staffing_ratio", "mean"),
    ).reset_index()

    col_t1, col_t2 = st.columns(2)

    def trend_chart(df_w, y_col, title, color, y_fmt=None):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_w["week_num"], y=df_w[y_col],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            fill="tozeroy",
            fillcolor="rgba(29, 158, 117, 0.1)",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#B4B2A9", size=11),
            height=200, margin=dict(l=0, r=0, t=30, b=20),
            title=dict(text=title, font=dict(size=12, color="#B4B2A9")),
            xaxis=dict(title="Week", gridcolor="#253347", tickfont=dict(size=9)),
            yaxis=dict(gridcolor="#253347", tickfont=dict(size=9),
                       tickformat=y_fmt or ""),
        )
        return fig

    with col_t1:
        st.plotly_chart(
            trend_chart(weekly, "revenue", "Weekly avg revenue (£)", "#1D9E75", "£,.0f"),
            use_container_width=True,
        )
        st.plotly_chart(
            trend_chart(weekly, "nps", "Weekly avg NPS", "#AFA9EC"),
            use_container_width=True,
        )
        st.plotly_chart(
            trend_chart(weekly, "staffing", "Weekly avg staffing ratio", "#FAC775", ".0%"),
            use_container_width=True,
        )

    with col_t2:
        st.plotly_chart(
            trend_chart(weekly, "conversion", "Weekly avg conversion rate", "#5DCAA5", ".1%"),
            use_container_width=True,
        )
        st.plotly_chart(
            trend_chart(weekly, "vc_computing", "Computing VoltCare attach rate", "#F7C1C1", ".1%"),
            use_container_width=True,
        )
        st.plotly_chart(
            trend_chart(weekly, "voltmobile", "VoltMobile attach rate", "#378ADD", ".1%"),
            use_container_width=True,
        )
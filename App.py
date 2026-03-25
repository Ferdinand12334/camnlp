"""
app.py
Cameroon NLP Societal Analysis System
Main Streamlit application — all pages in one file.

Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

from database import (
    init_db,
    get_session,
    TextData,
    AnalysisResult,
    Topic,
    User,
    Report,
    verify_user
)
# ── Local modules ──────────────────────────────────────────────────────────────
from nlp_engine import analyze_text
from data_collector import run_collection


# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CamNLP — Societal Analysis System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise DB ──────────────────────────────────────────────────────────────
init_db()

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1e2d3d; }
[data-testid="stSidebar"] * { color: #c9d6df !important; }
.main { background-color: #f8f9fa; }
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.stButton>button {
    background-color: #2E75B6; color: white;
    border-radius: 6px; border: none; font-weight: 600;
}
.stButton>button:hover { background-color: #1F4E79; }
.section-header {
    background: linear-gradient(90deg, #1F4E79, #2E75B6);
    color: white; padding: 8px 16px; border-radius: 6px;
    margin-bottom: 12px; font-weight: 700; font-size: 1rem;
}
.info-box {
    background: #e8f4f8; border-left: 4px solid #2E75B6;
    padding: 10px 14px; border-radius: 4px; margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION HELPERS
# ═════════════════════════════════════════════════════════════════════════════
def is_logged_in():
    return st.session_state.get("logged_in", False)

def current_user():
    return st.session_state.get("user", "")

def current_role():
    return st.session_state.get("role", "Analyst")

def logout():
    for k in ["logged_in", "user", "role", "user_id"]:
        st.session_state.pop(k, None)


# ═════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def load_all_records(limit=2000):
    session = get_session()
    rows = (
        session.query(
            TextData.data_id, TextData.content, TextData.platform,
            TextData.language, TextData.collection_date,
            AnalysisResult.sentiment_label, AnalysisResult.polarity_score,
            Topic.topic_name,
        )
        .join(AnalysisResult, AnalysisResult.data_id == TextData.data_id, isouter=True)
        .join(Topic, Topic.topic_id == AnalysisResult.topic_id, isouter=True)
        .limit(limit)
        .all()
    )
    session.close()
    df = pd.DataFrame(rows, columns=["ID","Text","Platform","Language","Date",
                                      "Sentiment","Score","Topic"])
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0)
    return df


@st.cache_data(ttl=300)
def load_topics():
    session = get_session()
    topics  = session.query(Topic).all()
    session.close()
    return {t.topic_id: {"name": t.topic_name,
                          "keywords": t.keywords,
                          "frequency": t.frequency} for t in topics}


SENTIMENT_ICON = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
TOPIC_COLORS   = ["#C00000","#D46B08","#2E75B6","#6f42c1","#217346"]


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ═════════════════════════════════════════════════════════════════════════════
def page_login():
    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align:center;padding:20px;
             background:linear-gradient(135deg,#1F4E79,#2E75B6);
             border-radius:12px;color:white;margin-bottom:24px;'>
            <div style='font-size:2.8rem'>🌍</div>
            <h2 style='color:white;margin:4px 0'>CamNLP Analysis System</h2>
            <p style='color:#BDD7EE;margin:0'>Cameroon Societal Trend Intelligence</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("#### Sign In")
            username = st.text_input("Username", placeholder="e.g. admin")
            password = st.text_input("Password", type="password")
            role_sel = st.selectbox("Role", ["Administrator", "Analyst"])
            submitted = st.form_submit_button("🔐  Login", use_container_width=True)

        if submitted:
            user = verify_user(username, password)
            if user:
                if user.role != role_sel:
                    st.error(f"Role mismatch — your account role is **{user.role}**.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.user      = user.username
                    st.session_state.role      = user.role
                    st.session_state.user_id   = user.user_id
                    st.success("Login successful! Loading dashboard…")
                    time.sleep(0.6)
                    st.rerun()
            else:
                st.error("Invalid username or password.")

        st.markdown("""
        <div class='info-box'>
            <b>Demo credentials:</b><br>
            Admin &nbsp;→ <code>admin</code> / <code>admin123</code><br>
            Analyst → <code>analyst</code> / <code>analyst123</code>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
PAGES_ADMIN  = ["📊 Dashboard","🔍 Data Collection","😊 Sentiment Analysis",
                "📌 Topic Modeling","📈 Trend Analysis","📄 Reports",
                "🔬 Analyze Text","👤 Users"]
PAGES_ANALYST = ["📊 Dashboard","😊 Sentiment Analysis",
                 "📌 Topic Modeling","📈 Trend Analysis","📄 Reports","🔬 Analyze Text"]

def sidebar_nav():
    with st.sidebar:
        st.markdown(f"""
        <div style='padding:14px;background:#1a3c5e;border-radius:8px;
                    margin-bottom:16px;text-align:center;'>
            <div style='color:white;font-size:1.1rem;font-weight:700'>🌍 CamNLP</div>
            <div style='color:#8899aa;font-size:.75rem'>Societal Analysis System</div>
        </div>
        """, unsafe_allow_html=True)

        pages = PAGES_ADMIN if current_role() == "Administrator" else PAGES_ANALYST
        page  = st.radio("Nav", pages, label_visibility="collapsed")

        st.markdown("---")
        st.markdown(f"👤 **{current_user()}** · `{current_role()}`")
        if st.button("Logout", use_container_width=True):
            logout(); st.rerun()

    return page


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    st.markdown("<div class='section-header'>📊 Dashboard — Overview</div>",
                unsafe_allow_html=True)
    df = load_all_records()

    if df.empty:
        st.warning("No data yet. Go to **Data Collection** to get started.")
        return

    total   = len(df)
    pos_pct = round(len(df[df.Sentiment=="Positive"]) / total * 100, 1)
    neg_pct = round(len(df[df.Sentiment=="Negative"]) / total * 100, 1)
    neu_pct = round(100 - pos_pct - neg_pct, 1)

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total Records",      f"{total:,}")
    c2.metric("🟢 Positive Sentiment", f"{pos_pct}%")
    c3.metric("🔴 Negative Sentiment", f"{neg_pct}%")
    c4.metric("🟡 Neutral Sentiment",  f"{neu_pct}%")
    st.markdown("---")

    col_l, col_r = st.columns([1.6, 1])

    # Sentiment trend line
    with col_l:
        st.markdown("**📈 Sentiment Trend Over Time**")
        monthly = (df.dropna(subset=["Sentiment"])
                     .groupby(["Month","Sentiment"]).size()
                     .reset_index(name="Count"))
        pivot = (monthly.pivot(index="Month", columns="Sentiment", values="Count")
                        .fillna(0).reset_index().sort_values("Month"))
        fig = go.Figure()
        cmap = {"Positive":"#217346","Negative":"#C00000","Neutral":"#D46B08"}
        for col in ["Positive","Negative","Neutral"]:
            if col in pivot.columns:
                fig.add_trace(go.Scatter(x=pivot["Month"], y=pivot[col], name=col,
                    line=dict(color=cmap[col], width=2.5),
                    mode="lines+markers", marker=dict(size=5)))
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(orientation="h",y=1.1),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    # Donut
    with col_r:
        st.markdown("**🍩 Sentiment Distribution**")
        donut = go.Figure(go.Pie(
            labels=["Positive","Negative","Neutral"],
            values=[pos_pct, neg_pct, neu_pct], hole=0.55,
            marker_colors=["#217346","#C00000","#D46B08"],
            textinfo="label+percent"))
        donut.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                            showlegend=False, paper_bgcolor="white")
        st.plotly_chart(donut, use_container_width=True)

    st.markdown("---")
    col_b, col_c = st.columns([1.6, 1])

    with col_b:
        st.markdown("**📌 Top Societal Topics**")
        tc = df["Topic"].value_counts().dropna().head(6).reset_index()
        tc.columns = ["Topic","Count"]
        fig2 = px.bar(tc, x="Count", y="Topic", orientation="h",
                      color="Count", color_continuous_scale="Blues")
        fig2.update_layout(height=270, showlegend=False,
                           coloraxis_showscale=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with col_c:
        st.markdown("**🕒 Recent Records**")
        for _, row in df.sort_values("Date", ascending=False).head(6).iterrows():
            icon = SENTIMENT_ICON.get(row["Sentiment"],"⚪")
            st.markdown(
                f"**{row['Platform']}** {icon} {row['Sentiment']}<br>"
                f"<small>{str(row['Date'])[:16]} · {str(row.get('Language','')).upper()}</small>",
                unsafe_allow_html=True)
            st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATA COLLECTION
# ═════════════════════════════════════════════════════════════════════════════
def page_data_collection():
    if current_role() != "Administrator":
        st.error("Access denied. Administrator role required.")
        return

    st.markdown("<div class='section-header'>🔍 Data Collection</div>",
                unsafe_allow_html=True)

    col_cfg, col_stat = st.columns([1, 1.2])

    with col_cfg:
        st.markdown("#### ⚙️ Configuration")
        with st.form("col_form"):
            keyword     = st.text_input("Keywords", value="#Cameroon, Anglophone, COVID")
            platform    = st.selectbox("Source",
                            ["All","Twitter","Facebook","CamTimes",
                             "237online","BBC Africa","Cameroon Tribune"])
            language    = st.selectbox("Language", ["All","English","French"])
            d1, d2      = st.columns(2)
            start_date  = d1.date_input("From", datetime(2020,1,1))
            end_date    = d2.date_input("To",   datetime.today())
            max_records = st.slider("Max Records", 10, 200, 50,
                                    help="Keep ≤50 for a fast demo")
            run_btn     = st.form_submit_button("▶  Start Collection",
                                                use_container_width=True)

    with col_stat:
        st.markdown("#### 📡 Progress")
        pb  = st.progress(0)
        log = st.empty()

        if run_btn:
            log_lines = []

            def cb(current, total, msg):
                pb.progress(min(current / max(total,1), 1.0))
                log_lines.append(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")
                log.code("\n".join(log_lines[-8:]))

            with st.spinner("Collecting…"):
                result = run_collection(
                    keyword=keyword, platform=platform, language=language,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.max.time()),
                    max_records=max_records, progress_callback=cb,
                )
            pb.progress(1.0)
            load_all_records.clear()
            st.success("✅ Collection complete!")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Collected",  result["collected"])
            m2.metric("Saved",      result["saved"])
            m3.metric("Duplicates", result["duplicates"])
            m4.metric("Filtered",   result["filtered"])

    st.markdown("---")
    st.markdown("#### 📋 Recent Records")
    df = load_all_records()
    if not df.empty:
        st.dataframe(
            df.sort_values("Date", ascending=False)
              .head(20)[["ID","Platform","Language","Date","Sentiment","Topic"]],
            use_container_width=True, height=320)
    else:
        st.info("No records yet. Run a collection above.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def page_sentiment():
    st.markdown("<div class='section-header'>😊 Sentiment Analysis</div>",
                unsafe_allow_html=True)
    df = load_all_records()
    if df.empty:
        st.warning("No data. Run Data Collection first."); return

    # Filters
    with st.expander("🔎 Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        sel_plat = f1.selectbox("Platform",  ["All"]+sorted(df["Platform"].dropna().unique().tolist()))
        sel_lang = f2.selectbox("Language",  ["All","en","fr"])
        sel_sent = f3.selectbox("Sentiment", ["All","Positive","Negative","Neutral"])
        sel_top  = f4.selectbox("Topic",     ["All"]+sorted(df["Topic"].dropna().unique().tolist()))
        d1, d2   = st.columns(2)
        d_from   = d1.date_input("From", df["Date"].min().date())
        d_to     = d2.date_input("To",   df["Date"].max().date())

    mask = (df["Date"].dt.date >= d_from) & (df["Date"].dt.date <= d_to)
    if sel_plat != "All": mask &= df["Platform"]  == sel_plat
    if sel_lang != "All": mask &= df["Language"]  == sel_lang
    if sel_sent != "All": mask &= df["Sentiment"] == sel_sent
    if sel_top  != "All": mask &= df["Topic"]     == sel_top
    fdf = df[mask].copy()

    st.markdown(f"**{len(fdf):,} records** match current filters.")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Annual Sentiment Distribution**")
        yearly = fdf.groupby(["Year","Sentiment"]).size().reset_index(name="Count")
        fig = px.bar(yearly, x="Year", y="Count", color="Sentiment", barmode="group",
                     color_discrete_map={"Positive":"#217346","Negative":"#C00000","Neutral":"#D46B08"})
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Polarity Score Distribution**")
        fig2 = px.histogram(fdf, x="Score", nbins=30,
                            color_discrete_sequence=["#2E75B6"])
        fig2.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Neutral")
        fig2.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("**📋 Analyzed Records**")
    display = fdf.sort_values("Date", ascending=False).head(200).copy()
    display["Date"]  = display["Date"].dt.strftime("%Y-%m-%d")
    display["Score"] = display["Score"].round(3)
    display["Text"]  = display["Text"].str[:80] + "…"
    display = display[["ID","Date","Platform","Language","Text","Sentiment","Score","Topic"]]

    def highlight_sent(val):
        c = {"Positive":"#d4edda","Negative":"#f8d7da","Neutral":"#fff3cd"}.get(val,"")
        return f"background-color:{c}"

    st.dataframe(display.style.map(highlight_sent, subset=["Sentiment"]),
                 use_container_width=True, height=380)

    st.download_button("⬇️ Download CSV", fdf.to_csv(index=False).encode(),
                       file_name="sentiment_results.csv", mime="text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TOPIC MODELING
# ═════════════════════════════════════════════════════════════════════════════
def page_topics():
    st.markdown("<div class='section-header'>📌 Topic Modeling</div>",
                unsafe_allow_html=True)
    df     = load_all_records()
    topics = load_topics()
    if df.empty:
        st.warning("No data available."); return

    # Topic cards
    st.markdown("**🃏 Identified Topics**")
    cols = st.columns(len(topics))
    for i, (tid, t) in enumerate(topics.items()):
        kws = [k.strip() for k in t["keywords"].split(",")][:6]
        col = TOPIC_COLORS[i % len(TOPIC_COLORS)]
        with cols[i]:
            kw_html = "".join(
                f'<span style="background:{col}22;color:{col};border:1px solid {col};'
                f'border-radius:10px;padding:1px 6px;font-size:.7rem;'
                f'margin:2px;display:inline-block">{k}</span>'
                for k in kws
            )
            st.markdown(f"""
            <div style='border:2px solid {col};border-radius:8px;padding:10px;min-height:160px'>
                <div style='background:{col};color:white;border-radius:4px;
                    padding:3px 8px;font-weight:700;font-size:.82rem;
                    text-align:center;margin-bottom:6px'>Topic {i+1}</div>
                <b style='color:{col}'>{t["name"]}</b><br>
                <small style='color:#666'>{t["frequency"]:,} docs</small><br><br>
                {kw_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**📊 Topic Distribution**")
        tc = df["Topic"].value_counts().reset_index()
        tc.columns = ["Topic","Count"]
        fig = px.bar(tc, x="Topic", y="Count", color="Topic",
                     color_discrete_sequence=TOPIC_COLORS)
        fig.update_layout(height=300, showlegend=False,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**🔥 Topic × Sentiment Heatmap**")
        cross = df.groupby(["Topic","Sentiment"]).size().reset_index(name="Count")
        if not cross.empty:
            piv = cross.pivot(index="Topic", columns="Sentiment",
                              values="Count").fillna(0)
            pct = piv.div(piv.sum(axis=1), axis=0).round(3) * 100
            fig2 = px.imshow(pct, text_auto=".0f",
                             color_continuous_scale="RdYlGn",
                             zmin=0, zmax=100)
            fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                               paper_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("**📈 Topic Prevalence Over Time**")
    trend = df.groupby(["Month","Topic"]).size().reset_index(name="Count")
    fig3 = px.line(trend.sort_values("Month"), x="Month", y="Count",
                   color="Topic", markers=True,
                   color_discrete_sequence=TOPIC_COLORS)
    fig3.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                       margin=dict(l=0,r=0,t=10,b=0),
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig3, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TREND ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def page_trends():
    st.markdown("<div class='section-header'>📈 Trend Analysis</div>",
                unsafe_allow_html=True)
    df = load_all_records()
    if df.empty:
        st.warning("No data available."); return

    # Monthly avg polarity
    st.markdown("**📉 Monthly Average Polarity Score**")
    ms = df.groupby("Month")["Score"].mean().reset_index().sort_values("Month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ms["Month"], y=ms["Score"],
        mode="lines+markers", fill="tozeroy",
        line=dict(color="#2E75B6", width=2.5),
        fillcolor="rgba(46,117,182,0.15)"))
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral")
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**📡 Records by Platform**")
        plat = df["Platform"].value_counts().reset_index()
        plat.columns = ["Platform","Count"]
        fig2 = px.pie(plat, names="Platform", values="Count", hole=0.4,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("**🌐 Language Distribution**")
        lang = df["Language"].value_counts().reset_index()
        lang.columns = ["Language","Count"]
        lang["Language"] = lang["Language"].map({"en":"English","fr":"French"}).fillna("Other")
        fig3 = px.bar(lang, x="Language", y="Count", color="Language",
                      color_discrete_map={"English":"#2E75B6","French":"#C00000"})
        fig3.update_layout(height=300, showlegend=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    # Yearly heatmap
    st.markdown("---")
    st.markdown("**🗓️ Year × Sentiment Heatmap**")
    heat = df.groupby(["Year","Sentiment"]).size().reset_index(name="Count")
    if not heat.empty:
        piv = heat.pivot(index="Sentiment", columns="Year", values="Count").fillna(0)
        fig4 = px.imshow(piv, text_auto=True, color_continuous_scale="Blues")
        fig4.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ═════════════════════════════════════════════════════════════════════════════
# Add this function BEFORE page_reports() in hoo.py

def generate_pdf_report(title, stats, records):
    try:
        from fpdf import FPDF

        def safe(text):
            """Convert any text to Latin-1 safe string for fpdf."""
            return str(text).encode("latin-1", errors="replace").decode("latin-1")

        pdf = FPDF()
        pdf.add_page()

        # ── Title ──────────────────────────────────────────
        pdf.set_font("Arial", "B", 16)
        pdf.set_fill_color(31, 78, 121)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 12, safe(title), ln=True, fill=True, align="C")
        pdf.ln(4)

        # ── Summary Stats ──────────────────────────────────
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 8, "Summary Statistics", ln=True)
        pdf.set_draw_color(31, 78, 121)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, safe(f"Date Range    : {stats['date_range']}"), ln=True)
        pdf.cell(0, 7, safe(f"Total Records : {stats['total']:,}"), ln=True)
        pdf.cell(0, 7, safe(f"Dominant Topic: {stats['top_topic']}"), ln=True)
        pdf.cell(0, 7, safe(
            f"Sentiment     : Positive {stats['positive_pct']}%  |  "
            f"Negative {stats['negative_pct']}%  |  "
            f"Neutral {stats['neutral_pct']}%"
        ), ln=True)
        pdf.ln(4)

        # ── Records Table ──────────────────────────────────
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 8, "Sample Records", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        # Table header
        pdf.set_fill_color(46, 117, 182)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", "B", 8)
        pdf.cell(22, 7, "Date",      border=1, fill=True)
        pdf.cell(25, 7, "Platform",  border=1, fill=True)
        pdf.cell(10, 7, "Lang",      border=1, fill=True)
        pdf.cell(75, 7, "Text",      border=1, fill=True)
        pdf.cell(22, 7, "Sentiment", border=1, fill=True)
        pdf.cell(16, 7, "Score",     border=1, fill=True)
        pdf.cell(20, 7, "Topic",     border=1, fill=True, ln=True)

        # Table rows
        pdf.set_font("Arial", "", 7)
        for i, r in enumerate(records):
            # Alternate row shading
            if i % 2 == 0:
                pdf.set_fill_color(235, 245, 255)
            else:
                pdf.set_fill_color(255, 255, 255)

            s = r.get("sentiment", "Neutral")

            # Sentiment colour for label
            if s == "Positive":
                pdf.set_text_color(33, 115, 70)
            elif s == "Negative":
                pdf.set_text_color(192, 0, 0)
            else:
                pdf.set_text_color(180, 120, 0)

            pdf.cell(22, 6, safe(str(r.get("date",     ""))[:10]), border=1, fill=True)
            pdf.cell(25, 6, safe(str(r.get("platform", ""))[:18]), border=1, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(10, 6, safe(str(r.get("language", ""))[:4]),  border=1, fill=True)
            pdf.cell(75, 6, safe(str(r.get("text",     ""))[:55]), border=1, fill=True)

            # Re-apply sentiment colour for sentiment + score columns
            if s == "Positive":
                pdf.set_text_color(33, 115, 70)
            elif s == "Negative":
                pdf.set_text_color(192, 0, 0)
            else:
                pdf.set_text_color(180, 120, 0)

            pdf.cell(22, 6, safe(s),                               border=1, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(16, 6, safe(str(r.get("score",    ""))[:6]),  border=1, fill=True)
            pdf.cell(20, 6, safe(str(r.get("topic",    ""))[:14]), border=1, fill=True, ln=True)

        # ── Footer ─────────────────────────────────────────
        pdf.ln(6)
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, safe(
            f"Generated by CamNLP Societal Analysis System - "
            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
        ), align="C")

        return bytes(pdf.output())

    except ImportError:
        return None
def page_reports():
    st.markdown("<div class='section-header'>📄 Reports</div>",
                unsafe_allow_html=True)
    df = load_all_records()

    c_form, c_prev = st.columns([1, 1.2])

    with c_form:
        st.markdown("#### ⚙️ Configure Report")
        with st.form("rep_form"):
            r_title = st.text_input("Title", "Cameroon Societal Trends Q1 2025")
            r_type  = st.selectbox("Type",
                        ["Full Analysis","Sentiment Analysis","Topic Modeling"])
            r_plat  = st.selectbox("Platform", ["All","Twitter","Facebook",
                                                 "CamTimes","237online","BBC Africa"])
            rd1, rd2 = st.columns(2)
            r_from  = rd1.date_input("From", datetime(2020,1,1))
            r_to    = rd2.date_input("To",   datetime.today())
            r_fmt   = st.selectbox("Format", ["PDF","CSV"])
            gen_btn = st.form_submit_button("📄 Generate", use_container_width=True)

    with c_prev:
        st.markdown("#### 👁️ Preview")
        total = len(df)
        pp = round(len(df[df.Sentiment=="Positive"]) / max(total,1) * 100, 1)
        np_ = round(len(df[df.Sentiment=="Negative"]) / max(total,1) * 100, 1)
        nu  = round(100 - pp - np_, 1)
        tt  = df["Topic"].value_counts().index[0] if not df.empty else "N/A"
        st.markdown(f"""
        <div style='border:1px solid #dee2e6;border-radius:8px;padding:16px;background:white;'>
            <h4 style='color:#1F4E79;text-align:center'>CAMEROON NLP ANALYSIS</h4>
            <hr/>
            Total records: <b>{total:,}</b><br>
            Dominant topic: <b>{tt}</b><br>
            🟢 {pp}%  🔴 {np_}%  🟡 {nu}%<br><br>
            Sections: Sentiment Overview · Topic Distribution ·
            Trend Analysis · Platform Breakdown · Sample Records
        </div>
        """, unsafe_allow_html=True)

    if gen_btn:
        mask = (df["Date"].dt.date >= r_from) & (df["Date"].dt.date <= r_to)
        if r_plat != "All": mask &= df["Platform"] == r_plat
        rdf = df[mask]
        tot = len(rdf)
        pp2  = round(len(rdf[rdf.Sentiment=="Positive"])/max(tot,1)*100,1)
        np2  = round(len(rdf[rdf.Sentiment=="Negative"])/max(tot,1)*100,1)
        nu2  = round(100-pp2-np2,1)
        tt2  = rdf["Topic"].value_counts().index[0] if not rdf.empty else "N/A"

        if r_fmt == "PDF":
            stats   = {"total":tot,"positive_pct":pp2,"negative_pct":np2,
                       "neutral_pct":nu2,"top_topic":tt2,
                       "date_range":f"{r_from} to {r_to}"}
            records = [{"date":str(r["Date"])[:10],"platform":r["Platform"],
                        "language":r["Language"],"text":str(r["Text"])[:60],
                        "sentiment":r["Sentiment"],"score":round(float(r["Score"]),3),
                        "topic":r["Topic"]}
                       for _, r in rdf.head(30).iterrows()]
            pdf_bytes = generate_pdf_report(r_title, stats, records)
            if pdf_bytes:
                session = get_session()
                session.add(Report(
                    user_id=st.session_state.get("user_id",1),
                    title=r_title,
                    content=f"{r_type} | {r_from}–{r_to}",
                    format="PDF"))
                session.commit(); session.close()
                st.success("✅ Report generated!")
                st.download_button("⬇️ Download PDF", pdf_bytes,
                    file_name=f"{r_title.replace(' ','_')}.pdf",
                    mime="application/pdf")
            else:
                st.error("fpdf2 not installed — run: pip install fpdf2")
        else:
            st.download_button("⬇️ Download CSV", rdf.to_csv(index=False).encode(),
                file_name=f"{r_title.replace(' ','_')}.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("#### 📚 Saved Reports")
    session = get_session()
    saved   = session.query(Report).order_by(Report.created_at.desc()).limit(20).all()
    session.close()
    if saved:
        st.dataframe(pd.DataFrame([{
            "Title": r.title, "Notes": r.content,
            "Format": r.format, "Created": str(r.created_at)[:16]
        } for r in saved]), use_container_width=True)
    else:
        st.info("No reports yet.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE TEXT
# ═════════════════════════════════════════════════════════════════════════════
def page_analyze_text():
    st.markdown("<div class='section-header'>🔬 Analyze Text</div>",
                unsafe_allow_html=True)
    st.markdown("Enter any text to instantly analyze its sentiment, language, and topic.")

    txt  = st.text_area("Text", height=140,
                        placeholder="Type or paste English / French text here…")
    c_a, c_s = st.columns([1, 4])
    run  = c_a.button("🔍 Analyze", use_container_width=True)
    save = c_s.checkbox("Save to database")

    if run and txt.strip():
        res = analyze_text(txt)
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Language",  {"en":"🇬🇧 English","fr":"🇫🇷 French"}.get(res["language"],"?"))
        c2.metric("Sentiment", res["sentiment_label"])
        c3.metric("Polarity",  f"{res['polarity_score']:+.3f}")
        c4.metric("Topic",     res["topic_name"])

        score = res["polarity_score"]
        pct   = int((score + 1) / 2 * 100)
        color = "#217346" if score > 0.1 else ("#C00000" if score < -0.1 else "#D46B08")
        st.markdown(f"""
        <div style='margin:12px 0 4px;font-weight:600;color:#444'>Polarity Score</div>
        <div style='background:#eee;border-radius:8px;height:18px;'>
            <div style='background:{color};width:{pct}%;height:18px;border-radius:8px;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;
                    font-size:.8rem;color:#888;margin-top:2px'>
            <span>-1.0 Very Negative</span>
            <span>0 Neutral</span>
            <span>+1.0 Very Positive</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Cleaned text after preprocessing:**")
        st.code(res["cleaned_text"])

        if save:
            session = get_session()
            td = TextData(content=txt, source="Manual Input",
                          platform="Manual", language=res["language"],
                          collection_date=datetime.utcnow())
            session.add(td); session.flush()
            ar = AnalysisResult(data_id=td.data_id,
                                sentiment_label=res["sentiment_label"],
                                polarity_score=res["polarity_score"],
                                topic_id=res["topic_id"],
                                analysis_date=datetime.utcnow())
            session.add(ar); session.commit(); session.close()
            load_all_records.clear()
            st.success("Result saved to database.")
    elif run:
        st.warning("Please enter some text first.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: USERS
# ═════════════════════════════════════════════════════════════════════════════
def page_users():
    if current_role() != "Administrator":
        st.error("Access denied."); return
    st.markdown("<div class='section-header'>👤 User Management</div>",
                unsafe_allow_html=True)
    import bcrypt as _bcrypt

    session = get_session()
    users   = session.query(User).all()
    session.close()

    st.dataframe(pd.DataFrame([{
        "ID":       u.user_id, "Username": u.username,
        "Role":     u.role,    "Email":    u.email or "",
        "Created":  str(u.created_at)[:16]
    } for u in users]), use_container_width=True)

    st.markdown("---")
    st.markdown("#### ➕ Add New User")
    with st.form("add_user"):
        nu = st.text_input("Username")
        np = st.text_input("Password", type="password")
        nr = st.selectbox("Role", ["Analyst","Administrator"])
        ne = st.text_input("Email")
        if st.form_submit_button("Add User", use_container_width=True):
            if nu and np:
                pw = _bcrypt.hashpw(np.encode(), _bcrypt.gensalt()).decode()
                session = get_session()
                if session.query(User).filter_by(username=nu).first():
                    st.error("Username already exists.")
                else:
                    session.add(User(username=nu, password_hash=pw, role=nr, email=ne))
                    session.commit()
                    st.success(f"User '{nu}' created.")
                session.close()
                st.rerun()
            else:
                st.warning("Username and password are required.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════
def main():
    if not is_logged_in():
        page_login()
        return

    page = sidebar_nav()

    routes = {
        "Dashboard":       page_dashboard,
        "Data Collection": page_data_collection,
        "Sentiment":       page_sentiment,
        "Topic":           page_topics,
        "Trend":           page_trends,
        "Reports":         page_reports,
        "Analyze Text":    page_analyze_text,
        "Users":           page_users,
    }
    for key, fn in routes.items():
        if key in page:
            fn()
            return


if __name__ == "__main__":
    main()
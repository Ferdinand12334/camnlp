"""
App.py
Cameroon NLP Societal Analysis System
Full bilingual (English / French) Streamlit application.
Run:  streamlit run App.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from PIL import Image
import os
import time

from database import (
    init_db, get_session,
    TextData, AnalysisResult, Topic, User, Report, verify_user
)
from nlp_engine     import analyze_text
from data_collector import run_collection

# ── Flag icon ─────────────────────────────────────────────────────────────────
_flag_path = os.path.join(os.path.dirname(__file__), "flag.png")
_flag_img  = Image.open(_flag_path) if os.path.exists(_flag_path) else "🇨🇲"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CamNLP — Societal Analysis System",
    page_icon=_flag_img,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialise DB ─────────────────────────────────────────────────────────────
init_db()

# ═════════════════════════════════════════════════════════════════════════════
# TRANSLATIONS
# ═════════════════════════════════════════════════════════════════════════════
TRANSLATIONS = {
    "en": {
        "app_title":            "CamNLP — Societal Analysis System",
        "app_subtitle":         "Cameroon Societal Trend Intelligence",
        "language_toggle":      "🌐 Language",
        "nav_dashboard":        "📊 Dashboard",
        "nav_collection":       "🔍 Data Collection",
        "nav_sentiment":        "😊 Sentiment Analysis",
        "nav_topics":           "📌 Topic Modeling",
        "nav_trends":           "📈 Trend Analysis",
        "nav_reports":          "📄 Reports",
        "nav_analyze":          "🔬 Analyze Text",
        "nav_users":            "👤 Users",
        "logout":               "Logout",
        "sign_in":              "Sign In",
        "username":             "Username",
        "password":             "Password",
        "role":                 "Role",
        "login_btn":            "🔐 Login",
        "login_success":        "Login successful! Loading dashboard…",
        "login_fail":           "Invalid username or password.",
        "role_mismatch":        "Role mismatch — your account role is",
        "demo_creds":           "Demo credentials",
        "dashboard_title":      "📊 Dashboard — Overview",
        "total_records":        "📦 Total Records",
        "positive_sentiment":   "🟢 Positive Sentiment",
        "negative_sentiment":   "🔴 Negative Sentiment",
        "neutral_sentiment":    "🟡 Neutral Sentiment",
        "sentiment_trend":      "📈 Sentiment Trend Over Time",
        "sentiment_dist":       "🍩 Sentiment Distribution",
        "top_topics":           "📌 Top Societal Topics",
        "recent_records":       "🕒 Recent Records",
        "no_data_dashboard":    "No data yet. Go to Data Collection to get started.",
        "refresh":              "🔄 Refresh",
        "collection_title":     "🔍 Data Collection",
        "configuration":        "⚙️ Configuration",
        "keywords":             "Keywords",
        "source":               "Source",
        "language_label":       "Language",
        "date_from":            "From",
        "date_to":              "To",
        "max_records":          "Max Records",
        "start_collection":     "▶ Start Collection",
        "progress_title":       "📡 Progress",
        "collecting":           "Collecting…",
        "collection_complete":  "✅ Collection complete!",
        "collected":            "Collected",
        "saved":                "Saved",
        "duplicates":           "Duplicates",
        "filtered":             "Filtered",
        "recent_records_tbl":   "📋 Recent Records",
        "no_records":           "No records yet. Run a collection above.",
        "access_denied":        "Access denied. Administrator role required.",
        "collecting_msg":       "Collecting record",
        "sentiment_title":      "😊 Sentiment Analysis",
        "filters":              "🔎 Filters",
        "platform":             "Platform",
        "all":                  "All",
        "records_match":        "records match current filters.",
        "annual_sentiment":     "Annual Sentiment Distribution",
        "polarity_dist":        "Polarity Score Distribution",
        "analyzed_records":     "📋 Analyzed Records",
        "download_csv":         "⬇️ Download CSV",
        "no_data_sentiment":    "No data. Run Data Collection first.",
        "topics_title":         "📌 Topic Modeling",
        "identified_topics":    "🃏 Identified Topics",
        "topic_distribution":   "📊 Topic Distribution",
        "topic_heatmap":        "🔥 Topic × Sentiment Heatmap",
        "topic_trend":          "📈 Topic Prevalence Over Time",
        "no_data_topics":       "No data available.",
        "docs":                 "docs",
        "trends_title":         "📈 Trend Analysis",
        "monthly_polarity":     "📉 Monthly Average Polarity Score",
        "by_platform":          "📡 Records by Platform",
        "language_dist":        "🌐 Language Distribution",
        "year_sentiment":       "🗓️ Year × Sentiment Heatmap",
        "no_data_trends":       "No data available.",
        "reports_title":        "📄 Reports",
        "configure_report":     "⚙️ Configure Report",
        "report_title_lbl":     "Title",
        "report_type":          "Type",
        "report_platform":      "Platform",
        "report_format":        "Format",
        "generate_btn":         "📄 Generate",
        "report_preview":       "👁️ Preview",
        "total_records_lbl":    "Total records",
        "dominant_topic":       "Dominant topic",
        "report_success":       "✅ Report generated!",
        "download_pdf":         "⬇️ Download PDF",
        "saved_reports":        "📚 Saved Reports",
        "no_reports":           "No reports yet.",
        "fpdf_error":           "fpdf2 not installed — run: pip install fpdf2",
        "analyze_title":        "🔬 Analyze Text",
        "analyze_intro":        "Enter any text to instantly analyze its sentiment, language, and topic.",
        "text_label":           "Text",
        "text_placeholder":     "Type or paste English / French text here…",
        "analyze_btn":          "🔍 Analyze",
        "save_db":              "Save to database",
        "language_result":      "Language",
        "sentiment_result":     "Sentiment",
        "polarity_result":      "Polarity",
        "topic_result":         "Topic",
        "polarity_label":       "Polarity Score",
        "very_negative":        "-1.0 Very Negative",
        "neutral_label":        "0 Neutral",
        "very_positive":        "+1.0 Very Positive",
        "cleaned_text":         "Cleaned text after preprocessing:",
        "saved_success":        "Result saved to database.",
        "enter_text":           "Please enter some text first.",
        "users_title":          "👤 User Management",
        "add_user":             "➕ Add New User",
        "new_username":         "Username",
        "new_password":         "Password",
        "new_role":             "Role",
        "new_email":            "Email",
        "add_user_btn":         "Add User",
        "user_exists":          "Username already exists.",
        "user_created":         "User created.",
        "user_required":        "Username and password are required.",
        "lang_en":              "🇬🇧 English",
        "lang_fr":              "🇫🇷 French",
        "lang_other":           "Other",
    },
    "fr": {
        "app_title":            "CamNLP — Système d'Analyse Sociétale",
        "app_subtitle":         "Intelligence sur les Tendances Sociétales du Cameroun",
        "language_toggle":      "🌐 Langue",
        "nav_dashboard":        "📊 Tableau de bord",
        "nav_collection":       "🔍 Collecte de données",
        "nav_sentiment":        "😊 Analyse des sentiments",
        "nav_topics":           "📌 Modélisation des sujets",
        "nav_trends":           "📈 Analyse des tendances",
        "nav_reports":          "📄 Rapports",
        "nav_analyze":          "🔬 Analyser un texte",
        "nav_users":            "👤 Utilisateurs",
        "logout":               "Déconnexion",
        "sign_in":              "Connexion",
        "username":             "Nom d'utilisateur",
        "password":             "Mot de passe",
        "role":                 "Rôle",
        "login_btn":            "🔐 Se connecter",
        "login_success":        "Connexion réussie ! Chargement du tableau de bord…",
        "login_fail":           "Nom d'utilisateur ou mot de passe incorrect.",
        "role_mismatch":        "Rôle incorrect — votre rôle est",
        "demo_creds":           "Identifiants de démonstration",
        "dashboard_title":      "📊 Tableau de bord — Vue d'ensemble",
        "total_records":        "📦 Total des enregistrements",
        "positive_sentiment":   "🟢 Sentiment positif",
        "negative_sentiment":   "🔴 Sentiment négatif",
        "neutral_sentiment":    "🟡 Sentiment neutre",
        "sentiment_trend":      "📈 Évolution du sentiment dans le temps",
        "sentiment_dist":       "🍩 Répartition des sentiments",
        "top_topics":           "📌 Principaux sujets sociétaux",
        "recent_records":       "🕒 Enregistrements récents",
        "no_data_dashboard":    "Pas de données. Allez dans Collecte de données pour commencer.",
        "refresh":              "🔄 Actualiser",
        "collection_title":     "🔍 Collecte de données",
        "configuration":        "⚙️ Configuration",
        "keywords":             "Mots-clés",
        "source":               "Source",
        "language_label":       "Langue",
        "date_from":            "Du",
        "date_to":              "Au",
        "max_records":          "Nombre max d'enregistrements",
        "start_collection":     "▶ Lancer la collecte",
        "progress_title":       "📡 Progression",
        "collecting":           "Collecte en cours…",
        "collection_complete":  "✅ Collecte terminée !",
        "collected":            "Collectés",
        "saved":                "Enregistrés",
        "duplicates":           "Doublons",
        "filtered":             "Filtrés",
        "recent_records_tbl":   "📋 Enregistrements récents",
        "no_records":           "Aucun enregistrement. Lancez une collecte ci-dessus.",
        "access_denied":        "Accès refusé. Rôle Administrateur requis.",
        "collecting_msg":       "Collecte de l'enregistrement",
        "sentiment_title":      "😊 Analyse des sentiments",
        "filters":              "🔎 Filtres",
        "platform":             "Plateforme",
        "all":                  "Tous",
        "records_match":        "enregistrements correspondent aux filtres.",
        "annual_sentiment":     "Répartition annuelle des sentiments",
        "polarity_dist":        "Distribution du score de polarité",
        "analyzed_records":     "📋 Enregistrements analysés",
        "download_csv":         "⬇️ Télécharger CSV",
        "no_data_sentiment":    "Pas de données. Lancez d'abord une collecte.",
        "topics_title":         "📌 Modélisation des sujets",
        "identified_topics":    "🃏 Sujets identifiés",
        "topic_distribution":   "📊 Répartition des sujets",
        "topic_heatmap":        "🔥 Carte thermique Sujet x Sentiment",
        "topic_trend":          "📈 Évolution des sujets dans le temps",
        "no_data_topics":       "Aucune donnée disponible.",
        "docs":                 "docs",
        "trends_title":         "📈 Analyse des tendances",
        "monthly_polarity":     "📉 Score de polarité mensuel moyen",
        "by_platform":          "📡 Enregistrements par plateforme",
        "language_dist":        "🌐 Répartition des langues",
        "year_sentiment":       "🗓️ Carte thermique Année x Sentiment",
        "no_data_trends":       "Aucune donnée disponible.",
        "reports_title":        "📄 Rapports",
        "configure_report":     "⚙️ Configurer le rapport",
        "report_title_lbl":     "Titre",
        "report_type":          "Type",
        "report_platform":      "Plateforme",
        "report_format":        "Format",
        "generate_btn":         "📄 Générer",
        "report_preview":       "👁️ Aperçu",
        "total_records_lbl":    "Total des enregistrements",
        "dominant_topic":       "Sujet dominant",
        "report_success":       "✅ Rapport généré !",
        "download_pdf":         "⬇️ Télécharger PDF",
        "saved_reports":        "📚 Rapports enregistrés",
        "no_reports":           "Aucun rapport pour l'instant.",
        "fpdf_error":           "fpdf2 non installé — exécutez : pip install fpdf2",
        "analyze_title":        "🔬 Analyser un texte",
        "analyze_intro":        "Entrez n'importe quel texte pour analyser instantanément son sentiment, sa langue et son sujet.",
        "text_label":           "Texte",
        "text_placeholder":     "Tapez ou collez du texte en anglais ou en français ici…",
        "analyze_btn":          "🔍 Analyser",
        "save_db":              "Enregistrer dans la base de données",
        "language_result":      "Langue",
        "sentiment_result":     "Sentiment",
        "polarity_result":      "Polarité",
        "topic_result":         "Sujet",
        "polarity_label":       "Score de polarité",
        "very_negative":        "-1.0 Très négatif",
        "neutral_label":        "0 Neutre",
        "very_positive":        "+1.0 Très positif",
        "cleaned_text":         "Texte nettoyé après prétraitement :",
        "saved_success":        "Résultat enregistré dans la base de données.",
        "enter_text":           "Veuillez d'abord saisir du texte.",
        "users_title":          "👤 Gestion des utilisateurs",
        "add_user":             "➕ Ajouter un utilisateur",
        "new_username":         "Nom d'utilisateur",
        "new_password":         "Mot de passe",
        "new_role":             "Rôle",
        "new_email":            "E-mail",
        "add_user_btn":         "Ajouter l'utilisateur",
        "user_exists":          "Ce nom d'utilisateur existe déjà.",
        "user_created":         "Utilisateur créé.",
        "user_required":        "Le nom d'utilisateur et le mot de passe sont requis.",
        "lang_en":              "🇬🇧 Anglais",
        "lang_fr":              "🇫🇷 Français",
        "lang_other":           "Autre",
    }
}

def t():
    return TRANSLATIONS.get(st.session_state.get("lang", "en"), TRANSLATIONS["en"])


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
# DATA HELPERS — no cache so dashboard always shows fresh data
# ═════════════════════════════════════════════════════════════════════════════
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
        .limit(limit).all()
    )
    session.close()
    df = pd.DataFrame(rows, columns=["ID","Text","Platform","Language","Date",
                                      "Sentiment","Score","Topic"])
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Year"]  = df["Date"].dt.year
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce").fillna(0)
    return df


def load_topics():
    session = get_session()
    topics  = session.query(Topic).all()
    session.close()
    return {tp.topic_id: {"name": tp.topic_name,
                           "keywords": tp.keywords,
                           "frequency": tp.frequency} for tp in topics}


SENTIMENT_ICON = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
TOPIC_COLORS   = ["#C00000","#D46B08","#2E75B6","#6f42c1","#217346"]


# ═════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATOR
# ═════════════════════════════════════════════════════════════════════════════
def generate_pdf_report(title, stats, records):
    try:
        from fpdf import FPDF

        def safe(text):
            return str(text).encode("latin-1", errors="replace").decode("latin-1")

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 16)
        pdf.set_fill_color(31, 78, 121)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 12, safe(title), ln=True, fill=True, align="C")
        pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 8, "Summary Statistics", ln=True)
        pdf.set_draw_color(31, 78, 121)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

        pdf.set_font("Arial", "", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, safe(f"Date Range    : {stats['date_range']}"), ln=True)
        pdf.cell(0, 7, safe(f"Total Records : {stats['total']:,}"),    ln=True)
        pdf.cell(0, 7, safe(f"Dominant Topic: {stats['top_topic']}"),  ln=True)
        pdf.cell(0, 7, safe(
            f"Sentiment     : Positive {stats['positive_pct']}%  |  "
            f"Negative {stats['negative_pct']}%  |  "
            f"Neutral {stats['neutral_pct']}%"), ln=True)
        pdf.ln(4)

        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(31, 78, 121)
        pdf.cell(0, 8, "Sample Records", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)

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

        pdf.set_font("Arial", "", 7)
        for i, r in enumerate(records):
            if i % 2 == 0:
                pdf.set_fill_color(235, 245, 255)
            else:
                pdf.set_fill_color(255, 255, 255)

            s = r.get("sentiment", "Neutral")
            if s == "Positive":   pdf.set_text_color(33, 115, 70)
            elif s == "Negative": pdf.set_text_color(192, 0, 0)
            else:                 pdf.set_text_color(180, 120, 0)

            pdf.cell(22, 6, safe(str(r.get("date",     ""))[:10]), border=1, fill=True)
            pdf.cell(25, 6, safe(str(r.get("platform", ""))[:18]), border=1, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(10, 6, safe(str(r.get("language", ""))[:4]),  border=1, fill=True)
            pdf.cell(75, 6, safe(str(r.get("text",     ""))[:55]), border=1, fill=True)

            if s == "Positive":   pdf.set_text_color(33, 115, 70)
            elif s == "Negative": pdf.set_text_color(192, 0, 0)
            else:                 pdf.set_text_color(180, 120, 0)

            pdf.cell(22, 6, safe(s),                               border=1, fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(16, 6, safe(str(r.get("score",    ""))[:6]),  border=1, fill=True)
            pdf.cell(20, 6, safe(str(r.get("topic",    ""))[:14]), border=1, fill=True, ln=True)

        pdf.ln(6)
        pdf.set_font("Arial", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, safe(
            f"Generated by CamNLP Societal Analysis System - "
            f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"), align="C")

        return bytes(pdf.output())
    except ImportError:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: LOGIN
# ═════════════════════════════════════════════════════════════════════════════
def page_login():
    lang = st.session_state.get("lang", "en")
    _, col, _ = st.columns([1, 1.3, 1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        lang_choice = st.radio(
            "🌐", ["English", "Français"], horizontal=True,
            index=0 if lang == "en" else 1
        )
        st.session_state.lang = "fr" if lang_choice == "Français" else "en"
        tr = t()

        st.markdown(f"""
        <div style='text-align:center;padding:20px;
             background:linear-gradient(135deg,#1F4E79,#2E75B6);
             border-radius:12px;color:white;margin-bottom:24px;'>
            <div style='font-size:2.8rem'>🇨🇲</div>
            <h2 style='color:white;margin:4px 0'>{tr["app_title"]}</h2>
            <p style='color:#BDD7EE;margin:0'>{tr["app_subtitle"]}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown(f"#### {tr['sign_in']}")
            username  = st.text_input(tr["username"], placeholder="e.g. admin")
            password  = st.text_input(tr["password"], type="password")
            role_sel  = st.selectbox(tr["role"], ["Administrator", "Analyst"])
            submitted = st.form_submit_button(tr["login_btn"], use_container_width=True)

        if submitted:
            user = verify_user(username, password)
            if user:
                if user.role != role_sel:
                    st.error(f"{tr['role_mismatch']} **{user.role}**.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.user      = user.username
                    st.session_state.role      = user.role
                    st.session_state.user_id   = user.user_id
                    st.success(tr["login_success"])
                    time.sleep(0.6)
                    st.rerun()
            else:
                st.error(tr["login_fail"])

        st.markdown(f"""
        <div class='info-box'>
            <b>{tr["demo_creds"]}:</b><br>
            Admin &nbsp;→ <code>admin</code> / <code>admin123</code><br>
            Analyst → <code>analyst</code> / <code>analyst123</code>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
def sidebar_nav():
    with st.sidebar:
        lang_choice = st.radio(
            "🌐 Language / Langue",
            ["English", "Français"], horizontal=True,
            index=0 if st.session_state.get("lang", "en") == "en" else 1,
            key="lang_radio"
        )
        st.session_state.lang = "fr" if lang_choice == "Français" else "en"
        tr = t()

        st.markdown(f"""
        <div style='padding:14px;background:#1a3c5e;border-radius:8px;
                    margin-bottom:16px;text-align:center;'>
            <div style='color:white;font-size:1.1rem;font-weight:700'>
                🇨🇲 CamNLP</div>
            <div style='color:#8899aa;font-size:.75rem'>{tr["app_subtitle"]}</div>
        </div>
        """, unsafe_allow_html=True)

        PAGES_ADMIN = [
            tr["nav_dashboard"], tr["nav_collection"], tr["nav_sentiment"],
            tr["nav_topics"],    tr["nav_trends"],     tr["nav_reports"],
            tr["nav_analyze"],   tr["nav_users"]
        ]
        PAGES_ANALYST = [
            tr["nav_dashboard"], tr["nav_sentiment"], tr["nav_topics"],
            tr["nav_trends"],    tr["nav_reports"],   tr["nav_analyze"]
        ]

        pages = PAGES_ADMIN if current_role() == "Administrator" else PAGES_ANALYST
        page  = st.radio("Nav", pages, label_visibility="collapsed")

        st.markdown("---")
        st.markdown(f"👤 **{current_user()}** · `{current_role()}`")
        if st.button(tr["logout"], use_container_width=True):
            logout(); st.rerun()

    return page


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    tr = t()

    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.markdown(f"<div class='section-header'>{tr['dashboard_title']}</div>",
                    unsafe_allow_html=True)
    with col_btn:
        if st.button(tr["refresh"], use_container_width=True):
            st.rerun()

    df = load_all_records()

    if df.empty:
        st.warning(tr["no_data_dashboard"]); return

    total   = len(df)
    pos_pct = round(len(df[df.Sentiment=="Positive"]) / total * 100, 1)
    neg_pct = round(len(df[df.Sentiment=="Negative"]) / total * 100, 1)
    neu_pct = round(100 - pos_pct - neg_pct, 1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(tr["total_records"],      f"{total:,}")
    c2.metric(tr["positive_sentiment"], f"{pos_pct}%")
    c3.metric(tr["negative_sentiment"], f"{neg_pct}%")
    c4.metric(tr["neutral_sentiment"],  f"{neu_pct}%")
    st.markdown("---")

    col_l, col_r = st.columns([1.6, 1])
    with col_l:
        st.markdown(f"**{tr['sentiment_trend']}**")
        monthly = (df.dropna(subset=["Sentiment"])
                     .groupby(["Month","Sentiment"]).size()
                     .reset_index(name="Count"))
        pivot = (monthly.pivot(index="Month", columns="Sentiment", values="Count")
                        .fillna(0).reset_index().sort_values("Month"))
        fig = go.Figure()
        cmap = {"Positive":"#217346","Negative":"#C00000","Neutral":"#D46B08"}
        for col in ["Positive","Negative","Neutral"]:
            if col in pivot.columns:
                fig.add_trace(go.Scatter(
                    x=pivot["Month"], y=pivot[col], name=col,
                    line=dict(color=cmap[col], width=2.5),
                    mode="lines+markers", marker=dict(size=5)))
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                          legend=dict(orientation="h",y=1.1),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(f"**{tr['sentiment_dist']}**")
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
        st.markdown(f"**{tr['top_topics']}**")
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
        st.markdown(f"**{tr['recent_records']}**")
        for _, row in df.sort_values("Date", ascending=False).head(6).iterrows():
            icon = SENTIMENT_ICON.get(row["Sentiment"],"⚪")
            st.markdown(
                f"**{row['Platform']}** {icon} {row['Sentiment']}<br>"
                f"<small>{str(row['Date'])[:16]} · "
                f"{str(row.get('Language','')).upper()}</small>",
                unsafe_allow_html=True)
            st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: DATA COLLECTION
# ═════════════════════════════════════════════════════════════════════════════
def page_data_collection():
    tr = t()
    if current_role() != "Administrator":
        st.error(tr["access_denied"]); return

    st.markdown(f"<div class='section-header'>{tr['collection_title']}</div>",
                unsafe_allow_html=True)

    col_cfg, col_stat = st.columns([1, 1.2])
    with col_cfg:
        st.markdown(f"#### {tr['configuration']}")
        with st.form("col_form"):
            keyword     = st.text_input(tr["keywords"],
                                        value="#Cameroon, Anglophone, COVID")
            platform    = st.selectbox(tr["source"],
                            [tr["all"],"Twitter","Facebook","CamTimes",
                             "237online","BBC Africa","Cameroon Tribune"])
            language    = st.selectbox(tr["language_label"],
                            [tr["all"],"English","French"])
            d1, d2      = st.columns(2)
            start_date  = d1.date_input(tr["date_from"], datetime(2020,1,1))
            end_date    = d2.date_input(tr["date_to"],   datetime.today())
            max_records = st.slider(tr["max_records"], 10, 200, 50)
            run_btn     = st.form_submit_button(tr["start_collection"],
                                                use_container_width=True)

    with col_stat:
        st.markdown(f"#### {tr['progress_title']}")
        pb  = st.progress(0)
        log = st.empty()

        if run_btn:
            log_lines = []
            def cb(current, total, msg):
                pb.progress(min(current / max(total,1), 1.0))
                log_lines.append(
                    f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}")
                log.code("\n".join(log_lines[-8:]))

            with st.spinner(tr["collecting"]):
                plat_val = None if platform == tr["all"] else platform
                lang_val = None if language == tr["all"] else language
                result = run_collection(
                    keyword=keyword, platform=plat_val, language=lang_val,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date,     datetime.max.time()),
                    max_records=max_records, progress_callback=cb,
                )
            pb.progress(1.0)
            st.success(tr["collection_complete"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(tr["collected"],  result["collected"])
            m2.metric(tr["saved"],      result["saved"])
            m3.metric(tr["duplicates"], result["duplicates"])
            m4.metric(tr["filtered"],   result["filtered"])
            time.sleep(2)
            st.rerun()  # ← auto refresh entire app with new data

    st.markdown("---")
    st.markdown(f"#### {tr['recent_records_tbl']}")
    df = load_all_records()
    if not df.empty:
        st.dataframe(
            df.sort_values("Date", ascending=False)
              .head(20)[["ID","Platform","Language","Date","Sentiment","Topic"]],
            use_container_width=True, height=320)
    else:
        st.info(tr["no_records"])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
def page_sentiment():
    tr = t()
    st.markdown(f"<div class='section-header'>{tr['sentiment_title']}</div>",
                unsafe_allow_html=True)
    df = load_all_records()
    if df.empty:
        st.warning(tr["no_data_sentiment"]); return

    with st.expander(tr["filters"], expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        sel_plat = f1.selectbox(tr["platform"],
                    [tr["all"]]+sorted(df["Platform"].dropna().unique().tolist()))
        sel_lang = f2.selectbox(tr["language_label"], [tr["all"],"en","fr"])
        sel_sent = f3.selectbox(tr["sentiment_result"],
                    [tr["all"],"Positive","Negative","Neutral"])
        sel_top  = f4.selectbox(tr["topic_result"],
                    [tr["all"]]+sorted(df["Topic"].dropna().unique().tolist()))
        d1, d2   = st.columns(2)
        d_from   = d1.date_input(tr["date_from"], df["Date"].min().date())
        d_to     = d2.date_input(tr["date_to"],   df["Date"].max().date())

    mask = (df["Date"].dt.date >= d_from) & (df["Date"].dt.date <= d_to)
    if sel_plat != tr["all"]: mask &= df["Platform"]  == sel_plat
    if sel_lang != tr["all"]: mask &= df["Language"]  == sel_lang
    if sel_sent != tr["all"]: mask &= df["Sentiment"] == sel_sent
    if sel_top  != tr["all"]: mask &= df["Topic"]     == sel_top
    fdf = df[mask].copy()

    st.markdown(f"**{len(fdf):,} {tr['records_match']}**")
    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{tr['annual_sentiment']}**")
        yearly = fdf.groupby(["Year","Sentiment"]).size().reset_index(name="Count")
        fig = px.bar(yearly, x="Year", y="Count", color="Sentiment",
                     barmode="group",
                     color_discrete_map={"Positive":"#217346",
                                         "Negative":"#C00000",
                                         "Neutral":"#D46B08"})
        fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(f"**{tr['polarity_dist']}**")
        fig2 = px.histogram(fdf, x="Score", nbins=30,
                            color_discrete_sequence=["#2E75B6"])
        fig2.add_vline(x=0, line_dash="dash", line_color="red",
                       annotation_text=tr["neutral_label"])
        fig2.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0),
                           plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown(f"**{tr['analyzed_records']}**")
    display = fdf.sort_values("Date", ascending=False).head(200).copy()
    display["Date"]  = display["Date"].dt.strftime("%Y-%m-%d")
    display["Score"] = display["Score"].round(3)
    display["Text"]  = display["Text"].str[:80] + "…"
    display = display[["ID","Date","Platform","Language",
                        "Text","Sentiment","Score","Topic"]]

    def highlight_sent(val):
        c = {"Positive":"#d4edda","Negative":"#f8d7da",
             "Neutral":"#fff3cd"}.get(val,"")
        return f"background-color:{c}"

    st.dataframe(display.style.map(highlight_sent, subset=["Sentiment"]),
                 use_container_width=True, height=380)
    st.download_button(tr["download_csv"], fdf.to_csv(index=False).encode(),
                       file_name="sentiment_results.csv", mime="text/csv")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: TOPIC MODELING
# ═════════════════════════════════════════════════════════════════════════════
def page_topics():
    tr     = t()
    st.markdown(f"<div class='section-header'>{tr['topics_title']}</div>",
                unsafe_allow_html=True)
    df     = load_all_records()
    topics = load_topics()
    if df.empty:
        st.warning(tr["no_data_topics"]); return

    st.markdown(f"**{tr['identified_topics']}**")
    cols = st.columns(len(topics))
    for i, (tid, top) in enumerate(topics.items()):
        kws = [k.strip() for k in top["keywords"].split(",")][:6]
        col = TOPIC_COLORS[i % len(TOPIC_COLORS)]
        with cols[i]:
            kw_html = "".join(
                f'<span style="background:{col}22;color:{col};'
                f'border:1px solid {col};border-radius:10px;'
                f'padding:1px 6px;font-size:.7rem;margin:2px;'
                f'display:inline-block">{k}</span>'
                for k in kws)
            st.markdown(f"""
            <div style='border:2px solid {col};border-radius:8px;
                        padding:10px;min-height:160px'>
                <div style='background:{col};color:white;border-radius:4px;
                    padding:3px 8px;font-weight:700;font-size:.82rem;
                    text-align:center;margin-bottom:6px'>Topic {i+1}</div>
                <b style='color:{col}'>{top["name"]}</b><br>
                <small style='color:#666'>
                    {top["frequency"]:,} {tr["docs"]}</small><br><br>
                {kw_html}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{tr['topic_distribution']}**")
        tc = df["Topic"].value_counts().reset_index()
        tc.columns = ["Topic","Count"]
        fig = px.bar(tc, x="Topic", y="Count", color="Topic",
                     color_discrete_sequence=TOPIC_COLORS)
        fig.update_layout(height=300, showlegend=False,
                          plot_bgcolor="white", paper_bgcolor="white",
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown(f"**{tr['topic_heatmap']}**")
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
    st.markdown(f"**{tr['topic_trend']}**")
    trend = df.groupby(["Month","Topic"]).size().reset_index(name="Count")
    fig3  = px.line(trend.sort_values("Month"), x="Month", y="Count",
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
    tr = t()
    st.markdown(f"<div class='section-header'>{tr['trends_title']}</div>",
                unsafe_allow_html=True)
    df = load_all_records()
    if df.empty:
        st.warning(tr["no_data_trends"]); return

    st.markdown(f"**{tr['monthly_polarity']}**")
    ms  = df.groupby("Month")["Score"].mean().reset_index().sort_values("Month")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ms["Month"], y=ms["Score"],
        mode="lines+markers", fill="tozeroy",
        line=dict(color="#2E75B6", width=2.5),
        fillcolor="rgba(46,117,182,0.15)"))
    fig.add_hline(y=0, line_dash="dash", line_color="red",
                  annotation_text=tr["neutral_label"])
    fig.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                      plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{tr['by_platform']}**")
        plat = df["Platform"].value_counts().reset_index()
        plat.columns = ["Platform","Count"]
        fig2 = px.pie(plat, names="Platform", values="Count", hole=0.4,
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=300, margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown(f"**{tr['language_dist']}**")
        lang = df["Language"].value_counts().reset_index()
        lang.columns = ["Language","Count"]
        lang["Language"] = lang["Language"].map(
            {"en": tr["lang_en"], "fr": tr["lang_fr"]}
        ).fillna(tr["lang_other"])
        fig3 = px.bar(lang, x="Language", y="Count", color="Language",
                      color_discrete_map={tr["lang_en"]:"#2E75B6",
                                          tr["lang_fr"]:"#C00000"})
        fig3.update_layout(height=300, showlegend=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown(f"**{tr['year_sentiment']}**")
    heat = df.groupby(["Year","Sentiment"]).size().reset_index(name="Count")
    if not heat.empty:
        piv  = heat.pivot(index="Sentiment", columns="Year",
                          values="Count").fillna(0)
        fig4 = px.imshow(piv, text_auto=True, color_continuous_scale="Blues")
        fig4.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0),
                           paper_bgcolor="white")
        st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: REPORTS
# ═════════════════════════════════════════════════════════════════════════════
def page_reports():
    tr = t()
    st.markdown(f"<div class='section-header'>{tr['reports_title']}</div>",
                unsafe_allow_html=True)
    df = load_all_records()

    c_form, c_prev = st.columns([1, 1.2])
    with c_form:
        st.markdown(f"#### {tr['configure_report']}")
        with st.form("rep_form"):
            r_title  = st.text_input(tr["report_title_lbl"],
                                     "Cameroon Societal Trends Q1 2025")
            r_type   = st.selectbox(tr["report_type"],
                         ["Full Analysis","Sentiment Analysis","Topic Modeling"])
            r_plat   = st.selectbox(tr["report_platform"],
                         [tr["all"],"Twitter","Facebook",
                          "CamTimes","237online","BBC Africa"])
            rd1, rd2 = st.columns(2)
            r_from   = rd1.date_input(tr["date_from"], datetime(2020,1,1))
            r_to     = rd2.date_input(tr["date_to"],   datetime.today())
            r_fmt    = st.selectbox(tr["report_format"], ["PDF","CSV"])
            gen_btn  = st.form_submit_button(tr["generate_btn"],
                                             use_container_width=True)

    with c_prev:
        st.markdown(f"#### {tr['report_preview']}")
        total = len(df)
        pp    = round(len(df[df.Sentiment=="Positive"]) / max(total,1) * 100, 1)
        np_   = round(len(df[df.Sentiment=="Negative"]) / max(total,1) * 100, 1)
        nu    = round(100 - pp - np_, 1)
        tc    = df["Topic"].value_counts()
        tt    = tc.index[0] if not tc.empty else "N/A"
        st.markdown(f"""
        <div style='border:1px solid #dee2e6;border-radius:8px;
                    padding:16px;background:white;'>
            <h4 style='color:#1F4E79;text-align:center'>
                CAMEROON NLP ANALYSIS</h4>
            <hr/>
            {tr["total_records_lbl"]}: <b>{total:,}</b><br>
            {tr["dominant_topic"]}: <b>{tt}</b><br>
            🟢 {pp}%  🔴 {np_}%  🟡 {nu}%
        </div>
        """, unsafe_allow_html=True)

    if gen_btn:
        mask = (df["Date"].dt.date >= r_from) & (df["Date"].dt.date <= r_to)
        if r_plat != tr["all"]: mask &= df["Platform"] == r_plat
        rdf  = df[mask]
        tot  = len(rdf)
        pp2  = round(len(rdf[rdf.Sentiment=="Positive"]) / max(tot,1)*100,1)
        np2  = round(len(rdf[rdf.Sentiment=="Negative"]) / max(tot,1)*100,1)
        nu2  = round(100-pp2-np2,1)
        tc2  = rdf["Topic"].value_counts()
        tt2  = tc2.index[0] if not tc2.empty else "N/A"

        if r_fmt == "PDF":
            stats   = {"total":tot,"positive_pct":pp2,"negative_pct":np2,
                       "neutral_pct":nu2,"top_topic":tt2,
                       "date_range":f"{r_from} to {r_to}"}
            records = [{"date":str(r["Date"])[:10],"platform":r["Platform"],
                        "language":r["Language"],"text":str(r["Text"])[:60],
                        "sentiment":r["Sentiment"],
                        "score":round(float(r["Score"]),3),
                        "topic":r["Topic"]}
                       for _, r in rdf.head(30).iterrows()]
            pdf_bytes = generate_pdf_report(r_title, stats, records)
            if pdf_bytes:
                session = get_session()
                session.add(Report(
                    user_id=st.session_state.get("user_id",1),
                    title=r_title,
                    content=f"{r_type} | {r_from} to {r_to}",
                    format="PDF"))
                session.commit(); session.close()
                st.success(tr["report_success"])
                st.download_button(tr["download_pdf"], pdf_bytes,
                    file_name=f"{r_title.replace(' ','_')}.pdf",
                    mime="application/pdf")
            else:
                st.error(tr["fpdf_error"])
        else:
            st.download_button(tr["download_csv"],
                rdf.to_csv(index=False).encode(),
                file_name=f"{r_title.replace(' ','_')}.csv",
                mime="text/csv")

    st.markdown("---")
    st.markdown(f"#### {tr['saved_reports']}")
    session = get_session()
    saved   = session.query(Report).order_by(
                Report.created_at.desc()).limit(20).all()
    session.close()
    if saved:
        st.dataframe(pd.DataFrame([{
            "Title":   r.title,  "Notes":   r.content,
            "Format":  r.format, "Created": str(r.created_at)[:16]
        } for r in saved]), use_container_width=True)
    else:
        st.info(tr["no_reports"])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYZE TEXT
# ═════════════════════════════════════════════════════════════════════════════
def page_analyze_text():
    tr = t()
    st.markdown(f"<div class='section-header'>{tr['analyze_title']}</div>",
                unsafe_allow_html=True)
    st.markdown(tr["analyze_intro"])

    txt  = st.text_area(tr["text_label"], height=140,
                        placeholder=tr["text_placeholder"])
    c_a, c_s = st.columns([1, 4])
    run  = c_a.button(tr["analyze_btn"], use_container_width=True)
    save = c_s.checkbox(tr["save_db"])

    if run and txt.strip():
        res = analyze_text(txt)
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(tr["language_result"],
                  {"en": tr["lang_en"], "fr": tr["lang_fr"]}.get(
                      res["language"],"?"))
        c2.metric(tr["sentiment_result"], res["sentiment_label"])
        c3.metric(tr["polarity_result"],  f"{res['polarity_score']:+.3f}")
        c4.metric(tr["topic_result"],     res["topic_name"])

        score = res["polarity_score"]
        pct   = int((score + 1) / 2 * 100)
        color = ("#217346" if score > 0.1
                 else "#C00000" if score < -0.1 else "#D46B08")
        st.markdown(f"""
        <div style='margin:12px 0 4px;font-weight:600;color:#444'>
            {tr["polarity_label"]}</div>
        <div style='background:#eee;border-radius:8px;height:18px;'>
            <div style='background:{color};width:{pct}%;
                        height:18px;border-radius:8px;'></div>
        </div>
        <div style='display:flex;justify-content:space-between;
                    font-size:.8rem;color:#888;margin-top:2px'>
            <span>{tr["very_negative"]}</span>
            <span>{tr["neutral_label"]}</span>
            <span>{tr["very_positive"]}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**{tr['cleaned_text']}**")
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
            st.success(tr["saved_success"])
    elif run:
        st.warning(tr["enter_text"])


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: USERS
# ═════════════════════════════════════════════════════════════════════════════
def page_users():
    tr = t()
    if current_role() != "Administrator":
        st.error(tr["access_denied"]); return
    st.markdown(f"<div class='section-header'>{tr['users_title']}</div>",
                unsafe_allow_html=True)
    import bcrypt as _bcrypt

    session = get_session()
    users   = session.query(User).all()
    session.close()

    st.dataframe(pd.DataFrame([{
        "ID":      u.user_id,  "Username": u.username,
        "Role":    u.role,     "Email":    u.email or "",
        "Created": str(u.created_at)[:16]
    } for u in users]), use_container_width=True)

    st.markdown("---")
    st.markdown(f"#### {tr['add_user']}")
    with st.form("add_user"):
        nu = st.text_input(tr["new_username"])
        np = st.text_input(tr["new_password"], type="password")
        nr = st.selectbox(tr["new_role"], ["Analyst","Administrator"])
        ne = st.text_input(tr["new_email"])
        if st.form_submit_button(tr["add_user_btn"], use_container_width=True):
            if nu and np:
                pw = _bcrypt.hashpw(np.encode(), _bcrypt.gensalt()).decode()
                session = get_session()
                if session.query(User).filter_by(username=nu).first():
                    st.error(tr["user_exists"])
                else:
                    session.add(User(username=nu, password_hash=pw,
                                     role=nr, email=ne))
                    session.commit()
                    st.success(f"{tr['user_created']} '{nu}'")
                session.close()
                st.rerun()
            else:
                st.warning(tr["user_required"])


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ═════════════════════════════════════════════════════════════════════════════
def main():
    if not is_logged_in():
        page_login()
        return

    tr   = t()
    page = sidebar_nav()

    routes = {
        tr["nav_dashboard"]:  page_dashboard,
        tr["nav_collection"]: page_data_collection,
        tr["nav_sentiment"]:  page_sentiment,
        tr["nav_topics"]:     page_topics,
        tr["nav_trends"]:     page_trends,
        tr["nav_reports"]:    page_reports,
        tr["nav_analyze"]:    page_analyze_text,
        tr["nav_users"]:      page_users,
    }

    fn = routes.get(page)
    if fn:
        fn()


if __name__ == "__main__":
    main()
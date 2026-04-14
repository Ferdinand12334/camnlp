import streamlit as st
import pandas as pd

# 👉 IMPORT YOUR REAL BACKEND FUNCTION
from hoo import run_collection   # make sure hoo.py is in same folder

# ------------------- CONFIG -------------------
st.set_page_config(
    page_title="CAM NLP Analyzer",
    page_icon="🚀",
    layout="wide"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
h1, h2, h3 {
    color: #FFD700;
}
.stButton>button {
    background-color: #FFD700;
    color: black;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ------------------- SIDEBAR -------------------
st.sidebar.title("🚀 CAM NLP")
page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "📥 Data Collection",
    "📊 Results Dashboard",
    "⚙️ Settings"
])

# ------------------- CACHE -------------------
@st.cache_data
def cached_collection(keyword, platform, language, max_records):
    return run_collection(
        keyword=keyword,
        platform=platform,
        language=language,
        max_records=max_records,
        progress_callback=None  # progress handled separately
    )

# ------------------- HOME -------------------
if page == "🏠 Home":
    st.title("🚀 CAM NLP Analyzer")
    st.write("""
    Welcome to your **AI Data Collection & Analysis Platform**.

    ### 🔥 Features:
    - Collect real-time data
    - Analyze sentiment
    - Visualize insights
    - Export results

    👉 Use the sidebar to begin.
    """)

# ------------------- DATA COLLECTION -------------------
elif page == "📥 Data Collection":
    st.title("📥 Data Collection")

    col1, col2 = st.columns(2)

    with col1:
        keyword = st.text_input("🔍 Keyword", placeholder="e.g AI, football")
        platform = st.selectbox("🌐 Platform", ["YouTube", "Twitter", "TikTok"])
    
    with col2:
        language = st.selectbox("🌍 Language", ["English", "French"])
        max_records = st.slider("📊 Max Records", 10, 500, 50)

    if st.button("🚀 Start Collection"):

        if not keyword:
            st.warning("⚠️ Please enter a keyword")

        else:
            progress_bar = st.progress(0)

            def update_progress(progress):
                progress_bar.progress(int(progress * 100))

            try:
                with st.spinner("Collecting real data... 🔍"):

                    df = run_collection(
                        keyword=keyword,
                        platform=platform,
                        language=language,
                        max_records=max_records,
                        progress_callback=update_progress
                    )

                # Convert to DataFrame if needed
                if isinstance(df, list):
                    df = pd.DataFrame(df)

                # Save data
                st.session_state["data"] = df

                st.success("✅ Data collected successfully!")

            except Exception as e:
                st.error(f"❌ Error: {e}")

# ------------------- RESULTS -------------------
elif page == "📊 Results Dashboard":
    st.title("📊 Results Dashboard")

    if "data" not in st.session_state:
        st.warning("⚠️ No data available. Please collect data first.")
    else:
        df = st.session_state["data"]

        st.subheader("📄 Dataset")
        st.dataframe(df, use_container_width=True)

        # Metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Records", len(df))

        with col2:
            if "Sentiment" in df.columns:
                positive = len(df[df["Sentiment"] == "Positive"])
                st.metric("Positive Count", positive)

        # Chart
        if "Sentiment" in df.columns:
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts()
            st.bar_chart(sentiment_counts)

        # Filter
        st.subheader("🔍 Filter Data")

        if "Sentiment" in df.columns:
            sentiment_filter = st.selectbox("Filter by Sentiment", ["All", "Positive", "Negative"])

            if sentiment_filter != "All":
                filtered_df = df[df["Sentiment"] == sentiment_filter]
            else:
                filtered_df = df
        else:
            filtered_df = df

        st.dataframe(filtered_df, use_container_width=True)

        # Download
        st.download_button(
            "⬇️ Download CSV",
            df.to_csv(index=False),
            "results.csv",
            mime="text/csv"
        )

# ------------------- SETTINGS -------------------
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")

    theme = st.selectbox("Theme", ["Dark", "Light"])
    notifications = st.checkbox("Enable Notifications")

    st.success("✅ Settings saved!")

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("👨‍💻 Built by **Clarck Tech** 🚀")
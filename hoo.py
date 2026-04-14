import pandas as pd
from transformers import pipeline
from googleapiclient.discovery import build

# ------------------- CONFIG -------------------
YOUTUBE_API_KEY = "AIzaSyC1nA7LRFhYIV8w5xudqE41KvN1VBa7AXg"

# Load AI model once (fast reuse)
sentiment_model = pipeline("sentiment-analysis")

# ------------------- YOUTUBE SCRAPER -------------------
def fetch_youtube_data(keyword, max_results):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    request = youtube.search().list(
        q=keyword,
        part="snippet",
        type="video",
        maxResults=max_results
    )
    response = request.execute()

    data = []

    for item in response["items"]:
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]

        text = title + " " + description

        data.append({
            "Text": text,
            "Source": "YouTube"
        })

    return data

# ------------------- TIKTOK SCRAPER (BASIC PLACEHOLDER) -------------------
def fetch_tiktok_data(keyword, max_results):
    # ⚠️ TikTok scraping is restricted (no official API)
    # This is a placeholder you can later upgrade

    data = []

    for i in range(max_results):
        data.append({
            "Text": f"{keyword} TikTok sample {i}",
            "Source": "TikTok"
        })

    return data

# ------------------- SENTIMENT ANALYSIS -------------------
def analyze_sentiment(texts):
    results = sentiment_model(texts)

    sentiments = []
    for r in results:
        label = r["label"]
        if label == "POSITIVE":
            sentiments.append("Positive")
        else:
            sentiments.append("Negative")

    return sentiments

# ------------------- MAIN FUNCTION -------------------
def run_collection(keyword, platform, language, max_records, progress_callback=None):
    try:
        all_data = []

        # ---------------- PLATFORM SELECTION ----------------
        if platform == "YouTube":
            all_data = fetch_youtube_data(keyword, max_records)

        elif platform == "TikTok":
            all_data = fetch_tiktok_data(keyword, max_records)

        else:
            raise ValueError("Unsupported platform")

        # ---------------- PROGRESS UPDATE ----------------
        if progress_callback:
            progress_callback(0.5)

        # ---------------- SENTIMENT ----------------
        texts = [item["Text"] for item in all_data]

        sentiments = analyze_sentiment(texts)

        # Attach sentiment
        for i, item in enumerate(all_data):
            item["Sentiment"] = sentiments[i]

        # ---------------- FINAL PROGRESS ----------------
        if progress_callback:
            progress_callback(1.0)

        return pd.DataFrame(all_data)

    except Exception as e:
        raise Exception(f"Backend error: {e}")
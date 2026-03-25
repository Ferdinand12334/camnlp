
# youtube_sentiment_single_video.py
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
from textblob import TextBlob
import re
import time

st.set_page_config(page_title="YouTube Sentiment Analyzer", layout="wide")
st.title("YouTube Single Video Comment Sentiment Analyzer")

# -----------------------------
# 1. USER INPUT
# -----------------------------
video_id_input = st.text_input("Enter YouTube Video ID or URL", "jEugVbzthAM")
API_KEY = st.text_input("Enter your YouTube API Key", "AIzaSyC1nA7LRFhYIV8w5xudqE41KvN1VBa7AXg")
max_comments = st.number_input("Max comments to scrape", min_value=10, max_value=1000, value=200)

# -----------------------------
# 2. HELPER FUNCTIONS
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    return text

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

def extract_video_id(url_or_id):
    # If user enters full URL, extract the ID
    if "youtu" in url_or_id:
        if "watch?v=" in url_or_id:
            return url_or_id.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url_or_id:
            return url_or_id.split("youtu.be/")[1].split("?")[0]
    return url_or_id

# -----------------------------
# 3. SCRAPING LOGIC
# -----------------------------
if st.button("Scrape & Analyze"):

    if not API_KEY or not video_id_input:
        st.error("Please enter both a YouTube video ID/URL and API Key!")
    else:
        video_id = extract_video_id(video_id_input)
        youtube = build("youtube", "v3", developerKey=API_KEY)
        comments = []

        st.info(f"Scraping comments for video: {video_id}")
        next_page_token = None
        comments_collected = 0

        while True:
            try:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat="plainText"
                )
                response = request.execute()
            except HttpError as e:
                if e.resp.status == 403:
                    st.warning("Comments disabled or inaccessible for this video.")
                    break
                else:
                    st.error(f"Error: {e}")
                    break

            for item in response.get("items", []):
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
                comments_collected += 1
                if comments_collected >= max_comments:
                    break

            next_page_token = response.get("nextPageToken")
            if not next_page_token or comments_collected >= max_comments:
                break

            time.sleep(1)

        st.success(f"Total comments collected: {len(comments)}")

        if comments:
            df = pd.DataFrame(comments, columns=["comment"])
            df["clean_comment"] = df["comment"].apply(clean_text)
            df["sentiment"] = df["clean_comment"].apply(get_sentiment)

            st.subheader("Sample Comments")
            st.dataframe(df.head(20))

            st.subheader("Sentiment Distribution")
            st.bar_chart(df["sentiment"].value_counts())

            df.to_csv("youtube_single_video_comments.csv", index=False)
            st.success("Dataset saved as 'youtube_single_video_comments.csv'")
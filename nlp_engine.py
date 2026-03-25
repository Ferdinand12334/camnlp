# nlp_engine.py
from textblob import TextBlob
import re

# Simple topic keywords mapping
TOPICS = {
    1: {"name": "Security",   "keywords": ["security","attack","protest","crisis","conflict","army","soldier","violence"]},
    2: {"name": "Economy",    "keywords": ["economy","price","trade","inflation","business","market","money","investment"]},
    3: {"name": "Education",  "keywords": ["school","student","education","university","teacher","reform","learning"]},
    4: {"name": "Healthcare", "keywords": ["health","hospital","doctor","medicine","disease","clinic","nurse","treatment"]},
    5: {"name": "Politics",   "keywords": ["government","president","election","policy","minister","parliament","law"]},
}

def detect_language(text):
    """Simple French/English detector based on common French words."""
    french_words = ["le","la","les","un","une","des","est","et","en","du","de","au","je","il","elle","nous","vous","ils"]
    words = text.lower().split()
    french_count = sum(1 for w in words if w in french_words)
    return "fr" if french_count >= 2 else "en"

def detect_topic(text):
    """Match text to the most relevant topic by keyword count."""
    text_lower = text.lower()
    scores = {}
    for topic_id, topic in TOPICS.items():
        scores[topic_id] = sum(1 for kw in topic["keywords"] if kw in text_lower)
    best = max(scores, key=scores.get)
    # Default to Politics if no match
    return best if scores[best] > 0 else 5

def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)           # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)      # strip hashtag symbol
    text = re.sub(r"[^\w\s]", " ", text)       # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # collapse spaces
    return text.lower()

def analyze_text(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        sentiment_label = "Positive"
    elif polarity < -0.1:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    topic_id   = detect_topic(text)
    topic_name = TOPICS[topic_id]["name"]
    language   = detect_language(text)
    cleaned    = clean_text(text)

    return {
        "sentiment_label": sentiment_label,   # ✅ matches app.py & data_collector.py
        "polarity_score":  round(polarity, 4),# ✅ matches app.py & data_collector.py
        "topic_id":        topic_id,          # ✅ needed for AnalysisResult
        "topic_name":      topic_name,        # ✅ needed for Analyze Text page
        "language":        language,          # ✅ needed for Analyze Text page
        "cleaned_text":    cleaned,           # ✅ needed for Analyze Text page
    }
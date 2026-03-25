# data_collector.py
import requests
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime
import time
import random

from database import get_session, TextData, AnalysisResult
from nlp_engine import analyze_text

# ── Headers to avoid being blocked ────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# ── Source configurations ──────────────────────────────────────────────────────
SOURCES = {
    "BBC Africa": {
        "type": "rss",
        "url":  "https://feeds.bbci.co.uk/africa/rss.xml",
        "platform": "BBC Africa",
        "language": "en",
    },
    "Cameroon Tribune": {
        "type": "rss",
        "url":  "https://www.cameroon-tribune.cm/rss.xml",
        "platform": "Cameroon Tribune",
        "language": "fr",
    },
    "237online": {
        "type": "scrape",
        "url":  "https://www.237online.com/",
        "platform": "237online",
        "language": "fr",
        "article_selector": "h2 a, h3 a",
    },
    "CamTimes": {
        "type": "scrape",
        "url":  "https://camtimes.org/",
        "platform": "CamTimes",
        "language": "en",
        "article_selector": "h2 a, h3 a",
    },
}

# ── Cameroon keywords for filtering relevant articles ─────────────────────────
CAMEROON_KEYWORDS = [
    "cameroon", "cameroun", "yaoundé", "yaounde", "douala", "bamenda",
    "anglophone", "francophone", "biya", "cpdm", "mrc", "sos", "ambazonia",
    "boko haram", "far north", "extrême-nord", "kribi", "buea", "limbe",
    "northwest", "southwest", "adamawa", "centre region", "littoral",
]


def is_cameroon_related(text):
    """Return True if text mentions Cameroon or related topics."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in CAMEROON_KEYWORDS)


# ── RSS Scraper ────────────────────────────────────────────────────────────────
def scrape_rss(source_name, config, max_records, start_date, end_date):
    """Scrape articles from an RSS feed."""
    articles = []
    try:
        feed = feedparser.parse(config["url"])
        for entry in feed.entries:
            if len(articles) >= max_records:
                break

            title   = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            text    = f"{title}. {summary}" if summary else title

            if not text or len(text) < 20:
                continue

            # Filter BBC to Cameroon-related only
            if source_name == "BBC Africa" and not is_cameroon_related(text):
                continue

            # Parse date
            published = entry.get("published_parsed") or entry.get("updated_parsed")
            if published:
                pub_date = datetime(*published[:6])
            else:
                pub_date = datetime.utcnow()

            # Date range filter
            if pub_date.date() < start_date.date() or pub_date.date() > end_date.date():
                continue

            articles.append({
                "text":     text,
                "platform": config["platform"],
                "language": config["language"],
                "date":     pub_date,
                "source":   entry.get("link", config["url"]),
            })

    except Exception as e:
        print(f"[RSS ERROR] {source_name}: {e}")

    return articles


# ── HTML Scraper ───────────────────────────────────────────────────────────────
def scrape_html(source_name, config, max_records, start_date, end_date):
    """Scrape article headlines from an HTML page."""
    articles = []
    try:
        resp = requests.get(config["url"], headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        links = soup.select(config["article_selector"])
        seen  = set()

        for tag in links:
            if len(articles) >= max_records:
                break

            text = tag.get_text(separator=" ").strip()
            href = tag.get("href", "")

            if not text or len(text) < 20 or text in seen:
                continue
            seen.add(text)

            # Try to fetch article body for more content
            full_text = text
            if href and href.startswith("http"):
                try:
                    art_resp = requests.get(href, headers=HEADERS, timeout=10)
                    art_soup = BeautifulSoup(art_resp.text, "html.parser")
                    paragraphs = art_soup.select("article p, .entry-content p, .post-content p")
                    body = " ".join(p.get_text() for p in paragraphs[:4])
                    if body.strip():
                        full_text = f"{text}. {body[:400]}"
                except Exception:
                    pass  # Fall back to headline only

            articles.append({
                "text":     full_text[:600],
                "platform": config["platform"],
                "language": config["language"],
                "date":     datetime.utcnow(),
                "source":   href or config["url"],
            })

            time.sleep(0.5)  # polite delay

    except Exception as e:
        print(f"[HTML ERROR] {source_name}: {e}")

    return articles


# ── Twitter-like scraper via Nitter (no API needed) ───────────────────────────
def scrape_nitter(keyword, max_records, start_date, end_date):
    """Scrape tweets via Nitter public instances (no API key needed)."""
    articles = []
    nitter_instances = [
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.1d4.us",
    ]
    query = keyword.replace("#", "%23").replace(" ", "+")

    for instance in nitter_instances:
        try:
            url  = f"{instance}/search?q={query}&f=tweets"
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                continue

            soup   = BeautifulSoup(resp.text, "html.parser")
            tweets = soup.select(".tweet-content")

            for tweet in tweets:
                if len(articles) >= max_records:
                    break
                text = tweet.get_text(separator=" ").strip()
                if not text or len(text) < 15:
                    continue
                articles.append({
                    "text":     text[:500],
                    "platform": "Twitter",
                    "language": "en",
                    "date":     datetime.utcnow(),
                    "source":   url,
                })

            if articles:
                break  # Stop trying instances once one works

        except Exception as e:
            print(f"[NITTER ERROR] {instance}: {e}")
            continue

    return articles


# ── Save to Database ───────────────────────────────────────────────────────────
def save_to_db(articles):
    """Analyze and save articles to database. Returns (saved, duplicates, filtered)."""
    session    = get_session()
    saved      = 0
    duplicates = 0
    filtered   = 0

    for art in articles:
        text = art["text"].strip()
        if not text:
            filtered += 1
            continue

        # Duplicate check
        existing = session.query(TextData).filter_by(
            content=text[:500], platform=art["platform"]
        ).first()
        if existing:
            duplicates += 1
            continue

        # NLP analysis
        try:
            analysis = analyze_text(text)
        except Exception as e:
            print(f"[NLP ERROR] {e}")
            filtered += 1
            continue

        # Save TextData
        td = TextData(
            content         = text[:1000],
            source          = art.get("source", ""),
            platform        = art["platform"],
            language        = analysis.get("language") or art["language"],
            collection_date = art["date"],
        )
        session.add(td)
        session.flush()

        # Save AnalysisResult
        ar = AnalysisResult(
            data_id         = td.data_id,
            sentiment_label = analysis["sentiment_label"],
            polarity_score  = analysis["polarity_score"],
            topic_id        = analysis["topic_id"],
            analysis_date   = datetime.utcnow(),
        )
        session.add(ar)
        saved += 1

    session.commit()
    session.close()
    return saved, duplicates, filtered


# ── Main Entry Point ───────────────────────────────────────────────────────────
def run_collection(keyword=None, platform=None, language=None,
                   start_date=None, end_date=None,
                   max_records=50, progress_callback=None):

    start_date = start_date or datetime(2020, 1, 1)
    end_date   = end_date   or datetime.utcnow()

    all_articles = []
    total_steps  = len(SOURCES) + 1  # +1 for Twitter
    step         = 0

    # ── Decide which sources to scrape ────────────────────────────────────────
    per_source = max(max_records // (total_steps), 5)

    # ── Scrape each news source ────────────────────────────────────────────────
    for source_name, config in SOURCES.items():

        # Platform filter
        if platform and platform != "All" and config["platform"] != platform:
            step += 1
            continue

        if progress_callback:
            progress_callback(step + 1, total_steps + 1,
                              f"Scraping {source_name}...")

        if config["type"] == "rss":
            articles = scrape_rss(source_name, config, per_source, start_date, end_date)
        else:
            articles = scrape_html(source_name, config, per_source, start_date, end_date)

        all_articles.extend(articles)
        step += 1

        if progress_callback:
            progress_callback(step, total_steps + 1,
                              f"Got {len(articles)} articles from {source_name}")
        time.sleep(1)  # polite delay between sources

    # ── Scrape Twitter via Nitter ──────────────────────────────────────────────
    if platform in (None, "All", "Twitter"):
        if progress_callback:
            progress_callback(step + 1, total_steps + 1,
                              f"Scraping Twitter for '{keyword}'...")

        kw       = keyword or "Cameroon"
        tweets   = scrape_nitter(kw, per_source, start_date, end_date)
        all_articles.extend(tweets)

        if progress_callback:
            progress_callback(step + 1, total_steps + 1,
                              f"Got {len(tweets)} tweets")

    # ── Shuffle & limit ───────────────────────────────────────────────────────
    random.shuffle(all_articles)
    all_articles = all_articles[:max_records]

    collected = len(all_articles)

    if progress_callback:
        progress_callback(total_steps + 1, total_steps + 1,
                          f"Saving {collected} articles to database...")

    # ── Save everything ───────────────────────────────────────────────────────
    saved, duplicates, filtered = save_to_db(all_articles)

    return {
        "collected":  collected,
        "saved":      saved,
        "duplicates": duplicates,
        "filtered":   filtered,
    }
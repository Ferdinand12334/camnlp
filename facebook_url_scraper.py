"""
facebook_url_scraper.py
=======================
Facebook URL-based Post Scraper Module
Cameroon NLP Societal Analysis System — Data Collection Extension

HOW TO INTEGRATE:
  1. Install dependencies:
       pip install requests beautifulsoup4 selenium webdriver-manager playwright
       playwright install chromium
  2. Add this file to your project folder alongside your main Streamlit app.
  3. In your main app, import and call render_facebook_scraper_tab() inside
     a Streamlit tab or sidebar section.

USAGE MODES (choose one based on your environment):
  - Mode A: requests + BeautifulSoup  → works for PUBLIC pages (no login needed)
  - Mode B: Selenium / Playwright     → works for posts that require login
  - Mode C: Facebook Graph API        → official, requires access token
"""

import re
import time
import sqlite3
import hashlib
import datetime
import requests
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urlencode

# ── Optional imports (graceful degradation) ──────────────────────────────────
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_facebook_url(url: str) -> bool:
    """Return True if *url* looks like a Facebook post / page / group URL."""
    parsed = urlparse(url)
    return parsed.netloc in ("www.facebook.com", "facebook.com", "m.facebook.com", "fb.com")


def clean_text(raw: str) -> str:
    """Basic cleaning: strip extra whitespace, remove control characters."""
    if not raw:
        return ""
    raw = re.sub(r"[\r\n\t]+", " ", raw)
    raw = re.sub(r"\s{2,}", " ", raw)
    return raw.strip()


def detect_language(text: str) -> str:
    """
    Very lightweight language guesser based on common stopwords.
    Replace with langdetect or the system's existing detector for better accuracy.
    """
    fr_words = {"le", "la", "les", "de", "du", "des", "et", "en", "un", "une",
                "est", "pour", "avec", "dans", "sur", "par", "que", "qui"}
    en_words = {"the", "is", "are", "was", "were", "and", "of", "in", "to",
                "a", "an", "for", "with", "on", "by", "that", "this"}
    tokens = set(text.lower().split())
    fr_score = len(tokens & fr_words)
    en_score = len(tokens & en_words)
    if fr_score == 0 and en_score == 0:
        return "unknown"
    return "fr" if fr_score > en_score else "en"


def make_record_id(url: str, text: str) -> str:
    """Deterministic dedup key from URL + first 100 chars of text."""
    return hashlib.md5(f"{url}{text[:100]}".encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODE A — requests + BeautifulSoup (public content, no login)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_with_requests(url: str) -> dict:
    """
    Fetch a public Facebook URL and extract visible text using BeautifulSoup.

    Works best for:
      - Public Facebook Pages  (facebook.com/<pagename>)
      - Public posts shared with 'Everyone'

    Does NOT work for:
      - Posts requiring login
      - Private / friends-only content
      - Dynamic content loaded by JavaScript

    Returns a dict with keys: url, text, author, timestamp, likes, comments,
                              platform, language, status
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,fr;q=0.8",
    }

    result = {
        "url": url,
        "text": "",
        "author": "Unknown",
        "timestamp": datetime.datetime.now().isoformat(),
        "likes": 0,
        "comments": 0,
        "platform": "Facebook",
        "language": "unknown",
        "status": "failed",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # ── Extract text from common Facebook HTML patterns ──────────────────
        # Meta description (most reliable for public pages)
        meta_desc = soup.find("meta", {"name": "description"}) or \
                    soup.find("meta", {"property": "og:description"})
        if meta_desc and meta_desc.get("content"):
            result["text"] = clean_text(meta_desc["content"])

        # og:title for author / page name
        og_title = soup.find("meta", {"property": "og:title"})
        if og_title and og_title.get("content"):
            result["author"] = og_title["content"].split("|")[0].strip()

        # Try article body or main content divs (varies by FB version)
        if not result["text"]:
            for selector in [
                {"data-testid": "post_message"},
                {"class": re.compile(r"userContent|_5pbx|_5rgt|_6a|story_body_container")},
            ]:
                el = soup.find("div", selector)
                if el:
                    result["text"] = clean_text(el.get_text(separator=" "))
                    break

        # Fallback: all paragraph text
        if not result["text"]:
            paras = [p.get_text(" ") for p in soup.find_all("p")]
            result["text"] = clean_text(" ".join(paras))

        if result["text"]:
            result["language"] = detect_language(result["text"])
            result["status"] = "success"
        else:
            result["status"] = "empty_content"

    except requests.exceptions.RequestException as exc:
        result["status"] = f"request_error: {exc}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODE B — Selenium (login-aware, JS-rendered content)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_with_selenium(url: str, email: str = "", password: str = "") -> dict:
    """
    Use a headless Chromium browser to load a Facebook URL, optionally log in,
    and extract post text.

    Parameters
    ----------
    url      : Full Facebook post or page URL
    email    : Facebook account email (required for private posts)
    password : Facebook account password (required for private posts)

    ⚠️  IMPORTANT: Logging into Facebook programmatically may violate Facebook's
        Terms of Service. Use only for posts you own or have explicit permission
        to scrape, and comply with local data-protection laws (e.g. GDPR).
    """
    if not SELENIUM_AVAILABLE:
        return {
            "url": url, "text": "", "status": "selenium_not_installed",
            "platform": "Facebook", "language": "unknown",
            "author": "", "timestamp": datetime.datetime.now().isoformat(),
            "likes": 0, "comments": 0,
        }

    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    result = {
        "url": url, "text": "", "author": "Unknown",
        "timestamp": datetime.datetime.now().isoformat(),
        "likes": 0, "comments": 0, "platform": "Facebook",
        "language": "unknown", "status": "failed",
    }

    driver = None
    try:
        driver = webdriver.Chrome(
            ChromeDriverManager().install(), options=options
        )
        driver.implicitly_wait(10)

        # ── Optional login ────────────────────────────────────────────────────
        if email and password:
            driver.get("https://www.facebook.com/login")
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.ID, "email")))
            driver.find_element(By.ID, "email").send_keys(email)
            driver.find_element(By.ID, "pass").send_keys(password)
            driver.find_element(By.NAME, "login").click()
            time.sleep(4)   # wait for redirect / 2FA prompt

        # ── Load target URL ───────────────────────────────────────────────────
        driver.get(url)
        time.sleep(3)       # allow JS to render

        # Dismiss cookie / login popups if present
        for dismiss_xpath in [
            "//div[@aria-label='Close']",
            "//button[contains(text(),'Not Now')]",
            "//button[contains(text(),'Decline')]",
        ]:
            try:
                btn = driver.find_element(By.XPATH, dismiss_xpath)
                btn.click()
                time.sleep(1)
            except Exception:
                pass

        # ── Extract text ──────────────────────────────────────────────────────
        text_elements = driver.find_elements(
            By.XPATH,
            "//div[@data-testid='post_message'] | "
            "//div[contains(@class,'userContent')] | "
            "//div[@role='article']//div[@dir='auto']"
        )
        collected = [clean_text(el.text) for el in text_elements if el.text.strip()]
        raw_text = " ".join(dict.fromkeys(collected))   # deduplicate while preserving order

        if raw_text:
            result["text"] = raw_text
            result["language"] = detect_language(raw_text)
            result["status"] = "success"
        else:
            result["status"] = "empty_content"

    except Exception as exc:
        result["status"] = f"selenium_error: {exc}"
    finally:
        if driver:
            driver.quit()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODE C — Facebook Graph API (official, token-required)
# ─────────────────────────────────────────────────────────────────────────────

def scrape_with_graph_api(post_url: str, access_token: str) -> dict:
    """
    Retrieve a Facebook post via the official Graph API.

    Requirements
    ------------
    - A valid user or page access token with at least:
        pages_read_engagement, pages_read_user_content
    - The post ID (extracted from the URL automatically here).

    Supported URL formats
    ---------------------
    - https://www.facebook.com/<page>/posts/<post_id>
    - https://www.facebook.com/photo?fbid=<id>
    - https://www.facebook.com/permalink.php?story_fbid=<id>&id=<page_id>
    """
    result = {
        "url": post_url, "text": "", "author": "Unknown",
        "timestamp": datetime.datetime.now().isoformat(),
        "likes": 0, "comments": 0, "platform": "Facebook",
        "language": "unknown", "status": "failed",
    }

    # ── Extract post ID from URL ──────────────────────────────────────────────
    post_id = None
    patterns = [
        r"facebook\.com/.+/posts/(\d+)",          # page post
        r"facebook\.com/photo\?fbid=(\d+)",        # photo
        r"story_fbid=(\d+)",                        # permalink
        r"facebook\.com/permalink\.php.*fbid=(\d+)",
        r"facebook\.com/(\d+)/posts/(\d+)",         # numeric page + post
    ]
    for pat in patterns:
        m = re.search(pat, post_url)
        if m:
            post_id = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
            break

    if not post_id:
        result["status"] = "could_not_extract_post_id"
        return result

    # ── Call Graph API ────────────────────────────────────────────────────────
    api_url = f"https://graph.facebook.com/v19.0/{post_id}"
    params = {
        "fields": "id,message,story,created_time,from,likes.summary(true),comments.summary(true)",
        "access_token": access_token,
    }
    try:
        resp = requests.get(api_url, params=params, timeout=15)
        data = resp.json()

        if "error" in data:
            result["status"] = f"api_error: {data['error'].get('message', 'unknown')}"
            return result

        text = data.get("message") or data.get("story") or ""
        result["text"] = clean_text(text)
        result["author"] = data.get("from", {}).get("name", "Unknown")
        result["timestamp"] = data.get("created_time", result["timestamp"])
        result["likes"] = data.get("likes", {}).get("summary", {}).get("total_count", 0)
        result["comments"] = data.get("comments", {}).get("summary", {}).get("total_count", 0)

        if result["text"]:
            result["language"] = detect_language(result["text"])
            result["status"] = "success"
        else:
            result["status"] = "empty_content"

    except requests.exceptions.RequestException as exc:
        result["status"] = f"request_error: {exc}"

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MULTI-URL BATCH SCRAPER
# ─────────────────────────────────────────────────────────────────────────────

def batch_scrape(
    urls: list[str],
    mode: str = "requests",
    email: str = "",
    password: str = "",
    access_token: str = "",
    delay: float = 2.0,
    progress_callback=None,
) -> list[dict]:
    """
    Scrape a list of Facebook URLs using the chosen mode.

    Parameters
    ----------
    urls              : List of Facebook post / page URLs
    mode              : 'requests' | 'selenium' | 'graph_api'
    email / password  : Only used for mode='selenium' with login
    access_token      : Only used for mode='graph_api'
    delay             : Seconds to wait between requests (be polite)
    progress_callback : Optional callable(current, total) for progress updates
    """
    results = []
    seen_ids = set()

    for i, url in enumerate(urls, 1):
        url = url.strip()
        if not url:
            continue
        if not validate_facebook_url(url):
            results.append({
                "url": url, "text": "", "status": "invalid_facebook_url",
                "platform": "Facebook", "language": "unknown",
                "author": "", "timestamp": datetime.datetime.now().isoformat(),
                "likes": 0, "comments": 0,
            })
            continue

        if mode == "selenium":
            rec = scrape_with_selenium(url, email, password)
        elif mode == "graph_api":
            rec = scrape_with_graph_api(url, access_token)
        else:
            rec = scrape_with_requests(url)

        # Deduplicate by content hash
        rid = make_record_id(url, rec["text"])
        if rid in seen_ids:
            rec["status"] = "duplicate"
        else:
            seen_ids.add(rid)

        rec["record_id"] = rid
        results.append(rec)

        if progress_callback:
            progress_callback(i, len(urls))

        if i < len(urls):
            time.sleep(delay)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DATABASE INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def save_to_database(records: list[dict], db_path: str = "nlp_system.db"):
    """
    Insert successfully scraped records into the TextData table used by
    the main Cameroon NLP Societal Analysis System.

    The schema matches Table 3.1 from Chapter 3 (Methodology) with an
    additional 'record_id' dedup column.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist (mirrors Chapter 3 schema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS TextData (
            data_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id      TEXT UNIQUE,
            content        TEXT NOT NULL,
            source         VARCHAR(255),
            platform       VARCHAR(100),
            language       VARCHAR(10),
            author         VARCHAR(255),
            likes          INTEGER DEFAULT 0,
            comments_count INTEGER DEFAULT 0,
            collection_date DATETIME
        )
    """)

    inserted = 0
    skipped = 0
    for rec in records:
        if rec["status"] != "success" or not rec["text"]:
            skipped += 1
            continue
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO TextData
                    (record_id, content, source, platform, language,
                     author, likes, comments_count, collection_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec.get("record_id", ""),
                rec["text"],
                rec["url"],
                rec["platform"],
                rec["language"],
                rec.get("author", "Unknown"),
                rec.get("likes", 0),
                rec.get("comments", 0),
                rec.get("timestamp", datetime.datetime.now().isoformat()),
            ))
            if cursor.rowcount:
                inserted += 1
            else:
                skipped += 1
        except sqlite3.Error:
            skipped += 1

    conn.commit()
    conn.close()
    return inserted, skipped


# ─────────────────────────────────────────────────────────────────────────────
# 7.  STREAMLIT UI  — call this inside your main Streamlit app
# ─────────────────────────────────────────────────────────────────────────────

def render_facebook_scraper_tab(db_path: str = "nlp_system.db"):
    """
    Render the Facebook URL Scraper tab inside your Streamlit application.

    Usage in your main app:
    -----------------------
        import streamlit as st
        from facebook_url_scraper import render_facebook_scraper_tab

        tab1, tab2 = st.tabs(["Main Dashboard", "Facebook URL Scraper"])
        with tab2:
            render_facebook_scraper_tab()
    """
    st.header("🔗 Facebook URL Scraper")
    st.markdown(
        "Paste Facebook post or page URLs below to extract text for NLP analysis. "
        "Scraped content is saved directly to the system database."
    )

    # ── Mode selector ─────────────────────────────────────────────────────────
    st.subheader("⚙️ Scraping Mode")
    mode = st.radio(
        "Choose scraping method:",
        options=["requests (public pages, no login)", "graph_api (official API)", "selenium (login-aware)"],
        index=0,
        help=(
            "• **requests** — fastest, works on fully public Facebook pages/posts. "
            "No login needed.\n"
            "• **graph_api** — official method; requires a Facebook access token. "
            "Most reliable for pages you manage.\n"
            "• **selenium** — launches a real browser; can log in to access "
            "friends-only posts. Slower and more complex."
        ),
    )
    mode_key = mode.split(" ")[0]   # 'requests' | 'graph_api' | 'selenium'

    # ── Credentials (shown only when needed) ──────────────────────────────────
    email, password, access_token = "", "", ""

    if mode_key == "selenium":
        st.info(
            "⚠️ Providing your Facebook credentials is optional. "
            "Only enter them if you need to access non-public posts. "
            "Credentials are used only for this session and never stored."
        )
        with st.expander("Facebook Login (optional)"):
            email    = st.text_input("Facebook Email", type="default", key="fb_email")
            password = st.text_input("Facebook Password", type="password", key="fb_pass")

    elif mode_key == "graph_api":
        st.info(
            "You need a Facebook User or Page Access Token. "
            "Generate one at: https://developers.facebook.com/tools/explorer/"
        )
        access_token = st.text_input(
            "Access Token", type="password", key="fb_token",
            placeholder="Paste your Facebook Graph API access token here"
        )

    # ── URL input ─────────────────────────────────────────────────────────────
    st.subheader("📋 Enter Facebook URLs")
    url_input = st.text_area(
        "One URL per line",
        placeholder=(
            "https://www.facebook.com/cameroon.info/posts/123456789\n"
            "https://www.facebook.com/groups/anglophonecrisis/posts/987654321\n"
            "https://www.facebook.com/rfi.afrique"
        ),
        height=180,
        key="fb_urls",
    )

    col1, col2 = st.columns(2)
    with col1:
        delay = st.slider(
            "Delay between requests (seconds)",
            min_value=1.0, max_value=10.0, value=2.0, step=0.5,
            help="Increase this if Facebook rate-limits you."
        )
    with col2:
        save_to_db = st.checkbox("Save to system database", value=True)

    # ── Run button ────────────────────────────────────────────────────────────
    if st.button("🚀 Start Scraping", type="primary"):
        urls = [u.strip() for u in url_input.splitlines() if u.strip()]

        if not urls:
            st.warning("Please enter at least one URL.")
            return

        if mode_key == "graph_api" and not access_token:
            st.error("Please provide a Facebook Access Token for Graph API mode.")
            return

        if not SELENIUM_AVAILABLE and mode_key == "selenium":
            st.error(
                "Selenium is not installed. Run: "
                "`pip install selenium webdriver-manager` then restart the app."
            )
            return

        # Progress tracking
        progress_bar  = st.progress(0)
        status_text   = st.empty()

        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Scraping {current} / {total} URLs…")

        with st.spinner("Scraping in progress…"):
            results = batch_scrape(
                urls=urls,
                mode=mode_key,
                email=email,
                password=password,
                access_token=access_token,
                delay=delay,
                progress_callback=update_progress,
            )

        progress_bar.progress(1.0)
        status_text.text("✅ Scraping complete!")

        # ── Save to DB ────────────────────────────────────────────────────────
        if save_to_db:
            inserted, skipped = save_to_database(results, db_path)
            st.success(f"Saved **{inserted}** new records to database. ({skipped} skipped / duplicates)")

        # ── Results table ─────────────────────────────────────────────────────
        st.subheader("📊 Scraping Results")
        df = pd.DataFrame(results)

        # Friendly column order
        cols = ["url", "author", "language", "likes", "comments", "status", "text"]
        df = df[[c for c in cols if c in df.columns]]

        # Truncate text for display
        if "text" in df.columns:
            df["text_preview"] = df["text"].str[:120] + "…"
            df = df.drop(columns=["text"])
            df = df.rename(columns={"text_preview": "text (preview)"})

        # Color-code status
        def highlight_status(val):
            if val == "success":
                return "background-color: #d4edda; color: #155724"
            elif val in ("duplicate", "empty_content"):
                return "background-color: #fff3cd; color: #856404"
            else:
                return "background-color: #f8d7da; color: #721c24"

        st.dataframe(
            df.style.applymap(highlight_status, subset=["status"]),
            use_container_width=True,
        )

        # Download CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download results as CSV",
            data=csv,
            file_name=f"facebook_scraped_{datetime.date.today()}.csv",
            mime="text/csv",
        )

        # Summary metrics
        success_count  = sum(1 for r in results if r["status"] == "success")
        failed_count   = len(results) - success_count
        lang_counts    = {}
        for r in results:
            if r["status"] == "success":
                lang_counts[r.get("language", "unknown")] = \
                    lang_counts.get(r.get("language", "unknown"), 0) + 1

        st.subheader("📈 Collection Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total URLs", len(results))
        m2.metric("✅ Successful", success_count)
        m3.metric("❌ Failed", failed_count)
        m4.metric("🇫🇷 French / 🇬🇧 English",
                  f"{lang_counts.get('fr', 0)} / {lang_counts.get('en', 0)}")

        # Pass to NLP pipeline hint
        if success_count > 0:
            st.info(
                "💡 **Next step:** Go to the **Sentiment Analysis** or **Topic Modeling** tab "
                "to run NLP analysis on the newly collected records."
            )


# ─────────────────────────────────────────────────────────────────────────────
# 8.  STANDALONE TEST  (run: python facebook_url_scraper.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_urls = [
        "https://www.facebook.com/rfi.afrique",           # public page
        "https://www.facebook.com/cameroon.info",         # public page
        "https://www.facebook.com/NOT_A_REAL_PAGE_XYZ",  # should fail gracefully
    ]

    print("Running standalone test with requests mode…\n")
    records = batch_scrape(test_urls, mode="requests", delay=1.5)

    for rec in records:
        print(f"URL     : {rec['url']}")
        print(f"Status  : {rec['status']}")
        print(f"Language: {rec['language']}")
        print(f"Text    : {rec['text'][:120]}…" if rec['text'] else "Text    : (empty)")
        print("-" * 60)

    # Test DB save
    inserted, skipped = save_to_database(records, "test_nlp.db")
    print(f"\nDatabase: {inserted} inserted, {skipped} skipped.")
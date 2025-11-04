# news_alert.py
import os, json, time, datetime as dt, hashlib
from pathlib import Path
from urllib.parse import quote_plus
import requests, feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------- CONFIG ----------
WATCHLIST = {
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    # add more: "MSFT": "Microsoft",
}
LOOKBACK_MIN = 180            # cluster window (minutes)
STRONG_POS = 0.6              # VADER strong positive threshold
STRONG_NEG = -0.6             # VADER strong negative threshold
CLUSTER_COUNT = 3             # strong headlines needed to alert
ALERT_COOLDOWN_MIN = 120      # min gap between alerts per ticker
POLL_NEWS_HOURS = 6           # hint window for Google News RSS

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_PATH = Path("state.json")
STATE_CHANGED_FLAG = Path(".state_changed")  # marker file for workflow to commit

an = SentimentIntensityAnalyzer()

def now_ts() -> int:
    return int(dt.datetime.utcnow().timestamp())

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"seen_ids": {}, "last_alert": {}}
    # seen_ids: {ticker: {sha1(url): pub_ts}}
    # last_alert: {ticker: ts}

def save_state(state):
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True))
    STATE_CHANGED_FLAG.write_text("changed")

def google_news_rss(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}+when:{POLL_NEWS_HOURS}h&hl=en-US&gl=US&ceid=US:en"

def senti(text: str) -> float:
    return an.polarity_scores((text or "")[:2000])["compound"]

def get_pub_ts(entry):
    for k in ("published_parsed", "updated_parsed"):
        t = getattr(entry, k, None)
        if t:
            return int(time.mktime(t))
    return now_ts()

def fetch_items(ticker: str, name: str):
    q = f'"{name}" OR {ticker} OR ${ticker}'
    feed = feedparser.parse(google_news_rss(q))
    items = []
    for e in feed.entries:
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        link = getattr(e, "link", "")
        score = senti(f"{title} {summary}")
        items.append({
            "title": title, "url": link, "score": score, "pub_ts": get_pub_ts(e)
        })
    return items

def notify_telegram(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Would notify (missing TELEGRAM_* env):\n", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("Telegram error:", e)

def maybe_alert(ticker: str, hits: list, state: dict):
    # Cooldown check
    last = state["last_alert"].get(ticker, 0)
    if now_ts() - last < ALERT_COOLDOWN_MIN * 60:
        return False
    # Prepare message
    verdict = "could rise significantly" if hits[0]["score"] > 0 else "could decrease significantly"
    top = sorted(hits, key=lambda x: -abs(x["score"]))[:5]
    lines = [f"{ticker} {verdict} (last {LOOKBACK_MIN}m):"]
    for h in top:
        lines.append(f"• {h['title']} [{h['url']}] (score {h['score']:+.2f})")
    notify_telegram("\n".join(lines))
    state["last_alert"][ticker] = now_ts()
    save_state(state)
    return True

def sha(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()

def run_once():
    changed = False
    state = load_state()
    cutoff_ts = now_ts() - LOOKBACK_MIN * 60

    for tkr, name in WATCHLIST.items():
        seen = state["seen_ids"].setdefault(tkr, {})
        # Pull and dedupe by URL hash
        items = fetch_items(tkr, name)
        window_items = []
        for it in items:
            h = sha(it["url"])
            if h not in seen:
                seen[h] = it["pub_ts"]
                changed = True
            # Keep cluster candidates in window
            if it["pub_ts"] >= cutoff_ts:
                window_items.append(it)

        pos = [x for x in window_items if x["score"] >= STRONG_POS]
        neg = [x for x in window_items if x["score"] <= STRONG_NEG]

        if len(pos) >= CLUSTER_COUNT:
            if maybe_alert(tkr, pos, state):
                changed = True
        elif len(neg) >= CLUSTER_COUNT:
            if maybe_alert(tkr, neg, state):
                changed = True

    if changed:
        save_state(state)

if __name__ == "__main__":
    run_once()

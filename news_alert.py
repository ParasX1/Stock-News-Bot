import os, json, time, datetime as dt, hashlib, re, requests, feedparser
from pathlib import Path
from urllib.parse import quote_plus
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ================= CONFIG =================
WATCHLIST = {
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
}
LOOKBACK_MIN = 180            # window for clustering (minutes)
STRONG_POS = 0.6              # VADER positive threshold
STRONG_NEG = -0.6              # VADER negative threshold
CLUSTER_COUNT = 3             # how many strong hits to alert
ALERT_COOLDOWN_MIN = 120      # min gap between alerts per ticker
POLL_NEWS_HOURS = 6           # how far back to query in Google News
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_PATH = Path("state.json")
an = SentimentIntensityAnalyzer()

# ================= UTILITIES =================
def now_ts(): return int(dt.datetime.utcnow().timestamp())

def google_news_rss(query: str):
    # URL-encode the query to avoid spaces/quotes issues
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}+when:{POLL_NEWS_HOURS}h&hl=en-US&gl=US&ceid=US:en"

def senti(text: str): return an.polarity_scores((text or "")[:2000])["compound"]

def get_pub_ts(entry):
    for k in ("published_parsed", "updated_parsed"):
        t = getattr(entry, k, None)
        if t: return int(time.mktime(t))
    return now_ts()

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"seen_ids": {}, "last_alert": {}}

def save_state(state): STATE_PATH.write_text(json.dumps(state, indent=2))

def domain_from_url(u: str):
    m = re.search(r"https?://([^/]+)/?", u)
    if not m: return ""
    d = m.group(1).lower()
    # unwrap Google News redirect if present
    q = re.search(r"[?&]url=([^&]+)", u)
    if q:
        inner = requests.utils.unquote(q.group(1))
        dm = re.search(r"https?://([^/]+)/?", inner)
        if dm: return dm.group(1).lower()
    return d

def unwrap_google_news(u: str):
    q = re.search(r"[?&]url=([^&]+)", u)
    return requests.utils.unquote(q.group(1)) if q else u

def human_ago(ts: int):
    delta = now_ts() - ts
    if delta < 60: return f"{delta}s ago"
    minutes = delta // 60
    if minutes < 60: return f"{minutes}m ago"
    hours = minutes // 60
    return f"{hours}h {minutes%60}m ago"

CRED_WEIGHTS = {
    "finance.yahoo.com": 1.2,
    "www.reuters.com": 1.3,
    "www.bloomberg.com": 1.3,
    "www.wsj.com": 1.2,
    "seekingalpha.com": 1.1,
    "marketbeat.com": 1.0,
}
DEFAULT_WEIGHT = 1.0
def source_weight(u: str): return CRED_WEIGHTS.get(domain_from_url(u), DEFAULT_WEIGHT)

def cluster_confidence(hits, window_min):
    if not hits: return "Low"
    vol = len(hits)
    avg_abs = sum(abs(h["score"]) for h in hits)/vol
    avg_w = sum(source_weight(h["url"]) for h in hits)/vol
    recency = max(0.5, min(1.0, 180 / max(1, window_min)))
    raw = vol * avg_abs * avg_w * recency
    if raw >= 4.0: return "High"
    if raw >= 2.0: return "Medium-High"
    if raw >= 1.0: return "Medium"
    return "Low"

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

def fetch_items(ticker, name):
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
    # newest first (helps recency)
    items.sort(key=lambda x: x["pub_ts"], reverse=True)
    return items

def sha(u): return hashlib.sha1(u.encode()).hexdigest()

# ================= ALERT FORMAT =================
def format_alert(ticker, window_min, pos_hits, neg_hits):
    side_hits = pos_hits if sum(h["score"] for h in pos_hits) >= abs(sum(h["score"] for h in neg_hits)) else neg_hits
    side = "Bullish" if side_hits and side_hits[0]["score"] > 0 else "Bearish"
    side_sorted = sorted(side_hits, key=lambda x: x["score"], reverse=True) if side == "Bullish" else sorted(side_hits, key=lambda x: x["score"])
    all_hits = pos_hits + neg_hits
    score_sum = sum(h["score"] for h in all_hits)
    conf = cluster_confidence(all_hits, window_min)
    pos_count = sum(1 for h in all_hits if h["score"] > 0)
    neg_count = sum(1 for h in all_hits if h["score"] < 0)

    lines = []
    lines.append(f"{ticker} — {side} cluster detected (last {window_min}m)")
    lines.append(f"Score sum: {score_sum:+.2f} | {pos_count}/{len(all_hits)} positive | Confidence: {conf}")
    lines.append("")
    lines.append("Top headlines:")

    for i, h in enumerate(side_sorted[:5], 1):
        url = unwrap_google_news(h["url"])
        dom = domain_from_url(url)
        lines.append(f"{i}) {h['title']}")
        lines.append(f"   {h['score']:+.2f} • {human_ago(h['pub_ts'])} • {dom}")
        lines.append(f"   {url}")

    opp = neg_hits if side == "Bullish" else pos_hits
    if opp:
        opp_pick = sorted(opp, key=lambda x: -abs(x["score"]))[0]
        url = unwrap_google_news(opp_pick["url"])
        dom = domain_from_url(url)
        lines.append("")
        lines.append("Counterpoint:")
        lines.append(f"• {opp_pick['title']}")
        lines.append(f"  {opp_pick['score']:+.2f} • {human_ago(opp_pick['pub_ts'])} • {dom}")
        lines.append(f"  {url}")

    return "\n".join(lines)

def maybe_alert(ticker, pos_hits, neg_hits, state):
    last = state["last_alert"].get(ticker, 0)
    if now_ts() - last < ALERT_COOLDOWN_MIN * 60:
        return False
    pos_cluster = len(pos_hits) >= CLUSTER_COUNT
    neg_cluster = len(neg_hits) >= CLUSTER_COUNT
    if not (pos_cluster or neg_cluster):
        return False
    msg = format_alert(ticker, LOOKBACK_MIN, pos_hits, neg_hits)
    notify_telegram(msg)
    state["last_alert"][ticker] = now_ts()
    save_state(state)
    return True

# ================= MAIN =================
def run_once():
    changed = False
    state = load_state()
    cutoff_ts = now_ts() - LOOKBACK_MIN * 60

    for tkr, name in WATCHLIST.items():
        seen = state["seen_ids"].setdefault(tkr, {})
        items = fetch_items(tkr, name)

        window_items = []
        for it in items:
            h = sha(it["url"])
            if h not in seen:
                seen[h] = it["pub_ts"]
                changed = True
            if it["pub_ts"] >= cutoff_ts:
                window_items.append(it)

        pos = [x for x in window_items if x["score"] >= STRONG_POS]
        neg = [x for x in window_items if x["score"] <= STRONG_NEG]

        if len(pos) >= CLUSTER_COUNT or len(neg) >= CLUSTER_COUNT:
            if maybe_alert(tkr, pos, neg, state):
                changed = True

    if changed:
        save_state(state)

if __name__ == "__main__":
    run_once()

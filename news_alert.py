import os, json, time, datetime as dt, hashlib, re, requests, feedparser, html
from pathlib import Path
from urllib.parse import quote_plus, urlparse, parse_qs, unquote
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ================= CONFIG =================
WATCHLIST = {

    "NVDA": "NVIDIA",
    
}
LOOKBACK_MIN = 180             # window for clustering (minutes)
STRONG_POS = 0.6               # VADER positive threshold
STRONG_NEG = -0.6              # VADER negative threshold
CLUSTER_COUNT = 3              # how many strong hits to alert
ALERT_COOLDOWN_MIN = 120       # min gap between alerts per ticker
POLL_NEWS_HOURS = 6            # how far back to query in Google News
SENT_TTL_HOURS = 24            # do not re-send the same article within this TTL

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

STATE_PATH = Path("state.json")
an = SentimentIntensityAnalyzer()

# ================= UTILITIES =================
def now_ts(): return int(dt.datetime.utcnow().timestamp())

def google_news_rss(query: str):
    q = quote_plus(query)  # encode spaces/quotes/$, etc.
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
    # seen_ids: {ticker: {sha1(canonical_url): pub_ts}}
    # last_alert: {ticker: ts}
    # sent_ids: {sha1(canonical_url): last_sent_ts}
    return {"seen_ids": {}, "last_alert": {}, "sent_ids": {}}

def save_state(state): STATE_PATH.write_text(json.dumps(state, indent=2))

def extract_canonical_url(u: str) -> str:
    """
    Unwrap Google News redirect URLs to their canonical target.
    """
    try:
        parsed = urlparse(u)
        if parsed.netloc.endswith("news.google.com"):
            qs = parse_qs(parsed.query)
            if "url" in qs and qs["url"]:
                return unquote(qs["url"][0])
        return u
    except Exception:
        return u

def domain_from_url(u: str) -> str:
    try:
        parsed = urlparse(u)
        d = parsed.netloc.lower()
        if d.startswith("www."): d = d[4:]
        return d
    except Exception:
        return ""

def human_ago(ts: int):
    delta = now_ts() - ts
    if delta < 60: return f"{delta}s ago"
    minutes = delta // 60
    if minutes < 60: return f"{minutes}m ago"
    hours = minutes // 60
    return f"{hours}h {minutes%60}m ago"

def hsha(u: str) -> str:
    return hashlib.sha1(u.encode("utf-8", "ignore")).hexdigest()

# Credibility weights (tweak as you like)
CRED_WEIGHTS = {
    "finance.yahoo.com": 1.2,
    "reuters.com": 1.3,
    "bloomberg.com": 1.3,
    "wsj.com": 1.2,
    "seekingalpha.com": 1.1,
    "marketbeat.com": 1.0,
}
DEFAULT_WEIGHT = 1.0
def source_weight(u: str): return CRED_WEIGHTS.get(domain_from_url(u), DEFAULT_WEIGHT)

def weighted_sum(hits):
    return sum(h["score"] * source_weight(h["url"]) for h in hits)

def cluster_confidence(hits, window_min):
    if not hits: return "Low"
    vol = len(hits)
    avg_abs = sum(abs(h["score"]) for h in hits)/vol
    avg_w = sum(source_weight(h["url"]) for h in hits)/vol
    recency = max(0.5, min(1.0, 180 / max(1, window_min)))  # decay if you widen the window
    raw = vol * avg_abs * avg_w * recency
    if raw >= 4.0: return "High"
    if raw >= 2.0: return "Medium-High"
    if raw >= 1.0: return "Medium"
    return "Low"

# ============ Telegram (HTML, short links) ============
def t_escape(s: str) -> str:
    # Telegram HTML needs escaping
    return html.escape(s or "")

def notify_telegram_html(text_html: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Would notify (missing TELEGRAM_* env):\n", text_html)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print("Telegram error:", e)

# ================= FETCH =================
def fetch_items(ticker, name):
    q = f'"{name}" OR {ticker} OR ${ticker}'
    feed = feedparser.parse(google_news_rss(q))
    items = []
    for e in feed.entries:
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        link = getattr(e, "link", "")
        can_url = extract_canonical_url(link)
        score = senti(f"{title} {summary}")
        items.append({
            "title": title, "url": can_url, "score": score, "pub_ts": get_pub_ts(e)
        })
    # newest first
    items.sort(key=lambda x: x["pub_ts"], reverse=True)
    return items

# ================= ALERT FORMAT =================
def format_alert_html(ticker, window_min, window_hits):
    # Partition into pos/neg for stats
    pos_hits = [h for h in window_hits if h["score"] >= STRONG_POS]
    neg_hits = [h for h in window_hits if h["score"] <= STRONG_NEG]
    side_hits = pos_hits if sum(h["score"] for h in pos_hits) >= abs(sum(h["score"] for h in neg_hits)) else neg_hits
    side = "Bullish" if side_hits and side_hits[0]["score"] > 0 else "Bearish"
    side_sorted = sorted(side_hits, key=lambda x: x["score"], reverse=True) if side == "Bullish" \
                  else sorted(side_hits, key=lambda x: x["score"])

    total = len(window_hits)
    pos_count = sum(1 for h in window_hits if h["score"] > 0)
    net = sum(h["score"] for h in window_hits)
    wsum = weighted_sum(window_hits)
    sources = len({domain_from_url(h["url"]) for h in window_hits if h["url"]})
    conf = cluster_confidence(window_hits, window_min)

    lines = []
    lines.append(f"<b>{t_escape(ticker)}</b> — {t_escape(side)} cluster (last {window_min}m)")
    lines.append(f"Net: {net:+.2f} | Pos/Total: {pos_count}/{total} | Weighted: {wsum:+.2f} | Sources: {sources} | Confidence: {t_escape(conf)}")
    lines.append("")
    lines.append("<b>Top headlines:</b>")

    for i, h in enumerate(side_sorted[:5], 1):
        dom = domain_from_url(h["url"])
        ago = human_ago(h["pub_ts"])
        title = t_escape(h["title"])
        link = t_escape(h["url"])
        lines.append(f"{i}) {title} — {t_escape(dom)} ({h['score']:+.2f}, {t_escape(ago)}) <a href=\"{link}\">open</a>")

    # Counterpoint (strongest on the other side)
    opp = neg_hits if side == "Bullish" else pos_hits
    if opp:
        opp_pick = sorted(opp, key=lambda x: -abs(x["score"]))[0]
        dom = domain_from_url(opp_pick["url"])
        ago = human_ago(opp_pick["pub_ts"])
        title = t_escape(opp_pick["title"])
        link = t_escape(opp_pick["url"])
        lines.append("")
        lines.append("<i>Counterpoint:</i>")
        lines.append(f"• {title} — {t_escape(dom)} ({opp_pick['score']:+.2f}, {t_escape(ago)}) <a href=\"{link}\">open</a>")

    return "\n".join(lines)

# ================= MAIN =================
def run_once():
    state = load_state()
    cutoff_ts = now_ts() - LOOKBACK_MIN * 60
    sent_ttl_cutoff = now_ts() - SENT_TTL_HOURS * 3600
    # prune old sent_ids
    state["sent_ids"] = {k:v for k,v in state.get("sent_ids", {}).items() if v >= sent_ttl_cutoff}

    changed = False

    for tkr, name in WATCHLIST.items():
        seen = state["seen_ids"].setdefault(tkr, {})
        items = fetch_items(tkr, name)

        # Build window and dedupe by canonical URL hash
        window_hits = []
        for it in items:
            can = it["url"]
            h = hsha(can)
            if h not in seen:
                seen[h] = it["pub_ts"]
                changed = True
            if it["pub_ts"] >= cutoff_ts:
                window_hits.append(it)

        # Gate: need enough strong hits either side to consider alerting
        pos = [x for x in window_hits if x["score"] >= STRONG_POS]
        neg = [x for x in window_hits if x["score"] <= STRONG_NEG]
        if not (len(pos) >= CLUSTER_COUNT or len(neg) >= CLUSTER_COUNT):
            continue

        # Prevent re-sending if we've just pushed the same top links
        # Build a signature of top items (by score, 5 max)
        side_hits = pos if sum(h["score"] for h in pos) >= abs(sum(h["score"] for h in neg)) else neg
        side_sorted = sorted(side_hits, key=lambda x: x["score"], reverse=True) if side_hits and side_hits[0]["score"] > 0 else \
                      sorted(side_hits, key=lambda x: x["score"])
        top_urls = [h["url"] for h in side_sorted[:5]]
        sig = hsha("|".join(top_urls))

        last_sent = state["sent_ids"].get(sig, 0)
        if now_ts() - last_sent < ALERT_COOLDOWN_MIN * 60:
            # skip duplicate-ish alert within cooldown
            continue

        # Cooldown per ticker
        last_tkr = state["last_alert"].get(tkr, 0)
        if now_ts() - last_tkr < ALERT_COOLDOWN_MIN * 60:
            continue

        # Send formatted HTML alert
        msg_html = format_alert_html(tkr, LOOKBACK_MIN, window_hits)
        notify_telegram_html(msg_html)

        # Update alert timestamps/signature cache
        state["last_alert"][tkr] = now_ts()
        state["sent_ids"][sig] = now_ts()
        changed = True

    if changed:
        save_state(state)

if __name__ == "__main__":
    run_once()

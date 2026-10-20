"""Collect Pakistani-context text from Wikipedia / Wikinews via the official MediaWiki API.

Licences: Wikipedia CC BY-SA 4.0, Wikinews CC BY 2.5 (credit in the report).
Usage:
    python scripts/scrape_wikipedia.py --email you@example.com --check      # validate categories only
    python scripts/scrape_wikipedia.py --email you@example.com --per-cat 25 --max-docs 300
Outputs (appended): data/raw/documents.jsonl, data/raw/source_log.csv
"""
import argparse, csv, json, time
from datetime import datetime, timezone
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
DOCS_PATH = RAW_DIR / "documents.jsonl"
LOG_PATH = RAW_DIR / "source_log.csv"
MIN_DELAY = 1.5  # seconds between API requests

SOURCES = {
    "wikipedia": dict(api="https://en.wikipedia.org/w/api.php", site="https://en.wikipedia.org/wiki/",
                      license="CC BY-SA 4.0", tier=1),
    "wikinews": dict(api="https://en.wikinews.org/w/api.php", site="https://en.wikinews.org/wiki/",
                     license="CC BY 2.5", tier=1),
}

CATEGORIES = {
    "wikipedia": [
        "Political parties in Pakistan", "Universities and colleges in Pakistan",
        "Judiciary of Pakistan", "Supreme Court of Pakistan", "Law of Pakistan", "Cities in Pakistan",
        "Stadiums in Pakistan", "Pakistan Super League",
        "Government agencies of Pakistan", "Provinces of Pakistan",
        "Television stations in Pakistan", "Newspapers published in Pakistan",
        "Elections in Pakistan", "Prime ministers of Pakistan",
    ],
    "wikinews": ["Pakistan"],
}

_last = 0.0


def get(session, url, params):
    global _last
    wait = MIN_DELAY - (time.time() - _last)
    if wait > 0:
        time.sleep(wait)
    _last = time.time()
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def members(session, source, cat, limit):
    j = get(session, SOURCES[source]["api"], {
        "action": "query", "format": "json", "list": "categorymembers",
        "cmtitle": f"Category:{cat}", "cmlimit": limit, "cmtype": "page"})
    return [m["title"] for m in j.get("query", {}).get("categorymembers", [])]


def page_text(session, source, title):
    j = get(session, SOURCES[source]["api"], {
        "action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
        "exsectionformat": "plain", "redirects": 1, "titles": title})
    page = next(iter(j["query"]["pages"].values()))
    return page.get("title", title), page.get("extract", "")


def existing_urls():
    if not DOCS_PATH.exists():
        return set()
    with open(DOCS_PATH, encoding="utf-8") as f:
        return {json.loads(l)["url"] for l in f if l.strip()}


def log_row(row):
    new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["date", "source", "tier", "license", "url", "title", "chars", "note"])
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="contact email for the User-Agent (Wikimedia policy)")
    ap.add_argument("--check", action="store_true", help="only list which categories exist")
    ap.add_argument("--per-cat", type=int, default=25)
    ap.add_argument("--max-docs", type=int, default=300, help="cap for wikipedia; wikinews capped at 50")
    ap.add_argument("--min-chars", type=int, default=800)
    a = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = f"NEDUET-TechFest-NER-Research/1.0 (student project, non-commercial; contact: {a.email})"
    caps = {"wikipedia": a.max_docs, "wikinews": 50}

    if a.check:
        for src, cats in CATEGORIES.items():
            for c in cats:
                n = len(members(s, src, c, 50))
                print(f"{'OK   ' if n else 'EMPTY'} [{src}] {c}: {n} pages")
        return

    seen = existing_urls()
    for src, cats in CATEGORIES.items():
        saved = sum(1 for u in seen if SOURCES[src]["site"] in u)
        for cat in cats:
            for title in members(s, src, cat, a.per_cat):
                if saved >= caps[src]:
                    print(f"cap reached for {src}")
                    break
                url = SOURCES[src]["site"] + title.replace(" ", "_")
                if url in seen:
                    continue
                t, text = page_text(s, src, title)
                if len(text) < a.min_chars:
                    continue
                now = datetime.now(timezone.utc).isoformat()
                doc = dict(source=src, tier=SOURCES[src]["tier"], license=SOURCES[src]["license"],
                           url=url, title=t, scraped_at=now, text=text)
                with open(DOCS_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                log_row(dict(date=now, source=src, tier=SOURCES[src]["tier"], license=SOURCES[src]["license"],
                             url=url, title=t, chars=len(text), note=f"category={cat}"))
                seen.add(url)
                saved += 1
                print(f"saved [{src}] {t}")
    print("done")


if __name__ == "__main__":
    main()

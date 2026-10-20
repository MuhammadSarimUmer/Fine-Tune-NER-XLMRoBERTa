"""Build one clean, balanced sentence corpus from every kept source.

Run from the project root:  python scripts/build_corpus.py [--seed 42] [--scale 1.0]

Reads : datasets/*, data/raw/documents.jsonl (Wikipedia/Wikinews from scrape_wikipedia.py)
Writes: data/raw/sentences.jsonl      one sentence per line: id, sentence, source, tier, group
        data/gazetteers/*.txt         provinces, divisions, districts, tehsils, penal code sections

Balancing: each source has a sentence cap and a per-document cap, so no single source
dominates (the Supreme Court text alone is ~65 MB) and every document contributes a few
sentences (diversity > volume). `group` = the source document, so Phase 5 can split
train/validation/benchmark BY DOCUMENT and avoid leakage.
"""
import argparse, codecs, glob, io, json, random, re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"
RAW = ROOT / "data" / "raw"
GAZ = ROOT / "data" / "gazetteers"

# source -> (tier, total sentence cap, max sentences per document)
PLAN = {
    "wikipedia":      (1, 4000, 15),
    "wikinews":       (1, 800, 20),
    "headlines":      (3, 4000, 1),    # each headline/excerpt is its own "document"
    "dawn":           (3, 3500, 6),
    "tribune":        (3, 3500, 3),
    "supreme_court":  (2, 3500, 2),
    "penal_code":     (2, 250, 1),     # generic wording, few entities -> small share
}


# ---------- tolerant reading (files mix UTF-8 with stray cp1252 bytes) ----------
def _cp1252_fallback(err):
    return err.object[err.start:err.end].decode("cp1252", errors="replace"), err.end

codecs.register_error("cp1252_fallback", _cp1252_fallback)


def read_text(path):
    return Path(path).read_bytes().decode("utf-8", errors="cp1252_fallback")


def read_csv(path, **kw):
    return pd.read_csv(io.StringIO(read_text(path)), **kw)


# ---------- cleaning and sentence splitting ----------
ABBR = ["Mr", "Mrs", "Ms", "Dr", "Prof", "Rs", "Ltd", "Pvt", "St", "No", "Nos", "vs", "v", "Jan", "Feb", "Mar",
        "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec", "Col", "Gen", "Maj", "Lt", "Sr", "Jr",
        "Hon", "Art", "Sec", "Ord", "Govt", "Dept", "Mst", "Syed", "Sh", "Mian", "Ch", "Brig", "Capt", "Cdr", "Engr", "Adv"]
JUNK = re.compile(r"(Published in|Also read|Read more|Read:|Click here|Follow us|File photo|Photo:|Copyright|"
                  r"Subscribe|Advertisement|www\.|http|@\w+|\.{4,}|\bAFP\b photo)", re.I)


def normalize(t):
    t = t.replace("\u00a0", " ").replace("\ufffd", "").replace("\u200b", "")
    t = t.replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2014", " - ").replace("\u2013", "-").replace("\u2026", "...")
    return re.sub(r"[ \t]+", " ", t)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    for a in ABBR:
        text = re.sub(rf"\b{a}\.", f"{a}<DOT>", text)
    text = re.sub(r"\b([A-Z])\.", r"\1<DOT>", text)          # initials: M. Khan
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts]


def good_sentence(s, headline=False):
    words = s.split()
    if len(words) < (4 if headline else 7) or len(words) > 55:
        return False
    if not headline and not re.search(r"[.!?\"']$", s):
        return False
    if not re.match(r"[A-Z0-9\"'(]", s):
        return False
    alpha = sum(c.isalpha() for c in s)
    if alpha / len(s) < 0.7:
        return False
    if sum(c.isupper() for c in s) / max(alpha, 1) > 0.4:      # ALL-CAPS headers
        return False
    if JUNK.search(s):
        return False
    return True


PK_HINT = re.compile(r"Pakistan|Karachi|Lahore|Islamabad|Punjab|Sindh|Khyber|Balochistan|Peshawar|Quetta|Rawalpindi|"
                     r"Multan|Faisalabad|Hyderabad|PTI|PML|PPP|MQM|Imran|Sharif|Zardari|Bilawal|Rs\s?\d|PKR|Dawn|"
                     r"Supreme Court|NAB|FIA|ISPR|SBP|Kashmir|Gilgit|Afghan", re.I)
OCR_BREAK = re.compile(r"\b[b-hj-z]\b|\w \-\d|\d \-\d|\bApproved for reporting\b")   # "Ordinanc e", "1 -1-1985"


def keep_sentence(source, s, rnd):
    """Quality/relevance filters applied per source (after generic cleaning)."""
    if source == "supreme_court":
        if OCR_BREAK.search(s) or sum(c.isdigit() for c in s) / len(s) > 0.08:
            return False
    # entity-light sentences (no capitalised word after the first, no digit) teach only 'O';
    # keep a small share as negatives
    ent_like = any(re.match(r"[A-Z][a-z]", w) for w in s.split()[1:]) or bool(re.search(r"\d", s))
    if not ent_like and rnd.random() > 0.35:
        return False
    # headlines include world news; favour Pakistan-related ones
    if source == "headlines" and not PK_HINT.search(s) and rnd.random() > 0.3:
        return False
    return True


def sentences_from(text, headline=False, by_paragraph=True):
    text = normalize(text)
    paras = [p for p in text.split("\n") if p.strip()] if by_paragraph else [text]
    out = []
    for p in paras:
        if re.search(r"file photo|photo:", p, re.I) and len(p) < 160:   # photo captions
            continue
        out.extend(s for s in split_sentences(p) if good_sentence(s, headline))
    return out


# ---------- loaders: each yields (group_id, text, is_headline, by_paragraph) ----------
def load_wiki(source):
    p = RAW / "documents.jsonl"
    if not p.exists():
        return
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        if d["source"] == source:
            yield f"{source}:{d['url']}", d["text"], False, True


def load_headlines():
    df = read_csv(next(iter(glob.glob(str(DS / "Pakistan-news" / "*.csv")))))
    for i, r in df.iterrows():
        ex = str(r.get("Story Excerpt", ""))
        if ex and ex != "nan":
            yield f"headlines:{i}", ex, False, True
        # headline itself is a fragment; kept only if it reads like a sentence
        hd = str(r.get("Story Heading", ""))
        if len(hd.split()) >= 8:
            yield f"headlines:{i}h", hd, True, True


def load_dawn():
    df = read_csv(DS / "Pakistan-news2" / "Dawn_2013_2018.csv")
    for i, r in df.iterrows():
        yield f"dawn:{i}", str(r["Article"]), False, True


def load_tribune():
    df = pd.read_excel(DS / "Pakistan-news2" / "The_Tribune_2013_2018.xlsx")
    for i, r in df.iterrows():
        yield f"tribune:{i}", str(r["Paragraph"]), False, True


def load_supreme_court():
    for f in sorted(glob.glob(str(DS / "Supreme_court_Of_Pakistan_judgments" / "**" / "*.txt"), recursive=True)):
        t = read_text(f)
        if len(t) < 3000:                    # empty / near-empty files
            continue
        yield f"supreme_court:{Path(f).name}", t[2500:], False, False   # skip header block (judges, case numbers)


def load_penal():
    df = read_csv(DS / "Pakistan-penal-code" / "processed_laws.csv")
    for i, r in df.iterrows():
        t = str(r.get("Applicable when", ""))
        if t and t != "nan":
            yield f"penal_code:{i}", t, False, True


LOADERS = {
    "wikipedia": lambda: load_wiki("wikipedia"),
    "wikinews": lambda: load_wiki("wikinews"),
    "headlines": load_headlines,
    "dawn": load_dawn,
    "tribune": load_tribune,
    "supreme_court": load_supreme_court,
    "penal_code": load_penal,
}


# ---------- gazetteers (for pre-annotation later) ----------
def build_gazetteers():
    GAZ.mkdir(parents=True, exist_ok=True)
    cen = read_csv(next(iter(glob.glob(str(DS / "Pakistan_census" / "*.csv")))))

    def clean(series, suffix):
        vals = set()
        for v in series.dropna().astype(str):
            v = re.sub(rf"\b{suffix}\b", "", v, flags=re.I).strip(" -")
            if v:
                vals.add(v.title())
        return sorted(vals)

    fixes = {"Kpk": "KPK", "Kpk/Fata": "KPK", "Fata": "FATA", "Ict": "Islamabad Capital Territory",
             "Islamabad": "Islamabad"}

    out = {
        "provinces": sorted({fixes.get(p, p) for p in clean(cen["PROVINCE"], r"PROVINCE")}),
        "divisions": clean(cen["DIVISION"], r"DIVISION"),
        "districts": clean(cen["DISTRICT"], r"DISTRICT"),
        "tehsils": clean(cen["SUB DIVISION"], r"(TEHSIL|SUB-?DIVISION|TALUKA|SUB TEHSIL)"),
    }
    pen = read_csv(DS / "Pakistan-penal-code" / "processed_laws.csv")
    out["penal_code_sections"] = sorted(set(pen["Name"].dropna().astype(str).str.strip()))
    for k, v in out.items():
        (GAZ / f"{k}.txt").write_text("\n".join(v), encoding="utf-8")
        print(f"gazetteer {k}: {len(v)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every source cap (e.g. 0.5 for half)")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    RAW.mkdir(parents=True, exist_ok=True)

    build_gazetteers()

    seen, seen_prefix, final, report = set(), set(), [], []
    for source, (tier, cap, per_doc) in PLAN.items():
        pool, n_docs = [], 0
        for group, text, headline, by_para in LOADERS[source]():
            sents = []
            for s in sentences_from(text, headline, by_para):
                key = re.sub(r"\W+", " ", s.lower()).strip()
                if key in seen or " ".join(key.split()[:12]) in seen_prefix:     # exact + near duplicates
                    continue
                if not keep_sentence(source, s, rnd):
                    continue
                sents.append((s, key))
            if not sents:
                continue
            n_docs += 1
            rnd.shuffle(sents)
            for s, key in sents[:per_doc]:
                pool.append((s, key, group))
        cap = int(cap * a.scale)
        rnd.shuffle(pool)
        chosen = pool[:cap]
        for s, key, group in chosen:
            seen.add(key)
            seen_prefix.add(" ".join(key.split()[:12]))
            final.append(dict(sentence=s, source=source, tier=tier, group=group))
        report.append((source, n_docs, len(pool), len(chosen)))

    rnd.shuffle(final)
    for i, r in enumerate(final):
        r["id"] = i
    with open(RAW / "sentences.jsonl", "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'source':14} {'docs':>6} {'available':>10} {'kept':>6}")
    for s, d, p, k in report:
        print(f"{s:14} {d:6} {p:10} {k:6}")
    tot = len(final)
    print(f"\ntotal sentences: {tot}")
    for t, c in sorted(Counter(r["tier"] for r in final).items()):
        print(f"tier {t}: {c} ({c / tot:.0%})")
    lens = [len(r["sentence"].split()) for r in final]
    print(f"avg words/sentence: {sum(lens) / len(lens):.1f}")


if __name__ == "__main__":
    main()

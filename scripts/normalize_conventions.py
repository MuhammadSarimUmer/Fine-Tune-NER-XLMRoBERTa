"""Apply the written annotation conventions (GUIDELINES.md) uniformly to a set of spans.

normalize(text, spans) is reused for the human-corrected data AND later for the silver data, so
every training sentence follows the same rules. Rules:
  1. Leading "The" is not part of a MEDIA_OUTLET span ("The Express Tribune" -> "Express Tribune").
  2. Durations are DATE ("two weeks", "23 months", "a decade", "22-year-old").
  3. Capitalised court names are COURT_LEGAL_BODY even when bare ("the High Court", "Supreme Court").
     Lowercase generic mentions ("the court", "apex court") stay unlabelled.
  4. A year that is part of a case/petition number is not a DATE ("No. 75-Q/2021").

Usage: python scripts/normalize_conventions.py   (reads data/annotated/corrected.jsonl,
       writes data/annotated/corrected_final.jsonl and prints every change)
"""
import json, re
from collections import Counter
from pathlib import Path

from tokenization import bio_to_spans, spans_to_bio

ROOT = Path(__file__).resolve().parent.parent
NUM = r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|several|few)"
UNIT = r"(?:second|minute|hour|day|week|month|year|decade|century)s?"
DURATION = re.compile(
    rf"(?<![\w-])(?:(?:next|last|past|previous|first|following)\s+)?{NUM}(?:[- ]){UNIT}(?:[- ](?:old|long))?(?![\w])(?!\s*\d+\))", re.I)
COURT = re.compile(r"(?<!\w)(?:(?:Lahore|Sindh|Peshawar|Balochistan|Islamabad)\s+)?High Courts?(?!\w)|(?<!\w)Supreme Court(?!\w)|(?<!\w)Federal Shariat Court(?!\w)")
CASE_NO = re.compile(r"(?:No\.?|Nos\.?)\s*[\w.\-]*\d[\w.\-]*/\s*$|\b\d+\s*-\s*[A-Z]/\s*$")
# case-type word, then a number, then "/" or "of", right before the year:  C.A.538/2022  Civil Appeal No.756 OF 2021  C.P. No.260-P of 2019
CASE_NO2 = re.compile(r"\b(?:C\.?\s?A|C\.?\s?P|C\.?\s?M\.?\s?A|Crl\.?\s?\w{0,4}|W\.?\s?P|Const\.?\s?P|S\.?\s?M\.?\s?C|Civil Appeal|Criminal Appeal|"
                      r"Appeal|Petition|Suit|Reference|Case)\b[^/]{0,30}?\d[\w\-]*\s*(?:/|\bof\b)\s*$", re.I)
# law-report citations: "2010 SCMR 1962", "PLD 2012 SC 25"
REPORTER = r"(?:SCMR|PLD|CLC|YLR|PCrLJ|MLD|CLD|PTD|GBLR|SCR|PLC|PLJ|CLD)"
CITE_BEFORE = re.compile(rf"\b{REPORTER}\s+(?:SC\s+|Lah\s+|Kar\s+|Pesh\s+|Quetta\s+|Isl\s+)?$")
CITE_AFTER = re.compile(rf"^\s*{REPORTER}\b")


def year_in_case_or_citation(text, s, e):
    before = text[max(0, s - 40):s]
    return bool(CASE_NO.search(before[-25:]) or CASE_NO2.search(before) or CITE_BEFORE.search(before)
                or CITE_AFTER.match(text[e:e + 12]))


def overlaps(spans, s, e):
    return any(not (b <= s or a >= e) for a, b, _ in spans)


def normalize(text, spans, log=None):
    spans = sorted(spans)
    note = (lambda *a: log.append(a)) if log is not None else (lambda *a: None)

    out = []
    for s, e, lab in spans:                                                    # rule 1 and 4
        if lab == "MEDIA_OUTLET" and text[s:e].lower().startswith("the "):
            note("strip-The", text[s:e]); s += 4
        if lab == "DATE" and re.fullmatch(r"(?:19|20)\d\d", text[s:e]) and year_in_case_or_citation(text, s, e):
            note("drop-year-in-case-no", text[s:e]); continue
        out.append((s, e, lab))
    spans = out

    for rx, lab, tag in ((COURT, "COURT_LEGAL_BODY", "court"), (DURATION, "DATE", "duration")):   # rules 2 and 3
        for m in rx.finditer(text):
            s, e = m.start(), m.end()
            if tag == "court" and text[s:e].split()[0][0].islower():
                continue
            if not overlaps(spans, s, e):
                spans.append((s, e, lab)); note("add-" + tag, text[s:e])
    return sorted(spans)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="corrected.jsonl")
    ap.add_argument("--dst", default="corrected_final.jsonl")
    a = ap.parse_args()
    src = ROOT / "data" / "annotated" / a.src
    dst = ROOT / "data" / "annotated" / a.dst
    log, n_changed, rows = [], 0, []
    for line in open(src, encoding="utf-8"):
        r = json.loads(line)
        before = bio_to_spans(r["text"], r["ner_tags"])
        local = []
        after = normalize(r["text"], before, local)
        if after != before:
            n_changed += 1
            log.extend((r["split"], *x) for x in local)
        r["tokens"], r["ner_tags"] = spans_to_bio(r["text"], after)
        rows.append(r)
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} sentences, {n_changed} changed -> {dst.name}")
    print(Counter((x[0], x[1]) for x in log))
    for kind in ("add-duration", "add-court", "strip-The", "drop-year-in-case-no"):
        items = [x[2] for x in log if x[1] == kind]
        print(f"\n{kind} ({len(items)}):", "; ".join(f"{w} x{n}" for w, n in Counter(items).most_common(40)))


if __name__ == "__main__":
    main()

"""Sanity-check a Label Studio tasks file (predictions[0]) and dump the corrected sentences for reading.

Automatic checks: span edges inside a word, leading/trailing punctuation or spaces, leading 'The',
unbalanced brackets, overlapping spans, entity text that is only a digit, label/text mismatches
(e.g. TIME without a time word, DATE with only a bare year inside a longer name).
Usage: python scripts/check_tasks.py data/annotated/labelstudio_tasks_extra_assistant.json
"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1])
tasks = json.load(open(path, encoding="utf-8"))
issues, lines = [], []
for i, t in enumerate(tasks):
    text = t["data"]["text"]
    spans = sorted((r["value"]["start"], r["value"]["end"], r["value"]["labels"][0]) for r in t["predictions"][0]["result"])
    prev_end = -1
    for s, e, lab in spans:
        w = text[s:e]
        if s < prev_end: issues.append((i, "overlap", w, lab))
        prev_end = e
        if (s > 0 and re.match(r"\w", text[s - 1]) and re.match(r"\w", w[0])) or (e < len(text) and re.match(r"\w", text[e]) and re.search(r"\w$", w)):
            issues.append((i, "edge inside word", w, lab))
        if w != w.strip() or re.match(r"[^\w(\"'$]", w) or re.search(r"[^\w).\"'%]$", w):
            issues.append((i, "edge punctuation", w, lab))
        if w.lower().startswith("the "): issues.append((i, "leading The", w, lab))
        if w.count("(") != w.count(")"): issues.append((i, "unbalanced bracket", w, lab))
        if re.fullmatch(r"\d{1,3}", w): issues.append((i, "bare number", w, lab))
        if lab == "TIME" and not re.search(r"\d|morning|evening|night|noon|midnight|afternoon|dawn|dusk", w, re.I):
            issues.append((i, "TIME without time word", w, lab))
        if lab == "DATE" and re.fullmatch(r"(?:19|20)\d\d", w) and re.search(r"[A-Z]\w*\s*$", text[max(0, s - 25):s]):
            issues.append((i, "year after capitalised word (maybe part of a name)", w, lab))
        if lab == "PERSON" and re.match(r"(Mr|Mrs|Ms|Dr|Prof|Justice|Gen|Col|Maj|Mst|Sheikh|Maulana)\b\.?\s", w):
            issues.append((i, "title inside PERSON", w, lab))
    s = text
    for st, en, lab in reversed(spans):
        s = s[:st] + f"[{text[st:en]}|{lab}]" + s[en:]
    lines.append(f"{i} | {t['data'].get('split', '?')[:6]} | {s}")
out = ROOT / "data" / "annotated_work" / (path.stem + "_checked.txt")
out.write_text("\n".join(lines), encoding="utf-8")
print(f"{len(tasks)} tasks, {len(issues)} automatic flags -> {out.name}")
for x in issues:
    print(x)

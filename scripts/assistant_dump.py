"""Dump Label Studio tasks as readable text for reviewing/correcting the draft by hand.
Usage: python scripts/assistant_dump.py   -> data/annotated_work/review_dump.txt
Format per line:  idx | split | sentence with draft spans inline as [text|LABEL]
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tasks = json.load(open(ROOT / "data" / "annotated" / "labelstudio_tasks.json", encoding="utf-8"))
out = ROOT / "data" / "annotated_work"
out.mkdir(exist_ok=True)
lines = []
for i, t in enumerate(tasks):
    text = t["data"]["text"]
    spans = sorted((r["value"]["start"], r["value"]["end"], r["value"]["labels"][0]) for r in t["predictions"][0]["result"])
    s = text
    for st, en, lab in reversed(spans):
        s = s[:st] + f"[{text[st:en]}|{lab}]" + s[en:]
    lines.append(f"{i} | {t['data']['split'][0]} | {s}")
(out / "review_dump.txt").write_text("\n".join(lines), encoding="utf-8")
print(len(lines), "lines ->", out / "review_dump.txt")

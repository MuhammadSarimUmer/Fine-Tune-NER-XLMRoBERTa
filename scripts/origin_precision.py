"""Precision of the automatic draft per (label, origin layer), measured on the REVIEW split only.
Tells us which layer (regex / gazetteer / spacy) to trust for which label when using silver data."""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tokenization import bio_to_spans

human = {}
for line in open(ROOT / "data/annotated/corrected_final.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["split"] == "review":
        human[r["id"]] = set(bio_to_spans(r["text"], r["ner_tags"]))

tp, n = Counter(), Counter()
miss = Counter()
for line in open(ROOT / "data/annotated/preannotated.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"] not in human:
        continue
    h = human[r["id"]]
    for s, e, lab, origin in r["spans"]:
        key = (lab, origin)
        n[key] += 1
        if (s, e, lab) in h:
            tp[key] += 1
print(f"{'label':24}{'origin':12}{'n':>5}{'precision':>10}")
for key in sorted(n, key=lambda k: (k[0], -n[k])):
    print(f"{key[0]:24}{key[1]:12}{n[key]:5}{tp[key] / n[key]:10.2f}")

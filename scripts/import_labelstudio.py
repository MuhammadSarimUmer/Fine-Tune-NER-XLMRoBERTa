"""Convert a Label Studio JSON export (after manual correction) back to our token/BIO format.

Usage:  python scripts/import_labelstudio.py path/to/export.json [--out data/annotated/corrected.jsonl]
Also prints how accurate the automatic pre-annotation was (auto vs human), per label.
Use this table in the report: it measures annotation quality and shows where auto-labels fail.
"""
import argparse, json
from collections import Counter
from pathlib import Path

from labels import ENTITY_TYPES
from tokenization import spans_to_bio

ROOT = Path(__file__).resolve().parent.parent


def result_spans(result):
    spans = []
    for r in result:
        v = r.get("value", {})
        if r.get("type") == "labels" and v.get("labels"):
            spans.append((v["start"], v["end"], v["labels"][0]))
    return sorted(spans)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--tasks", default=str(ROOT / "data" / "annotated" / "labelstudio_tasks_assistant.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "annotated" / "corrected.jsonl"))
    a = ap.parse_args()

    tasks = json.load(open(a.export, encoding="utf-8"))
    ref = {}                                       # sentence id -> predictions from the tasks file we imported
    if Path(a.tasks).exists():
        for rt in json.load(open(a.tasks, encoding="utf-8")):
            ref[rt["data"].get("id")] = rt.get("predictions", [])
    rows = []
    stats = {}                                     # model_version -> (tp, fp, fn) counters
    for t in tasks:
        anns = [x for x in t.get("annotations", []) if not x.get("was_cancelled")]
        if not anns:
            continue                                   # not reviewed yet
        text = t["data"]["text"]
        human = result_spans(anns[-1]["result"])
        h = set(human)
        preds = t.get("predictions", [])
        if not preds or not isinstance(preds[0], dict):          # export holds only prediction ids
            preds = ref.get(t["data"].get("id"), [])
        for pr in preds:
            tp, fp, fn = stats.setdefault(pr.get("model_version", "draft"), (Counter(), Counter(), Counter()))
            au = set(result_spans(pr["result"]))
            for s in h & au: tp[s[2]] += 1
            for s in au - h: fp[s[2]] += 1
            for s in h - au: fn[s[2]] += 1
        tokens, tags = spans_to_bio(text, human)
        rows.append(dict(id=t["data"].get("id", ""), text=text, tokens=tokens, ner_tags=tags,
                         source=t["data"].get("source", ""), group=t["data"].get("group", ""),
                         split=t["data"].get("split", "review"), reviewed=True))
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} reviewed sentences -> {a.out}")
    for ver, (tp, fp, fn) in stats.items():
        print(f"\n[{ver}] quality vs human (exact span+label match)\n{'label':24}{'prec':>7}{'rec':>7}{'tp':>6}{'fp':>6}{'fn':>6}")
        for lab in ENTITY_TYPES:
            p = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else float('nan')
            r = tp[lab] / (tp[lab] + fn[lab]) if tp[lab] + fn[lab] else float('nan')
            print(f"{lab:24}{p:7.2f}{r:7.2f}{tp[lab]:6}{fp[lab]:6}{fn[lab]:6}")


if __name__ == "__main__":
    main()

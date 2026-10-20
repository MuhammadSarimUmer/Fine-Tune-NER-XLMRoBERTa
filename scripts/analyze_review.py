"""Per-split quality of the automatic draft vs the human labels, plus the concrete errors.

Usage: python scripts/analyze_review.py data/annotated/export-400.json
Bulk-fix decisions must come from the REVIEW split only; the TEST split stays clean for the final report.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import ENTITY_TYPES


def spans(result):
    return {(r["value"]["start"], r["value"]["end"], r["value"]["labels"][0])
            for r in result if r.get("type") == "labels" and r["value"].get("labels")}


def main():
    export = json.load(open(sys.argv[1], encoding="utf-8"))
    ref = {t["data"]["id"]: t["predictions"] for t in json.load(open(ROOT / "data/annotated/labelstudio_tasks_assistant.json", encoding="utf-8"))}
    cnt = {s: (Counter(), Counter(), Counter()) for s in ("test", "review")}
    errors = defaultdict(list)                       # (label, kind) -> examples, review only
    for t in export:
        anns = [a for a in t["annotations"] if not a.get("was_cancelled")]
        if not anns:
            continue
        split = t["data"].get("split", "review")
        text = t["data"]["text"]
        human = spans(anns[-1]["result"])
        draft = spans([p for p in ref[t["data"]["id"]] if p["model_version"] == "preannotation-v1"][0]["result"])
        tp, fp, fn = cnt[split]
        for s in human & draft: tp[s[2]] += 1
        for s in draft - human:
            fp[s[2]] += 1
            if split == "review": errors[(s[2], "draft-wrong")].append(text[s[0]:s[1]])
        for s in human - draft:
            fn[s[2]] += 1
            if split == "review": errors[(s[2], "draft-missed")].append(text[s[0]:s[1]])
    for split, (tp, fp, fn) in cnt.items():
        print(f"\n== draft vs human, {split.upper()} split ==\n{'label':24}{'gold':>6}{'prec':>7}{'rec':>7}")
        for lab in ENTITY_TYPES:
            g = tp[lab] + fn[lab]
            p = tp[lab] / (tp[lab] + fp[lab]) if tp[lab] + fp[lab] else float('nan')
            r = tp[lab] / g if g else float('nan')
            print(f"{lab:24}{g:6}{p:7.2f}{r:7.2f}")
    print("\n== most common REVIEW-split draft errors ==")
    for (lab, kind), items in sorted(errors.items(), key=lambda kv: -len(kv[1])):
        if len(items) >= 4:
            top = ", ".join(f"{w} x{n}" for w, n in Counter(items).most_common(8))
            print(f"{lab:24}{kind:14}{len(items):4}  {top}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strict entity-level scorer for the TechFest NER Challenge.

Usage:  python score.py --gold sample_answer_key.jsonl --pred my_submission.jsonl

Gold / prediction files are JSONL, one object per sentence:
  {"id": "TFN-DEV-0001", "entities": [{"start": 0, "end": 5, "label": "CITY"}, ...]}
Offsets are character offsets into the sentence text (end exclusive).
An entity counts as correct only if start, end and label all match exactly.
Sentences missing from the prediction file are treated as "no entities".
Labels must be exactly one of the 20 official labels. Any other label counts as a wrong
prediction (false positive) and is reported in a warning.
Ranking metric: micro-F1 (macro-F1 is the tiebreaker).
"""
import argparse, collections, json

LABELS = ["PERSON", "CITY", "PROVINCE_REGION", "COUNTRY", "LANDMARK_FACILITY",
          "GOVERNMENT_BODY", "POLITICAL_PARTY", "ORGANIZATION", "EDUCATIONAL_INSTITUTION",
          "COURT_LEGAL_BODY", "LAW_ACT", "EVENT", "MEDIA_OUTLET", "SPORTS_TEAM", "DATE",
          "TIME", "MONEY", "PERCENT", "PHONE_NUMBER", "ID_NUMBER"]


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out[d["id"]] = {(e["start"], e["end"], e["label"]) for e in d.get("entities", [])}
    return out


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--json", help="optional path to save results as JSON")
    a = ap.parse_args()
    gold, pred = load(a.gold), load(a.pred)
    extra = set(pred) - set(gold)
    if extra:
        print(f"Warning: {len(extra)} predicted ids are not in the gold file (ignored).")

    c = {l: [0, 0, 0] for l in LABELS}  # tp, fp, fn
    bad = collections.Counter()
    for sid, g in gold.items():
        for e in pred.get(sid, set()):
            if e[2] not in c:
                bad[e[2]] += 1
        p = {e for e in pred.get(sid, set()) if e[2] in c}
        for e in g & p: c[e[2]][0] += 1
        for e in p - g: c[e[2]][1] += 1
        for e in g - p: c[e[2]][2] += 1

    print(f"{'ENTITY':26s}{'P':>8s}{'R':>8s}{'F1':>8s}{'support':>9s}")
    res, f1s = {}, []
    for l in LABELS:
        tp, fp, fn = c[l]
        p, r, f = prf(tp, fp, fn)
        res[l] = {"precision": p, "recall": r, "f1": f, "support": tp + fn}
        if tp + fn:
            f1s.append(f)
        print(f"{l:26s}{p:8.3f}{r:8.3f}{f:8.3f}{tp + fn:9d}")
    TP, FP, FN = (sum(c[l][i] for l in LABELS) for i in range(3))
    FP += sum(bad.values())  # unofficial labels are wrong predictions
    p, r, f = prf(TP, FP, FN)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    print("-" * 59)
    print(f"{'MICRO':26s}{p:8.3f}{r:8.3f}{f:8.3f}{TP + FN:9d}")
    print(f"{'MACRO-F1':26s}{'':8s}{'':8s}{macro:8.3f}")
    if bad:
        print(f"WARNING: {sum(bad.values())} predictions used labels outside the official 20 "
              f"({', '.join(sorted(map(repr, bad)))}). They were counted as wrong.")
    if a.json:
        res["micro"] = {"precision": p, "recall": r, "f1": f}
        res["macro_f1"] = macro
        json.dump(res, open(a.json, "w"), indent=2)


if __name__ == "__main__":
    main()

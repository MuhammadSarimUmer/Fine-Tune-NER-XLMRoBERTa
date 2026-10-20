"""Compare training runs on the two HUMAN test sets, pooled (random test + extra_test = 764 gold entities).

Usage: python scripts/compare_runs.py trained_reports/v1_xlmr_base trained_reports/v2_xlmr_base trained_reports/v3_xlmr_base
For each run it rebuilds true-positive / predicted counts from the saved metrics (tp = recall x support, predicted = tp / precision),
pools the two test sets, and prints overall micro P/R/F1, macro-F1 over labels, and per-label pooled F1.
Pooling is used because several labels have only 2-15 test entities per set, so one set alone is too noisy.
"""
import json, sys
from pathlib import Path

SETS = ("test", "extra_test")


def pooled(run):
    tp, pred, sup = {}, {}, {}
    for s in SETS:
        m = json.loads((Path(run) / f"metrics_{s}.json").read_text(encoding="utf-8"))
        for lab, v in m.items():
            if lab.endswith("avg") or v["support"] == 0: continue
            t = round(v["recall"] * v["support"])
            p = round(t / v["precision"]) if v["precision"] > 0 else None
            if p is None:                       # precision 0 -> predicted count unknown unless tp=0; fall back to 'predicted' field
                p = v.get("predicted", 0)
            tp[lab] = tp.get(lab, 0) + t; pred[lab] = pred.get(lab, 0) + p; sup[lab] = sup.get(lab, 0) + v["support"]
    return tp, pred, sup


def prf(t, p, s):
    P = t / p if p else 0.0; R = t / s if s else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


runs = {Path(r).name: pooled(r) for r in sys.argv[1:]}
print("POOLED over test (random, 200) + extra_test (100)")
print(f"{'run':18}{'micro P':>9}{'micro R':>9}{'micro F1':>10}{'macro F1':>10}  entities")
for name, (tp, pred, sup) in runs.items():
    P, R, F = prf(sum(tp.values()), sum(pred.values()), sum(sup.values()))
    macro = sum(prf(tp[l], pred[l], sup[l])[2] for l in sup) / len(sup)
    print(f"{name:18}{P:9.3f}{R:9.3f}{F:10.3f}{macro:10.3f}  {sum(sup.values())}")
labs = sorted(set().union(*[set(r[2]) for r in runs.values()]))
print("\nper-label pooled F1 (support = pooled gold entities)")
print(f"{'label':24}{'support':>8}" + "".join(f"{n[:12]:>14}" for n in runs))
for l in labs:
    sup = max(r[2].get(l, 0) for r in runs.values())
    print(f"{l:24}{sup:8}" + "".join(f"{prf(r[0].get(l, 0), r[1].get(l, 0), r[2].get(l, 0))[2]:14.3f}" for r in runs.values()))

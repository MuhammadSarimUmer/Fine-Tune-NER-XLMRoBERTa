"""How many labels score above 90 / 80 / 70 / 60 F1? Uses the pooled human test sets (test + extra_test) per run, and, when a
pattern_test report exists, adds PHONE_NUMBER and ID_NUMBER (synthetic test) to reach all 20 labels.
Usage: python scripts/label_bands.py trained_reports/v1_xlmr_base trained_reports/v2_xlmr_base trained_reports/v3_xlmr_base
"""
import json, sys
from pathlib import Path

def prf(t, p, s):
    P = t / p if p else 0.0; R = t / s if s else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0

def pooled(run):
    tp, pred, sup = {}, {}, {}
    for s in ("test", "extra_test"):
        for lab, v in json.loads((Path(run) / f"metrics_{s}.json").read_text(encoding="utf-8")).items():
            if lab.endswith("avg") or v["support"] == 0: continue
            t = round(v["recall"] * v["support"]); p = round(t / v["precision"]) if v["precision"] > 0 else v.get("predicted", 0)
            tp[lab] = tp.get(lab, 0) + t; pred[lab] = pred.get(lab, 0) + p; sup[lab] = sup.get(lab, 0) + v["support"]
    f1 = {l: prf(tp[l], pred[l], sup[l]) for l in sup}
    pat = Path(run) / "metrics_pattern_test.json"
    if pat.exists():
        m = json.loads(pat.read_text(encoding="utf-8"))
        for l in ("PHONE_NUMBER", "ID_NUMBER"):
            if l in m and m[l]["support"]: f1[l] = m[l]["f1-score"]; sup[l] = m[l]["support"]
    return f1, sup

BANDS = [(0.90, ">= 0.90"), (0.80, ">= 0.80"), (0.70, ">= 0.70"), (0.60, ">= 0.60"), (0.50, ">= 0.50")]
for run in sys.argv[1:]:
    f1, sup = pooled(run)
    print(f"\n=== {Path(run).name}: {len(f1)} labels scored ===")
    for th, name in BANDS:
        labs = [l for l, v in f1.items() if v >= th]
        print(f"  F1 {name}: {len(labs):2} labels")
    print("  F1 below 0.60:", ", ".join(f"{l} {v:.2f}" for l, v in sorted(f1.items(), key=lambda x: x[1]) if v < 0.60) or "none")
    print("  sorted:", ", ".join(f"{l} {v:.2f} (n={sup[l]})" for l, v in sorted(f1.items(), key=lambda x: -x[1])))

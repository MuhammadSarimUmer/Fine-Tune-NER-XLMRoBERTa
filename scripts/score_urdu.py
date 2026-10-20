"""Score the Urdu zero-shot check offline from the downloaded predictions.
Usage: python scripts/score_urdu.py trained_reports/urdu-test
Reads the expected entities from notebooks/04_urdu_check_v1.ipynb (cell with URDU = [...]).
"""
import json, sys, re
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "trained_reports/urdu-test")
nb = json.load(open("notebooks/04_urdu_check_v1.ipynb", encoding="utf-8"))
src = next("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and "URDU = [" in "".join(c["source"]))
ns = {}
exec(src.split("with open(")[0], ns)
URDU = ns["URDU"]
preds = [json.loads(l) for l in open(d / "urdu_preds.jsonl", encoding="utf-8")]
tp = fp = fn = 0
by = {}
lines = []
for (sent, exp), p in zip(URDU, preds):
    got = [(e["text"], e["label"]) for e in p["entities"]]
    gd = {}
    for t, l in got: gd.setdefault(t, []).append(l)
    lines.append(f"\n{sent}")
    for t, l in exp:
        a = by.setdefault(l, [0, 0]); a[1] += 1
        if (t, l) in got: lines.append(f"  OK    {l:24}{t}"); tp += 1; a[0] += 1
        elif t in gd: lines.append(f"  LABEL {l:24}{t} (model: {gd[t]})"); fn += 1
        else: lines.append(f"  MISS  {l:24}{t}"); fn += 1
    for t, l in got:
        if (t, l) not in exp: lines.append(f"  EXTRA {l:24}{t}"); fp += 1
P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
F = 2 * P * R / (P + R) if P + R else 0
lines.append(f"\nTP {tp} FP {fp} FN {fn}  P {P:.3f} R {R:.3f} F1 {F:.3f}")
for l, (a, b) in sorted(by.items()): lines.append(f"  {l:24}{a}/{b}")
(d / "urdu_score.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

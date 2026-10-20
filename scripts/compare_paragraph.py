"""Side-by-side view of the three models' predictions on the big test paragraph.
Usage: python scripts/compare_paragraph.py trained_reports/evals
"""
import json, sys
d = sys.argv[1] if len(sys.argv) > 1 else "trained_reports/evals"
M = {}
for v in ("v1", "v2", "v3"):
    for line in open(f"{d}/predictions_{v}.jsonl", encoding="utf-8"):
        for e in json.loads(line)["entities"]:
            M.setdefault((e["start"], e["end"], e["text"]), {})[v] = e["label"]
print(f"{'TEXT':42}{'v1':16}{'v2':16}{'v3':16}")
same_n = 0
for (s, e, t), lab in sorted(M.items()):
    same = len(lab) == 3 and len(set(lab.values())) == 1
    same_n += same
    print(f"{t[:41]:42}" + "".join(f"{lab.get(v, '-')[:15]:16}" for v in ("v1", "v2", "v3")) + ("" if same else "<<"))
print(f"\n{same_n} of {len(M)} identical in all three")
for v in ("v1", "v2", "v3"):
    print(v, "found", sum(1 for x in M.values() if v in x))

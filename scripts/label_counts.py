"""Entity counts per label in each data source we can train on (B- tags = entity count)."""
import json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from labels import ENTITY_TYPES


def cnt(path, flt=lambda r: True):
    c, n = Counter(), 0
    for line in open(ROOT / path, encoding="utf-8"):
        r = json.loads(line)
        if flt(r):
            n += 1
            for t in r["ner_tags"]:
                if t.startswith("B-"):
                    c[t[2:]] += 1
    return c, n


syn, ns = cnt("data/synthetic/synthetic.jsonl")
sil, nt = cnt("data/annotated/silver.jsonl", lambda r: r["trusted"])
unt, nu = cnt("data/annotated/silver.jsonl", lambda r: not r["trusted"])
hum, nh = cnt("data/annotated/corrected_final.jsonl", lambda r: r["split"] == "review")
tst, nte = cnt("data/annotated/corrected_final.jsonl", lambda r: r["split"] == "test")
print(f"sentences: synthetic {ns}, trusted silver {nt}, untrusted silver {nu}, human review {nh}, test {nte}")
print(f"{'label':24}{'synth':>7}{'trusted':>8}{'untrust':>8}{'human':>6}{'test':>6}{'real(t+h)':>10}")
for lab in ENTITY_TYPES:
    print(f"{lab:24}{syn[lab]:7}{sil[lab]:8}{unt[lab]:8}{hum[lab]:6}{tst[lab]:6}{sil[lab] + hum[lab]:10}")

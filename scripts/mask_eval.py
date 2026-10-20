"""Does the round-2 mask rule hit missed entities and avoid true O words? Evidence from the REVIEW split only (human labels).
For each review sentence we take the silver-style draft (after bulk fixes), find tokens the draft labelled O, and compare with the human label."""
import json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bulk_fix_silver import fix
from tokenization import spans_to_bio
from mask_unlabelled import unlabelled_name_mask
ANN = Path(__file__).resolve().parent.parent / "data" / "annotated"

human = {}
for l in open(ANN / "corrected_final.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["split"] == "review": human[r["id"]] = r
pre = {}
for l in open(ANN / "preannotated.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if r["id"] in human: pre[r["id"]] = r

missed_total = missed_hit = trueO_total = trueO_masked = 0
by_label = Counter(); hit_label = Counter(); ex_bad, ex_good = [], []
for i, h in human.items():
    spans, _ = fix(pre[i]["text"], pre[i]["spans"])
    toks, dtags = spans_to_bio(pre[i]["text"], list(spans))
    assert toks == h["tokens"]
    m = unlabelled_name_mask(toks, dtags)
    for k, (dt, ht) in enumerate(zip(dtags, h["ner_tags"])):
        if dt != "O": continue
        if ht != "O":
            missed_total += 1; by_label[ht[2:]] += 1
            if k in m: missed_hit += 1; hit_label[ht[2:]] += 1
        else:
            trueO_total += 1
            if k in m: trueO_masked += 1; ex_bad.append(toks[k])
print(f"review sentences: {len(human)}")
print(f"draft-O tokens that the HUMAN labelled as entity (missed by draft): {missed_total}; covered by mask: {missed_hit} ({missed_hit/max(missed_total,1):.0%})")
print(f"draft-O tokens that are truly O: {trueO_total}; masked anyway (cost): {trueO_masked} ({trueO_masked/max(trueO_total,1):.1%})")
print("missed-entity tokens by label (covered/total):", {k: f"{hit_label[k]}/{v}" for k, v in by_label.most_common()})
print("most common truly-O tokens that get masked:", Counter(ex_bad).most_common(25))

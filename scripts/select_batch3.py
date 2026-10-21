"""Pick 400 NEW sentences for hand annotation (batch 3), aimed at the weakest entity types.

Purpose: more HUMAN-verified TRAINING data for ORGANIZATION, GOVERNMENT_BODY, LAW_ACT, LANDMARK_FACILITY,
EDUCATIONAL_INSTITUTION and EVENT (our weakest types). These sentences are for training, not testing.

Rules that keep the existing test sets valid:
  * never from the original test documents (test_groups.json) or the extra_test documents (extra_test_groups.json)
  * never a sentence that is already human-annotated (400) or in the extra batch (300)
  * at most one sentence per document
  * the order of the tasks is round-robin over the weak labels, so ANY prefix of the list is balanced:
    the annotator can stop early and use only what is submitted.
Draft labels = the fixed draft (bulk_fix_silver.fix); the assistant then corrects them (batch3_corr_*.json) and the
human reviews every sentence in Label Studio.

Run:  python scripts/select_batch3.py
Out:  data/annotated/labelstudio_tasks_batch3.json, data/annotated/batch3_ids.json, data/annotated_work/batch3_dump.txt
"""
import json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bulk_fix_silver import fix

ROOT = Path(__file__).resolve().parent.parent
ANN, WORK = ROOT / "data" / "annotated", ROOT / "data" / "annotated_work"
WEAK = ["ORGANIZATION", "GOVERNMENT_BODY", "LAW_ACT", "LANDMARK_FACILITY", "EDUCATIONAL_INSTITUTION", "EVENT"]
N, SEED = 400, 21


def main():
    rnd = random.Random(SEED)
    used_ids = {json.loads(l)["id"] for l in open(ANN / "corrected_final.jsonl", encoding="utf-8")}
    used_ids |= {json.loads(l)["id"] for l in open(ANN / "corrected_extra_final.jsonl", encoding="utf-8")}
    extra_tasks = json.load(open(ANN / "labelstudio_tasks_extra_assistant.json", encoding="utf-8"))
    used_ids |= {t["data"]["id"] for t in extra_tasks}
    banned_groups = set(json.load(open(ANN / "test_groups.json", encoding="utf-8")))
    banned_groups |= set(json.load(open(ANN / "extra_test_groups.json", encoding="utf-8")))

    pool = []
    for line in open(ANN / "preannotated.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in used_ids or r["group"] in banned_groups or not 8 <= len(r["tokens"]) <= 45:
            continue
        spans, _ = fix(r["text"], r["spans"])
        labs = {k[2] for k in spans}
        if labs & set(WEAK):
            r["spans"] = [[s, e, lab, o] for (s, e, lab), o in sorted(spans.items())]
            r["labs"] = labs
            pool.append(r)

    by = {l: [r for r in pool if l in r["labs"]] for l in WEAK}
    for v in by.values(): rnd.shuffle(v)
    chosen, ids, groups, picked_for = [], set(), set(), []
    while len(chosen) < N and any(by.values()):
        for lab in WEAK:
            while by[lab]:
                r = by[lab].pop()
                if r["id"] not in ids and r["group"] not in groups:
                    ids.add(r["id"]); groups.add(r["group"]); chosen.append(r); picked_for.append(lab); break
            if len(chosen) >= N: break

    def ls_task(r):
        res = [{"from_name": "label", "to_name": "text", "type": "labels",
                "value": {"start": s, "end": e, "text": r["text"][s:e], "labels": [lab]}} for s, e, lab, _ in r["spans"]]
        return {"data": {"text": r["text"], "id": r["id"], "source": r["source"], "group": r["group"], "split": "batch3"},
                "predictions": [{"model_version": "preannotation-v2", "result": res}]}

    (ANN / "labelstudio_tasks_batch3.json").write_text(json.dumps([ls_task(r) for r in chosen], ensure_ascii=False, indent=1), encoding="utf-8")
    (ANN / "batch3_ids.json").write_text(json.dumps([r["id"] for r in chosen]), encoding="utf-8")
    WORK.mkdir(exist_ok=True)
    lines = []
    for i, (r, why) in enumerate(zip(chosen, picked_for)):
        s = r["text"]
        for st, en, lab, _ in sorted(r["spans"], reverse=True):
            s = s[:st] + f"[{r['text'][st:en]}|{lab}]" + s[en:]
        lines.append(f"{i} | {why[:4]} | {s}")
    (WORK / "batch3_dump.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"pool {len(pool)}; chosen {len(chosen)} from {len(groups)} documents")
    print("picked for:", dict(Counter(picked_for)))
    print("weak labels present in the drafts:", dict(Counter(l for r in chosen for l in r["labs"] if l in WEAK)))
    assert not ({r['group'] for r in chosen} & banned_groups) and not ({r['id'] for r in chosen} & used_ids)


if __name__ == "__main__":
    main()

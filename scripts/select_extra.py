"""Pick targeted extra sentences for the WEAK classes (human-verified later).

  extra_train (200): more real training examples of weak labels
  extra_test  (100): a second gold test set, from documents that never appear in training

Rules that keep it leak-free:
  * only sentences that are not among the 400 already human-annotated
  * never from the original test documents (data/annotated/test_groups.json)
  * extra_test documents are disjoint from every training document (review + extra_train) and are
    saved to data/annotated/extra_test_groups.json so the dataset builder excludes them from training
  * at most one sentence per document, so one article cannot dominate
Draft labels = the fixed draft (bulk_fix_silver.fix), to be corrected by hand, like the first 400.
"""
import json, random, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bulk_fix_silver import fix

ROOT = Path(__file__).resolve().parent.parent
ANN, WORK = ROOT / "data" / "annotated", ROOT / "data" / "annotated_work"
WEAK = ["LANDMARK_FACILITY", "EDUCATIONAL_INSTITUTION", "EVENT", "TIME", "SPORTS_TEAM", "MEDIA_OUTLET",
        "ORGANIZATION", "LAW_ACT", "GOVERNMENT_BODY"]
N_TRAIN, N_TEST, SEED = 200, 100, 13


def sample(pool, n, rnd):
    by_label = {l: [r for r in pool if l in r["labs"]] for l in WEAK}
    for v in by_label.values():
        rnd.shuffle(v)
    chosen, ids, groups = [], set(), set()
    labels = sorted(WEAK, key=lambda l: len(by_label[l]))
    while len(chosen) < n and any(by_label.values()):
        for lab in labels:
            while by_label[lab]:
                r = by_label[lab].pop()
                if r["id"] not in ids and r["group"] not in groups:
                    ids.add(r["id"]); groups.add(r["group"]); chosen.append(r); break
            if len(chosen) >= n:
                break
    return chosen


def main():
    rnd = random.Random(SEED)
    human = [json.loads(l) for l in open(ANN / "corrected_final.jsonl", encoding="utf-8")]
    human_ids = {r["id"] for r in human}
    human_groups = {r["group"] for r in human}
    test_groups = set(json.load(open(ANN / "test_groups.json", encoding="utf-8")))

    pool = []
    for line in open(ANN / "preannotated.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in human_ids or r["group"] in test_groups:
            continue
        if not 8 <= len(r["tokens"]) <= 50:
            continue
        spans, _ = fix(r["text"], r["spans"])
        labs = {k[2] for k in spans}
        if labs & set(WEAK):
            r["spans"] = [[s, e, lab, o] for (s, e, lab), o in sorted(spans.items())]
            r["labs"] = labs
            pool.append(r)

    groups = sorted({r["group"] for r in pool} - human_groups)
    rnd.shuffle(groups)
    test_pool_groups = set(groups[: len(groups) // 3])               # documents reserved for extra_test
    train_pool = [r for r in pool if r["group"] not in test_pool_groups]
    test_pool = [r for r in pool if r["group"] in test_pool_groups]
    extra_train = sample(train_pool, N_TRAIN, rnd)
    extra_test = sample(test_pool, N_TEST, rnd)
    tr_groups, te_groups = {r["group"] for r in extra_train}, {r["group"] for r in extra_test}
    assert not (te_groups & (tr_groups | human_groups | test_groups)), "extra_test documents leaked into training"
    for r in extra_train: r["split"] = "extra_train"
    for r in extra_test: r["split"] = "extra_test"

    def ls_task(r):
        res = [{"from_name": "label", "to_name": "text", "type": "labels",
                "value": {"start": s, "end": e, "text": r["text"][s:e], "labels": [lab]}} for s, e, lab, _ in r["spans"]]
        return {"data": {"text": r["text"], "id": r["id"], "source": r["source"], "group": r["group"], "split": r["split"]},
                "predictions": [{"model_version": "preannotation-v2", "result": res}]}

    rows = extra_test + extra_train                                  # test first, like the first batch
    (ANN / "labelstudio_tasks_extra.json").write_text(json.dumps([ls_task(r) for r in rows], ensure_ascii=False, indent=1), encoding="utf-8")
    (ANN / "extra_test_groups.json").write_text(json.dumps(sorted(te_groups)), encoding="utf-8")
    WORK.mkdir(exist_ok=True)
    lines = []
    for i, r in enumerate(rows):
        s = r["text"]
        for st, en, lab, _ in sorted(r["spans"], reverse=True):
            s = s[:st] + f"[{r['text'][st:en]}|{lab}]" + s[en:]
        lines.append(f"{i} | {r['split'][6]} | {s}")
    (WORK / "extra_dump.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"pool {len(pool)} sentences; extra_test {len(extra_test)} from {len(te_groups)} docs; extra_train {len(extra_train)} from {len(tr_groups)} docs")
    for name, rs in (("extra_test", extra_test), ("extra_train", extra_train)):
        print(name, dict(Counter(l for r in rs for l in r["labs"] if l in WEAK)))


if __name__ == "__main__":
    main()

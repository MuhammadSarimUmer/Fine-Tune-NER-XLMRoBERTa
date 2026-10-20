"""Build the final train / validation / test / extra_test files, with leakage checks.

Inputs (all under data/):
  annotated/corrected_final.jsonl        400 human-verified (200 test + 200 review)
  annotated/corrected_extra_final.jsonl  100 human-verified extra_test (gold, weak-label enriched)
  annotated/labelstudio_tasks_extra_assistant.json   extra_train (200) = assistant labels, NOT human verified
  annotated/silver.jsonl                 ~15k automatic labels after bulk fix (trusted flag + untrusted spans)
  synthetic/synthetic.jsonl              template sentences (fake phones / CNICs etc.)
  annotated/test_groups.json, extra_test_groups.json   documents that must never be trained on

Output: data/dataset/{train,validation,test,extra_test}.jsonl + dataset_report.txt
Row fields: id, text, tokens, ner_tags (BIO strings), ignore (token indices excluded from the loss),
            source, group, origin
"""
import json, random, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tokenization import TOKEN_RE, spans_to_bio
from normalize_conventions import normalize
from mask_unlabelled import unlabelled_name_mask
from labels import LABELS

MASK_UNLABELLED = True        # round 2 (Step 21); set False to rebuild the round-1 dataset

ROOT = Path(__file__).resolve().parent.parent
ANN, SYN, OUT = ROOT / "data" / "annotated", ROOT / "data" / "synthetic", ROOT / "data" / "dataset"
SEED, VAL_FRACTION = 42, 0.08
report = []


def say(*a):
    s = " ".join(str(x) for x in a); print(s); report.append(s)


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def row(r, origin, ignore=()):
    return dict(id=r["id"], text=r["text"], tokens=r["tokens"], ner_tags=r["ner_tags"], ignore=sorted(ignore),
                source=r["source"], group=r["group"], origin=origin)


# ---------------------------------------------------------------- load
human = read(ANN / "corrected_final.jsonl")
test = [r for r in human if r["split"] == "test"]
review = [r for r in human if r["split"] == "review"]
extra_test = read(ANN / "corrected_extra_final.jsonl")
test_groups = set(json.load(open(ANN / "test_groups.json", encoding="utf-8")))
extra_test_groups = set(json.load(open(ANN / "extra_test_groups.json", encoding="utf-8")))
silver = read(ANN / "silver.jsonl")
synthetic = read(SYN / "synthetic.jsonl")

# extra_train: assistant-v1 predictions turned into BIO (same convention pass as human data)
extra_train = []
for t in json.load(open(ANN / "labelstudio_tasks_extra_assistant.json", encoding="utf-8")):
    if t["data"]["split"] != "extra_train": continue
    pred = next(p for p in t["predictions"] if p["model_version"] == "assistant-v1")
    spans = [(r["value"]["start"], r["value"]["end"], r["value"]["labels"][0]) for r in pred["result"]]
    spans = normalize(t["data"]["text"], spans, None)
    toks, tags = spans_to_bio(t["data"]["text"], spans)
    extra_train.append(dict(id=t["data"]["id"], text=t["data"]["text"], tokens=toks, ner_tags=tags,
                            source=t["data"]["source"], group=t["data"]["group"]))
assert len(test) == 200 and len(review) == 200 and len(extra_test) == 100 and len(extra_train) == 200

# ---------------------------------------------------------------- 1. exclude test DOCUMENTS from training
test_ids = {r["id"] for r in test + extra_test}
held_groups = {r["group"] for r in test + extra_test}
assert held_groups <= (test_groups | extra_test_groups) | held_groups
say("== leakage control ==")
say(f"test documents: {len(held_groups & test_groups)} (random test) + {len(held_groups & extra_test_groups)} (extra test)")

human_ids = {r["id"] for r in human + extra_test + extra_train}    # never use the silver version of these
pool, dropped = [], Counter()
for r in silver:
    if r["id"] in human_ids: dropped["silver copy of a human/assistant sentence"] += 1; continue
    if r["group"] in held_groups: dropped["sibling from a test document"] += 1; continue
    ign = set()
    if not r["trusted"]:                                           # mask unreliable spaCy spans (-100), keep rest
        spans = [tuple(b) for b in r["untrusted_spans"]]
        for i, m in enumerate(TOKEN_RE.finditer(r["text"])):
            if any(m.start() < e and m.end() > s for s, e, _ in spans): ign.add(i)
    n_before = len(ign)
    if MASK_UNLABELLED:                                            # round 2: unlabelled capitalised names are not taught as 'O'
        ign |= unlabelled_name_mask(r["tokens"], r["ner_tags"])
        dropped["tokens masked as unlabelled names"] += len(ign) - n_before
    pool.append(row(r, "silver_trusted" if r["trusted"] else "silver_masked", ign))
for r in review: pool.append(row(r, "human_review"))
for r in extra_train: pool.append(row(r, "assistant_extra"))
say("removed from silver:", dict(dropped))

# ---------------------------------------------------------------- 2. near-duplicate check vs both test sets
def words(t): return re.findall(r"\w+", t.lower())
def grams(t, n=8):
    w = words(t); return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} if len(w) >= n else {" ".join(w)}

test_texts = [r["text"] for r in test + extra_test]
test_grams = set().union(*(grams(t) for t in test_texts))
test_prefix = {" ".join(words(t)[:12]) for t in test_texts}
clean, dup = [], Counter()
for r in pool:
    if r["origin"] == "human_review" or True:
        if " ".join(words(r["text"])[:12]) in test_prefix: dup["same first 12 words as a test sentence"] += 1; continue
        if grams(r["text"]) & test_grams: dup["shares an 8-word span with a test sentence"] += 1; continue
    clean.append(r)
say("removed as near-duplicates of test sentences:", dict(dup))
pool = clean

# ---------------------------------------------------------------- 3. group-level train / validation split
groups = sorted({r["group"] for r in pool})
random.Random(SEED).shuffle(groups)
val_groups = set(groups[: int(len(groups) * VAL_FRACTION)])
train = [r for r in pool if r["group"] not in val_groups]
validation = [r for r in pool if r["group"] in val_groups]
syn_rows = [row(r, "synthetic") for r in synthetic]
train += syn_rows                                                  # synthetic is training only (never validation/test)
test_rows = [row(r, "human_test") for r in test]
extra_rows = [row(r, "human_extra_test") for r in extra_test]

# ---------------------------------------------------------------- 4. hard assertions
g = lambda rows: {r["group"] for r in rows if r["origin"] != "synthetic"}
assert not g(train) & g(validation), "train/validation share a document"
assert not g(train + validation) & held_groups, "test document found in train/validation"
assert not {r["id"] for r in train + validation} & test_ids, "test sentence id in train/validation"
assert not {r["text"] for r in train + validation} & set(test_texts), "identical test text in train/validation"
assert not g(test_rows) & g(extra_rows), "test and extra_test share a document"
assert all(len(r["tokens"]) == len(r["ner_tags"]) for r in train + validation + test_rows + extra_rows)
assert all(t in LABELS for r in train + validation + test_rows + extra_rows for t in r["ner_tags"])
say("ALL LEAKAGE ASSERTIONS PASSED (disjoint document groups, no test ids/texts in train, no near-duplicates)")

# ---------------------------------------------------------------- write + report
OUT.mkdir(parents=True, exist_ok=True)
for name, rows in (("train", train), ("validation", validation), ("test", test_rows), ("extra_test", extra_rows)):
    with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

say("\n== sizes ==")
for name, rows in (("train", train), ("validation", validation), ("test", test_rows), ("extra_test", extra_rows)):
    say(f"{name:11}{len(rows):7} sentences  {len({r['group'] for r in rows}):6} groups  origins {dict(Counter(r['origin'] for r in rows))}")
say("\n== entity counts per label (B- tags, masked tokens not counted) ==")
ents = [e for e in LABELS if e.startswith("B-")]
cnt = {n: Counter() for n in ("train", "validation", "test", "extra_test")}
for n, rows in (("train", train), ("validation", validation), ("test", test_rows), ("extra_test", extra_rows)):
    for r in rows:
        for i, t in enumerate(r["ner_tags"]):
            if t.startswith("B-") and i not in set(r["ignore"]): cnt[n][t[2:]] += 1
say(f"{'label':24}{'train':>8}{'valid':>8}{'test':>7}{'extra':>7}")
for e in ents:
    l = e[2:]; say(f"{l:24}{cnt['train'][l]:8}{cnt['validation'][l]:8}{cnt['test'][l]:7}{cnt['extra_test'][l]:7}")
masked = sum(len(r["ignore"]) for r in train)
say(f"\nmasked tokens in train (ignored by the loss): {masked}")
(OUT / "dataset_report.txt").write_text("\n".join(report), encoding="utf-8")

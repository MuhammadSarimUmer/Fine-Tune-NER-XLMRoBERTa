"""Fine-tune a transformer for token-classification NER and evaluate it.

Run (Colab GPU or any machine with torch + transformers):
    python scripts/train_ner.py --model xlm-roberta-base --out outputs/xlmr_base
Quick smoke test (tiny subset, 1 epoch):
    python scripts/train_ner.py --smoke

Data: data/dataset/{train,validation,test,extra_test}.jsonl  (made by scripts/build_dataset.py)
  - `ignore` = token indices masked with -100 (unreliable silver spans)
  - extra sub-word pieces of a word also get -100 (only the FIRST piece carries the word's label)
Outputs (in --out): model files, metrics_{test,extra_test}.json, report_{...}.txt, errors_{...}.txt

NOTE: we deliberately do NOT import the Hugging Face `datasets` library, because this project has a folder
called `datasets/` that would shadow it when running from the project root.
"""
import argparse, inspect, json, random, sys
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

sys.path.insert(0, str(Path(__file__).parent))
from labels import ENTITY_TYPES, LABELS, LABEL2ID, ID2LABEL

ROOT = Path(__file__).resolve().parent.parent


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


class NerData(torch.utils.data.Dataset):
    """Tokenises pre-split words and aligns labels to sub-word pieces."""

    def __init__(self, rows, tok, max_len):
        self.items = []
        self.truncated = 0
        for r in rows:
            enc = tok(r["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
            ignore, prev, labels = set(r["ignore"]), None, []
            for w in enc.word_ids():
                if w is None or w == prev:
                    labels.append(-100)                       # special token or 2nd/3rd piece of a word
                elif w in ignore:
                    labels.append(-100)                       # masked silver span
                else:
                    labels.append(LABEL2ID[r["ner_tags"][w]])
                prev = w
            if len(set(w for w in enc.word_ids() if w is not None)) < len(r["tokens"]): self.truncated += 1
            self.items.append(dict(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels,
                                   word_ids=enc.word_ids()))

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        return dict(input_ids=it["input_ids"], attention_mask=it["attention_mask"], labels=it["labels"])


def predict_tags(trainer, ds, rows):
    """Predicted BIO strings per ORIGINAL word (first sub-piece decides; truncated words default to O)."""
    out = trainer.predict(ds)
    logits = out.predictions
    preds = []
    for i, r in enumerate(rows):
        tags, prev = ["O"] * len(r["tokens"]), None
        for pos, w in enumerate(ds.items[i]["word_ids"]):
            if w is not None and w != prev:
                tags[w] = ID2LABEL[int(np.argmax(logits[i][pos]))]
            prev = w
        preds.append(tags)
    return preds


def fix_bio(tags):
    """An I-X that does not follow B-X/I-X becomes B-X (standard repair, so a lone I- tag still counts)."""
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]: t = "B-" + t[2:]
        out.append(t); prev = t
    return out


def get_entities(tags):
    """BIO tags -> set of (start, end, type). Same exact-match unit that seqeval/conlleval use."""
    ents, start, typ = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if typ is not None: ents.append((start, i, typ))
            start, typ = (i, t[2:]) if t != "O" else (None, None)
    return set(ents)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(gold, pred):
    """Entity-level exact-match scores. Returns {label: {precision, recall, f1-score, support}} + avg rows."""
    tp, fp, fn = {}, {}, {}
    for g, p in zip(gold, pred):
        G, P = get_entities(g), get_entities(p)
        for e in G & P: tp[e[2]] = tp.get(e[2], 0) + 1
        for e in P - G: fp[e[2]] = fp.get(e[2], 0) + 1
        for e in G - P: fn[e[2]] = fn.get(e[2], 0) + 1
    labs = sorted(set(ENTITY_TYPES) | set(tp) | set(fp) | set(fn))    # always list all 20 labels
    out, rows = {}, []
    for l in labs:
        p, r, f = prf(tp.get(l, 0), fp.get(l, 0), fn.get(l, 0))
        out[l] = {"precision": p, "recall": r, "f1-score": f, "support": tp.get(l, 0) + fn.get(l, 0),
                  "predicted": tp.get(l, 0) + fp.get(l, 0)}
    T, F, N = sum(tp.values()), sum(fp.values()), sum(fn.values())
    p, r, f = prf(T, F, N)
    out["micro avg"] = {"precision": p, "recall": r, "f1-score": f, "support": T + N}
    with_support = [l for l in labs if out[l]["support"] > 0]
    out["macro avg"] = {m: float(np.mean([out[l][m] for l in with_support])) if with_support else 0.0
                        for m in ("precision", "recall", "f1-score")} | {"support": T + N}
    tot = sum(out[l]["support"] for l in labs) or 1
    out["weighted avg"] = {m: sum(out[l][m] * out[l]["support"] for l in labs) / tot
                           for m in ("precision", "recall", "f1-score")} | {"support": T + N}
    return out


def format_report(m):
    lines = [f"{'':26}{'precision':>10}{'recall':>10}{'f1-score':>10}{'support':>9}"]
    for k, v in m.items():
        if k == "micro avg": lines.append("-" * 65)
        if v["support"] == 0 and v.get("predicted", 1) == 0:      # label absent from this test set and never predicted
            lines.append(f"{k:26}{'n/a':>10}{'n/a':>10}{'n/a':>10}{0:9}   (no examples in this test set)")
            continue
        lines.append(f"{k:26}{v['precision']:10.3f}{v['recall']:10.3f}{v['f1-score']:10.3f}{int(v['support']):9}")
    return "\n".join(lines)


def evaluate(name, trainer, tok, rows, max_len, outdir):
    ds = NerData(rows, tok, max_len)
    pred = [fix_bio(p) for p in predict_tags(trainer, ds, rows)]
    gold = [r["ner_tags"] for r in rows]
    metrics = score(gold, pred)
    (outdir / f"metrics_{name}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    text = format_report(metrics)
    (outdir / f"report_{name}.txt").write_text(text, encoding="utf-8")
    print(f"\n=== {name} ({len(rows)} sentences) ===\n{text}")
    # error dump: every sentence where prediction != gold, for error analysis
    with open(outdir / f"errors_{name}.txt", "w", encoding="utf-8") as f:
        for r, g, p in zip(rows, gold, pred):
            if g != p:
                f.write(r["text"] + "\n")
                for t, a, b in zip(r["tokens"], g, p):
                    if a != b: f.write(f"    {t:25} gold={a:24} pred={b}\n")
                f.write("\n")
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "xlmr_base"))
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--human_x", type=int, default=3, help="repeat human-verified training sentences this many times")
    ap.add_argument("--assistant_x", type=int, default=2, help="repeat AI-labelled extra sentences")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check that everything works")
    ap.add_argument("--eval_only", default=None, help="folder of an already trained model (.../best): skip training, just score it "
                                                      "on test, extra_test and pattern_test and write the reports to --out")
    a = ap.parse_args()
    set_seed(a.seed)
    data, out = Path(a.data), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.eval_only:                                    # score a saved model, no training
        tok = AutoTokenizer.from_pretrained(a.eval_only)
        model = AutoModelForTokenClassification.from_pretrained(a.eval_only)
        trainer = Trainer(model=model, data_collator=DataCollatorForTokenClassification(tok),
                          args=TrainingArguments(output_dir=str(out / "tmp"), per_device_eval_batch_size=32, report_to="none",
                                                 fp16=torch.cuda.is_available()))
        for name, fn in (("test", "test.jsonl"), ("extra_test", "extra_test.jsonl"), ("pattern_test", "pattern_test.jsonl")):
            if (data / fn).exists(): evaluate(name, trainer, tok, read(data / fn), a.max_len, out)
        return

    train, val = read(data / "train.jsonl"), read(data / "validation.jsonl")
    test, extra = read(data / "test.jsonl"), read(data / "extra_test.jsonl")
    if a.smoke:
        random.Random(0).shuffle(train); train, val, a.epochs = train[:300], val[:60], 1
    # oversample the small, high-quality human-verified part of the training set
    rep = {"human_review": a.human_x, "human_batch3": a.human_x, "assistant_extra": a.assistant_x}
    train = [r for r in train for _ in range(rep.get(r["origin"], 1))]
    random.Random(a.seed).shuffle(train)
    print(f"train {len(train)}  validation {len(val)}  test {len(test)}  extra_test {len(extra)}")

    tok = AutoTokenizer.from_pretrained(a.model, add_prefix_space=True) if "roberta" in a.model and "xlm" not in a.model \
        else AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model, num_labels=len(LABELS), id2label=ID2LABEL,
                                                            label2id=LABEL2ID)
    dtrain, dval = NerData(train, tok, a.max_len), NerData(val, tok, a.max_len)
    print("sentences truncated by max_len:", dtrain.truncated, dval.truncated)

    def compute_metrics(p):
        logits, labels = p
        pred = np.argmax(logits, axis=-1)
        P, G = [], []
        for pr, lb in zip(pred, labels):
            keep = lb != -100
            P.append(fix_bio([ID2LABEL[int(x)] for x in pr[keep]])); G.append([ID2LABEL[int(x)] for x in lb[keep]])
        m = score(G, P)["micro avg"]
        return dict(precision=m["precision"], recall=m["recall"], f1=m["f1-score"])

    kw = dict(output_dir=str(out / "checkpoints"), num_train_epochs=a.epochs, learning_rate=a.lr,
              per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs * 2, weight_decay=0.01,
              save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
              metric_for_best_model="f1", greater_is_better=True, logging_steps=100, report_to="none",
              fp16=torch.cuda.is_available(), seed=a.seed)
    key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters \
        else "evaluation_strategy"
    kw[key] = "epoch"
    # warm-up over the first 10% of steps; newer transformers dropped `warmup_ratio` (a float < 1 in warmup_steps = ratio)
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in params: kw["warmup_ratio"] = 0.1
    else: kw["warmup_steps"] = 0.1
    unknown = [k for k in kw if k not in params]               # arguments this transformers version no longer has
    if unknown: print("WARNING: ignoring unsupported TrainingArguments:", unknown)
    kw = {k: v for k, v in kw.items() if k in params}
    trainer = Trainer(model=model, args=TrainingArguments(**kw), train_dataset=dtrain, eval_dataset=dval,
                      data_collator=DataCollatorForTokenClassification(tok), compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(str(out / "best")); tok.save_pretrained(str(out / "best"))
    print("validation:", trainer.evaluate())

    # Final scores. The test sets were never used for training, tuning or choosing the epoch.
    evaluate("test", trainer, tok, test, a.max_len, out)
    evaluate("extra_test", trainer, tok, extra, a.max_len, out)
    pat = data / "pattern_test.jsonl"          # synthetic held-out test for PHONE_NUMBER / ID_NUMBER (see make_pattern_test.py)
    if pat.exists():
        print("\n(pattern_test is SYNTHETIC: it measures the phone/CNIC pattern, not real-text performance)")
        evaluate("pattern_test", trainer, tok, read(pat), a.max_len, out)


if __name__ == "__main__":
    main()


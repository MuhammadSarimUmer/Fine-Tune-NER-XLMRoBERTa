from pathlib import Path
p = Path(__file__).parent.parent / "README.md"
s = p.read_text(encoding="utf-8")

def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)

sub("pooled%20F1-0.814", "pooled%20F1-0.826")
sub("precision-0.806-", "precision-0.803-")
sub("recall-0.822-", "recall-0.851-")
sub("fine-tuned for 4 epochs on a Colab T4", "fine-tuned for 6 epochs on a Colab T4 (final run: v4)")
sub("+ 5,390 synthetic sentences", "+ 6,260 synthetic sentences")
sub("**500 human-verified sentences** for testing", "**764 human-verified sentences** (300 gold for testing, 464 for training)")
sub("**P 0.806 / R 0.822 / F1 0.814** on 764", "**P 0.803 / R 0.851 / F1 0.826** on 764")
sub("zero-shot Urdu check: 21 of 27 expected entities found (tiny sample)", "zero-shot Urdu check (done with v1): 21 of 27 expected entities found (tiny sample)")
sub("| Random test (gold, human-verified) | 200 | 426 | 0.805 | 0.826 | **0.816** |", "| Random test (gold, human-verified) | 200 | 426 | 0.793 | 0.847 | **0.820** |")
sub("| Extra test (gold, weak-label enriched) | 100 | 338 | 0.807 | 0.817 | **0.812** |", "| Extra test (gold, weak-label enriched) | 100 | 338 | 0.816 | 0.855 | **0.835** |")
sub("| **Pooled gold** | 300 | 764 | **0.806** | **0.822** | **0.814** |", "| **Pooled gold** | 300 | 764 | **0.803** | **0.851** | **0.826** |")
sub("| 188 | 102 | 0.981 | 1.000 | 0.990 |", "| 188 | 102 | 0.981 | 0.990 | 0.985 |")
sub("| 188 | 84 | 0.848 | 1.000 | 0.918 |", "| 188 | 84 | 0.810 | 0.964 | 0.880 |")
sub("14 of the 20 types reach F1 of at least 0.80. The weak types are `ORGANIZATION` (0.39), `LAW_ACT` (0.45) and `LANDMARK_FACILITY` (0.56).",
    "12 of the 20 types reach F1 of at least 0.80. The weak types are `ORGANIZATION` (0.48) and `LAW_ACT` (0.49).")
sub("### Three runs, one honest conclusion", "### Four runs, one honest conclusion")
sub("| **v1 (final)** | 0.814 | 0.807 | first dataset, 4 epochs |", "| v1 | 0.814 | 0.807 | first dataset, 4 epochs |")
sub("| v3 | 0.810 | 0.800 | + masking of unlabelled names |",
    "| v3 | 0.810 | 0.800 | + masking of unlabelled names |\n| **v4 (final)** | **0.826** | **0.814** | v3 recipe + 264 human-verified batch-3 sentences |")
sub("Two improvement rounds changed the *kind* of error but not the overall score beyond run-to-run noise (about 0.02 F1). The final model was chosen by a rule fixed in advance (pooled micro F1, then macro F1 over all 20 labels, then smallest precision-recall gap), not by the best single number.",
    "Rounds 1 and 2 (more synthetic data, masking) changed the *kind* of error but not the overall score beyond run-to-run noise (about 0.02 F1). Round 3 added only human-verified labels for the weakest types and gave the best scores on both test sets (LANDMARK_FACILITY 0.56 to 0.72, ORGANIZATION 0.39 to 0.48), although +0.012 over v1 is still inside the noise band. The final model was chosen by a rule fixed in advance (pooled micro F1, then macro F1 over all 20 labels, then smallest precision-recall gap), not by the best single number.")
sub("3. **Synthetic data:**", "3. **Synthetic data:**")
sub("5. **Human verify:** 400 + 100 sentences corrected in Label Studio become gold test data.",
    "5. **Human verify:** 400 + 100 sentences corrected in Label Studio become gold test data; a later batch of 264 verified sentences (drafted by an AI assistant, checked by a human) is used for training only.")
sub("AdamW, lr 3e-5, batch 16, best epoch by validation F1.", "AdamW, lr 3e-5, batch 16, 6 epochs, best epoch by validation F1.")
sub("outputs/xlmr_base/best` (about 1.1 GB", "outputs/xlmr_v4/best` (about 1.1 GB")
s = s.replace("outputs/xlmr_base/best", "outputs/xlmr_v4/best")
sub("--out outputs/eval_v1", "--out outputs/eval_v4")
sub("| [`04_urdu_check_v1.ipynb`](notebooks/04_urdu_check_v1.ipynb) | zero-shot Urdu sanity check |",
    "| [`04_urdu_check_v1.ipynb`](notebooks/04_urdu_check_v1.ipynb) | zero-shot Urdu sanity check (run with v1) |\n| [`05_train_v4_colab.ipynb`](notebooks/05_train_v4_colab.ipynb) | train the final model v4 (adds batch-3 human data) |")
sub("metrics, reports and error lists for v1, v2, v3, the evaluation re-scores", "metrics, reports and error lists for v1 to v4, the evaluation re-scores")
sub("only about 500 sentences are human-verified.", "only 764 sentences are human-verified.")

extra = '''
## Organisers' submission files

The folder `student_pack_techfest/student_pack/` holds the organisers' files. `make_submission.py` there is already filled in with our model (self-contained: tokenizer, windows and span conversion). Put the model folder next to it as `xlmr_base/best` (or set `NER_MODEL_DIR`), then:

```bash
cd student_pack_techfest/student_pack
python make_submission.py --input sample_benchmark.jsonl --output my_submission.jsonl
python optional_self_check/score.py --gold optional_self_check/sample_answer_key.jsonl --pred my_submission.jsonl
```

On the day, use the real benchmark file as `--input` and hand in the file it writes. The answer key is for scoring only, never for training.

'''
sub("## Known limitations", extra.lstrip("\n") + "## Known limitations")
p.write_text(s, encoding="utf-8")
print("ok")

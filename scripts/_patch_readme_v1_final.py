from pathlib import Path
p = Path(__file__).parent.parent / "README.md"
s = p.read_text(encoding="utf-8")
repl = [
("pooled%20F1-0.826", "pooled%20F1-0.814"),
("precision-0.803-", "precision-0.806-"),
("recall-0.851-", "recall-0.822-"),
("fine-tuned for 6 epochs on a Colab T4 (final run: v4)", "fine-tuned for 4 epochs on a Colab T4 (**final model: v1**; v4 trained as experiment)"),
("+ 6,260 synthetic sentences", "+ 5,390 synthetic sentences (v1 training set)"),
("**P 0.803 / R 0.851 / F1 0.826** on 764", "**P 0.806 / R 0.822 / F1 0.814** on 764 (v1; our internal gold)"),
("zero-shot Urdu check (done with v1):", "zero-shot Urdu check (v1):"),
("| Random test (gold, human-verified) | 200 | 426 | 0.793 | 0.847 | **0.820** |", "| Random test (gold, human-verified) | 200 | 426 | 0.805 | 0.826 | **0.816** |"),
("| Extra test (gold, weak-label enriched) | 100 | 338 | 0.816 | 0.855 | **0.835** |", "| Extra test (gold, weak-label enriched) | 100 | 338 | 0.807 | 0.817 | **0.812** |"),
("| **Pooled gold** | 300 | 764 | **0.803** | **0.851** | **0.826** |", "| **Pooled gold (v1)** | 300 | 764 | **0.806** | **0.822** | **0.814** |"),
("| 188 | 102 | 0.981 | 0.990 | 0.985 |", "| 188 | 102 | 0.981 | 1.000 | 0.990 |"),
("| 188 | 84 | 0.810 | 0.964 | 0.880 |", "| 188 | 84 | 0.848 | 1.000 | 0.918 |"),
("12 of the 20 types reach F1 of at least 0.80. The weak types are `ORGANIZATION` (0.48) and `LAW_ACT` (0.49).",
 "14 of the 20 types reach F1 of at least 0.80 on internal gold. The weak types are `ORGANIZATION` (0.39), `LAW_ACT` (0.45) and `LANDMARK_FACILITY` (0.56)."),
("| **v4 (final)** | **0.826** | **0.814** | v3 recipe + 264 human-verified batch-3 sentences |",
 "| v4 (experiment) | 0.826 | 0.814 | v3 recipe + 264 human-verified batch-3 sentences |"),
("Rounds 1 and 2 (more synthetic data, masking) changed the *kind* of error but not the overall score beyond run-to-run noise (about 0.02 F1). Round 3 added only human-verified labels for the weakest types and gave the best scores on both test sets (LANDMARK_FACILITY 0.56 to 0.72, ORGANIZATION 0.39 to 0.48), although +0.012 over v1 is still inside the noise band. The final model was chosen by a rule fixed in advance (pooled micro F1, then macro F1 over all 20 labels, then smallest precision-recall gap), not by the best single number.",
 "**Deployed model: v1.** On our internal gold, v4 scores highest (micro F1 0.826 vs 0.814), but the gap is inside noise. On the organisers' 80-sentence sample with their answer key, **v1 micro F1 0.979 vs v4 0.958**, so we submit v1 from `student_pack_techfest/student_pack/xlmr_base/best`. Rounds 1–2 changed error kinds without a clear overall gain; round 3 (v4) helped weak types on internal tests but over-predicted on the sample."),
("batch 16, 6 epochs, best epoch", "batch 16, 4 epochs (v1), best epoch"),
("--out outputs/eval_v4", "--out outputs/eval_v1"),
("| [`05_train_v4_colab.ipynb`](notebooks/05_train_v4_colab.ipynb) | train the final model v4 (adds batch-3 human data) |",
 "| [`05_train_v4_colab.ipynb`](notebooks/05_train_v4_colab.ipynb) | optional v4 training (batch-3 human data) |"),
]
# paths (multiple occurrences)
s = s.replace("outputs/xlmr_v4/best", "outputs/xlmr_base/best")
if "student_pack_techfest/student_pack/xlmr_base/best" not in s:
    s = s.replace(
        "place them at `outputs/xlmr_base/best`.",
        "place v1 at `outputs/xlmr_base/best` and copy the same `best/` folder to `student_pack_techfest/student_pack/xlmr_base/best` for submission.",
    )
for a,b in repl:
    if s.count(a) != 1:
        raise SystemExit(f"count {s.count(a)} for {a[:50]}")
    s = s.replace(a,b)
# add organiser scores line after four runs section if not there
needle = "**Deployed model: v1.**"
if "Organisers' sample" not in s:
    s = s.replace(needle, needle + "\n\n| Check | v1 | v4 |\n|---|---|---|\n| Organisers' sample (80 sentences, 239 entities) | **0.979** micro F1 | 0.958 micro F1 |")
p.write_text(s, encoding="utf-8")
print("readme ok")

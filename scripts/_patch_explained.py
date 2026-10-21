from pathlib import Path
p = Path(__file__).parent.parent / "PROJECT_EXPLAINED.md"
s = p.read_text(encoding="utf-8")

def sub(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)

sub("## F6. Results of the three runs and the honest conclusion", """**Round 3 (v4), more human labels (Step 33-36).** Rounds 1 and 2 only re-processed the same noisy labels. Round 3 added real human work: 400 new sentences rich in the weak types (ORGANIZATION, GOVERNMENT_BODY, LAW_ACT, LANDMARK_FACILITY, EDUCATIONAL_INSTITUTION, EVENT) were selected one per document, never from a test document (`scripts/select_batch3.py`). An AI assistant corrected the drafts (`batch3_corr_01..06.json`, applied by `apply_corrections.py`: 333 spans removed, 223 added); a human then verified **264** of them in Label Studio. These were imported (`import_labelstudio.py`), normalised (`normalize_conventions.py`, 2 sentences changed) and added to training only (repeated x3 like other human data; flag `USE_BATCH3` in `build_dataset.py`). Leakage was checked three ways: document groups and ids disjoint from both test sets (now asserted in the build), no identical text, and only common phrases ("Election Commission of Pakistan (ECP)", "according to an MQM press release") shared with test sentences. v4 uses exactly the v3 recipe, so v3 vs v4 isolates the effect of the human data.

How good was the assistant draft against the human (exact span and label, precision / recall)? PERSON 0.96 / 0.95, GOVERNMENT_BODY 0.95 / 0.91, POLITICAL_PARTY 0.98 / 0.98, EVENT 0.98 / 0.96, LAW_ACT 0.77 / 0.79 (weakest), SPORTS_TEAM 0.85 / 0.61. The raw rules+spaCy draft was far lower (ORGANIZATION precision 0.19, LAW_ACT 0.49 / 0.51): LLM-assisted pre-annotation saves a lot of human effort.

## F6. Results of the four runs and the honest conclusion""")
sub("| v3 | 0.786 | 0.836 | 0.810 | 0.800 |\n", "| v3 | 0.786 | 0.836 | 0.810 | 0.800 |\n| **v4 (final)** | 0.803 | 0.851 | **0.826** | **0.814** |\n")
sub("**Conclusion.** The three runs tie. Both rounds changed the *kind* of error but not the total. What limits the score is **label quality**",
    "**Conclusion.** v1, v2 and v3 tie. Rounds 1 and 2 changed the *kind* of error but not the total. v4, which adds only human-verified labels, is best on both test sets (random test 0.820 vs v1 0.816; extra_test 0.835 vs v1 0.812) and on the targeted types: LANDMARK_FACILITY 0.56 to 0.72, ORGANIZATION 0.39 to 0.48, EDUCATIONAL_INSTITUTION 0.71 to 0.93 (only 14 test entities), LAW_ACT 0.45 to 0.49. TIME, COURT_LEGAL_BODY and EVENT dipped slightly (10 to 18 entities each). The +0.012 over v1 is still inside the noise band, so v4 is the best by our rule, not a proven improvement. What limits the score is **label quality**")
sub("## F7. How the final model was chosen", "## F7. How the final model was chosen (rule and decision)")
sub("applied to all three models scored by the same code", "applied to all models scored by the same code")
sub("Step 1 tie (0.814 / 0.814 / 0.810). Step 2: **v1 0.807** vs v3 0.800 vs v2 0.798. **v1 is the final model** and also the simplest.",
    "With three models (v1, v2, v3): step 1 tie (0.814 / 0.814 / 0.810), step 2 gave v1 0.807 vs v3 0.800 vs v2 0.798, so **v1 was chosen**. After round 3, the same rule on four models: step 1 v4 0.826 vs v1 0.814 = v2 0.814 vs v3 0.810; step 2 v4 0.814 vs v1 0.807. **v4 is the final model** (v1 is the fallback). v4 leans to recall (P 0.803 / R 0.851, gap 0.047 vs v1's 0.016); the gap only breaks ties.")
sub("**Label bands (final model, 20 labels).** F1 at least 0.90: 6; at least 0.80: 14; at least 0.70: 17; below 0.60: 3 (ORGANIZATION, LAW_ACT, LANDMARK_FACILITY).",
    "**Label bands (final model v4, 20 labels).** F1 at least 0.90: 6; at least 0.80: 12; below 0.60: 2 (ORGANIZATION, LAW_ACT).")
sub("`python scripts/predict.py --model outputs/xlmr_base/best --text \"...\"`", "`python scripts/predict.py --model outputs/xlmr_v4/best --text \"...\"`")
sub("The benchmark's required format is unknown; adapt the output writer when the organisers publish it.",
    "The organisers' required format is JSONL with character offsets; their script `student_pack_techfest/student_pack/make_submission.py` has our model filled in (same steps as `predict.py`, self-contained). Rules: exactly the 20 labels, no overlaps, end exclusive, strict exact match, micro F1 ranks (macro F1 tie-break). The 80-sentence `sample_benchmark.jsonl` with `sample_answer_key.jsonl` is used only for scoring, never training.")
sub("we ran the final model on 8 hand-written Urdu sentences", "we ran the first model (v1; the check predates v4) on 8 hand-written Urdu sentences")
sub("`03_eval_predict_all_models` (re-score all runs, paragraph test), `04_urdu_check_v1`.", "`03_eval_predict_all_models` (re-score all runs, paragraph test), `04_urdu_check_v1`, `05_train_v4_colab` (final model).")
sub("- **How did you choose the final model?** A rule fixed before re-scoring: pooled micro F1, then macro F1 over 20 labels, then precision-recall balance.",
    "- **How did you choose the final model?** A rule fixed before re-scoring: pooled micro F1, then macro F1 over 20 labels, then precision-recall balance. v1 won among the first three; after adding 264 human-verified sentences, v4 won on both metrics and on both test sets, though only slightly (+0.012, inside noise).")
sub("- **Why did your improvements not help?** They changed the type of error (fewer missed, more spurious) but not the total; the limit is human label volume, not model size.",
    "- **Why did your improvements not help?** The first two rounds (synthetic data, masking) changed the type of error but not the total. The one thing that helped, a little, was more human-verified labels (v4): the limit is human label volume, not model size.")
sub("An AI assistant drafted first-pass corrections for the 400 and 100 gold sentences and labelled 200 extra training sentences. Every gold test sentence was reviewed by a human; the 200 extra training sentences were not, and we disclose it.",
    "An AI assistant drafted first-pass corrections for the 400 and 100 gold sentences, the 400 batch-3 sentences and labelled 200 extra training sentences. Every gold test sentence and the 264 batch-3 training sentences were reviewed by a human; the 200 extra training sentences and the 136 unreviewed batch-3 sentences were not used as human data, and we disclose it.")
p.write_text(s, encoding="utf-8")
print("ok")

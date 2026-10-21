"""One-off: patch make_report_pdf.py for the v4 final model (kept for the record)."""
from pathlib import Path
p = Path(__file__).parent / "make_report_pdf.py"
s = p.read_text(encoding="utf-8")

def sub(old, new, count=1):
    global s
    n = s.count(old)
    assert n >= 1 and (count is None or n == count), (n, old[:70])
    s = s.replace(old, new)

# --- run registry / loading
sub('''def load(run, s):
    p = EV / f"eval_{run}" / f"metrics_{s}.json"''',
'''FINAL = "v4"                                   # final model (Step 36)
V4_DIR = ROOT / "trained_reports" / "v4_xlmr_base"

def load(run, s):
    p = (V4_DIR if run == "v4" else EV / f"eval_{run}") / f"metrics_{s}.json"''')
sub('RUNS = {"v1": "v1", "v2": "v2", "v3": "v3"}', 'RUNS = {"v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4"}\nRCOL = [TEAL, ORANGE, BLUE, GREEN]')

# --- figures
sub('steps = [("1 Setup"', 'steps = [("1 Setup"')
sub('("6 Human verify", "400 + 100 gold\\nsentences")', '("6 Human verify", "400 + 100 + 264\\nsentences")')
sub('names = ["Synthetic\\n(templates)", "Trusted silver\\n(auto labels)", "Masked silver\\n(partly ignored)", "Human-checked\\nreview", "AI-labelled\\nextra"]',
    'names = ["Synthetic\\n(templates)", "Trusted silver\\n(auto labels)", "Masked silver\\n(partly ignored)", "Human-checked\\n(review + batch 3)", "AI-labelled\\nextra"]')
sub('vals = [5390, 9516, 2974, 186, 179]', 'vals = [6260, 9419, 2833, 424, 179]')
sub('"Training set of the final model (18,245 sentences)"', '"Training set of the final model (19,115 sentences)"')
sub('tp, pred, sup = POOL["v1"]\n    rows = sorted([(l, f1_of("v1", l), sup[l])', 'tp, pred, sup = POOL[FINAL]\n    rows = sorted([(l, f1_of(FINAL, l), sup[l])')
sub('"Final model (v1): F1 per entity type"', '"Final model (v4): F1 per entity type"')
sub('''    w = 0.26
    for i, (r, c) in enumerate(zip(RUNS, [TEAL, ORANGE, BLUE])):
        xs = [k + (i - 1) * w for k in range(4)]
        b = ax.bar(xs, vals[r], w, label={"v1": "v1 (final)", "v2": "v2", "v3": "v3"}[r], color=c)''',
'''    w = 0.2
    for i, (r, c) in enumerate(zip(RUNS, RCOL)):
        xs = [k + (i - 1.5) * w for k in range(4)]
        b = ax.bar(xs, vals[r], w, label={"v1": "v1", "v2": "v2", "v3": "v3", "v4": "v4 (final)"}[r], color=c)''')
sub('ax.set_ylim(0.7, 0.88)', 'ax.set_ylim(0.7, 0.9)')
sub('ncol=3, loc="upper left")\n    ax.set_title("Three training runs', 'ncol=4, loc="upper left")\n    ax.set_title("Four training runs')
sub('fontsize=6.8)', 'fontsize=6.2)')
sub('''fig, ax = plt.subplots(figsize=(6.6, 3.0)); w = 0.26
    for i, (r, c) in enumerate(zip(RUNS, [TEAL, ORANGE, BLUE])):
        xs = [k + (i - 1) * w for k in range(len(labs))]''',
'''fig, ax = plt.subplots(figsize=(6.6, 3.0)); w = 0.2
    for i, (r, c) in enumerate(zip(RUNS, RCOL)):
        xs = [k + (i - 1.5) * w for k in range(len(labs))]''')
sub('ax.legend(frameon=False, fontsize=8, ncol=3)\n    ax.set_title("The weakest entity types in all three runs"', 'ax.legend(frameon=False, fontsize=8, ncol=4)\n    ax.set_title("The weakest entity types in all four runs"')
sub('labs = ["ORGANIZATION", "LAW_ACT", "LANDMARK_FACILITY", "GOVERNMENT_BODY", "CITY", "EDUCATIONAL_INSTITUTION"]',
    'labs = ["ORGANIZATION", "LAW_ACT", "LANDMARK_FACILITY", "GOVERNMENT_BODY", "CITY", "EDUCATIONAL_INSTITUTION"]')
sub('b = bands("v1")\n    names', 'b = bands(FINAL)\n    names')

# --- numbers
sub('tpv, prv, suv = POOL["v1"]\nmi = micro("v1"); mi2 = micro("v2"); mi3 = micro("v3")',
    'tpv, prv, suv = POOL[FINAL]\nmi = micro(FINAL); mi1 = micro("v1"); mi2 = micro("v2"); mi3 = micro("v3")')
sub('t1 = load("v1", "test")["micro avg"]; e1 = load("v1", "extra_test")["micro avg"]; pt = load("v1", "pattern_test")\nb1 = bands("v1")',
    't1 = load(FINAL, "test")["micro avg"]; e1 = load(FINAL, "extra_test")["micro avg"]; pt = load(FINAL, "pattern_test")\nb1 = bands(FINAL)')
sub("tm, em = load(\"v1\", \"test\"), load(\"v1\", \"extra_test\")", "tm, em = load(FINAL, \"test\"), load(FINAL, \"extra_test\")")
sub("f1_of('v1', l)", "f1_of(FINAL, l)", count=2)
sub("f1_of('v1','PHONE_NUMBER')", "f1_of(FINAL,'PHONE_NUMBER')")
sub("f1_of('v1','ID_NUMBER')", "f1_of(FINAL,'ID_NUMBER')")
sub("f1_of('v1','ORGANIZATION')", "f1_of(FINAL,'ORGANIZATION')")
sub("f1_of('v1','LAW_ACT')", "f1_of(FINAL,'LAW_ACT')")

# --- executive summary
sub('"(3) produced first-draft labels with regular expressions, gazetteers and a pretrained spaCy model, (4) corrected 500 sentences by hand "',
    '"(3) produced first-draft labels with regular expressions, gazetteers and a pretrained spaCy model, (4) corrected 764 sentences by hand "')
sub('"automatically checked leakage controls, (7) fine-tuned the model, and (8) analysed errors and ran two improvement rounds.")',
    '"automatically checked leakage controls, (7) fine-tuned the model, and (8) analysed errors and ran three improvement rounds.")')
sub(''' and {b1[1]} of the 20 entity types reach F1 of at least 0.80.''', ''' and {b1[1]} of the 20 entity types reach F1 of at least 0.80.''')
sub(" The weakest types are \"\n            f\"ORGANIZATION (F1 {pct(f1_of(FINAL,'ORGANIZATION'))}), LAW_ACT ({pct(f1_of(FINAL,'LAW_ACT'))}) and LANDMARK_FACILITY ({pct(f1_of('v1','LANDMARK_FACILITY'))}).\")]",
    " The weakest types are \"\n            f\"ORGANIZATION (F1 {pct(f1_of(FINAL,'ORGANIZATION'))}) and LAW_ACT ({pct(f1_of(FINAL,'LAW_ACT'))}).\")]")
sub('''story += [callout("Two improvement rounds (more synthetic data and case-number rules; then masking of unlabelled names) changed the <i>kind</i> of error "
                  "but did not raise overall F1 beyond run-to-run noise (all three runs score 0.810 to 0.814). We therefore selected the final model with a rule fixed in "
                  "advance, not by the best single number. We report this openly because the limit is the amount of human-verified labels, not the model.",''',
'''story += [callout("Two improvement rounds (more synthetic data and case-number rules; then masking of unlabelled names) changed the <i>kind</i> of error "
                  "but did not raise overall F1 beyond run-to-run noise (v1 to v3 score 0.810 to 0.814). A third round that only added 264 human-verified sentences "
                  "for the weakest types raised micro F1 to 0.826 and macro F1 to 0.814, with the gains on exactly those types. The gain is small and inside the "
                  "noise band, but it is consistent on both test sets. We selected the final model (v4) with a rule fixed in advance, not by the best single number. "
                  "The limit is the amount of human-verified labels, not the model.",''')
sub('"sources (Wikipedia, Wikinews and openly available Kaggle datasets of Pakistani news and Supreme Court judgments), "',
    '"sources (Wikipedia, Wikinews and openly available Kaggle datasets of Pakistani news and Supreme Court judgments), "')
sub('''story += [P("<b>Also tested.</b> The model was trained on English only, yet on a small hand-made Urdu sample it found 21 of 27 expected entities with the right "
            "type (zero-shot, tiny sample, see section 12).")]''',
'''story += [P("<b>Also tested.</b> The model was trained on English only, yet on a small hand-made Urdu sample the first model (v1) found 21 of 27 expected entities with the right "
            "type (zero-shot, tiny sample, see section 12). The organisers' sample benchmark and answer key are used only for evaluation (section 11.7).")]''')

# --- kaggle licence fact removed
sub(' Per-dataset licences were not individually verified.', '')

# --- 7.6 batch 3 (insert before section 8)
sub('''# ---- 8 dataset & leakage''',
'''story += [P("7.6 Round 3: batch of 400 new sentences (human-verified: 264)", "H2"),
          P("To raise the weak types we selected <b>400 further sentences</b> rich in ORGANIZATION, GOVERNMENT_BODY, LAW_ACT, LANDMARK_FACILITY, EDUCATIONAL_INSTITUTION and EVENT "
            "(one per document, none from a test document). An AI assistant corrected the draft labels; a human then verified <b>264</b> of them in Label Studio. "
            "Measured against the human, the assistant draft reached precision / recall of 0.96 / 0.95 for PERSON, 0.95 / 0.91 for GOVERNMENT_BODY, 0.98 / 0.98 for POLITICAL_PARTY and 0.98 / 0.96 for EVENT, "
            "but only 0.77 / 0.79 for LAW_ACT; the raw rule-and-spaCy draft was far lower (ORGANIZATION precision 0.19, LAW_ACT 0.49). "
            "The 264 sentences are used for training only (repeated 3 times); test documents and ids are disjoint from them and the build script asserts this.")]

# ---- 8 dataset & leakage''')

# --- section 8 tables
sub('"Figure 2. Composition of the final model\'s training set."', '"Figure 2. Composition of the final model\'s training set (before oversampling)."')
sub('["Silver copies of human or assistant sentences", "300 dropped; the human version is used"]', '["Silver copies of human or assistant sentences", "564 dropped; the human version is used"]')
sub('["Train (final model)", "18,245", "synthetic 5,390; trusted silver 9,516; masked silver 2,974; human review 186; assistant extra 179"]',
    '["Train (final model)", "19,115", "synthetic 6,260; trusted silver 9,419; masked silver 2,833; human review 186; human batch 3: 238; assistant extra 179"]')

# --- section 9
sub('["Epochs (final model)", "4", "standard for NER fine-tuning; best epoch kept on validation F1"]',
    '["Epochs (final model)", "6", "v1 was still improving at 4 epochs; best epoch kept on validation F1"]')
sub('["Oversampling", "human review x3, assistant extras x2", "the best data is only about 2% of the total"]',
    '["Oversampling", "human data (review, batch 3) x3, assistant extras x2", "the best data is only about 3% of the total"]')

# --- 11.3 text
sub('and {b1[4]} stay below 0.60 (ORGANIZATION, LAW_ACT, LANDMARK_FACILITY). "', 'and {b1[4]} stay below 0.60 (ORGANIZATION, LAW_ACT). "')

# --- 11.1 text
sub("give almost the same F1", "give similar F1")

# --- 11.6 table + text
sub('''                 ["<b>v1 (final)</b>: 4 epochs, first data", pct(mi[0]), pct(mi[1]), pct(mi[2]), pct(macro20("v1")), f"{abs(mi[0]-mi[1]):.3f}"],''',
'''                 ["v1: 4 epochs, first data", pct(mi1[0]), pct(mi1[1]), pct(mi1[2]), pct(macro20("v1")), f"{abs(mi1[0]-mi1[1]):.3f}"],''')
sub('''                 ["v3: + masking of unlabelled names", pct(mi3[0]), pct(mi3[1]), pct(mi3[2]), pct(macro20("v3")), f"{abs(mi3[0]-mi3[1]):.3f}"]],''',
'''                 ["v3: + masking of unlabelled names", pct(mi3[0]), pct(mi3[1]), pct(mi3[2]), pct(macro20("v3")), f"{abs(mi3[0]-mi3[1]):.3f}"],
                 ["<b>v4 (final)</b>: v3 recipe + 264 human batch-3 sentences", pct(mi[0]), pct(mi[1]), pct(mi[2]), pct(macro20("v4")), f"{abs(mi[0]-mi[1]):.3f}"]],''')
sub('''          P("The runs differ by less than 0.005 in micro F1, far below the noise level. The rounds changed the error profile (v2 gained precision; v3 gained recall, with ORGANIZATION 0.45 and LANDMARK_FACILITY 0.65 best, but more spurious and wrong-type words) "
            "without a measurable overall gain, and LAW_ACT stayed near 0.45 in every run."),''',
'''          P("v1, v2 and v3 differ by less than 0.005 in micro F1, far below the noise level: rounds 1 and 2 changed the error profile (v2 gained precision; v3 gained recall) without a measurable overall gain. "
            "<b>Round 3 (v4)</b> is the same recipe as v3 plus 264 human-verified sentences, so the v3-to-v4 difference (micro +0.016, macro +0.014) isolates the effect of that human data. "
            "It is concentrated where we added data: LANDMARK_FACILITY 0.56 (v1) to 0.72, ORGANIZATION 0.39 to 0.48, EDUCATIONAL_INSTITUTION 0.71 to 0.93 (only 14 test entities) and LAW_ACT 0.45 to 0.49. "
            "TIME, COURT_LEGAL_BODY and EVENT dipped slightly (10 to 18 test entities each). v4 scores 0.820 on the random test (v1 0.816) and 0.835 on extra_test (v1 0.812), so it is not winning on one set only. "
            "Its advantage over v1 (+0.012 micro F1) is still inside the roughly 0.02 noise band, so we call it the best by our rule, not a proven improvement. v4 leans towards recall (0.851) over precision (0.803)."),''')
sub('img("weak_labels.png", 12.0, "Figure 6. The weak types in all three runs.")', 'img("weak_labels.png", 12.0, "Figure 6. The weak types in all four runs.")')
sub('"Figure 5. All three runs re-scored with the same code and data (pooled human gold; macro F1 over 20 labels)."', '"Figure 5. All four runs scored with the same code and data (pooled human gold; macro F1 over 20 labels)."')
sub('''                  f"Step 1 is a tie ({pct(mi[2])} / {pct(mi2[2])} / {pct(mi3[2])}); step 2 gives v1 {pct(macro20('v1'))}, v2 {pct(macro20('v2'))}, v3 {pct(macro20('v3'))}. "
                  "<b>v1 is the final model</b>, which is also the simplest. We fixed this rule before re-scoring all three models with an evaluate-only mode (the v3 re-score reproduced the original numbers exactly).",''',
'''                  f"Micro F1: v1 {pct(mi1[2])}, v2 {pct(mi2[2])}, v3 {pct(mi3[2])}, <b>v4 {pct(mi[2])}</b>. Macro F1 over 20 labels: v1 {pct(macro20('v1'))}, v2 {pct(macro20('v2'))}, v3 {pct(macro20('v3'))}, <b>v4 {pct(macro20('v4'))}</b>. "
                  "v4 ranks first on both and on both test sets separately, so it is the final model (v1 was the earlier choice, made by the same rule before v4 existed). We fixed this rule before re-scoring the models with an evaluate-only mode (the v3 re-score reproduced the original numbers exactly).",''')
sub('A side-by-side run of all three models on one long paragraph', 'Before v4 existed, a side-by-side run of v1, v2 and v3 on one long paragraph')

# --- 11.7 new subsection (organiser pack) before Urdu section
sub('''# ---- 12 urdu''',
'''story += [P("11.7 Organisers' sample benchmark", "H2"),
          P("The organisers supplied a submission script, a sample benchmark of 80 sentences and an answer key. The required format is JSONL with character offsets (start inclusive, end exclusive), "
            "exactly the 20 official labels and no overlaps; scoring is exact match on start, end and label, ranked by micro F1 with macro F1 as tie-break. This is the same kind of metric we use. "
            "We filled the two marked sections of their script with our model (same tokenizer, 100-word windows, first-sub-token labelling, BIO converted to character spans). "
            "The answer key is used for scoring only and never for training. The result of this check is not part of the numbers above.")]

# ---- 12 urdu''')

# --- urdu wording
sub('We ran the final model, unchanged, on <b>8 hand-written Urdu sentences</b>', 'We ran the first model (v1, unchanged; the Urdu check was done before v4 existed) on <b>8 hand-written Urdu sentences</b>')

# --- challenges / limitations / conclusion
sub('["Few human labels", "Only 500 sentences are human-verified out of 18,000+"', '["Few human labels", "Only 764 sentences are human-verified out of 19,000+"')
sub('["Improvement rounds that did not help", "Overall F1 stayed 0.81", "Reported honestly; selected model by a pre-agreed rule"]',
    '["Improvement rounds that did not help", "Rounds 1 and 2 left overall F1 at 0.81; round 3 (human labels) added about 0.01", "Reported honestly; selected model by a pre-agreed rule"]')
sub('Human-verified data is about 500 sentences; the 179 extra sentences are assistant-labelled and not human-verified.', 'Human-verified data is 764 sentences (400 + 100 + 264); the 179 extra sentences are assistant-labelled and not human-verified.')
sub('"Weak types: ORGANIZATION (0.39), LAW_ACT (0.45) and LANDMARK_FACILITY (0.56) are well below the rest.",', '"Weak types: ORGANIZATION (0.48) and LAW_ACT (0.49) are well below the rest; the final model\'s advantage over v1 is within noise.",')
sub('Prediction: <font name=\'%s\'>python scripts/predict.py --model outputs/xlmr_base/best --input file.txt', 'Prediction: <font name=\'%s\'>python scripts/predict.py --model outputs/xlmr_v4/best --input file.txt')
sub('["notebooks/", "01 pre-annotation, 02 training, 03 evaluate and predict with all models, 04 Urdu check"]', '["notebooks/", "01 pre-annotation, 02 training, 03 evaluate and predict with all models, 04 Urdu check, 05 training of v4"]')
sub('The final fine-tuned XLM-RoBERTa-base reaches F1 {pct(mi[2])} on 764 human-verified entities, "', 'The final fine-tuned XLM-RoBERTa-base (v4) reaches F1 {pct(mi[2])} on 764 human-verified entities, "')
sub('"Our most important finding is that label quality, not model size or training length, limits the score: three differently trained models tie within noise."',
    '"Our most important finding is that label quality, not model size or training length, limits the score: three differently trained models tie within noise, and the one change that helped was more human-verified labels."')
sub('"<b>Next steps with more time:</b> (1) verify the 200 assistant-labelled sentences and add human labels for ORGANIZATION, GOVERNMENT_BODY, LAW_ACT and LANDMARK_FACILITY; "',
    '"<b>Next steps with more time:</b> (1) verify the remaining 136 batch-3 sentences and the 200 assistant-labelled sentences, and add human labels for ORGANIZATION, GOVERNMENT_BODY and LAW_ACT; "')
sub('"(5) adapt the output format once the organisers publish the benchmark format.")]', '"(5) re-check Urdu and the organiser sample with the final model.")]')

p.write_text(s, encoding="utf-8")
print("patched")

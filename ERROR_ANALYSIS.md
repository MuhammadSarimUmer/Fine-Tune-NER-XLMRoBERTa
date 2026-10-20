# Error Analysis - what we did, how, and why

Error analysis = studying *where and why* the trained model is wrong, so that the next improvement fixes causes instead of guessing.
Round 1 compares **v1** (first training run, 4 epochs) with **v2** (after the fixes below). Files:
`trained_reports/v1_xlmr_base/` (and `v2/` once the second run is done): `report_*.txt`, `metrics_*.json`, `errors_*.txt`, `error_summary.txt`.

---------------------------------------------------------------------------------------------------

## Step 1 - Measure first (overall and per label)
- **What:** read `report_test.txt` and `report_extra_test.txt`: precision, recall and F1 per entity type and overall.
- **How:** `scripts/train_ner.py` scores at entity level: an entity is correct only if start, end and type all match.
  A missing, extra, half-right or wrong-type entity each counts as a mistake.
- **Why:** an overall number hides where the problems are. Per-label numbers show which labels to work on, and the same metric
  is what the hidden benchmark uses (overall and per entity).
- **Result (v1):** test F1 0.816, extra_test F1 0.812. Weak labels: LAW_ACT (~0.45), ORGANIZATION (0.30-0.45),
  LANDMARK_FACILITY (recall 0.25-0.50), GOVERNMENT_BODY (0.71-0.84).

## Step 2 - Beware small supports
- **What:** look at the `support` column (number of true entities) before trusting any F1.
- **Why:** EDUCATIONAL_INSTITUTION, EVENT, TIME and SPORTS_TEAM have only 2-3 spans in the random test, so a perfect 1.0 or a 0.33
  means almost nothing. That is exactly why we built `extra_test` (12-18 spans per weak label): it is the better evidence for rare labels.
- **How to report:** quote random-test numbers for big labels (PERSON, DATE, GOVERNMENT_BODY...) and extra_test numbers for rare ones.

## Step 3 - Is it overfitting or just the data? (compare the two test sets)
- **What:** compare test (0.816) with extra_test (0.812).
- **Why:** if the model had memorised easy patterns it would do much worse on the harder, weak-label-enriched set. Nearly equal
  scores mean it generalises about equally on both. (Validation F1 is only used for choosing the epoch.)

## Step 4 - Dump every wrong sentence
- **What:** `errors_test.txt` / `errors_extra_test.txt` list each sentence with a mistake and, under it, each wrong word with
  `gold=` (human label) and `pred=` (model label).
- **How:** written by `evaluate()` in `train_ner.py` after the final scoring.
- **Why:** numbers say *how much*, examples say *why*. You can only fix a cause you have seen.

## Step 5 - Count the kinds of mistakes (`scripts/error_summary.py`)
- **What:** every wrong word is put in one of four kinds: **missed** (gold entity, model said O), **spurious** (gold O, model said
  entity), **confusion** (wrong type), **boundary** (right type, wrong start/end).
- **How:** `python scripts/error_summary.py trained_reports/v1_xlmr_base` (rerun for v2 to compare).
- **Why:** each kind has a different cure: missed = model lacks examples; spurious = noisy/over-eager training labels;
  confusion = unclear or inconsistent type boundaries; boundary = inconsistent span rules.
- **Result (v1, wrong words):**

| | missed | confusion | spurious | boundary |
|---|---|---|---|---|
| random test (157 wrong words, 66 sentences) | 61 | 43 | 38 | 15 |
| extra_test (153 wrong words, 49 sentences) | 76 | 50 | 17 | 10 |

  - Missed words by gold label: random test GOVERNMENT_BODY 16, LAW_ACT 12, LANDMARK_FACILITY 10, ORGANIZATION 7;
    extra_test EVENT 13, LANDMARK_FACILITY 13, GOVERNMENT_BODY 11, ORGANIZATION 10, LAW_ACT 10.
  - Spurious words: **DATE is 18 of 38** on the random test (and 5 of 17 on extra_test): the model invents dates.
  - Top confusions: ORGANIZATION -> GOVERNMENT_BODY (7 + 4), CITY -> PERSON (6 + 6), ORGANIZATION -> POLITICAL_PARTY (4),
    LANDMARK_FACILITY -> CITY (6, extra_test), SPORTS_TEAM -> COUNTRY (3).

## Step 6 - Group the examples into causes (reading the error files)
For each frequent pattern we asked: *is the model wrong, is the training data wrong, or is the test label questionable?*

| # | Pattern (examples) | Likely cause | Fix |
|---|---|---|---|
| 1 | GOVERNMENT_BODY vs ORGANIZATION vs POLITICAL_PARTY (abbreviations and bodies the model has never seen) | few clean examples; the silver spaCy ORGANIZATION layer is only ~0.3 precise and was masked, so the model got little signal | more distinct bodies/organisations in synthetic data (associations, companies, IGOs, commissions) |
| 2 | Long law names broken up (names with brackets, years, "and") | training laws were mostly short | synthetic laws with brackets/years/long titles; law-specific sentences |
| 3 | Landmarks missed or tagged CITY/PERSON (rivers, dams, ports, shrines, temples, police stations) | LANDMARK_FACILITY was thin (about 540 train spans) and spaCy's landmark layer is masked | many distinct landmarks of those kinds + templates |
| 4 | **Spurious DATE** on years inside case numbers and citations (`C.A.538/2022`, `2010 SCMR 1962`) | the silver data contained them as DATE (88 cases) | rule: such years are never DATE (silver 88 -> 20) + synthetic case-number sentences with unlabelled years |
| 5 | Spurious DATE on vague phrases ("recent weeks", "period") | the human labels also treat *"last year", "a day"* as DATE, so these are not clearly junk; the gold for "recent weeks" itself is inconsistent | **no change**: not enough evidence, and changing it would risk worse consistency |
| 6 | COUNTRY vs SPORTS_TEAM ("Pakistan tour", "South Africa call off ...") | context-dependent; national teams only labelled with strong match cues | synthetic national-team sentences (touring, toss, series) |
| 7 | Unfamiliar place names read as PERSON (village names) | the model leans on capitalised-name-looks-like-person | more place variety; mostly a data-size limit, so not fixed in round 1 |
| 8 | A few **gold labels look wrong** (e.g. one "Kashmir" tagged CITY; "Jamaica" is gold COUNTRY, model said CITY = model error) | annotation slips are normal at this scale | **not edited** (see rule below); mention in the report |

## Step 7 - Decide what to fix, and what NOT to fix
- **Fix:** causes that are clearly about training data and that human labels support (rows 1-4, 6).
- **Don't fix:** things where the evidence is unclear (row 5), single-instance oddities, and anything that would mean editing test labels.
- **Why:** each change has to be justified from the *human* labels, otherwise we are just tuning the data to the test sentences.

## Step 8 - Check each rule against human labels BEFORE applying (`scripts/date_audit.py`)
- **What:** for the DATE problems we counted how the human review labels treat the same patterns.
- **How:** the audit shows human review sentences never label a year inside a case/citation as DATE (0 cases), but do label
  "last year", "a day" etc. as DATE (13 spans). So: apply the case-year rule, drop the "vague phrase" idea.
- **Why:** a rule that contradicts the annotator would make the data *less* consistent. We also confirmed the new rule changes
  no existing human label (`normalize_conventions.py` still changes the same 9 human sentences as before).

## Step 9 - Make the fixes (`Step 17` in `PROJECT_LOG.md`)
1. `normalize_conventions.py`: years in case numbers and law-report citations are never DATE (applies to silver and human data).
2. `generate_synthetic.py`: synthetic data 5,390 -> 6,260 sentences; more landmarks, government bodies, organisations, laws with
   brackets/years, national teams in match context; new CASE_NOISE templates.
3. `train_ner.py` unchanged; training 6 epochs instead of 4 (validation was still improving).
4. Dataset rebuilt with `build_dataset.py`; all leakage assertions passed.

## Step 10 - The honesty rule (data leakage in error analysis)
- **The risk:** if we read test errors and then add *those exact names* to our data, the model would "know the answers" and the score
  would be fake. Even the idea of a fix can leak information from the test set.
- **What we did to limit it:**
  - no entity string from the test/extra_test errors was added to any gazetteer or template (we used *other* names of the same kind);
  - rules were validated on the review split (human labels), not on test labels;
  - test labels were never edited, even where they look wrong;
  - only categories of errors (not individual sentences) drove changes.
- **Residual caveat (say it in the report):** because the *categories* came from test errors, the random test is no longer fully
  unseen for round 1. Report v2 as "after one error-analysis round". The hidden benchmark (Oct 22) is the clean measure.

## Step 11 - Retrain and compare (v2 vs v1)
- **What:** run `notebooks/02_train_colab.ipynb` (writes `outputs/xlmr_v2`), put the report files in `trained_reports/v2/`, then:
  `python scripts/error_summary.py trained_reports/v2`
- **Compare:** overall F1 on both test sets, per-label F1 for the targeted labels (LAW_ACT, ORGANIZATION, LANDMARK_FACILITY,
  GOVERNMENT_BODY, SPORTS_TEAM), and the count of spurious DATE words.
- **Why:** an improvement round is only a result if we show before and after. If a targeted label did not improve, say so
  and say what you would try next; that is also a valid finding.

| Metric | v1 | v2 |
|---|---|---|
| Test micro F1 | 0.816 | 0.825 |
| Extra-test micro F1 | 0.812 | 0.801 |
| LAW_ACT F1 (test / extra) | 0.48 / 0.42 | 0.59 / 0.38 |
| ORGANIZATION F1 (test / extra) | 0.45 / 0.30 | 0.44 / 0.24 |
| LANDMARK_FACILITY F1 (test / extra) | 0.36 / 0.64 | 0.50 / 0.53 |
| GOVERNMENT_BODY F1 (test / extra) | 0.71 / 0.84 | 0.74 / 0.83 |
| Spurious DATE words (test / extra) | 18 / 5 | 16 / 4 |

**v2 verdict (honest reading).** Overall change is within noise (+0.009 on test, -0.011 on extra_test; with 338-426 entities one or two entities swing F1 by about 0.5 point and the difference between runs is about +-2 points). Small real gains: spurious DATE words 18 -> 16 and 5 -> 4 (the case-number rule helps a little), LANDMARK_FACILITY test recall 0.25 -> 0.375, LAW_ACT test F1 0.48 -> 0.59. No gain, or worse: ORGANIZATION (recall 0.35 -> 0.30 and 0.21 -> 0.14), LAW_ACT on extra_test, missed words on test 61 -> 74.
**Why synthetic additions did not move real-text scores much:** template sentences with listed names teach the model the *name lists* and the template wording, not how real news phrases organisations, bodies and laws. The model also still sees thousands of real silver sentences in which real organisations (spaCy ORGANIZATION is masked only where spaCy found them) and landmarks are left unlabelled = O, so it learns that such names are not entities (low recall). This is the main remaining cause, and more templates do not remove it.
**What this teaches (say in the report):** improving rare labels needs more *real, correctly labelled* examples (or fewer wrongly-O training examples), not just more synthetic text.

## Step 12 - What to say in the report and presentation
1. Per-label results and which labels are weak, with supports.
2. The error taxonomy (missed / spurious / confusion / boundary) with the numbers above.
3. The causes and the fixes (table in step 6), including what was *not* changed and why.
4. The leakage precautions in step 10 and the residual caveat.
5. Remaining limits: mostly silver labels, small human gold set, rare labels with few test spans, no real PHONE/ID test data.
6. Next ideas if there were more time: annotate more human gold for ORGANIZATION / GOVERNMENT_BODY / LANDMARK_FACILITY,
   a gazetteer for Pakistani villages/towns, a second model for comparison, class-weighted loss.

## Round 2 (v3): masking unlabelled capitalised names
See PROJECT_LOG.md Step 21 for method, evidence (review split only: covers 46% of missed-entity tokens at a 0.7% cost) and how to judge it (pooled F1 on both test sets, per-label recall, v1/v2/v3 side by side). Note: v1 and v2 pooled F1 are 0.814 and 0.815, a tie; do not report round 1 as an improvement.

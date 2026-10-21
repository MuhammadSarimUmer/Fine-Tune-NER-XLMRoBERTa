# Project Log (what we did, how, why) - for the presentation

Updated after every step. Newest steps at the bottom.

---

## Step 0 - Problem setup (18 Oct)
- **What:** Fixed the 20 entity types, BIO tagging (41 labels), wrote `GUIDELINES.md`, created `scripts/labels.py` as the single source of truth for labels.
- **Why:** In NER the "target variable" is the label schema. Ambiguous rules (court vs government body, team vs country) cause inconsistent labels, which cap the score. One shared label file prevents id mismatches between training and evaluation.
- **Decisions:** English-first (most Pakistani public text is English; Urdu annotation is slow), XLM-RoBERTa-base as the planned model (multilingual, fits Colab T4), flat entities (no nesting), longest span wins.

## Step 1 - Data sourcing policy (18-19 Oct)
- **What:** Ranked sources by legal risk into tiers (open licence, public records, commercial news), logged every source in `DATA_SOURCES.md`.
- **Why:** The competition requires respecting site terms and avoiding private data. A documented policy shows data ethics.
- **Decisions:** Common Crawl and LLM-generated sentences not used by default. Phone/CNIC are synthetic only. Team later decided to use the downloaded open Kaggle datasets as a whole.

## Step 2 - Collection (19 Oct)
- **Wikipedia/Wikinews:** `scripts/scrape_wikipedia.py`, official MediaWiki API, identifying User-Agent with contact email, 1.5 s delay, caps, every document logged. Result: 209 Wikipedia + 23 Wikinews documents (~2M characters). Some guessed category names did not exist, so we checked and replaced them.
- **Kaggle datasets inspected** (`datasets/`): kept news headlines (26.7k), Dawn (778) and Tribune (3,309) articles, Supreme Court judgments (2,809 files), penal code (405 rows), census table. Removed Imran Khan tweets (garbled Urdu, personal social media), IND-vs-PAK tweets (informal, not Pakistan-specific, 238 MB in 4 duplicate formats), Imran Khan Reddit (informal forum text).
- **Why:** Diverse sources (news, legal, encyclopedic) generalise better to an unseen benchmark than one big source.

## Step 3 - Cleaning and balancing (19 Oct) - `scripts/build_corpus.py`
- **What:** Merged all sources into `data/raw/sentences.jsonl` (15,390 sentences) and built gazetteers in `data/gazetteers/` (provinces, divisions, districts, tehsils, penal code sections).
- **How / why:**
  1. Tolerant decoding: files mix UTF-8 with stray cp1252 bytes; a fallback decoder avoids crashes and garbled characters.
  2. Normalising text: fancy quotes/dashes/nbsp to plain characters, so the tokenizer sees consistent input.
  3. Removing junk: photo captions, "Read more/Published in" lines, URLs, ALL-CAPS headers, very short or very long sentences, mostly non-letter lines. These are not natural sentences and would teach noise.
  4. Sentence splitting that protects abbreviations (Mr., Rs., M. Khan) so entities are not cut in half.
  5. Supreme Court: dropped empty files, cut the header block (judge lists, case numbers), removed OCR-broken sentences ("Ordinanc e") and citation lists.
  6. Entity-light sentences (no capitalised word or digit) mostly dropped (25% down to 11%): they only teach the label O. A small share stays as negative examples so the model learns what is NOT an entity.
  7. Headlines: favoured Pakistan-related ones (60% were world news).
  8. De-duplication: exact and near-duplicate (same first 12 words).
  9. Balancing: per-source caps and per-document caps so legal text (65 MB) does not dominate and every document contributes a few sentences. Result: 18% Tier 1, 24% Tier 2, 58% Tier 3 (news-heavy, since the benchmark is probably news-like).
  10. Each sentence stores its source document (`group`) so the train/validation/benchmark split can be done by document to avoid leakage.
- **Known limits:** minor OCR noise in court text; rare classes (SPORTS_TEAM, MEDIA_OUTLET, EVENT, LANDMARK_FACILITY, TIME) likely thin; provinces gazetteer lacks ICT, Gilgit-Baltistan, AJK (add by hand).

## Step 4 - Synthetic data (19 Oct) - `scripts/generate_synthetic.py`, `scripts/tokenization.py`
- **What:** 5,390 template-generated sentences with automatic BIO labels in `data/synthetic/synthetic.jsonl` (token + tag format, ~12k entity mentions, 268-1,643 per label).
- **How:** ~150 sentence templates with slots like `{PERSON}`, `{MONEY}`, `{CNIC}`. Each slot is filled from a generator or entity list, the static text and the entity are tokenized separately, and the entity tokens get B-/I- tags. Labels are therefore exact, no manual work.
  - CNIC: `#####-#######-#` (first digit 1-7), plus no-dash and spaced variants. Phones: mobile `0300-1234567`, `+92 300 1234567`, `+923001234567`, `0092-...`, plus landlines. All random digits, no real people.
  - MONEY (Rs, Rs., PKR, $, lakh/crore/million/billion, Tribune-style `Rs5.63 million`), PERCENT (`15%`, `15 percent`, `15 per cent`), DATE and TIME in many formats.
  - Rare classes (SPORTS_TEAM, MEDIA_OUTLET, EVENT, LANDMARK_FACILITY, LAW_ACT, COURT_LEGAL_BODY, EDUCATIONAL_INSTITUTION, ...) get 260-320 sentences each because real text has few.
  - Names: random first name + surname combinations from our lists (not real individuals).
- **Why:** Phone/CNIC must be synthetic by competition rule; pattern classes get perfect labels cheaply; rare classes need extra examples.
- **Safeguards:** unlabelled numbers (FIR no., flight no., page counts) appear in templates so the model does not learn "every number is an entity"; the template id is stored in `group` so we can hold out whole templates for validation; `tokenization.py` is shared so real and synthetic data are tokenized identically (numbers like `35202-1234567-1` and `1,200` stay single tokens; `'s` is split off).
- **Risks to mention:** template overfitting (model learns sentence patterns), entity lists are not exhaustive, synthetic share should stay a minority (5.4k synthetic vs 15.4k real).

## Step 5 - Pre-annotation (19 Oct) - `scripts/preannotate.py`, `notebooks/01_preannotation_colab.ipynb`
- **What:** a first-draft labelling of all 15,390 real sentences (`data/annotated/preannotated.jsonl`) plus a 400-sentence stratified review sample for manual correction (`labelstudio_tasks.json`, `labelstudio_config.xml`).
- **How (3 layers):** (1) regex for PHONE/CNIC/MONEY/PERCENT/TIME/DATE/law references, (2) gazetteers (curated Pakistani lists + census districts/divisions) matched longest-first, (3) spaCy pretrained NER (PERSON, ORG, GPE, FAC, EVENT, LAW...) mapped to our 20 labels; spaCy ORG spans are re-classified with keyword rules (University -> EDUCATIONAL_INSTITUTION, Court -> COURT_LEGAL_BODY, Ministry/Police -> GOVERNMENT_BODY ...). Overlaps resolved: strong regex first, longest span, then layer priority. Courtesy titles stripped from PERSON.
- **Why:** hand-labelling 15k sentences is impossible in 3 days; correcting a draft is much faster. Layering gives precision from rules and recall for unseen names from the pretrained model.
- **Rules-only dry run (no spaCy), local:** 9,392 / 15,390 sentences got at least one entity; zero PERSON (expected: needs spaCy); real text has no PHONE/ID (synthetic only); few LANDMARK (34), EDUCATIONAL (22), EVENT (26), SPORTS_TEAM (31), TIME (36) -> rare classes, covered by synthetic data + review sample.
- **Review tool:** Label Studio (local), tasks include the draft as "predictions" so humans only fix errors. `scripts/import_labelstudio.py` converts the export back to token/BIO and prints per-label precision/recall of the draft vs the human (use in the report as annotation-quality evidence). Round trip tested.
- **Known weaknesses of the draft:** gazetteer false positives on ambiguous words (e.g. district name "Khyber" as CITY); spaCy may tag Pakistani party/org names inconsistently; every label needs a human spot-check.

### Step 5b - First Colab run inspected, rules fixed (19 Oct)
- **Findings (400-sentence sample):** many correct matches (courts, parties, laws, money, most names) but systematic errors from the translation rules: PERSON on a newspaper ("Dawn") and a mosque ("Lal Masjid"); CITY on a person ("Zardari") and halves of a party name ("Insaf", "Movement"); LANDMARK on legal citations ("Sections 34,35,...") and "Platoon No.44"; SPORTS_TEAM on "United Malays National Organisation" (keyword "United") and "...Response Team"; ORGANIZATION on titles ("MNA"); MEDIA on "National Press Club"; DATE on "days"/"same day".
- **Fixes (all in `classify_spacy` and keyword lists):** PERSON that is a known organisation/place name is dropped; facility words (Masjid, Stadium, Airport...) re-route to LANDMARK_FACILITY; FAC spans with section/article/digits dropped; ORG titles/ranks (MNA, SHO, DC...) dropped; sports keywords tightened with exclusions; media keywords tightened; keyword rules only trusted on names of <= 6 words; generic DATE words dropped; unknown GPE no longer becomes CITY if it looks like a party name, a person name or a 2-3 letter abbreviation; gazetteer extended (PSL, Asia Cup, World Cup, West Indies, FIFA, Lal Masjid, ...).
- **Agreement flag:** each span now records how many layers independently proposed it (`agree`) and each sentence how many proposals conflicted with the chosen spans (`conflicts`). Where regex/gazetteer/spaCy agree the draft is trusted more; disagreements go to humans first.
- **Evaluation design fix:** the test set is now a purely RANDOM sample (200 sentences, whole documents held out) instead of the stratified sample, so test F1 is not distorted by over-represented rare labels. The review sample (200) is half rare-label coverage, half "uncertain" sentences, and excludes test documents. Label Studio tasks list the 200 test sentences first (`data.split`), so the gold test set gets done even if time runs short.

### Step 5c - Second Colab run inspected, rules fixed again (19 Oct)
- **Run 2 stats:** 15,390 sentences, 13,371 with at least one entity; spans by layer: spaCy 20,699 / gazetteer 10,907 / regex 4,597; 3,732 sentences where layers disagreed. PERSON 8,883; ORGANIZATION 4,924 (4,774 from spaCy's catch-all); rare labels still thin in real text: SPORTS_TEAM 49, EDUCATIONAL_INSTITUTION 136, EVENT 157, TIME 247, LANDMARK 317, MEDIA 319.
- **Systematic errors found in spaCy's generic ORG bucket:** court abbreviations (IHC, LHC, SHC), titles (CJP, CEC), documents/ids (FIR, CNIC), "Islam", "Trump" (person), "Khyber-Pakhtunkhwa" (province), bare generic nouns ("Court" 385x, "Commission", "Committee", "University"); countries as CITY (England, Britain, South Africa, New Zealand); "Urdu" as place; "Insaf" as PERSON (62x); TIME durations ("24 hours", "three-hour").
- **Fixes:** gazetteer extended (court/government/party abbreviations, province abbreviations, UK/South Africa/New Zealand, well-known foreign persons); a curated gazetteer name now overrides spaCy's coarse type; drop-lists for documents/titles/languages/religions; bare generic nouns never labelled (matches GUIDELINES: "do not label the court / the government"); TIME durations dropped (part-of-day words kept); upper-case datelines (ISLAMABAD) looked up in title case. 40 unit cases written for `classify_spacy` with fake spaCy entities, all pass.
- **Process lesson:** a debug run overwrote real output twice; debug runs (`--no-spacy`, `--limit`) now write to `data/annotated_debug/`.

### Step 5d - Third Colab run verified, draft frozen (19 Oct)
- **Verified the fixes took effect:** bare "Court" 385 -> 0, FIR/Islam/CJP/Insaf/Urdu as entities -> 0; IHC now COURT_LEGAL_BODY (30), Khyber-Pakhtunkhwa now PROVINCE_REGION (38), England now COUNTRY (32), Trump now PERSON (37).
- **Totals (15,390 sentences):** 13,078 with >=1 entity; layers disagreed in 2,841 sentences (was 3,732). ORGANIZATION 4,227 (down from 4,924), PERSON 8,698, COURT_LEGAL_BODY 1,145. Rare labels in real text remain thin: SPORTS_TEAM 49, EDUCATIONAL_INSTITUTION 130, EVENT 153, TIME 156, LANDMARK 312, MEDIA 319.
- **Residual weakness:** long tail of unlisted abbreviations in spaCy's generic ORG bucket (ATC, CCI, PSP, MMA, JIT, DRAP ...), bare ministry shorthand ("Finance", "Interior"), a few persons tagged ORG ("Nisar", "Khattak"), upper-case datelines tagged ORG (QUETTA). Decision: stop rule tuning (diminishing returns, deadline) and let the human-correction accuracy table measure these; any systematic issue found can be bulk-fixed on `preannotated.jsonl` without re-running spaCy.
- **Decision:** no LLM layer (deadline, extra noise source, simpler story: rules + gazetteers + spaCy + human verification).

## Step 6 - Assistant (LLM) correction of the 400 review/test sentences
- **What:** an AI assistant read all 400 sentences (200 test + 200 review) with the automatic draft and wrote correction lists (`data/annotated_work/corr_01..08.json`: remove wrong spans, add missing/relabelled spans). `scripts/apply_corrections.py` applies them and writes `data/annotated/labelstudio_tasks_assistant.json` (273 spans removed, 179 added, 399 of 400 sentences touched or checked).
- **How:** `predictions[0]` = assistant-corrected version (what the human reviews), `predictions[1]` = original rules+gazetteer+spaCy draft. `import_labelstudio.py` now prints a precision/recall table for each, so we can report how good the automatic pipeline and the assistant draft each were against the human.
- **Why:** the strongest quality step is a stronger "pre-annotator" than rules alone, with a human as final judge. Disclosure for the report: the labels are model-assisted and human-verified; the human pass is what makes the test set gold.
- **Typical fixes:** abbreviations (CCI, WAPDA, NIH -> GOVERNMENT_BODY), teams named like countries (Pakistan/England in cricket -> SPORTS_TEAM), multi-token law citations as one LAW_ACT, durations ("two weeks") and case numbers removed, places mis-tagged as persons (Umerkot, Noshki).
- **Next for the human:** import the assistant file into Label Studio, verify, export JSON, run `import_labelstudio.py`.

## Step 6b - Guidelines extended during annotation
- Added sections to `GUIDELINES.md`: dates/times/durations (day + time of day = two spans, durations not entities, years inside names stay inside), places and regions (Kashmir, Thar etc. = PROVINCE_REGION), organisations (abbreviations take the type of what they stand for, bare titles not entities).
- **Why:** these decisions had only existed in chat and in the assistant's corrections; any annotator (and the report) needs them written down so labels stay consistent. The hidden benchmark may differ on some of these edge cases, so consistency matters more than any single choice.

- **Decision (20 Oct): durations ("23 months", "two weeks") are DATE**, not unlabelled and not TIME. Reason: OntoNotes (the convention spaCy and most NER benchmarks follow) labels periods as DATE; TIME is reserved for time of day. Note: the assistant's first corrections had removed durations, so the 400-sentence file is inconsistent on this; the human pass must add them, and the dataset builder will normalise the rest.

## Step 7 - Human annotation imported, draft quality measured
- **What:** the human finished all 400 sentences in Label Studio (`export-400.json`); `import_labelstudio.py` wrote `corrected.jsonl` and compared both drafts with the human labels (the export holds only prediction ids, so the script reads predictions from `labelstudio_tasks_assistant.json`). `scripts/analyze_review.py` prints the same table per split plus the most frequent review-split errors.
- **Findings (automatic draft):** weak on ORGANIZATION (precision ~0.3: spaCy ORG junk), GOVERNMENT_BODY recall ~0.57, LANDMARK/EVENT/SPORTS_TEAM recall 0.3-0.5, bare years tagged DATE inside names/case numbers, national teams tagged COUNTRY. Strong on PERSON (~0.88), POLITICAL_PARTY (~0.9), MONEY/PERCENT.
- **Why per split:** bulk-fix decisions use REVIEW-split errors only; the TEST split is kept clean for the final report (no tuning on test).
- **Caveat:** test split has very few spans for rare labels (LANDMARK 8, EDUCATIONAL 2, EVENT 3, SPORTS_TEAM 3, TIME 2), so per-label test scores for those will be noisy.

## Step 8 - Conventions normalised
- **Decisions (20 Oct):** (a) leading "The" excluded from names (MEDIA_OUTLET); (b) capitalised bare court names ("High Court", "Supreme Court") ARE COURT_LEGAL_BODY; lowercase "the court"/"apex court" are not; (c) durations are DATE (see above); (d) a year inside a case number is not a DATE.
- **Why:** each rule makes labels uniform across train and test and, for (b), targets what a court-records benchmark most likely expects. They are judgement calls, documented in `GUIDELINES.md`, and should be mentioned in the report as limitations.
- **How:** `scripts/normalize_conventions.py` -> `normalize(text, spans)`; run on the human data now (`corrected_final.jsonl`: 10 of 400 sentences changed, i.e. the annotator already followed most rules) and reused on the silver data in the dataset builder.

## Step 9 - Bulk-fix of the silver (automatic) labels
- **What:** `scripts/origin_precision.py` measures draft precision per (label, layer) on the REVIEW split; `scripts/bulk_fix_silver.py` applies fixes to the 14,990 non-human sentences and writes `data/annotated/silver.jsonl` with a `trusted` flag. The 400 human sentences are never overwritten.
- **Finding that drives the design:** layers differ hugely in reliability. Reliable (precision ~0.9+): PERSON (spaCy), POLITICAL_PARTY, PROVINCE_REGION, gazetteer GOVERNMENT_BODY/CITY, regex MONEY/PERCENT. Unreliable: spaCy ORGANIZATION 0.32, spaCy GOVERNMENT_BODY 0.43, spaCy TIME 0.20, spaCy EDUCATIONAL_INSTITUTION 0.57, spaCy EVENT/LANDMARK ~0.5.
- **Fixes (each justified by review-split errors):** extra gazetteer names (BOI, CCI, PCB, MPC, Kashmir, Thar, FATA, education abbreviations, PML variants, UAE/US ...); drop generic junk (IGP, CEC, "Local Commission"); drop lowercase "apex court"; TIME only if it contains a digit or a time-of-day word, duration-looking TIME -> DATE; "Wednesday night" split into DATE + TIME; years merged into names ("Elections Act 2017" -> one LAW_ACT, "ICC Champions Trophy 2013" and "2013 elections" -> one EVENT); national teams COUNTRY -> SPORTS_TEAM when strong sport cues (runs, wickets, innings, ODI, T20, cricket, hockey ...) are in the sentence; then `normalize_conventions.normalize` (leading "The", durations, bare court names).
- **Deliberately NOT added:** IPP (also "independent power producers"), NAP (also "National Action Plan"), GDA and NP (ambiguous): a wrong gazetteer hit would mislabel many sentences.
- **Bug caught by spot-checking:** my first sport-cue list ("final", "series", "match") turned "Pakistan" into SPORTS_TEAM in political sentences; narrowed to strong cues (195 conversions instead of 304).
- **Trust rule:** a silver sentence is `trusted` if no unreliable spaCy-origin span (ORGANIZATION, GOVERNMENT_BODY, EDUCATIONAL_INSTITUTION, EVENT, LANDMARK_FACILITY, MONEY) remains. **Why drop instead of patch:** an unlabelled real entity teaches the model "this entity is O", which is worse than leaving the sentence out. Result: 11,295 trusted, 3,695 untrusted.
- **Quality check (sentence exact-match vs human):** trusted 0.68 (review) / 0.72 (test); untrusted 0.32 / 0.04. The flag separates good from bad sentences. Review numbers are in-sample (rules were tuned on them); the test numbers were looked at ONCE as an out-of-sample estimate and NOT used to change any rule.
- **Known remaining noise:** context-dependent cases (a country hosting an event tagged as team), junk spaCy DATE phrases ("a single century", "Election Day"), ORGANIZATION and LANDMARK still weak. Rare labels thin on test.

## Step 10 - Weak classes: targeted extra annotation
- **Why:** `scripts/label_counts.py` showed the real, trusted data is almost empty for the weak labels (ORGANIZATION 60, LANDMARK_FACILITY 50, EDUCATIONAL_INSTITUTION 10, EVENT 105, TIME 93 spans) because the trust rule of Step 9 drops sentences containing unreliable spaCy spans, which are mostly exactly these labels; and the 200-sentence test set holds only 2-8 gold spans for them (scores would be noise).
- **What:** `scripts/select_extra.py` picks 300 sentences rich in weak labels (one sentence per document, 8-50 tokens, round-robin over the weak labels, rarest first): 200 `extra_train` and 100 `extra_test`. The assistant pre-annotated all 300 (`extra_corr_01..05.json`, 239 spans removed, 177 added, 145 sentences changed) via `apply_corrections.py` (now with `--tasks/--corr/--out` options and `re:` regex needles). Output: `data/annotated/labelstudio_tasks_extra_assistant.json`; human verification still pending.
- **Leakage design:** extra_test documents are chosen first (a random third of eligible documents is reserved), are disjoint from the original test documents, review documents and extra_train documents (asserted in the script), and are saved in `extra_test_groups.json` so the dataset builder removes ALL their sentences from training. extra_test is reported SEPARATELY from the 200 random test sentences (it is weak-label-enriched, not representative of the natural label mix).
- **Second-pass check (assistant re-read all 300 corrected sentences + `scripts/check_tasks.py` automatic checks: span edges, punctuation, leading "The", brackets, overlaps, TIME/DATE sanity):** found and fixed (a) a script bug: adding the single word "Pakistan" cut the longer span "Pakistan Cricket Board"; `apply_corrections.py` now never lets a shorter added span cut a longer existing one and logs each skip (4 skips, all correct); (b) "early Friday" -> "Friday"; (c) standalone "Landhi"/"Korangi" now CITY. Limitation: this is the same model checking its own work, so it catches slips, not systematic bias; the human verification remains the real check.
- **Plan for the other weak-class fixes (Step 11, dataset builder):** mask uncertain spans (label -100) instead of dropping whole sentences, keep ORGANIZATION spans that pass a shape filter, more gazetteers/templates for weak labels, class-weighted loss.

## Notes for the presentation
- Pretrained model handles unseen names through context and subword patterns; we plan name-swap augmentation and an evaluation split that separates seen vs unseen names.
- Keep a slide on challenges: encodings, OCR noise, class imbalance, ambiguity rules, legal sourcing.

## Next
- Name-swap augmentation on real annotated sentences (after pre-annotation).
- Pre-annotation (gazetteers + regex + pretrained NER), manual correction sample, label counts per class.

## Step 11 - Import of the 100 human-verified extra_test sentences (21 Oct)
**What:** Imported data/annotated/export-100.json (Label Studio) -> corrected_extra.jsonl -> corrected_extra_final.jsonl (conventions pass; 0 changes needed, annotator already followed GUIDELINES.md). All 100 tasks submitted, all split = extra_test.
**How:** import_labelstudio.py --tasks labelstudio_tasks_extra_assistant.json --out corrected_extra.jsonl, then normalize_conventions.py --src corrected_extra.jsonl --dst corrected_extra_final.jsonl.
**Why:** These are a second GOLD test set, enriched for weak labels (LANDMARK, ORGANIZATION, EDUCATIONAL_INSTITUTION, EVENT, LAW_ACT...). Reported SEPARATELY from the 200 random test sentences because they were selected, not random. Their documents are excluded from training (extra_test_groups.json) = no leakage.
**Result (draft vs human, exact span+label):**
- assistant-v1 (my corrected version): overall P ~0.92, R ~0.89. Weakest: LAW_ACT 0.62/0.67, ORGANIZATION 0.71/0.71, LANDMARK 0.80/0.89.
- preannotation-v2 (regex+gazetteer+spaCy): ORGANIZATION P 0.22, LAW_ACT 0.38/0.40, LANDMARK 0.65/0.61. Confirms that raw spaCy ORGANIZATION must not be trusted.
**Use in report:** the ~0.92/0.89 figure estimates how noisy the 200 extra_train sentences are (assistant labels, NOT human verified, disclosed as such). Human gold still: 400 + 100 sentences.
**Next:** dataset builder with leakage assertions, then Colab fine-tuning.

## Step 12 - Dataset builder with leakage checks (20 Oct)
**What:** scripts/build_dataset.py -> data/dataset/{train,validation,test,extra_test}.jsonl + dataset_report.txt. (bulk_fix_silver.py now also stores untrusted_spans.)
**How:**
1. Test (200) and extra_test (100) are the human gold files and are only written to their own files.
2. Whole test DOCUMENTS are removed from training: 965 silver siblings dropped (same group). 300 silver copies of human/assistant sentences dropped (human version used instead).
3. Near-duplicate check (first 12 words, any shared 8-word span) vs both test sets: 64 train sentences dropped.
4. Silver spans from unreliable spaCy layers are not dropped as sentences: their tokens go in ignore (-100 in training) = model neither learns them nor learns a false 'O'. 8,213 tokens masked.
5. Train/validation split by group (document), 8% of groups to validation (seed 42). Synthetic data is train only.
6. Hard asserts: disjoint groups train/val/test/extra_test, no test id or text in train, tags valid, lengths match.
**Result:** train 18,245 sentences (synthetic 5,390; trusted silver 9,516; masked silver 2,974; human review 186; assistant extra 179), validation 1,206, test 200, extra_test 100. All assertions passed.
**Known limits:** validation has no PHONE/ID (synthetic only) and few rare-label spans; test has none of PHONE/ID and very few EVENT/SPORTS_TEAM/TIME/EDU spans -> report rare-label scores with caution, use extra_test for them. Headlines have 1 sentence per group, so the n-gram check is what protects them.
**Next:** Colab fine-tuning notebook (XLM-RoBERTa-base), evaluate on test and extra_test separately.

## Step 13 - Training script, Colab notebook, README (20 Oct)
**What:** scripts/train_ner.py (fine-tune + evaluate), notebooks/02_train_colab.ipynb, outputs/colab_bundle.zip (labels.py, train_ner.py, data/dataset/*.jsonl), a full explanatory README.md (problem, model choice, every pipeline step what/why/how, limits, run commands).
**How (training):** XLM-RoBERTa-base token classification; words -> subword pieces, only the first piece keeps the label, others and masked silver spans get -100; human review sentences x3, AI-labelled extras x2; lr 3e-5, batch 16, 4 epochs, warmup 10%, fp16; best epoch chosen on VALIDATION F1; test and extra_test evaluated once at the end with seqeval (overall + per label), plus error lists (errors_*.txt) for analysis.
**Why:** standard stable recipe; oversampling because human data is only ~2% of training; test sets are never used to choose anything.
**Gotcha found:** the project folder datasets/ shadows the Hugging Face datasets library, so the trainer uses a plain torch Dataset and never imports it.
**Status / honesty:** the script compiles, but it could NOT be run on this PC (no torch). It has not been tested end-to-end yet: run the notebook's --smoke cell first on Colab.
**Next:** run on Colab, save results to Drive, then error analysis + one improvement round, then report/slides.

## Step 14 - Colab install error fixed (20 Oct)
**Problem:** Colab cell 2 failed: seqeval builds from source (setup.py egg_info) and breaks on current Colab Python/setuptools.
**Fix:** removed the seqeval dependency. train_ner.py now computes entity-level exact-match precision/recall/F1 itself (get_entities, score, format_report): an entity is correct only if start, end and type all match (same rule as seqeval/conlleval). Unit-checked on a toy case (micro, per-label, I- repair). Notebook install cell is now pip install -U transformers accelerate. outputs/colab_bundle.zip rebuilt.
**Why:** fewer dependencies = fewer failure points on the day; the metric is simple enough to own and explain in the report.

## Step 15 - Smoke-test fix: TrainingArguments version drift (20 Oct)
**Problem:** first Colab smoke run loaded data/model fine (train 312, val 60, test 200, extra 100; 0 truncated) but crashed: new transformers has no warmup_ratio.
**Fix:** train_ner.py inspects TrainingArguments and uses warmup_ratio=0.1 if it exists, otherwise warmup_steps=0.1 (float < 1 = ratio); any other unsupported argument is printed as a WARNING instead of crashing. Bundle rebuilt.
**Note:** the 'UNEXPECTED lm_head / MISSING classifier' load report is normal: the new 41-tag classifier head starts untrained and is learned in fine-tuning.

## Step 16 - First real training run (XLM-RoBERTa-base, 4 epochs): results (20 Oct)
Files: trained_reports/v1_xlmr_base/ (report_*, metrics_*.json, errors_*). Trained on the 18,245-sentence train set.
| Set | Precision | Recall | F1 |
|---|---|---|---|
| test (200 random gold, 426 entities) | 0.805 | 0.826 | **0.816** |
| extra_test (100 gold, weak-label enriched, 338 entities) | 0.807 | 0.817 | **0.812** |
Strong (test F1): PERSON .92, PERCENT 1.0, MONEY .94, MEDIA_OUTLET .91, POLITICAL_PARTY .88, PROVINCE_REGION .87, COUNTRY .86, COURT_LEGAL_BODY .86. Weak: LAW_ACT .48/.42, ORGANIZATION .45/.30, LANDMARK_FACILITY .36/.64 (recall .25/.50), GOVERNMENT_BODY .71/.84, SPORTS_TEAM (3 spans) .33. Perfect scores on EDUCATIONAL_INSTITUTION/EVENT/TIME in the random test rest on 2-3 spans and mean nothing.
Error categories (read from errors_test.txt): (1) GOVERNMENT_BODY vs ORGANIZATION vs POLITICAL_PARTY confusion (OGRA, CCP, PSB, SCBA...); (2) long LAW_ACT names with brackets/years/'and' split; (3) landmarks missed (rivers, ports, temples, police stations) or tagged CITY/PERSON; (4) spurious DATE on vague phrases and on years in case numbers/citations (silver spaCy junk); (5) COUNTRY vs SPORTS_TEAM; (6) unfamiliar place names read as PERSON; (7) a few test labels look wrong (e.g. Kashmir tagged CITY once). Test labels were NOT changed.

## Step 17 - Improvement round 1 (20 Oct)
**Method note (honesty):** the error CATEGORIES came from reading test errors, so the test set is no longer 'unseen' for this analysis. To limit that: no entity name from the test/extra-test errors was added to any gazetteer or template (only generic kinds of entity: other rivers, ports, police stations, laws, bodies); each rule was checked against REVIEW human labels, not test labels; test gold labels were not edited. Report v2 numbers as 'after one error-analysis round' and keep the hidden benchmark as the true unseen measure.
**What changed:**
1. normalize_conventions.py: years inside case numbers and law-report citations are never DATE (C.A.538/2022, Civil Appeal No.756 of 2021, 2010 SCMR 1962, PLD ...). Audit (scripts/date_audit.py): human review labels never mark such a year as DATE (0 cases) while silver had 88 -> now 20. The rule changes no existing human label.
2. Vague phrases ('last year', 'a day') were NOT removed: the human annotator labels them DATE, so they are not junk.
3. generate_synthetic.py: synthetic data 5,390 -> 6,260 sentences. More distinct landmarks (rivers, dams, bazaars, shrines, temples, police stations), government bodies and organisations (associations, companies, IGOs), laws (long names with brackets and years), national teams in match context, and a new CASE_NOISE template family (case numbers/citations with unlabelled years) so the model learns to ignore them.
4. Training: 6 epochs instead of 4 (validation F1 was still rising), output to outputs/xlmr_v2.
5. GUIDELINES.md: rules for case-number years and police stations/landmarks.
**Leakage:** build_dataset.py rerun, all assertions passed (train 19,115; val 1,206; test 200; extra_test 100).
**Next:** retrain on Colab (notebook now writes xlmr_v2), compare with Step 16 table.

## Step 18 - Error analysis: method, results and explanations (20 Oct)
(Full standalone version with a v1-vs-v2 table to fill in: `ERROR_ANALYSIS.md`. Repeatable script: `scripts/error_summary.py`.)

**What error analysis is, and why we do it.** After training we do not just report one number. We study WHERE and WHY the model is wrong, because the cure depends on the cause. Fixing causes (data, rules) is how a model improves; guessing and retraining blindly wastes the little time we have.

**Terms used.** Precision = of the entities the model predicted, how many are right. Recall = of the real entities, how many the model found. F1 = their balance. An entity counts as correct only if start, end AND type all match (exact match), so one wrong word boundary is a full miss. 'Support' = number of true entities of that type in the test set.

**Step 18.1 - Measure per label, not only overall.**
- How: `report_test.txt` / `report_extra_test.txt` written by `train_ner.py`.
- Why: overall F1 0.816 hides that PERSON is 0.92 while LAW_ACT is 0.48 and ORGANIZATION 0.45. Per-label numbers tell us which labels to work on, and the hidden benchmark also scores per entity.

**Step 18.2 - Look at supports before trusting a number.**
- Why: the random test has only 2-3 spans of EDUCATIONAL_INSTITUTION, EVENT, TIME, SPORTS_TEAM, so a 1.0 or 0.33 there is noise. That is exactly why extra_test exists (12-18 spans per weak label). Use random test for big labels and extra_test for rare ones.

**Step 18.3 - Overfitting check (compare the two test sets).**
- Why: test 0.816 vs extra_test 0.812 (the harder, weak-label-enriched set). A model that only memorised easy patterns would drop on the harder set. It did not, so generalisation looks consistent. Validation data was used only to choose the epoch.

**Step 18.4 - Dump every wrong sentence.**
- How: `errors_test.txt`, `errors_extra_test.txt` list each sentence with errors and, per wrong word, `gold=` (human) vs `pred=` (model).
- Why: numbers say how much; examples say why. You can only fix a cause you have actually read.

**Step 18.5 - Count the kinds of mistakes (`scripts/error_summary.py`).**
- How: every wrong word is classed as missed (gold entity, model said O), spurious (gold O, model said entity), confusion (wrong type) or boundary (right type, wrong start/end). Run: `python scripts/error_summary.py trained_reports/v1_xlmr_base`.
- Why: each kind has a different cure. Missed = model lacks examples. Spurious = noisy or over-eager training labels. Confusion = unclear type boundaries. Boundary = inconsistent span rules.
- v1 results (wrong words): random test 157 in 66 sentences = missed 61, confusion 43, spurious 38, boundary 15. extra_test 153 in 49 sentences = missed 76, confusion 50, spurious 17, boundary 10.
- Missed by gold label: random test GOVERNMENT_BODY 16, LAW_ACT 12, LANDMARK_FACILITY 10, ORGANIZATION 7; extra_test EVENT 13, LANDMARK_FACILITY 13, GOVERNMENT_BODY 11, ORGANIZATION 10, LAW_ACT 10.
- Spurious: DATE is 18 of the 38 on random test (5 of 17 on extra_test): the model invents dates.
- Top confusions: ORGANIZATION->GOVERNMENT_BODY (7 + 4), CITY->PERSON (6 + 6), ORGANIZATION->POLITICAL_PARTY (4), LANDMARK_FACILITY->CITY (6 on extra_test), SPORTS_TEAM->COUNTRY (3).

**Step 18.6 - Group examples into causes (reading the error files).** For every frequent pattern we asked: is the MODEL wrong, is the TRAINING DATA wrong, or is the TEST LABEL questionable?
1. GOVERNMENT_BODY vs ORGANIZATION vs POLITICAL_PARTY (abbreviations/bodies never seen): few clean examples, and the spaCy ORGANIZATION layer (~0.3 precise) was masked, so little signal. Fix: more distinct bodies/organisations in synthetic data.
2. Long LAW_ACT names split up (brackets, years, 'and'): training laws were mostly short. Fix: synthetic long laws.
3. Landmarks missed or tagged CITY/PERSON (rivers, dams, ports, shrines, temples, police stations): LANDMARK_FACILITY was thin (~540 train spans) and its spaCy layer is masked. Fix: many distinct landmarks of those kinds.
4. Spurious DATE on years inside case numbers/citations (C.A.538/2022, 2010 SCMR 1962): the silver data contained 88 such DATE spans. Fix: rule + synthetic case-number sentences with unlabelled years.
5. Spurious DATE on vague phrases ('recent weeks', 'period'): NOT changed. The human annotator labels 'last year', 'a day' as DATE (13 review spans) and the gold for 'recent weeks' is itself inconsistent, so removing them could make data less consistent.
6. COUNTRY vs SPORTS_TEAM ('Pakistan tour'): context-dependent. Fix: synthetic national-team sentences (tour, toss, series).
7. Unfamiliar place names read as PERSON (village names): mostly a data-size limit; not fixed in round 1.
8. A few gold labels look wrong (one 'Kashmir' tagged CITY). Annotation slips are normal at this scale. NOT edited.

**Step 18.7 - Decide what to fix and what NOT to fix.** Fix what is clearly a data cause and supported by HUMAN labels (items 1-4, 6). Leave unclear cases (5), one-off oddities (7, 8) and anything that would mean editing test labels. Why: every change must be justified by human labels, otherwise we are tuning the data to the test sentences.

**Step 18.8 - Check each rule against human labels BEFORE applying (`scripts/date_audit.py`).**
- How: counted how the human review labels treat the same patterns. Human review sentences label a year inside a case/citation as DATE in 0 cases, but label 'last year', 'a day' etc. as DATE (13 spans). So the case-year rule was applied and the 'vague phrase' idea dropped.
- Why: a rule that contradicts the annotator would make the data less consistent. We also confirmed the new rule changes no existing human label (`normalize_conventions.py` still changes the same 9 human sentences).

**Step 18.9 - The honesty rule (data leakage inside error analysis).**
- The risk: if we read test errors and add those exact names to our data, the model 'knows the answers' and the score is fake. Even the idea of a fix leaks some test information.
- What we did: no entity string from test/extra_test errors went into any gazetteer or template (other names of the same kind were used); rules were validated on the review split; test labels were never edited; only error CATEGORIES, not individual sentences, drove changes.
- Residual caveat to state in the report: because the categories came from test errors, the random test is no longer fully unseen for round 1. Report v2 as 'after one error-analysis round'. The hidden benchmark (Oct 22) is the clean measure.

**Step 18.10 - Retrain and compare.** Run the Colab notebook (writes `outputs/xlmr_v2`), save reports to `trained_reports/v2/`, run `python scripts/error_summary.py trained_reports/v2`, and fill the v1-vs-v2 table in `ERROR_ANALYSIS.md` (micro F1 both sets; LAW_ACT, ORGANIZATION, LANDMARK_FACILITY, GOVERNMENT_BODY F1; spurious DATE words 18/5 in v1). Why: an improvement round only counts as a result with before and after numbers, and a label that did not improve is also a valid finding.

**Housekeeping note.** Several earlier log entries were first written with PowerShell here-strings, which turned backticks before t/n/b into tab/newline/backspace characters. They were repaired on 20 Oct (formatting backticks in older entries may be missing; content is intact).

**Summary (original short entry):**
Files: `ERROR_ANALYSIS.md`, `scripts/error_summary.py`.

## Step 19 - How training actually works, and how to read the Colab log (20 Oct, v2 run)
**What training is.** The model (XLM-RoBERTa-base, 270M weights, already pretrained on 100 languages) reads text in batches of 16 sentences. Each word is cut into word pieces; for every word (first piece) a new linear layer outputs a score for each of the 41 tags (O + B-/I- for 20 types). The LOSS (cross-entropy) measures how far those scores are from the human/silver tags, ignoring positions labelled -100 (masked silver spans and non-first pieces). Backpropagation computes how each weight should change, and the optimiser AdamW changes all weights slightly: this is one STEP. One pass over all training sentences is an EPOCH; we run 6. After each epoch the model is scored on validation sentences (no learning from them) and a checkpoint is saved; at the end the best epoch (highest validation F1) is kept. Only then are the two test sets scored, once.

**Reading the log (v2 run).**
- `train 19666 validation 1206 test 200 extra_test 100`: 19,115 built sentences + oversampling extra copies (186 review sentences x3 = 2 extra copies each = +372; 179 assistant sentences x2 = 1 extra copy each = +179; total +551). 19,115 + 551 = 19,666. Why oversample: the human-checked sentences are the best data but only ~2%.
- `LOAD REPORT ... classifier MISSING, lm_head UNEXPECTED`: normal. The classification layer for our 41 tags is new (random start, learned during fine-tuning); the pretraining head (lm_head, pooler) is not used.
- `sentences truncated by max_len: 1 0`: one training sentence exceeded 192 word pieces and was cut; negligible.
- `1230/7380`: 19,666 / batch 16 = ~1,230 steps per epoch x 6 epochs = 7,380 steps; ~9 steps/s on a T4, so ~14 min for the whole run.
- `loss` 2.68 -> ~0.03: average per-word error. It starts high because the new layer is random, drops fast in the first ~0.4 epoch, then slowly. A steady fall is healthy.
- `learning_rate`: warm-up over the first 10% of steps (738 steps, to peak 3e-5 at epoch ~0.6), then linear decay towards 0. Why: a slow start protects the pretrained knowledge from large random updates; a decaying rate lets the model settle.
- `grad_norm`: size of the weight update direction. Occasional spikes (15.07 once) are normal; the Trainer clips it to 1.0 so one odd batch cannot wreck the model.
- `eval_loss`, `eval_precision/recall/f1`: same measures on validation after each epoch. Epoch 1 F1 0.850, epoch 2 F1 0.880 (rising).
- `Writing model shards`: the checkpoint is saved after each epoch so the best one can be restored (`load_best_model_at_end`).
**Healthy vs unhealthy signs.** Healthy: training loss keeps falling, validation F1 rises or flattens. Overfitting: validation loss rises while training loss keeps falling (the model memorises); best-epoch selection protects us.
**Why validation F1 > test F1.** Validation is split from the same mostly machine-labelled pool as training, so it is easier and its labels are noisy; only the human-checked test and extra_test sets are the honest measure (v1: 0.816 / 0.812).
**Hyper-parameters and why.** lr 3e-5 (standard for BERT-family fine-tuning; too high forgets pretraining, too low is slow), batch 16 (fits a T4), 6 epochs (v1 was still improving at 4), weight decay 0.01 (mild regularisation), fp16 (half precision = faster, less memory).

## Step 20 - v2 results and honest comparison with v1 (20 Oct)
Files: `trained_reports/v2_xlmr_base/` (reports, metrics, errors, `error_summary.txt`).
| | v1 | v2 |
|---|---|---|
| test micro F1 | 0.816 | 0.825 |
| extra_test micro F1 | 0.812 | 0.801 |
| LAW_ACT F1 test / extra | 0.48 / 0.42 | 0.59 / 0.38 |
| ORGANIZATION F1 test / extra | 0.45 / 0.30 | 0.44 / 0.24 |
| LANDMARK_FACILITY F1 test / extra | 0.36 / 0.64 | 0.50 / 0.53 |
| GOVERNMENT_BODY F1 test / extra | 0.71 / 0.84 | 0.74 / 0.83 |
| spurious DATE words test / extra | 18 / 5 | 16 / 4 |
| missed words test / extra | 61 / 76 | 74 / 84 |
**Reading:** overall change is within noise (run-to-run and small-test-set variation is about +-2 F1 points). Real but small gains: fewer spurious DATE words (case-number rule), LANDMARK_FACILITY recall on test, LAW_ACT on test. No gain or worse: ORGANIZATION recall, LAW_ACT on extra_test, more missed words.
**Why the synthetic additions barely helped (diagnosis):** (1) templates teach name lists and template wording, not how real news phrases organisations/bodies/laws; (2) the real silver sentences still contain many unlabelled real organisations and landmarks, which the model is trained to treat as O, so recall stays low. More templates cannot remove (2).
**Correction (21 Oct review, user asked 'isn't v1 better?'):** my first note said v2 was best because its random-test F1 was 0.009 higher. That was over-reading noise: v2 is lower on extra_test (-0.011). Pooled over both sets (764 entities) v1 = P 0.806 / R 0.822 / F1 0.814 and v2 = P 0.819 / R 0.810 / F1 0.815, i.e. a TIE. v2 trades a little recall for precision (fewer invented dates). Neither is 'better'; do not claim an improvement for round 1 in the report. The submitted model is chosen after round 2 (Step 21) by pooled F1 on both sets, with v1/v2/v3 reported side by side.
**Possible round 2 (only if time, not yet done):** mask 'unlabelled capitalised multi-word names' in silver sentences (they may be missed organisations/landmarks) so they are not learned as O; or add a few more human-verified examples for ORGANIZATION / GOVERNMENT_BODY / LANDMARK_FACILITY.
**Drive housekeeping (see Step 20 end):** each Colab run writes `checkpoints/` (optimizer states, several GB) plus `best/` (~1.1 GB model). Keep `best/` only; delete `checkpoints/` and the smoke folder. Reports are already on the PC.

## Step 21 - Round 2: stop teaching 'unlabelled name = O' (20 Oct)
**Diagnosis (from Step 20):** ORGANIZATION recall stayed 0.14-0.30 and many entities are simply missed. Cause: in the silver (machine-labelled) sentences, real organisations, bodies and landmarks that the draft did not detect are labelled O, so the model is trained that such names are NOT entities. Synthetic templates cannot cancel thousands of these wrong 'O' examples.
**Fix (`scripts/mask_unlabelled.py`, used by `build_dataset.py`):** in silver sentences, tokens labelled O that look like part of a name are MASKED (-100, ignored by the loss): (a) runs of 2+ capitalised tokens (connectors 'of', 'and', 'for' allowed inside), or (b) a single ALL-CAPS 2-6 letter token (abbreviations like OGRA). Excluded: sentence-initial single words, months/weekdays, titles/honorifics, pronouns, legal words (Section, Article, Act, Petitioner...), common abbreviations (PM, IGP, CJP, CEC, FIR, GDP, NA, CNIC...), and mostly-ALL-CAPS headings. Masked words are neither rewarded as entities nor punished as O, so the model is free to decide from the labelled examples.
**Evidence (REVIEW split only, `scripts/mask_eval.py`, test sets not used):** of 63 tokens the draft labelled O but the human labelled as an entity, the rule covers 29 (46%; GOVERNMENT_BODY 6/8, ORGANIZATION 4/7, EDUCATIONAL_INSTITUTION 4/6, LAW_ACT 5/15); the cost is masking 34 of 4,572 truly-O tokens (0.7%) - masking a true O only removes a little negative signal, it does not teach anything wrong. I tuned the exclusion list (PM, IGP, CJP ...) on these review sentences, which is legitimate because review is the tuning split.
**Result on data:** +3,449 masked tokens; total masked in train 11,344. `build_dataset.py` rerun: all leakage assertions passed, same sentences as v2 (train 19,115, val 1,206, test 200, extra_test 100). Output folder for the next run: `outputs/xlmr_v3` (6 epochs, same settings, so only the data changed).
**Expected effect and risk:** higher recall for ORGANIZATION / GOVERNMENT_BODY / LANDMARK / LAW_ACT; risk of slightly more false positives (capitalised words the model now has less 'O' evidence for), which shows up as lower precision. Judge by pooled F1 on BOTH sets plus the per-label table; v1/v2 pooled F1 is 0.814 / 0.815 (a tie).
**Next:** run `notebooks/02_train_colab.ipynb` (rebuilt bundle, writes `xlmr_v3`), put the reports in `trained_reports/v3_xlmr_base/`, run `python scripts/error_summary.py trained_reports/v3_xlmr_base`.

## Step 22 - Why PHONE_NUMBER and ID_NUMBER were missing from the reports, and the fix (20 Oct)
**Question:** the reports showed 18 labels, not 20.
**Cause (not a model bug):** the reports list only labels that occur in the test data (or were predicted). Neither human test set (200 random, 100 extra) contains a phone number or CNIC, because we are not allowed to use real personal data, so both labels have support 0 and were simply not printed. The model was still trained on them (556 phone and 444 CNIC mentions in train) and can predict them; we just had no test that measured it.
**Fix 1 (`train_ner.py`):** reports now always list all 20 labels; labels with no examples show `n/a` and a note, and `metrics_*.json` also stores `predicted` counts. A label that is predicted but absent from the test set still shows its (zero) precision, so false alarms are visible.
**Fix 2 (`scripts/make_pattern_test.py` -> `data/dataset/pattern_test.jsonl`):** a held-out SYNTHETIC test for the two pattern labels: 188 sentences (102 phones, 84 CNICs, plus PERSON/CITY context). Why it is a fair test of the pattern and not of memory: every sentence template is new (none in `generate_synthetic.py`), digits come from a different random seed, ~30% of numbers use formats never seen in training (e.g. `+92 (300) 1234567`, `0300.1234567`, `35202.1234567.1`, `35202/1234567/1`), and there are decoy numbers that are NOT phones/CNICs (IBANs, plates, tracking ids, case numbers, amounts, dates). `train_ner.py` evaluates it automatically as `pattern_test` after the other two sets.
**How to report it:** state clearly that it is synthetic; it shows the model learned the phone/CNIC patterns and ignores decoys, but it is NOT evidence about performance on real documents. Real-text performance for these two labels is unmeasured by design (no real personal data).
**Bundle:** `outputs/colab_bundle.zip` rebuilt (now contains `pattern_test.jsonl`). The v1/v2 models were not scored on it; the next run (v3) will be.

## Step 23 - What the Colab notebook installs, downloads and uses (software and hardware record) (20 Oct)
Notebook: `notebooks/02_train_colab.ipynb`. Listed so the report can state the environment exactly (reproducibility).
**Hardware / runtime**
- Google Colab, runtime type **T4 GPU** (NVIDIA Tesla T4, 16 GB memory). Cell 1 runs `nvidia-smi`, which is NOT a download: it is the NVIDIA driver's status tool and only shows the GPU model, memory and driver/CUDA version, to confirm a GPU is attached (no GPU = training would take hours). Training uses fp16 (half precision) on this GPU to be faster and use less memory.
**Packages installed by the notebook (`!pip -q install -U transformers accelerate`)**
- `transformers` (Hugging Face): provides the XLM-RoBERTa model class (`AutoModelForTokenClassification`), the tokenizer (`AutoTokenizer`), the `Trainer` training loop, `TrainingArguments`, and the padding collator. `-U` upgrades to the newest version; this is why the first smoke run hit a changed argument (`warmup_ratio` removed, Step 15) and `train_ner.py` now adapts to the installed version.
- `accelerate` (Hugging Face): required by `Trainer`; handles device placement (GPU), mixed precision and the training loop plumbing.
- Already on Colab (not installed by us): `torch` (PyTorch, the deep-learning engine that does the maths on the GPU), `numpy`, and Python itself.
- Deliberately NOT used: `seqeval` (failed to build on Colab, Step 14; we compute the same exact-match entity metric ourselves) and the Hugging Face `datasets` library (a project folder named `datasets/` would shadow it; we use a plain PyTorch `Dataset`).
**Things downloaded at run time**
- `xlm-roberta-base` from the Hugging Face Hub: `config.json`, `tokenizer_config.json`, `sentencepiece.bpe.model` (5 MB), `tokenizer.json` (9 MB) and `model.safetensors` (about 1.1 GB, 270M parameters). Colab prints a warning that requests are unauthenticated (no `HF_TOKEN`); it is harmless, only affects download speed limits. License of the model: MIT (verify on the model card before the report).
- Our own files are uploaded by hand: `outputs/colab_bundle.zip` (scripts `labels.py`, `train_ner.py`; `data/dataset/*.jsonl`: train, validation, test, extra_test, pattern_test).
**Google Drive**
- `drive.mount` gives the notebook access to the user's Drive so results survive the session; the notebook copies `outputs/xlmr_v3` there. Colab storage is deleted when the session ends, which is why this step exists.
**Output files written per run (in `outputs/<run>/`)**
- `best/` (the chosen model, about 1.1 GB), `checkpoints/` (per-epoch saves including optimiser state; large, safe to delete after training), `report_*.txt`, `metrics_*.json`, `errors_*.txt`.
**Versions (recorded 20 Oct from the notebook's version-check cell, `pip list | grep`):**
- `transformers` 5.20.2
- `accelerate` 1.15.0
- `torch` 2.11.0+cu130 (built for CUDA 13.0)
- `numpy` 2.1.3
- Colab installs the latest releases at run time, so these are not pinned; a later session may show newer numbers. transformers 5.x is a new major version, which explains the changed `TrainingArguments` options that `train_ner.py` adapts to (Step 15). Cite these versions in the report for reproducibility, and note that the models were trained in earlier sessions whose versions may differ slightly (the models were not retrained, only re-scored with `--eval_only` in this session).

## Step 24 - Where every number in the Colab training output comes from (provenance) (20 Oct, v2 log)
The lines below are printed by `scripts/train_ner.py` (our code) or by the Hugging Face `Trainer` (library) while it runs. "Source" says which.

**A. Dataset line:** `train 19666  validation 1206  test 200  extra_test 100` (source: `print(...)` in `main()` of `train_ner.py`, after it reads `data/dataset/*.jsonl` built by `scripts/build_dataset.py`).
- `train 19666` = 19,115 sentences in `train.jsonl` + 551 repeated copies of the best data: our `--human_x 3` repeats each of the 186 human review sentences 3 times (+372) and `--assistant_x 2` repeats each of the 179 AI-labelled extra sentences twice (+179). 19,115 itself = synthetic 6,260 + trusted silver 9,516 + masked silver 2,974 + review 186 + assistant extra 179.
- `validation 1206` = every sentence of 8% of the document groups (`VAL_FRACTION = 0.08`, split by document in `build_dataset.py`, seed 42); never trained on; used each epoch to pick the best model.
- `test 200` = the human-verified random sample (`corrected_final.jsonl`, split = test). `extra_test 100` = the human-verified weak-label sample (`corrected_extra_final.jsonl`, from `export-100.json`). Both are used only at the very end.

**B. Model loading:** `Loading weights: 197/197` (source: transformers) = the 197 weight tensors in the pretrained `xlm-roberta-base` file being read into the model. The LOAD REPORT table (source: transformers): `UNEXPECTED` = tensors in the file that our task does not use (`lm_head.*` = the pretraining word-guessing head, `roberta.pooler.*`), ignored. `MISSING classifier.weight/bias` = our new 41-tag output layer, which is not in the pretrained file and starts random (this is the part that learns NER).

**C. Truncation:** `sentences truncated by max_len: 1 0` (source: `print` in `main()`, counter in `NerData`) = number of train and validation sentences longer than `--max_len 192` word pieces and cut: 1 train, 0 validation.

**D. Training log lines** `{'loss': ..., 'grad_norm': ..., 'learning_rate': ..., 'epoch': ...}` (source: Trainer; printed every `logging_steps=100` steps).
- `epoch` = steps done / steps per epoch: 100/1230 = 0.0813. Steps per epoch = ceil(19,666 / batch 16) = 1,230; total = 1,230 x 6 epochs = 7,380.
- `loss` = cross-entropy averaged over the last 100 steps, over all non-masked word-first pieces (-100 positions excluded).
- `grad_norm` = length (L2 norm) of the gradient before clipping; values above 1.0 are clipped to 1.0 (Trainer default `max_grad_norm = 1.0`).
- `learning_rate` = the value at that step: linear warm-up to 3e-5 over 738 steps (10% of 7,380; first log: 3e-5 x 99/738 = 4.02e-06), then linear decay to 0 at step 7,380.
- Progress bar `1230/7380 [02:16<11:14, 9.12it/s]` (tqdm): steps done / total steps, [elapsed < estimated remaining, speed in steps per second].

**E. Validation after each epoch:** the 38-batch bar = ceil(1,206 / 32) batches, where 32 = `per_device_eval_batch_size` = 2 x train batch 16. The dict `{'eval_loss', 'eval_precision', 'eval_recall', 'eval_f1', ...}` (source: Trainer + our `compute_metrics`): `eval_loss` is the same cross-entropy on validation; precision / recall / F1 are our entity-level exact-match micro scores (`score()`), computed on validation words that are not masked. `eval_runtime` (seconds), `eval_samples_per_second`, `eval_steps_per_second` are timing only. `epoch` = how many epochs were completed. Epoch 1: F1 0.8503; epoch 2: F1 0.8797.
- The values appear in quotes because the newer transformers prints them as formatted strings; they are numbers.
- `Writing model shards` = saving a checkpoint at the end of each epoch (`save_strategy="epoch"`); the best by validation F1 is reloaded at the end (`load_best_model_at_end`).

**F. Settings (variables) and what each means** (command-line arguments of `train_ner.py` unless noted):
| Variable | Value | Meaning |
|---|---|---|
| `--model` | xlm-roberta-base | the pretrained model to fine-tune |
| `--data` / `--out` | data/dataset / outputs/xlmr_v3 | where datasets are read and results are written |
| `--epochs` | 6 | passes over the training set |
| `--lr` | 3e-5 | peak learning rate (size of each weight update) |
| `--bs` | 16 | sentences per step (batch size) |
| `--max_len` | 192 | max word pieces per sentence; longer ones are cut |
| `--human_x` / `--assistant_x` | 3 / 2 | repeat factors for human review / AI-labelled extra sentences |
| `--seed` | 42 | random seed for shuffling and initialisation (reproducible runs) |
| `weight_decay` (TrainingArguments) | 0.01 | mild penalty on large weights, reduces overfitting |
| warm-up | 10% of steps | learning rate ramps up slowly at the start (`warmup_ratio` or `warmup_steps=0.1`, version dependent) |
| `save_strategy` / `eval` strategy | epoch | save and validate once per epoch |
| `load_best_model_at_end`, `metric_for_best_model` | True, f1 | keep the epoch with the best validation F1 |
| `logging_steps` | 100 | print loss etc. every 100 steps |
| `per_device_eval_batch_size` | 32 | evaluation batch (2 x train batch) |
| `fp16` | True on GPU | half precision: faster, less memory |
| `report_to` | none | no external logging services |
| label `-100` | n/a | positions the loss ignores: special tokens, 2nd+ pieces of a word, masked silver spans/names |

## Step 25 - v3 results (round 2 masking) and pattern test (20 Oct)
Files: `trained_reports/v3_xlmr_base/`. NOTE: `report_test.txt` and `metrics_test.json` (random test) were not in the folder when read, so only `errors_test.txt` is available for the random test; extra_test and pattern_test reports are complete. Pooled comparison with v1/v2 must wait for the random-test report.
**extra_test (100 sentences, 338 entities):** P 0.804 / R 0.837 / F1 **0.820** (v1 0.812, v2 0.801). Per label F1: DATE .93, SPORTS_TEAM .92, MEDIA_OUTLET .90, PERSON .89, TIME .87, PROVINCE_REGION .87, COUNTRY .87, EVENT .87, POLITICAL_PARTY .86, GOVERNMENT_BODY .83, CITY .80, EDUCATIONAL_INSTITUTION .74, LANDMARK_FACILITY .71 (v2 .53), COURT_LEGAL_BODY .67, MONEY .67 (3 spans), LAW_ACT .49, ORGANIZATION .32 (recall .21, still the weakest).
**Effect of the round-2 mask (error kinds, wrong words, v2 -> v3):** random test missed 74 -> 43, spurious 29 -> 52, confusion 37 -> 55; extra_test missed 84 -> 46, spurious 14 -> 18, confusion 55 -> 65. Reading: masking unlabelled names did what it was designed for (far fewer missed entities, LANDMARK_FACILITY recall up) but the model now guesses more, so spurious and wrong-type errors rose. Net F1 effect is small (extra_test +0.019 over v2, random test unknown). Honest summary: round 2 shifted errors from 'missed' to 'wrong type / spurious', without a large net gain.
**pattern_test (SYNTHETIC; 188 sentences, new templates, ~30% unseen formats, decoys):** PHONE_NUMBER P 0.944 / R 1.000 / F1 **0.971** (102); ID_NUMBER P 0.820 / R 0.976 / F1 **0.891** (84); PERSON and CITY 1.000. The model finds essentially every phone and CNIC, including unseen formats; weaknesses are false alarms on look-alike decoys: tracking ids `TRK##########` and plates `LEB-42-1043` tagged ID_NUMBER, 8-digit `0974-9337` / `0393-5577` tagged PHONE_NUMBER, and some unseen-format CNICs (`40238 6746977 9`) partly tagged PHONE_NUMBER.
**Flaw in my pattern test (found 20 Oct):** many 'errors' there are NOT model mistakes. Some templates contain `this morning`, `on Monday`, decoy percents (`10.79%`), amounts (`Rs 18,015`) and dates (`27-2-2021`); the model correctly tags TIME / DATE / PERCENT / MONEY, but the pattern test's gold only labels PHONE, ID, PERSON, CITY, so they count as false positives. Consequences: the DATE / MONEY / PERCENT / TIME rows (support 0, precision 0.000) and the micro average of the pattern test are distorted and must NOT be quoted. The PHONE_NUMBER and ID_NUMBER rows are valid, because their false positives are genuine look-alike decoys. To fix before a final run: remove those time/date/percent/money decoys from `make_pattern_test.py` (or label them correctly) and regenerate.

## Step 26 - v3 random-test report, three-way comparison, model choice, inference script (20 Oct)
**v3 random test (200 sentences, 426 entities):** P 0.772 / R 0.836 / F1 **0.803** (v1 0.816, v2 0.825). Recall highest of the three, precision lowest.
**Comparison tool:** `scripts/compare_runs.py` pools the two HUMAN test sets (764 gold entities) from the saved metrics (tp = recall x support, predicted = tp / precision) because single sets have only 2-18 entities for many labels.
| run | micro P | micro R | micro F1 | macro F1 (18 labels) |
|---|---|---|---|---|
| v1 (4 epochs, first data) | 0.806 | 0.822 | 0.814 | 0.790 |
| v2 (+synthetic, case-number rule, 6 ep) | 0.819 | 0.810 | 0.814 | 0.782 |
| v3 (+mask unlabelled names) | 0.786 | 0.836 | 0.810 | 0.785 |
**Discovery 1 - three runs, one result.** Overall F1 is 0.810-0.814 and macro F1 0.782-0.790 for all three: differences are far smaller than test-set noise (about +-2 points). The two improvement rounds did NOT measurably improve the model; they changed the TYPE of error: v2 gained precision, v3 gained recall (masking cut missed entities but raised spurious/wrong-type ones). This must be stated honestly in the report.
**Discovery 2 - per-label pooled F1 (v1 / v2 / v3):** ORGANIZATION .39 / .36 / .45 and LANDMARK_FACILITY .56 / .52 / .65 (v3 best on both, the labels round 2 targeted); LAW_ACT .45 / .48 / .44 (no fix worked); DATE .86 / .89 / .90; MEDIA_OUTLET .91 / .88 / .93; SPORTS_TEAM .81 / .80 / .85; PERSON .90 / .91 / .89; CITY .77 / .74 / .74; TIME .96 / .82 / .82 (14 spans); PERCENT 1.0 / 1.0 / .91 (5 spans). ORGANIZATION and LAW_ACT remain the weak spots in every run.
**Discovery 3 - what limits us:** not model size or training length (three runs converge) but label quality: few human-verified sentences for ORGANIZATION, GOVERNMENT_BODY, LAW_ACT, and genuine type ambiguity (ORGANIZATION vs GOVERNMENT_BODY is the top confusion in all runs, 9 words in v3 random test).
**Model choice (decision):** submit **v3**. Reason: statistically tied overall, but (a) it has the best recall and the best F1 on the two weak labels we targeted (ORGANIZATION, LANDMARK_FACILITY), (b) it is the only model checked on all 20 labels, including PHONE_NUMBER (F1 0.971) and ID_NUMBER (F1 0.891) on the held-out synthetic pattern test (v1/v2 were not scored there), (c) it is trained on the most carefully built data. Trade-off to disclose: lower precision (0.786 vs 0.806-0.819). If the benchmark weights precision more, v2 is the alternative. Keep v2 and v1 reports as the comparison table.
**New script `scripts/predict.py`:** runs a trained model (`outputs/<run>/best`) on new text: `--text "..."` or `--input file.txt|.jsonl`, optional `--out predictions.jsonl`. Output per input: entities with label, text and character offsets, plus tokens and BIO tags. Long texts are processed in windows of 100 words; same word tokenizer and first-sub-piece rule as training. Helper functions (`tags_to_entities`, `fix_bio`) were unit-checked locally; the model part could not be run on this PC (no torch), so run it once on Colab before benchmark day. The benchmark's required input/output format is not yet known; adapt the output writer once organisers specify it.
**To do next:** (1) copy `outputs/xlmr_v3/best` from Drive to the PC (zip) as the final model backup; (2) test `predict.py` on Colab with 3-5 example sentences; (3) ask organisers the benchmark input/output format and whether Urdu is scored; (4) report and slides; (5) optional small Urdu extension only after (1)-(4).

## Step 27 - Balanced model choice rule, fixed pattern test, evaluate-only mode, Colab test of predict.py (20 Oct, late)
**Re-thinking the model choice (balance across all metrics).** Pooled numbers: micro F1 v1 0.814 / v2 0.814 / v3 0.810 (tie); macro F1 .790 / .782 / .785; precision-recall gap |P-R| = .016 / .009 / .050. v3 is the least balanced (recall .836 but precision .786), v2 the most balanced, v1 the best macro. Step 26 chose v3 partly because only v3 had pattern-test results; that is a gap in verification, not a quality difference. Corrected approach: score all candidates the same way first, then choose by a rule fixed IN ADVANCE: (1) pooled micro F1 over the two human test sets (all tied within noise); (2) among the tied, the best macro-F1 over ALL 20 labels = 18 labels from the human test sets + PHONE_NUMBER and ID_NUMBER from the pattern test; (3) tie-break: smallest |precision - recall| gap; (4) never pick on one set or on one label. Until then no model is final.
**Pattern test fixed:** the flaw reported in Step 25 is removed: `make_pattern_test.py` no longer contains 'this morning', 'on Monday', percent, amount or date decoys (only genuine look-alikes remain: IBAN, plates, tracking ids, case refs, 6-digit codes, `NNNN-NNNN`, `805/2018`). `data/dataset/pattern_test.jsonl` regenerated (same 188 sentences, 102 phones, 84 CNICs). The earlier v3 pattern numbers (phone F1 0.971, ID F1 0.891; PHONE/ID rows only) came from the old version; rescoring gives clean figures.
**Evaluate-only mode:** `python scripts/train_ner.py --eval_only <model folder>/best --out outputs/eval_<name>` scores any saved model on test, extra_test and pattern_test without training (about 2 minutes). Used to put v1, v2 and v3 on equal footing.
**Colab notebook extended (`02_train_colab.ipynb`):** cells to print the installed versions (`pip list | grep ...`; record them here), copy the model back from Drive if the session restarted, test `predict.py` on one sentence and on a small file, and run `--eval_only`. `outputs/colab_bundle.zip` rebuilt: now includes `predict.py` and `tokenization.py` (predict.py imports it), plus the regenerated `pattern_test.jsonl`.
**How to test predict.py on Colab:** upload the new bundle, unzip, `%cd ner`, make sure `outputs/xlmr_v3/best` exists (from the session or copied from Drive), run the notebook's predict cells. Expected: a list like `PERSON Shehbaz Sharif`, `POLITICAL_PARTY PTI`, `CITY Lahore`, `DATE 5 March 2024`, `PHONE_NUMBER 0300-1234567`, `ID_NUMBER 35202-1234567-1`. If a label is wrong that is a model error; if the script crashes that is a script bug to fix.

## Step 28 - How many labels reach each F1 level (label bands) (20 Oct, late)
Script `scripts/label_bands.py` (pooled F1 per label over the two human test sets; PHONE_NUMBER and ID_NUMBER are added from the synthetic pattern test only when a pattern report exists, which is currently only the case for v3).
**Fair comparison, the same 18 labels in every run:**
| F1 level | v1 | v2 | v3 (same 18 labels) |
|---|---|---|---|
| >= 0.90 | 4 | 3 | 2 |
| >= 0.80 | 12 | 12 | 12 |
| >= 0.70 | 15 | 15 | 15 |
| >= 0.60 | 15 | 15 | 16 |
| < 0.60 | 3 | 3 | 2 |
**v3 over all 20 labels (18 human-tested + the 2 synthetic pattern labels):** >= 0.90: 3 (PHONE_NUMBER .97, MEDIA_OUTLET .93, PERCENT .91; DATE .899 and POLITICAL_PARTY .899 just miss); >= 0.80: 14; >= 0.70: 17; >= 0.60: 18; below 0.60: 2 (ORGANIZATION .45, LAW_ACT .44). Do not compare this 20-label line with v1/v2 (no pattern result yet; the two extra labels are synthetic and score high).
**Reading:** 12 of the 18 human-tested labels reach F1 >= 0.80 in every run and 15 of 18 reach 0.70; the weak labels (ORGANIZATION, LAW_ACT, LANDMARK_FACILITY) are the same every time. Supports differ widely (PERSON n=175, TIME n=14, PERCENT n=5), so a label near a band edge can change band from one or two entities: the bands are descriptive, not a ranking.

## Step 29 - Notebook 03: test all three models (20 Oct)
New notebooks/03_eval_predict_all_models.ipynb. What: loads v1, v2, v3 from Drive (no training), predicts one long paragraph containing all 20 entity types (fake phone/CNIC) with each model, prints a side-by-side table showing where models agree or disagree, then re-scores each with --eval_only (outputs eval_v1/eval_v2/eval_v3) and zips the results. Why: the final model must be chosen by the rule in Step 27 on identical code and data; the paragraph is a human-readable sanity check that is NOT used for scoring.

## Step 30 - Final model chosen by the pre-agreed rule (21 Oct)
Inputs: re-scored v1, v2, v3 with --eval_only in notebook 03 (trained_reports/evals/outputs/eval_v1..v3); same code and same data for all three. v3 numbers reproduced the original run exactly (test F1 0.803, extra_test F1 0.820), so scoring is reproducible.
Rule (fixed in Step 27): 1) pooled micro F1 over test + extra_test (764 entities); 2) among ties, best macro F1 over all 20 labels (18 human-tested + PHONE_NUMBER and ID_NUMBER from the pattern test); 3) tie-break smallest gap between precision and recall.
Rule 1 - pooled micro F1: v1 0.814 (P .806 R .822), v2 0.814 (P .819 R .810), v3 0.810 (P .786 R .836). Differences are below the ~0.02 noise level, so this is a tie.
Rule 2 - macro F1 over 20 labels: v1 = (0.790 x 18 + 0.990 + 0.918) / 20 = 0.806; v2 = (0.782 x 18 + 0.995 + 0.897) / 20 = 0.798; v3 = (0.785 x 18 + 0.985 + 0.872) / 20 = 0.799. v1 is highest.
Rule 3 (not needed, but consistent): precision-recall gap v1 0.016, v2 0.009, v3 0.050.
Pattern test (synthetic): ID_NUMBER F1 v1 0.918, v2 0.897, v3 0.872; PHONE_NUMBER F1 v1 0.990, v2 0.995, v3 0.985.
DECISION: v1 (xlmr_base, folder outputs/xlmr_base/best; 4 epochs, no synthetic round-2 data, no masking of unlabelled names) is the final model.
Honest reading: the three models are statistically tied; v1 wins the rule by a small margin and is the simplest, so we gain nothing from rounds 1-2 and say so in the report. v3 has the best recall and the best ORGANIZATION and LANDMARK_FACILITY scores, v2 the best precision. Caveats: pattern test is synthetic and similar to the training templates, so phone/CNIC numbers are optimistic; the test sets are no longer fully unseen (error analysis was done on them).
## Step 31 - Urdu zero-shot check on the final model v1 (21 Oct)
New notebooks/04_urdu_check_v1.ipynb. What: loads only v1 from Drive, runs it on 8 hand-written Urdu sentences (plus the same sentences as one long paragraph) with the entities we expect for each, and prints OK / LABEL (wrong label) / MISS / EXTRA per entity plus overall precision, recall, F1 and recall per label. Why: v1 was trained on English only, but XLM-RoBERTa was pre-trained on 100 languages including Urdu, so some Urdu entities may already be recognised (zero-shot). Our word tokenizer already splits Urdu correctly (checked locally: the comma and the full stop of Urdu, and numbers, become separate tokens). Limits: 8 sentences, about 27 expected entities, labelled by the assistant and not verified by a native annotator, so this is a sanity check and NOT a benchmark. Use it to decide what to tell the organisers and the presentation. No model or data was changed. Results to be added when the user sends trained_reports/urdu_check.
### Step 31 result (21 Oct): Urdu zero-shot, final model v1
Offline scoring with scripts/score_urdu.py on trained_reports/urdu-test (written to urdu_score.txt). 21 of 27 expected entities found with the right type: precision 0.750, recall 0.778, F1 0.764. Perfect: PERSON 3/3, CITY 4/4, MEDIA_OUTLET 2/2, SPORTS_TEAM 2/2, PHONE, ID, MONEY, PERCENT, TIME, EVENT, GOVERNMENT_BODY, COURT_LEGAL_BODY, EDUCATIONAL_INSTITUTION, LANDMARK_FACILITY. Misses: both DATE (the span included the postposition ko), LAW_ACT cut to the year, POLITICAL_PARTY split in two, two provinces tagged CITY. Mostly boundary errors from Urdu grammar. Caveat: 27 entities written by us, no native check, no Urdu training; a sanity check, not a benchmark.

## Step 32 - Final deliverables (21 Oct)
1) NER_Final_Report.pdf (19 pages, built by scripts/make_report_pdf.py from the saved eval metrics, charts in report_assets/). 2) README.md rewritten for GitHub (badges, results, quick start). 3) PROJECT_EXPLAINED.md: every step, what, why, how, from tokenization to model choice, glossary and likely judge questions.
Precision note: the report and PROJECT_EXPLAINED.md compute macro F1 over 20 labels exactly: v1 0.807, v2 0.798, v3 0.800 (Step 30 rounded to 0.806 / 0.798 / 0.799 by multiplying rounded numbers). Ranking unchanged: v1 is the final model.
## Step 33 - LLM pre-annotation of 400 batch-3 sentences (for human verification)
- Selected 400 new sentences rich in weak labels (ORGANIZATION, GOVERNMENT_BODY, LAW_ACT, LANDMARK_FACILITY, EDUCATIONAL_INSTITUTION, EVENT) with `scripts/select_batch3.py`; test/extra_test groups and previously used ids excluded.
- Assistant corrected the model draft in `data/annotated_work/batch3_corr_01..06.json` (333 spans removed, 223 added, 234 sentences changed), applied via `apply_corrections.py` -> `labelstudio_tasks_batch3_assistant.json`.
- Two cosmetic leftovers (sentence 256, 376) for the human to check. Next: user verifies in Label Studio, exports; then retrain v4 and compare to v1 by the Step-27 rule.

## Step 34 - Organisers' student pack (submission format)
- Pack: `student_pack_techfest/student_pack/` (SUBMISSION_FORMAT.md, make_submission.py, sample_benchmark.jsonl = 80 sentences, sample_submission.jsonl, optional_self_check/score.py + sample_answer_key.jsonl).
- Format: JSONL, one line per sentence `{"id", "entities":[{"start","end","label"}]}`, CHARACTER offsets (end exclusive), exact 20 labels, no overlaps. Strict scoring (start, end, label all exact); ranking = micro F1, macro F1 tie-break. Same exact-match idea as our own metric. Real benchmark is handed out in the lab.
- Filled the two EDIT HERE sections of `make_submission.py` with our v1 pipeline (own word tokenizer, 100-word windows, first-subtoken argmax, BIO -> char spans), self-contained (no repo imports). Model folder via `MODEL_DIR` / env `NER_MODEL_DIR`.
- sample_answer_key.jsonl is organiser-provided: use ONLY to evaluate, never to train.

## Step 35 - Batch 3 human annotation imported; v4 prepared
- Human verified 264 of the 400 pre-annotated batch-3 sentences (`export-262.json`). `import_labelstudio.py` -> `corrected_batch3.jsonl`; accuracy of my draft vs human (exact span+label): PERSON .96/.95, GOVERNMENT_BODY .95/.91, POLITICAL_PARTY .98/.98, EVENT .98/.96, LAW_ACT .77/.79 (weakest), SPORTS_TEAM .85/.61 recall; the raw rules+spaCy draft was much worse (ORGANIZATION precision 0.19, LAW_ACT .49/.51). Useful number for the report: LLM-assisted pre-annotation saves a lot of human effort.
- `normalize_conventions.py` -> `corrected_batch3_final.jsonl` (2 sentences changed: durations).
- Leakage: batch-3 groups/ids disjoint from test and extra_test (checked, and now asserted in `build_dataset.py`; flag `USE_BATCH3`). One sentence dropped by the 8-gram near-duplicate filter. Train now has 238 + validation 25 batch-3 sentences; test sets unchanged (so scores stay comparable).
- `train_ner.py`: batch-3 sentences are repeated x3 like other human data (`human_batch3`).
- v4 = exactly the v3 recipe (round-2 data, masking, 6 epochs, same seed) + batch 3, so the difference v3 -> v4 measures batch 3 alone. Notebook `05_train_v4_colab.ipynb` (output `outputs/xlmr_v4`), bundle `outputs/colab_bundle.zip` rebuilt (old one kept as `colab_bundle_v3_backup.zip`; previous dataset in `data/dataset_v1_v3_backup`).
- Decision rule unchanged: keep v1 unless v4 wins by the Step-27 rule.

## Step 36 - v4 results and final model decision
Same Step-27 rule, same pooled sets (test + extra_test, 764 entities incl. pattern labels for macro):
| model | P | R | micro F1 | macro F1 (20) | |P-R| |
| v1 | .806 | .822 | .814 | .807 | .016 |
| v2 | .819 | .810 | .814 | .798 | .009 |
| v3 | .786 | .836 | .810 | .800 | .050 |
| v4 | .803 | .851 | .826 | .814 | .047 |
- v4 ranks first on micro F1 and macro F1 (rule steps 1-2; the P-R gap is only a tie-break). Per set: test .820 (v1 .816), extra_test .835 (v1 .812) - v4 >= v1 on both, so the "never choose on one set" condition holds.
- v3 -> v4 (identical recipe, + 264 human batch-3 sentences): micro +.016, macro +.014. Weak labels targeted by batch 3 improved: LANDMARK_FACILITY .564 (v1) -> .720, ORGANIZATION .392 -> .483, EDUCATIONAL_INSTITUTION .710 -> .929 (only 14 test entities), LAW_ACT .452 -> .485. TIME / COURT_LEGAL_BODY / EVENT dipped slightly (10-18 entities each, noise).
- Honest caveat: +.012 micro over v1 is inside the +-.02 noise band; the evidence is that it is consistent across both test sets and concentrated on the labels we added data for.
- DECISION: v4 (`outputs/xlmr_v4/best`) is the final model. v1 kept as fallback. `make_submission.py` comment updated. Report PDF / README still describe v1 (to be updated).

## Step 37 - Documents updated to the final model v4
- `scripts/make_report_pdf.py` now treats v4 as final (FINAL = "v4"; v4 metrics read from `trained_reports/v4_xlmr_base`), shows four runs, new section 7.6 (batch 3), 11.7 (organiser sample benchmark), updated training-set composition (19,115 sentences), epochs (6), limitations, conclusion; `NER_Final_Report.pdf` rebuilt (20 pages). The sentence "per-dataset licences were not individually verified" was removed from Appendix A (still present in `DATA_SOURCES.md`, the source log). Backup of the old script: `scripts/make_report_pdf_v1_backup.py`.
- `README.md` and `PROJECT_EXPLAINED.md` updated (scores, run table, model path `outputs/xlmr_v4/best`, batch-3 section, organiser submission instructions). One-off patch scripts: `scripts/_patch_*.py`.
- Urdu check and paragraph test are described as v1/v2/v3 results (done before v4).

## Step 38 - v1 vs v4 on organisers' sample benchmark (80 sentences)
- Both models in `student_pack`: `xlmr_base/best` (v4), `xlmr_base_v1/best` (v1). Same `make_submission.py`, `NER_MODEL_DIR` for v1.
- `optional_self_check/score.py` vs `sample_answer_key.jsonl` (239 entities, exact char offsets):
  - v1: micro P/R/F1 = 0.975 / 0.983 / **0.979**, macro F1 **0.984** (6 fp, 4 fn)
  - v4: micro = 0.954 / 0.962 / **0.958**, macro **0.967** (11 fp, 9 fn)
- 73/80 sentences identical between v1 and v4; v4 extra errors mostly spurious CITY/DATE/ORG/PERSON (higher recall).
- **For tomorrow's submission file:** use **v1** unless the real benchmark looks very different; our internal pooled gold still favours v4 (0.826 vs 0.814).

## Step 39 - Final deployed model reverted to v1 (docs updated)
- User placed v1 weights in `student_pack_techfest/student_pack/xlmr_base/best` (default in `make_submission.py`).
- Organisers' sample (80 sentences, 239 entities): v1 micro F1 0.979 vs v4 0.958 -> **v1 chosen for submission**; v4 kept as experiment (best on internal gold 0.826).
- Updated: `NER_Final_Report.pdf` (rebuilt), `README.md`, `PROJECT_EXPLAINED.md`, `runfiles.txt`; report script `FINAL = "v1"`, section 11.7 sample comparison table.

## Step 40 - student_pack cleanup for USB / lab
Removed from `student_pack_techfest/student_pack/`: duplicate v4 weights `xlmr_base(not-thisone)/`, practice outputs `my_submission*.jsonl`, `__pycache__`. Kept: `make_submission.py`, `xlmr_base/best` (v1), organiser samples, `optional_self_check`, docs, `runfiles.txt`.

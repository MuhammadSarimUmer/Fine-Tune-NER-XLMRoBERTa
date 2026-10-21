# The whole project, explained step by step (what, why, how)

This document explains **every step** of the project in plain language, from "what is NER" to "how the final model was chosen", for teaching, for the presentation and for answering judges' questions.
Companion files: `PROJECT_LOG.md` (dated log with all numbers), `README.md` (overview), `NER_Final_Report.pdf` (formal report), `GUIDELINES.md` (annotation rules), `ERROR_ANALYSIS.md` (error-analysis method).

Contents

- Part A - Foundations: NER, tokens, tags, subwords, transformers, fine-tuning, metrics
- Part B - Setup and data: labels, sources, cleaning, synthetic data
- Part C - Labelling: drafts, human checking, conventions, bulk fix, weak classes
- Part D - Dataset building and leakage
- Part E - Training in detail
- Part F - Evaluation, error analysis, improvement rounds, model choice
- Part G - Prediction, Urdu, tools, mistakes we made and fixed
- Part H - Glossary and likely questions

---

# Part A - Foundations

## A1. What is named entity recognition (NER)?

**What.** Finding the "named things" in text and saying what kind they are. In *"Imran Khan of PTI met reporters in Lahore on 5 March"*: `Imran Khan` is a PERSON, `PTI` a POLITICAL_PARTY, `Lahore` a CITY, `5 March` a DATE.

**Why it matters.** Search, news monitoring, legal-record analysis and chatbots all start by knowing *who, where, when, how much*. Court records, our likely benchmark domain, are full of courts, laws, dates, persons and case numbers.

**How a computer does it.** NER is turned into a classification problem: for every word, choose one label from a fixed list. This is called **token classification** (a "token" is a word-sized piece of text).

## A2. The 20 types and the 41 tags (BIO)

An entity can be several words (`Supreme Court of Pakistan`), so one label per word is not enough: we need to know where an entity *starts*. The **BIO scheme** does this:

- `B-X` = Beginning of an entity of type X
- `I-X` = Inside (continuation of) an entity of type X
- `O` = Outside (not an entity)

Example from our tokenizer:

| Prime | Minister | Shehbaz | Sharif | visited | Lahore | on | 5 | March | 2024 | . |
|---|---|---|---|---|---|---|---|---|---|---|
| O | O | B-PERSON | I-PERSON | O | B-CITY | O | B-DATE | I-DATE | I-DATE | O |

("Prime Minister" is a courtesy title, so it stays `O`; `5 March 2024` is one DATE of three words.)

20 types x 2 (B and I) + 1 (O) = **41 tags**. They are defined once in `scripts/labels.py` (`LABELS`, `LABEL2ID`, `ID2LABEL`) so training, evaluation and prediction can never disagree about which number means which tag. `O` is id 0.

**Design decisions and why**
- *Flat entities (no nesting):* "Lahore High Court" is one COURT_LEGAL_BODY; we do not also label "Lahore" inside it. Nested labels would need a different, harder model, and the benchmark almost certainly uses flat spans.
- *Longest sensible span wins:* "Elections Act 2017" is one LAW_ACT, not a LAW_ACT plus a DATE.

## A3. Two kinds of tokenization (this confuses everybody, so read slowly)

There are **two different tokenizers** in the project. They do different jobs.

### 1) Our word tokenizer: `scripts/tokenization.py`

**What.** Cuts a sentence into *words and punctuation*. These are the units that receive tags.

**Why we wrote our own.** Real and synthetic data must be cut in exactly the same way, and some of our entity types have special shapes (CNIC `35202-1234567-1`, phone `0300-1234567`, money `Rs5.63`, time `3:30`). A normal `split()` would break `35202-1234567-1` or leave punctuation glued to words.

**How.** One regular expression, `TOKEN_RE`, tried left to right:

```
\d+(?:[,.:/\-]\d+)*     numbers; commas, dots, colons, slashes, hyphens INSIDE a number stay in it
|\w+(?:[-.]\w+)*        words; a hyphen or dot INSIDE a word stays in it (Minar-e-Pakistan, PML-N)
|'s                     possessive 's becomes its own token (so it is never part of an entity)
|[^\w\s]                every other punctuation mark is its own token
```

Real output:

```
Rs. 1,200 for Imran Khan's CNIC 35202-1234567-1 at 3:30 PM, up 15%.
-> Rs | . | 1,200 | for | Imran | Khan | 's | CNIC | 35202-1234567-1 | at | 3:30 | PM | , | up | 15 | % | .
```

Note how `1,200`, `35202-1234567-1` and `3:30` each stay **one** token, `Khan's` becomes `Khan` + `'s`, and `15%` becomes `15` + `%`.

**Why these rules matter.** If `35202-1234567-1` were split into five tokens, the model would have to tag five pieces and any slip would break the entity (exact-match scoring). If `'s` stayed glued, "Khan's" would be a different word from "Khan".

**Two helper functions.** `spans_to_bio(text, spans)` converts character spans (start, end, label) into tokens and BIO tags (a token takes the label if it overlaps the span; the first token gets `B-`, later ones `I-`). `bio_to_spans(text, tags)` does the reverse. Everything (draft labels, human labels, synthetic data, predictions) is converted through these two functions, so they are all comparable.

**Works for Urdu too.** Python's `\w` matches Urdu letters, so Urdu words stay whole, while the Urdu comma `،` and the Urdu full stop `۔` are not word characters and become their own tokens. We checked: `کوئٹہ، پشاور ... بڑھ گئی۔` splits into words, `،` and `۔` correctly.

### 2) The model's subword tokenizer: XLM-RoBERTa SentencePiece

**What.** The pretrained model has a fixed vocabulary of about 250,000 *pieces* (frequent words and word fragments). It cannot read a word it has never seen as one unit, so it breaks it into known pieces. A common word may be one piece; an unusual name may be three or four.

**Why subwords exist.** A vocabulary of whole words would never be large enough (every name, every typo). Pieces let the model represent *any* string and share knowledge between similar words (the pieces of "Nawabshah" resemble pieces of other Sindhi place names).

**The problem this creates.** Our labels are per *word*, but the model sees per *piece*. We solve it with the **first-subtoken rule**:

- the **first piece** of a word carries the word's label;
- the other pieces get the special label `-100`, which the loss function **ignores**;
- at prediction time we read the tag from each word's first piece.

So one word = one prediction, and the scoring sees the same words as the labels.

Other special tokens (start `<s>`, end `</s>`, padding) also get `-100`. Sentences longer than 192 pieces are cut (`--max_len 192`); in training, exactly one sentence was affected.

## A4. What is a transformer, and what is "fine-tuning"?

**Transformer.** A neural network that reads all words of a sentence at once and, through "attention", lets every word look at every other word to decide what it means in context. That is why the model can tell `Pakistan` (COUNTRY) from `Pakistan` in "Pakistan beat India" (SPORTS_TEAM): the surrounding words differ.

**Pretraining.** XLM-RoBERTa-base (270 million adjustable numbers called *weights*) was trained by Facebook AI on huge amounts of text in 100 languages by guessing hidden words. It knows grammar and general world knowledge but knows nothing about our 20 labels.

**Fine-tuning.** We add one small new layer on top that turns each word's representation into 41 scores (one per tag), then continue training the whole network on our labelled sentences. The model keeps its language knowledge and learns our task. This needs far less data than training from scratch, which is why a few thousand good sentences are enough.

**Why XLM-RoBERTa-base** (model choice, full table in the report): strong NER baseline; vocabulary that handles Pakistani and Urdu-origin names; multilingual (Urdu possible); small enough for a free Colab T4 GPU; DeBERTa/large models were slower or riskier on a short deadline; spaCy's own NER was used only as a noisy labeller; LLM prompting is not "fine-tuning".

## A5. The metrics, with a worked example

NER is scored **per entity** with **exact match**: a predicted entity counts as correct only if its start, end *and* type are all identical to the gold entity.

Suppose the gold text has 4 entities and the model predicts 4 entities:

| | gold | predicted |
|---|---|---|
| `Lahore High Court` (COURT_LEGAL_BODY) | yes | `Lahore High Court` COURT_LEGAL_BODY: **correct** |
| `Imran Khan` (PERSON) | yes | `Imran` PERSON: **wrong boundary** (counts as one miss and one false alarm) |
| `Lahore` (CITY) | yes | `Lahore` CITY: **correct** |
| `Dawn` (MEDIA_OUTLET) | yes | not found: **missed** |
| (nothing) | no | `Monday` TIME: **false alarm** |

- Correct (true positives) TP = 2 (`Lahore High Court`, `Lahore`); predicted = 4 (`Lahore High Court`, `Imran`, `Lahore`, `Monday`); gold = 4 (`Lahore High Court`, `Imran Khan`, `Lahore`, `Dawn`). So precision = 2/4 = 0.50, recall = 2/4 = 0.50, F1 = 0.50.
- **Precision** = TP / predicted = "of what the model said, how much was right".
- **Recall** = TP / gold = "of what was really there, how much did the model find".
- **F1** = 2PR / (P + R), the harmonic mean: it is low if either P or R is low, so you cannot cheat by predicting almost nothing or everything.

**Micro vs macro.** *Micro* pools all entities of all types and computes one P/R/F1 (big types like PERSON dominate). *Macro* computes F1 per type and averages (every type counts equally, so rare, hard types matter). Our selection rule uses both.

**Support** = number of gold entities of a type in the test set. A score on 2 entities (`EDUCATIONAL_INSTITUTION` in the random test) is almost meaningless; that is why we pool test sets and built `extra_test`.

**Noise.** On sets this small, two runs differing by under about 0.02 F1 are statistically tied. We never claim "better" below that margin.

**Why we implemented the scorer ourselves** (`train_ner.py`: `get_entities`, `score`, `format_report`): the `seqeval` package failed to build on Colab. The rule is the same one seqeval/conlleval use, and owning the code means we can explain it.

---

# Part B - Setup and data

## B1. Step 0 - Setup: labels and rules (18 Oct)

- **What:** fixed the 20 types, BIO tagging, wrote `GUIDELINES.md` and `scripts/labels.py`.
- **Why:** the labels are the "target variable" of NER. If two annotators label "Supreme Court" differently, the data contradicts itself and the score is capped. A single label file prevents id mismatches.
- **Decisions:** English first (most Pakistani public text is English; Urdu annotation is slow); XLM-RoBERTa-base; flat entities.

## B2. Step 1 - Data-sourcing policy (18-19 Oct)

- **What:** ranked sources by legal risk (Tier 1 openly licensed; Tier 2 public records; Tier 3 commercial news, kept small) and logged every source in `DATA_SOURCES.md`.
- **Why:** the competition requires respecting site terms and avoiding private data. A written policy proves it.
- **Rules we follow:** check robots.txt and terms; rate-limit (at least 1 s between requests; we used 1.5 s); identify our scraper with a User-Agent that contains a contact email; use text only to train and evaluate, never redistribute it; no full articles pasted into the report; no personal data; phone and CNIC synthetic only; credit Wikipedia (CC BY-SA 4.0).
- **Not used:** Common Crawl and LLM-generated sentences by default.

## B3. Step 2 - Collection (19 Oct)

- **Wikipedia and Wikinews:** `scripts/scrape_wikipedia.py` uses the official MediaWiki API (not HTML scraping), identifies itself, waits 1.5 s between calls, caps the number of pages and logs every document. Result: 209 Wikipedia + 23 Wikinews documents (about 2 million characters). Some category names we guessed did not exist; we checked and replaced them.
- **Kaggle datasets** kept: Pakistani news headlines (26.7k), Dawn (778) and Express Tribune (3,309) articles, Supreme Court judgments (2,809 files), Pakistan Penal Code (405 rows), census table (for gazetteers).
- **Removed after inspection:** Imran Khan tweets (garbled Urdu, personal social media), India-Pakistan cricket tweets (informal, not Pakistan-specific, 238 MB in 4 duplicate formats), Imran Khan Reddit (informal forum text).
- **Why a mix:** encyclopedic, news and legal text together generalise better to an unseen benchmark than one big source.

## B4. Step 3 - Cleaning and balancing (`scripts/build_corpus.py`)

Result: **15,390 sentences** in `data/raw/sentences.jsonl` plus gazetteers (provinces, divisions, districts, tehsils, penal-code sections).

Each cleaning step and its reason:

1. **Tolerant decoding.** Files mix UTF-8 with stray cp1252 bytes. A fallback decoder avoids crashes and garbled characters ("Ã¢â‚¬").
2. **Normalising text.** Curly quotes, long dashes and non-breaking spaces become plain ones, so the tokenizer sees consistent input.
3. **Junk removal.** Photo captions, "Read more / Published in", URLs, ALL-CAPS headers, very short or very long lines and lines that are mostly non-letters are not natural sentences and would teach noise.
4. **Abbreviation-safe sentence splitting.** "Mr.", "Rs.", "M. Khan" must not end a sentence; otherwise entities are cut in half.
5. **Court-judgment clean-up.** Empty files dropped; header blocks (judge lists, case numbers) cut; OCR-broken sentences ("Ordinanc e") and citation lists removed.
6. **Entity-light sentences mostly dropped** (25% down to 11%). A sentence with no capitalised word and no digit teaches only "O". A small share remains so the model also learns what is *not* an entity.
7. **Headlines.** 60% were world news; Pakistan-related ones favoured.
8. **De-duplication.** Exact and near-duplicates (same first 12 words).
9. **Balancing.** Caps per source and per document: otherwise 65 MB of legal text would dominate. Final mix: 18% Tier 1, 24% Tier 2, 58% Tier 3 (news-heavy because the benchmark is probably news-like).
10. **Document id (`group`).** Each sentence remembers its source document. This one field makes the leakage-proof split in Part D possible.

Known limits: some OCR noise remains in court text; rare classes are thin; the province gazetteer lacked ICT, Gilgit-Baltistan and AJK (added by hand).

## B5. Step 4 - Synthetic data (`scripts/generate_synthetic.py`)

- **What:** template-generated sentences with exact labels. 5,390 for the final model; 6,260 in later runs.
- **How:** about 150 templates with slots such as `{PERSON}`, `{CNIC}`, `{MONEY}`. Each slot is filled from a generator or list; static text and entity are tokenized separately; entity tokens get `B-`/`I-`. The labels are exact with zero manual work.
- **Why:** (1) phone and CNIC must be synthetic by the rules; (2) pattern classes (money, percent, date, time) get perfect labels cheaply; (3) rare classes (SPORTS_TEAM, MEDIA_OUTLET, EVENT, LANDMARK_FACILITY, LAW_ACT, COURT_LEGAL_BODY, EDUCATIONAL_INSTITUTION) get 260-320 sentences each because real text has few.
- **Formats generated.** CNIC `#####-#######-#` with first digit 1-7, plus no-dash and spaced variants. Phones: `0300-1234567`, `+92 300 1234567`, `+923001234567`, `0092-...`, landlines. All digits random; no real person.
- **Decoys.** Templates include numbers that are NOT entities (FIR numbers, flight numbers, page counts) so the model does not learn "every number is an entity".
- **Risks.** The model may learn template wording; name lists are not exhaustive; synthetic share must stay a minority. Mitigations: 5,390 synthetic vs 15,390 real; templates are stored in `group` so whole templates can be held out; a separate pattern test uses brand-new templates.

---

# Part C - Labelling

## C1. Why we label in layers

Hand-labelling 15,000 sentences in three days is impossible. Correcting a draft is several times faster than labelling from scratch. So we build a ladder of label quality:

| Quality | Name | Source | Use |
|---|---|---|---|
| Highest | **Gold** | human-verified | testing (and a little training) |
| Medium | **Silver** | rules + gazetteers + spaCy, then bulk-fixed | most of the training |
| Exact but artificial | **Synthetic** | templates | phone/CNIC, rare types |

## C2. Step 5 - Pre-annotation (`scripts/preannotate.py`)

Three layers, in order of reliability; each span remembers its **origin layer**:

1. **Regex** for PHONE, CNIC, MONEY, PERCENT, TIME, DATE and law references. Patterns have a fixed shape, so rules are precise.
2. **Gazetteers**: curated lists (provinces, divisions, districts, courts, parties, abbreviations). Matched longest-first so "Lahore High Court" beats "Lahore".
3. **spaCy NER** (pretrained): finds PERSON and generic ORG/GPE/FAC/EVENT that rules cannot list. Its coarse ORG is re-classified by keyword rules: University becomes EDUCATIONAL_INSTITUTION, Court becomes COURT_LEGAL_BODY, Ministry/Police become GOVERNMENT_BODY.

**Overlap rule:** strong regex first, then longest span, then layer priority. Courtesy titles are stripped from PERSON. Each span stores `agree` (how many layers proposed it); each sentence stores `conflicts`.

**Three review rounds (5b, 5c, 5d)** looked at real output and fixed systematic mistakes: a newspaper tagged PERSON, a mosque tagged PERSON, "Insaf" (half of a party name) tagged PERSON 62 times, bare "Court" tagged 385 times, countries tagged CITY, TIME durations, upper-case datelines. After each round we verified the numbers moved (bare "Court" 385 to 0; IHC to COURT_LEGAL_BODY; Khyber-Pakhtunkhwa to PROVINCE_REGION). Unit cases (40) protect the rules. We then **stopped** tuning (diminishing returns, deadline) and let the human check measure what was left.

**Process lesson:** a debug run overwrote real output twice; debug runs now write to `data/annotated_debug/`.

## C3. Step 6-7 - Human verification (Label Studio)

- **What:** 400 sentences corrected by hand: **200 test** (random sample; whole documents held out; never trained on) + **200 review** (half rare-label coverage, half "uncertain" sentences; used for training and to learn the drafts' errors). Later **100 more** (`extra_test`).
- **AI-assisted first pass (disclosed).** An AI assistant corrected the drafts first (`scripts/apply_corrections.py`: 273 spans removed, 179 added), and the human then reviewed every sentence. This is "model-assisted, human-verified"; the human pass is what makes the test set gold.
- **Why the test sample is random.** A hand-picked test set would be biased toward rare labels and not represent the natural label mix. The review sample is the opposite (enriched) because training wants hard cases.
- **Measured draft quality** (`import_labelstudio.py` and `analyze_review.py`, exact span+label vs human): strong PERSON (~0.88), POLITICAL_PARTY (~0.9), MONEY, PERCENT; weak ORGANIZATION (precision ~0.3), GOVERNMENT_BODY (recall ~0.57), LANDMARK/EVENT/SPORTS_TEAM (recall 0.3-0.5).
- **Why we compute draft-vs-human tables per split:** bulk-fix decisions use REVIEW errors only; the TEST split stays clean.

## C4. Step 8 - Conventions (`GUIDELINES.md`, `scripts/normalize_conventions.py`)

Rules that surfaced during annotation, written down and applied automatically. **Why:** the model learns whatever is consistent; inconsistent rules directly lower the score.

| Rule | Reason |
|---|---|
| Durations ("two weeks", "23 months") are DATE | OntoNotes/spaCy convention; TIME is only time of day |
| "Thursday morning" = DATE + TIME (two spans) | each part has its own type |
| A year inside a name stays inside ("Elections Act 2017" = one LAW_ACT) | longest sensible span |
| A year inside a case number or citation is never DATE (C.A.538/2022, 2010 SCMR 1962) | human labels never mark them; silver had 88 such errors |
| Leading "The" is not part of a name | annotators excluded it consistently |
| Capitalised bare "High Court"/"Supreme Court" = COURT_LEGAL_BODY; lowercase "the court" is not | benchmark is probably court records, so bare mentions likely count; generic words are not entities |
| Kashmir, Thar etc. = PROVINCE_REGION; districts and towns = CITY | consistency |
| Rivers, stadiums, police stations, bazaars, shrines = LANDMARK_FACILITY | one home for facilities |
| National teams in a match context = SPORTS_TEAM, otherwise COUNTRY | context decides |
| Abbreviations take the type of what they stand for (WAPDA = GOVERNMENT_BODY) | consistent |
| FIR, flight, case numbers, page counts are not ID or PHONE | they are numbers but not our ID types |
| Bare job titles (IGP, CEC) and generic nouns are not entities | not named things |

These are judgement calls. The hidden benchmark may decide some differently; we state that as a limitation.

## C5. Step 9 - Bulk fix of silver labels (`scripts/origin_precision.py`, `scripts/bulk_fix_silver.py`)

- **The idea.** A human correction fixes only its own sentence, but a *pattern* of errors repeats thousands of times. We measure precision for every (label, layer) pair on the **review split**, then write rules that fix the patterns everywhere.
- **What the measurements showed.** Reliable (precision about 0.9+): PERSON (spaCy), POLITICAL_PARTY, PROVINCE_REGION, gazetteer GOVERNMENT_BODY/CITY, regex MONEY/PERCENT. Unreliable: spaCy ORGANIZATION 0.32, GOVERNMENT_BODY 0.43, TIME 0.20, EDUCATIONAL_INSTITUTION 0.57, EVENT/LANDMARK about 0.5.
- **Fixes (each justified by review errors).** Extra gazetteer names; drop junk (IGP, CEC, "Local Commission"); drop lowercase "apex court"; TIME kept only with a digit or a time-of-day word; duration-like TIME becomes DATE; "Wednesday night" split into DATE + TIME; years folded into law/event names; national teams become SPORTS_TEAM only with strong sport cues (runs, wickets, innings, ODI, T20...). A first, broader cue list ("final", "series", "match") turned "Pakistan" into a team in political sentences; **spot-checking caught it** (195 conversions instead of 304).
- **Deliberately not added:** IPP, NAP, GDA, NP: each also means something else; one wrong gazetteer hit would mislabel many sentences.
- **The `trusted` flag.** A silver sentence is trusted if no unreliable spaCy-origin span remains. Result: 11,295 trusted, 3,695 untrusted. Agreement with the human: trusted 0.68 (review) / 0.72 (test); untrusted 0.32 / 0.04. The flag separates good from bad.
- **Why we drop or mask instead of patching:** an unlabelled real entity teaches the model "this entity is O", which is worse than leaving the sentence out. (Part D explains masking.)
- **Honesty:** the test numbers were looked at once as an out-of-sample estimate and never used to change a rule.

## C6. Step 10 - Weak classes (`scripts/label_counts.py`, `scripts/select_extra.py`)

- **Why.** After the trust rule, real data was almost empty for the weak labels (ORGANIZATION 60, LANDMARK 50, EDUCATIONAL 10, EVENT 105, TIME 93 spans) and the random test had only 2-8 gold spans for them.
- **What.** 300 extra sentences rich in weak labels (one per document, 8-50 tokens, round-robin, rarest first): **100 `extra_test`** (human-verified, a second gold set) and **200 `extra_train`** (assistant-labelled, **not human-verified**, disclosed).
- **Leakage design.** extra_test documents are chosen first, are disjoint from test, review and extra_train documents (asserted in code) and are saved in `extra_test_groups.json` so the builder removes all their sentences from training. extra_test is reported separately because it is selected, not random.
- **Second-pass check.** The assistant re-read all 300 and `scripts/check_tasks.py` ran automatic checks. This caught a script bug (adding "Pakistan" cut "Pakistan Cricket Board", now impossible), "early Friday" becoming "Friday", and Landhi/Korangi as CITY. Limit: the same model checking itself finds slips, not systematic bias.
- **Quality estimate.** On the 100 human-verified sentences the assistant's labels had precision about 0.92 and recall about 0.89; this tells us roughly how noisy extra_train is.

---

# Part D - Dataset building and leakage

## D1. What is data leakage?

**Leakage** = the model sees, during training, something that is also in the test. The score then measures *memory*, not *skill*, and collapses on the hidden benchmark.

Why it is sneaky in NER: sentences from the same document share people, places and phrasing. If sentence 3 of a court judgment is in training and sentence 4 is in the test, the model has effectively seen the answer. News headlines can also be near-duplicates of each other.

## D2. How we prevent it (`scripts/build_dataset.py`)

1. **Split by document, not by sentence.** Every sentence has a `group` (source document). Whole groups go to one split.
2. **Remove test siblings.** All other sentences from any test document are removed from training (965 silver sentences).
3. **Never train on a silver copy of a human sentence.** 300 dropped; the human version is used.
4. **Near-duplicate check.** Any training sentence sharing its first 12 words or an 8-word run with a test sentence is dropped (64).
5. **Separate validation split.** 8% of document groups (seed 42), used only to pick the best epoch.
6. **Synthetic data is training-only.**
7. **Hard assertions.** The script *stops with an error* if train/validation/test/extra_test groups overlap, a test id or text appears in train, tags are invalid or lengths mismatch. Result: all assertions passed in every build.

## D3. Masking: why not just delete or keep noisy spans?

Silver spans from unreliable layers (spaCy ORGANIZATION, GOVERNMENT_BODY, EDUCATIONAL_INSTITUTION, EVENT, LANDMARK, MONEY) have three possible treatments:

| Treatment | Effect | Verdict |
|---|---|---|
| Keep the label | teaches false labels | bad |
| Relabel as `O` | teaches "this real entity is not an entity" | bad |
| Drop the sentence | loses the correct labels in it | wasteful |
| **Mask (`-100`)** | words stay as context; loss ignores them | **chosen** |

`-100` is the value PyTorch's cross-entropy ignores. A masked word is neither rewarded nor punished.

## D4. What went into the final model's training set

18,245 sentences: synthetic 5,390; trusted silver 9,516; masked silver 2,974; human review 186; assistant extra 179. Validation 1,206, test 200, extra_test 100. Later builds (v2, v3) have 19,115 training sentences because synthetic data grew to 6,260.

**Why is validation higher than test?** Validation is split from the same mostly machine-labelled pool as training, so it is easier and its labels carry the same noise. Only the human-verified test sets are the honest measure.

**Known limits.** Validation has no PHONE/ID (synthetic data is training-only); the random test has no PHONE/ID and very few EVENT/SPORTS_TEAM/TIME/EDUCATIONAL spans.

---

# Part E - Training in detail

## E1. What happens in one training step

1. Take **16 sentences** (batch size) from the training set (shuffled each epoch).
2. Words are cut into subword pieces and padded to equal length; special tokens are added.
3. The model produces, for every piece, 41 scores (the **logits**).
4. **Loss (cross-entropy)** compares the scores for each first-piece with the target tag and averages. Positions labelled `-100` are skipped.
5. **Backpropagation** works out, for each of the 270 million weights, how a tiny change would reduce the loss (the **gradient**).
6. **AdamW** (the optimiser) changes every weight a little, scaled by the **learning rate**.

One pass over all training data = **epoch**. After each epoch the model is scored on the validation set (not trained on) and a checkpoint is saved; at the end the epoch with the best validation F1 is restored.

## E2. Hyper-parameters, and why each value

| Setting | Final model | Why |
|---|---|---|
| Learning rate | 3e-5 | standard for BERT-family fine-tuning; higher forgets pretraining, lower is slow |
| Warm-up | first 10% of steps | slow start protects pretrained knowledge from large random updates |
| Decay | linear to 0 | lets the model settle at the end |
| Batch size | 16 | fits a 16 GB T4 |
| Epochs | 4 (v1); 6 (v2, v3) | v1's validation F1 was still rising at 4, so later runs used 6 |
| Weight decay | 0.01 | mild penalty on large weights, reduces overfitting |
| fp16 | on | half precision: faster and uses less memory |
| Max length | 192 pieces | covers all but one sentence |
| Oversampling | human review x3, assistant extras x2 | the best data is only about 2% of the total |
| Seed | 42 | reproducible shuffling and initialisation |
| `load_best_model_at_end` | F1 | keep the best epoch, not the last |

**Oversampling arithmetic (v2 log):** 19,115 sentences + 186 review x (3-1) = +372 + 179 assistant x (2-1) = +179 gives 19,666. At batch 16 that is 1,230 steps per epoch; 6 epochs = 7,380 steps; about 9 steps per second on a T4 means about 14 minutes.

## E3. Reading the training log

- `train 19666 validation 1206 test 200 extra_test 100` - the sizes above.
- `LOAD REPORT: classifier MISSING, lm_head UNEXPECTED` - normal. The 41-tag layer is new (starts random and is what learns NER); the pretraining word-guessing head is not used.
- `sentences truncated by max_len: 1 0` - one training sentence was cut.
- `loss` falls from about 2.7 to about 0.03: it starts high because the new layer is random, then drops fast and slowly improves. A steady fall is healthy.
- `learning_rate` ramps up over the first 738 steps then decays.
- `grad_norm` is the size of the update direction; spikes happen; the Trainer clips it to 1.0 so one odd batch cannot wreck the model.
- `eval_f1` after each epoch (v2: 0.850, 0.880 ...). **Overfitting** looks like training loss falling while validation loss rises; best-epoch selection protects us.

## E4. Environment

Google Colab, NVIDIA Tesla T4 (16 GB). `nvidia-smi` is not a download; it is the NVIDIA driver's status tool that proves a GPU is attached. Libraries: `transformers 5.20.2`, `accelerate 1.15.0`, `torch 2.11.0+cu130`, `numpy 2.1.3` (Colab installs the latest each session). We avoided `seqeval` (build failure) and Hugging Face `datasets` (our project folder `datasets/` would shadow it) and used a plain PyTorch `Dataset`.

---

# Part F - Evaluation, error analysis and model choice

## F1. Evaluation protocol

- Exact entity match, overall and per label (Part A5).
- Sets: random **test** (200 sentences, 426 entities), **extra_test** (100, 338), both **pooled** (764), and the synthetic **pattern test** (188).
- Test sets are scored only after training; validation picks the epoch. No tuning on test.

## F2. Why 18 labels appeared, not 20

The reports only listed labels present in the test data. We may not use real phones or CNICs, so neither human test set contains them: `PHONE_NUMBER` and `ID_NUMBER` had support 0 and were silently omitted. The model *was* trained on them (556 phones, 444 CNICs). We fixed the report to always list all 20 labels (`n/a` when there is no example) and built the pattern test.

## F3. The pattern test (`scripts/make_pattern_test.py`)

188 synthetic sentences (102 phones, 84 CNICs, plus PERSON/CITY context). It is a fair test of the *pattern*, not of memory, because: every template is new; digits come from a different seed; about 30% of numbers use formats never seen in training (`+92 (300) 1234567`, `0300.1234567`, `35202/1234567/1`); and **decoys** that are not phones/CNICs are included (IBANs, number plates, tracking ids, case references, 6-digit codes, `NNNN-NNNN`).

**A flaw we found and fixed.** The first version included "this morning", "on Monday", percentages, amounts and dates. The model correctly tagged TIME/DATE/PERCENT/MONEY, but the gold only labelled phone/ID/person/city, so these counted as false positives. Those decoys were removed and the sets re-scored; we quote only clean numbers.

**How to report it.** It shows the model learned the patterns and ignores decoys; it is *not* evidence about real documents.

## F4. Error analysis, step by step

Why: one overall number says nothing about *where* the model fails, and the cure depends on the cause. (Repeatable script: `scripts/error_summary.py`; method file: `ERROR_ANALYSIS.md`.)

1. **Per-label scores** (overall F1 0.816 hides PERSON 0.92 vs LAW_ACT 0.48).
2. **Check supports** before trusting a number.
3. **Compare the two test sets** (0.816 vs 0.812: no sign of memorising easy patterns).
4. **Dump every wrong sentence** (`errors_*.txt`: gold vs predicted per word).
5. **Count error kinds**: *missed* (gold entity, model said O), *spurious* (gold O, model said entity), *confusion* (wrong type), *boundary* (right type, wrong edges). Each has a different cure. v1: random test 157 wrong words = 61 missed, 43 confusion, 38 spurious, 15 boundary; extra_test 153 = 76 / 50 / 17 / 10.
6. **Group into causes** and ask for each: is the MODEL wrong, the TRAINING DATA wrong, or the TEST LABEL questionable?
   - GOVERNMENT_BODY vs ORGANIZATION vs POLITICAL_PARTY confusion (few clean examples, spaCy ORG layer masked)
   - long LAW_ACT names split (training laws were short)
   - landmarks missed or called CITY/PERSON (rivers, dams, ports, shrines, police stations)
   - spurious DATE on years inside case numbers (88 in silver)
   - vague DATE phrases (kept: the human labels "last year", "a day" as DATE)
   - COUNTRY vs SPORTS_TEAM (context)
   - unfamiliar village names read as PERSON (data-size limit)
   - a few test labels look wrong (one Kashmir tagged CITY; **not edited**)
7. **Decide what to fix and what not to**: fix clear data causes supported by *human* labels.
8. **Check each rule against human labels first** (`scripts/date_audit.py`).
9. **The honesty rule (leakage inside error analysis).** Reading test errors leaks a little information. We added **no** test entity string to any gazetteer or template, validated rules on the review split, never edited test labels, and let only error *categories* drive changes. We still state that the test set is no longer fully unseen.
10. **Retrain and compare** with before/after numbers.

## F5. Improvement rounds

**Round 1 (v2).** Case-number-year rule; synthetic data 5,390 to 6,260 (more distinct landmarks, bodies, long laws, national-team sentences, `CASE_NOISE` templates); 6 epochs.

**Round 2 (v3), masking unlabelled names (`scripts/mask_unlabelled.py`, `scripts/mask_eval.py`).** Diagnosis: ORGANIZATION recall stayed low because silver sentences contain real organisations and landmarks the draft missed, labelled `O`, which teaches "that name is not an entity". Fix: mask runs of 2+ capitalised `O` tokens (connectors of/and/for allowed) or single ALL-CAPS 2-6 letter tokens, with exclusions (sentence-initial words, months, weekdays, titles, legal words, common abbreviations PM IGP CJP CEC FIR GDP NA CNIC...). Evidence on the review split only: covers 46% of draft-missed entity tokens (29 of 63) while masking only 0.7% of truly-O tokens (34 of 4,572).

**What happened.** Round 2 moved errors from *missed* (74 to 43 on the random test) to *spurious* (29 to 52) and *confusion*. Net F1 change was within noise.

**Round 3 (v4), more human labels (Step 33-36).** Rounds 1 and 2 only re-processed the same noisy labels. Round 3 added real human work: 400 new sentences rich in the weak types (ORGANIZATION, GOVERNMENT_BODY, LAW_ACT, LANDMARK_FACILITY, EDUCATIONAL_INSTITUTION, EVENT) were selected one per document, never from a test document (`scripts/select_batch3.py`). An AI assistant corrected the drafts (`batch3_corr_01..06.json`, applied by `apply_corrections.py`: 333 spans removed, 223 added); a human then verified **264** of them in Label Studio. These were imported (`import_labelstudio.py`), normalised (`normalize_conventions.py`, 2 sentences changed) and added to training only (repeated x3 like other human data; flag `USE_BATCH3` in `build_dataset.py`). Leakage was checked three ways: document groups and ids disjoint from both test sets (now asserted in the build), no identical text, and only common phrases ("Election Commission of Pakistan (ECP)", "according to an MQM press release") shared with test sentences. v4 uses exactly the v3 recipe, so v3 vs v4 isolates the effect of the human data.

How good was the assistant draft against the human (exact span and label, precision / recall)? PERSON 0.96 / 0.95, GOVERNMENT_BODY 0.95 / 0.91, POLITICAL_PARTY 0.98 / 0.98, EVENT 0.98 / 0.96, LAW_ACT 0.77 / 0.79 (weakest), SPORTS_TEAM 0.85 / 0.61. The raw rules+spaCy draft was far lower (ORGANIZATION precision 0.19, LAW_ACT 0.49 / 0.51): LLM-assisted pre-annotation saves a lot of human effort.

## F6. Results of the four runs and the honest conclusion

| Run | Micro P | Micro R | Micro F1 | Macro F1 (20 labels) |
|---|---|---|---|---|
| v1 | 0.806 | 0.822 | 0.814 | 0.807 |
| v2 | 0.819 | 0.810 | 0.814 | 0.798 |
| v3 | 0.786 | 0.836 | 0.810 | 0.800 |
| v4 (experiment) | 0.803 | 0.851 | **0.826** | **0.814** |

(Macro values computed exactly by the report script; the earlier log entry rounded them as 0.806 / 0.798 / 0.799 - the ranking is the same.)

**Conclusion.** v1, v2 and v3 tie on internal gold. v4 adds human-verified labels and reaches micro F1 0.826 (+0.012 over v1, inside noise), with gains on weak types but more spurious predictions. **Deployed final model: v1** — on the organisers' sample benchmark (80 sentences, their scorer), v1 micro F1 **0.979** vs v4 **0.958**, so `make_submission.py` uses `xlmr_base/best` (v1 weights). What limits the score is **label quality**, not model size.

## F7. How the final model was chosen (rule and decision)

First we noticed our own earlier mistakes: a claim "v2 is best" over-read noise (+0.009 on one set), and a pick of v3 because only v3 had pattern-test numbers (a gap in verification, not in quality). The fix was a **rule fixed in advance**, applied to all models scored by the same code on the same data (`--eval_only` mode):

1. pooled micro F1 over both human test sets;
2. among ties, best macro F1 over all 20 labels (18 human-tested + PHONE_NUMBER and ID_NUMBER from the pattern test);
3. tie-break: smallest |precision - recall| gap;
4. never choose on one set or on one label.

Internal rule on four models: v4 ranks first on pooled gold (micro 0.826, macro 0.814). For the TechFest **submission file** we added a second check: both models on the organisers' sample with `optional_self_check/score.py`. **v1 wins (0.979 vs 0.958 micro F1)** — fewer spurious CITY/DATE/ORG spans. **v1 is the final deployed model**; v4 remains documented as an experiment.

**Label bands (final model v1, 20 labels).** F1 at least 0.90: 6; at least 0.80: 14; below 0.60: 3 (ORGANIZATION, LAW_ACT, LANDMARK_FACILITY).

**Paragraph test.** All 20 types in one long paragraph: 50 of 54 predicted entities identical across v1, v2, v3. Differences: only v1 found "Dawn"; v2 cut "Anwaar-ul-Haq Kakar"; v1 called "Karachi Press Club" MEDIA_OUTLET (v2/v3 said LANDMARK_FACILITY, correct); all three cut "Bilawal Bhutto Zardari" to two words. A sanity check, not a score.

---

# Part G - Prediction, Urdu, tools, mistakes

## G1. Prediction (`scripts/predict.py`)

`python scripts/predict.py --model outputs/xlmr_base/best --text "..."` or `--input file.txt|.jsonl --out predictions.jsonl`.

How: (1) cut the text into words with our word tokenizer, (2) run the model in windows of 100 words (so long texts fit), (3) take each word's tag from its first piece, (4) `fix_bio` repairs an `I-X` that has no `B-X` before it, (5) `tags_to_entities` turns tags into character spans. Output per input: `id`, `text`, `entities` (text, label, start, end), `tokens`, `tags`. The organisers' required format is JSONL with character offsets; their script `student_pack_techfest/student_pack/make_submission.py` has our model filled in (same steps as `predict.py`, self-contained). Rules: exactly the 20 labels, no overlaps, end exclusive, strict exact match, micro F1 ranks (macro F1 tie-break). The 80-sentence `sample_benchmark.jsonl` with `sample_answer_key.jsonl` is used only for scoring, never training. v1 scored 0.979 micro F1 on that sample; v4 scored 0.958.

## G2. Urdu (zero-shot)

The organisers said Urdu may earn extra points. We did **not** train on Urdu, but XLM-RoBERTa saw Urdu in pretraining, so we ran the first model (v1; the check predates v4) on 8 hand-written Urdu sentences with 27 expected entities (`notebooks/04_urdu_check_v1.ipynb`, scored offline by `scripts/score_urdu.py`).

Result: **21 of 27 found with the right type** (precision 0.750, recall 0.778, F1 0.764). Perfect for PERSON, CITY, MEDIA_OUTLET, SPORTS_TEAM, PHONE, ID, MONEY, PERCENT, TIME, EVENT and others. Misses are mostly *boundary* errors from Urdu grammar (the postposition "ko" attached to a date; a law name cut to its year; a party name split in two) and two province names tagged CITY.

Caveats: 27 entities written by us, not checked by a native annotator, no Urdu training: a sanity check, not a benchmark. What a proper Urdu round would take (not done): synthetic Urdu templates, 60-100 hand-labelled Urdu sentences, an Urdu section in the guidelines, mixed English+Urdu fine-tuning, and a check that English scores do not drop.

## G3. Tools and why each

| Tool | Use | Why |
|---|---|---|
| Label Studio | human annotation | free, local, shows drafts as pre-annotations |
| spaCy | draft labels only | knows persons/organisations; too noisy to be the product |
| PyTorch + transformers + accelerate | fine-tuning | standard stack |
| Google Colab T4 | GPU | free; fits base-size model |
| Google Drive | storing models | Colab storage is erased after each session |
| matplotlib + reportlab | charts and PDF | `scripts/make_report_pdf.py` rebuilds the report from saved metrics |

Notebooks: `01_preannotation_colab` (drafts), `02_train_colab` (train and evaluate), `03_eval_predict_all_models` (re-score all runs, paragraph test), `04_urdu_check_v1`, `05_train_v4_colab` (final model).

## G4. Mistakes we made and how we fixed them (good presentation material)

- `seqeval` would not install on Colab: wrote our own metric.
- `TrainingArguments` dropped `warmup_ratio` in new transformers: code inspects the installed version and adapts.
- Folder `datasets/` shadowed the Hugging Face library: plain PyTorch dataset.
- Debug runs overwrote real outputs: debug writes to a separate folder.
- A sport-cue list was too broad (Pakistan became a team in politics): caught by spot-check.
- Pattern test had decoys that were actually entities: found, removed, re-scored.
- Claimed "v2 best" from a +0.009 difference: retracted, replaced by a pre-agreed rule.
- Reports showed 18 labels: pattern test added.
- PowerShell here-strings corrupted markdown backticks in the log: repaired.
- Notebook unzip placed scripts one folder too deep: fixed the cell.

---

# Part H - Glossary and likely questions

## Glossary

| Term | Meaning |
|---|---|
| NER | named entity recognition |
| Token | a word-sized unit (our tokenizer) or a piece (the model's tokenizer) |
| BIO | B- begin, I- inside, O outside |
| Gold / silver / synthetic | human-verified / automatic noisy / template-made labels |
| Gazetteer | a list of known names (provinces, courts...) |
| Pre-annotation | automatic first-draft labels that humans correct |
| Fine-tuning | continuing training of a pretrained model on our task |
| Loss | number that says how wrong the model is; training lowers it |
| Epoch / step / batch | one pass over data / one weight update / sentences per step |
| Logit | raw score for each tag before choosing |
| Masking (-100) | positions the loss ignores |
| Leakage | test information visible in training |
| Support | number of gold entities of a type in a set |
| Exact match | start, end and type must all be identical |
| Micro / macro | pool all entities / average per label |
| Zero-shot | applying a model to something it was not trained on |
| Overfitting | memorising training data, doing worse on new data |

## Likely questions from judges, with short answers

- **Why XLM-RoBERTa and not an English model?** Better handling of Pakistani and Urdu-origin names, multilingual, fits a free GPU, strong NER baseline.
- **How did you avoid leakage?** Splits by document; siblings and near-duplicates removed; validation separate; assertions stop the build on any overlap.
- **Is your test set truly unseen?** It was never trained on, but we studied its error categories, so it is not fully unseen; no test text was added to data and no label edited. The hidden benchmark is the clean measure.
- **Where did the phone and CNIC data come from?** Randomly generated by us; no real person. Measured on a separate synthetic test with unseen formats and decoys.
- **Why is ORGANIZATION so low?** Few human labels, and the type overlaps with GOVERNMENT_BODY and POLITICAL_PARTY; unreliable automatic labels had to be masked.
- **Why did your improvements not help?** The first two rounds (synthetic data, masking) changed the type of error but not the total. The one thing that helped, a little, was more human-verified labels (v4): the limit is human label volume, not model size.
- **How did you choose the final model?** Internal gold favoured v4 slightly; the organisers' sample favoured v1 clearly (0.979 vs 0.958). We submit v1.
- **Did an AI do your labelling?** An AI assistant drafted first-pass corrections for the 400 and 100 gold sentences, the 400 batch-3 sentences and labelled 200 extra training sentences. Every gold test sentence and the 264 batch-3 training sentences were reviewed by a human; the 200 extra training sentences and the 136 unreviewed batch-3 sentences were not used as human data, and we disclose it.
- **What about Urdu?** Zero-shot sanity check found 21 of 27 expected entities; not trained, not a benchmark; a proper Urdu round is the next step.
- **What would you do with another week?** Human-verify more ORGANIZATION, GOVERNMENT_BODY, LAW_ACT and LANDMARK sentences; Urdu fine-tuning round; several seeds to shrink noise; adapt the output to the benchmark format.

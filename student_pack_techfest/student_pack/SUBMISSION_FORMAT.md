# TechFest NER Challenge: Submission Format

Your answers must be in **one file** in the layout below. A file in any other layout cannot be scored.

## Layout: JSONL (one line per sentence)

You will be given a test file where each line looks like this:

```
{"id": "TFN-DEV-0001", "text": "Shazia Baig (CNIC 71786-3754747-3) was issued a new smart card by NADRA."}
```

Your submission must have **one line for each sentence**, with the same `id` and the entities your model found:

```
{"id": "TFN-DEV-0001", "entities": [{"start": 0, "end": 11, "label": "PERSON"}, {"start": 18, "end": 33, "label": "ID_NUMBER"}, {"start": 66, "end": 71, "label": "GOVERNMENT_BODY"}]}
```

- `start` and `end` are **character positions** in the sentence text, counting from 0. `end` is exclusive, so `text[start:end]` is the entity (`text[0:11]` is "Shazia Baig").
- `label` must be **exactly** one of the 20 official labels below (upper case, same spelling). No other label names are allowed, so `PER`, `ORG`, `LOC`, `B-CITY` and similar are all invalid. A file with any other label is rejected: `make_submission.py` will not create it and tells you which labels were wrong. The 20 labels: PERSON, CITY, PROVINCE_REGION, COUNTRY, LANDMARK_FACILITY, GOVERNMENT_BODY, POLITICAL_PARTY, ORGANIZATION, EDUCATIONAL_INSTITUTION, COURT_LEGAL_BODY, LAW_ACT, EVENT, MEDIA_OUTLET, SPORTS_TEAM, DATE, TIME, MONEY, PERCENT, PHONE_NUMBER, ID_NUMBER.
- A sentence with no entities still needs a line, with `"entities": []`.
- Entities must not overlap. A sentence missing from your file counts as "no entities found".
- Scoring is strict: an entity counts only if `start`, `end` and `label` are all correct.

A complete example is in `sample_submission.jsonl`: it is the correct answer file for `sample_benchmark.jsonl` (80 sentences).

## How to create the file

Use `make_submission.py`. Open it and edit only the two sections marked `EDIT HERE`:

1. `load_model()`: load your trained model.
2. `predict(model, text)`: return the entities your model finds in one sentence, using only the 20 official labels. The file includes ready-made helpers `spans_from_hf` (Hugging Face pipeline output) and `spans_from_bio` (word tokens plus B-/I-/O tags), and examples for spaCy.

Then practise on the sample benchmark:

```
python make_submission.py --input sample_benchmark.jsonl --output my_submission.jsonl
```

It builds your file and checks the format. If your model produces any label outside the 20 official ones, it stops with an error and no file is created. Compare it with `sample_submission.jsonl` to see what a correct file looks like.

**In the lab** you will receive the real benchmark file in the same layout as `sample_benchmark.jsonl`. Run the same command with that file as `--input` and hand in the file it creates. Optional: the `optional_self_check` folder lets you score your sample file yourself with `python score.py --gold sample_answer_key.jsonl --pred my_submission.jsonl`.

To check an existing file: `python make_submission.py --check my_submission.jsonl --input sample_benchmark.jsonl`

"""Pre-annotate the real sentences with automatic labels (a draft for humans to correct).

Run from project root:
    python scripts/preannotate.py --model en_core_web_lg          # full (needs spaCy; Colab notebook does this)
    python scripts/preannotate.py --no-spacy                      # rules + gazetteers only

Layers (strongest evidence first):
  1. REGEX      phone, CNIC, money, percent, time, date, law references   (patterns are reliable)
  2. GAZETTEER  curated Pakistani lists + census districts               (known names, exact match)
  3. SPACY      pretrained NER (OntoNotes labels) mapped to our 20 labels (catches unseen PERSON/ORG/etc.)
Overlaps are resolved by: strong regex (phone/CNIC) first, then longest span, then layer priority.

Outputs (data/annotated/):
  preannotated.jsonl          every sentence: tokens, ner_tags (BIO), spans, origin of each span
  labelstudio_tasks.json      stratified REVIEW SAMPLE with predictions, for Label Studio
  labelstudio_config.xml      the 20-label annotation interface for Label Studio
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path

from labels import ENTITY_TYPES
from tokenization import spans_to_bio
import generate_synthetic as G

ROOT = Path(__file__).resolve().parent.parent
SENT = ROOT / "data" / "raw" / "sentences.jsonl"
GAZ = ROOT / "data" / "gazetteers"
OUT = ROOT / "data" / "annotated"

# ------------------------------------------------------------------ layer 1: regex
MON = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
ORD = r"(?:st|nd|rd|th)?"
CUR = r"(?:Rs\.?|PKR|USD|US\$|\$|EUR|GBP|€|£)"
SCALE = r"(?:thousand|million|billion|trillion|lakh|crore|bn|m|k)"
REGEX_RULES = [  # (label, compiled pattern, strong?)
    ("ID_NUMBER", re.compile(r"(?<![\d-])\d{5}-\d{7}-\d(?!\d)|(?<![\d-])[1-7]\d{12}(?![\d-])|(?<![\d-])\d{5} \d{7} \d(?!\d)"), True),
    ("PHONE_NUMBER", re.compile(r"(?<![\d-])(?:\+92|0092|0)[\s-]?3\d{2}[\s-]?\d{7}(?![\d-])"
                                r"|(?<![\d-])\(0\d{2,3}\)\s?\d{3}-\d{3}-\d{3}"
                                r"|(?<![\d-])0(?:21|22|41|42|51|52|55|61|81|91)-\d{7,8}(?![\d-])"), True),
    ("MONEY", re.compile(rf"{CUR}\s?\d[\d,]*(?:\.\d+)?(?:\s?{SCALE}\b)?|\b\d[\d,]*(?:\.\d+)?(?:\s{SCALE})?\s(?:rupees|dollars|Rupees)\b"), True),
    ("PERCENT", re.compile(r"\b\d+(?:\.\d+)?(?:\s?%|\s?(?:percent|per cent|pc)\b)"), True),
    ("TIME", re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:a\.m\.|p\.m\.|am|pm|AM|PM)(?!\w)|(?<![\d:/.-])\b(?:[01]?\d|2[0-3]):[0-5]\d\b(?![:/.-]?\d)|\b(?:noon|midnight)\b"), False),
    ("DATE", re.compile(rf"\b\d{{1,2}}{ORD}\s{MON}\b\.?(?:,?\s\d{{4}})?|\b{MON}\s\d{{1,2}}{ORD}(?:,\s\d{{4}})?\b|\b{MON}\s(?:19|20)\d{{2}}\b"
                        r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b"
                        r"|\b(?:19|20)\d{2}(?:-\d{2})?\b"), False),
    ("LAW_ACT", re.compile(r"\b(?:Article|Section)s?\s\d+(?:-?[A-Za-z])?(?:\(\d+\))?(?:\sof\sthe\s(?:[A-Z][\w-]*\s?){1,6})?"
                           r"|\b(?:[A-Z][\w-]*\s){1,5}(?:Act|Ordinance|Order|Code)(?:,?\s\d{4})?\b"
                           r"|\bConstitution\b"), False),
]


def regex_spans(text):
    out = []
    for label, pat, strong in REGEX_RULES:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            while e > s and text[e - 1] in " .,": e -= 1      # trim trailing space/punct
            if e > s:
                out.append((s, e, label, "regex", strong))
    return out


# ------------------------------------------------------------------ layer 2: gazetteer
EXTRA_COUNTRIES = ("Afghanistan Albania Algeria Argentina Armenia Australia Austria Azerbaijan Bahrain Bangladesh Belarus Belgium "
                   "Brazil Bulgaria Canada Chile China Colombia Cuba Denmark Egypt Ethiopia Finland France Germany Ghana Greece "
                   "Hungary India Indonesia Iran Iraq Ireland Israel Italy Japan Jordan Kazakhstan Kenya Kuwait Kyrgyzstan Lebanon "
                   "Libya Malaysia Mexico Morocco Myanmar Nepal Netherlands Nigeria Norway Oman Palestine Philippines Poland Portugal "
                   "Qatar Romania Russia Rwanda Singapore Somalia Spain Sudan Sweden Switzerland Syria Tajikistan Thailand Tunisia "
                   "Turkey Turkmenistan Uganda Ukraine Uzbekistan Venezuela Vietnam Yemen Zimbabwe").split()
DROP = {"Lions", "Dolphins", "Pakistan A", "Public News", "Special Court", "Customs", "Family Court", "Dawn", "Jazz", "Basant"}
EXTRA = [  # names that the first runs showed as missing / mislabelled
    ("COURT_LEGAL_BODY", ["IHC", "LHC", "SHC", "PHC", "BHC", "apex court", "Apex Court", "Supreme Court Bar Association"]),
    ("GOVERNMENT_BODY", ["Parliament", "SBP", "ISI", "BISP", "PIB", "NACTA", "PEMRA", "Federal Government", "Provincial Assembly",
                         "Election Commission", "Government of Pakistan", "Establishment Division", "Board of Revenue"]),
    ("POLITICAL_PARTY", ["PML", "APML", "PTM", "PDM", "GDA", "Pakistan Tehrik", "N-League", "Muttahida", "Pakistan Tehrik-e-Insaf",
                         "Muslim League", "Awami League"]),
    ("PROVINCE_REGION", ["K-P", "KP", "Khyber-Pakhtunkhwa", "AJK", "FATA", "Fata", "South Punjab", "East Pakistan", "West Pakistan",
                         "East Bengal", "Karachi Division"]),
    ("COUNTRY", ["England", "Britain", "Scotland", "Wales", "South Africa", "New Zealand", "Great Britain", "Sri Lanka"]),
    ("PERSON", ["Trump", "Donald Trump", "Biden", "Joe Biden", "Modi", "Narendra Modi", "Obama", "Putin", "Federer"]),
    ("EVENT", ["PSL", "Pakistan Super League", "Asia Cup", "World Cup", "Cricket World Cup", "Australian Open", "Olympics", "Champions Trophy"]),
    ("MEDIA_OUTLET", ["DawnNews", "Dawn News", "Express 24/7", "Express Media Group", "GNN", "Tribune", "The Express Tribune", "Gazette"]),
    ("SPORTS_TEAM", ["West Indies", "Pakistan Navy", "Pakistan Army", "Pakistan Eagles", "Friends Hockey Club"]),
    ("ORGANIZATION", ["FIFA", "ICC", "UNICEF", "UNHCR", "UNDP", "OIC", "SAARC", "NATO", "OPEC", "BBC Urdu"]),
    ("LANDMARK_FACILITY", ["Lal Masjid", "Thar Desert", "Gaddafi Stadium", "Bilawal House", "Peshawar Stadium", "Orange Line Metro Train Project"]),
]
# ambiguous common words: kept out of the plain gazetteer; spaCy / review decides


def read_gaz(name):
    p = GAZ / f"{name}.txt"
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def build_gazetteer():
    g = {}
    lists = [
        ("COURT_LEGAL_BODY", G.COURT), ("GOVERNMENT_BODY", G.GOVBODY), ("POLITICAL_PARTY", G.PARTY),
        ("EDUCATIONAL_INSTITUTION", G.EDU), ("MEDIA_OUTLET", G.MEDIA), ("SPORTS_TEAM", G.TEAM), ("EVENT", G.EVENT),
        ("LANDMARK_FACILITY", G.LANDMARK), ("ORGANIZATION", G.ORG), ("LAW_ACT", G.LAW_NAMES),
        ("PROVINCE_REGION", G.PROVINCE + read_gaz("provinces") + ["Gilgit Baltistan"]),
        ("COUNTRY", G.COUNTRY + EXTRA_COUNTRIES),
        ("CITY", [c.replace("_", " ") for c in G.CITY] + [d for d in read_gaz("districts") + read_gaz("divisions") if len(d) > 3 and not re.search(r"\d", d)]),
    ]
    lists += EXTRA
    for label, names in lists:
        for n in names:
            if n and n not in DROP and n not in g:
                g[n] = label            # first list wins on conflicts
    return g


def gazetteer_matcher(g):
    names = sorted(g, key=len, reverse=True)
    pat = re.compile(r"(?<![\w-])(" + "|".join(re.escape(n) for n in names) + r")(?![\w-])")
    def run(text):
        return [(m.start(), m.end(), g[m.group(1)], "gazetteer", False) for m in pat.finditer(text)]
    return run


# ------------------------------------------------------------------ layer 3: spaCy (optional)
TITLES = re.compile(r"^(?:Prime Minister|Chief Minister|Chief Justice|Justice|Senator|Minister|President|General|Gen|Dr|Mr|Mrs|Ms|"
                    r"Prof|Professor|Sheikh|Begum|Maulana|Mufti|Captain|Colonel|Sir|Hazrat)\.?\s+", re.I)
ORG_RULES = [  # keyword -> more specific label for spaCy ORG spans
    ("EDUCATIONAL_INSTITUTION", r"University|College|Institute|School|Academy|Madrasa|Campus"),
    ("COURT_LEGAL_BODY", r"Court|Tribunal|Bench|Judicial"),
    ("POLITICAL_PARTY", r"\bParty\b|\bLeague\b|Movement|Tehreek|Jamaat|Awami|Muttahida|\bPTI\b|\bPPP\b|\bMQM\b"),
    ("GOVERNMENT_BODY", r"Ministry|Commission|Authority|Police|Bureau|Assembly|Senate|Department|Agency|Board|Army|Navy|Air Force|Council|Secretariat|Government|Division|Rangers|Force|Cabinet|Directorate"),
    ("MEDIA_OUTLET", r"\bNews\b|Times|Tribune|Radio|\bTV\b|Daily|Herald|Gazette|\bChannel\b|\bJang\b|DawnNews"),
    ("SPORTS_TEAM", r"\bXI\b|Kings|Qalandars|Zalmi|Sultans|Gladiators|\bFC\b|Hockey Club|Cricket Club|\bSquad\b"),
]
# keyword rules are skipped when these words show the span is something else
ORG_EXCLUDE = {"SPORTS_TEAM": r"Organi[sz]ation|Response|Committee|Association|Party|Movement",
               "MEDIA_OUTLET": r"Club|Association|Union"}
ORG_MAX_WORDS = 6                     # keyword rules only trust short names
NOT_ORG = re.compile(r"^(?:M[NP]As?|SHO|DC|DPO|SSP|DSP|IG|DIG|PM|CM|AG|Const|Source|Sources)$")
SPACY_MAP = {"PERSON": "PERSON", "FAC": "LANDMARK_FACILITY", "EVENT": "EVENT", "LAW": "LAW_ACT", "DATE": "DATE",
             "TIME": "TIME", "MONEY": "MONEY", "PERCENT": "PERCENT"}
FACILITY_WORDS = re.compile(r"Masjid|Mosque|Hospital|Stadium|Airport|Station|Chowk|Park|Fort|Canal|Dam|Bridge|Motorway|Railway|Shrine|Darbar", re.I)
FAC_JUNK = re.compile(r"\bSections?\b|\bArticles?\b|Platoon|\bLevel\b|\d{2,}")
DATE_JUNK = re.compile(r"^(?:the )?(?:same )?(?:day|days|week|weeks|month|months|year|years|time|period)$", re.I)  # relative dates like "last year" stay DATE
PARTY_WORDS = re.compile(r"Movement|Insaf|\bParty\b|Tehreek|Jamaat|League|Awami|Qaumi", re.I)
PERSON_NAMES = set(G.SURNAME + G.MALE + G.FEMALE)
# things spaCy tags as entities that are not entities for us (documents, titles, religions, languages ...)
DROP_TEXT = {"Islam", "FIR", "CNIC", "ODI", "State", "Justice", "CJP", "CEC", "CPC", "CMA", "Criminal Petition", "Urdu",
             "Province", "Indus", "Committee", "Govt", "Insaf", "Law", "Capt", "Jo", "Muslim", "Hindu", "English", "Punjabi"}
# bare generic nouns: guidelines say never label "the court", "the government", "the party" ...
GENERIC_WORDS = {"court", "courts", "tribunal", "bench", "commission", "council", "board", "authority", "government",
                 "committee", "university", "college", "school", "party", "cabinet", "department", "ministry", "force",
                 "assembly", "agency", "bureau", "division", "state", "federation", "organisation", "organization"}
ORG_TYPES = {"ORGANIZATION", "GOVERNMENT_BODY", "COURT_LEGAL_BODY", "POLITICAL_PARTY", "EDUCATIONAL_INSTITUTION",
             "MEDIA_OUTLET", "SPORTS_TEAM"}
TIME_JUNK = re.compile(r"\bhours?\b|-hour|\bminutes?\b|overnight|Earlier|night of|\bdays?\b|^(?:AM|PM|am|pm)$", re.I)
EVENT_LAW_GENERIC = {"war", "cup", "games", "act", "constitutional", "amendment", "olympic", "championship", "elections"}


def classify_spacy(ent, gaz):
    lab = _classify(ent, gaz)
    txt = ent.text.strip()
    if lab in ORG_TYPES and txt.lower() in GENERIC_WORDS:
        return None
    return lab


def _classify(ent, gaz):
    t, txt = ent.label_, ent.text.strip()
    if txt in DROP_TEXT:
        return None
    if t in ("ORG", "GPE", "FAC", "NORP") and txt in gaz:      # our curated list outranks spaCy's coarse type
        return gaz[txt]
    if t in ("EVENT", "LAW") and txt.lower() in EVENT_LAW_GENERIC:
        return None
    if t == "TIME":
        return None if TIME_JUNK.search(txt) else "TIME"
    if t == "PERSON":
        if re.search(r"[\d/]", txt) or txt[:1].islower():     # "Form-45", "EOBI/", lower-case junk
            return None
        if txt in gaz or txt in ("Dawn",):               # a known organisation/place/newspaper, not a person
            return None
        if FACILITY_WORDS.search(txt):
            return "LANDMARK_FACILITY"                   # "Lal Masjid" etc.
        return "PERSON"
    if t == "FAC":
        return None if (FAC_JUNK.search(txt) or len(txt.split()) > 7) else "LANDMARK_FACILITY"
    if t == "DATE":
        return None if DATE_JUNK.match(txt) else "DATE"
    if t in SPACY_MAP:
        return SPACY_MAP[t]
    if t == "ORG":
        if NOT_ORG.match(txt) or len(txt) <= 2:
            return None
        if len(txt.split()) <= ORG_MAX_WORDS:
            for lab, kw in ORG_RULES:
                if re.search(kw, txt) and not (lab in ORG_EXCLUDE and re.search(ORG_EXCLUDE[lab], txt)):
                    return lab
        return "ORGANIZATION"
    if t == "GPE":
        lab = gaz.get(txt) or gaz.get(txt.title())      # DATELINES are upper-case: ISLAMABAD -> Islamabad
        if lab in ("COUNTRY", "PROVINCE_REGION", "CITY"):
            return lab
        # unknown place: default CITY, but not for obvious non-places
        if PARTY_WORDS.search(txt) or txt in PERSON_NAMES or (txt.isupper() and len(txt) <= 3):
            return None
        return "CITY"
    return None   # NORP, LOC, CARDINAL, ... not mapped (left to gazetteer / human review)


def spacy_spans_batch(nlp, texts, gaz, batch=256):
    for doc in nlp.pipe(texts, batch_size=batch):
        out = []
        for ent in doc.ents:
            lab = classify_spacy(ent, gaz)
            if not lab:
                continue
            s, e = ent.start_char, ent.end_char
            txt = doc.text[s:e]
            if txt.lower().startswith("the "):
                s += 4
            if lab == "PERSON":                               # courtesy titles are not part of the name
                m = TITLES.match(doc.text[s:e])
                while m:
                    s += m.end(); m = TITLES.match(doc.text[s:e])
            if doc.text[s:e].endswith("'s"):
                e -= 2
            if e > s:
                out.append((s, e, lab, "spacy", False))
        yield out


# ------------------------------------------------------------------ merge
def resolve(spans):
    """Greedy, non-overlapping: strong regex first, then longest, then layer priority."""
    prio = {"regex": 3, "gazetteer": 2, "spacy": 1}
    order = sorted(spans, key=lambda x: (0 if x[4] else 1, -(x[1] - x[0]), -prio[x[3]], x[0]))
    taken, chosen = [], []
    for s, e, lab, origin, strong in order:
        if all(e <= ts or s >= te for ts, te in taken):
            taken.append((s, e)); chosen.append((s, e, lab, origin))
    chosen = sorted(chosen)
    # agreement = how many different layers independently proposed exactly this span+label
    agree = [len({o for s2, e2, l2, o, _ in spans if (s2, e2, l2) == (s, e, lab)}) for s, e, lab, _ in chosen]
    # conflicts = proposals that overlap a chosen span but differ in boundary or label (a layer disagreed)
    keys = {(s, e, lab) for s, e, lab, _ in chosen}
    conflicts = sum(1 for s2, e2, l2, o, _ in spans
                    if (s2, e2, l2) not in keys and any(s2 < e and e2 > s for s, e, _, _ in chosen))
    return chosen, agree, conflicts


# ------------------------------------------------------------------ outputs
def ls_config():
    labels = "\n".join(f'    <Label value="{l}"/>' for l in ENTITY_TYPES)
    return f'<View>\n  <Labels name="label" toName="text">\n{labels}\n  </Labels>\n  <Text name="text" value="$text"/>\n</View>\n'


def ls_task(row):
    result = [{"from_name": "label", "to_name": "text", "type": "labels",
               "value": {"start": s, "end": e, "text": row["text"][s:e], "labels": [lab]}}
              for s, e, lab, _ in row["spans"]]
    return {"data": {"text": row["text"], "id": row["id"], "source": row["source"], "group": row["group"],
                     "split": row.get("split", "review")},
            "predictions": [{"model_version": "preannotation-v1", "result": result}]}


def stratified_sample(rows, n, rnd):
    """Pick n sentences covering every label (round-robin over labels, rarest first)."""
    by_label = {l: [] for l in ENTITY_TYPES}
    for r in rows:
        for lab in {sp[2] for sp in r["spans"]}:
            by_label[lab].append(r)
    for v in by_label.values():
        rnd.shuffle(v)
    chosen, ids = [], set()
    labels = sorted(ENTITY_TYPES, key=lambda l: len(by_label[l]))
    while len(chosen) < n and any(by_label.values()):
        for lab in labels:
            while by_label[lab]:
                r = by_label[lab].pop()
                if r["id"] not in ids:
                    ids.add(r["id"]); chosen.append(r); break
            if len(chosen) >= n:
                break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="en_core_web_lg")
    ap.add_argument("--no-spacy", action="store_true")
    ap.add_argument("--test-n", type=int, default=200, help="random sentences for the gold TEST set")
    ap.add_argument("--review-n", type=int, default=200, help="sentences for the training/validation review sample")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N sentences")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    global OUT
    if a.no_spacy or a.limit:                  # partial/debug runs must never overwrite the real Colab output
        OUT = ROOT / "data" / "annotated_debug"
        print(f"[debug run -> writing to {OUT}, not data/annotated]")
    OUT.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(SENT, encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    gaz = build_gazetteer()
    gaz_run = gazetteer_matcher(gaz)
    print(f"{len(rows)} sentences; gazetteer entries: {len(gaz)}")

    nlp = None
    if not a.no_spacy:
        import spacy
        try:
            nlp = spacy.load(a.model, disable=["lemmatizer"])
        except OSError:
            raise SystemExit(f"spaCy model missing: python -m spacy download {a.model}   (or use --no-spacy)")
        print("spaCy model:", a.model)
        sp_iter = spacy_spans_batch(nlp, [r["text"] if "text" in r else r["sentence"] for r in rows], gaz)
    else:
        sp_iter = iter([[] for _ in rows])

    out_rows = []
    for r, sp in zip(rows, sp_iter):
        text = r["sentence"]
        spans, agree, conflicts = resolve(regex_spans(text) + gaz_run(text) + sp)
        tokens, tags = spans_to_bio(text, [(s, e, l) for s, e, l, _ in spans])
        out_rows.append(dict(id=f"real{r['id']}", text=text, tokens=tokens, ner_tags=tags, spans=spans,
                             agree=agree, conflicts=conflicts,
                             source=r["source"], tier=r["tier"], group=r["group"]))

    with open(OUT / "preannotated.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 1) TEST sample: purely random (incl. sentences with no entity) => unbiased F1 later.
    #         Whole documents are held out: every sentence of a test document is excluded from review/training.
    test = rnd.sample(out_rows, min(a.test_n, len(out_rows)))
    test_groups = {r["group"] for r in test}
    json.dump(sorted(test_groups), open(OUT / "test_groups.json", "w", encoding="utf-8"))
    pool = [r for r in out_rows if r["group"] not in test_groups]

    # ---- 2) REVIEW sample (for training/validation): half rare-label coverage, half "uncertain" sentences
    #         uncertain = layers disagreed (conflicts) or a span rests on spaCy alone (agree == 1, origin spacy)
    half = a.review_n // 2
    strat = stratified_sample(pool, half, rnd)
    ids = {r["id"] for r in strat}
    uncertain = [r for r in pool if r["id"] not in ids and
                 (r["conflicts"] > 0 or any(g == 1 and sp[3] == "spacy" for g, sp in zip(r["agree"], r["spans"])))]
    rnd.shuffle(uncertain)
    review = strat + uncertain[:a.review_n - len(strat)]
    for r in test: r["split"] = "test"
    for r in review: r["split"] = "review"

    tasks = [ls_task(r) for r in test + review]       # test first: if time runs out, the gold test set is done
    (OUT / "labelstudio_tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "labelstudio_config.xml").write_text(ls_config(), encoding="utf-8")
    sample = test + review
    n_conf = sum(1 for r in out_rows if r["conflicts"] > 0)
    print(f"test (random): {len(test)} from {len(test_groups)} held-out documents; review: {len(review)}; "
          f"sentences with layer conflicts overall: {n_conf}")

    cnt = Counter(sp[2] for r in out_rows for sp in r["spans"])
    org = Counter((sp[2], sp[3]) for r in out_rows for sp in r["spans"])
    n_with = sum(1 for r in out_rows if r["spans"])
    print(f"\nsentences with >=1 entity: {n_with}/{len(out_rows)}")
    print(f"{'label':24}{'total':>7}{'regex':>8}{'gaz':>7}{'spacy':>7}")
    for lab in ENTITY_TYPES:
        print(f"{lab:24}{cnt[lab]:7}{org[(lab, 'regex')]:8}{org[(lab, 'gazetteer')]:7}{org[(lab, 'spacy')]:7}")
    print(f"\nLabel Studio tasks: {len(sample)} sentences -> labelstudio_tasks.json (first {len(test)} = test, rest = review)")


if __name__ == "__main__":
    main()

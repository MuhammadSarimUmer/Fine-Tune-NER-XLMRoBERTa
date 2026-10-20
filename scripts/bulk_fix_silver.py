"""Bulk-fix the automatic (silver) labels using what the human review showed, then decide which
silver sentences are trustworthy enough to train on.

Input : data/annotated/preannotated.jsonl      (draft: spans with origin layer)
        data/annotated/corrected_final.jsonl   (the 400 human-verified sentences)
Output: data/annotated/silver.jsonl            (every NON-human sentence, fixed, with trusted flag)

Every rule below is justified by a REVIEW-split error pattern (never by the test split).
Philosophy: a sentence with an unlabelled real entity teaches the model "this entity is O", which is
worse than leaving the sentence out. So untrusted sentences are dropped from training, not patched.
"""
import json, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize_conventions import DURATION, normalize
from tokenization import bio_to_spans, spans_to_bio

ROOT = Path(__file__).resolve().parent.parent
ANN = ROOT / "data" / "annotated"

# 1. names the draft missed (all from review-split errors, generalised to their relatives)
GAZ_ADD = {
    "GOVERNMENT_BODY": ["BOI", "Board of Investment", "CCI", "Council of Common Interests", "MPC", "Monetary Policy Committee",
                        "PCB", "Pakistan Cricket Board", "PSX", "Pakistan Stock Exchange", "NIH", "WAPDA", "AGPR", "USC",
                        "Utility Stores Corporation", "SECP", "OGRA", "NEPRA", "PEMRA", "PTA", "FBR", "SBP", "NADRA", "ISPR",
                        "Pakistan Academy of Letters", "Prime Minister's Office", "Prime Minister's Secretariat",
                        "Parliamentary Subcommittee on Electoral Reforms"],
    "POLITICAL_PARTY": ["PML(N)", "PML-N", "PML-F", "PML-Q", "PIB", "SUP", "ITP", "Baath party", "Ba'ath party",
                        "Grand Democratic Alliance", "Pakistan Tehreek-i-Insaf", "Pakistan Tehrik-e-Insaf", "Istehkam-e-Pakistan Party",
                        "Awami National Party", "Jamaat-e-Islami", "JUI-F", "JI", "TLP", "BNP", "BAP", "PkMAP", "PSP", "MMA"],
    # deliberately NOT added: IPP (also "independent power producers"), NAP (also "National Action Plan"),
    # GDA / NP (ambiguous abbreviations): a wrong gazetteer hit would create false labels in many sentences
    "EDUCATIONAL_INSTITUTION": ["LUMS", "Lums", "NEDUET", "NED University", "SZABIST", "UCP", "UMT", "DUHS", "Ziauddin University",
                                "GIKI", "NUST", "COMSATS", "Habib University", "IBA", "Lahore Grammar School", "Law College Quetta"],
    "PROVINCE_REGION": ["Kashmir", "Azad Kashmir", "Azad Jammu and Kashmir", "Jammu and Kashmir", "AJK", "Gilgit-Baltistan",
                        "Gilgit Baltistan", "Islamabad Capital Territory", "FATA", "Thar", "Cholistan", "Potohar", "Middle East",
                        "Kalat", "Makran", "Kurram Agency", "Orakzai Agency", "Khyber Agency", "Mohmand Agency", "Bajaur Agency",
                        "Khyber-Pakhtunkhwa", "Khyber Pakhtunkhwa", "KPK", "K-P"],
    "COUNTRY": ["Islamic Republic of Pakistan", "United Arab Emirates", "UAE", "US", "USA", "UK"],
    "MEDIA_OUTLET": ["Express Tribune", "Geo Super", "Geo News", "ARY News", "Dawn News", "Samaa", "Reuters"],
}
# 2. generic words / titles / junk that the draft labelled but the human removed (exact text)
DROP_TEXT = {"IGP", "CEC", "AAG", "SHO", "Inspector General of Police", "Coming Into Force", "Society", "Local Commission",
             "Contesting Candidate", "Fourth Schedule", "Provincial Assemblies", "Company", "QRF", "TNE", "CoD", "TCS"}
# 3. spaCy layers whose measured precision on the review split is too low to trust (see origin_precision.py)
UNTRUSTED_SPACY = {"ORGANIZATION", "GOVERNMENT_BODY", "EDUCATIONAL_INSTITUTION", "EVENT", "LANDMARK_FACILITY", "MONEY"}

TEAM_COUNTRIES = ["Pakistan", "India", "England", "Australia", "New Zealand", "South Africa", "Sri Lanka", "Bangladesh",
                  "West Indies", "Zimbabwe", "Afghanistan", "Scotland", "Ireland", "Netherlands", "Kenya", "Nepal", "Oman", "UAE", "USA"]
# strong sport cues only (an earlier, broader list such as "final"/"series"/"match" turned "Pakistan" into a
# team in political sentences; found by spot-checking the output)
MATCH_CUES = re.compile(r"\b(?:\d+\s+runs?|run chase|wickets?|innings|ODIs?|T20Is?|T20s?|Twenty20|Tests?|[Cc]ricket|[Hh]ockey|"
                        r"[Ss]quash|[Ff]ootball|bowlers?|batsm[ae]n|batters?|World Cup|Champions Trophy)\b")
TOD = re.compile(r"\d|morning|evening|night|noon|midnight|afternoon|dawn|dusk|o'clock", re.I)
DAY_TOD = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|yesterday|tomorrow)\s+"
                     r"(morning|afternoon|evening|night)$", re.I)
LAW_YEAR = re.compile(r"[A-Z][\w()\-]*(?:\s+(?:[A-Z][\w()\-]*|of|and|the|for)){0,8}\s+"
                      r"(?:Act|Ordinance|Bill|Order|Rules|Regulations|Code),?\s+(?:19|20)\d\d")
EVENT_YEAR = re.compile(r"[A-Z][\w\-]*(?:\s+[A-Z][\w\-]*){0,4}\s+(?:Trophy|Cup|Championship|Games)\s+(?:19|20)\d\d"
                        r"|(?:19|20)\d\d\s+(?:general\s+)?elections?")


def word_re(name):
    return re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w$-])")        # "US$" is money, not a country


GAZ_RE = [(lab, n, word_re(n)) for lab, names in GAZ_ADD.items() for n in sorted(names, key=len, reverse=True)]
TEAM_RE = [word_re(c) for c in TEAM_COUNTRIES]


def overlap(a, b):
    return not (a[1] <= b[0] or a[0] >= b[1])


def add_span(spans, new, force=False):
    """spans: dict (s,e,label)->origin. A new span replaces overlapped spans (unless they came from a
    curated layer and are longer, in which case we keep them)."""
    hits = [k for k in spans if overlap(k, new)]
    if not force and any(spans[k] in ("gazetteer", "regex", "fix") and (k[1] - k[0]) > (new[1] - new[0]) for k in hits):
        return
    for k in hits:
        spans.pop(k)
    spans[new] = "fix"


def fix(text, draft_spans):
    """draft_spans: [(s,e,label,origin)] -> (dict (s,e,label)->origin, notes)"""
    notes = []
    spans = {}
    for s, e, lab, origin in draft_spans:
        t = text[s:e]
        if t in DROP_TEXT:
            notes.append("drop-generic"); continue
        if lab == "COURT_LEGAL_BODY" and t[:1].islower():
            notes.append("drop-lowercase-court"); continue
        if lab == "TIME" and origin == "spacy":
            if DURATION.fullmatch(t):
                lab = "DATE"; origin = "fix"
            elif not TOD.search(t):
                notes.append("drop-junk-time"); continue
        if lab == "DATE" and re.fullmatch(r"(?:19|20)\d\d", t) and origin == "regex":
            pass                                       # years inside names are merged below
        spans[(s, e, lab)] = origin

    for s, e, lab in list(spans):                      # "Wednesday night" -> DATE + TIME
        m = DAY_TOD.match(text[s:e])
        if m and lab in ("DATE", "TIME"):
            del spans[(s, e, lab)]
            ds, de = s, s + len(m.group(1))
            spans[(ds, de, "DATE")] = "fix"; spans[(e - len(m.group(2)), e, "TIME")] = "fix"; notes.append("split-day-time")

    for rx, lab in ((LAW_YEAR, "LAW_ACT"), (EVENT_YEAR, "EVENT")):    # year belongs inside the name
        for m in rx.finditer(text):
            add_span(spans, (m.start(), m.end(), lab), force=True); notes.append("merge-year-" + lab)

    for lab, name, rx in GAZ_RE:
        for m in rx.finditer(text):
            add_span(spans, (m.start(), m.end(), lab)); notes.append("gaz-add")

    if MATCH_CUES.search(text):                         # national team in a match context
        for rx in TEAM_RE:
            for m in rx.finditer(text):
                k = (m.start(), m.end(), "COUNTRY")
                if k in spans:
                    del spans[k]; spans[(k[0], k[1], "SPORTS_TEAM")] = "fix"; notes.append("country->team")

    # remove years left over inside a longer date/name span, then convention normalisation
    origin_of = dict(spans)
    final = normalize(text, list(spans), None)
    out = {}
    for s, e, lab in final:
        out[(s, e, lab)] = origin_of.get((s, e, lab)) or next(
            (o for (a, b, l), o in origin_of.items() if l == lab and a <= s < b), "fix")
    # drop pieces fully nested in another span (flat entities only)
    keys = sorted(out, key=lambda k: (k[0], -(k[1] - k[0])))
    flat, last_end = {}, -1
    for k in keys:
        if k[0] >= last_end:
            flat[k] = out[k]; last_end = k[1]
    return flat, notes


def trusted(spans):
    bad = [k for k, o in spans.items() if o == "spacy" and k[2] in UNTRUSTED_SPACY]
    return not bad, bad


def main():
    human = {}
    for line in open(ANN / "corrected_final.jsonl", encoding="utf-8"):
        r = json.loads(line); human[r["id"]] = r
    out, stats, eval_rows = [], Counter(), []
    for line in open(ANN / "preannotated.jsonl", encoding="utf-8"):
        r = json.loads(line)
        spans, notes = fix(r["text"], r["spans"])
        ok, bad = trusted(spans)
        if r["id"] in human:                            # evaluation only; human version is never replaced
            eval_rows.append((human[r["id"]], spans, ok)); continue
        toks, tags = spans_to_bio(r["text"], list(spans))
        out.append(dict(id=r["id"], text=r["text"], tokens=toks, ner_tags=tags, source=r["source"], group=r["group"],
                        trusted=ok, n_spans=len(spans), untrusted_spans=[list(b) for b in bad]))
        stats["trusted" if ok else "untrusted"] += 1
        stats["has_entity" if spans else "no_entity"] += 1
        for n in notes: stats["fix:" + n] += 1
    with open(ANN / "silver.jsonl", "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"silver sentences: {len(out)}; "); [print(f"  {k:28}{v}") for k, v in sorted(stats.items())]

    # how good is the fixed silver? sentence-level exact match vs human, per split and trusted/untrusted
    print("\nfixed draft vs human (sentence exact match = every span right)")
    for split in ("review", "test"):
        for flag in (True, False):
            rows = [(h, sp) for h, sp, ok in eval_rows if h["split"] == split and ok == flag]
            if rows:
                exact = sum(set(bio_to_spans(h["text"], h["ner_tags"])) == set(sp) for h, sp in rows)
                print(f"  {split:7} {'trusted  ' if flag else 'untrusted'} {len(rows):4} sentences, exact {exact / len(rows):.2f}")
    tp, fp, fn = Counter(), Counter(), Counter()
    for h, sp, ok in eval_rows:
        hs, ds = set(bio_to_spans(h["text"], h["ner_tags"])), set(sp)
        for s in hs & ds: tp[(h["split"], s[2])] += 1
        for s in ds - hs: fp[(h["split"], s[2])] += 1
        for s in hs - ds: fn[(h["split"], s[2])] += 1
    print("\nspan P/R of fixed draft (review = tuned on, test = never tuned on)")
    labs = sorted({k[1] for k in list(tp) + list(fp) + list(fn)})
    print(f"{'label':24}{'rev P':>7}{'rev R':>7}{'test P':>8}{'test R':>8}")
    for lab in labs:
        row = []
        for sp in ("review", "test"):
            t, f_, n_ = tp[(sp, lab)], fp[(sp, lab)], fn[(sp, lab)]
            row += [t / (t + f_) if t + f_ else float("nan"), t / (t + n_) if t + n_ else float("nan")]
        print(f"{lab:24}{row[0]:7.2f}{row[1]:7.2f}{row[2]:8.2f}{row[3]:8.2f}")


if __name__ == "__main__":
    main()

"""Which DATE spans in the silver data are junk? Evidence comes from REVIEW-split human labels + silver counts only."""
import json, re, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tokenization import bio_to_spans
ANN = Path(__file__).resolve().parent.parent / "data" / "annotated"
rd = lambda p: [json.loads(l) for l in open(ANN / p, encoding="utf-8")]

review = [r for r in rd("corrected_final.jsonl") if r["split"] == "review"]
extra_t = rd("corrected_extra_final.jsonl")           # gold, but NOT used for decisions; only printed at the end
silver = rd("silver.jsonl")

def dates(r):
    return [r["text"][s:e] for s, e, l in bio_to_spans(r["text"], r["ner_tags"]) if l == "DATE"]

print("silver DATE spans:", sum(len(dates(r)) for r in silver))
c = Counter(t.lower() for r in silver for t in dates(r))
print("most common silver DATE texts:", c.most_common(40))
vague = re.compile(r"^(?:the |a |an )?(?:past|last|recent|next|coming|previous|following|first|second|third|few|several|many|some)?\s*(?:period|time|weeks?|days?|months?|years?|decades?|centur(?:y|ies)|moment|date|future|past|present)$", re.I)
print("vague-looking silver DATE spans:", sum(1 for r in silver for t in dates(r) if vague.match(t)))
print("   examples:", Counter(t.lower() for r in silver for t in dates(r) if vague.match(t)).most_common(25))
print("\nREVIEW (human) DATE spans that look vague:", [t for r in review for t in dates(r) if vague.match(t)])
print("REVIEW sentences containing 'recent weeks' / 'period' unlabelled:",
      sum(1 for r in review if re.search(r"\brecent (?:weeks|days|months|years)\b|\bperiod\b", r["text"]) and not any(vague.match(t) for t in dates(r))))
# year inside case numbers / citations
case_year = re.compile(r"\b\d{4}\s+(?:SCMR|PLD|CLC|YLR|PCrLJ|MLD|CLD|PTD|GBLR)\b|\b(?:PLD|SCMR|CLC|YLR)\s+\d{4}\b|\bNo\.?\s*\d+[-/]?\w*\s+of\s+(?:19|20)\d\d\b|\b[A-Z]{1,4}\.?[A-Z]?\.?\s*(?:No\.?)?\s*\d+(?:[-/]\w+)?\s*/\s*(?:19|20)\d\d\b", re.I)
bad = [(r["text"][:100], t) for r in silver for t in dates(r) if re.fullmatch(r"(?:19|20)\d\d", t) and case_year.search(r["text"])]
print("\nsilver 4-digit DATE years in sentences with case/citation patterns:", len(bad)); print(bad[:6])
print("REVIEW human years labelled DATE in such sentences:",
      sum(1 for r in review for t in dates(r) if re.fullmatch(r"(?:19|20)\d\d", t) and case_year.search(r["text"])))
print("\n(extra_test gold, for information only) vague gold DATE:", [t for r in extra_t for t in dates(r) if vague.match(t)])

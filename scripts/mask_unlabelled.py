"""Round 2: stop teaching the model that unlabelled capitalised names are 'O'.

Problem: in the machine-labelled (silver) sentences, real organisations / bodies / landmarks that the draft missed are labelled O,
so the model learns "such names are not entities" (low recall for ORGANIZATION, GOVERNMENT_BODY, LANDMARK_FACILITY, LAW_ACT).
Fix: in silver sentences, tokens that are O but look like part of a name are MASKED (-100): the loss ignores them, so they are neither
rewarded as entities nor punished as O. Rules (deliberately simple, checked on the REVIEW split only - see mask_eval.py):
  * a run of 2+ capitalised tokens labelled O (connectors 'of','and','for' allowed between them), or
  * a single ALL-CAPS token of 2-6 letters labelled O (abbreviations such as OGRA, NCCS)
  * not: sentence-start words, months/weekdays, titles and honorifics, pronouns, common capitalised function words
  * sentences that are mostly ALL-CAPS (headings) are skipped
"""
import re

CONNECT = {"of", "and", "for", "the", "de", "al", "e", "ul", "&"}
NOT_NAME = {w.lower() for w in """
The A An In On At By For With From To Of And But Or If As He She It They We You I His Her Its Their Our This That These Those There
However According Meanwhile Also Following After Before During While When Where Who Which What Why How Yesterday Today Tomorrow
January February March April May June July August September October November December
Monday Tuesday Wednesday Thursday Friday Saturday Sunday
Mr Mrs Ms Dr Prof Justice Judge Sir Begum Maulana Hafiz Syed Sardar Chaudhry Raja Mian Khan Prime Minister Chief President Chairman
Secretary Senator Governor Captain General Colonel Major Lieutenant Inspector Commissioner Director Advocate Counsel Respondent
Respondents Petitioner Petitioners Appellant Appellants Section Article Clause Rule Order Act Ordinance Schedule Para Paragraph Chapter
PM IGP CJP CEC DIG SSP SP DPO AAG AG GDP NA PA MNA MNAs MPA MPAs LGBT CNIC FIR CPEC TV Rs PKR USD No Nos Vs Versus Page Pages Table Figure Appendix Annexure Exhibit Case Cases Appeal Petition Civil Criminal Supreme High
""".split()}
TOK = re.compile(r"^[A-Z][\w\-.']*$")


def _cap(t):
    return bool(TOK.match(t)) and t.lower() not in NOT_NAME and any(c.isalpha() for c in t)


def _abbr(t):
    return t.isalpha() and t.isupper() and 2 <= len(t) <= 6 and t.lower() not in NOT_NAME


def unlabelled_name_mask(tokens, tags):
    """Return the set of token indices (labelled O) that look like part of an unlabelled name."""
    n = len(tokens)
    caps = sum(1 for t in tokens if t.isupper() and len(t) > 1)
    if n and caps / n > 0.5: return set()                       # heading in capitals
    out, i = set(), 0
    while i < n:
        if tags[i] != "O" or not _cap(tokens[i]):
            i += 1; continue
        run, j = [i], i + 1
        while j < n:
            if tags[j] == "O" and _cap(tokens[j]):
                run.append(j); j += 1
            elif (tags[j] == "O" and tokens[j].lower() in CONNECT and j + 1 < n and tags[j + 1] == "O" and _cap(tokens[j + 1])
                  and len(run) >= 1):
                run += [j, j + 1]; j += 2
            else:
                break
        caplen = sum(1 for k in run if _cap(tokens[k]))
        if caplen >= 2 or (caplen == 1 and _abbr(tokens[run[0]])):
            # a lone sentence-first capitalised word is not enough evidence
            if not (caplen == 1 and run[0] == 0):
                out.update(run)
        i = j if j > i else i + 1
    return out

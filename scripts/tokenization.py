"""Shared word tokenizer so synthetic and real (annotated) data are tokenized identically.

Rules (kept simple on purpose):
- numbers with internal , . : / - stay ONE token: 1,200  5.63  3:30  5/10/2024  2024-25  35202-1234567-1  0300-1234567
- words keep internal - and . : Minar-e-Pakistan, PML-N, Rs5.63
- possessive 's is split off (so it is never part of an entity)
- every other punctuation mark is its own token (%, comma, quotes, brackets ...)
"""
import re

TOKEN_RE = re.compile(r"\d+(?:[,.:/\-]\d+)*|\w+(?:[-.]\w+)*|'s|[^\w\s]")


def tokenize(text):
    return TOKEN_RE.findall(text)


def spans_to_bio(text, spans):
    """Convert character spans [(start, end, label), ...] to (tokens, bio_tags).

    A token takes a span's label if it overlaps the span. Spans must not overlap each other.
    """
    toks = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    tags = ["O"] * len(toks)
    for s, e, lab in sorted(spans):
        first = True
        for i, (_, ts, te) in enumerate(toks):
            if ts < e and te > s:
                tags[i] = ("B-" if first else "I-") + lab
                first = False
    return [t for t, _, _ in toks], tags


def bio_to_spans(text, tags):
    """Inverse of spans_to_bio: BIO tags -> [(start, end, label)] character spans."""
    toks = [(m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
    assert len(toks) == len(tags)
    spans, cur = [], None
    for (ts, te), tag in zip(toks, tags):
        if tag.startswith("B-") or (tag.startswith("I-") and (cur is None or cur[2] != tag[2:])):
            if cur: spans.append(tuple(cur))
            cur = [ts, te, tag[2:]]
        elif tag.startswith("I-"):
            cur[1] = te
        else:
            if cur: spans.append(tuple(cur)); cur = None
    if cur: spans.append(tuple(cur))
    return spans


if __name__ == "__main__":
    print(tokenize("Rs. 1,200 for Imran Khan's CNIC 35202-1234567-1 at 3:30 PM, up 15%."))

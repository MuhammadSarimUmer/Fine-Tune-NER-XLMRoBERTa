"""Summarise errors_*.txt written by train_ner.py: what kinds of token mistakes, and which label pairs get confused.

Usage: python scripts/error_summary.py trained_reports/v1_xlmr_base [more folders...]
Token-level categories (B-/I- prefix ignored except for 'boundary'):
  missed      gold = entity,  pred = O
  spurious    gold = O,       pred = entity
  confusion   gold = X,       pred = Y (different types)
  boundary    same type, but B-/I- differs (entity starts/ends in the wrong place)
"""
import re, sys
from collections import Counter
from pathlib import Path

LINE = re.compile(r"^\s+(\S+)\s+gold=(\S+)\s+pred=(\S+)")


def parse(path):
    sents, cur = [], None
    for line in open(path, encoding="utf-8"):
        m = LINE.match(line)
        if m:
            if cur is not None: cur["tok"].append(m.groups())
        elif line.strip():
            cur = {"text": line.strip(), "tok": []}; sents.append(cur)
    return sents


def summarise(path):
    sents = parse(path)
    cat, missed, spurious, conf = Counter(), Counter(), Counter(), Counter()
    for s in sents:
        for tok, g, p in s["tok"]:
            gt, pt = g[2:], p[2:]
            if g == "O" and p != "O": cat["spurious"] += 1; spurious[pt] += 1
            elif g != "O" and p == "O": cat["missed"] += 1; missed[gt] += 1
            elif gt != pt: cat["confusion"] += 1; conf[(gt, pt)] += 1
            else: cat["boundary"] += 1
    out = [f"{path.name}: {len(sents)} sentences with at least one error, {sum(cat.values())} wrong tokens"]
    out.append("  token errors by kind: " + ", ".join(f"{k} {v}" for k, v in cat.most_common()))
    out.append("  missed (gold entity predicted O), by gold label: " + ", ".join(f"{k} {v}" for k, v in missed.most_common(8)))
    out.append("  spurious (gold O predicted entity), by predicted label: " + ", ".join(f"{k} {v}" for k, v in spurious.most_common(8)))
    out.append("  most common confusions gold->pred: " + ", ".join(f"{a}->{b} {n}" for (a, b), n in conf.most_common(10)))
    return "\n".join(out)


if __name__ == "__main__":
    for folder in sys.argv[1:]:
        f = Path(folder)
        text = "\n\n".join(summarise(f / n) for n in ("errors_test.txt", "errors_extra_test.txt") if (f / n).exists())
        print(f"=== {folder} ===\n{text}\n")
        (f / "error_summary.txt").write_text(text, encoding="utf-8")

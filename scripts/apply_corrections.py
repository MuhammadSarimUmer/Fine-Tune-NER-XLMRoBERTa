"""Apply hand-written corrections (data/annotated_work/corr_*.json) to Label Studio tasks.

Correction format per sentence:  {"i": index, "r": [[text, LABEL, optional_occurrence]], "a": [[text, LABEL]]}
  r = remove the draft span with this text+label (occurrence n among identical draft spans; default: all)
  a = add a span for every occurrence of text not already covered; an added span REPLACES any
      overlapping draft span (so a longer correct span wins over shorter wrong ones)
  A needle starting with "re:" is a regular expression (used for odd characters such as soft hyphens).

Default run (first 400 sentences):
  python scripts/apply_corrections.py
Extra weak-class batch:
  python scripts/apply_corrections.py --tasks data/annotated/labelstudio_tasks_extra.json \
      --corr "extra_corr_*.json" --out labelstudio_tasks_extra_assistant.json --draft-name preannotation-v2

Output: predictions[0] = corrected version (model_version "assistant-v1") <- what the human reviews
        predictions[1] = the original automatic draft, kept to measure its accuracy
"""
import argparse, glob, json, re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = ROOT / "data" / "annotated" / "labelstudio_tasks.json"
WORK = ROOT / "data" / "annotated_work"


def spans_of(task):
    return sorted((r["value"]["start"], r["value"]["end"], r["value"]["labels"][0]) for r in task["predictions"][0]["result"])


def result_of(text, spans):
    return [{"from_name": "label", "to_name": "text", "type": "labels",
             "value": {"start": s, "end": e, "text": text[s:e], "labels": [lab]}} for s, e, lab in sorted(spans)]


def occurrences(text, needle):
    if needle.startswith("re:"):
        return [(m.start(), m.end()) for m in re.finditer(needle[3:], text)]
    left = r"(?<!\w)" if re.match(r"\w", needle) else ""
    right = r"(?!\w)" if re.search(r"\w$", needle) else ""
    return [(m.start(), m.end()) for m in re.finditer(left + re.escape(needle) + right, text)]


def matches(needle, span_text):
    if needle.startswith("re:"):
        return re.fullmatch(needle[3:], span_text) is not None
    return span_text == needle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(TASKS))
    ap.add_argument("--corr", default="corr_*.json", help="glob inside data/annotated_work")
    ap.add_argument("--out", default="labelstudio_tasks_assistant.json")
    ap.add_argument("--draft-name", default="preannotation-v1")
    a = ap.parse_args()

    tasks = json.load(open(a.tasks, encoding="utf-8"))
    corr = {}
    for f in sorted(glob.glob(str(WORK / a.corr))):
        for c in json.load(open(f, encoding="utf-8")):
            corr.setdefault(c["i"], []).append(c)

    problems, stats = [], Counter()
    for i, t in enumerate(tasks):
        text = t["data"]["text"]
        draft = spans_of(t)
        spans = list(draft)
        for c in corr.get(i, []):
            for item in c.get("r", []):
                needle, lab = item[0], item[1]
                occ = item[2] if len(item) > 2 else None
                hits = [sp for sp in spans if sp[2] == lab and matches(needle, text[sp[0]:sp[1]])]
                if not hits:
                    problems.append((i, "remove not found", needle, lab)); continue
                if occ:
                    hits = [hits[occ - 1]] if occ <= len(hits) else []
                for sp in hits:
                    spans.remove(sp); stats["removed"] += 1
            for needle, lab in c.get("a", []):
                occs = occurrences(text, needle)
                if not occs:
                    problems.append((i, "add not found", needle, lab)); continue
                for s, e in occs:
                    if any(sp[0] <= s and e <= sp[1] and (sp[1] - sp[0]) > (e - s) for sp in spans):
                        problems.append((i, "skipped add (inside longer span)", text[s:e], lab)); continue   # never cut a longer existing span (e.g. "Pakistan" inside "Pakistan Cricket Board")
                    spans = [sp for sp in spans if sp[1] <= s or sp[0] >= e]      # added span replaces overlaps
                    spans.append((s, e, lab)); stats["added"] += 1
        t["predictions"] = [{"model_version": "assistant-v1", "result": result_of(text, spans)},
                            {"model_version": a.draft_name, "result": result_of(text, draft)}]
        if sorted(spans) != draft:
            stats["sentences_changed"] += 1

    out = ROOT / "data" / "annotated" / a.out
    out.write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tasks)} tasks -> {out.name}; corrections for {len(corr)} sentences; {dict(stats)}")
    for p in problems:
        print("PROBLEM", p)


if __name__ == "__main__":
    main()

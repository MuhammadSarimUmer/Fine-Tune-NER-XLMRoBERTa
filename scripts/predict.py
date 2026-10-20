"""Run the trained NER model on new text and print/save the entities.

Examples (Colab or any machine with torch + transformers; model folder = the `best/` folder from training):
  python scripts/predict.py --model outputs/xlmr_v3/best --text "Imran Khan met Shehbaz Sharif in Lahore on 5 March 2024."
  python scripts/predict.py --model outputs/xlmr_v3/best --input my_sentences.txt --out predictions.jsonl
Input file: .txt (one text per line) or .jsonl (each line {"text": "..."} or {"id": ..., "text": "..."}).
Output (.jsonl, one line per input): {"id", "text", "entities": [{"text","label","start","end"}], "tokens", "tags"}
`start`/`end` are character offsets into the original text. Long texts are processed in windows of 100 words.
"""
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tokenization import TOKEN_RE
from labels import ID2LABEL

WINDOW = 100


def fix_bio(tags):
    out, prev = [], "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]: t = "B-" + t[2:]
        out.append(t); prev = t
    return out


def tags_to_entities(text, offsets, tags):
    """BIO tags per word + word character offsets -> list of entity dicts."""
    ents, cur = [], None
    for (s, e), t in zip(offsets, tags):
        if t.startswith("B-") or (t.startswith("I-") and (cur is None or cur["label"] != t[2:])):
            if cur: ents.append(cur)
            cur = {"label": t[2:], "start": s, "end": e}
        elif t.startswith("I-"):
            cur["end"] = e
        else:
            if cur: ents.append(cur); cur = None
    if cur: ents.append(cur)
    for x in ents: x["text"] = text[x["start"]:x["end"]]
    return [{"text": x["text"], "label": x["label"], "start": x["start"], "end": x["end"]} for x in ents]


def load_inputs(a):
    if a.text: return [{"id": 0, "text": a.text}]
    rows, p = [], Path(a.input)
    for i, line in enumerate(open(p, encoding="utf-8")):
        line = line.rstrip("\n")
        if not line.strip(): continue
        if p.suffix == ".jsonl":
            d = json.loads(line); rows.append({"id": d.get("id", i), "text": d["text"]})
        else:
            rows.append({"id": i, "text": line})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder with the trained model (outputs/<run>/best)")
    ap.add_argument("--text"); ap.add_argument("--input"); ap.add_argument("--out")
    ap.add_argument("--max_len", type=int, default=192)
    a = ap.parse_args()
    if not a.text and not a.input: ap.error("give --text or --input")

    import numpy as np, torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model).to(dev).eval()

    results = []
    for row in load_inputs(a):
        text = row["text"]
        words = [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]
        tags = []
        for w0 in range(0, len(words), WINDOW):
            chunk = [w for w, _, _ in words[w0:w0 + WINDOW]]
            enc = tok(chunk, is_split_into_words=True, truncation=True, max_length=a.max_len, return_tensors="pt").to(dev)
            with torch.no_grad():
                logits = model(**enc).logits[0].cpu().numpy()
            chunk_tags, prev = ["O"] * len(chunk), None
            for pos, wid in enumerate(enc.word_ids()):
                if wid is not None and wid != prev:
                    chunk_tags[wid] = ID2LABEL[int(np.argmax(logits[pos]))]
                prev = wid
            tags += chunk_tags
        tags = fix_bio(tags)
        ents = tags_to_entities(text, [(s, e) for _, s, e in words], tags)
        results.append({"id": row["id"], "text": text, "entities": ents, "tokens": [w for w, _, _ in words], "tags": tags})
        print(f"[{row['id']}] {text}")
        for e in ents: print(f"    {e['label']:24} {e['text']}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for r in results: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("saved", a.out)


if __name__ == "__main__":
    main()

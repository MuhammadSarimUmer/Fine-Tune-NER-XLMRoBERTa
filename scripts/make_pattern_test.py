"""Held-out SYNTHETIC test for PHONE_NUMBER and ID_NUMBER (CNIC) -> data/dataset/pattern_test.jsonl

Why: neither human test set contains phone numbers or CNICs (rules forbid real personal data, so they exist only as synthetic data).
This set measures whether the model really learned the *pattern*, not the training templates:
  * every sentence template here is NEW (none appears in generate_synthetic.py) and digits come from a different random seed
  * ~30% of numbers use formats the training data never contained (see UNSEEN_*)
  * decoy numbers that look similar but are NOT phones/CNICs (IBANs, plates, case numbers, 13-digit tracking ids, amounts)
It is synthetic, so report it separately and say so; it is NOT evidence about real-text performance.
All numbers are random digits (no real person's data).
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tokenization import spans_to_bio

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dataset" / "pattern_test.jsonl"
r = random.Random(20261022)
D = lambda n: "".join(r.choice("0123456789") for _ in range(n))
MOB = ["300", "301", "302", "303", "321", "322", "333", "334", "345", "346", "315", "316"]
NAMES = ["Hira Siddiqui", "Bilal Chaudhry", "Sana Qureshi", "Kamran Baloch", "Mehwish Raza", "Talha Memon", "Rabia Sheikh", "Zain Butt"]
CITIES = ["Lahore", "Karachi", "Quetta", "Multan", "Peshawar", "Hyderabad", "Sialkot"]


def phone(unseen):
    m = r.choice(MOB)
    if unseen:
        return r.choice([f"+92 ({m}) {D(7)}", f"0{m}-{D(3)}-{D(4)}", f"+92-{m[:2]}-{m[2]}{D(7)}"[:0] + f"0{m} {D(3)} {D(4)}",
                         f"92{m}{D(7)}", f"0{m}.{D(7)}"])
    return r.choice([f"0{m}-{D(7)}", f"0{m}{D(7)}", f"+92 {m} {D(7)}", f"+92{m}{D(7)}", f"0092-{m}-{D(7)}", f"+92-{m}-{D(7)}",
                     f"0{r.choice(['21', '42', '51'])}-{D(8)}"])


def cnic(unseen):
    a, b, c = str(r.randint(1, 7)) + D(4), D(7), D(1)
    if unseen:
        return r.choice([f"{a}.{b}.{c}", f"{a}-{b} {c}", f"{a} - {b} - {c}", f"{a}-{b}-{c}".replace("-", "\u2013")[:0] + f"{a}/{b}/{c}"])
    return r.choice([f"{a}-{b}-{c}", f"{a}{b}{c}", f"{a} {b} {c}"])


def decoy():
    return r.choice([f"PK{D(2)} SCBL {D(4)} {D(4)} {D(4)} {D(2)}", f"LEB-{r.randint(10, 99)}-{D(4)}", f"ABC-{D(3)}",
                     f"{r.randint(100, 999)}/{r.randint(2010, 2024)}", f"TRK{D(10)}", f"{D(6)}",
                     f"{r.randint(1, 31)}-{r.randint(1, 12)}-{r.randint(2015, 2024)}", f"{D(4)}-{D(4)}"])


# {P} phone  {I} cnic  {D} decoy  {N} person  {C} city
PHONE_T = ["Reach the duty officer on {P} anytime.", "Whatsapp: {P}", "{N} left a voicemail from {P}.",
           "Dial {P} for the complaint cell in {C}.", "For booking, ring {P} or {P}.", "Emergency contact (sister): {P}",
           "The caller, later identified as {N}, used {P}.", "Tel / Fax {P}", "Customer care {P} (open 9 to 5).",
           "Send a text message to {P} with your roll number.", "The clinic in {C} is reachable at {P}.",
           "Mobile# {P} was registered in the name of {N}.", "Please save {P} and call back after prayers."]
ID_T = ["Applicant CNIC {I}.", "{N} produced a card numbered {I} at the counter in {C}.", "CNIC # {I} verified by the clerk.",
        "National ID card: {I}", "Registered voter identity {I}, resident of {C}.", "Attach a photocopy of CNIC {I} with the form.",
        "The deceased was identified through the number {I}.", "Father's CNIC: {I}  Mother's CNIC: {I}",
        "{N} (ID {I}) signed the affidavit.", "The bank requires your {I} to open an account."]
DECOY_T = ["The invoice number is {D}.", "Vehicle {D} was found abandoned in {C}.", "Transfer to account {D} was declined.",
           "The shipment {D} reached {C} safely.", "Case reference {D} is pending.", "{N} paid {D} for the ticket.",
           "Use reference code {D} when you call the helpline.", "The file number is {D}."]
MIX_T = ["{N} of {C} gave CNIC {I} and phone {P}; the invoice was {D}.", "Contact {N} on {P}, ID card {I}, case {D}.",
         "In {C}, {N} (CNIC {I}) can be reached at {P}."]

rows = []
def make(template, label_kind):
    text, spans, pos = "", [], 0
    import re
    unseen = r.random() < 0.3
    for m in re.finditer(r"\{([PIDNC])\}", template):
        text += template[pos:m.start()]; pos = m.end()
        k = m.group(1)
        v = {"P": lambda: phone(unseen), "I": lambda: cnic(unseen), "D": decoy, "N": lambda: r.choice(NAMES), "C": lambda: r.choice(CITIES)}[k]()
        lab = {"P": "PHONE_NUMBER", "I": "ID_NUMBER", "N": "PERSON", "C": "CITY"}.get(k)
        if lab: spans.append((len(text), len(text) + len(v), lab))
        text += v
    text += template[pos:]
    toks, tags = spans_to_bio(text, spans)
    rows.append(dict(id=f"pat{len(rows)}", text=text, tokens=toks, ner_tags=tags, ignore=[], source="pattern_test",
                     group=f"pattern:{label_kind}:{template[:20]}", origin="pattern_test"))


for t in PHONE_T:
    for _ in range(6): make(t, "phone")
for t in ID_T:
    for _ in range(6): make(t, "cnic")
for t in DECOY_T:
    for _ in range(4): make(t, "decoy")
for t in MIX_T:
    for _ in range(6): make(t, "mixed")
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for x in rows: f.write(json.dumps(x, ensure_ascii=False) + "\n")
n = {k: sum(1 for x in rows for t in x["ner_tags"] if t == f"B-{k}") for k in ("PHONE_NUMBER", "ID_NUMBER", "PERSON", "CITY")}
print(f"wrote {len(rows)} sentences -> {OUT}\nentities: {n}")
for x in rows[::25]: print(" ", x["text"])

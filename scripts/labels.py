"""Single source of truth for the NER label schema (20 entity types, BIO tagging)."""

ENTITY_TYPES = [
    "PERSON", "CITY", "PROVINCE_REGION", "COUNTRY", "LANDMARK_FACILITY",
    "GOVERNMENT_BODY", "POLITICAL_PARTY", "ORGANIZATION", "EDUCATIONAL_INSTITUTION",
    "COURT_LEGAL_BODY", "LAW_ACT", "EVENT", "MEDIA_OUTLET", "SPORTS_TEAM",
    "DATE", "TIME", "MONEY", "PERCENT", "PHONE_NUMBER", "ID_NUMBER",
]

# 'O' first so it gets id 0, then B-/I- for each type -> 41 labels total
LABELS = ["O"] + [f"{p}-{e}" for e in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

if __name__ == "__main__":
    print(len(ENTITY_TYPES), "entity types ->", len(LABELS), "BIO labels")
    print(LABELS)

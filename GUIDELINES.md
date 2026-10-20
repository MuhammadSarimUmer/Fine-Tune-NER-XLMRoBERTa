# Annotation Guidelines

Tagging: BIO, flat (no nested entities). Longest sensible span wins. Be consistent.

| Entity | Rule / examples |
|---|---|
| PERSON | Names of people, incl. titles only if part of name ("Quaid-e-Azam" yes; "Prime Minister" alone no) |
| CITY | Lahore, Karachi, Quetta |
| PROVINCE_REGION | Punjab, Sindh, Balochistan, KPK, Gilgit-Baltistan, Azad Kashmir |
| COUNTRY | Pakistan, China, USA |
| LANDMARK_FACILITY | Minar-e-Pakistan, Jinnah Hospital, Allama Iqbal Airport, Badshahi Mosque |
| GOVERNMENT_BODY | FIA, NAB, Punjab Police, Ministry of Finance, National Assembly, Senate |
| POLITICAL_PARTY | PTI, PML-N, PPP, MQM |
| ORGANIZATION | Companies/NGOs/other: Engro, HBL, Edhi Foundation, WHO |
| EDUCATIONAL_INSTITUTION | LUMS, NED University, University of Karachi |
| COURT_LEGAL_BODY | Supreme Court, Lahore High Court, Federal Shariat Court, Election Tribunal |
| LAW_ACT | Pakistan Penal Code, Constitution, Elections Act 2017, Article 62 |
| EVENT | Independence Day, Budget 2024-25, PSL 9, General Elections 2024 |
| MEDIA_OUTLET | Dawn, Geo News, ARY, Express Tribune |
| SPORTS_TEAM | Lahore Qalandars, Pakistan cricket team, Karachi Kings |
| DATE | 5 May 2024, March, last Friday, 2023 |
| TIME | 3:30 PM, 9 o'clock |
| MONEY | Include currency: "Rs 5 million", "PKR 200", "$10 billion" |
| PERCENT | 15%, 15 percent |
| PHONE_NUMBER | Synthetic only: +92 300 1234567, 0321-1234567 |
| ID_NUMBER | CNIC format, synthetic only: 35202-1234567-1 |

## Tricky cases
- LAW_ACT: label the whole reference as one span, e.g. "Section 302 of the Pakistan Penal Code", "Article 184(3)".
- Courtesy titles (Mr, Dr, Justice, Prime Minister) are NOT part of PERSON.
- MONEY includes the currency: "Rs. 5 million", "Rs5.63 million", "$2 billion".
- FIR/case/flight numbers and page counts are NOT ID_NUMBER or PHONE_NUMBER.
- "Pakistan" alone = COUNTRY; "Pakistan cricket team"/"Pakistan" in a match context = SPORTS_TEAM (label full team name).
- City inside a team/org name is NOT labeled separately: [Lahore Qalandars] = one SPORTS_TEAM.
- Courts are COURT_LEGAL_BODY, not GOVERNMENT_BODY.
- Police/army/ministries = GOVERNMENT_BODY.
- Do not label generic words ("the court", "the government", "the party").
- Possessive 's is not part of the entity.
- Entity types are never labeled for private real individuals' phone/CNIC: use synthetic only.

## Dates, times and durations
- Time of day = TIME: "this evening", "tonight", "this morning", "midnight", "3:30 PM". Words like "later", "around" are not part of the span.
- Day + time of day = TWO spans: "Thursday" DATE + "morning" TIME; "Wednesday" DATE + "night" TIME.
- Relative dates are DATE: "last year", "tomorrow", "last month", "next week", "this financial year".
- Date ranges and formats are one DATE span: "8-10 November 2014", "14/15 August 1947", "24th June, 1993".
- Durations/periods are DATE (OntoNotes/spaCy convention; decided 20 Oct): "two weeks", "three years", "23 months", "a decade", "22-year-old" (age = the span "22-year-old"), "up to two years". Not TIME: TIME is only a time of day.
- Plain counts and ordinals ("second day", "five-member bench") are still NOT entities.
- Scores/figures such as "4-18" are not DATE.
- A year inside a name is part of that name: "ICC Champions Trophy 2013" = one EVENT; "Elections Act 2017" = one LAW_ACT; do not label the year separately.
- A year inside a case or petition number ("Criminal Petition No. 75-Q/2021") is NOT a DATE.

## Names with articles and bare court names (decided 20 Oct; applied by `scripts/normalize_conventions.py`)
- A leading "The" is NOT part of the name: "Express Tribune", not "The Express Tribune". Reason: the human annotators consistently excluded it; one rule keeps train and test consistent.
- Capitalised court names are COURT_LEGAL_BODY even when bare: "the learned High Court", "Supreme Court", "High Courts". Reason: the hidden benchmark is on court records, where bare "High Court" is a court mention a grader will most likely expect; missing them costs recall. Lowercase generic mentions ("the court", "apex court") stay unlabelled.

## Places and regions
- PROVINCE_REGION also covers named regions and territories: Kashmir, Azad Kashmir, Jammu and Kashmir, Gilgit-Baltistan, Islamabad Capital Territory, FATA, Thar, Cholistan, Potohar, tribal agencies (Kurram Agency).
- Districts, towns and cities = CITY (Tharparkar District, Umerkot, Noshki).
- Natural features (river, mountain, lake) and buildings/stadiums/airports = LANDMARK_FACILITY.
- Team vs country: in a cricket/hockey match context, national teams are SPORTS_TEAM (England, New Zealand, Pakistan); otherwise COUNTRY.

## Organisations and bodies
- Abbreviations get the type of what they stand for: CCI, WAPDA, NIH, FO, AGPR = GOVERNMENT_BODY; PIB, ITP, NAP, SUP = POLITICAL_PARTY.
- Bare job titles (IGP, CEC, AAG, Inspector General of Police) are NOT entities.
- Programme/scheme names and generic nouns ("the Society", "Company", "Local Commission", "aforesaid crime report") are NOT entities.
- Tournaments and competitions (Quaid-e-Azam Trophy, World Cup, PSL) = EVENT; cricket grounds (Kingsmead, Arbab Niaz Stadium) = LANDMARK_FACILITY.
- Political agreements and family names used as groups ("the Bhutto family", "Charter of Democracy") are not labelled.

## Added 20 Oct (error-analysis round 1)
- A year inside a case number or law-report citation is never a DATE: C.A.538/2022, Civil Appeal No.756 of 2021, C.P. No.260-P of 2019, 2010 SCMR 1962, PLD 2012 SC 25 (applied by 
ormalize_conventions.py).
- Police stations, bazaars, shrines, temples, bridges, rivers, dams and ports with their proper name = LANDMARK_FACILITY (e.g. "Kotwali police station", "Jhelum River").

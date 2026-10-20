# Data Sources, Rules and Source Log

> Not legal advice. Terms were checked on the dates logged below; re-check before use.
> Competition rule: scrape/gather *publicly available* Pakistani sources, respect site terms, avoid private/personal data.
> PHONE_NUMBER and ID_NUMBER must be synthetic.

## Our data policy (use this in the report/presentation)
1. Prefer openly licensed text (Tier 1), then official public records (Tier 2); keep commercial news small (Tier 3).
2. Check robots.txt and terms of use for every site before scraping; rate-limit requests (>= 1 s between requests); identify our bot with a User-Agent.
3. Use text only for training/evaluation of our model. We do NOT redistribute the scraped dataset.
4. No full articles pasted in the report or slides; short snippets only, with attribution.
5. No personal/private data. Phone numbers and CNICs are generated, never taken from real people.
6. Every document is logged below with URL, licence and date.
7. Wikipedia-derived text: CC BY-SA 4.0, credited in the report.

## Tiers (planned mix, adjustable)
| Tier | Sources | Licence / risk | Target share |
|---|---|---|---|
| 1 | Wikipedia, Wikinews, World Bank docs on Pakistan, Global Voices | CC BY / CC BY-SA (VERIFY each) | ~50% |
| 2 | Court judgments, Pakistan Code, NA/Senate, govt press releases (PID, MOFA, SBP), HEC/NED pages | Public records; read terms (SECP/FBR: single pages only) | ~35% |
| 3 | 2-3 news outlets (Dawn, Tribune, The News ...) | Commercial copyright; tens of articles only | ~15% |

Open questions for organizers: (a) may public NER datasets be used for warm-up? (b) may LLM-generated sentences be used beyond phone/CNIC?

## Source log
## Downloaded public datasets (Kaggle; open, publicly available; per-dataset licence not individually verified)
Decision: all kept datasets are used as a whole (team decision, 19 Oct 2026). Used for model training/evaluation only, not redistributed.

| Local folder | Kaggle source | Used for |
|---|---|---|
| `datasets/Pakistan-news` | https://www.kaggle.com/datasets/zusmani/pakistan-news-headlines | ~26.7k headlines/excerpts (news style) |
| `datasets/Pakistan_census` | https://www.kaggle.com/datasets/mabdullahsajid/population-of-pakistan-dataset | Gazetteer: provinces, districts, tehsils |
| `datasets/Pakistan-news2` | https://www.kaggle.com/datasets/zincly/pakistan-news-articles-2013-and-2018 | Dawn (778) + Tribune (3,309) articles |
| `datasets/Pakistan-penal-code` | https://www.kaggle.com/datasets/shanza30/pakistan-penal-code-sections-for-llm-training | LAW_ACT / legal-term gazetteer |
| `datasets/Supreme_court_Of_Pakistan_judgments` | source URL not yet recorded (TODO: add) | Legal text, 2,809 judgments |

Removed after inspection (not used): Imran Khan tweets (garbled Urdu encoding, personal social media), T20 IND vs PAK tweets (not Pakistan-specific, informal, 4 duplicate formats), Imran Khan Reddit (informal forum posts). Their Kaggle links remain in `kagglesources.txt`.

## Source log (scraped/API sources)
| Date checked | Source | URL | Licence / terms note | Robots OK? | Approx docs | Notes |
|---|---|---|---|---|---|---|
| 2026-10-19 | Wikipedia (API) | https://en.wikipedia.org | CC BY-SA 4.0 | API | 209 | scripts/scrape_wikipedia.py |
| 2026-10-19 | Wikinews (API) | https://en.wikinews.org | CC BY 2.5 | API | 23 | scripts/scrape_wikipedia.py |

"""Generate synthetic, automatically BIO-labelled sentences (Pakistani context).

Run from project root:  python scripts/generate_synthetic.py [--seed 42] [--scale 1.0]
Output: data/synthetic/synthetic.jsonl  (tokens, ner_tags, text, group=template id)

Why synthetic?
- PHONE_NUMBER / ID_NUMBER MUST be synthetic (competition rule; no real people's details).
- MONEY / PERCENT / DATE / TIME follow patterns, so templates give perfect labels cheaply.
- Rare classes (SPORTS_TEAM, MEDIA_OUTLET, EVENT, LANDMARK_FACILITY, LAW_ACT, COURT_LEGAL_BODY...)
  need extra examples; real text has few of them.
Labels are exact because we insert the entity ourselves: no manual annotation needed.

All phone numbers / CNICs are random digits. Names are combined randomly from first-name and
surname lists, so they do not refer to real individuals. Template ids are stored in `group`
so Phase 5 can hold out whole templates for validation (prevents template memorisation scoring).
"""
import argparse, json, random, re
from collections import Counter
from pathlib import Path

from labels import ENTITY_TYPES
from tokenization import tokenize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synthetic" / "synthetic.jsonl"

# ------------------------------------------------------------------ entity value lists
MALE = ("Muhammad Ali Ahmed Hassan Hussain Usman Bilal Imran Kamran Faisal Asad Hamza Zain Saad Talha Salman Omar Umar "
        "Tariq Nadeem Shahid Waqar Junaid Rizwan Adnan Fahad Danish Haider Zubair Shoaib Naveed Irfan Kashif Yasir "
        "Abdullah Ibrahim Ayaz Farhan Hammad Raza Asif Sajid Khurram Rashid Shahzad Arslan Waseem Mubashir Noman").split()
FEMALE = ("Ayesha Fatima Zainab Maryam Sana Hina Sadia Nida Amna Bushra Rabia Mehwish Saira Iqra Noor Hira Laiba "
          "Khadija Sidra Anum Farah Nazia Uzma Samina Shazia Rukhsana Tahira Aliya Mahnoor Komal").split()
SURNAME = ("Khan Ahmed Ali Malik Sheikh Butt Chaudhry Qureshi Siddiqui Hashmi Raza Shah Baloch Bhutto Memon Abbasi Niazi "
           "Mirza Javed Iqbal Hussain Rehman Akhtar Farooqi Ansari Lodhi Gilani Zardari Sharif Afridi Yousafzai Bajwa "
           "Cheema Gondal Mughal Rana Tarar Awan Jatoi Mengal Bugti Kakar Durrani Saeed Mehmood Ghani Anwar").split()
CITY = ("Lahore Karachi Islamabad Rawalpindi Faisalabad Multan Peshawar Quetta Hyderabad Gujranwala Sialkot Sargodha "
        "Bahawalpur Sukkur Larkana Mardan Abbottabad Mirpur Gwadar Jhelum Sahiwal Rahim_Yar_Khan Okara Kasur Swat "
        "Mingora Muzaffarabad Gilgit Skardu Chitral Nawabshah Dera_Ghazi_Khan Gujrat Sheikhupura Mansehra Kohat Turbat").split()
PROVINCE = ["Punjab", "Sindh", "Balochistan", "Khyber Pakhtunkhwa", "KPK", "Gilgit-Baltistan", "Azad Kashmir",
            "Islamabad Capital Territory", "Southern Punjab", "Upper Sindh", "Malakand", "Waziristan", "Kurram"]
COUNTRY = ["Pakistan", "China", "India", "Afghanistan", "Iran", "Saudi Arabia", "United States", "United Kingdom", "UAE",
           "Turkey", "Bangladesh", "Sri Lanka", "Qatar", "Japan", "Germany", "Canada", "Malaysia", "Russia", "Australia"]
LANDMARK = ["Minar-e-Pakistan", "Badshahi Mosque", "Faisal Mosque", "Mazar-e-Quaid", "Lahore Fort", "Shalimar Gardens",
            "Jinnah International Airport", "Allama Iqbal International Airport", "Islamabad International Airport",
            "Jinnah Hospital", "Mayo Hospital", "Services Hospital", "Civil Hospital Karachi", "Gaddafi Stadium",
            "National Stadium", "Rawalpindi Cricket Stadium", "Multan Cricket Stadium", "Margalla Hills", "Mohatta Palace",
            "Faisalabad Clock Tower", "Karakoram Highway", "Motorway M-2", "Lahore Railway Station", "Cantt Station",
            "Karachi Port", "Jinnah Convention Centre", "Pakistan Monument", "Data Darbar", "Tarbela Dam", "Mangla Dam",
            "Ayub National Park", "Liaquat National Hospital", "Shaukat Khanum Hospital", "Sehwan Sharif shrine"]
GOVBODY = ["FIA", "NAB", "Federal Investigation Agency", "National Accountability Bureau", "Ministry of Finance",
           "Ministry of Interior", "Ministry of Foreign Affairs", "Ministry of Health", "National Assembly", "Senate",
           "Punjab Police", "Sindh Police", "Election Commission of Pakistan", "ECP", "Federal Board of Revenue", "FBR",
           "State Bank of Pakistan", "NADRA", "PTA", "ISPR", "Punjab Assembly", "Sindh Assembly", "Pakistan Army",
           "Rangers", "Higher Education Commission", "HEC", "Capital Development Authority", "CDA", "NDMA", "PDMA Punjab",
           "Planning Commission", "Cabinet Division", "Khyber Pakhtunkhwa Assembly", "Customs", "Pakistan Railways",
           "Anti-Narcotics Force", "Counter Terrorism Department", "SECP", "Provincial Assembly of Balochistan"]
PARTY = ["PTI", "PML-N", "PPP", "MQM", "MQM-P", "JUI-F", "ANP", "Jamaat-e-Islami", "BAP", "TLP", "PML-Q", "BNP",
         "Pakistan Tehreek-e-Insaf", "Pakistan Muslim League-Nawaz", "Pakistan Peoples Party", "Awami National Party",
         "Muttahida Qaumi Movement", "Istehkam-e-Pakistan Party", "National Party", "Pashtunkhwa Milli Awami Party"]
ORG = ["Engro", "HBL", "UBL", "MCB Bank", "Meezan Bank", "Allied Bank", "Edhi Foundation", "Pakistan International Airlines",
       "PIA", "Lucky Cement", "K-Electric", "Jazz", "Telenor Pakistan", "Zong", "Ufone", "WHO", "UNICEF", "World Bank",
       "IMF", "Asian Development Bank", "Pakistan Steel Mills", "Oil and Gas Development Company", "OGDC", "PSO",
       "Fauji Foundation", "Habib Metropolitan Bank", "Nestle Pakistan", "Unilever Pakistan", "Shaukat Khanum Memorial Trust",
       "Pakistan Medical Association", "Lahore Chamber of Commerce and Industry", "Pakistan Cricket Board", "PCB",
       "Karachi Stock Exchange", "Pakistan Stock Exchange", "Alkhidmat Foundation", "Chhipa Welfare Association", "Daraz",
       "Careem", "Foodpanda", "Systems Limited", "Pakistan State Oil", "United Nations", "Amnesty International"]
EDU = ["LUMS", "NED University of Engineering and Technology", "University of Karachi", "COMSATS University Islamabad",
       "IBA Karachi", "FAST-NUCES", "Aga Khan University", "Quaid-i-Azam University", "GIKI", "University of the Punjab",
       "NUST", "Government College University Lahore", "Dow University of Health Sciences", "University of Peshawar",
       "Bahria University", "Lahore College for Women University", "King Edward Medical University", "UET Lahore",
       "University of Engineering and Technology", "Islamia University Bahawalpur", "Habib University", "Karachi University",
       "Institute of Business Administration", "PIEAS", "Forman Christian College", "Beaconhouse School System",
       "The City School", "Allama Iqbal Open University", "University of Sindh", "Punjab University", "Virtual University of Pakistan"]
COURT = ["Supreme Court of Pakistan", "Supreme Court", "Lahore High Court", "Sindh High Court", "Islamabad High Court",
         "Peshawar High Court", "Balochistan High Court", "Federal Shariat Court", "Anti-Terrorism Court",
         "Accountability Court", "Election Tribunal", "Banking Court", "Sessions Court", "District and Sessions Court",
         "Military Court", "Supreme Appellate Court", "Rawalpindi Bench", "Multan Bench", "Bahawalpur Bench",
         "Judicial Commission of Pakistan", "Supreme Judicial Council", "Federal Constitutional Court", "Family Court",
         "Federal Service Tribunal", "Customs Appellate Tribunal", "Special Court"]
LAW_NAMES = ["Pakistan Penal Code", "Constitution of Pakistan", "Elections Act 2017", "Anti-Terrorism Act 1997",
             "Code of Criminal Procedure", "Companies Act 2017", "Prevention of Electronic Crimes Act 2016",
             "National Accountability Ordinance 1999", "Qanun-e-Shahadat Order 1984", "Income Tax Ordinance 2001",
             "Contempt of Court Ordinance", "Protection of Pakistan Act 2014", "Pakistan Army Act 1952",
             "Right of Access to Information Act 2017", "Constitution", "Criminal Procedure Code", "Civil Procedure Code",
             "Local Government Act 2022", "Land Revenue Act 1967", "Sales Tax Act 1990", "Control of Narcotic Substances Act 1997",
             "Punjab Tenancy Act", "Sindh Elections Act", "Transfer of Property Act 1882", "Pakistan Telecommunication Act 1996",
             "Protection of Women Act 2006", "Muslim Family Laws Ordinance 1961", "Official Secrets Act 1923"]
LAW_ARTICLES = ["Article 6", "Article 62", "Article 63", "Article 63-A", "Article 184(3)", "Article 199", "Article 10-A",
                "Article 25", "Article 245", "Article 140-A", "Article 19-A", "Article 17", "Section 144", "Section 302",
                "Section 10", "Section 497", "Section 420", "Section 144 of the CrPC", "Section 7 of the Anti-Terrorism Act",
                "Section 302 of the Pakistan Penal Code", "Section 9 of the Elections Act 2017", "Article 8 of the Constitution",
                "Section 496-A", "Section 376", "Section 489-F", "Section 4 of the Official Secrets Act"]
EVENT = ["Independence Day", "Pakistan Day", "Defence Day", "Kashmir Solidarity Day", "Eid-ul-Fitr", "Eid-ul-Adha",
         "Budget 2024-25", "Budget 2023-24", "General Elections 2024", "General Elections 2018", "PSL 9", "PSL 8", "Asia Cup 2023",
         "ICC T20 World Cup", "ICC Champions Trophy", "Cricket World Cup 2023", "Lahore Literary Festival",
         "Karachi Literature Festival", "Muharram", "Ramzan", "Ashura", "SCO Summit", "Independence Day parade", "Pakistan Super League",
         "National Dialogue Conference", "Lahore Marathon", "Karachi Eat Festival", "Faiz Festival", "Operation Zarb-e-Azb",
         "2022 floods", "2005 earthquake", "Kartarpur Corridor opening", "Azadi March", "Long March", "Shandur Polo Festival",
         "Sindh Festival", "Basant", "Quaid-e-Azam Day", "Youm-e-Takbeer", "Women's Day", "Labour Day", "Hockey World Cup"]
MEDIA = ["Dawn", "Geo News", "ARY News", "Express Tribune", "The News International", "Samaa TV", "Dunya News", "Hum TV",
         "Business Recorder", "BBC Urdu", "Bol News", "PTV", "Radio Pakistan", "APP", "Reuters", "Al Jazeera", "CNN", "BBC",
         "The Nation", "Daily Jang", "Express News", "92 News", "Aaj News", "Neo News", "Pakistan Today", "Daily Times",
         "Voice of America", "Associated Press of Pakistan", "Geo TV", "ARY Digital", "Hum News", "Public News", "Roze News",
         "Nawa-i-Waqt", "Daily Express", "The Friday Times", "Gwadar Pro", "Bloomberg", "Anadolu Agency"]
TEAM = ["Lahore Qalandars", "Karachi Kings", "Islamabad United", "Peshawar Zalmi", "Quetta Gladiators", "Multan Sultans",
        "Pakistan cricket team", "Pakistan hockey team", "Pakistan national football team", "Pakistan women's cricket team",
        "Pakistan Shaheens", "Karachi Whites", "Lahore Blues", "Sindh Cricket team", "Punjab Cricket team", "Peshawar Panthers",
        "National Bank of Pakistan cricket team", "WAPDA", "Pakistan Army football team", "Khan Research Laboratories FC",
        "Pakistan squash team", "Pakistan Under-19 team", "Rawalpindi Rams", "Northern cricket team", "Balochistan Bears",
        "Pakistan A", "Pakistan Eagles", "Hyderabad Hawks", "Faisalabad Wolves", "Dolphins", "Lions", "Pakistan Navy"]
# ---- round-2 additions (Step 17): weak labels. Names were chosen from general knowledge, NOT copied from test/extra-test errors.
LANDMARK += ["Jhelum River", "Chenab River", "Ravi River", "Sutlej River", "Kabul River", "Neelum River", "Swat River",
             "Keenjhar Lake", "Manchar Lake", "Saif-ul-Malook Lake", "Hub Dam", "Port Qasim", "Anarkali Bazaar", "Liaquat Bagh",
             "Lal Masjid", "Katas Raj Temple", "Bhong Mosque", "Shah Rukn-e-Alam shrine", "Moenjodaro", "Hingol National Park",
             "Deosai National Plains", "Khunjerab Pass", "Torkham border crossing", "Wagah border", "Kotwali police station",
             "Model Town police station", "Gulberg police station", "Anarkali police station", "Race Course police station",
             "Clifton Beach", "Mall Road", "Lahore Zoo", "Sukkur Barrage", "Guddu Barrage", "Chaukhandi tombs", "Rohtas Fort",
             "Hazuri Bagh", "Karachi Press Club", "Bolan Pass", "Lowari Tunnel", "Neelum Valley", "Jhelum Bridge"]
GOVBODY += ["PEMRA", "NEPRA", "FPSC", "NTDC", "SNGPL", "SSGC", "LESCO", "WASA Lahore", "National Highway Authority",
            "Securities and Exchange Commission of Pakistan", "Pakistan Telecommunication Authority", "Punjab Healthcare Commission",
            "Lahore Development Authority", "Karachi Water and Sewerage Board", "Intelligence Bureau", "Pakistan Air Force",
            "Ministry of Climate Change", "Ministry of Education", "Provincial Disaster Management Authority",
            "Rawalpindi Development Authority", "Lahore Police", "Karachi Police", "Frontier Corps", "Pakistan Coast Guards",
            "Federal Public Service Commission", "Economic Coordination Committee", "Council of Common Interests", "Senate Secretariat"]
ORG += ["Pakistan Tobacco Company", "Fatima Fertilizer", "Packages Limited", "Bank Alfalah", "Standard Chartered Pakistan",
        "Pakistan Bar Council", "Lahore High Court Bar Association", "Karachi Bar Association", "Pakistan Hindu Council",
        "Edhi Trust", "Pakistan Red Crescent Society", "Human Rights Commission of Pakistan", "Pakistan Federal Union of Journalists",
        "Karachi Chamber of Commerce and Industry", "All Pakistan Textile Mills Association", "Pakistan Hotels Association",
        "UNESCO", "UNHCR", "UNDP", "WTO", "OPEC", "NATO", "SAARC", "OIC", "Islamic Development Bank", "Pakistan Sugar Mills Association",
        "Oil Marketing Companies Association of Pakistan", "Hum Network Limited", "Sui Southern Gas Company Employees Union",
        "Pakistan Muslim League Women Wing Trust", "Aurat Foundation", "Sindh Bar Council"]
LAW_NAMES += ["Customs Act 1969", "Foreign Exchange Regulation Act 1947", "Punjab Local Government (Amendment) Act 2021",
              "Sindh Rented Premises Ordinance 1979", "Industrial Relations Act 2012", "Punjab Protection of Women Against Violence Act 2016",
              "Khyber Pakhtunkhwa Right to Information Act 2013", "Code of Civil Procedure 1908", "Anti-Money Laundering Act 2010",
              "Pakistan Citizenship Act 1951", "Sindh Child Marriage Restraint Act 2013", "Punjab Land Revenue (Amendment) Act 2020",
              "Prevention of Corruption Act 1947", "Specific Relief Act 1877", "Arms Ordinance 1965", "Explosive Substances Act 1908",
              "Payment and Settlement Systems Act 2007", "Civil Servants (Efficiency and Discipline) Rules 1973",
              "Removal from Service (Special Powers) Ordinance 2000", "Hudood Ordinances 1979", "Telegraph Act 1885",
              "Press, Newspapers, News Agencies and Books Registration Ordinance 2002", "Registration Act 1908",
              "Punjab Civil Servants Act 1974", "Sindh Civil Servants Act 1973", "Contract Act 1872", "Limitation Act 1908",
              "Factories Act 1934", "Employees' Old-Age Benefits Act 1976", "Environmental Protection Act 1997"]
TEAM += ["India", "England", "Australia", "New Zealand", "Sri Lanka", "Bangladesh", "Zimbabwe", "Ireland", "Pakistan women",
         "Pakistan Under-19s"]

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
RELDATES = ["yesterday", "today", "tomorrow", "last week", "next month", "last year", "this week", "next Friday", "last Monday", "next week"]
TITLES_M = ["Mr", "Dr", "Justice", "Prime Minister", "Chief Minister", "Senator", "Minister", "Chief Justice", "General", "Sheikh"]
TITLES_F = ["Ms", "Dr", "Mrs", "Justice", "Senator", "Begum"]


# ------------------------------------------------------------------ generators
def ordinal(d):
    return f"{d}{'th' if 10 <= d % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')}"


def g_person(r):
    if r.random() < 0.55:
        first = r.choice(MALE)
    else:
        first = r.choice(FEMALE)
    form = r.random()
    if form < 0.65:
        return f"{first} {r.choice(SURNAME)}"
    if form < 0.8:
        return f"{first} {r.choice(MALE)} {r.choice(SURNAME)}"
    if form < 0.9:
        return first
    return r.choice(SURNAME) if r.random() < 0.5 else f"{first} {r.choice(SURNAME)}"


def g_city(r): return r.choice(CITY).replace("_", " ")
def g_prov(r): return r.choice(PROVINCE)
def g_country(r): return r.choice(COUNTRY)


def g_date(r):
    d, m, y = r.randint(1, 28), r.choice(MONTHS), r.randint(2008, 2026)
    k = r.random()
    if k < 0.2: return f"{d} {m} {y}"
    if k < 0.35: return f"{m} {d}, {y}"
    if k < 0.45: return f"{ordinal(d)} {m}"
    if k < 0.55: return f"{d} {m}"
    if k < 0.65: return f"{m} {y}"
    if k < 0.72: return str(y)
    if k < 0.8: return r.choice(WEEKDAYS)
    if k < 0.9: return r.choice(RELDATES)
    if k < 0.95: return f"{d}/{r.randint(1, 12)}/{y}"
    return f"{ordinal(d)} of {m}"


def g_time(r):
    h, mi = r.randint(1, 12), r.choice([0, 15, 30, 45, 5, 10, 20, 40, 50])
    k = r.random()
    if k < 0.3: return f"{h}:{mi:02d} {r.choice(['AM', 'PM', 'am', 'pm'])}"
    if k < 0.45: return f"{h} {r.choice(['am', 'pm', 'AM', 'PM'])}"
    if k < 0.6: return f"{r.randint(0, 23)}:{mi:02d}"
    if k < 0.7: return f"{h}:{mi:02d}{r.choice(['am', 'pm'])}"
    if k < 0.8: return f"{h}{r.choice(['am', 'pm'])}"
    if k < 0.9: return r.choice(["noon", "midnight", "dawn", "dusk"])
    return f"{h}:{mi:02d} {r.choice(['a.m.', 'p.m.'])}"


def g_money(r):
    scale = r.choice(["", "", "thousand", "million", "million", "billion", "lakh", "crore", "trillion"])
    cur = r.choices(["Rs", "Rs.", "PKR", "Rs", "$", "USD", "Rupees", "US$"], [8, 3, 5, 0, 2, 2, 1, 1])[0]
    num = r.choice([r.randint(1, 99), round(r.uniform(1, 99), 2), round(r.uniform(1, 9), 1), r.randint(100, 99999)])
    if isinstance(num, int) and num >= 1000 and not scale:
        num = f"{num:,}"
    k = r.random()
    if cur == "Rupees": return f"{num} Rupees" if not scale else f"{num} {scale} Rupees"
    if k < 0.3 and cur in ("Rs", "PKR"): return f"{cur}{num}{(' ' + scale) if scale else ''}"   # Rs5.63 million (Tribune style)
    return f"{cur} {num}{(' ' + scale) if scale else ''}"


def g_percent(r):
    v = r.choice([r.randint(1, 100), round(r.uniform(0.1, 99.9), 1), round(r.uniform(0.01, 9.99), 2)])
    return r.choice([f"{v}%", f"{v}%", f"{v} percent", f"{v} per cent", f"{v}pc"])


def digits(r, n): return "".join(r.choice("0123456789") for _ in range(n))


def g_cnic(r):
    # 5-digit locality code (first digit 1-7 = region), 7-digit family number, 1 digit (odd male / even female)
    a = str(r.randint(1, 7)) + digits(r, 4)
    b, c = digits(r, 7), digits(r, 1)
    k = r.random()
    if k < 0.75: return f"{a}-{b}-{c}"
    if k < 0.9: return f"{a}{b}{c}"
    return f"{a} {b} {c}"


def g_phone(r):
    mob = r.choice(["300", "301", "302", "303", "304", "305", "306", "307", "308", "309", "310", "311", "312", "313", "314",
                    "315", "316", "317", "318", "319", "320", "321", "322", "323", "324", "325", "330", "331", "332", "333",
                    "334", "335", "336", "337", "340", "341", "342", "343", "344", "345", "346", "347"])
    rest = digits(r, 7)
    k = r.random()
    if k < 0.18: return f"0{mob}-{rest}"
    if k < 0.34: return f"0{mob}{rest}"
    if k < 0.5: return f"+92 {mob} {rest}"
    if k < 0.62: return f"+92{mob}{rest}"
    if k < 0.7: return f"0092-{mob}-{rest}"
    if k < 0.78: return f"0{mob} {rest}"
    if k < 0.84: return f"+92-{mob}-{rest}"
    if k < 0.92:
        area = r.choice(["21", "42", "51", "91", "52", "41", "61", "81", "22", "55"])
        return f"0{area}-{digits(r, 8 if len(area) == 2 else 7)}"
    area = r.choice(["21", "42", "51"])
    return f"(0{area}) {digits(r, 3)}-{digits(r, 3)}-{digits(r, 3)}"


def g_fir(r): return f"{r.randint(1, 999)}/{r.randint(2008, 2025)}"
def g_flight(r): return f"PK-{r.randint(100, 899)}"
def g_page(r): return str(r.randint(2, 480))
def g_caseno(r):
    """Case numbers / law-report citations: contain years that are NOT dates (teaches the model to leave them alone)."""
    y, n = r.randint(1995, 2024), r.randint(1, 3999)
    return r.choice([f"C.A.{n}/{y}", f"CA.{n}/{y}", f"Civil Appeal No.{n} of {y}", f"C.P. No.{n}-P of {y}", f"Crl.P.{n}/{y}",
                     f"W.P. No.{n} of {y}", f"{y} SCMR {r.randint(1, 2400)}", f"PLD {y} SC {r.randint(1, 900)}",
                     f"{y} CLC {r.randint(1, 1900)}", f"{y} PCrLJ {r.randint(1, 1900)}", f"Suit No.{n}/{y}", f"Const.P. {n}/{y}"])
def pick(lst): return lambda r: r.choice(lst)


# label -> generator (None-labelled "noise" slots get label None and stay O)
GEN = {
    "PERSON": g_person, "CITY": g_city, "PROVINCE_REGION": g_prov, "COUNTRY": g_country,
    "LANDMARK_FACILITY": pick(LANDMARK), "GOVERNMENT_BODY": pick(GOVBODY), "POLITICAL_PARTY": pick(PARTY),
    "ORGANIZATION": pick(ORG), "EDUCATIONAL_INSTITUTION": pick(EDU), "COURT_LEGAL_BODY": pick(COURT),
    "LAW_ACT": lambda r: r.choice(LAW_NAMES) if r.random() < 0.5 else r.choice(LAW_ARTICLES),
    "EVENT": pick(EVENT), "MEDIA_OUTLET": pick(MEDIA), "SPORTS_TEAM": pick(TEAM),
    "DATE": g_date, "TIME": g_time, "MONEY": g_money, "PERCENT": g_percent,
    "PHONE_NUMBER": g_phone, "ID_NUMBER": g_cnic,
}
NOISE = {"FIRNO": g_fir, "FLIGHT": g_flight, "PAGE": g_page, "CASENO": g_caseno}
assert set(GEN) == set(ENTITY_TYPES), "generators must cover all 20 labels"

# ------------------------------------------------------------------ templates (focus label -> templates)
# {LABEL} = labelled entity slot; {FIRNO}/{FLIGHT}/{PAGE} = unlabelled digits (teach the model not to tag every number)
T = {
 "PHONE_NUMBER": [
  "For queries, please call {PHONE_NUMBER}.", "Contact number: {PHONE_NUMBER}", "He can be reached at {PHONE_NUMBER} or {PHONE_NUMBER}.",
  "The helpline {PHONE_NUMBER} is available 24 hours a day.", "Please WhatsApp us on {PHONE_NUMBER} to book an appointment.",
  "{PERSON} gave reporters the number {PHONE_NUMBER} for further details.", "To register a complaint dial {PHONE_NUMBER}.",
  "Office: {PHONE_NUMBER}, Mobile: {PHONE_NUMBER}", "Call {PHONE_NUMBER} between {TIME} and {TIME} on weekdays.",
  "Admissions office of {EDUCATIONAL_INSTITUTION} can be contacted at {PHONE_NUMBER}.", "Phone: {PHONE_NUMBER} Fax: {PHONE_NUMBER}",
  "Police in {CITY} urged citizens to report suspicious activity on {PHONE_NUMBER}.", "Send your feedback to {PHONE_NUMBER} by SMS.",
  "Mobile No. {PHONE_NUMBER}", "The complainant, {PERSON}, said the caller used the number {PHONE_NUMBER}.",
 ],
 "ID_NUMBER": [
  "CNIC number {ID_NUMBER}", "The applicant must attach a copy of the CNIC ({ID_NUMBER}).", "{PERSON}, holder of CNIC No. {ID_NUMBER}, appeared before the court.",
  "The suspect, identified by CNIC {ID_NUMBER}, was taken into custody in {CITY}.", "Please enter your 13-digit CNIC {ID_NUMBER} to continue.",
  "NADRA verified the card bearing the number {ID_NUMBER}.", "{PERSON} S/O {PERSON}, CNIC {ID_NUMBER}, resident of {CITY}.",
  "FIR No. {FIRNO} was registered against {PERSON} (CNIC {ID_NUMBER}) at {CITY} police station.",
  "ID card number: {ID_NUMBER}", "Voter {PERSON} with identity card number {ID_NUMBER} was not found in the list.",
  "The passport applicant's CNIC is {ID_NUMBER}.", "Borrower: {PERSON}, CNIC: {ID_NUMBER}, Phone: {PHONE_NUMBER}",
 ],
 "MONEY": [
  "The government allocated {MONEY} for the project.", "The price of petrol was raised by {MONEY} per litre.", "{ORGANIZATION} reported a profit of {MONEY} in the last quarter.",
  "{PERSON} was fined {MONEY} by the {COURT_LEGAL_BODY}.", "The ticket costs {MONEY}.", "Pakistan received {MONEY} from the {ORGANIZATION}.",
  "The monthly salary was increased from {MONEY} to {MONEY}.", "Remittances reached {MONEY} in {DATE}.", "The bail was set at surety bonds worth {MONEY}.",
  "The {GOVERNMENT_BODY} recovered {MONEY} from the accused.", "The flood relief fund has collected {MONEY} so far.",
  "Tuition fee at {EDUCATIONAL_INSTITUTION} is {MONEY} per semester.", "Gold was sold at {MONEY} per tola in {CITY}.",
 ],
 "PERCENT": [
  "Inflation rose to {PERCENT} in {DATE}.", "The policy rate was cut by {PERCENT}.", "Turnout in {CITY} was only {PERCENT}.",
  "{ORGANIZATION} shares fell {PERCENT} on the stock exchange.", "Nearly {PERCENT} of the population in {PROVINCE_REGION} lacks clean water.",
  "The budget increases defence spending by {PERCENT}.", "{PARTY} secured {PERCENT} of the votes.", "GDP growth is projected at {PERCENT} for the next fiscal year.",
  "The tax rate was raised from {PERCENT} to {PERCENT}.", "Literacy in {PROVINCE_REGION} stands at {PERCENT}.", "A discount of {PERCENT} was announced by {ORGANIZATION}.",
 ],
 "DATE": [
  "The hearing was adjourned until {DATE}.", "{PERSON} was born on {DATE} in {CITY}.", "The attack took place on {DATE}.", "Elections will be held on {DATE}.",
  "The bill was passed by the {GOVERNMENT_BODY} on {DATE}.", "The exam will begin on {DATE} and end on {DATE}.", "The petition was filed in {DATE}.",
  "According to the report published in {DATE}, the situation has worsened.", "{EVENT} is celebrated every year on {DATE}.", "The deadline is {DATE}.",
  "The match between {SPORTS_TEAM} and {SPORTS_TEAM} will be played on {DATE}.", "The prime minister is expected to visit {COUNTRY} {DATE}.",
 ],
 "TIME": [
  "The meeting will start at {TIME}.", "The explosion occurred at around {TIME} near {LANDMARK_FACILITY}.", "Prayers will be offered at {TIME}.",
  "The flight {FLIGHT} to {CITY} departs at {TIME}.", "Offices remain open from {TIME} to {TIME}.", "The match starts at {TIME} local time.",
  "The press conference by {PERSON} is scheduled for {TIME} on {DATE}.", "Police reached the spot at {TIME}.", "The court resumed hearing at {TIME}.",
  "The suspects fled at {TIME}.", "Polling begins at {TIME} and closes at {TIME}.",
 ],
 "PERSON": [
  "{PERSON} said the matter would be taken up in the next session.", "Prime Minister {PERSON} met the delegation in {CITY}.", "Justice {PERSON} observed that the law was clear.",
  "{PERSON} and {PERSON} were appointed as members of the committee.", "The accused, {PERSON}, denied all allegations.", "Dr {PERSON} is the head of the department at {EDUCATIONAL_INSTITUTION}.",
  "{PERSON}'s lawyer told the {COURT_LEGAL_BODY} that his client was innocent.", "Chief Minister {PERSON} inaugurated the hospital in {CITY}.",
  "Senator {PERSON} of {PARTY} criticised the budget.", "Captain {PERSON} led {SPORTS_TEAM} to victory.", "According to {PERSON}, the project will be completed by {DATE}.",
  "{PERSON}, a resident of {CITY}, was injured in the incident.", "The report was prepared by {PERSON} and reviewed by {PERSON}.", "{PERSON} resigned as chairman of {ORGANIZATION}.",
  "Former minister {PERSON} was granted bail by the {COURT_LEGAL_BODY}.", "Speaking to {MEDIA_OUTLET}, {PERSON} rejected the claims.",
 ],
 "CITY": [
  "The bus left {CITY} for {CITY} early in the morning.", "Heavy rain lashed {CITY} and its surrounding areas.", "A new metro line will connect {CITY} with {CITY}.",
  "Traders in {CITY} observed a strike on {DATE}.", "The factory in {CITY} employs over 2,000 workers.", "{PERSON} was shot dead in {CITY}.",
  "Flights to {CITY} were suspended due to fog.", "The seminar will be held in {CITY} next week.",
 ],
 "PROVINCE_REGION": [
  "The {PROVINCE_REGION} government announced a relief package.", "Floods have affected millions in {PROVINCE_REGION}.", "Chief Minister of {PROVINCE_REGION} {PERSON} visited {CITY}.",
  "Schools across {PROVINCE_REGION} will remain closed on {DATE}.", "Wheat production in {PROVINCE_REGION} fell by {PERCENT}.", "Security forces launched an operation in {PROVINCE_REGION}.",
  "The {PROVINCE_REGION} Assembly passed the resolution unanimously.", "The highway connects {CITY} to northern {PROVINCE_REGION}.",
 ],
 "COUNTRY": [
  "Pakistan signed an agreement with {COUNTRY} on {DATE}.", "Foreign Minister {PERSON} will travel to {COUNTRY} next week.", "Exports to {COUNTRY} grew by {PERCENT}.",
  "Thousands of Pakistanis work in {COUNTRY}.", "The delegation from {COUNTRY} arrived in {CITY}.", "Relations between Pakistan and {COUNTRY} remain strong.",
  "{COUNTRY} offered {MONEY} in aid.", "A cricket series against {COUNTRY} was announced.",
 ],
 "LANDMARK_FACILITY": [
  "Thousands gathered at {LANDMARK_FACILITY} on {DATE}.", "The blast occurred near {LANDMARK_FACILITY} in {CITY}.", "Security has been tightened around {LANDMARK_FACILITY}.",
  "{PERSON} was admitted to {LANDMARK_FACILITY} with injuries.", "Tourists visit {LANDMARK_FACILITY} in large numbers every year.", "The rally will end at {LANDMARK_FACILITY}.",
  "Renovation of {LANDMARK_FACILITY} will cost {MONEY}.", "The match will be played at {LANDMARK_FACILITY}.", "Traffic was blocked on the road leading to {LANDMARK_FACILITY}.",
 ],
 "GOVERNMENT_BODY": [
  "The {GOVERNMENT_BODY} has summoned {PERSON} on {DATE}.", "A spokesperson for the {GOVERNMENT_BODY} confirmed the arrest.", "The {GOVERNMENT_BODY} issued a notification on {DATE}.",
  "The {GOVERNMENT_BODY} and the {GOVERNMENT_BODY} will hold talks.", "{PERSON} was appointed chairman of the {GOVERNMENT_BODY}.", "The {GOVERNMENT_BODY} approved {MONEY} for the scheme.",
  "Officials of the {GOVERNMENT_BODY} raided a warehouse in {CITY}.", "The {GOVERNMENT_BODY} has rejected the proposal.",
 ],
 "POLITICAL_PARTY": [
  "{PARTY} announced a protest in {CITY} on {DATE}.", "{PERSON} joined {PARTY} in {CITY}.", "The alliance of {PARTY} and {PARTY} won seats in {PROVINCE_REGION}.",
  "{PARTY} leaders met {PERSON} to discuss the coalition.", "{PARTY} has nominated {PERSON} as its candidate.", "Workers of {PARTY} staged a rally outside {GOVERNMENT_BODY}.",
  "The manifesto of {PARTY} promises a {PERCENT} cut in taxes.", "{PARTY} won the by-election in {CITY}.",
 ],
 "ORGANIZATION": [
  "{ORGANIZATION} announced a partnership with {ORGANIZATION}.", "{PERSON} is the chief executive of {ORGANIZATION}.", "{ORGANIZATION} opened a new branch in {CITY}.",
  "Employees of {ORGANIZATION} went on strike on {DATE}.", "{ORGANIZATION} donated {MONEY} to flood victims.", "The report by {ORGANIZATION} warned of a food crisis in {COUNTRY}.",
  "{ORGANIZATION} has laid off staff in {CITY}.", "Shares of {ORGANIZATION} rose {PERCENT}.",
 ],
 "EDUCATIONAL_INSTITUTION": [
  "Students of {EDUCATIONAL_INSTITUTION} protested against the fee hike.", "{PERSON} graduated from {EDUCATIONAL_INSTITUTION} in {DATE}.", "{EDUCATIONAL_INSTITUTION} has announced a scholarship worth {MONEY}.",
  "The vice chancellor of {EDUCATIONAL_INSTITUTION} met {PERSON}.", "A seminar was held at {EDUCATIONAL_INSTITUTION} in {CITY}.", "{EDUCATIONAL_INSTITUTION} will reopen on {DATE}.",
  "Admissions at {EDUCATIONAL_INSTITUTION} close on {DATE}.", "Professor {PERSON} of {EDUCATIONAL_INSTITUTION} led the research.",
 ],
 "COURT_LEGAL_BODY": [
  "The {COURT_LEGAL_BODY} reserved its verdict in the case.", "{PERSON} approached the {COURT_LEGAL_BODY} against the decision.", "The {COURT_LEGAL_BODY} dismissed the petition on {DATE}.",
  "The {COURT_LEGAL_BODY} directed the {GOVERNMENT_BODY} to submit a report.", "A bench of the {COURT_LEGAL_BODY} headed by Justice {PERSON} heard the matter.",
  "The {COURT_LEGAL_BODY} granted bail to {PERSON} against surety bonds of {MONEY}.", "The case was transferred to the {COURT_LEGAL_BODY} in {CITY}.",
  "The {COURT_LEGAL_BODY} sentenced {PERSON} to ten years in prison.", "The {COURT_LEGAL_BODY} issued notices to {PARTY} and {GOVERNMENT_BODY}.",
 ],
 "LAW_ACT": [
  "He was booked under {LAW_ACT}.", "The petition challenges {LAW_ACT}.", "The {COURT_LEGAL_BODY} declared the provision of {LAW_ACT} unconstitutional.",
  "The {GOVERNMENT_BODY} registered a case under {LAW_ACT} against {PERSON}.", "The National Assembly passed an amendment to {LAW_ACT} on {DATE}.", "Lawyers argued that {LAW_ACT} was violated.",
  "The accused is charged under {LAW_ACT} and {LAW_ACT}.", "Imposition of {LAW_ACT} in {CITY} was extended for {DATE}.", "{PERSON} was disqualified under {LAW_ACT}.",
 ],
 "EVENT": [
  "Celebrations for {EVENT} were held across {PROVINCE_REGION}.", "{EVENT} will take place in {CITY} on {DATE}.", "Security was beefed up in {CITY} for {EVENT}.",
  "{PERSON} addressed a ceremony marking {EVENT}.", "Thousands attended {EVENT} at {LANDMARK_FACILITY}.", "The government declared a public holiday on {EVENT}.",
  "{SPORTS_TEAM} reached the final of {EVENT}.", "Prices rose sharply ahead of {EVENT}.", "{EVENT} was postponed due to the security situation.",
 ],
 "MEDIA_OUTLET": [
  "Speaking to {MEDIA_OUTLET}, {PERSON} denied the allegations.", "According to {MEDIA_OUTLET}, the deal was signed on {DATE}.", "{MEDIA_OUTLET} reported that {PARTY} would boycott the session.",
  "The {GOVERNMENT_BODY} suspended the licence of {MEDIA_OUTLET}.", "{PERSON} told {MEDIA_OUTLET} that talks were under way.", "A journalist of {MEDIA_OUTLET} was injured in {CITY}.",
  "{MEDIA_OUTLET} quoted officials as saying the budget would be presented on {DATE}.", "The channel {MEDIA_OUTLET} aired the interview on {DATE}.",
 ],
 "SPORTS_TEAM": [
  "{SPORTS_TEAM} beat {SPORTS_TEAM} by five wickets at {LANDMARK_FACILITY}.", "{PERSON} was named captain of {SPORTS_TEAM}.", "{SPORTS_TEAM} will face {SPORTS_TEAM} on {DATE}.",
  "{SPORTS_TEAM} announced the squad for the tour of {COUNTRY}.", "Fans of {SPORTS_TEAM} gathered outside {LANDMARK_FACILITY}.", "{PERSON} signed a two-year deal with {SPORTS_TEAM}.",
  "{SPORTS_TEAM} lost the series against {COUNTRY} on {DATE}.", "Coach {PERSON} said {SPORTS_TEAM} were ready for {EVENT}.",
 ],
 "MIXED": [
  "On {DATE}, {PERSON} of {PARTY} told {MEDIA_OUTLET} in {CITY} that {GOVERNMENT_BODY} had released {MONEY}.",
  "The {COURT_LEGAL_BODY} in {CITY} issued notices to {GOVERNMENT_BODY} and {ORGANIZATION} on {DATE}.",
  "{PERSON}, a student of {EDUCATIONAL_INSTITUTION}, was arrested at {LANDMARK_FACILITY} on {DATE} at {TIME}.",
  "FIR No. {FIRNO} was lodged in {CITY} against {PERSON} under {LAW_ACT} on {DATE}.",
  "{ORGANIZATION} said exports to {COUNTRY} rose {PERCENT} to {MONEY} in {DATE}.",
  "Flight {FLIGHT} from {CITY} to {CITY} carrying {PERSON} landed at {TIME}.",
  "The report ({PAGE} pages) by {ORGANIZATION} was submitted to the {COURT_LEGAL_BODY} on {DATE}.",
  "{PERSON} (CNIC {ID_NUMBER}, phone {PHONE_NUMBER}) filed a petition before the {COURT_LEGAL_BODY}.",
  "Speaking at {EVENT} in {CITY}, {PERSON} said {PARTY} would contest from {PROVINCE_REGION}.",
  "{SPORTS_TEAM} owner {PERSON} announced {MONEY} in prize money ahead of {EVENT}.",
 ],
}

# ---- round-2 templates (Step 17)
T["LANDMARK_FACILITY"] += [
  "Floodwater from the {LANDMARK_FACILITY} entered villages in {PROVINCE_REGION}.", "A case was registered at the {LANDMARK_FACILITY} against {PERSON}.",
  "Police reached the {LANDMARK_FACILITY} soon after the incident.", "Residents living near {LANDMARK_FACILITY} in {CITY} complained of noise.",
  "Vendors at {LANDMARK_FACILITY} were evicted on {DATE}.", "{PERSON} visited {LANDMARK_FACILITY} to inspect the arrangements.",
  "The water level in {LANDMARK_FACILITY} rose sharply after heavy rain.", "The authority was set up to manage water from {LANDMARK_FACILITY} among the provinces.",
  "The FIR was lodged at {LANDMARK_FACILITY}, {CITY}.", "Pilgrims gathered at {LANDMARK_FACILITY} on {DATE}.",
]
T["LAW_ACT"] += [
  "The {GOVERNMENT_BODY} sought action against {PERSON} under {LAW_ACT} read with {LAW_ACT}.", "The appellant was dismissed from service under {LAW_ACT}.",
  "The tribunal held that {LAW_ACT} does not apply to the case.", "Counsel for {PERSON} relied on {LAW_ACT}.",
  "The amendment to {LAW_ACT} was challenged in the {COURT_LEGAL_BODY}.", "After the promulgation of {LAW_ACT}, the scope of the offence was widened.",
  "The respondent was charged under {LAW_ACT} read with {LAW_ACT}.", "The scope of {LAW_ACT} was examined by the {COURT_LEGAL_BODY}.",
  "{PERSON} was tried under {LAW_ACT} in {CITY}.", "The provisions of {LAW_ACT} were extended to {PROVINCE_REGION} in {DATE}.",
]
T["ORGANIZATION"] += [
  "{PERSON} was elected president of {ORGANIZATION} on {DATE}.", "{ORGANIZATION} called a countrywide strike against the decision.",
  "The members of {ORGANIZATION} met in {CITY} to discuss the matter.", "{ORGANIZATION} welcomed the ruling of the {COURT_LEGAL_BODY}.",
  "A joint statement by {ORGANIZATION} and {ORGANIZATION} was issued on {DATE}.", "The finance secretary of {ORGANIZATION} requested the bank to freeze the accounts.",
  "{ORGANIZATION} warned of a shortage in {CITY} unless the {GOVERNMENT_BODY} acted.", "{PERSON} resigned from the board of {ORGANIZATION}.",
]
T["GOVERNMENT_BODY"] += [
  "The {GOVERNMENT_BODY} was asked to review its rules.", "The {GOVERNMENT_BODY} has been told to submit a report to the {GOVERNMENT_BODY}.",
  "A summary was sent by the {GOVERNMENT_BODY} to the {GOVERNMENT_BODY} with a recommendation.", "The director general of the {GOVERNMENT_BODY} in {CITY} was removed.",
  "The chairman of the {GOVERNMENT_BODY} instructed officials to verify the complaint against {PERSON}.", "The {GOVERNMENT_BODY} approved the plan on {DATE}.",
]
T["SPORTS_TEAM"] += [
  "{SPORTS_TEAM} have called off their tour of {COUNTRY}.", "{SPORTS_TEAM} will tour {COUNTRY} in {DATE}.", "{SPORTS_TEAM} won the toss and elected to bat against {SPORTS_TEAM}.",
  "{SPORTS_TEAM} thrashed {SPORTS_TEAM} by 8 wickets in {CITY}.", "The series between {SPORTS_TEAM} and {SPORTS_TEAM} will begin on {DATE}.",
  "{PERSON} scored a century as {SPORTS_TEAM} beat {SPORTS_TEAM}.",
]
# case numbers and citations: years inside them are NOT dates (no labelled slot except context entities)
T["CASE_NOISE"] = [
  "The appeal, registered as {CASENO}, was heard by the {COURT_LEGAL_BODY}.", "In {CASENO}, the petitioner challenged the order of the {GOVERNMENT_BODY}.",
  "The case reported as {CASENO} was relied on by the learned counsel.", "Reference was made to {CASENO} decided by the {COURT_LEGAL_BODY}.",
  "{PERSON} filed {CASENO} before the {COURT_LEGAL_BODY} on {DATE}.", "The facts of {CASENO} are similar to the present case.",
  "Learned counsel for {PERSON} cited {CASENO} and {CASENO}.", "The judgment in {CASENO} was set aside.",
]

SLOT = re.compile(r"\{([A-Z_]+)\}")
# slot alias: {PARTY} -> POLITICAL_PARTY in templates for brevity
ALIAS = {"PARTY": "POLITICAL_PARTY"}


def build(template, r):
    """Fill a template -> (tokens, bio_tags, text)."""
    tokens, tags, text = [], [], ""
    pos = 0
    for m in SLOT.finditer(template):
        static = template[pos:m.start()]
        if static:
            toks = tokenize(static)
            tokens += toks; tags += ["O"] * len(toks); text += static
        name = ALIAS.get(m.group(1), m.group(1))
        if name in GEN:
            val = GEN[name](r)
            toks = tokenize(val)
            tokens += toks
            tags += [f"B-{name}"] + [f"I-{name}"] * (len(toks) - 1)
        else:
            val = NOISE[name](r)
            toks = tokenize(val)
            tokens += toks; tags += ["O"] * len(toks)
        text += val
        pos = m.end()
    tail = template[pos:]
    if tail:
        toks = tokenize(tail)
        tokens += toks; tags += ["O"] * len(toks); text += tail
    return tokens, tags, text


PLAN = {  # focus -> number of sentences (scaled by --scale)
    "PHONE_NUMBER": 400, "ID_NUMBER": 400, "MONEY": 250, "PERCENT": 220, "DATE": 220, "TIME": 260,
    "PERSON": 250, "CITY": 100, "PROVINCE_REGION": 130, "COUNTRY": 100, "LANDMARK_FACILITY": 450,
    "GOVERNMENT_BODY": 320, "POLITICAL_PARTY": 200, "ORGANIZATION": 400, "EDUCATIONAL_INSTITUTION": 260,
    "COURT_LEGAL_BODY": 280, "LAW_ACT": 450, "EVENT": 320, "MEDIA_OUTLET": 280, "SPORTS_TEAM": 420, "MIXED": 400,
    "CASE_NOISE": 150,
}


def valid_bio(tags):
    prev = "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            return False
        prev = t
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    r = random.Random(a.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    rows, seen = [], set()
    for focus, n in PLAN.items():
        templates = T[focus]
        made, tries = 0, 0
        target = int(n * a.scale)
        while made < target and tries < target * 20:
            tries += 1
            ti = r.randrange(len(templates))
            toks, tags, text = build(templates[ti], r)
            if text in seen or not valid_bio(tags):
                continue
            seen.add(text)
            rows.append(dict(tokens=toks, ner_tags=tags, text=text, source="synthetic", tier=0,
                             group=f"synthetic:{focus}:{ti}"))
            made += 1
    r.shuffle(rows)
    for i, row in enumerate(rows):
        row["id"] = f"syn{i}"
    with open(OUT, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    cnt = Counter(t[2:] for row in rows for t in row["ner_tags"] if t.startswith("B-"))
    print(f"wrote {len(rows)} sentences -> {OUT}")
    print("entity mentions per label:")
    for lab in ENTITY_TYPES:
        print(f"  {lab:24} {cnt[lab]}")
    print("\nexamples:")
    for row in rows[:4]:
        print(" ", row["text"])
        print("  ", [f"{t}/{g}" for t, g in zip(row["tokens"], row["ner_tags"]) if g != "O"])


if __name__ == "__main__":
    main()

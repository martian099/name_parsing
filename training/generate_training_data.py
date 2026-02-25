"""Generate synthetic training data for name/address NER.

Pre-processing: SpaCy tokenizes each raw word before label assignment.
This aligns with the standard BERT fine-tuning pipeline and with the
inference pipeline in model.py — both use SpaCy's blank English tokenizer.
SpaCy separates punctuation from words (e.g. "Doe," → ["Doe", ","]),
so entity words in the training data are clean from the start.

OCR noise: ~15% of words have realistic OCR character errors introduced
(substitution, deletion, insertion, character swap, bigram confusion) to
make the model robust to imperfect scanned text.

Data variations:
  Person types:
    - Individual:     "John Doe, 1234 Braddock Ave, Denver CO"
    - Shared last:    "John or Mary Doe, 1234 Braddock Ave, Denver CO"
    - Separate names: "John Doe or Mary Smith, 1234 Braddock Ave, Denver CO"
    - Business:       "Fairfax SushiMax LLC, 1234 Braddock Ave, Denver CO"
      (business first_name = business name word, last_name = LLC/Inc/Corp/etc.)

  Street types:
    - Regular:   "1234 Braddock Ave"
    - Numbered:  "1234 5th Ave"  (ordinal street names)
    - P.O. Box:  "P.O. Box 1234"  (street_name = "Box")

Output format:
    {
        "text":   "John Doe, 1234 Braddock Ave, Denver CO",
        "words":  ["John", "Doe", ",", "1234", "Braddock", "Ave", ",", "Denver", "CO"],
        "labels": [1, 3, 0, 0, 5, 0, 0, 0, 0]
    }
    where labels align with the SpaCy-tokenized words list.

Usage:
    python training/generate_training_data.py --num-examples 5000 --output data/raw/train.json
"""

import argparse
import json
import random
import string
from pathlib import Path

import spacy
from faker import Faker

# SpaCy blank English tokenizer (rule-based, no ML model download needed)
_nlp = spacy.blank("en")

fake = Faker("en_US")


# --- OCR noise ---

# Common character confusions in OCR output (visually similar glyphs)
_OCR_CHAR_MAP = {
    'l': '1', '1': 'l', 'I': '1',
    'O': '0', '0': 'O',
    'S': '5', 'Z': '2', 'G': '6',
    'B': '8', 'n': 'u', 'u': 'n',
}

# Common multi-character OCR confusions
_OCR_BIGRAMS = [
    ('rn', 'm'), ('m', 'rn'),
    ('vv', 'w'), ('w', 'vv'),
    ('cl', 'd'),
]

# Probability that any given word gets OCR noise applied
_NOISE_PROB = 0.15


def _add_ocr_noise(word: str) -> str:
    """Apply one realistic OCR error to a word.

    Chooses randomly from: character substitution, deletion, insertion,
    adjacent-character swap, or bigram substitution.
    Only applied to words with 4+ characters that contain at least one letter,
    to avoid corrupting short structural tokens (connectors, state codes, etc.).
    """
    if len(word) < 4 or not any(c.isalpha() for c in word):
        return word

    noise_type = random.choices(
        ['substitute', 'delete', 'insert', 'swap', 'bigram'],
        weights=[0.40, 0.20, 0.15, 0.10, 0.15],
    )[0]

    if noise_type == 'substitute':
        candidates = [i for i, c in enumerate(word) if c in _OCR_CHAR_MAP]
        if candidates:
            i = random.choice(candidates)
            word = word[:i] + _OCR_CHAR_MAP[word[i]] + word[i + 1:]

    elif noise_type == 'delete' and len(word) > 3:
        # Delete an interior character (preserve first/last to keep word recognizable)
        i = random.randint(1, len(word) - 2)
        word = word[:i] + word[i + 1:]

    elif noise_type == 'insert':
        # Duplicate an adjacent character (smudge / ink bleed)
        i = random.randint(1, len(word) - 1)
        word = word[:i] + word[i - 1] + word[i:]

    elif noise_type == 'swap' and len(word) >= 2:
        # Swap two adjacent characters
        i = random.randint(0, len(word) - 2)
        word = word[:i] + word[i + 1] + word[i] + word[i + 2:]

    elif noise_type == 'bigram':
        for src, dst in _OCR_BIGRAMS:
            if src in word.lower():
                idx = word.lower().index(src)
                word = word[:idx] + dst + word[idx + len(src):]
                break

    return word


def _spacy_tokenize_word(word: str) -> list[str]:
    """Tokenize a single word with SpaCy (splits off punctuation, etc.)."""
    tokens = [token.text for token in _nlp(word)]
    return tokens if tokens else [word]


# --- Name / address data lists ---

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
    "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
    "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen",
    "Charles", "Lisa", "Daniel", "Nancy", "Matthew", "Betty", "Anthony",
    "Margaret", "Mark", "Sandra", "Donald", "Ashley", "Steven", "Dorothy",
    "Paul", "Kimberly", "Andrew", "Emily", "Joshua", "Donna", "Kenneth",
    "Michelle", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca",
    "Jason", "Sharon", "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob",
    "Kathleen", "Gary", "Amy", "Nicholas", "Angela", "Eric", "Shirley",
    "Jonathan", "Anna", "Stephen", "Brenda", "Larry", "Pamela", "Justin",
    "Emma", "Scott", "Nicole", "Brandon", "Helen", "Benjamin", "Samantha",
    "Samuel", "Katherine", "Raymond", "Christine", "Gregory", "Debra",
    "Frank", "Rachel", "Alexander", "Carolyn", "Patrick", "Janet",
    "Jack", "Catherine", "Dennis", "Maria", "Jerry", "Heather",
    "Tyler", "Diane", "Aaron", "Ruth",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
    "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris",
    "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan",
    "Cooper", "Peterson", "Bailey", "Reed", "Kelly", "Howard", "Ramos",
    "Kim", "Cox", "Ward", "Richardson", "Watson", "Brooks", "Chavez",
    "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz", "Hughes",
    "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long",
    "Ross", "Foster", "Jimenez",
]

STREET_NAMES = [
    "Braddock", "Maple", "Oak", "Cedar", "Elm", "Pine", "Walnut",
    "Chestnut", "Willow", "Birch", "Magnolia", "Sycamore", "Aspen",
    "Juniper", "Cypress", "Laurel", "Poplar", "Spruce", "Hickory",
    "Redwood", "Washington", "Jefferson", "Lincoln", "Madison", "Monroe",
    "Franklin", "Hamilton", "Jackson", "Adams", "Kennedy", "Roosevelt",
    "Harrison", "Cleveland", "Garfield", "Grant", "Sherman", "Sheridan",
    "Pershing", "Patton", "Bradley", "Eisenhower", "MacArthur",
    "Highland", "Meadow", "Valley", "Summit", "Ridge", "Crest",
    "Lakeview", "Riverview", "Hillside", "Woodland", "Fairview",
    "Sunset", "Sunrise", "Clearwater", "Stonebridge", "Brookhaven",
    "Foxwood", "Greenfield", "Oakmont", "Pinehurst", "Maplewood",
    "Cedarwood", "Willowbrook", "Birchwood", "Ashford", "Westgate",
    "Eastgate", "Northgate", "Southgate", "Kensington", "Burlington",
    "Canterbury", "Devonshire", "Exeter", "Greenwich",
    "Hampton", "Lancaster", "Manchester", "Norfolk", "Oxford",
    "Princeton", "Stratford", "Victoria", "Westminster", "Windsor",
    "Harbor", "Bayview", "Cliffside", "Riverside", "Lakewood",
    "Glendale", "Thornwood", "Silverwood", "Ironwood", "Fernwood",
]

STREET_SUFFIXES = [
    "St", "Ave", "Blvd", "Dr", "Ln", "Ct", "Pl", "Way", "Rd",
    "Ter", "Cir", "Trl", "Pkwy", "Loop", "Run",
    "Street", "Avenue", "Boulevard", "Drive", "Lane", "Court",
    "Place", "Road", "Terrace", "Circle", "Trail", "Parkway",
]

DIRECTIONS = ["N", "S", "E", "W", "NE", "NW", "SE", "SW",
              "North", "South", "East", "West"]

STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

BUSINESS_NAMES = [
    # Tech
    "TechVision", "DataFirst", "NetCom", "CyberPath", "SmartCore",
    "CloudBase", "AppForge", "CodeMax", "DevPro", "SoftWave",
    "PixelForge", "ByteWorks", "DataLink", "SysPro", "TechEdge",
    # Food / restaurant
    "SushiMax", "PizzaKing", "TacoTime", "NoodleBox", "BurgerPro",
    "RamenHouse", "WokFire", "FusionBite", "SpiceRoute", "GrillMaster",
    "SteakHouse", "WingStop", "TaqueriaX", "PastaBar", "SaladWorks",
    # Professional services
    "LegalEagle", "ProBuild", "FastFix", "MedFirst", "FinServe",
    "AuditPro", "TaxMax", "LoanFirst", "InsureAll", "RealtyPro",
    "LawBridge", "CpaFirst", "TitlePro", "EscrowMax", "NotaryHub",
    # Retail / commerce
    "ShopMax", "BuyPro", "DealFirst", "ValuePlus", "BargainHub",
    "MegaMart", "QuickShop", "FreshMart", "HomeBase", "AutoMax",
    # Construction / property
    "BuildMax", "HomePro", "PropFirst", "RenovatePro", "StructMax",
    "ConcreteKing", "SteelPro", "RoofFirst", "FloorMax", "WindowPro",
    # Healthcare
    "CareFirst", "HealthMax", "MediPro", "WellnessHub", "ClinicPro",
    # Nature-inspired
    "BlueRidge", "GreenPath", "RedRock", "SilverLine", "GoldCrest",
    "IronBridge", "CopperCraft", "TimberMax", "StonePath", "WaterWorks",
    # General
    "ApexGroup", "ZenithCo", "NexusNet", "VortexPro", "OmegaCare",
    "AlphaBuilt", "PrimeFirst", "EliteServ", "TopChoice", "FirstRate",
]

BUSINESS_TYPES = ["LLC", "Inc", "Corp", "Corporation", "Ltd", "Co", "LP", "LLP"]

LOCATION_PREFIXES = [
    "Fairfax", "Arlington", "Alexandria", "Richmond", "Norfolk",
    "Springfield", "Herndon", "Reston", "McLean", "Vienna",
    "Boston", "Chicago", "Atlanta", "Houston", "Phoenix",
    "Denver", "Portland", "Seattle", "Dallas", "Miami",
    "Nashville", "Memphis", "Louisville", "Cincinnati", "Columbus",
    "Raleigh", "Charlotte", "Tampa", "Orlando", "Sacramento",
    "Tucson", "Fresno", "Oakland", "Bakersfield", "Anaheim",
]

ORDINAL_STREETS = [
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
    "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th",
    "21st", "22nd", "23rd", "24th", "25th", "30th", "40th", "50th",
]


def _random_first(): return random.choice(FIRST_NAMES)
def _random_last(): return random.choice(LAST_NAMES)
def _random_street_name(): return random.choice(STREET_NAMES)
def _random_suffix(): return random.choice(STREET_SUFFIXES)
def _random_direction(): return random.choice(DIRECTIONS)
def _random_state(): return random.choice(STATES)
def _random_business_name(): return random.choice(BUSINESS_NAMES)
def _random_business_type(): return random.choice(BUSINESS_TYPES)
def _random_location_prefix(): return random.choice(LOCATION_PREFIXES)
def _random_ordinal(): return random.choice(ORDINAL_STREETS)


def _maybe_add_middle(tokens: list, probability: float = 0.25) -> None:
    """Maybe append a middle name token (abbreviated or full) labeled O.

    Abbreviated: "M."  (single uppercase letter + period)
    Full:        "Monroe"  (any given name used as a middle name)
    """
    if random.random() >= probability:
        return
    if random.random() < 0.5:
        # Abbreviated middle initial: "M."
        tokens.append((random.choice(string.ascii_uppercase) + ".", "O"))
    else:
        # Full middle name drawn from the given-names pool
        tokens.append((_random_first(), "O"))


def _random_street_number():
    return str(random.randint(1, 99999))


def _random_zip():
    if random.random() < 0.3:
        return f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}"
    return str(random.randint(10000, 99999))


def _random_email(first: str, last: str):
    domain = random.choice(["gmail.com", "yahoo.com", "outlook.com", "aol.com",
                            "hotmail.com", "icloud.com", "mail.com"])
    sep = random.choice([".", "_", ""])
    return f"{first.lower()}{sep}{last.lower()}@{domain}"


def generate_example() -> dict:
    """Generate one training example with SpaCy-tokenized words and BIO labels.

    Flow:
      1. Build (raw_word, label) pairs for each field (name, street, city/state)
      2. Apply OCR noise to ~15% of words (simulates scanned-text imperfections)
      3. SpaCy-tokenize each (possibly noised) word into sub-tokens:
           - First sub-token keeps the word's BIO label
           - Additional sub-tokens (e.g. "," split from "Doe,") get label "O"
      4. Return {"text": original_text, "words": spacy_tokens, "labels": label_ids}

    Training and inference both operate on SpaCy-tokenized input, ensuring
    consistent label alignment between training data and inference.

    Output format:
        {
            "text":   "John Doe, 1234 Braddock Ave, Denver CO 80201",
            "words":  ["John", "Doe", ",", "1234", "Braddock", "Ave", ",", ...],
            "labels": [1, 3, 0, 0, 5, 0, 0, ...]
        }
    """
    from name_parsing.config import LABEL2ID

    # Each element: (word_string, label_string)
    # Words may have punctuation attached (e.g. "Doe,", "Ave,")
    tokens: list[tuple[str, str]] = []

    # --- Name section ---
    template = random.choices(
        ["single", "shared_last", "separate_names", "business"],
        weights=[0.30, 0.20, 0.20, 0.30],
    )[0]

    if template == "single":
        first1 = _random_first()
        last1 = _random_last()
        tokens.append((first1, "B-FIRST_NAME"))
        _maybe_add_middle(tokens, probability=0.25)
        tokens.append((last1 + ",", "B-LAST_NAME"))

    elif template == "shared_last":
        first1 = _random_first()
        first2 = _random_first()
        last1 = _random_last()
        connector = random.choice(["or", "and", "&"])
        tokens.append((first1, "B-FIRST_NAME"))
        _maybe_add_middle(tokens, probability=0.20)
        tokens.append((connector, "O"))
        tokens.append((first2, "O"))
        tokens.append((last1 + ",", "B-LAST_NAME"))

    elif template == "separate_names":
        first1 = _random_first()
        last1 = _random_last()
        first2 = _random_first()
        last2 = _random_last()
        connector = random.choice(["or", "and", "&"])
        tokens.append((first1, "B-FIRST_NAME"))
        _maybe_add_middle(tokens, probability=0.20)
        tokens.append((last1, "B-LAST_NAME"))
        tokens.append((connector, "O"))
        tokens.append((first2, "O"))
        tokens.append((last2 + ",", "O"))

    elif template == "business":
        biz_name = _random_business_name()
        biz_type = _random_business_type()
        # Optional location prefix (e.g. "Fairfax SushiMax LLC")
        if random.random() < 0.5:
            prefix = _random_location_prefix()
            tokens.append((prefix, "O"))
        tokens.append((biz_name, "B-FIRST_NAME"))
        tokens.append((biz_type + ",", "B-LAST_NAME"))

    # --- Street address ---
    street_variant = random.choices(
        ["regular", "ordinal", "po_box"],
        weights=[0.60, 0.25, 0.15],
    )[0]

    if street_variant == "po_box":
        box_num = str(random.randint(1, 9999))
        po_form = random.choice(["P.O.", "PO"])
        tokens.append((po_form, "O"))
        tokens.append(("Box", "B-STREET_NAME"))
        tokens.append((box_num + ",", "O"))
    else:
        street_num = _random_street_number()
        tokens.append((street_num, "O"))
        # Optional directional prefix (e.g. "N Braddock Ave")
        if random.random() < 0.15:
            tokens.append((_random_direction(), "O"))
        if street_variant == "ordinal":
            street = _random_ordinal()
        else:
            street = _random_street_name()
        suffix = _random_suffix()
        tokens.append((street, "B-STREET_NAME"))
        tokens.append((suffix + ",", "O"))

    # --- City, State, Zip ---
    # fake.city() may return multi-word names like "San Francisco"
    for word in fake.city().split():
        tokens.append((word, "O"))
    tokens.append((_random_state(), "O"))
    if random.random() < 0.7:
        tokens.append((_random_zip(), "O"))

    # Optional email
    if template != "business" and random.random() < 0.2:
        first_word = tokens[0][0].rstrip(",")
        last_token = next((tok for tok, lbl in tokens if lbl == "B-LAST_NAME"), "doe,")
        last_word = last_token.rstrip(",")
        tokens.append((_random_email(first_word, last_word), "O"))

    # --- Apply OCR noise, then SpaCy-tokenize each word ---
    spacy_tokens: list[tuple[str, str]] = []
    for word, label in tokens:
        # Optionally corrupt word with OCR noise (preserves the label)
        if random.random() < _NOISE_PROB:
            word = _add_ocr_noise(word)
        # SpaCy splits punctuation off (e.g. "Doe," → ["Doe", ","])
        sub_tokens = _spacy_tokenize_word(word)
        # First sub-token keeps the word's label; additional sub-tokens get "O"
        spacy_tokens.append((sub_tokens[0], label))
        for st in sub_tokens[1:]:
            spacy_tokens.append((st, "O"))

    words = [w for w, _ in spacy_tokens]
    labels = [LABEL2ID[lbl] for _, lbl in spacy_tokens]
    # Keep original (pre-noise) whitespace text for human readability
    text = " ".join(w for w, _ in tokens)

    return {"text": text, "words": words, "labels": labels}


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    parser = argparse.ArgumentParser(description="Generate synthetic NER training data")
    parser.add_argument("--num-examples", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/raw/train.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    from name_parsing.config import ID2LABEL

    examples = []
    for _ in range(args.num_examples):
        ex = generate_example()
        examples.append(ex)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Generated {len(examples)} examples -> {output_path}")

    print("\n--- Sample examples ---")
    for ex_data in examples[:5]:
        labeled = [
            (word, ID2LABEL.get(lab_id, "?"))
            for word, lab_id in zip(ex_data["words"], ex_data["labels"])
            if lab_id != 0
        ]
        print(f"Text:     {ex_data['text']}")
        print(f"Words:    {ex_data['words']}")
        print(f"Entities: {labeled}")
        print()


if __name__ == "__main__":
    main()

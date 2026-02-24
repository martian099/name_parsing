"""Generate synthetic training data for OCR customer record NER.

V3: Preprocessing-first approach.
1. Build clean text with known character spans for each entity
2. Apply OCR noise (character errors and word merging)
3. Run preprocess_ocr_text() to split merged tokens back into words
4. Tokenize with is_split_into_words=True
5. Assign BIO labels per word by checking character span membership

This matches the inference pipeline exactly: the model always sees
preprocessed, word-split text.

Usage:
    python training/generate_training_data.py --num-examples 4000 --output data/raw/train.json
"""

import argparse
import json
import random
import re
import string
from pathlib import Path

from faker import Faker
from transformers import AutoTokenizer

fake = Faker("en_US")

# --- Name lists ---

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
    "Canterbury", "Devonshire", "Exeter", "Fairfax", "Greenwich",
    "Hampton", "Lancaster", "Manchester", "Norfolk", "Oxford",
    "Princeton", "Stratford", "Victoria", "Westminster", "Windsor",
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


# --- OCR noise ---

OCR_CONFUSIONS = {
    "l": ["1", "I", "|"],
    "1": ["l", "I", "|"],
    "O": ["0", "Q"],
    "0": ["O", "Q"],
    "I": ["l", "1", "|"],
    "S": ["5", "$"],
    "5": ["S"],
    "B": ["8"],
    "8": ["B"],
    "G": ["6"],
    "6": ["G"],
    "Z": ["2"],
    "2": ["Z"],
    "m": ["rn", "nn"],
    "n": ["ri"],
    "rn": ["m"],
    "d": ["cl"],
    "w": ["vv"],
    "a": ["o"],
    "e": ["c"],
    "c": ["e"],
    "h": ["b"],
    "u": ["v"],
    "v": ["u"],
}

# Junk characters OCR might insert at merge points
MERGE_JUNK = ["", "/", "|", "-", ".", "~", "\\"]


def preprocess_ocr_text(text: str) -> str:
    """Mirror of the inference-time preprocessor. Must stay in sync with model.py."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    text = re.sub(r' +', ' ', text).strip()
    return text


def inject_ocr_noise_with_spans(
    text: str, spans: list[dict], error_rate: float = 0.03, merge_rate: float = 0.05
) -> tuple[str, list[dict]]:
    """Inject OCR character errors and space-merges, keeping spans aligned.

    Returns (noisy_text, adjusted_spans) with updated character positions.
    """
    if error_rate <= 0 and merge_rate <= 0:
        return text, spans

    old_to_new: dict[int, int] = {}
    result: list[str] = []
    new_pos = 0

    i = 0
    while i < len(text):
        old_to_new[i] = new_pos
        ch = text[i]

        if ch == " " and random.random() < merge_rate:
            junk = random.choice(MERGE_JUNK)
            if junk:
                result.append(junk)
                new_pos += len(junk)
            i += 1
            continue

        if random.random() < error_rate and ch not in (" ", ","):
            error_type = random.choices(
                ["substitute", "drop", "double"], weights=[0.6, 0.2, 0.2]
            )[0]
            if error_type == "substitute":
                bigram = text[i:i + 2] if i + 1 < len(text) else ""
                if bigram in OCR_CONFUSIONS:
                    replacement = random.choice(OCR_CONFUSIONS[bigram])
                    result.append(replacement)
                    new_pos += len(replacement)
                    old_to_new[i + 1] = new_pos
                    i += 2
                    continue
                elif ch in OCR_CONFUSIONS:
                    replacement = random.choice(OCR_CONFUSIONS[ch])
                    result.append(replacement)
                    new_pos += len(replacement)
                else:
                    result.append(ch)
                    new_pos += 1
            elif error_type == "drop":
                pass
            elif error_type == "double":
                result.append(ch)
                result.append(ch)
                new_pos += 2
        else:
            result.append(ch)
            new_pos += 1
        i += 1

    old_to_new[len(text)] = new_pos
    noisy_text = "".join(result)

    adjusted_spans = []
    for span in spans:
        new_start = old_to_new.get(span["start"], 0)
        new_end = old_to_new.get(span["end"], new_pos)
        if new_start < new_end:
            adjusted_spans.append(
                {"start": new_start, "end": new_end, "label": span["label"]}
            )

    return noisy_text, adjusted_spans


# --- Template helpers ---

class TextBuilder:
    """Build text while tracking character-level entity spans."""

    def __init__(self):
        self.parts: list[str] = []
        self.spans: list[dict] = []
        self._pos = 0

    def add(self, text: str, label: str | None = None):
        start = self._pos
        self.parts.append(text)
        self._pos += len(text)
        if label:
            self.spans.append({"start": start, "end": self._pos, "label": label})

    def add_space(self):
        self.add(" ")

    def add_comma_space(self):
        self.add(", " if random.random() < 0.85 else " ")

    def build(self) -> tuple[str, list[dict]]:
        return "".join(self.parts), self.spans


def _random_first(): return random.choice(FIRST_NAMES)
def _random_last(): return random.choice(LAST_NAMES)
def _random_street_name(): return random.choice(STREET_NAMES)
def _random_suffix(): return random.choice(STREET_SUFFIXES)
def _random_direction(): return random.choice(DIRECTIONS)
def _random_state(): return random.choice(STATES)

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

def _random_phone():
    return f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}"


def generate_example(tokenizer, ocr_noise_rate: float = 0.0, merge_rate: float = 0.0):
    """Generate one training example with word-level BIO labels.

    Pipeline:
    1. Build clean text + character spans
    2. Apply OCR noise (updates spans to match noisy positions)
    3. Run preprocess_ocr_text() to split merged tokens
    4. Tokenize with is_split_into_words=True
    5. Assign BIO labels per word by checking span membership
    """
    template = random.choices(
        ["single", "shared_last", "separate_names", "multi_names"],
        weights=[0.35, 0.30, 0.25, 0.10],
    )[0]

    first1 = _random_first()
    last1 = _random_last()
    tb = TextBuilder()

    if template == "single":
        has_middle = random.random() < 0.15
        tb.add(first1, "FIRST_NAME")
        tb.add_space()
        if has_middle:
            mi = random.choice(string.ascii_uppercase) + ("." if random.random() < 0.7 else "")
            tb.add(mi)
            tb.add_space()
        tb.add(last1, "LAST_NAME")

    elif template == "shared_last":
        first2 = _random_first()
        connector = random.choice([" or ", " & ", " and ", " / "])
        tb.add(first1, "FIRST_NAME")
        tb.add(connector)
        tb.add(first2)
        tb.add_space()
        tb.add(last1, "LAST_NAME")

    elif template == "separate_names":
        first2 = _random_first()
        last2 = _random_last()
        connector = random.choice([" or ", " & ", " and ", " / "])
        tb.add(first1, "FIRST_NAME")
        tb.add_space()
        tb.add(last1, "LAST_NAME")
        tb.add(connector)
        tb.add(first2)
        tb.add_space()
        tb.add(last2)

    elif template == "multi_names":
        tb.add(first1, "FIRST_NAME")
        tb.add_space()
        tb.add(last1, "LAST_NAME")
        for _ in range(random.randint(1, 3)):
            tb.add(random.choice([", ", " and ", " & ", " / "]))
            tb.add(_random_first())
            tb.add_space()
            tb.add(_random_last())

    # Address
    tb.add_comma_space()
    tb.add(_random_street_number())
    tb.add_space()
    if random.random() < 0.2:
        tb.add(_random_direction())
        tb.add_space()
    tb.add(_random_street_name(), "STREET_NAME")
    tb.add_space()
    tb.add(_random_suffix())

    # City, State, Zip
    tb.add_comma_space()
    tb.add(fake.city())
    tb.add_space()
    tb.add(_random_state())
    if random.random() < 0.8:
        tb.add_space()
        tb.add(_random_zip())

    # Optional email / phone
    if random.random() < 0.3:
        tb.add_comma_space()
        tb.add(_random_email(first1, last1))
    if random.random() < 0.1:
        tb.add_comma_space()
        tb.add(_random_phone())

    text, spans = tb.build()

    # Step 2: OCR noise (spans are updated to match noisy character positions)
    if ocr_noise_rate > 0 or merge_rate > 0:
        text, spans = inject_ocr_noise_with_spans(text, spans, ocr_noise_rate, merge_rate)

    # Step 3: Preprocess — same function as used at inference time
    preprocessed = preprocess_ocr_text(text)
    words = preprocessed.split()

    # Step 4 & 5: Tokenize (for word_ids) and assign BIO labels per word
    from name_parsing.config import LABEL2ID

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=64,
        return_offsets_mapping=False,
    )
    word_ids = encoding.word_ids()

    # For each entity span, preprocess its substring and find the matching words
    # in the word list by consecutive case-insensitive match.
    entity_word_sets: list[tuple[int, int, str]] = []  # (word_start_idx, word_end_idx, label)

    for span in spans:
        # Get the entity text from the noisy text
        entity_text_noisy = text[span["start"]:span["end"]]
        # Preprocess the entity substring the same way
        entity_text_pre = preprocess_ocr_text(entity_text_noisy)
        if not entity_text_pre:
            continue
        entity_words = entity_text_pre.split()
        if not entity_words:
            continue

        # Find these entity words in the word list (case-insensitive search)
        first_word = entity_words[0].lower()
        for w_idx, word in enumerate(words):
            if word.lower() == first_word:
                # Check if all entity words match consecutively
                if all(
                    w_idx + j < len(words) and words[w_idx + j].lower() == entity_words[j].lower()
                    for j in range(len(entity_words))
                ):
                    entity_word_sets.append(
                        (w_idx, w_idx + len(entity_words) - 1, span["label"])
                    )
                    break

    # Build a per-word label map: word_idx -> label string
    word_label: dict[int, str] = {}
    for w_start, w_end, label in entity_word_sets:
        for w_idx in range(w_start, w_end + 1):
            if w_idx == w_start:
                word_label[w_idx] = f"B-{label}"
            else:
                word_label[w_idx] = f"I-{label}"

    # Assign label IDs per word (one per word, aligned with word list)
    # Labels are stored at word granularity — train.py will re-expand to subtokens.
    word_labels = [
        LABEL2ID.get(word_label.get(w_idx, "O"), 0)
        for w_idx in range(len(words))
    ]

    return {
        "text": text,
        "preprocessed": preprocessed,
        "labels": word_labels,
    }


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    parser = argparse.ArgumentParser(description="Generate synthetic NER training data")
    parser.add_argument("--num-examples", type=int, default=4000)
    parser.add_argument("--output", type=str, default="data/raw/train.json")
    parser.add_argument("--noise-rate", type=float, default=0.03)
    parser.add_argument("--merge-rate", type=float, default=0.05)
    parser.add_argument("--clean-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    from name_parsing.config import BASE_MODEL_NAME, ID2LABEL
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    examples = []
    for i in range(args.num_examples):
        if random.random() < args.clean_fraction:
            noise, merge = 0.0, 0.0
        else:
            noise = random.uniform(0.01, args.noise_rate * 2)
            merge = random.uniform(0.01, args.merge_rate * 2)
        ex = generate_example(tokenizer, ocr_noise_rate=noise, merge_rate=merge)
        examples.append(ex)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Generated {len(examples)} examples -> {output_path}")

    print("\n--- Sample examples ---")
    for ex_data in examples[:5]:
        words = ex_data["preprocessed"].split()
        labeled = [
            (word, ID2LABEL.get(lab_id, "?"))
            for word, lab_id in zip(words, ex_data["labels"])
            if lab_id not in (0,)
        ]
        print(f"Text: {ex_data['text']}")
        print(f"Preprocessed: {ex_data['preprocessed']}")
        print(f"Entities: {labeled}")
        print()


if __name__ == "__main__":
    main()

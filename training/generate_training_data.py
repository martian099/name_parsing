"""Generate synthetic training data for OCR customer record NER.

Produces subword-level BIO-tagged examples covering common OCR-scanned
customer record formats with OCR noise injection including word merging/splitting.

The new approach works at the character-span level:
1. Build clean text with known character spans for each entity
2. Apply OCR noise (including merging adjacent words)
3. Tokenize with DistilBERT's WordPiece tokenizer
4. Assign BIO labels to each subword based on character overlap with entity spans

This means the model learns to handle merged tokens like "37/harbor" or
"johndoe" because each subword gets its own label.

Usage:
    python training/generate_training_data.py --num-examples 4000 --output data/raw/train.json
"""

import argparse
import json
import random
import string
from pathlib import Path

from faker import Faker
from transformers import AutoTokenizer

fake = Faker("en_US")

# --- Name lists (common US names for diversity) ---

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


# --- OCR noise injection ---

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

# Junk characters that OCR might insert between merged words
MERGE_JUNK = ["", "/", "|", "-", ".", "~", "\\"]


def inject_ocr_noise_with_spans(text: str, spans: list[dict], error_rate: float = 0.03,
                                 merge_rate: float = 0.05) -> tuple[str, list[dict]]:
    """Inject OCR errors and return updated text + adjusted spans.

    This operates character-by-character, tracking how each original position
    maps to the new position so entity spans stay aligned.

    Args:
        text: Original clean text
        spans: List of {"start": int, "end": int, "label": str}
        error_rate: Per-character error probability
        merge_rate: Probability of removing a space (merging adjacent words)

    Returns:
        (noisy_text, adjusted_spans)
    """
    if error_rate <= 0 and merge_rate <= 0:
        return text, spans

    # Build a mapping: old_pos -> new_pos
    old_to_new = {}
    result = []
    new_pos = 0

    i = 0
    while i < len(text):
        old_to_new[i] = new_pos
        ch = text[i]

        # Space merging: remove space to merge adjacent words
        if ch == " " and random.random() < merge_rate:
            # Sometimes insert junk between merged words
            junk = random.choice(MERGE_JUNK)
            if junk:
                result.append(junk)
                new_pos += len(junk)
            i += 1
            continue

        # Character-level noise (skip spaces and commas)
        if random.random() < error_rate and ch not in (" ", ","):
            error_type = random.choices(
                ["substitute", "drop", "double"],
                weights=[0.6, 0.2, 0.2],
            )[0]

            if error_type == "substitute":
                bigram = text[i:i+2] if i + 1 < len(text) else ""
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
                pass  # skip char, don't advance new_pos
            elif error_type == "double":
                result.append(ch)
                result.append(ch)
                new_pos += 2
        else:
            result.append(ch)
            new_pos += 1
        i += 1

    # Map the end position too
    old_to_new[len(text)] = new_pos

    noisy_text = "".join(result)

    # Adjust spans using the mapping
    adjusted_spans = []
    for span in spans:
        new_start = old_to_new.get(span["start"], 0)
        new_end = old_to_new.get(span["end"], new_pos)
        if new_start < new_end:
            adjusted_spans.append({
                "start": new_start,
                "end": new_end,
                "label": span["label"],
            })

    return noisy_text, adjusted_spans


# --- Template generators ---

def _random_first():
    return random.choice(FIRST_NAMES)

def _random_last():
    return random.choice(LAST_NAMES)

def _random_middle_initial():
    return random.choice(string.ascii_uppercase)

def _random_street_number():
    return str(random.randint(1, 99999))

def _random_street_name():
    return random.choice(STREET_NAMES)

def _random_suffix():
    return random.choice(STREET_SUFFIXES)

def _random_direction():
    return random.choice(DIRECTIONS)

def _random_city():
    return fake.city()

def _random_state():
    return random.choice(STATES)

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


class TextBuilder:
    """Build text while tracking character-level entity spans."""

    def __init__(self):
        self.parts: list[str] = []
        self.spans: list[dict] = []
        self._pos = 0

    def add(self, text: str, label: str | None = None):
        """Append text, optionally marking it with an entity label."""
        start = self._pos
        self.parts.append(text)
        self._pos += len(text)
        if label:
            self.spans.append({"start": start, "end": self._pos, "label": label})

    def add_space(self):
        self.add(" ")

    def add_comma_space(self):
        if random.random() < 0.85:
            self.add(", ")
        else:
            self.add(" ")

    def build(self) -> tuple[str, list[dict]]:
        return "".join(self.parts), self.spans


def generate_example(tokenizer, ocr_noise_rate: float = 0.0, merge_rate: float = 0.0):
    """Generate a single training example with subword-level BIO labels.

    Returns dict with keys: text, input_ids, labels
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
            mi = _random_middle_initial() + ("." if random.random() < 0.7 else "")
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
        num_extra = random.randint(1, 3)
        for _ in range(num_extra):
            sep = random.choice([", ", " and ", " & ", " / "])
            tb.add(sep)
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
    tb.add(_random_city())
    tb.add_space()
    tb.add(_random_state())
    if random.random() < 0.8:
        tb.add_space()
        tb.add(_random_zip())

    # Optional email
    if random.random() < 0.3:
        tb.add_comma_space()
        tb.add(_random_email(first1, last1))

    # Optional phone
    if random.random() < 0.1:
        tb.add_comma_space()
        tb.add(_random_phone())

    text, spans = tb.build()

    # Apply OCR noise (including word merging)
    if ocr_noise_rate > 0 or merge_rate > 0:
        text, spans = inject_ocr_noise_with_spans(text, spans, ocr_noise_rate, merge_rate)

    # Tokenize the raw text
    encoding = tokenizer(text, truncation=True, max_length=64, return_offsets_mapping=True)
    input_ids = encoding["input_ids"]
    offset_mapping = encoding["offset_mapping"]

    # Assign BIO labels to each subword token based on character overlap
    labels = []
    for token_idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:
            # Special token ([CLS], [SEP])
            labels.append(-100)
            continue

        # Find which entity span (if any) this subword overlaps with
        matched_label = None
        for span in spans:
            # Check if this subword's characters overlap with the entity span
            overlap_start = max(tok_start, span["start"])
            overlap_end = min(tok_end, span["end"])
            if overlap_start < overlap_end:
                # There's overlap — check if it's mostly entity characters
                overlap_len = overlap_end - overlap_start
                token_len = tok_end - tok_start
                if overlap_len / token_len >= 0.5:
                    matched_label = span["label"]
                    break

        if matched_label is None:
            labels.append(0)  # O
        else:
            # Determine B- vs I-: is this the first subword of this entity?
            is_begin = True
            if token_idx > 0:
                prev_start, prev_end = offset_mapping[token_idx - 1]
                if prev_start != prev_end:  # not special token
                    for span in spans:
                        if span["label"] == matched_label:
                            prev_overlap_start = max(prev_start, span["start"])
                            prev_overlap_end = min(prev_end, span["end"])
                            prev_overlap_len = prev_overlap_end - prev_overlap_start
                            prev_token_len = prev_end - prev_start
                            if prev_token_len > 0 and prev_overlap_len / prev_token_len >= 0.5:
                                is_begin = False
                                break

            if is_begin:
                label_str = f"B-{matched_label}"
            else:
                label_str = f"I-{matched_label}"

            from name_parsing.config import LABEL2ID
            labels.append(LABEL2ID.get(label_str, 0))

    return {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": encoding["attention_mask"],
        "labels": labels,
        "spans": spans,  # keep for debugging/evaluation
    }


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    parser = argparse.ArgumentParser(description="Generate synthetic NER training data")
    parser.add_argument("--num-examples", type=int, default=4000)
    parser.add_argument("--output", type=str, default="data/raw/train.json")
    parser.add_argument("--noise-rate", type=float, default=0.03,
                        help="OCR character noise rate")
    parser.add_argument("--merge-rate", type=float, default=0.05,
                        help="Probability of removing spaces (merging words)")
    parser.add_argument("--clean-fraction", type=float, default=0.3,
                        help="Fraction of examples with no noise")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    Faker.seed(args.seed)

    from name_parsing.config import BASE_MODEL_NAME, ID2LABEL
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    examples = []
    for i in range(args.num_examples):
        if random.random() < args.clean_fraction:
            noise = 0.0
            merge = 0.0
        else:
            noise = random.uniform(0.01, args.noise_rate * 2)
            merge = random.uniform(0.01, args.merge_rate * 2)
        ex = generate_example(tokenizer, ocr_noise_rate=noise, merge_rate=merge)

        # Convert to serializable format (drop spans for smaller file)
        examples.append({
            "text": ex["text"],
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["labels"],
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"Generated {len(examples)} examples -> {output_path}")

    # Print a few samples with subword labels
    print("\n--- Sample examples ---")
    for ex_data in examples[:5]:
        tokens = tokenizer.convert_ids_to_tokens(ex_data["input_ids"])
        labeled = [
            (tok, ID2LABEL.get(lab, "?"))
            for tok, lab in zip(tokens, ex_data["labels"])
            if lab not in (-100, 0)
        ]
        print(f"Text: {ex_data['text']}")
        print(f"Entities: {labeled}")
        print()


if __name__ == "__main__":
    main()

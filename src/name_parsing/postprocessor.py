"""Post-process NER predictions to extract structured fields.

Works with raw (un-preprocessed) words and word_ids() from is_split_into_words=True
tokenization. Words may have punctuation attached (e.g. "Doe,", "Ave,") since the
model is trained on raw text. Punctuation is stripped from entity words before output.

For each word, the prediction of its FIRST subtoken is used (standard HuggingFace
NER pattern). The BIO state machine then runs over words, not subtokens.
"""

import re

from name_parsing.config import GENERIC_STREET_WORDS, ID2LABEL


def _clean_word(word: str) -> str:
    """Strip leading and trailing punctuation from a raw word."""
    return word.strip(".,;:!?\"'()[]{}/-\\")


def _clean_name_word(word: str) -> str:
    """Clean a name word: strip punctuation, possessives, and single-letter dot-prefixes.

    Examples:
        "GG's"        -> "GG"
        "A.Professional" -> "Professional"
        "Smith,"      -> "Smith"
    """
    word = _clean_word(word)
    # Strip possessive 's (e.g. "GG's" -> "GG")
    word = re.sub(r"'s$", "", word, flags=re.IGNORECASE)
    # Strip leading single-letter prefix followed by a dot (e.g. "A.Professional" -> "Professional")
    word = re.sub(r"^[A-Za-z]\.", "", word)
    return word


def extract_entities(
    predictions: list[int],
    words: list[str],
    word_ids: list[int | None],
) -> dict[str, list[list[str]]]:
    """Extract entity spans from BIO predictions on word-level tokens.

    Takes the first subtoken prediction for each word, then runs a BIO
    state machine over the word list.

    Args:
        predictions: Predicted label IDs, one per subtoken (including special tokens).
        words: The whitespace-split word list fed to the tokenizer.
        word_ids: Output of encoding.word_ids() — maps each subtoken position
                  to its word index, or None for special tokens ([CLS], [SEP]).

    Returns:
        Dict mapping entity type to list of entity spans, where each span
        is a list of raw word strings (may include punctuation).
    """
    # Collect the first-subtoken prediction for each word index
    word_predictions: dict[int, int] = {}
    for pred_id, word_idx in zip(predictions, word_ids):
        if word_idx is None:
            continue
        if word_idx not in word_predictions:  # first subtoken wins
            word_predictions[word_idx] = pred_id

    entities: dict[str, list[list[str]]] = {
        "FIRST_NAME": [],
        "LAST_NAME": [],
        "STREET_NAME": [],
    }
    current_entity: str | None = None
    current_words: list[str] = []

    for word_idx, word in enumerate(words):
        pred_id = word_predictions.get(word_idx, 0)
        label = ID2LABEL.get(pred_id, "O")

        if label.startswith("B-"):
            if current_entity and current_words:
                entities[current_entity].append(current_words)
            entity_type = label[2:]
            if entity_type in entities:
                current_entity = entity_type
                current_words = [word]
            else:
                current_entity = None
                current_words = []

        elif label.startswith("I-") and current_entity == label[2:]:
            current_words.append(word)

        else:
            if current_entity and current_words:
                entities[current_entity].append(current_words)
            current_entity = None
            current_words = []

    if current_entity and current_words:
        entities[current_entity].append(current_words)

    return entities


def _is_numeric(word: str) -> bool:
    """Return True if the word is a pure number (not an ordinal like '5th')."""
    if not word:
        return False
    # Ordinal numbers (1st, 2nd, 3rd, 5th, etc.) are valid street names
    if re.match(r'^\d+(st|nd|rd|th)$', word, re.IGNORECASE):
        return False
    digit_count = sum(1 for c in word if c.isdigit())
    return digit_count / len(word) >= 0.5


def filter_street_name(street_spans: list[list[str]]) -> str:
    """Pick the most distinctive street name word from extracted spans.

    Flattens all spans into individual words, strips punctuation, filters
    out generic street suffixes/directions and pure numeric tokens, then
    returns the first distinctive word.

    Ordinal numbers (1st, 5th, etc.) are kept as valid street names.
    "Box" is kept for P.O. Box addresses.
    """
    if not street_spans:
        return ""

    all_words = [word for span in street_spans for word in span]

    if not all_words:
        return ""

    distinct = []
    for raw_word in all_words:
        cleaned = _clean_word(raw_word)
        if cleaned.lower() not in GENERIC_STREET_WORDS and not _is_numeric(cleaned):
            distinct.append(cleaned)

    return distinct[0] if distinct else _clean_word(all_words[0])


def postprocess(
    predictions: list[int],
    words: list[str],
    word_ids: list[int | None],
) -> dict[str, str]:
    """Full post-processing pipeline: extract entities and build result dict.

    Punctuation is stripped from all entity words (e.g. "Doe," -> "Doe").
    Street names additionally have generic type words filtered out.

    Args:
        predictions: Predicted label IDs per subtoken.
        words: Whitespace-split word list from raw text.
        word_ids: Subtoken-to-word-index mapping from encoding.word_ids().

    Returns:
        {"first_name": "...", "last_name": "...", "street_name": "..."}
    """
    entities = extract_entities(predictions, words, word_ids)

    first_name = (
        " ".join(_clean_name_word(w) for w in entities["FIRST_NAME"][0])
        if entities["FIRST_NAME"] else ""
    )
    last_name = (
        " ".join(_clean_name_word(w) for w in entities["LAST_NAME"][0])
        if entities["LAST_NAME"] else ""
    )
    street_name = filter_street_name(entities["STREET_NAME"])

    return {
        "first_name": first_name,
        "last_name": last_name,
        "street_name": street_name,
    }

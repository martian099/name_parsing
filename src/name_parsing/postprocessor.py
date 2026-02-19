"""Post-process NER predictions to extract structured fields.

V2: Works with subword tokens and offset_mapping instead of word_ids.
Reconstructs entities by joining consecutive B/I-tagged subword tokens,
stripping WordPiece prefixes (##). Uses character offsets to detect word
boundaries — if there's a gap between consecutive subword tokens, it means
there was a space in the original text and we insert one when joining.
"""

from name_parsing.config import GENERIC_STREET_WORDS, ID2LABEL


# Each collected token is (clean_text, char_start, char_end)
TokenInfo = tuple[str, int, int]


def extract_entities(
    predictions: list[int],
    tokens: list[str],
    offset_mapping: list[tuple[int, int]],
) -> dict[str, list[list[TokenInfo]]]:
    """Extract entity spans from BIO predictions on subword tokens.

    Args:
        predictions: Predicted label IDs for each subword token.
        tokens: Subword token strings (from tokenizer.convert_ids_to_tokens).
        offset_mapping: List of (start, end) character offsets for each token.

    Returns:
        Dict mapping entity type ("FIRST_NAME", "LAST_NAME", "STREET_NAME")
        to list of entity spans, where each span is a list of
        (clean_text, char_start, char_end) tuples.
    """
    entities: dict[str, list[list[TokenInfo]]] = {
        "FIRST_NAME": [],
        "LAST_NAME": [],
        "STREET_NAME": [],
    }

    current_entity = None
    current_tokens: list[TokenInfo] = []

    for idx, (pred_id, token, (tok_start, tok_end)) in enumerate(
        zip(predictions, tokens, offset_mapping)
    ):
        # Skip special tokens ([CLS], [SEP], [PAD])
        if tok_start == tok_end:
            if current_entity and current_tokens:
                entities[current_entity].append(current_tokens)
            current_entity = None
            current_tokens = []
            continue

        label = ID2LABEL.get(pred_id, "O")

        # Clean subword token: strip ## prefix
        clean_token = token.lstrip("#") if token.startswith("##") else token

        if label.startswith("B-"):
            # Start of a new entity — save previous if exists
            if current_entity and current_tokens:
                entities[current_entity].append(current_tokens)
            entity_type = label[2:]
            if entity_type in entities:
                current_entity = entity_type
                current_tokens = [(clean_token, tok_start, tok_end)]
            else:
                current_entity = None
                current_tokens = []

        elif label.startswith("I-") and current_entity:
            entity_type = label[2:]
            if entity_type == current_entity:
                current_tokens.append((clean_token, tok_start, tok_end))
            else:
                # Mismatched I- tag, close current entity
                entities[current_entity].append(current_tokens)
                current_entity = None
                current_tokens = []
        else:
            # O tag or -100
            if current_entity and current_tokens:
                entities[current_entity].append(current_tokens)
            current_entity = None
            current_tokens = []

    # Don't forget last entity
    if current_entity and current_tokens:
        entities[current_entity].append(current_tokens)

    return entities


def _join_token_infos(token_infos: list[TokenInfo]) -> str:
    """Join subword tokens into text, inserting spaces where the original had gaps.

    If consecutive tokens have a gap in their character offsets (i.e., the
    next token's start > the previous token's end), that means there was a
    space (or other separator) in the original text. We insert a space there.
    If they're adjacent (next start == prev end), they were part of the same
    word and get concatenated directly.
    """
    if not token_infos:
        return ""

    parts = [token_infos[0][0]]
    for i in range(1, len(token_infos)):
        prev_end = token_infos[i - 1][2]
        curr_start = token_infos[i][1]
        if curr_start > prev_end:
            # There was a gap (space) between these tokens in the original text
            parts.append(" ")
        parts.append(token_infos[i][0])

    return "".join(parts)


def filter_street_name(street_spans: list[list[TokenInfo]]) -> str:
    """Pick the most distinctive street name word from extracted spans.

    Joins subword tokens within each span (gap-aware), splits multi-word
    results into individual words, then filters out generic street words
    and numeric tokens. Returns the first distinctive word.
    """
    if not street_spans:
        return ""

    # Join subword tokens within each span, then split into individual words
    all_words = []
    for span in street_spans:
        joined = _join_token_infos(span)
        # Split on spaces in case the span covers multiple words
        for word in joined.split():
            if word:
                all_words.append(word)

    if not all_words:
        return ""

    def _is_numeric(s: str) -> bool:
        """Check if token looks like a street number (mostly digits, possibly OCR-mangled)."""
        cleaned = s.rstrip(".,").replace("-", "")
        if not cleaned:
            return False
        digit_count = sum(1 for c in cleaned if c.isdigit())
        return digit_count / len(cleaned) >= 0.5

    # Filter out generic words and numeric tokens
    distinct = [
        w for w in all_words
        if w.lower().rstrip(".,") not in GENERIC_STREET_WORDS and not _is_numeric(w)
    ]

    if distinct:
        return distinct[0]
    return all_words[0]


def postprocess(
    predictions: list[int],
    tokens: list[str],
    offset_mapping: list[tuple[int, int]],
) -> dict[str, str]:
    """Full post-processing pipeline: extract entities and build result dict.

    Args:
        predictions: Predicted label IDs per subword token.
        tokens: Subword token strings.
        offset_mapping: Character offsets per subword token.

    Returns:
        {"first_name": "...", "last_name": "...", "street_name": "..."}
    """
    entities = extract_entities(predictions, tokens, offset_mapping)

    # Take first occurrence of each name entity, join subwords (gap-aware)
    first_name = ""
    if entities["FIRST_NAME"]:
        first_name = _join_token_infos(entities["FIRST_NAME"][0])

    last_name = ""
    if entities["LAST_NAME"]:
        last_name = _join_token_infos(entities["LAST_NAME"][0])

    street_name = filter_street_name(entities["STREET_NAME"])

    # Clean up any trailing punctuation from OCR
    first_name = first_name.rstrip(".,;:")
    last_name = last_name.rstrip(".,;:")
    street_name = street_name.rstrip(".,;:")

    return {
        "first_name": first_name,
        "last_name": last_name,
        "street_name": street_name,
    }

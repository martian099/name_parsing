"""Post-process NER predictions to extract structured fields.

With SpaCy pre-tokenization, input words are already clean — punctuation is
split into separate tokens and labeled "O". The postprocessor's only job is:
  1. Extract BIO entity spans from predictions (using first-subtoken-per-word rule)
  2. Return the first span per entity category, joining its words

No additional filtering, cleaning, or generic-word removal is applied.
The raw model predictions are returned directly.

For each word, the prediction of its FIRST subtoken is used (standard HuggingFace
NER pattern). The BIO state machine then runs over words, not subtokens.
"""

from name_parsing.config import ID2LABEL


def extract_entities(
    predictions: list[int],
    words: list[str],
    word_ids: list[int | None],
) -> dict[str, list[list[str]]]:
    """Extract entity spans from BIO predictions on SpaCy-tokenized words.

    Takes the first-subtoken prediction for each word, then runs a BIO
    state machine over the word list.

    Args:
        predictions: Predicted label IDs, one per subtoken (including special tokens).
        words: The SpaCy-tokenized word list fed to the tokenizer.
        word_ids: Output of encoding.word_ids() — maps each subtoken position
                  to its word index, or None for special tokens ([CLS], [SEP]).

    Returns:
        Dict mapping entity type to list of entity spans, where each span
        is a list of word strings.
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


def postprocess(
    predictions: list[int],
    words: list[str],
    word_ids: list[int | None],
) -> dict[str, str]:
    """Extract the first entity span per category and return as strings.

    No additional post-processing is applied beyond selecting the first span.
    With SpaCy pre-tokenization, punctuation is already a separate "O" token,
    so entity words are already clean.

    Args:
        predictions: Predicted label IDs per subtoken.
        words: SpaCy-tokenized word list from input text.
        word_ids: Subtoken-to-word-index mapping from encoding.word_ids().

    Returns:
        {"first_name": "...", "last_name": "...", "street_name": "..."}
    """
    entities = extract_entities(predictions, words, word_ids)

    first_name = " ".join(entities["FIRST_NAME"][0]) if entities["FIRST_NAME"] else ""
    last_name = " ".join(entities["LAST_NAME"][0]) if entities["LAST_NAME"] else ""
    street_name = " ".join(entities["STREET_NAME"][0]) if entities["STREET_NAME"] else ""

    return {
        "first_name": first_name,
        "last_name": last_name,
        "street_name": street_name,
    }

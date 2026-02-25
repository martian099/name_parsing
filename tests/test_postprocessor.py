"""Tests for the post-processing pipeline.

With SpaCy pre-tokenization, words passed to the model are already clean:
punctuation is a separate "O" token, not attached to entity words. The
postprocessor extracts the first BIO span per entity type and joins words
directly — no additional cleaning is applied.

Tests pass words and word_ids directly — no tokenizer needed.
word_ids mirrors what encoding.word_ids() returns: None for special tokens,
an int index for each subtoken pointing to its source word.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from name_parsing.config import LABEL2ID
from name_parsing.postprocessor import (
    extract_entities,
    postprocess,
)


def _make_word_ids(words: list[str], subtokens_per_word: list[int] | None = None) -> list[int | None]:
    """Build a word_ids list as if from encoding.word_ids().

    By default each word produces exactly one subtoken.
    Pass subtokens_per_word to simulate multi-subtoken words.
    Always wraps with None for [CLS] and [SEP].
    """
    if subtokens_per_word is None:
        subtokens_per_word = [1] * len(words)
    ids: list[int | None] = [None]  # [CLS]
    for word_idx, n in enumerate(subtokens_per_word):
        ids.extend([word_idx] * n)
    ids.append(None)  # [SEP]
    return ids


def _make_predictions(words: list[str], labels: list[str],
                      subtokens_per_word: list[int] | None = None) -> list[int]:
    """Build a predictions list aligned with word_ids.

    For multi-subtoken words, the first subtoken gets the real label
    and subsequent subtokens get 0 (O).
    Wraps with 0 for [CLS] and [SEP] special token positions.
    """
    if subtokens_per_word is None:
        subtokens_per_word = [1] * len(words)
    preds: list[int] = [0]  # [CLS] -> ignored (word_id=None)
    for label, n in zip(labels, subtokens_per_word):
        preds.append(LABEL2ID[label])
        preds.extend([0] * (n - 1))  # continuation subtokens: O
    preds.append(0)  # [SEP]
    return preds


class TestExtractEntities:
    def test_single_name_spacy_words(self):
        """SpaCy-tokenized words: comma is a separate 'O' token, not attached."""
        words = ["Alex", "Doe", ",", "1201", "Braddock", "Ave", ","]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "B-STREET_NAME", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["Alex"]]
        assert entities["LAST_NAME"] == [["Doe"]]
        assert entities["STREET_NAME"] == [["Braddock"]]

    def test_shared_last_name(self):
        words = ["Alex", "or", "Mary", "Doe", ",", "1201", "Braddock", "Ave", ","]
        labels = ["B-FIRST_NAME", "O", "O", "B-LAST_NAME", "O", "O", "B-STREET_NAME", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["Alex"]]
        assert entities["LAST_NAME"] == [["Doe"]]
        assert entities["STREET_NAME"] == [["Braddock"]]

    def test_separate_names(self):
        words = ["Alex", "Doe", "or", "Mary", "Smith", ",", "500", "Oak", "Ave", ","]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "O", "O", "O", "B-STREET_NAME", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["Alex"]]
        assert entities["LAST_NAME"] == [["Doe"]]
        assert entities["STREET_NAME"] == [["Oak"]]

    def test_business_payor(self):
        """Business: first_name=business word, last_name=LLC/Inc/etc."""
        words = ["Fairfax", "SushiMax", "LLC", ",", "1201", "Braddock", "Ave", ","]
        labels = ["O", "B-FIRST_NAME", "B-LAST_NAME", "O", "O", "B-STREET_NAME", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["SushiMax"]]
        assert entities["LAST_NAME"] == [["LLC"]]
        assert entities["STREET_NAME"] == [["Braddock"]]

    def test_multi_subtoken_word_uses_first_subtoken(self):
        """Word 'Braddock' splits into 2 BERT subtokens; first subtoken label wins."""
        words = ["Braddock"]
        labels = ["B-STREET_NAME"]
        preds = _make_predictions(words, labels, subtokens_per_word=[2])
        word_ids = _make_word_ids(words, subtokens_per_word=[2])

        entities = extract_entities(preds, words, word_ids)

        assert entities["STREET_NAME"] == [["Braddock"]]

    def test_special_tokens_only(self):
        preds = [0, 0]
        words = []
        word_ids = [None, None]

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == []
        assert entities["LAST_NAME"] == []
        assert entities["STREET_NAME"] == []

    def test_mismatched_I_tag_closes_entity(self):
        """I-LAST_NAME after B-FIRST_NAME should close the first entity."""
        words = ["Alex", "Doe"]
        labels = ["B-FIRST_NAME", "I-LAST_NAME"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["Alex"]]
        assert entities["LAST_NAME"] == []


class TestPostprocess:
    def test_single_name_spacy_tokenized(self):
        """SpaCy already separated 'Doe,' into 'Doe' + ','; output is clean."""
        words = ["Alex", "Doe", ",", "1201", "Braddock", "Ave", ",", "Richmond", "VA"]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "B-STREET_NAME", "O", "O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "Alex"
        assert result["last_name"] == "Doe"
        assert result["street_name"] == "Braddock"

    def test_shared_last_name(self):
        words = ["Alex", "or", "Mary", "Doe", ",", "1201", "Braddock", "Ave", ",",
                 "Richmond", "VA", "22312"]
        labels = ["B-FIRST_NAME", "O", "O", "B-LAST_NAME", "O", "O",
                  "B-STREET_NAME", "O", "O", "O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "Alex"
        assert result["last_name"] == "Doe"
        assert result["street_name"] == "Braddock"

    def test_business_payor_output(self):
        words = ["Fairfax", "SushiMax", "LLC", ",", "1201", "Braddock", "Ave", ",", "Denver", "CO"]
        labels = ["O", "B-FIRST_NAME", "B-LAST_NAME", "O", "O", "B-STREET_NAME", "O", "O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "SushiMax"
        assert result["last_name"] == "LLC"
        assert result["street_name"] == "Braddock"

    def test_ordinal_street(self):
        words = ["John", "Doe", ",", "1234", "5th", "Ave", ",", "Denver", "CO"]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "B-STREET_NAME", "O", "O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "John"
        assert result["last_name"] == "Doe"
        assert result["street_name"] == "5th"

    def test_po_box(self):
        words = ["John", "Doe", ",", "P.O.", "Box", "1234", ",", "Denver", "CO"]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "B-STREET_NAME", "O", "O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "John"
        assert result["last_name"] == "Doe"
        assert result["street_name"] == "Box"

    def test_empty_input(self):
        result = postprocess([0, 0], [], [None, None])
        assert result == {"first_name": "", "last_name": "", "street_name": ""}

    def test_multi_word_first_name(self):
        words = ["Mary", "Jane", "Doe", ","]
        labels = ["B-FIRST_NAME", "I-FIRST_NAME", "B-LAST_NAME", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "Mary Jane"
        assert result["last_name"] == "Doe"

    def test_no_entities_returns_empty_strings(self):
        words = ["Denver", "CO", "80201"]
        labels = ["O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result == {"first_name": "", "last_name": "", "street_name": ""}

"""Tests for the post-processing pipeline.

Words may include attached punctuation (e.g. "Doe,", "Ave,") since the model
is trained on raw whitespace-split text. Post-processing strips punctuation
from entity words before returning results.

Tests pass words and word_ids directly — no tokenizer needed.
word_ids mirrors what encoding.word_ids() returns: None for special tokens,
an int index for each subtoken pointing to its source word.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from name_parsing.config import LABEL2ID
from name_parsing.postprocessor import (
    _clean_word,
    extract_entities,
    filter_street_name,
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


class TestCleanWord:
    def test_strips_trailing_comma(self):
        assert _clean_word("Doe,") == "Doe"

    def test_strips_trailing_period(self):
        assert _clean_word("Ave.") == "Ave"

    def test_no_punctuation_unchanged(self):
        assert _clean_word("Braddock") == "Braddock"

    def test_strips_leading_punctuation(self):
        assert _clean_word(",John") == "John"

    def test_ordinal_unchanged(self):
        assert _clean_word("5th") == "5th"


class TestExtractEntities:
    def test_single_name_raw_words(self):
        """Raw words with punctuation attached to last name."""
        words = ["Alex", "Doe,", "1201", "Braddock", "Ave,"]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "B-STREET_NAME", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["Alex"]]
        assert entities["LAST_NAME"] == [["Doe,"]]  # raw word, punctuation present
        assert entities["STREET_NAME"] == [["Braddock"]]

    def test_shared_last_name(self):
        words = ["Alex", "or", "Mary", "Doe,", "1201", "Braddock", "Ave,"]
        labels = ["B-FIRST_NAME", "O", "O", "B-LAST_NAME", "O", "B-STREET_NAME", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["Alex"]]
        assert entities["LAST_NAME"] == [["Doe,"]]
        assert entities["STREET_NAME"] == [["Braddock"]]

    def test_separate_names(self):
        words = ["Alex", "Doe", "or", "Mary", "Smith,", "500", "Oak", "Ave,"]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "O", "O", "B-STREET_NAME", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["Alex"]]
        assert entities["LAST_NAME"] == [["Doe"]]
        assert entities["STREET_NAME"] == [["Oak"]]

    def test_business_payor(self):
        """Business: first_name=business word, last_name=LLC/Inc/etc."""
        words = ["Fairfax", "SushiMax", "LLC,", "1201", "Braddock", "Ave,"]
        labels = ["O", "B-FIRST_NAME", "B-LAST_NAME", "O", "B-STREET_NAME", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["SushiMax"]]
        assert entities["LAST_NAME"] == [["LLC,"]]
        assert entities["STREET_NAME"] == [["Braddock"]]

    def test_multi_subtoken_word_uses_first_subtoken(self):
        """Word 'Braddock' splits into 2 subtokens; first subtoken label wins."""
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
        words = ["Alex", "Doe,"]
        labels = ["B-FIRST_NAME", "I-LAST_NAME"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        entities = extract_entities(preds, words, word_ids)

        assert entities["FIRST_NAME"] == [["Alex"]]
        assert entities["LAST_NAME"] == []


class TestFilterStreetName:
    def test_single_distinct_word(self):
        assert filter_street_name([["Braddock"]]) == "Braddock"

    def test_strips_punctuation_from_words(self):
        assert filter_street_name([["Braddock,"]]) == "Braddock"

    def test_filters_generic_suffix(self):
        # "Ave," is labeled — strip comma, check against GENERIC_STREET_WORDS
        assert filter_street_name([["Braddock"], ["Ave,"]]) == "Braddock"

    def test_ordinal_street_kept(self):
        """Ordinals like '5th' must not be filtered as numeric."""
        assert filter_street_name([["5th"]]) == "5th"
        assert filter_street_name([["1st"]]) == "1st"
        assert filter_street_name([["22nd"]]) == "22nd"

    def test_ordinal_with_generic_suffix(self):
        assert filter_street_name([["5th"], ["Ave,"]]) == "5th"

    def test_po_box_returns_box(self):
        """'Box' is the street_name for P.O. Box addresses."""
        assert filter_street_name([["Box"]]) == "Box"

    def test_filters_pure_numeric(self):
        assert filter_street_name([["1201,"], ["Braddock"]]) == "Braddock"

    def test_multi_word_span_returns_first_distinct(self):
        # "Silver Lake" — neither is generic, returns first
        assert filter_street_name([["Silver", "Lake"]]) == "Silver"

    def test_all_generic_returns_first_word_cleaned(self):
        assert filter_street_name([["North,"], ["Ave,"]]) == "North"

    def test_empty(self):
        assert filter_street_name([]) == ""


class TestPostprocess:
    def test_single_name_punctuation_stripped(self):
        """Punctuation on raw words is stripped in output."""
        words = ["Alex", "Doe,", "1201", "Braddock", "Ave,", "Richmond", "VA"]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "B-STREET_NAME", "O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "Alex"
        assert result["last_name"] == "Doe"
        assert result["street_name"] == "Braddock"

    def test_shared_last_name(self):
        words = ["Alex", "or", "Mary", "Doe,", "1201", "Braddock", "Ave,",
                 "Richmond", "VA", "22312"]
        labels = ["B-FIRST_NAME", "O", "O", "B-LAST_NAME", "O",
                  "B-STREET_NAME", "O", "O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "Alex"
        assert result["last_name"] == "Doe"
        assert result["street_name"] == "Braddock"

    def test_business_payor_output(self):
        words = ["Fairfax", "SushiMax", "LLC,", "1201", "Braddock", "Ave,", "Denver", "CO"]
        labels = ["O", "B-FIRST_NAME", "B-LAST_NAME", "O", "B-STREET_NAME", "O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "SushiMax"
        assert result["last_name"] == "LLC"
        assert result["street_name"] == "Braddock"

    def test_ordinal_street(self):
        words = ["John", "Doe,", "1234", "5th", "Ave,", "Denver", "CO"]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "B-STREET_NAME", "O", "O", "O"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "John"
        assert result["last_name"] == "Doe"
        assert result["street_name"] == "5th"

    def test_po_box(self):
        words = ["John", "Doe,", "P.O.", "Box", "1234,", "Denver", "CO"]
        labels = ["B-FIRST_NAME", "B-LAST_NAME", "O", "B-STREET_NAME", "O", "O", "O"]
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
        words = ["Mary", "Jane", "Doe,"]
        labels = ["B-FIRST_NAME", "I-FIRST_NAME", "B-LAST_NAME"]
        preds = _make_predictions(words, labels)
        word_ids = _make_word_ids(words)

        result = postprocess(preds, words, word_ids)

        assert result["first_name"] == "Mary Jane"
        assert result["last_name"] == "Doe"

"""Tests for the post-processing pipeline (V2: subword-based).

Tests use mock subword tokens and offset_mapping to simulate what the
tokenizer + model would produce, without needing a real model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from name_parsing.config import LABEL2ID
from name_parsing.postprocessor import (
    _join_token_infos,
    extract_entities,
    filter_street_name,
    postprocess,
)


def _make_mock_data(text_tokens: list[str], labels: list[str]):
    """Build mock tokens, predictions, and offset_mapping for testing.

    Simulates realistic offset_mapping: non-## tokens are separated by 1-char
    gaps (representing spaces), while ## tokens are adjacent to their predecessor
    (no gap).

    Args:
        text_tokens: List of subword tokens (use ## prefix for continuations).
        labels: List of BIO label strings, same length as text_tokens.

    Returns:
        (predictions, tokens, offset_mapping) with [CLS] and [SEP] added.
    """
    tokens = ["[CLS]"] + text_tokens + ["[SEP]"]
    preds = [-100] + [LABEL2ID[l] for l in labels] + [-100]

    # Build offset_mapping with realistic gaps:
    # - ## tokens are adjacent (no gap) to previous token
    # - non-## tokens have a 1-char gap (space) from previous token
    offset_mapping = [(0, 0)]  # [CLS]
    pos = 0
    for i, tok in enumerate(text_tokens):
        if i > 0 and not tok.startswith("##"):
            pos += 1  # space gap before non-## tokens
        clean = tok.lstrip("#") if tok.startswith("##") else tok
        end = pos + len(clean)
        offset_mapping.append((pos, end))
        pos = end
    offset_mapping.append((0, 0))  # [SEP]

    return preds, tokens, offset_mapping


def _extract_texts(entity_spans):
    """Helper: extract just the text strings from entity spans for easier assertion."""
    return [[text for text, _, _ in span] for span in entity_spans]


class TestExtractEntities:
    def test_single_name(self):
        preds, tokens, offsets = _make_mock_data(
            ["alex", "doe", "1201", "brad", "##dock", "ave"],
            ["B-FIRST_NAME", "B-LAST_NAME", "O", "B-STREET_NAME", "I-STREET_NAME", "O"],
        )
        entities = extract_entities(preds, tokens, offsets)

        assert _extract_texts(entities["FIRST_NAME"]) == [["alex"]]
        assert _extract_texts(entities["LAST_NAME"]) == [["doe"]]
        assert _extract_texts(entities["STREET_NAME"]) == [["brad", "dock"]]

    def test_shared_last_name(self):
        preds, tokens, offsets = _make_mock_data(
            ["alex", "or", "mary", "doe", "1201", "brad", "##dock", "ave"],
            ["B-FIRST_NAME", "O", "O", "B-LAST_NAME", "O", "B-STREET_NAME", "I-STREET_NAME", "O"],
        )
        entities = extract_entities(preds, tokens, offsets)

        assert _extract_texts(entities["FIRST_NAME"]) == [["alex"]]
        assert _extract_texts(entities["LAST_NAME"]) == [["doe"]]
        assert _extract_texts(entities["STREET_NAME"]) == [["brad", "dock"]]

    def test_separate_names(self):
        preds, tokens, offsets = _make_mock_data(
            ["alex", "doe", "or", "mary", "smith", "1201", "oak", "st"],
            ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "O", "O", "B-STREET_NAME", "O"],
        )
        entities = extract_entities(preds, tokens, offsets)

        assert _extract_texts(entities["FIRST_NAME"]) == [["alex"]]
        assert _extract_texts(entities["LAST_NAME"]) == [["doe"]]
        assert _extract_texts(entities["STREET_NAME"]) == [["oak"]]

    def test_multi_subword_entity(self):
        """Test entity spanning multiple subwords (like 'braddock' -> 'brad' + '##dock')."""
        preds, tokens, offsets = _make_mock_data(
            ["brad", "##dock"],
            ["B-STREET_NAME", "I-STREET_NAME"],
        )
        entities = extract_entities(preds, tokens, offsets)

        assert _extract_texts(entities["STREET_NAME"]) == [["brad", "dock"]]

    def test_multi_token_first_name(self):
        """Test two-word first name like 'Mary Jane'."""
        preds, tokens, offsets = _make_mock_data(
            ["mary", "jane", "doe"],
            ["B-FIRST_NAME", "I-FIRST_NAME", "B-LAST_NAME"],
        )
        entities = extract_entities(preds, tokens, offsets)

        assert _extract_texts(entities["FIRST_NAME"]) == [["mary", "jane"]]

    def test_special_tokens_only(self):
        tokens = ["[CLS]", "[SEP]"]
        preds = [-100, -100]
        offsets = [(0, 0), (0, 0)]

        entities = extract_entities(preds, tokens, offsets)

        assert entities["FIRST_NAME"] == []
        assert entities["LAST_NAME"] == []
        assert entities["STREET_NAME"] == []


class TestJoinTokenInfos:
    def test_single_word_subwords(self):
        """Subwords of one word (no gap) are concatenated directly."""
        infos = [("brad", 0, 4), ("dock", 4, 8)]
        assert _join_token_infos(infos) == "braddock"

    def test_multi_word_with_gaps(self):
        """Tokens with gaps (spaces) get spaces inserted."""
        infos = [("silver", 0, 6), ("lake", 7, 11)]
        assert _join_token_infos(infos) == "silver lake"

    def test_mixed_subwords_and_words(self):
        """Mix of ## continuations and separate words."""
        infos = [("mary", 0, 4), ("jane", 5, 9)]
        assert _join_token_infos(infos) == "mary jane"

    def test_empty(self):
        assert _join_token_infos([]) == ""

    def test_single_token(self):
        assert _join_token_infos([("alex", 0, 4)]) == "alex"


class TestFilterStreetName:
    def test_single_distinct(self):
        assert filter_street_name([[("braddock", 0, 8)]]) == "braddock"

    def test_joined_subwords(self):
        # Subwords of one word (no gap) -> "braddock"
        assert filter_street_name([[("brad", 0, 4), ("dock", 4, 8)]]) == "braddock"

    def test_multi_word_street_filters_generic(self):
        """'silver lake' span — neither is generic so it returns 'silver'."""
        # "silver lake" as a single span with a gap
        assert filter_street_name([[("silver", 0, 6), ("lake", 7, 11)]]) == "silver"

    def test_filters_generic(self):
        # Two separate spans: street name and suffix
        assert filter_street_name([[("braddock", 0, 8)], [("ave", 10, 13)]]) == "braddock"

    def test_all_generic_returns_first(self):
        assert filter_street_name([[("north", 0, 5)], [("ave", 6, 9)]]) == "north"

    def test_empty(self):
        assert filter_street_name([]) == ""

    def test_multiple_spans(self):
        assert filter_street_name([[("oak", 0, 3)], [("maple", 5, 10)]]) == "oak"

    def test_filters_numeric(self):
        assert filter_street_name([[("1201", 0, 4)], [("braddock", 5, 13)]]) == "braddock"

    def test_ocr_mangled_number(self):
        assert filter_street_name([[("553s7", 0, 5)], [("braddock", 6, 14)]]) == "braddock"


class TestPostprocess:
    def test_full_pipeline(self):
        preds, tokens, offsets = _make_mock_data(
            ["alex", "or", "mary", "doe", ",", "1201", "brad", "##dock", "ave",
             ",", "rich", "##mond", "va", "22312"],
            ["B-FIRST_NAME", "O", "O", "B-LAST_NAME", "O", "O", "B-STREET_NAME",
             "I-STREET_NAME", "O", "O", "O", "O", "O", "O"],
        )
        result = postprocess(preds, tokens, offsets)

        assert result["first_name"] == "alex"
        assert result["last_name"] == "doe"
        assert result["street_name"] == "braddock"

    def test_strips_trailing_punctuation(self):
        preds, tokens, offsets = _make_mock_data(
            ["alex", ",", "doe", ","],
            ["B-FIRST_NAME", "O", "B-LAST_NAME", "O"],
        )
        result = postprocess(preds, tokens, offsets)

        assert result["first_name"] == "alex"
        assert result["last_name"] == "doe"

    def test_empty_input(self):
        tokens = ["[CLS]", "[SEP]"]
        preds = [-100, -100]
        offsets = [(0, 0), (0, 0)]
        result = postprocess(preds, tokens, offsets)
        assert result == {"first_name": "", "last_name": "", "street_name": ""}

    def test_merged_ocr_token(self):
        """Test handling of OCR-merged token like '37/harbor'."""
        preds, tokens, offsets = _make_mock_data(
            ["alex", "doe", "37", "/", "harbor", "way"],
            ["B-FIRST_NAME", "B-LAST_NAME", "O", "O", "B-STREET_NAME", "O"],
        )
        result = postprocess(preds, tokens, offsets)

        assert result["first_name"] == "alex"
        assert result["last_name"] == "doe"
        assert result["street_name"] == "harbor"

    def test_multi_word_street_name(self):
        """'silver lake' as B-STREET + I-STREET should produce 'silver'
        as the first distinct word, since neither word is generic."""
        preds, tokens, offsets = _make_mock_data(
            ["james", "par", "##ker", "90", "silver", "lake", "dr"],
            ["B-FIRST_NAME", "B-LAST_NAME", "I-LAST_NAME", "O",
             "B-STREET_NAME", "I-STREET_NAME", "O"],
        )
        result = postprocess(preds, tokens, offsets)

        assert result["first_name"] == "james"
        assert result["last_name"] == "parker"
        # "silver lake" span -> split into ["silver", "lake"] -> both distinct -> first = "silver"
        assert result["street_name"] == "silver"

    def test_multi_word_first_name_with_space(self):
        """Two-word first name like 'Mary Jane' should have a space."""
        preds, tokens, offsets = _make_mock_data(
            ["mary", "jane", "doe"],
            ["B-FIRST_NAME", "I-FIRST_NAME", "B-LAST_NAME"],
        )
        result = postprocess(preds, tokens, offsets)

        assert result["first_name"] == "mary jane"
        assert result["last_name"] == "doe"

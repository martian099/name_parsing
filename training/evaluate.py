"""Evaluate model on held-out test data and print per-field accuracy.

V3: Uses preprocessed word-level data format. Ground truth is reconstructed
directly from the word list + word-level labels — no subword joining needed.

Expects a separate test file (generated with a different seed than training)
to avoid data leakage.

Usage:
    python training/evaluate.py --model models/onnx/quantized --data data/raw/test.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from name_parsing.config import GENERIC_STREET_WORDS, ID2LABEL
from name_parsing.model import NameAddressParser


def extract_expected_from_example(ex: dict) -> dict[str, str]:
    """Reconstruct ground-truth entity values from a training example.

    Uses the word list (from 'preprocessed') and word-level labels to find
    the first span of each entity type, then joins the words.
    """
    words = ex["preprocessed"].split()
    labels = ex["labels"]  # one label ID per word

    # Build per-word label map directly (labels are already word-level)
    word_label: dict[int, str] = {}
    for word_idx, lab_id in enumerate(labels):
        if lab_id not in (0,):
            word_label[word_idx] = ID2LABEL.get(lab_id, "O")

    expected = {"first_name": "", "last_name": "", "street_name": ""}

    for field_prefix, field_key in [
        ("FIRST_NAME", "first_name"),
        ("LAST_NAME", "last_name"),
        ("STREET_NAME", "street_name"),
    ]:
        entity_words: list[str] = []
        in_entity = False

        for w_idx, word in enumerate(words):
            lbl = word_label.get(w_idx, "O")
            if lbl == f"B-{field_prefix}":
                in_entity = True
                entity_words = [word]
            elif lbl == f"I-{field_prefix}" and in_entity:
                entity_words.append(word)
            elif in_entity:
                break  # end of first span

        if entity_words:
            value = " ".join(entity_words)
            if field_key == "street_name" and value.lower() in GENERIC_STREET_WORDS:
                value = ""
            expected[field_key] = value

    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/onnx/quantized")
    parser.add_argument("--data", type=str, default="data/raw/test.json",
                        help="Held-out test file (different seed from training data)")
    parser.add_argument("--max-examples", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = NameAddressParser(args.model)

    print(f"Loading test data from {args.data}...")
    with open(args.data) as f:
        examples = json.load(f)

    test_examples = examples[:args.max_examples]
    print(f"Evaluating on {len(test_examples)} examples...\n")

    correct = {"first_name": 0, "last_name": 0, "street_name": 0}
    total = {"first_name": 0, "last_name": 0, "street_name": 0}
    errors = []

    for ex in test_examples:
        expected = extract_expected_from_example(ex)
        predicted = model.parse(ex["text"])

        for field in ["first_name", "last_name", "street_name"]:
            if not expected[field]:
                continue
            total[field] += 1
            if predicted[field].lower() == expected[field].lower():
                correct[field] += 1
            else:
                errors.append({
                    "text": ex["text"],
                    "field": field,
                    "expected": expected[field],
                    "predicted": predicted[field],
                })

    print("=" * 60)
    print(f"{'Field':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 60)
    for field in ["first_name", "last_name", "street_name"]:
        acc = correct[field] / total[field] * 100 if total[field] > 0 else 0
        print(f"{field:<15} {correct[field]:<10} {total[field]:<10} {acc:.1f}%")
    print("=" * 60)

    overall_correct = sum(correct.values())
    overall_total = sum(total.values())
    overall_acc = overall_correct / overall_total * 100 if overall_total > 0 else 0
    print(f"{'Overall':<15} {overall_correct:<10} {overall_total:<10} {overall_acc:.1f}%")

    if errors:
        print(f"\n--- Sample errors (first 10 of {len(errors)}) ---")
        for err in errors[:10]:
            print(f"  Text:      {err['text'][:80]}")
            print(f"  Field:     {err['field']}")
            print(f"  Expected:  '{err['expected']}' | Predicted: '{err['predicted']}'")
            print()


if __name__ == "__main__":
    main()

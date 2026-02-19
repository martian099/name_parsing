"""Evaluate model on test data and print per-entity metrics.

V2: Works with the new data format (text + character spans) instead of
word-level tokens/tags.

Usage:
    python training/evaluate.py --model models/onnx/quantized --data data/raw/train.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from transformers import AutoTokenizer

from name_parsing.config import GENERIC_STREET_WORDS, ID2LABEL
from name_parsing.model import NameAddressParser


def extract_expected_from_example(ex: dict, tokenizer) -> dict[str, str]:
    """Extract ground-truth values from a V2 training example.

    In V2, training data has input_ids and labels (subword-level BIO IDs).
    We reconstruct expected entity values by joining subword tokens that have
    B-/I- labels.
    """
    tokens = tokenizer.convert_ids_to_tokens(ex["input_ids"])
    labels = ex["labels"]

    expected = {"first_name": "", "last_name": "", "street_name": ""}

    for field_prefix in ["FIRST_NAME", "LAST_NAME", "STREET_NAME"]:
        field_key = field_prefix.lower()
        field_tokens = []
        in_entity = False

        for tok, lab_id in zip(tokens, labels):
            if lab_id == -100 or lab_id == 0:
                if in_entity:
                    break  # End of first occurrence
                continue

            label = ID2LABEL.get(lab_id, "O")
            if label == f"B-{field_prefix}":
                in_entity = True
                clean = tok.lstrip("#") if tok.startswith("##") else tok
                field_tokens.append(clean)
            elif label == f"I-{field_prefix}" and in_entity:
                clean = tok.lstrip("#") if tok.startswith("##") else tok
                field_tokens.append(clean)
            elif in_entity:
                break  # End of entity

        if field_tokens:
            value = "".join(field_tokens)
            # For street_name, filter out generic words
            if field_key == "street_name":
                if value.lower().rstrip(".,") in GENERIC_STREET_WORDS:
                    value = ""
            expected[field_key] = value.rstrip(".,;:")

    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/onnx/quantized")
    parser.add_argument("--data", type=str, default="data/raw/train.json")
    parser.add_argument("--max-examples", type=int, default=500)
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    model = NameAddressParser(args.model)

    print(f"Loading tokenizer...")
    from name_parsing.config import BASE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print(f"Loading data from {args.data}...")
    with open(args.data) as f:
        examples = json.load(f)

    # Use last N examples as test set (training used the first ones)
    test_examples = examples[-args.max_examples:]
    print(f"Evaluating on {len(test_examples)} examples...\n")

    correct = {"first_name": 0, "last_name": 0, "street_name": 0}
    total = {"first_name": 0, "last_name": 0, "street_name": 0}
    errors = []

    for ex in test_examples:
        expected = extract_expected_from_example(ex, tokenizer)
        text = ex["text"]
        predicted = model.parse(text)

        # Compare (case-insensitive)
        for field in ["first_name", "last_name", "street_name"]:
            if not expected[field]:
                continue
            total[field] += 1
            if predicted[field].lower() == expected[field].lower():
                correct[field] += 1
            else:
                errors.append({
                    "text": text,
                    "field": field,
                    "expected": expected[field],
                    "predicted": predicted[field],
                })

    # Print results
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

    # Print some errors
    if errors:
        print(f"\n--- Sample errors (showing first 10 of {len(errors)}) ---")
        for err in errors[:10]:
            print(f"  Text: {err['text'][:80]}...")
            print(f"  Field: {err['field']}")
            print(f"  Expected: '{err['expected']}' | Predicted: '{err['predicted']}'")
            print()


if __name__ == "__main__":
    main()

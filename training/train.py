"""Fine-tune ModernBERT for token classification on name/address NER.

Training data contains raw text and word-level labels aligned with text.split().
This script tokenizes on-the-fly using is_split_into_words=True and expands
word-level labels to subtoken labels (first subtoken gets the label, rest get -100).

Usage:
    python training/train.py --data data/raw/train.json --output models/finetuned
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from name_parsing.config import (
    BASE_MODEL_NAME,
    BATCH_SIZE,
    ID2LABEL,
    LABEL2ID,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    NUM_LABELS,
    TRAIN_EPOCHS,
    TRAIN_TEST_SPLIT,
)


def tokenize_and_align_labels(examples: dict, tokenizer) -> dict:
    """Tokenize SpaCy-pre-tokenized word lists and expand labels to subtoken labels.

    Each example has 'words' (SpaCy-tokenized list) and 'labels'
    (one integer label ID per word). We tokenize with is_split_into_words=True
    and assign: first subtoken of each word → word label, rest → -100.
    """
    word_lists = examples["words"]
    tokenized = tokenizer(
        word_lists,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

    all_labels = []
    for i, word_labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        subtoken_labels = []
        seen: set[int] = set()
        for word_idx in word_ids:
            if word_idx is None:
                subtoken_labels.append(-100)
            elif word_idx in seen:
                subtoken_labels.append(-100)
            else:
                seen.add(word_idx)
                subtoken_labels.append(word_labels[word_idx])
        all_labels.append(subtoken_labels)

    tokenized["labels"] = all_labels
    return tokenized


def load_data(data_path: str, tokenizer) -> DatasetDict:
    """Load JSON training data (SpaCy-tokenized words), tokenize on-the-fly, and split into train/eval."""
    with open(data_path) as f:
        examples = json.load(f)

    records = [{"words": ex["words"], "labels": ex["labels"]} for ex in examples]
    dataset = Dataset.from_list(records)
    dataset = dataset.map(
        lambda batch: tokenize_and_align_labels(batch, tokenizer),
        batched=True,
        remove_columns=["words"],
    )

    split = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT, seed=42)
    return DatasetDict({"train": split["train"], "eval": split["test"]})


def compute_metrics(eval_pred):
    """Compute token-level precision, recall, F1 for NER."""
    from seqeval.metrics import f1_score, precision_score, recall_score

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Convert IDs back to label strings, ignoring -100
    true_labels = []
    pred_labels = []
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_clean = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_seq.append(ID2LABEL[l])
            pred_seq_clean.append(ID2LABEL[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_clean)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/train.json")
    parser.add_argument("--output", type=str, default="models/finetuned")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training device: {device}")

    print(f"Loading tokenizer and model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print(f"Loading data from {args.data}...")
    datasets = load_data(args.data, tokenizer)
    print(f"Train: {len(datasets['train'])} examples, Eval: {len(datasets['eval'])} examples")

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        save_total_limit=2,
        fp16=False,  # MPS and CPU don't support fp16 training
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    print(f"\nEval metrics: {metrics}")

    # Save best model
    output_path = Path(args.output)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"\nModel saved to {output_path}")

    # Save label map alongside model
    with open(output_path / "label_map.json", "w") as f:
        json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID}, f, indent=2)


if __name__ == "__main__":
    main()

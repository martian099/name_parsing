"""Fine-tune DistilBERT for token classification on OCR customer record NER.

This version uses character-level tokenization (no is_split_into_words).
Training data already contains pre-tokenized input_ids and per-subword labels.

Usage:
    python training/train.py --data data/raw/train.json --output models/finetuned
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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


def load_data(data_path: str) -> DatasetDict:
    """Load pre-tokenized JSON training data and split into train/eval."""
    with open(data_path) as f:
        examples = json.load(f)

    # Data is already tokenized with input_ids, attention_mask, labels
    records = []
    for ex in examples:
        records.append({
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["labels"],
        })

    dataset = Dataset.from_list(records)
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

    print(f"Loading data from {args.data}...")
    datasets = load_data(args.data)
    print(f"Train: {len(datasets['train'])} examples, Eval: {len(datasets['eval'])} examples")

    print(f"Loading tokenizer and model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

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
        fp16=False,  # CPU training
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

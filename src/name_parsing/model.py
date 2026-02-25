"""Inference pipeline for OCR customer record NER extraction.

Input text is tokenized with SpaCy (blank English model) before being
passed to DistilBERT. This matches the training-time tokenization used
in generate_training_data.py for consistent label alignment.

SpaCy separates punctuation from words (e.g. "Doe," → ["Doe", ","]), so
entity words in the output are already clean — no punctuation stripping needed.

Loads a quantized ONNX DistilBERT model and provides a simple parse() API.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import spacy
from transformers import AutoTokenizer

from name_parsing.config import MAX_SEQ_LENGTH, ONNX_MODEL_DIR
from name_parsing.postprocessor import postprocess

# SpaCy blank English tokenizer (rule-based, no ML model download needed)
_nlp = spacy.blank("en")


class NameAddressParser:
    """Extract first_name, last_name, and street_name from text.

    Args:
        model_dir: Path to directory containing the quantized ONNX model
                   and tokenizer files. Defaults to models/onnx/quantized.
    """

    def __init__(self, model_dir: str | Path | None = None):
        if model_dir is None:
            model_dir = ONNX_MODEL_DIR / "quantized"
        model_dir = Path(model_dir)

        # Find the ONNX model file
        onnx_files = list(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx file found in {model_dir}")
        model_path = onnx_files[0]

        # Load ONNX runtime session (CPU)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    def parse(self, text: str) -> dict[str, str]:
        """Parse text and extract structured data.

        Text is first tokenized with SpaCy (matching training-time preprocessing),
        then fed to DistilBERT as pre-split words via is_split_into_words=True.
        SpaCy separates punctuation (e.g. "Doe," → ["Doe", ","]), so entity words
        in the output are already clean.

        Args:
            text: Input text, e.g. "Alex or Mary Doe, 1201 Braddock Ave, Richmond VA"

        Returns:
            Dict with keys: first_name, last_name, street_name
        """
        if not text or not text.strip():
            return {"first_name": "", "last_name": "", "street_name": ""}

        # SpaCy tokenization (matches training-time preprocessing)
        words = [token.text for token in _nlp(text)]

        if not words:
            return {"first_name": "", "last_name": "", "street_name": ""}

        # Tokenize as pre-split words — clean word_ids() alignment
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
            return_tensors="np",
        )

        # Run ONNX inference
        input_feed = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
        outputs = self.session.run(None, input_feed)
        logits = outputs[0]  # shape: (1, seq_len, num_labels)

        # Get predicted label IDs and word alignment
        predictions = np.argmax(logits[0], axis=-1).tolist()
        word_ids = encoding.word_ids()  # list[int | None], one per subtoken

        return postprocess(predictions, words, word_ids)

    def parse_batch(self, texts: list[str]) -> list[dict[str, str]]:
        """Parse multiple texts sequentially.

        Args:
            texts: List of text strings.

        Returns:
            List of result dicts.
        """
        return [self.parse(text) for text in texts]

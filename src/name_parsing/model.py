"""Inference pipeline for OCR customer record NER extraction.

V3: Preprocesses raw text before tokenization to split OCR-merged tokens
(e.g. "JohnDoe" -> "John Doe", "37/harbor" -> "37 harbor"), then uses
standard word-level NER with is_split_into_words=True and word_ids() for
clean, unambiguous subword-to-word alignment.

Loads a quantized ONNX DistilBERT model and provides a simple parse() API.
"""

import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from name_parsing.config import MAX_SEQ_LENGTH, ONNX_MODEL_DIR
from name_parsing.postprocessor import postprocess


def preprocess_ocr_text(text: str) -> str:
    """Split OCR-merged tokens into separate words before NER.

    Handles:
    - Special-char merges: "37/harbor" -> "37 harbor", "North|Gate" -> "North Gate"
    - CamelCase merges:    "JohnDoe"   -> "John Doe",  "MaryDoe"   -> "Mary Doe"
    - Digit<->letter:      "37harbor"  -> "37 harbor", "12Braddock" -> "12 Braddock"
    - Punctuation:         "Doe,"      -> "Doe"  (stripped, not split into separate token)

    All-lowercase merges ("johndoe") are unrecoverable without a dictionary
    and are left as-is. OCR on printed text almost always preserves capitals.
    """
    # CamelCase: insert space before uppercase preceded by lowercase (do this before stripping)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Digit<->letter boundary splits
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    # Remove all non-alphanumeric, non-space characters (punctuation, special chars)
    text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)
    # Collapse extra spaces
    text = re.sub(r' +', ' ', text).strip()
    return text


class NameAddressParser:
    """Extract first_name, last_name, and street_name from OCR-scanned text.

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
        """Parse OCR-scanned text and extract structured data.

        Args:
            text: Raw OCR text from a scanned document, e.g.
                  "Alex or Mary Doe, 1201 Braddock Ave, Richmond VA, 22312"

        Returns:
            Dict with keys: first_name, last_name, street_name
        """
        if not text or not text.strip():
            return {"first_name": "", "last_name": "", "street_name": ""}

        # Preprocess: split OCR-merged tokens before passing to the model
        preprocessed = preprocess_ocr_text(text)
        words = preprocessed.split()

        if not words:
            return {"first_name": "", "last_name": "", "street_name": ""}

        # Tokenize as pre-split words — clean word_ids() alignment, no offset_mapping needed
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
            texts: List of OCR text strings.

        Returns:
            List of result dicts.
        """
        return [self.parse(text) for text in texts]

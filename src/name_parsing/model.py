"""Inference pipeline for OCR customer record NER extraction.

V2: Tokenizes raw text directly (no is_split_into_words) and uses
offset_mapping for subword-to-character alignment. This handles
OCR-merged tokens like "37/harbor" correctly.

Loads a quantized ONNX DistilBERT model and provides a simple parse() API.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from name_parsing.config import MAX_SEQ_LENGTH, ONNX_MODEL_DIR
from name_parsing.postprocessor import postprocess


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

        # Tokenize raw text — no is_split_into_words, let WordPiece handle everything
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
            return_tensors="np",
            return_offsets_mapping=True,
        )

        # Run ONNX inference (offset_mapping not needed by model)
        input_feed = {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
        outputs = self.session.run(None, input_feed)
        logits = outputs[0]  # shape: (1, seq_len, num_labels)

        # Get predicted label IDs
        predictions = np.argmax(logits[0], axis=-1).tolist()

        # Get subword tokens and offset mapping for post-processing
        tokens = self.tokenizer.convert_ids_to_tokens(
            encoding["input_ids"][0].tolist()
        )
        offset_mapping = encoding["offset_mapping"][0].tolist()

        # Convert offset_mapping to list of tuples
        offset_mapping = [(int(s), int(e)) for s, e in offset_mapping]

        # Post-process to extract entities
        return postprocess(predictions, tokens, offset_mapping)

    def parse_batch(self, texts: list[str]) -> list[dict[str, str]]:
        """Parse multiple texts. Currently processes sequentially.

        Args:
            texts: List of OCR text strings.

        Returns:
            List of result dicts.
        """
        return [self.parse(text) for text in texts]

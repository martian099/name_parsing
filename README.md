# Name & Address Parser for OCR-Scanned Documents

Extract structured data from messy OCR-scanned customer records using a fine-tuned DistilBERT NER model.

```python
from name_parsing import NameAddressParser

parser = NameAddressParser("models/onnx/quantized")
result = parser.parse("Alex or Mary Doe, 1201 Braddock Ave, Richmond VA, 22312")
# {'first_name': 'alex', 'last_name': 'doe', 'street_name': 'braddock'}
```

**Key features:**
- Handles OCR noise (character swaps, drops, doubles)
- Handles OCR-merged tokens (e.g., `"JohnDoe"` → `"John"` / `"Doe"`, `"37/harbor"` → `"harbor"`)
- Extracts only the first person's name when multiple names appear
- Returns the most distinctive street name word (filters out "Ave", "St", etc.)
- CPU-only, ~10ms inference latency (p99)
- 67 MB quantized ONNX model — no GPU or PyTorch needed at runtime

## Quick Start

### 1. Set Up the Environment

```bash
# Clone the repo
git clone https://github.com/<your-username>/name-parsing.git
cd name-parsing

# Create the conda environment (all dependencies pinned)
conda env create -f environment.yaml

# Activate it
conda activate name-parsing

# Install the package in development mode
pip install -e .
```

### 2. Inference (Using the Pre-Trained Model)

If the repo includes a pre-trained model in `models/onnx/quantized/`, you can start parsing immediately:

```python
from name_parsing import NameAddressParser

parser = NameAddressParser("models/onnx/quantized")

# Single name
parser.parse("John Smith, 500 Oak Ave, Denver CO 80201")
# {'first_name': 'john', 'last_name': 'smith', 'street_name': 'oak'}

# Multiple names — extracts only the first person
parser.parse("Alex or Mary Doe, 1201 Braddock Ave, Richmond VA 22312")
# {'first_name': 'alex', 'last_name': 'doe', 'street_name': 'braddock'}

# OCR-merged CamelCase names — splits before NER
parser.parse("JohnDoe, 1201 Braddock Ave, Richmond VA 22312")
# {'first_name': 'john', 'last_name': 'doe', 'street_name': 'braddock'}

# OCR-merged token with special character
parser.parse("sarah martinez 37/harbor way coastal city, ca 90210")
# {'first_name': 'sarah', 'last_name': 'martinez', 'street_name': 'harbor'}

# Batch processing
parser.parse_batch(["John Smith, 100 Main St", "Jane Doe, 200 Oak Ave"])
```

### 3. Training From Scratch

If you want to retrain the model (e.g., with different data or parameters):

```bash
# Step 1: Generate synthetic training data (4000 examples)
python training/generate_training_data.py \
    --num-examples 4000 \
    --output data/raw/train.json \
    --seed 42

# Step 2: Generate a held-out test set (different seed to avoid leakage)
python training/generate_training_data.py \
    --num-examples 1000 \
    --output data/raw/test.json \
    --seed 99

# Step 3: Fine-tune DistilBERT (takes ~6 min on CPU)
python training/train.py \
    --data data/raw/train.json \
    --output models/finetuned \
    --epochs 15 \
    --batch-size 16 \
    --lr 5e-5

# Step 4: Export to ONNX + quantize (265 MB → 67 MB)
python training/export_onnx.py \
    --model models/finetuned \
    --output models/onnx

# Step 5: Evaluate accuracy on the held-out test set
python training/evaluate.py \
    --model models/onnx/quantized \
    --data data/raw/test.json \
    --max-examples 500
```

### 4. Adding Your Own Labeled Examples

The training data format is intentionally simple — just raw text, preprocessed text, and word-level labels:

```json
{
  "text": "JohnDoe, 500 Oak Ave, Denver CO",
  "preprocessed": "John Doe 500 Oak Ave Denver CO",
  "labels": [1, 3, 0, 5, 0, 0, 0, 0]
}
```

Labels are integer IDs mapping to:
```
0=O, 1=B-FIRST_NAME, 2=I-FIRST_NAME, 3=B-LAST_NAME, 4=I-LAST_NAME,
5=B-STREET_NAME, 6=I-STREET_NAME
```

The `preprocessed` field is the output of `preprocess_ocr_text()` applied to `text` — it splits
CamelCase merges, digit↔letter boundaries, and removes punctuation/special characters.
You can create manually labeled examples and mix them into `train.json` before retraining.

### 5. Run Tests

```bash
# Run all tests (unit tests + integration tests + benchmark)
pytest tests/ -v

# Just the postprocessor unit tests (no model needed)
pytest tests/test_postprocessor.py -v

# Integration tests (requires trained model in models/onnx/quantized/)
pytest tests/test_inference.py -v
```

## Project Structure

```
name-parsing/
├── src/name_parsing/          # Main package (used at inference time)
│   ├── __init__.py            # Exports NameAddressParser
│   ├── config.py              # Labels, paths, hyperparameters
│   ├── model.py               # NameAddressParser: preprocess → tokenize → ONNX → postprocess
│   └── postprocessor.py       # Word-level entity extraction and street filtering
│
├── training/                  # Training pipeline (run once)
│   ├── generate_training_data.py  # Synthetic data with OCR noise
│   ├── train.py               # Fine-tune DistilBERT (tokenizes on-the-fly)
│   ├── export_onnx.py         # ONNX export + INT8 quantization
│   └── evaluate.py            # Per-field accuracy evaluation
│
├── tests/
│   ├── test_postprocessor.py  # 20 unit tests for post-processing logic
│   └── test_inference.py      # 12 integration tests + latency benchmark
│
├── notebooks/
│   ├── play.ipynb             # Interactive playground
│   └── examples.ipynb         # Usage examples and edge cases
│
├── models/                    # Model artifacts (generated by training)
│   ├── finetuned/             # PyTorch model checkpoint
│   └── onnx/
│       ├── onnx_export/       # Full-precision ONNX model
│       └── quantized/         # INT8 quantized model (used for inference)
│
├── data/raw/                  # Training/test data (generated)
│   ├── train.json             # 4000 examples (seed 42)
│   └── test.json              # 1000 examples (seed 99, held out)
│
├── environment.yaml           # Conda environment (all versions pinned)
├── pyproject.toml             # Package config + dependency groups
├── DOCUMENTATION.md           # Detailed technical documentation
└── README.md                  # This file
```

## How It Works

This project uses a **preprocessing-first** pipeline:

1. **Preprocess** the raw OCR text to split merged tokens before the model sees them:
   - CamelCase: `"JohnDoe"` → `"John Doe"`
   - Digit↔letter: `"37harbor"` → `"37 harbor"`
   - Special chars: `"37/harbor"` → `"37 harbor"`
   - Punctuation stripped: `"Doe,"` → `"Doe"`

2. **Tokenize** the cleaned word list with WordPiece using `is_split_into_words=True`

3. **Predict** a BIO label for each word using the quantized ONNX model:
   ```
   O  B-FIRST_NAME  I-FIRST_NAME  B-LAST_NAME  I-LAST_NAME  B-STREET_NAME  I-STREET_NAME
   ```

4. **Post-process**: extract entity spans from BIO predictions, filter generic street words

The preprocessing step is the critical insight — by splitting OCR-merged tokens *before* the model, each word gets a clean label. The model sees `"John"` and `"Doe"` as separate words with separate predictions, not a merged `"JohnDoe"` that it would have to decode internally.

For a deep dive into the architecture, design decisions, and implementation details, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Runtime Dependencies

For **inference only**, the minimal dependencies are:

- `onnxruntime` — runs the quantized model (~30 MB)
- `transformers` — provides the WordPiece tokenizer
- `numpy`

No PyTorch needed at runtime. Total footprint: ~67 MB model + ~50 MB libraries.

## Performance

| Metric | Value |
|--------|-------|
| Training F1 | 99.75% |
| first_name accuracy | 100% (500/500) |
| last_name accuracy | 99.0% (495/500) |
| street_name accuracy | 97.2% (485/499) |
| Overall accuracy | 98.7% (1480/1499) |
| Inference latency (p99) | ~10ms |
| Model size (quantized) | 67 MB |

Remaining errors are primarily OCR corruption that garbles the entity itself (e.g., `"ond"` OCR'd from `"and"` getting predicted as a last name) or digit-in-word substitutions (`"High1and"` → `"High 1 and"` where only the first word is predicted).

## License

MIT

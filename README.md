# Name & Address Parser

Extract structured data from customer records using a fine-tuned DistilBERT NER model.

```python
from name_parsing import NameAddressParser

parser = NameAddressParser("models/onnx/quantized")
result = parser.parse("Alex or Mary Doe, 1201 Braddock Ave, Richmond VA, 22312")
# {'first_name': 'Alex', 'last_name': 'Doe', 'street_name': 'Braddock'}
```

**Key features:**
- Handles individual persons, shared last names, and separate names ("John or Mary Doe", "John Doe or Mary Smith")
- Handles business payors — "Fairfax SushiMax LLC" → first_name="SushiMax", last_name="LLC"
- Handles middle names — both abbreviated ("M.") and full ("Monroe"), skipped cleanly
- Handles numbered streets — "1234 5th Ave" → street_name="5th"
- Handles P.O. Box addresses — street_name="Box"
- Labels are trained directly on raw input text — no preprocessing step
- Post-processing strips punctuation, possessives (`"GG's"` → `"GG"`), single-letter dot-prefixes (`"A.Professional"` → `"Professional"`), and filters generic street words after inference
- CPU-only inference, ~10ms latency (p99)
- 67 MB quantized ONNX model — no GPU or PyTorch at runtime

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

# Single person
parser.parse("John Smith, 500 Oak Ave, Denver CO 80201")
# {'first_name': 'John', 'last_name': 'Smith', 'street_name': 'Oak'}

# With abbreviated middle name — middle name is skipped
parser.parse("Madison M. Jackson, 500 Oak St, Denver CO 80201")
# {'first_name': 'Madison', 'last_name': 'Jackson', 'street_name': 'Oak'}

# With full middle name
parser.parse("Madison Monroe Jackson, 500 Oak St, Denver CO 80201")
# {'first_name': 'Madison', 'last_name': 'Jackson', 'street_name': 'Oak'}

# Multiple names — extracts only the first person
parser.parse("Alex or Mary Doe, 1201 Braddock Ave, Richmond VA 22312")
# {'first_name': 'Alex', 'last_name': 'Doe', 'street_name': 'Braddock'}

# Business payor — business name is first_name, type (LLC/Inc/etc.) is last_name
parser.parse("Fairfax SushiMax LLC, 1201 Braddock Ave, Richmond VA")
# {'first_name': 'SushiMax', 'last_name': 'LLC', 'street_name': 'Braddock'}

# Numbered / ordinal street
parser.parse("John Smith, 1234 5th Ave, Denver CO 80201")
# {'first_name': 'John', 'last_name': 'Smith', 'street_name': '5th'}

# P.O. Box address
parser.parse("Jane Doe, P.O. Box 1234, Arlington VA 22201")
# {'first_name': 'Jane', 'last_name': 'Doe', 'street_name': 'Box'}

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

# Step 3: Fine-tune DistilBERT
# Automatically uses MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise
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

The training data format is simple — raw text and word-level labels aligned with `text.split()`:

```json
{
  "text": "John Doe, 500 Oak Ave, Denver CO",
  "labels": [1, 3, 0, 5, 0, 0, 0]
}
```

Labels are integer IDs corresponding to `text.split()` words:
```
0=O, 1=B-FIRST_NAME, 2=I-FIRST_NAME, 3=B-LAST_NAME, 4=I-LAST_NAME,
5=B-STREET_NAME, 6=I-STREET_NAME
```

In the example above, `text.split()` produces `["John", "Doe,", "500", "Oak", "Ave,", "Denver", "CO"]`
and `labels[1]=3` means `"Doe,"` is the last name. Post-processing strips the comma automatically.

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
│   ├── model.py               # NameAddressParser: tokenize → ONNX → postprocess
│   └── postprocessor.py       # Entity extraction, punctuation stripping, street filtering
│
├── training/                  # Training pipeline (run once)
│   ├── generate_training_data.py  # Synthetic data generation
│   ├── train.py               # Fine-tune DistilBERT (MPS / CUDA / CPU)
│   ├── export_onnx.py         # ONNX export + INT8 quantization
│   └── evaluate.py            # Per-field accuracy evaluation
│
├── tests/
│   ├── test_postprocessor.py  # 29 unit tests for post-processing logic
│   └── test_inference.py      # Integration tests + latency benchmark
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

The pipeline has two phases: **training** (labels on raw text, no preprocessing) and **inference** (raw split → model → post-process).

### Training

Labels are assigned directly to whitespace-split words from raw text. No preprocessing step — what the model sees at inference is identical to what it was trained on.

```
"John Doe, 1234 Braddock Ave, Denver CO"
 ↓ text.split()
["John", "Doe,", "1234", "Braddock", "Ave,", "Denver", "CO"]
  FN      LN      O       SN          O        O         O
```

Punctuation like `"Doe,"` and `"Ave,"` is part of the raw word. The model learns to label these words correctly, and post-processing strips the punctuation at output time.

### Inference

```
1. Split on whitespace  →  raw words (no preprocessing)
2. Tokenize             →  WordPiece subtokens via is_split_into_words=True
3. ONNX inference       →  BIO label per word (~10ms on CPU)
4. Post-process         →  strip punctuation, filter generic street words, return result
```

**Why no preprocessing?** Preprocessing before labeling creates shifting ground truth — the labels end up on transformed text that differs from the original input. By labeling directly on raw text, training and inference are perfectly consistent.

**Post-processing handles:**
- Punctuation stripping: `"Doe,"` → `"Doe"`, `"LLC,"` → `"LLC"`
- Possessive stripping (names only): `"GG's"` → `"GG"`
- Single-letter dot-prefix stripping (names only): `"A.Professional"` → `"Professional"`
- Generic street word filtering: `"Ave"`, `"St"`, `"Blvd"` filtered out
- Ordinal number preservation: `"5th"`, `"1st"` kept as valid street names
- P.O. Box: `"Box"` returned directly as street_name

For a deep dive into the architecture, design decisions, and implementation details, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Runtime Dependencies

For **inference only**, the minimal dependencies are:

- `onnxruntime` — runs the quantized model
- `transformers` — provides the WordPiece tokenizer
- `numpy`

No PyTorch needed at runtime. Total footprint: ~67 MB model + ~50 MB libraries.

## Performance

| Metric | Value |
|--------|-------|
| Training F1 (seqeval) | 100% |
| Inference latency (p99) | ~10ms |
| Model size (quantized) | 67 MB |
| Training time (M1 Pro MPS) | ~5 min |
| Training examples | 4,000 synthetic |
| Test examples | 1,000 synthetic (separate seed) |

## License

MIT

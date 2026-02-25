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
- **SpaCy pre-tokenization** at both training and inference time — standard BERT fine-tuning practice; punctuation is automatically separated before labeling (e.g. "Doe," → ["Doe", ","])
- **OCR noise training** — 15% of training words have realistic OCR errors (character substitution, deletion, insertion, swaps) for robustness on scanned text
- **Minimal post-processing** — only the first predicted span per entity category is selected; raw model predictions returned directly
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

# OCR-noisy input (model is trained to handle these)
parser.parse("Rache1 Mend0za, 1201 Bradd0ck Ave, Richmond VA")
# {'first_name': 'Rache1', 'last_name': 'Mend0za', 'street_name': 'Bradd0ck'}

# Batch processing
parser.parse_batch(["John Smith, 100 Main St", "Jane Doe, 200 Oak Ave"])
```

### 3. Training From Scratch

```bash
# Step 1: Generate synthetic training data (5000 examples, SpaCy-tokenized + OCR noise)
python training/generate_training_data.py \
    --num-examples 5000 \
    --output data/raw/train.json \
    --seed 42

# Step 2: Generate a held-out test set (different seed to avoid leakage)
python training/generate_training_data.py \
    --num-examples 1000 \
    --output data/raw/test.json \
    --seed 123

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

The training data format stores SpaCy-tokenized words alongside integer labels:

```json
{
  "text": "John Doe, 500 Oak Ave, Denver CO",
  "words": ["John", "Doe", ",", "500", "Oak", "Ave", ",", "Denver", "CO"],
  "labels": [1, 3, 0, 0, 5, 0, 0, 0, 0]
}
```

The `words` list is SpaCy-tokenized (punctuation is a separate token). Labels align with `words`:
```
0=O, 1=B-FIRST_NAME, 2=I-FIRST_NAME, 3=B-LAST_NAME, 4=I-LAST_NAME,
5=B-STREET_NAME, 6=I-STREET_NAME
```

In the example: `words[1]="Doe"` has `labels[1]=3` (B-LAST_NAME), `words[2]=","` has `labels[2]=0` (O).
SpaCy already separates the comma so no punctuation stripping is needed in post-processing.

You can create manually labeled examples and mix them into `train.json` before retraining.

### 5. Run Tests

```bash
# Just the postprocessor unit tests (no model needed)
pytest tests/test_postprocessor.py -v

# Integration tests (requires trained model in models/onnx/quantized/)
pytest tests/test_inference.py -v

# All tests
pytest tests/ -v
```

## Project Structure

```
name-parsing/
├── src/name_parsing/          # Main package (used at inference time)
│   ├── __init__.py            # Exports NameAddressParser
│   ├── config.py              # Labels, paths, hyperparameters
│   ├── model.py               # NameAddressParser: SpaCy → ONNX → postprocess
│   └── postprocessor.py       # BIO entity extraction, first-span selection
│
├── training/                  # Training pipeline (run once)
│   ├── generate_training_data.py  # Synthetic data: SpaCy-tokenized + OCR noise
│   ├── train.py               # Fine-tune DistilBERT (MPS / CUDA / CPU)
│   ├── export_onnx.py         # ONNX export + INT8 quantization
│   └── evaluate.py            # Per-field accuracy evaluation
│
├── tests/
│   ├── test_postprocessor.py  # Unit tests for BIO extraction and postprocess
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
│   ├── train.json             # 5000 examples (seed 42)
│   └── test.json              # 1000 examples (seed 123, held out)
│
├── environment.yaml           # Conda environment (all versions pinned)
├── pyproject.toml             # Package config + dependency groups
├── DOCUMENTATION.md           # Detailed technical documentation
└── README.md                  # This file
```

## How It Works

The pipeline has two phases: **training** (SpaCy-tokenized + OCR noise → model) and **inference** (SpaCy tokenize → model → pick first span).

### Training

Text is SpaCy-tokenized before label assignment. SpaCy separates punctuation into distinct tokens, so labels land on clean words — no punctuation stripping needed later.

```
"John Doe, 1234 Braddock Ave, Denver CO"
 ↓ SpaCy tokenize
["John", "Doe", ",", "1234", "Braddock", "Ave", ",", "Denver", "CO"]
  FN      LN    O     O       SN          O     O     O          O
```

~15% of words also have realistic OCR noise applied (character substitutions, deletions, insertions) to make the model robust to scanned text imperfections.

### Inference

```
1. SpaCy tokenize  →  clean word tokens (punctuation separated)
2. BERT tokenize   →  WordPiece subtokens via is_split_into_words=True
3. ONNX inference  →  BIO label per word (~10ms on CPU)
4. Post-process    →  pick first span per entity category, join words
```

**Why SpaCy pre-tokenization?** It is the standard approach for BERT fine-tuning NLP tasks. Using the same tokenizer at training and inference time ensures the model always sees the same token boundaries. SpaCy's blank English model is rule-based (no ML model download needed) and consistently separates punctuation from words.

**Post-processing** is minimal: extract BIO entity spans, select only the first span per category (first_name, last_name, street_name), join its words. No filtering or cleaning — entity words are already clean because SpaCy separated punctuation as "O" tokens.

For a deep dive into the architecture, design decisions, and implementation details, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Runtime Dependencies

For **inference only**, the minimal dependencies are:

- `onnxruntime` — runs the quantized model
- `transformers` — provides the WordPiece tokenizer
- `numpy`
- `spacy` — pre-tokenizes input text before inference (blank English model, no download needed)

No PyTorch needed at runtime. Total footprint: ~67 MB model + libraries.

## Performance

| Metric | Value |
|--------|-------|
| Training F1 (seqeval) | 99.84% |
| Inference latency (p99) | ~10ms |
| Model size (quantized) | 67 MB |
| Training time (M1 Pro MPS) | ~7 min |
| Training examples | 5,000 synthetic (with OCR noise) |
| Test examples | 1,000 synthetic (seed 123, held out) |

## License

MIT

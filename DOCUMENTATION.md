# Name & Address Parser — Technical Documentation

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Key Concepts](#key-concepts)
3. [Architecture and Design Decisions](#architecture-and-design-decisions)
4. [How It All Fits Together](#how-it-all-fits-together)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [Performance Characteristics](#performance-characteristics)
7. [How to Improve the Model](#how-to-improve-the-model)

---

## What This Project Does

This project takes text from customer records and extracts three pieces of structured data:

```
Input:  "Alex or Mary Doe, 1201 Braddock Ave, Richmond VA, 22312"
Output: {"first_name": "Alex", "last_name": "Doe", "street_name": "Braddock"}
```

The challenge is that this text is messy and takes many forms:
- Multiple names: "Alex or Mary Doe", "Alex Doe or Mary Smith"
- Middle names: "Madison M. Jackson", "Madison Monroe Jackson"
- Business payors: "Fairfax SushiMax LLC" → first_name="SushiMax", last_name="LLC"
- Numbered streets: "1234 5th Ave" → street_name="5th"
- P.O. Box addresses: "P.O. Box 1234" → street_name="Box"
- OCR-corrupted text: "Rache1 Mend0za" (l→1, o→0 substitutions)
- Varying formatting (commas missing, emails appended, phone numbers mixed in)

### Constraints

- **No LLM**: Generative large language models are out of scope
- **CPU-only inference**: No GPU assumed in production
- **Under 100ms latency**: Hard latency budget per parse call
- **Extract first entity only**: When multiple names appear, extract only the first person's (or business's) first name and last name

---

## Key Concepts

### What is NER (Named Entity Recognition)?

NER labels each word in text with what type of thing it is:

```
Alex     or    Mary   Doe    1201   Braddock  Ave   Richmond  VA
FIRST    -     -      LAST   -      STREET    -     -         -
```

The model looks at each word in context and decides: first name? last name? street name? none of the above?

**Why NER over regex?** Customer records have too many format variations for rules to handle reliably. A learned model generalizes across variations rather than needing a hand-written rule for each edge case.

### What is BIO Tagging?

BIO stands for **B**egin, **I**nside, **O**utside — a labeling scheme that marks where entities start and continue:

- **B-FIRST_NAME** = this word begins a first name entity
- **I-FIRST_NAME** = this word continues a first name
- **O** = not part of any tracked entity

Our full label set has 7 labels:
```
O, B-FIRST_NAME, I-FIRST_NAME, B-LAST_NAME, I-LAST_NAME, B-STREET_NAME, I-STREET_NAME
```

**Why only these three entity types?** The requirement is to extract first name, last name, and street name. Labeling only what we need keeps the model focused. Everything else (city, state, zip, email, phone) is labeled "O" — which helps the model understand structure by contrast.

### What is DistilBERT?

DistilBERT is a smaller, faster version of BERT — a transformer encoder model (it reads and classifies text, not generates it). Key facts:

- 66M parameters, 6 layers (BERT has 110M and 12 layers)
- Uses **WordPiece tokenization**: words are split into subwords. "Braddock" → `["brad", "##dock"]`. The `##` prefix means "continuation of the previous token"
- We use `distilbert-base-uncased`, which lowercases all input

**Why DistilBERT over alternatives?**
- **vs. BERT**: 40% fewer parameters, 60% faster, retains 97% of performance — sufficient for our narrow domain
- **vs. RoBERTa**: Larger and slower; unnecessary capacity for short single-line inputs
- **vs. CRF**: CRFs require hand-crafted features and don't handle text variation as well as learned embeddings
- **vs. spaCy NER**: spaCy's built-in models target general entities (PERSON, ORG, GPE); fine-tuning for custom labels on our domain gives better control

### What is SpaCy Pre-Tokenization?

Before text is fed to DistilBERT, it is first tokenized with SpaCy's blank English tokenizer — a rule-based tokenizer that properly handles punctuation separation, abbreviations, and other edge cases.

**Why SpaCy before BERT?** This is the standard approach for BERT fine-tuning NLP tasks. SpaCy tokenization is applied identically at both training time (in `generate_training_data.py`) and inference time (in `model.py`). This guarantees that the model always sees the same token boundaries it was trained on.

**Key behavior**: SpaCy splits punctuation into separate tokens:
```
"Doe,"  →  ["Doe", ","]
"Ave,"  →  ["Ave", ","]
"LLC,"  →  ["LLC", ","]
```

This means entity words are already clean (no trailing comma) — the "," gets label "O" during training, and no punctuation stripping is needed in post-processing.

**What SpaCy model is used?** `spacy.blank("en")` — the blank English tokenizer with rule-based tokenization only. No ML model download is needed; just `pip install spacy`.

### What is OCR Noise Augmentation?

Approximately 15% of words in each training example have realistic OCR errors applied before labeling. This teaches the model to extract entities from imperfect scanned text.

**OCR error types simulated:**
- **Character substitution** (40%): visually similar glyph swaps — `l`↔`1`, `O`↔`0`, `I`→`1`, `n`↔`u`, `S`→`5`, etc.
- **Character deletion** (20%): an interior character is dropped — "Braddock" → "Bradock"
- **Character insertion** (15%): a character is doubled (smudge) — "Smith" → "Smiith"
- **Adjacent swap** (10%): two neighboring characters transposed — "John" → "Jonh"
- **Bigram confusion** (15%): multi-character misreads — "rn"↔"m", "vv"↔"w", "cl"→"d"

Only words with 4+ characters that contain at least one letter are noised. This preserves short structural tokens ("or", "and", state codes) that are critical for parsing.

### What is ONNX and Quantization?

- **ONNX** (Open Neural Network Exchange) is a format for saving models that run without PyTorch. ONNX Runtime applies graph-level optimizations (operator fusion, memory planning) not available in PyTorch's eager mode.
- **Quantization** converts 32-bit float weights to 8-bit integers. The model becomes ~4× smaller and faster with minimal accuracy loss. We use **dynamic quantization** — weights are INT8, activations are quantized at inference time without needing a calibration dataset.

### What is Fine-Tuning?

DistilBERT was pre-trained on general English text (Wikipedia, books) and understands English. **Fine-tuning** trains it further on our specific task — labeling customer record text — so it learns the domain patterns. Even 5,000 synthetic examples are sufficient because the base model already understands language; it just needs to learn what "first name", "last name", and "street name" mean in this context.

---

## Architecture and Design Decisions

### Decision 1: SpaCy Pre-Tokenization (Training and Inference)

The most important architectural change from raw-whitespace-split approaches.

**The approach:** SpaCy's blank English tokenizer is applied to text before BERT tokenization, both when generating training data and at inference time. This is the standard NLP pipeline for BERT fine-tuning.

**Why SpaCy and not raw whitespace split?**
1. **Proper token boundaries**: SpaCy correctly splits "Doe," into ["Doe", ","] — the entity word is separated from punctuation. Labels then land on clean words.
2. **Consistency**: The same tokenizer runs identically at training and inference time. What the model was trained on is exactly what it sees at inference.
3. **Handles edge cases**: SpaCy's English rules handle abbreviations ("P.O." stays one token), contractions, and other linguistic patterns correctly.

**Training data impact**: A label like B-LAST_NAME is assigned to the word "Doe" (clean), not "Doe," (with punctuation). The "," becomes a separate "O" token. This produces cleaner supervision signal.

**Inference impact**: `parse()` calls `[token.text for token in _nlp(text)]` instead of `text.split()`. The resulting word list is fed to BERT with `is_split_into_words=True`.

### Decision 2: Minimal Post-Processing (First Span Only)

Post-processing is reduced to a single operation: select the first predicted BIO span per entity category and join its words.

**What was removed:** The previous postprocessor applied punctuation stripping (`_clean_word`), possessive/dot-prefix removal (`_clean_name_word`), numeric filtering, and a generic street word filter (`filter_street_name`). These are no longer needed because:

1. **Punctuation stripping**: SpaCy already separates punctuation as "O" tokens. Entity words in predictions are inherently clean.
2. **Generic word filter**: Only the street name (e.g., "Braddock") is labeled B-STREET_NAME in training data; the suffix (e.g., "Ave") is always labeled "O". The model learns this distinction directly — no filter needed.
3. **Possessive/prefix handling**: Edge cases like "GG's" or "A.Professional" are better handled by training data coverage or deferred to downstream processing per the user's needs.

**What remains:** `extract_entities()` (BIO state machine, unchanged) and `postprocess()` which selects `entities["FIRST_NAME"][0]` etc. and joins with spaces.

### Decision 3: OCR Noise Augmentation in Training Data

OCR noise is applied to ~15% of words during synthetic data generation to make the model robust to scanned-text imperfections.

**Design choices:**
- **Noise probability 15%**: Enough to expose the model to errors without overwhelming the signal. At 15% per word, a 12-word example has ~1.8 noisy words on average.
- **4+ character threshold**: Short words (connectors, state codes, directional prefixes) are preserved so the model can still parse structure.
- **Noise applied before SpaCy tokenization**: Noise changes the text representation but not the label. SpaCy tokenizes the noised word, first sub-token keeps the original label.
- **Noise retained in output**: The model predicts labels on the noised tokens. If "Rache1" is labeled B-FIRST_NAME, the output is "Rache1". Users can apply their own OCR correction if needed.

### Decision 4: Training Data Format with SpaCy Words List

Training data stores a `words` list (SpaCy-tokenized) alongside the original `text`:

```json
{
  "text": "John Doe, 1234 Braddock Ave, Denver CO",
  "words": ["John", "Doe", ",", "1234", "Braddock", "Ave", ",", "Denver", "CO"],
  "labels": [1, 3, 0, 0, 5, 0, 0, 0, 0]
}
```

`labels[i]` corresponds directly to `words[i]`. The `text` field is kept for human readability. The `words` list is what training and evaluation actually use.

**Why not store only `text`?** With SpaCy tokenization, `text.split()` gives different tokens than SpaCy — using the pre-computed `words` list avoids re-tokenizing at training time and ensures consistency.

### Decision 5: Word-Level NER with `is_split_into_words=True`

SpaCy words are pre-split and passed to the BERT tokenizer with `is_split_into_words=True`. The tokenizer's `word_ids()` output gives a direct word index for each BERT subtoken, making label alignment simple:

- **First-subtoken rule**: The prediction of the first BERT subtoken of each SpaCy word represents that word's label. Continuation subtokens (same `word_id`) are ignored.
- **No offset arithmetic**: `word_ids()` eliminates the need for character offset mapping
- **Simple entity reconstruction**: Entity text is just `" ".join(entity_words)` — no gap detection needed
- **Training data portability**: Labels stored as one-integer-per-word, so data is human-readable and tokenizer-agnostic

### Decision 6: Synthetic Training Data

We don't have a labeled dataset of real records. Instead we generate synthetic data from templates using real name lists (100 first names, 100 last names), real street names, and Faker-generated cities.

**Data variations** (to train the model on the full expected input space):
- **Person types**: individual (30%), shared last name (20%), separate persons (20%), business payor (30%)
- **Middle names**: abbreviated `"M."` or full `"Monroe"` at ~25% probability, labeled `O` so only the given name is the first_name entity
- **Street types**: regular named streets (60%), ordinal numbered streets like `"5th Ave"` (25%), P.O. Box addresses (15%)
- **Business payors**: business name word is `B-FIRST_NAME`, business type (LLC/Inc/Corp/Ltd/Co/LP/LLP) is `B-LAST_NAME`; optional location prefix word (labeled `O`)
- **OCR noise**: ~15% of words per example have realistic character-level errors

**Why separate train/test seeds?** Training data uses `--seed 42`; test data uses `--seed 123`. This ensures the evaluation dataset exercises different random name/address combinations, measuring generalization rather than memorization.

### Decision 7: First-Entity-Only Extraction

Extraction of only the first person/business is achieved at two layers:

1. **Training data**: Only the first first-name and last-name tokens are labeled. All subsequent names are labeled `O`. The model learns to label only the first entity.
2. **Postprocessor**: `postprocess()` takes `entities["FIRST_NAME"][0]` and `entities["LAST_NAME"][0]` — the first detected span. This acts as a safety net if the model occasionally mislabels a secondary name.

### Decision 8: GPU Acceleration for Training, CPU for Inference

Training uses the best available accelerator automatically:
- **MPS** (Metal Performance Shaders) on Apple Silicon (M1/M2/M3)
- **CUDA** on NVIDIA GPUs
- **CPU** as fallback

The HuggingFace Trainer + Accelerate library handles device selection. `fp16=False` is set because neither MPS nor CPU support 16-bit float training.

Production inference uses CPU-only ONNX Runtime, which is appropriate because:
- Inputs are tiny (typically <64 tokens) — GPU parallelism doesn't help at this scale
- Single-threaded ONNX is predictable (~10ms) and avoids overhead from parallelism on short inputs
- ONNX Runtime has no dependency on PyTorch or any GPU libraries at runtime

### Decision 9: ONNX Runtime with Single-Thread Configuration

The inference pipeline uses `intra_op_num_threads=1` and `inter_op_num_threads=1`.

Multi-threaded inference only pays off for large inputs where parallelizing matrix operations is beneficial. For our short inputs (~40-60 SpaCy tokens), thread coordination overhead outweighs any parallelism gain. Single-threaded gives predictable, consistent ~10ms latency.

In a production server, concurrency is handled at the request level (multiple threads or processes, each running single-threaded ONNX) — not within a single inference call.

### Decision 10: MAX_SEQ_LENGTH = 64

Customer records are short text — a name, an address, and sometimes an email or phone. 64 tokens covers even the longest realistic inputs. (Note: with SpaCy tokenization, punctuation is split, so the token count is slightly higher than with whitespace splitting — 64 BERT tokens still covers all typical inputs comfortably.)

Transformer self-attention is O(n²) in sequence length. Padding to 128 or 512 would waste compute on padding tokens since real inputs rarely exceed ~50-60 SpaCy tokens.

### Decision 11: Training Hyperparameters

- **Learning rate 5e-5**: Standard for BERT fine-tuning. Too low → slow convergence; too high → destabilizes pre-trained weights
- **15 epochs**: More than typical fine-tuning (3-5 epochs) because the dataset is small (5,000 examples). More passes help with generalization
- **Batch size 16**: Stable gradient estimates while fitting in memory on CPU/MPS
- **Warmup ratio 0.1**: First 10% of steps use linearly increasing LR. Prevents the randomly-initialized classification head from producing large gradients that damage the pre-trained weights
- **Weight decay 0.01**: L2 regularization to reduce overfitting on a small dataset
- **85/15 train/eval split**: ~750 eval examples for reliable in-training metrics; the separate test set (seed 123) measures true generalization
- **`load_best_model_at_end=True` with `metric_for_best_model="f1"`**: Saves the checkpoint with highest seqeval F1. Seqeval evaluates at entity level — an entity is correct only if all its B-/I- tokens are predicted correctly, stricter than per-token accuracy

---

## How It All Fits Together

### Training Phase

```
1. generate_training_data.py
       Builds (raw_word, label) token lists from templates
       Applies OCR noise to ~15% of words (char substitutions, deletions, etc.)
       SpaCy-tokenizes each (possibly noised) word:
         "Doe,"  → ["Doe", ","]  →  ("Doe", B-LAST_NAME), (",", O)
         "Rache1" → ["Rache1"]   →  ("Rache1", B-FIRST_NAME)
       Stores: {text, words, labels}   ← words are SpaCy tokens
         |
         v
2. train.py
       Reads words list directly from data (no re-tokenization)
       Tokenizes with is_split_into_words=True → BERT WordPiece subtokens
       Expands word labels to subtoken labels (first subtoken = label, rest = -100)
       Fine-tunes DistilBERT on MPS / CUDA / CPU
       85% train / 15% eval; saves best checkpoint by seqeval F1
         |
         v
3. export_onnx.py
       Traces PyTorch model → ONNX graph (265 MB)
       Applies dynamic INT8 quantization → 67 MB
       Copies tokenizer to quantized directory (self-contained for deployment)
         |
         v
4. evaluate.py
       Loads held-out test.json (seed 123, not seen during training)
       Runs full inference pipeline for each example
       Reconstructs expected values from words + labels (no cleaning needed)
       Reports per-field accuracy + sample errors
```

### Inference Phase

```
Input: "Fairfax SushiMax LLC, 1201 5th Ave, Richmond VA"
    |
    v
 NameAddressParser.parse()
    |
    |-- 1. SpaCy tokenize
    |      words = ["Fairfax", "SushiMax", "LLC", ",", "1201", "5th", "Ave", ",", "Richmond", "VA"]
    |
    |-- 2. BERT tokenize with is_split_into_words=True
    |      subtokens: [CLS] fairfax sushi ##max llc , 1201 5th ave , richmond va [SEP]
    |      word_ids:  [None  0       1     1     2  3 4    5   6  7 8        9   None]
    |
    |-- 3. ONNX inference (~10ms on CPU)
    |      logits → argmax → label IDs per subtoken
    |
    |-- 4. postprocess()
    |      First-subtoken rule: word 0→O, word 1→B-FIRST_NAME, word 2→B-LAST_NAME,
    |                           word 3→O, word 4→O, word 5→B-STREET_NAME, ...
    |
    |      BIO state machine:
    |        FIRST_NAME:  [["SushiMax"]]
    |        LAST_NAME:   [["LLC"]]
    |        STREET_NAME: [["5th"]]
    |
    |      Select first span per category, join words:
    |        first_name  = "SushiMax"
    |        last_name   = "LLC"
    |        street_name = "5th"
    |
    v
{"first_name": "SushiMax", "last_name": "LLC", "street_name": "5th"}
```

---

## File-by-File Breakdown

### `pyproject.toml` — Project Configuration

Three dependency groups:

- **Runtime** (always needed): `onnxruntime`, `transformers` (for tokenizer), `numpy`, `spacy` (for pre-tokenization). No PyTorch at runtime.
- **Training** (`pip install -e ".[train]"`): adds `torch`, `datasets`, `optimum[onnxruntime]`, `faker`, `seqeval`, `accelerate`
- **Dev** (`pip install -e ".[dev]"`): adds `pytest`, `pytest-benchmark`

---

### `src/name_parsing/config.py` — Central Configuration

Single source of truth for all constants used across training and inference.

**Label definitions:**
```python
LABEL_LIST = ["O", "B-FIRST_NAME", "I-FIRST_NAME", "B-LAST_NAME", "I-LAST_NAME", "B-STREET_NAME", "I-STREET_NAME"]
LABEL2ID = {"O": 0, "B-FIRST_NAME": 1, ...}
ID2LABEL = {0: "O", 1: "B-FIRST_NAME", ...}
```

`"O"` must be index 0 because it is the default label. `-100` is the PyTorch convention for `CrossEntropyLoss` ignored positions (used for continuation subtokens and special tokens).

---

### `src/name_parsing/model.py` — Inference Pipeline

Contains the `NameAddressParser` class.

**Module-level SpaCy tokenizer**: `_nlp = spacy.blank("en")` — initialized once at import time.

**`__init__(model_dir)`**: Finds the `.onnx` file in the model directory, creates a single-threaded `ort.InferenceSession` with all graph optimizations enabled, loads the tokenizer from the same directory.

**`parse(text)`**:
1. Guard clause: returns empty fields for empty input
2. `words = [token.text for token in _nlp(text)]` — SpaCy tokenization, same as training
3. `tokenizer(words, is_split_into_words=True, return_tensors="np")` — NumPy arrays are ONNX Runtime's native format, no PyTorch dependency at inference
4. ONNX inference: feeds `input_ids` and `attention_mask`, receives logits shape `(1, seq_len, 7)`
5. Argmax → predicted label IDs per subtoken
6. `postprocess(predictions, words, word_ids)` → final dict

**`parse_batch(texts)`**: Processes texts sequentially. Batching multiple inputs into one ONNX call would improve throughput (not latency) if needed.

---

### `src/name_parsing/postprocessor.py` — Entity Extraction

**`extract_entities(predictions, words, word_ids)`**:
1. First-subtoken rule: build `word_predictions` dict mapping word_idx → label_id using first occurrence only
2. BIO state machine over words:
   - B- tag: close any open entity, start new span
   - I- tag matching current entity: append word to span
   - I- tag mismatching or O tag: close current span
3. Returns `{"FIRST_NAME": [[words], ...], "LAST_NAME": [...], "STREET_NAME": [...]}`

**`postprocess(predictions, words, word_ids)`**: Main entry point. Calls `extract_entities()`, takes `[0]` of each entity type (first occurrence = first person/business), joins words with space. No additional cleaning — SpaCy-tokenized words are already clean.

---

### `training/generate_training_data.py` — Synthetic Training Data

Generates `(word, label)` token lists, applies OCR noise, SpaCy-tokenizes, and stores the result.

**Token-building approach**: Each element in the `tokens` list is a `(word_string, label_string)` tuple with punctuation still attached (e.g., `("Doe,", "B-LAST_NAME")`). Then:

1. OCR noise applied per word at 15% probability
2. SpaCy tokenizes each word: `"Doe,"` → `["Doe", ","]`
3. First sub-token inherits the word's label; additional sub-tokens get `"O"`

```python
# After noise + SpaCy tokenization:
# ("Rache1", "B-FIRST_NAME"), ("Mend0za,", "B-LAST_NAME") → SpaCy →
# ("Rache1", "B-FIRST_NAME"), ("Mend0za", "B-LAST_NAME"), (",", "O")
```

**Templates**:
- **single** (30%): `"John Doe, 1234 Braddock Ave, Denver CO"` — one person
- **shared_last** (20%): `"John or Mary Doe, 1234 Braddock Ave"` — two first names, one last
- **separate_names** (20%): `"John Doe or Mary Smith, 1234 Braddock Ave"` — two full names; only first labeled
- **business** (30%): `"Fairfax SushiMax LLC, 1234 Braddock Ave"` — optional prefix (O), business name (B-FIRST_NAME), type (B-LAST_NAME)

**Street variants**:
- Regular (60%): `"1234 Braddock Ave,"` — street name from `STREET_NAMES` list
- Ordinal (25%): `"1234 5th Ave,"` — from `ORDINAL_STREETS` list (`"1st"` through `"50th"`)
- P.O. Box (15%): `"P.O. Box 1234,"` — only `"Box"` is labeled `B-STREET_NAME`

**Output format**:
```json
{
  "text": "John Doe, 1234 Braddock Ave, Denver CO",
  "words": ["John", "Doe", ",", "1234", "Braddock", "Ave", ",", "Denver", "CO"],
  "labels": [1, 3, 0, 0, 5, 0, 0, 0, 0]
}
```

---

### `training/train.py` — Fine-Tuning DistilBERT

**Device detection**: Prints and uses MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU automatically.

**`tokenize_and_align_labels(examples, tokenizer)`**: Reads `examples["words"]` directly (pre-computed SpaCy tokens), tokenizes with `is_split_into_words=True`, then expands word-level labels to subtoken labels. First subtoken → word's label; continuation subtokens and special tokens → `-100` (ignored in loss).

**`load_data(data_path, tokenizer)`**: Reads JSON, extracts `words` and `labels` fields, applies tokenization via `.map()`, splits 85/15 into train/eval with `seed=42`.

**`DataCollatorForTokenClassification`**: Dynamically pads each batch to the longest sequence in that batch (more efficient than always padding to `MAX_SEQ_LENGTH`). Pads `labels` with `-100`.

**`compute_metrics()`**: Uses `seqeval` for entity-level F1 — an entity is correct only if every token in the span has the right label.

---

### `training/export_onnx.py` — ONNX Export and Quantization

**Step 1 — ONNX Export**: `optimum`'s `ORTModelForTokenClassification.from_pretrained(..., export=True)` traces the PyTorch model and produces an ONNX graph.

**Step 2 — INT8 Dynamic Quantization**: `ORTQuantizer` with `AutoQuantizationConfig.avx2(is_static=False)`. The `avx2` config targets x86 CPUs with AVX2 support (most modern Intel/AMD production servers). Dynamic quantization requires no calibration dataset — activation ranges are computed at inference time.

**Size reduction**: 265 MB (FP32) → 67 MB (INT8) — 75% reduction.

The quantized directory is self-contained for deployment: ONNX model + tokenizer files + label map JSON.

---

### `training/evaluate.py` — Model Evaluation

Measures **end-to-end accuracy** by running the full inference pipeline against held-out test data — separate seed from training, so no data leakage.

**`extract_expected_from_example(ex)`**: Reconstructs expected values from training labels. Reads the `words` list directly and walks the BIO state machine to find the first span of each entity. No cleaning applied — SpaCy words are already clean; joins entity words with space to match inference output.

**Comparison**: Case-insensitive. Fields with empty expected value are skipped. Reports per-field accuracy + overall + first 10 sample errors.

---

### `tests/test_postprocessor.py` — Unit Tests

**`TestExtractEntities`** (7 tests): BIO predictions → entity spans with SpaCy-tokenized inputs:
- SpaCy words (clean, no attached punctuation): `"Doe"`, `"LLC"`, `"Braddock"`
- Single name, shared last name, separate names, business payor
- Multi-BERT-subtoken words (first-subtoken rule)
- Empty words list; mismatched I- tag (closes entity, discards word)

**`TestPostprocess`** (8 tests): End-to-end post-processing:
- Clean SpaCy words — output is already punctuation-free
- Shared last name, business payor, ordinal street, P.O. Box
- Empty input; multi-word first name; no-entity input

---

### `tests/test_inference.py` — Integration Tests

Tests require a trained ONNX model (skipped if not present):

**`TestInference`** (11 tests): Full parse pipeline:
- Single name, shared last name, separate names
- Business payor with and without location prefix
- Ordinal street, P.O. Box
- Middle initial (result includes only given name, middle skipped)
- Returns all expected keys; empty input; punctuation not in output

**`TestBenchmark`** (1 test): 100 inference calls after 5 warmup calls. Asserts p99 < 100ms.

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Training F1 (seqeval) | 99.84% |
| Training precision | 99.82% |
| Training recall | 99.87% |
| Inference latency (p50) | ~10ms |
| Inference latency (p99) | ~10ms |
| Model size (quantized) | 67 MB |
| Model size (FP32 ONNX) | 265 MB |
| Training time (M1 Pro MPS) | ~7 min |
| Training examples | 5,000 synthetic (with OCR noise) |
| Test examples | 1,000 synthetic (seed 123, held out) |

---

## How to Improve the Model

1. **Add real labeled examples**: The model was trained on synthetic data only. Even 200-500 manually labeled real records would improve accuracy on edge cases the templates don't cover. The format (`words`, `labels` per SpaCy token) is designed to make manual labeling straightforward.

2. **Increase OCR noise coverage**: Raise `_NOISE_PROB` in `generate_training_data.py` (currently 0.15) or add new noise types (e.g., character case errors, whole-word deletions) if the deployment environment has particularly noisy OCR.

3. **Expand template coverage**: Add more patterns to `generate_training_data.py` — apartment/suite numbers, international addresses, additional business type formats, records without commas.

4. **Add post-processing as needed**: The minimal postprocessor returns raw model predictions. If evaluation reveals common false positives (e.g., "Ave" labeled as STREET_NAME), add targeted rules. The user decides what post-processing is warranted based on observed errors.

5. **Target specific failures**: Run `evaluate.py`, examine the errors, and add training examples that directly address the failing pattern.

6. **Try a larger base model**: Swap `distilbert-base-uncased` for `bert-base-uncased` in `config.py`. This roughly doubles inference time (~20ms) but may gain 1-2% accuracy. Still well under the 100ms budget.

7. **Increase training data volume**: Generate 10,000–20,000 examples instead of 5,000. More data with more random name/city combinations generally improves generalization, especially for rare patterns.

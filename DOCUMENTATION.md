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
- Varying formatting (commas missing, emails appended, phone numbers mixed in)

### Constraints

- **No LLM**: Generative large language models are out of scope
- **CPU-only inference**: No GPU assumed in production
- **Under 100ms latency**: Hard latency budget per parse call
- **Extract first entity only**: When multiple names appear, extract only the first person's (or business's) first name and last name
- **Most distinctive street word**: "Braddock" from "1201 Braddock Ave", not "Ave" or "1201"

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

### What is the Raw-Input Labeling Approach?

The core design principle: **labels are assigned directly to whitespace-split words from the raw input text. No preprocessing before labeling.**

When training data is generated, words are split by whitespace from the raw sentence. Labels correspond to those raw words, including any attached punctuation:

```
Text:   "John Doe, 1234 Braddock Ave, Denver CO"
Words:  ["John", "Doe,", "1234", "Braddock", "Ave,", "Denver", "CO"]
Labels: [B-FIRST  B-LAST  O       B-STREET   O       O         O  ]
```

At inference time, the model receives the same raw whitespace-split words it was trained on. Post-processing then handles cleaning — stripping punctuation from entity words, filtering generic street suffixes, etc.

**Why raw-input labeling instead of preprocessing first?**

Preprocessing before labeling creates *shifting ground truth*: the labels end up on transformed text that differs from the original input. For example, if preprocessing removes punctuation, the label `B-LAST_NAME` gets assigned to `"Doe"` — but at inference time, the model might see `"Doe,"` (with comma), a different token. Training on one distribution and inferring on another introduces subtle mismatches. By labeling on raw text, training and inference are guaranteed to be identical.

### What is ONNX and Quantization?

- **ONNX** (Open Neural Network Exchange) is a format for saving models that run without PyTorch. ONNX Runtime applies graph-level optimizations (operator fusion, memory planning) not available in PyTorch's eager mode.
- **Quantization** converts 32-bit float weights to 8-bit integers. The model becomes ~4× smaller and faster with minimal accuracy loss. We use **dynamic quantization** — weights are INT8, activations are quantized at inference time without needing a calibration dataset.

### What is Fine-Tuning?

DistilBERT was pre-trained on general English text (Wikipedia, books) and understands English. **Fine-tuning** trains it further on our specific task — labeling customer record text — so it learns the domain patterns. Even 4,000 synthetic examples are sufficient because the base model already understands language; it just needs to learn what "first name", "last name", and "street name" mean in this context.

---

## Architecture and Design Decisions

### Decision 1: Raw-Input Labeling (No Preprocessing Before the Model)

The most important architectural decision.

**The rejected alternative (preprocessing-first):** Apply text transformations — split CamelCase, strip punctuation, split digit/letter boundaries — before assigning labels. The model then trains on cleaned text.

**The problem with preprocessing-first:** Ground truth becomes unstable. Labels are assigned to preprocessed text, but the original raw text is what matters to users. If preprocessing changes token boundaries (e.g., "Doe," → "Doe"), the label for the last name now applies to a token the model won't see when given raw input from a different formatting context. Any mismatch between the preprocessing logic applied at training vs. inference time silently degrades accuracy.

**The raw-input approach:** Labels are assigned to `text.split()` words directly. At inference time, `parse()` does exactly `words = text.split()` — nothing else before tokenization. Whatever punctuation or formatting is in the raw text, the model is trained on the same.

**Post-processing handles cleanup:** Punctuation stripping, generic word filtering, and field extraction are post-inference steps applied after the model has already predicted labels. This is the correct separation: the model predicts *where* entities are; post-processing decides how to *clean and format* the output.

### Decision 2: Post-Processing for Punctuation and Field Extraction

Since labels live on raw words (e.g., `"Doe,"`, `"LLC,"`), the post-processor must clean entity words before returning them.

**`_clean_word(word)`** strips leading and trailing punctuation characters (`.,;:!?"'()[]{}/-\`). This is applied to every word in extracted entity spans:
- `"Doe,"` → `"Doe"`
- `"LLC,"` → `"LLC"`
- `"5th"` → `"5th"` (no change — no punctuation)

**`_clean_name_word(word)`** extends `_clean_word` with two additional normalizations applied only to first and last name words:
1. **Strip possessive `'s`**: removes a trailing `'s` (case-insensitive) — e.g., `"GG's"` → `"GG"`
2. **Strip single-letter dot-prefix**: removes a leading `X.` pattern where X is one letter — e.g., `"A.Professional"` → `"Professional"`

**`filter_street_name()`** additionally filters out generic words that label as STREET_NAME but aren't the distinctive part:
1. Strip punctuation from each word
2. Filter words in `GENERIC_STREET_WORDS` (suffixes: ave, st, blvd, dr...; directions: n, s, e, w, north...)
3. Filter pure numeric tokens (≥50% digit characters) — catches house numbers
4. Preserve ordinal numbers: `"5th"`, `"1st"`, `"22nd"` match `\d+(st|nd|rd|th)` and are kept
5. Return the first remaining word; fall back to first word if all filtered

**Why a hardcoded generic word list?** The set of common street suffixes and directions is finite and stable. A learned filtering approach would add complexity without meaningful benefit.

**Why ordinal exemption for numerics?** The ≥50% digit threshold would incorrectly filter "5th" (1 digit / 3 chars = 33%) — but "10th" (2 digits / 4 chars = 50%) would be caught by the threshold. The explicit ordinal regex (`\d+(st|nd|rd|th)`) handles all ordinals before the numeric check runs.

### Decision 3: Word-Level NER with `is_split_into_words=True`

Rather than character-level tokenization, words are pre-split on whitespace and passed to the tokenizer with `is_split_into_words=True`. The tokenizer's `word_ids()` output gives a direct word index for each subtoken, making label alignment simple:

- **First-subtoken rule**: The prediction of the first subtoken of each word represents that word's label. Continuation subtokens (same `word_id` as a previous position) are ignored.
- **No offset arithmetic**: `word_ids()` eliminates the need for character offset mapping used in character-level approaches
- **Simple entity reconstruction**: Since words are already split, entity text is just `" ".join(entity_words)` — no gap detection needed
- **Training data portability**: Labels stored as one-integer-per-word (not per-subtoken), so data is human-readable and tokenizer-agnostic

### Decision 4: Synthetic Training Data

We don't have a labeled dataset of real records. Instead we generate synthetic data from templates using real name lists (100 first names, 100 last names), real street names, and Faker-generated cities.

**Data variations** (to train the model on the full expected input space):
- **Person types**: individual (30%), shared last name (20%), separate persons (20%), business payor (30%)
- **Middle names**: abbreviated `"M."` or full `"Monroe"` at ~25% probability, labeled `O` so only the given name is the first_name entity
- **Street types**: regular named streets (60%), ordinal numbered streets like `"5th Ave"` (25%), P.O. Box addresses (15%)
- **Business payors**: business name word is `B-FIRST_NAME`, business type (LLC/Inc/Corp/Ltd/Co/LP/LLP) is `B-LAST_NAME`; optional location prefix word (labeled `O`)

**Why no word jumbling or OCR noise?** OCR noise and word merging are out of scope for this repo. The model is trained on clean text with natural punctuation. If OCR preprocessing is needed for a specific deployment, it can be applied to the input before calling `parse()` — keeping that concern separate and explicit.

**Why separate train/test seeds?** Training data uses `--seed 42`; test data uses `--seed 99`. This ensures the evaluation dataset exercises different random name/address combinations, measuring generalization rather than memorization.

### Decision 5: Portable Training Data Format

Training data stores `text` (raw) and `labels` (one per `text.split()` word). No pre-tokenized `input_ids`:

```json
{
  "text": "John Doe, 500 Oak Ave, Denver CO",
  "labels": [1, 3, 0, 5, 0, 0, 0]
}
```

`labels[i]` corresponds directly to `text.split()[i]`. The data is human-readable and model-agnostic — if you switch to a different tokenizer or model, the data doesn't need to be regenerated.

Tokenization happens on-the-fly in `train.py`, which expands word-level labels to subtoken labels (first subtoken gets the word's label, continuation subtokens get -100 so they're ignored in the loss).

### Decision 6: First-Entity-Only Extraction

Extraction of only the first person/business is achieved at two layers:

1. **Training data**: Only the first first-name and last-name tokens are labeled. All subsequent names are labeled `O`. The model learns to label only the first entity.
2. **Postprocessor**: `postprocess()` takes `entities["FIRST_NAME"][0]` and `entities["LAST_NAME"][0]` — the first detected span. This acts as a safety net if the model occasionally mislabels a secondary name.

### Decision 7: GPU Acceleration for Training, CPU for Inference

Training uses the best available accelerator automatically:
- **MPS** (Metal Performance Shaders) on Apple Silicon (M1/M2/M3)
- **CUDA** on NVIDIA GPUs
- **CPU** as fallback

The HuggingFace Trainer + Accelerate library handles device selection. `fp16=False` is set because neither MPS nor CPU support 16-bit float training.

Production inference uses CPU-only ONNX Runtime, which is appropriate because:
- Inputs are tiny (typically <64 tokens) — GPU parallelism doesn't help at this scale
- Single-threaded ONNX is predictable (~10ms) and avoids overhead from parallelism on short inputs
- ONNX Runtime has no dependency on PyTorch or any GPU libraries at runtime

### Decision 8: ONNX Runtime with Single-Thread Configuration

The inference pipeline uses `intra_op_num_threads=1` and `inter_op_num_threads=1`.

Multi-threaded inference only pays off for large inputs where parallelizing matrix operations is beneficial. For our short inputs (~40-50 tokens), thread coordination overhead outweighs any parallelism gain. Single-threaded gives predictable, consistent ~10ms latency.

In a production server, concurrency is handled at the request level (multiple threads or processes, each running single-threaded ONNX) — not within a single inference call.

### Decision 9: MAX_SEQ_LENGTH = 64

Customer records are short text — a name, an address, and sometimes an email or phone. 64 tokens covers even the longest realistic inputs.

Transformer self-attention is O(n²) in sequence length. Padding to 128 or 512 would waste compute on padding tokens since real inputs rarely exceed ~40-50 tokens.

### Decision 10: Training Hyperparameters

- **Learning rate 5e-5**: Standard for BERT fine-tuning. Too low → slow convergence; too high → destabilizes pre-trained weights
- **15 epochs**: More than typical fine-tuning (3-5 epochs) because the dataset is small (4,000 examples). More passes help with generalization
- **Batch size 16**: Stable gradient estimates while fitting in memory on CPU/MPS
- **Warmup ratio 0.1**: First 10% of steps use linearly increasing LR. Prevents the randomly-initialized classification head from producing large gradients that damage the pre-trained weights
- **Weight decay 0.01**: L2 regularization to reduce overfitting on a small dataset
- **85/15 train/eval split**: 600 eval examples for reliable in-training metrics; the separate test set (seed 99) measures true generalization
- **`load_best_model_at_end=True` with `metric_for_best_model="f1"`**: Saves the checkpoint with highest seqeval F1. Seqeval evaluates at entity level — an entity is correct only if all its B-/I- tokens are predicted correctly, stricter than per-token accuracy

---

## How It All Fits Together

### Training Phase

```
1. generate_training_data.py
       Builds (word, label) token lists from templates
       Joins to raw text: "John Doe, 1234 Braddock Ave, Denver CO"
       Labels aligned with text.split() words
       Variations: individual / shared last / separate / business,
                   abbreviated/full middle names (O),
                   regular / ordinal / P.O. Box streets
       Stores: {text, labels}   ← no preprocessing, human-readable
         |
         v
2. train.py
       Reads text, splits by whitespace → word list
       Tokenizes on-the-fly with is_split_into_words=True
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
       Loads held-out test.json (seed 99, not seen during training)
       Runs full inference pipeline for each example
       Reconstructs expected values from labels (with same punctuation stripping)
       Reports per-field accuracy + sample errors
```

### Inference Phase

```
Input: "Fairfax SushiMax LLC, 1201 5th Ave, Richmond VA"
    |
    v
NameAddressParser.parse()
    |
    |-- 1. Split on whitespace (raw, no preprocessing)
    |      words = ["Fairfax", "SushiMax", "LLC,", "1201", "5th", "Ave,", "Richmond", "VA"]
    |
    |-- 2. Tokenize with is_split_into_words=True
    |      subtokens: [CLS] fairfax sushi ##max llc , 1201 5th ave , richmond va [SEP]
    |      word_ids:  [None  0       1     1     2  2 3    4   5  5 6        7   None]
    |
    |-- 3. ONNX inference (~10ms on CPU)
    |      logits → argmax → label IDs per subtoken
    |
    |-- 4. postprocess()
    |      First-subtoken rule: word 0→O, word 1→B-FIRST_NAME, word 2→B-LAST_NAME,
    |                           word 3→O, word 4→B-STREET_NAME, word 5→O, ...
    |
    |      BIO state machine:
    |        FIRST_NAME:  [["SushiMax"]]
    |        LAST_NAME:   [["LLC,"]]
    |        STREET_NAME: [["5th"]]
    |
    |      _clean_name_word("SushiMax") → "SushiMax"
    |      _clean_name_word("LLC,")     → "LLC"
    |      filter_street_name([["5th"]]):
    |        "5th" matches ordinal pattern → not filtered → "5th"
    |
    v
{"first_name": "SushiMax", "last_name": "LLC", "street_name": "5th"}
```

---

## File-by-File Breakdown

### `pyproject.toml` — Project Configuration

Three dependency groups:

- **Runtime** (always needed): `onnxruntime`, `transformers` (for tokenizer), `numpy`. No PyTorch at runtime.
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

**`GENERIC_STREET_WORDS`**: ~45 words filtered out from STREET_NAME entity spans during post-processing. Includes suffixes (ave, st, blvd, dr, rd, ln, ct, pl, way...), their full forms (avenue, street, boulevard...), and cardinal directions (north, south, n, s, ne, sw...).

---

### `src/name_parsing/model.py` — Inference Pipeline

Contains the `NameAddressParser` class.

**`__init__(model_dir)`**: Finds the `.onnx` file in the model directory, creates a single-threaded `ort.InferenceSession` with all graph optimizations enabled, loads the tokenizer from the same directory.

**`parse(text)`**:
1. Guard clause: returns empty fields for empty input
2. `words = text.split()` — raw whitespace split, no preprocessing
3. `tokenizer(words, is_split_into_words=True, return_tensors="np")` — NumPy arrays are ONNX Runtime's native format, no PyTorch dependency at inference
4. ONNX inference: feeds `input_ids` and `attention_mask`, receives logits shape `(1, seq_len, 7)`
5. Argmax → predicted label IDs per subtoken
6. `postprocess(predictions, words, word_ids)` → final dict

**`parse_batch(texts)`**: Processes texts sequentially. Batching multiple inputs into one ONNX call would improve throughput (not latency) if needed.

---

### `src/name_parsing/postprocessor.py` — Entity Extraction and Filtering

**`_clean_word(word)`**: Strips leading/trailing punctuation (`.,;:!?"'()[]{}/-\`). Applied to all entity words in output.

**`_clean_name_word(word)`**: Extends `_clean_word` with name-specific cleaning — strips trailing possessive `'s` (`"GG's"` → `"GG"`) and a leading single-letter dot-prefix (`"A.Professional"` → `"Professional"`). Used for first_name and last_name only.

**`_is_numeric(word)`**: Returns True if ≥50% of characters are digits AND the word is not an ordinal (`\d+(st|nd|rd|th)`). Used to filter house numbers from street name spans.

**`extract_entities(predictions, words, word_ids)`**:
1. First-subtoken rule: build `word_predictions` dict mapping word_idx → label_id using first occurrence only
2. BIO state machine over words:
   - B- tag: close any open entity, start new span
   - I- tag matching current entity: append word to span
   - I- tag mismatching or O tag: close current span
3. Returns `{"FIRST_NAME": [[words], ...], "LAST_NAME": [...], "STREET_NAME": [...]}`

**`filter_street_name(street_spans)`**: Flatten spans → strip punctuation → filter generic words + numeric tokens + ordinal exemption → return first remaining word, fallback to first word.

**`postprocess(predictions, words, word_ids)`**: Main entry point. Calls `extract_entities()`, takes `[0]` of each entity type (first occurrence = first person/business), applies `_clean_name_word()` to each word in first_name and last_name spans (strips punctuation, possessives, and single-letter dot-prefixes), calls `filter_street_name()` for street_name.

---

### `training/generate_training_data.py` — Synthetic Training Data

Generates `(word, label)` token lists that are joined into a sentence. No preprocessing, no OCR noise.

**Token-building approach**: Each element in the `tokens` list is a `(word_string, label_string)` tuple. The raw text is `" ".join(words)` and labels align with `text.split()`:

```python
tokens.append(("John", "B-FIRST_NAME"))
tokens.append(("Doe,", "B-LAST_NAME"))    # comma attached naturally
tokens.append(("1234", "O"))
tokens.append(("Braddock", "B-STREET_NAME"))
tokens.append(("Ave,", "O"))              # suffix is O; post-processor already filters it
```

**Templates**:
- **single** (30%): `"John Doe, 1234 Braddock Ave, Denver CO"` — one person
- **shared_last** (20%): `"John or Mary Doe, 1234 Braddock Ave"` — two first names, one last
- **separate_names** (20%): `"John Doe or Mary Smith, 1234 Braddock Ave"` — two full names; only first labeled
- **business** (30%): `"Fairfax SushiMax LLC, 1234 Braddock Ave"` — optional prefix (O), business name (B-FIRST_NAME), type (B-LAST_NAME)

**Middle names** (`_maybe_add_middle()`): Added after the first name token at 25% probability for `single`, 20% for `shared_last` and `separate_names`. Randomly either abbreviated (`"M."`) or full (`"Monroe"`). Always labeled `O` — only the given name word is the first_name entity. Business payors never get middle names.

**Street variants**:
- Regular (60%): `"1234 Braddock Ave,"` — street name from `STREET_NAMES` list
- Ordinal (25%): `"1234 5th Ave,"` — from `ORDINAL_STREETS` list (`"1st"` through `"50th"`)
- P.O. Box (15%): `"P.O. Box 1234,"` — only `"Box"` is labeled `B-STREET_NAME`

**City handling**: `fake.city()` may return multi-word names like "San Francisco". Each word is added as a separate `O` token to keep label-word alignment with `text.split()` correct.

**Output format**:
```json
{"text": "John Doe, 1234 Braddock Ave, Denver CO", "labels": [1, 3, 0, 5, 0, 0, 0]}
```

---

### `training/train.py` — Fine-Tuning DistilBERT

**Device detection**: Prints and uses MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU automatically. The HuggingFace Trainer + Accelerate library handles device routing. `fp16=False` since MPS and CPU don't support float16 training.

**`tokenize_and_align_labels(examples, tokenizer)`**: Splits `text` by whitespace to get the word list, tokenizes with `is_split_into_words=True`, then expands word-level labels to subtoken labels. First subtoken → word's label; continuation subtokens and special tokens → `-100` (ignored in loss).

**`load_data(data_path, tokenizer)`**: Reads JSON, applies tokenization via `.map()`, splits 85/15 into train/eval with `seed=42`.

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

**`extract_expected_from_example(ex)`**: Reconstructs expected values from training labels. Splits `text` by whitespace to get words, walks the BIO state machine to find the first span of each entity, applies `_clean_word()` and street filtering (same logic as postprocessor) to get the expected string.

**Comparison**: Case-insensitive. Fields with empty expected value are skipped. Reports per-field accuracy + overall + first 10 sample errors.

---

### `tests/test_postprocessor.py` — 29 Unit Tests

**`TestCleanWord`** (5 tests): `_clean_word()` strips trailing commas, periods; leaves ordinals and clean words unchanged.

**`TestExtractEntities`** (7 tests): BIO predictions → entity spans:
- Raw words with punctuation (`"Doe,"`, `"LLC,"`)
- Single name, shared last name, separate names, business payor
- Multi-subtoken words (first-subtoken rule)
- Empty words list; mismatched I- tag (closes entity, discards word)

**`TestFilterStreetName`** (10 tests): Street filtering:
- Punctuation stripped before filtering (`"Braddock,"` → `"Braddock"`)
- Generic suffix filtered (`"Ave,"` → filtered)
- Ordinals preserved (`"5th"`, `"1st"`, `"22nd"` returned as-is)
- P.O. Box: `"Box"` returned directly
- Numeric tokens filtered; ordinal exemption tested
- All-generic fallback; empty input

**`TestPostprocess`** (7 tests): End-to-end post-processing:
- Punctuation stripped in first/last name output
- Shared last name, business payor, ordinal street, P.O. Box
- Empty input; multi-word first name

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
| Training F1 (seqeval) | 100% |
| Inference latency (p50) | ~10ms |
| Inference latency (p99) | ~10ms |
| Model size (quantized) | 67 MB |
| Model size (FP32 ONNX) | 265 MB |
| Training time (M1 Pro MPS) | ~5 min |
| Training examples | 4,000 synthetic |
| Test examples | 1,000 synthetic (seed 99, held out) |

---

## How to Improve the Model

1. **Add real labeled examples**: The model was trained on synthetic data only. Even 200-500 manually labeled real records would improve accuracy on edge cases the templates don't cover. The format (`text`, `labels` per `text.split()` word) is designed to make manual labeling straightforward.

2. **Expand template coverage**: Add more patterns to `generate_training_data.py` — apartment/suite numbers, international addresses, additional business type formats, records without commas.

3. **Target specific failures**: Run `evaluate.py`, examine the errors, and add training examples that directly address the failing pattern.

4. **Try a larger base model**: Swap `distilbert-base-uncased` for `bert-base-uncased` in `config.py`. This roughly doubles inference time (~20ms) but may gain 1-2% accuracy. Still well under the 100ms budget.

5. **Increase training data volume**: Generate 10,000–20,000 examples instead of 4,000. More data with more random name/city combinations generally improves generalization.

6. **Post-processing sanity checks**: For highest precision, add rule-based checks on model output — e.g., flag last_name values that are common connector words ("and", "or", "the") and return empty instead.

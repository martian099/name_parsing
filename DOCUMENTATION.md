# OCR Customer Record Parser — Technical Documentation

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Key Concepts](#key-concepts)
3. [Architecture and Design Decisions](#architecture-and-design-decisions)
4. [How It All Fits Together](#how-it-all-fits-together)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [Evolution: V1 → V2 → V3](#evolution-v1--v2--v3)
7. [Performance Characteristics](#performance-characteristics)
8. [How to Improve the Model](#how-to-improve-the-model)

---

## What This Project Does

This project takes raw text from OCR-scanned customer records — such as scanned forms, mailing labels, or contact cards — and extracts three pieces of structured data:

```
Input:  "Alex or Mary Doe, 1201 Braddock Ave, Richmond VA, 22312"
Output: {"first_name": "alex", "last_name": "doe", "street_name": "braddock"}
```

The challenge is that this text is messy:
- There can be multiple names ("Alex or Mary Doe", "Alex Doe or Mary Smith")
- The formatting varies wildly (commas might be missing, emails might be appended, phone numbers mixed in)
- OCR sometimes makes character-level mistakes (letters swapped, characters dropped, doubled)
- OCR sometimes **merges adjacent words** — e.g., `"JohnDoe"` (missing space), `"37/harbor"` (space replaced with `/`)

### Constraints

- **No LLM allowed**: Generative large language models are out of scope for this task
- **CPU-only inference**: Must run on CPU (no GPU assumed)
- **Under 100ms latency**: Hard latency budget for each parse call
- **Extract first person only**: When multiple names appear, extract only the first person's first name and last name
- **Most distinctive street word**: Extract the most distinctive part of the street name — "Braddock" from "1201 Braddock Ave", not "Ave" or "1201"

---

## Key Concepts

### What is NER (Named Entity Recognition)?

NER is a task where you label each piece of text with what "type" of thing it is. For example:

```
Alex     or    Mary   Doe    1201   Braddock  Ave   Richmond  VA    22312
FIRST    -     -      LAST   -      STREET    -     -         -     -
```

The model looks at each word and decides: "Is this a first name? A last name? A street name? Or none of the above?"

**Why NER over regex or rule-based parsing?** OCR-scanned customer records have too many format variations for rules to handle reliably. Names can appear in any order, separators vary (commas, spaces, slashes), and OCR errors corrupt the text unpredictably. A learned model generalizes across these variations instead of needing a rule for every edge case.

### What is BIO Tagging?

BIO stands for **B**egin, **I**nside, **O**utside. It's a labeling scheme that tells the model where entities start and continue:

- **B-FIRST_NAME** = this word is the *beginning* of a first name
- **I-FIRST_NAME** = this word *continues* a first name
- **O** = this word is not part of any entity we care about

**Why BIO instead of simple entity labels?** Without B/I distinction, the model couldn't tell where one entity ends and another begins. The B- prefix marks "this is the start of a new entity," while I- means "this continues the previous one."

Our full label set has 7 labels:
```
O, B-FIRST_NAME, I-FIRST_NAME, B-LAST_NAME, I-LAST_NAME, B-STREET_NAME, I-STREET_NAME
```

**Why only these three entity types?** The requirement is to extract first name, last name, and street name. Labeling only what we need keeps the model's task focused. Everything else (city, state, zip, email, phone) is labeled "O" — which actually helps the model understand structure, since text after the street suffix is reliably "O".

### What is DistilBERT?

DistilBERT is a smaller, faster version of BERT — a well-known transformer model. Key facts:

- It is an **encoder** model (reads text and classifies it), NOT a generative/LLM model
- It has 66 million parameters and 6 layers (BERT has 110M and 12 layers)
- It uses **WordPiece tokenization**: words get split into subwords. For example, "Braddock" might become `["brad", "##dock"]`. The `##` prefix means "this is a continuation of the previous token, not a new word"
- We use `distilbert-base-uncased`, which lowercases all input — case information is not needed because the preprocessing step has already split CamelCase tokens before the model sees them

**Why DistilBERT and not BERT, RoBERTa, or a CRF?**
- **vs. BERT**: DistilBERT retains 97% of BERT's performance with 40% fewer parameters and 60% faster inference. Since our input is short (a single address line) and our entity types are few, the full BERT capacity is unnecessary.
- **vs. RoBERTa**: RoBERTa is larger and slower. Our task is narrow enough that DistilBERT's capacity suffices.
- **vs. CRF (Conditional Random Fields)**: CRFs require hand-crafted features and struggle with OCR noise. DistilBERT learns features automatically from text.
- **vs. spaCy NER**: spaCy's built-in NER models are trained for general entities (PERSON, ORG, GPE). Fine-tuning spaCy for custom labels is possible but offers less control over the tokenization pipeline.

### What is the Preprocessing-First Approach?

The core insight of this project's V3 architecture: **split OCR-merged tokens in Python before the model sees them**.

OCR frequently merges adjacent words by dropping the space between them, sometimes inserting junk characters (`/`, `|`, `~`). Examples:
- `"JohnDoe"` — CamelCase merge (no space between words)
- `"37/harbor"` — junk character merge
- `"37harbor"` — digit/letter boundary merge

Instead of asking the model to figure out where boundaries should be, a simple regex preprocessor handles this deterministically:

```python
text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)   # CamelCase → "John Doe"
text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)    # digit→letter → "37 harbor"
text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)    # letter→digit
text = re.sub(r'[^A-Za-z0-9\s]', ' ', text)         # strip all special chars
text = re.sub(r' +', ' ', text).strip()
```

After preprocessing, the text is a clean sequence of alphanumeric words. The model then uses `is_split_into_words=True` tokenization and `word_ids()` for clean word-to-subtoken alignment.

### What is ONNX and Quantization?

- **ONNX** (Open Neural Network Exchange) is a format for saving models that can run without PyTorch. It's faster for inference because the ONNX Runtime engine applies graph-level optimizations (operator fusion, memory planning) that PyTorch's eager execution mode doesn't.
- **Quantization** shrinks the model by converting 32-bit floating point numbers to 8-bit integers. This makes the model ~4x smaller and faster with minimal accuracy loss. We use **dynamic quantization**, meaning weights are stored as INT8 but activations are quantized on-the-fly during inference.

**Why dynamic over static quantization?** Static quantization requires a calibration dataset to determine the quantization ranges for activations. Dynamic quantization requires no calibration — it determines activation ranges at inference time. For our short inputs, the overhead of dynamic range computation is negligible, and it avoids the complexity of maintaining a calibration set.

### What is Fine-Tuning?

DistilBERT was originally trained on general English text (Wikipedia, books). It understands English but doesn't know anything about OCR-scanned customer records. **Fine-tuning** means we take this pre-trained model and train it further on our specific task (labeling OCR text) so it learns our domain.

**Why fine-tuning instead of training from scratch?** Training a transformer from scratch requires millions of examples and days of compute. Fine-tuning leverages the language understanding BERT already has and just teaches it the specific task. This works well even with only 4,000 synthetic examples.

---

## Architecture and Design Decisions

### Decision 1: Preprocessing-First (The V2-to-V3 Pivot)

The most important architectural decision is **where to handle OCR-merged tokens**.

**V2 (character-level, abandoned):** Feed raw text directly to the tokenizer and use `return_offsets_mapping=True` to label each subword token. The model was expected to figure out that "harbor" in the merged token "37/harbor" is a street name.

**V3 (preprocessing-first, current):** Split merged tokens in Python *before* the model, then use word-level tokenization with `is_split_into_words=True` and `word_ids()`.

**Why V2 had a hidden flaw:** Even though the model correctly handled simple special-char merges like "37/harbor" (because WordPiece naturally splits on "/"), CamelCase merges like "JohnDoe" caused a subtle bug. WordPiece tokenizes "JohnDoe" as `["john", "##D", "##oe"]` — the continuation subwords "##D" and "##oe" are fragments, not real name tokens. The first-subtoken rule picks "john" for the first word prediction correctly, but the second name "doe" is represented only by fragments "##D" and "##oe" which the model never saw as standalone names during training.

**Why V3 works:** After `preprocess_ocr_text("JohnDoe")` → `"John Doe"`, the tokenizer sees `["john", "doe"]` as two proper words, each with their own first-subtoken prediction. The model has abundant training signal for "john" (first name) and "doe" (last name) as standalone tokens.

**Additional benefit of V3:** No need for gap-aware token joining logic (the `_join_token_infos` function in V2). Since words are pre-split, multi-word entities like "Silver Lake" are already separated before tokenization, and reconstructing entity text is just `" ".join(words_in_span)`.

### Decision 2: Strip All Punctuation in Preprocessing

V3's preprocessor removes all non-alphanumeric, non-space characters with `re.sub(r'[^A-Za-z0-9\s]', ' ', text)`.

**Why not selectively handle specific punctuation?** Earlier versions tried to only detach commas (e.g., `"Doe,"` → `"Doe ,"`). This was fragile:
- Emails like `"john.smith@gmail.com"` would break: `"john .smith@gmail .com"`
- Any new punctuation type required a new rule
- The set of OCR junk characters is open-ended

**Why stripping is safe for this task:** The entity values we care about (first names, last names, street names) are all pure alphanumeric text. Punctuation is noise for our task — commas separate fields, periods appear in email addresses or abbreviations, slashes/pipes are OCR merge artifacts. Removing it all uniformly is the simplest and most robust approach.

**Why do CamelCase splits happen before stripping?** The CamelCase split rule relies on letter case: `([a-z])([A-Z])`. If we stripped special characters first, an input like `"John.Doe"` would become `"John Doe"` (correct) directly from the strip. But `"JohnDoe"` has no special characters to strip — the CamelCase rule is still needed for those. The order is: split CamelCase → split digit/letter boundaries → strip everything else → collapse spaces.

### Decision 3: Word-Level NER with `is_split_into_words=True`

With the preprocessing step guaranteeing clean word boundaries, word-level tokenization is strictly better than character-level:

- **Cleaner alignment**: `word_ids()` gives a direct word index for each subtoken, no offset arithmetic needed
- **Simpler label assignment**: One label per word during training; first-subtoken rule maps it to subtokens
- **Simpler postprocessing**: No gap detection, no subword joining — just `" ".join(entity_words)`
- **Training data portability**: Labels stored as one-integer-per-word, not one-integer-per-subtoken, so the data is tokenizer-agnostic and easy to inspect/edit manually

### Decision 4: Portable Training Data Format

Training data stores only `text`, `preprocessed`, and `labels` — NOT `input_ids` or `attention_mask`.

```json
{
  "text": "JohnDoe, 500 Oak Ave, Denver CO",
  "preprocessed": "John Doe 500 Oak Ave Denver CO",
  "labels": [1, 3, 0, 5, 0, 0, 0, 0]
}
```

**Why remove pre-tokenized IDs?** Earlier versions pre-tokenized everything into `input_ids` during data generation. This tightly coupled the data to a specific tokenizer and model. If you switch models, all data must be regenerated. Worse, manually labeled examples become difficult to create — you'd have to run them through the tokenizer just to add them.

With the new format, `train.py` tokenizes on-the-fly at training time. The data file is human-readable: `labels[i]` corresponds directly to `preprocessed.split()[i]`. You can open `train.json`, find an example, and add your own labeled records without any tooling.

### Decision 5: Separate Train and Test Sets

Training data is generated with `--seed 42`; the test set uses `--seed 99`. The evaluate script defaults to `data/raw/test.json`.

**Why this matters:** If you evaluate on the training distribution (same seed), you measure how well the model memorizes your templates, not how well it generalizes. Different seeds produce different random name/address combinations, so the test set exercises the model on examples it hasn't seen.

### Decision 6: Synthetic Training Data with OCR Noise

We don't have a labeled dataset of real OCR-scanned records. Instead, we generate synthetic data by:

1. Building realistic text from templates using name lists, street name lists, and Faker-generated cities
2. Tracking exact character positions of each entity during construction
3. Injecting OCR noise that simulates real scanner errors

**Why synthetic data works here:** Customer records follow a limited set of patterns (name(s) + address). The variation is in the names, addresses, and formatting — all of which we can generate. The hardest part (OCR noise) is simulated directly.

**Why track character positions during noise injection?** OCR noise changes character counts. If we add a character (doubling "l" → "ll"), everything after it shifts by 1. By tracking an `old_to_new` position mapping during noise injection, entity spans remain accurate after noise is applied.

### Decision 7: First-Entity-Only Extraction

The requirement is to extract only the first person's name. The system achieves this through two layers:

1. **Training data labels only the first person**: In all templates, only the first first-name and last-name are labeled. Additional names are labeled "O". The model learns to label only the first person's name as entities.
2. **Postprocessor takes the first occurrence**: `postprocess()` takes `entities["FIRST_NAME"][0]` — the first span it finds. This acts as a safety net even if the model occasionally labels a second name.

### Decision 8: Street Name Filtering (Generic Word Removal)

Street addresses contain generic words like "Ave", "St", "Dr", "North" that aren't distinctive. The postprocessor filters these out because the requirement is the **most distinctive** part of the street name.

The filtering pipeline:
1. Collect all words from STREET_NAME entity spans
2. Filter out words in `GENERIC_STREET_WORDS` (45 common street suffixes and directions)
3. Filter out numeric tokens (>=50% digit characters) — catches street numbers even after OCR mangling
4. Return the first remaining word; if all words were filtered, return the first word anyway

**Why the >=50% digit heuristic instead of `isdigit()`?** OCR can mangle street numbers — "1201" might become "12O1" or "l201". A strict `isdigit()` check would miss these. The 50% digit threshold catches OCR-corrupted numbers while not accidentally filtering words like "1st".

**Why a hardcoded set instead of a learned filter?** The list of generic street words is finite and well-known. A learned approach would add complexity without benefit — the model already labels both "Braddock" and "Ave" as STREET_NAME (correctly), and filtering is better done as a deterministic postprocessing step.

### Decision 9: ONNX Runtime with Single-Thread Configuration

The inference pipeline uses ONNX Runtime with `intra_op_num_threads=1` and `inter_op_num_threads=1`.

**Why single-threaded?** Multi-threaded inference adds parallelism overhead (thread synchronization, cache contention) that only pays off for large inputs. Our inputs are tiny (typically <64 tokens). Single-threaded execution eliminates this overhead and gives more predictable, consistent latency (~10ms p99). In a production server handling many concurrent requests, each request should be single-threaded while the server itself handles concurrency at the request level.

### Decision 10: MAX_SEQ_LENGTH = 64

OCR-scanned customer records are short text — typically a name, an address, and maybe an email or phone number. 64 tokens is enough to cover even the longest realistic inputs while keeping inference fast.

**Why not 128 or 512 (BERT's max)?** Transformer self-attention is O(n²) in sequence length. Since our inputs rarely exceed ~40-50 tokens in practice, padding to 128 or 512 would waste compute on padding tokens.

### Decision 11: Training Hyperparameters

- **Learning rate 5e-5**: Standard for BERT fine-tuning. Lower rates underfit; higher rates can destabilize training.
- **15 epochs**: More epochs than typical BERT fine-tuning (usually 3-5) because our dataset is small (4,000 examples). The model needs more passes to learn the patterns.
- **Batch size 16**: Small enough to fit in memory, large enough for stable gradient estimates.
- **Warmup ratio 0.1**: The first 10% of training steps use a linearly increasing learning rate. This prevents the randomly-initialized classification head from producing large gradients that corrupt the pre-trained weights.
- **Weight decay 0.01**: L2 regularization to prevent overfitting on our small synthetic dataset.
- **85/15 train/eval split**: 600 examples for in-training evaluation is enough for reliable metrics. The held-out test set (separate seed) provides the true accuracy estimate.
- **`load_best_model_at_end=True` with `metric_for_best_model="f1"`**: Saves the model with the highest seqeval F1 across all epochs. Seqeval evaluates at the entity level — an entity is correct only if all its B- and I- tokens are predicted correctly. This is stricter and more meaningful than per-token accuracy.

---

## How It All Fits Together

The project has two phases: **Training** (done once or when you want to retrain) and **Inference** (done repeatedly in production).

### Training Phase

```
1. generate_training_data.py    Creates 4000 synthetic labeled examples
         |                      (with OCR noise including word merging)
         |                      Uses TextBuilder for character-span tracking
         |                      Applies preprocess_ocr_text() to each example
         |                      Stores: text, preprocessed, labels (word-level)
         v
2. train.py                     Fine-tunes DistilBERT on those examples
         |                      Tokenizes on-the-fly with is_split_into_words=True
         |                      Expands word-level labels to subtoken labels
         |                      85% train / 15% eval split; saves best by F1
         v
3. export_onnx.py               Converts PyTorch model to ONNX format
         |                      Applies INT8 dynamic quantization
         |                      265 MB → 67 MB
         v
4. evaluate.py                  Runs full inference pipeline on held-out test set
                                Reports per-field accuracy + sample errors
```

### Inference Phase

```
Raw OCR text (e.g., "JohnDoe, 1201 BraddockAve, Richmond VA")
    |
    v
model.py: NameAddressParser.parse()
    |
    |-- 1. preprocess_ocr_text()
    |      Input:  "JohnDoe, 1201 BraddockAve, Richmond VA"
    |      Output: "John Doe 1201 Braddock Ave Richmond VA"
    |              (CamelCase split, punctuation stripped)
    |
    |-- 2. Split into word list and tokenize with is_split_into_words=True
    |      words    = ["John", "Doe", "1201", "Braddock", "Ave", "Richmond", "VA"]
    |      subtokens = ["[CLS]", "john", "doe", "1201", "brad", "##dock", "ave", "richmond", "va", "[SEP]"]
    |      word_ids  = [None,   0,      1,     2,      3,      3,         4,     5,           6,    None]
    |
    |-- 3. Run ONNX inference (CPU, ~10ms)
    |      Argmax predictions (per subtoken):
    |      [-, B-FIRST, B-LAST, O, B-STREET, I-STREET, O, O, O, -]
    |
    |-- 4. Post-process (postprocessor.py)
    |      First-subtoken rule: word 3 ("Braddock") = B-STREET, word 3 cont. ignored
    |      Entity spans (word indices):
    |        FIRST_NAME: [0]       → words[0]   = "John"
    |        LAST_NAME:  [1]       → words[1]   = "Doe"
    |        STREET_NAME:[3, 4]    → words[3:5] = ["Braddock", "Ave"]
    |      filter_street_name(["Braddock", "Ave"]):
    |        "Ave" is in GENERIC_STREET_WORDS → filtered out → "Braddock"
    |
    v
{"first_name": "john", "last_name": "doe", "street_name": "braddock"}
```

---

## File-by-File Breakdown

### `pyproject.toml` — Project Configuration

Defines the project's package metadata and three dependency groups:

- **Runtime** (always needed): `onnxruntime` (runs the ONNX model), `transformers` (provides the tokenizer), `numpy`. These are the only dependencies needed in production.
- **Training** (`pip install -e ".[train]"`): adds `torch`, `datasets`, `optimum[onnxruntime]`, `faker`, `seqeval`, `accelerate`.
- **Dev** (`pip install -e ".[dev]"`): adds `pytest` and `pytest-benchmark` for testing.

**Design note:** The runtime dependencies are intentionally minimal. In production, you don't need PyTorch (200+ MB) — only `onnxruntime` (~30 MB), `transformers` (for the tokenizer), and `numpy`.

Uses `build-backend = "setuptools.build_meta"` with a src-layout package structure, which prevents accidental imports of the source directory during development.

---

### `src/name_parsing/__init__.py` — Package Entry Point

Exports only `NameAddressParser` — the single class users need to interact with.

---

### `src/name_parsing/config.py` — Central Configuration

Single source of truth for all constants used across training and inference.

**Label definitions:**
```python
LABEL_LIST = ["O", "B-FIRST_NAME", "I-FIRST_NAME", "B-LAST_NAME", "I-LAST_NAME", "B-STREET_NAME", "I-STREET_NAME"]
LABEL2ID = {"O": 0, "B-FIRST_NAME": 1, ...}  # label string → integer
ID2LABEL = {0: "O", 1: "B-FIRST_NAME", ...}  # integer → label string
```

The model outputs numbers (0-6), not strings. These dictionaries convert back and forth. "O" must be index 0 because it's the default label, and -100 is used for special tokens that should be ignored during loss computation (a PyTorch convention for `CrossEntropyLoss`).

**Model paths:** `PROJECT_ROOT` is computed relative to the config file's location, making paths work regardless of the working directory.

**Training hyperparameters:** `MAX_SEQ_LENGTH = 64`, `TRAIN_EPOCHS = 15`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 16`, `TRAIN_TEST_SPLIT = 0.15`. See [Decision 11](#decision-11-training-hyperparameters) for rationale.

**`GENERIC_STREET_WORDS`**: A set of ~45 words that appear in street addresses but aren't the distinctive part of the street name. Includes suffixes (ave, st, blvd, dr...), their abbreviations, and cardinal directions.

---

### `src/name_parsing/model.py` — Inference Pipeline

The main file used in production. Contains `preprocess_ocr_text()` and the `NameAddressParser` class.

**`preprocess_ocr_text(text)`** — The core preprocessing function. Must stay identical to the copy in `generate_training_data.py`. Steps:
1. CamelCase split: `re.sub(r'([a-z])([A-Z])', r'\1 \2', text)`
2. Digit→letter split: `re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)`
3. Letter→digit split: `re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)`
4. Strip all punctuation/special chars: `re.sub(r'[^A-Za-z0-9\s]', ' ', text)`
5. Collapse whitespace

**`NameAddressParser.__init__(model_dir)`**:
- Finds the `.onnx` file in the model directory (flexible naming, robust to ONNX export toolchain naming changes)
- Creates an `ort.InferenceSession` with CPU-optimized settings: single-threaded, all graph optimizations enabled
- Loads the tokenizer from the same directory

**`parse(text)`** — The main method:
1. **Guard clause**: Returns empty fields for empty/whitespace input
2. **Preprocess**: Calls `preprocess_ocr_text(text)` and splits into word list
3. **Tokenize**: Calls `self.tokenizer(words, is_split_into_words=True, ..., return_tensors="np")`. `return_tensors="np"` gives NumPy arrays directly (ONNX Runtime's native format), avoiding a PyTorch dependency at inference time.
4. **ONNX inference**: Feeds `input_ids` and `attention_mask` to the ONNX session
5. **Argmax**: Converts raw logits (7 scores per token) into predicted label IDs
6. **Postprocess**: Passes predictions, words, and `word_ids()` to `postprocess()`

**`parse_batch(texts)`**: Processes texts sequentially. A future optimization could batch multiple inputs into a single ONNX call for throughput (not latency) improvement.

---

### `src/name_parsing/postprocessor.py` — Entity Extraction and Filtering

Converts raw model predictions into the final JSON output.

**`extract_entities(predictions, words, word_ids)`**:

1. **First-subtoken rule**: For each position in `word_ids`, take the prediction of the *first* subtoken seen for each word index. Continuation subtokens (same word_idx as a previous position) are ignored.
2. **BIO state machine**: Walk through word predictions in order, grouping consecutive B-/I- tags into entity spans.

Key behaviors:
- **B- tag**: Start a new entity span (closes any previous one)
- **I- tag matching current entity**: Append word to current span
- **I- tag mismatching current entity**: Close current span, discard the mismatched word (treat as noise)
- **O tag**: Close any open entity

**`filter_street_name(street_spans)`**:

1. Flatten all spans into a single word list
2. Filter out generic street words (`GENERIC_STREET_WORDS` set)
3. Filter out numeric tokens (>=50% digits)
4. Return the first remaining word; falls back to the first word if all were filtered

**`postprocess(predictions, words, word_ids)`** — The main entry point:

1. Calls `extract_entities()` to get all entity spans
2. Takes `[0]` of each entity type (first occurrence = first person)
3. Joins words within each entity with `" ".join(...)` — no gap detection needed since words were pre-split
4. Calls `filter_street_name()` for street name
5. Returns `{"first_name": ..., "last_name": ..., "street_name": ...}`

---

### `training/generate_training_data.py` — Synthetic Training Data

Since we don't have a pre-existing labeled dataset, this script generates one with known entity positions.

**`TextBuilder` class** — Builds text piece by piece while recording the exact character range of each entity:

```python
tb = TextBuilder()
tb.add("Alex", "FIRST_NAME")  # records span {"start": 0, "end": 4, "label": "FIRST_NAME"}
tb.add_space()
tb.add("Doe", "LAST_NAME")    # records span {"start": 5, "end": 8, "label": "LAST_NAME"}
```

**Why a builder pattern?** String concatenation doesn't track positions. With TextBuilder, we always know exactly where each entity is in the output string, even as the string grows. Critical because OCR noise shifts these positions.

**Templates** — Four patterns with weighted random selection:
- `"Alex Doe, 1201 Braddock Ave, ..."` (single name, 35%)
- `"Alex or Mary Doe, ..."` (shared last name, 30%)
- `"Alex Doe or Mary Smith, ..."` (separate full names, 25%)
- `"Alex Doe, Mary Smith, John Brown, ..."` (multiple names, 10%)

Each template labels only the first person's first name and last name. Additional names are added as unlabeled text ("O"). The street name is always labeled.

**`inject_ocr_noise_with_spans()`** — Simulates OCR errors while keeping entity spans aligned.

Builds an `old_to_new` character position mapping as it processes each character:
- **Character substitution** (60% of errors): Uses `OCR_CONFUSIONS` dictionary (e.g., `l → 1`, `O → 0`, `rn → m`)
- **Character dropping** (20%): A character is removed
- **Character doubling** (20%): A character is repeated
- **Word merging** (controlled by `merge_rate`, separate from character errors): Spaces are randomly removed, optionally replaced with junk characters (`/`, `|`, `-`, `.`, `~`, `\`)

**Label assignment** — After noise injection, `preprocess_ocr_text()` is applied and the text is split into words. For each entity span, the entity's substring is also preprocessed and split into words, then those words are located in the word list by consecutive case-insensitive match. The matching word range is assigned B-/I- labels.

**Output format**: JSON with `text` (noisy OCR text), `preprocessed` (after `preprocess_ocr_text()`), and `labels` (one integer per word in `preprocessed`). No pre-tokenized IDs — labels are model-agnostic.

**Data mix**: 30% of examples are clean (no noise), 70% have noise with randomly varying intensity.

---

### `training/train.py` — Fine-Tuning DistilBERT

**`tokenize_and_align_labels(examples, tokenizer)`**: Tokenizes word lists from `preprocessed` field using `is_split_into_words=True`, then expands word-level labels to subtoken labels. The first subtoken of each word gets the word's label; all continuation subtokens get -100 (ignored in loss).

**`load_data(data_path, tokenizer)`**: Reads the JSON file, applies `tokenize_and_align_labels` via `.map()`, then splits 85/15 into train/eval with `seed=42`.

**Model setup**: Loads `distilbert-base-uncased` from HuggingFace with a fresh `TokenClassification` head — a single linear layer mapping 768-dim hidden states to 7 label logits. The base model weights are pre-trained; only the classification head starts random.

**`DataCollatorForTokenClassification`**: Handles dynamic padding. Since examples have variable lengths, the collator pads each batch to the length of the longest example in that batch — more efficient than padding everything to `MAX_SEQ_LENGTH`. Also pads the `labels` tensor with -100 so padded positions don't contribute to the loss.

**`compute_metrics()`**: Uses `seqeval` for entity-level F1. An entity is correct only if all its tokens (B- and I-) are predicted correctly — stricter than per-token accuracy.

---

### `training/export_onnx.py` — ONNX Export and Quantization

Converts the fine-tuned PyTorch model into a production-ready ONNX format in two steps.

**Step 1 — ONNX Export**: Uses `optimum`'s `ORTModelForTokenClassification.from_pretrained(..., export=True)` to trace the PyTorch model and produce an ONNX graph.

**Step 2 — INT8 Dynamic Quantization**: Uses `ORTQuantizer` with `AutoQuantizationConfig.avx2(is_static=False, per_channel=False)`. The `avx2` config generates an ONNX model optimized for x86 CPUs with AVX2 support (most modern Intel/AMD processors).

**Why copy the tokenizer to the quantized directory?** The ONNX model needs a co-located tokenizer for inference. By copying the tokenizer alongside the quantized model, the quantized directory is self-contained for deployment.

**Size reduction**: 265 MB (FP32 ONNX) → 67 MB (INT8 quantized), a 75% reduction.

---

### `training/evaluate.py` — Model Evaluation

Measures **end-to-end accuracy** by running the full inference pipeline (model + postprocessor) against held-out test data.

**`extract_expected_from_example(ex)`**: Reads the word list from `preprocessed` and the word-level `labels` to reconstruct expected entity values. BIO state machine finds the first span of each entity type, joins the words, and applies street name filtering.

**Why evaluate end-to-end instead of just model accuracy?** The model's token-level F1 (measured by seqeval during training) can be high even if the postprocessor introduces errors. End-to-end evaluation catches bugs in the postprocessor and filtering logic.

Compares predicted vs expected values case-insensitively and reports per-field accuracy plus sample errors.

---

### `tests/test_postprocessor.py` — Unit Tests for Post-Processing

20 tests organized into four test classes:

**`TestExtractEntities`** (7 tests): BIO-tagged word predictions grouped into entity spans:
- Single names, shared last names, separate names
- Multi-subtoken words (the first-subtoken rule)
- Edge case: only special tokens, mismatched I- tags

**`TestFilterStreetName`** (8 tests): Street name filtering:
- Single distinct word, multi-word spans
- Generic word filtering, numeric filtering, OCR-mangled number filtering
- Empty input, all-generic fallback, multiple spans

**`TestPostprocess`** (5 tests): End-to-end postprocessor:
- Full pipeline, empty input
- OCR-merge split by preprocessor (`"37/harbor"` → `"harbor"`)
- CamelCase split (`"JohnDoe"` → `"john"`, `"doe"`)
- Multi-word first name with space preserved

Helper functions `_make_word_ids(words, subtokens_per_word)` and `_make_predictions(words, labels, subtokens_per_word)` build realistic mock data without requiring a real tokenizer.

---

### `tests/test_inference.py` — Integration Tests and Benchmarking

12 tests that require the trained ONNX model (skip if not present):

**`TestInference`** (11 tests): Full parse pipeline on representative inputs:
- Single name, shared last name, separate names
- Text with email, middle initial
- Returns all expected keys; empty input handling
- OCR-merged CamelCase name: `"JohnDoe"` → first_name="john", last_name="doe"
- OCR-merged name + special-char street: `"MaryDoe, 37/harbor way"`
- Special-char merge only: `"37/harbor"`
- Digit-letter merge: `"37harbor"`

**`TestBenchmark`** (1 test): Measures inference latency over 100 runs after 5 warmup runs. Asserts p99 < 100ms. Currently achieves ~10ms p99.

**Why warmup runs?** The first few calls are slower due to ONNX Runtime JIT compilation, tokenizer loading, and CPU cache warming. Warmup ensures the benchmark measures steady-state performance.

---

## Evolution: V1 → V2 → V3

### V1: Word-Level Tokenization (is_split_into_words, abandoned)

V1 used `is_split_into_words=True` on the *raw* text — split by whitespace first, then tokenize each word.

**V1's fatal flaw**: When OCR merged "37 Harbor" into "37/harbor", V1 treated it as a single "word" because it split on whitespace. The model could only assign the entire merged token one label — it had no way to say "the '37' part is O but 'harbor' is STREET_NAME."

### V2: Character-Level Tokenization (return_offsets_mapping, abandoned)

V2 fed raw text directly to the tokenizer, letting WordPiece split on learned subword boundaries. Each subword got its own label prediction, and `return_offsets_mapping=True` provided character positions for gap-aware joining.

**V2's improvement**: "37/harbor" → `["37", "/", "harbor"]` — each subword predicted independently. This worked for simple special-character merges.

**V2's subtle flaw**: CamelCase merges like "JohnDoe" → `["john", "##D", "##oe"]`. The `##D` and `##oe` are continuation subtokens, not standalone tokens. When the first-subtoken rule assigned the label of "##D" to represent "Doe", the model was being asked to predict a name label from a lowercase-"d" fragment, which it had no training signal for. This caused "JohnDoe" to be extracted as first_name="John", last_name="D" (or empty) — a real user-reported bug.

V2 also required `_join_token_infos()` — a gap-aware subword joining function using character offsets. This was complex and fragile, and became entirely unnecessary in V3.

### V3: Preprocessing-First (current)

V3 splits merged tokens in Python *before* tokenization using `preprocess_ocr_text()`, then uses `is_split_into_words=True` on the cleaned word list.

**V3 fixes all the V2 issues:**
- "JohnDoe" → "John Doe" → `["john"]`, `["doe"]` — clean word tokens with full training signal
- No gap-aware joining needed — words are pre-split, `" ".join()` is correct
- No `return_offsets_mapping=True` — simpler tokenizer call
- Training data is tokenizer-agnostic (word-level labels, not subtoken-level)
- Preprocessing is deterministic and debuggable — you can inspect `ex["preprocessed"]`

**V3 results**: 99.75% training F1, 98.7% end-to-end accuracy on the held-out test set.

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Training F1 (seqeval) | 99.75% |
| first_name accuracy | 100.0% (500/500) |
| last_name accuracy | 99.0% (495/500) |
| street_name accuracy | 97.2% (485/499) |
| Overall accuracy | 98.7% (1480/1499) |
| Inference latency (p50) | ~10ms |
| Inference latency (p99) | ~10ms |
| Model size (quantized) | 67 MB |
| Model size (FP32 ONNX) | 265 MB |
| Training time | ~6 minutes (CPU) |
| Training examples | 4,000 synthetic |
| Test examples | 1,000 synthetic (separate seed) |

**Remaining error patterns** in the test set:
- OCR noise corrupts the entity itself — e.g., `"ond"` (from `"and"`) between two names gets labeled as a last name because the corrupted connector looks like a word
- Digit-in-word substitution — `"High1and"` → `"High 1 and"` where the model correctly labels `"High"` as STREET_NAME but not `"and"` (which doesn't look like a street name fragment after splitting)

---

## How to Improve the Model

If accuracy isn't sufficient for your production needs, here are the most effective improvements in order of impact:

1. **Add real labeled examples**: The model was trained on synthetic data only. Even 200-500 real OCR-scanned examples (manually labeled) would significantly improve accuracy, especially for edge cases the templates don't cover. The data format (`text`, `preprocessed`, `labels`) is designed to make this easy — just add records to `train.json` and retrain.

2. **Increase training data variety**: Add more templates to `generate_training_data.py` — business names, apartment/suite numbers, PO boxes, international addresses, multi-line OCR where lines are concatenated.

3. **Target specific failure modes**: Run `evaluate.py` and examine the errors. If the model consistently confuses certain patterns, add more training examples that emphasize the difference.

4. **Adjust OCR noise parameters**: If your real OCR has a different error profile than the synthetic noise, tune the `OCR_CONFUSIONS` dictionary, `error_rate`, and `merge_rate` to match your actual OCR output.

5. **Try a larger model**: Swap `distilbert-base-uncased` for `bert-base-uncased` in `config.py`. This will roughly double inference time (~20ms) but may improve accuracy 1-2%. Still well under the 100ms budget.

6. **Increase training data volume**: Generate 10,000–20,000 examples instead of 4,000. More data generally helps, especially for the noisy examples where the model needs to learn many different corruption patterns.

7. **Ensemble with rules**: For the highest accuracy, combine the model's predictions with simple pattern-matching rules as a post-processing sanity check. For example, flag predictions where the last_name field contains common English connector words ("and", "or", "the") and return empty instead.

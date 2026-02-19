# OCR Customer Record Parser — Technical Documentation

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Key Concepts](#key-concepts)
3. [Architecture and Design Decisions](#architecture-and-design-decisions)
4. [How It All Fits Together](#how-it-all-fits-together)
5. [File-by-File Breakdown](#file-by-file-breakdown)
6. [The V1 to V2 Evolution: Lessons Learned](#the-v1-to-v2-evolution-lessons-learned)
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
- OCR sometimes **merges adjacent words** (e.g., "37/harbor" instead of "37 Harbor")
- Multi-word street names like "Silver Lake" need to be reconstructed correctly, not concatenated into "SilverLake"

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

The model looks at each piece of text and decides: "Is this a first name? A last name? A street name? Or none of the above?"

**Why NER over regex or rule-based parsing?** OCR-scanned customer records have too many format variations for rules to handle reliably. Names can appear in any order, separators vary (commas, spaces, slashes), and OCR errors corrupt the text unpredictably. A learned model generalizes across these variations instead of needing a rule for every edge case.

### What is BIO Tagging?

BIO stands for **B**egin, **I**nside, **O**utside. It's a labeling scheme that tells the model where entities start and continue:

- **B-FIRST_NAME** = this token is the *beginning* of a first name
- **I-FIRST_NAME** = this token *continues* a first name
- **O** = this token is not part of any entity we care about

**Why BIO instead of simple entity labels?** Without B/I distinction, the model couldn't tell where one entity ends and another begins. If "Alex Doe" were both labeled FIRST_NAME/LAST_NAME without B/I, the model wouldn't know that "Alex" is a complete first name versus the start of a multi-word first name like "Mary Jane". The B- prefix marks "this is the start of a new entity," while I- means "this continues the previous one."

Our full label set has 7 labels:
```
O, B-FIRST_NAME, I-FIRST_NAME, B-LAST_NAME, I-LAST_NAME, B-STREET_NAME, I-STREET_NAME
```

**Why only these three entity types?** The requirement is to extract first name, last name, and street name. We label only what we need — adding labels for city, state, zip, etc. would increase the model's task difficulty without benefit. The model learns to recognize these unlabeled fields implicitly as "O" (outside), which actually helps it understand the structure of the text (e.g., text after the street suffix is probably city/state, so don't label it as a name).

### What is DistilBERT?

DistilBERT is a smaller, faster version of BERT — a well-known transformer model. Key facts:

- It is an **encoder** model (reads text and classifies it), NOT a generative/LLM model (it doesn't generate text)
- It has 66 million parameters and 6 layers (BERT has 110M and 12 layers)
- It uses **WordPiece tokenization**: words get split into subwords. For example, "Braddock" might become ["brad", "##dock"]. The ## prefix means "this is a continuation of the previous token, not a new word."

**Why DistilBERT and not BERT, RoBERTa, or a CRF?**
- **vs. BERT**: DistilBERT retains 97% of BERT's performance with 40% fewer parameters and 60% faster inference. Since our input is short (a single address line) and our entity types are few, the full BERT capacity is unnecessary.
- **vs. RoBERTa**: RoBERTa is larger and slower. Our task is narrow enough that DistilBERT's capacity suffices.
- **vs. CRF (Conditional Random Fields)**: CRFs require hand-crafted features and struggle with OCR noise. DistilBERT learns features automatically from text and handles misspellings gracefully because WordPiece tokenization breaks words into subwords that overlap with correct spellings — "Brddock" still shares the "br" subword with "Braddock."
- **vs. spaCy NER**: spaCy's built-in NER models are trained for general entities (PERSON, ORG, GPE). Fine-tuning spaCy for custom labels is possible but offers less control over tokenization, which is critical for our OCR-merged-token problem.

### What is WordPiece Tokenization and Why It Matters?

WordPiece is a subword tokenization algorithm. Instead of treating each word as an indivisible unit, it breaks words into commonly-seen pieces:

```
"Braddock"  → ["brad", "##dock"]
"unfamiliar" → ["un", "##fa", "##mil", "##iar"]
"37/harbor"  → ["37", "/", "harbor"]
```

**Why this matters for OCR**: When OCR produces "Brddock" (missing the 'a'), WordPiece might produce ["br", "##dd", "##ock"]. Even though the spelling is wrong, the model still sees "br" and "ock" — fragments it recognizes from training. This gives the model robustness to character-level corruption that word-level approaches (which would see "Brddock" as a completely unknown word) lack.

**Why this matters for merged tokens**: When OCR merges "37 Harbor" into "37/harbor" (removing the space, adding junk), WordPiece doesn't care — it splits based on learned subword boundaries, not whitespace. Each resulting subword gets its own prediction, so the model can label "harbor" as STREET_NAME while labeling "37" and "/" as O.

### What is ONNX and Quantization?

- **ONNX** (Open Neural Network Exchange) is a format for saving models that can run without PyTorch. It's faster for inference because the ONNX Runtime engine applies graph-level optimizations (operator fusion, memory planning) that PyTorch's eager execution mode doesn't.
- **Quantization** shrinks the model by converting 32-bit floating point numbers to 8-bit integers. This makes the model ~4x smaller and faster with minimal accuracy loss. We use **dynamic quantization**, meaning weights are stored as INT8 but activations are quantized on-the-fly during inference.

**Why dynamic over static quantization?** Static quantization requires a calibration dataset to determine the quantization ranges for activations. Dynamic quantization requires no calibration — it determines activation ranges at inference time. For our short inputs, the overhead of dynamic range computation is negligible, and it avoids the complexity of maintaining a calibration set.

### What is Fine-Tuning?

DistilBERT was originally trained on general English text (Wikipedia, books). It understands English but doesn't know anything about OCR-scanned customer records. **Fine-tuning** means we take this pre-trained model and train it further on our specific task (labeling OCR text) so it learns our domain.

**Why fine-tuning instead of training from scratch?** Training a transformer from scratch requires millions of examples and days of compute. Fine-tuning leverages the language understanding BERT already has (grammar, word meanings, common patterns) and just teaches it the specific task of labeling customer record fields. This works well even with only 4,000 synthetic examples.

---

## Architecture and Design Decisions

### Decision 1: Character-Level Tokenization (The V1-to-V2 Pivot)

The most important architectural decision in this project is **how we feed text to the tokenizer**.

**V1 (word-level, abandoned):** Split input by whitespace first, then tokenized each word individually using `is_split_into_words=True`. Each word was treated as a pre-defined unit.

**V2 (character-level, current):** Feed the raw text string directly to the tokenizer, let WordPiece split it into subwords freely, then use `return_offsets_mapping=True` to know which characters each subword covers.

**Why V1 was fundamentally broken:** When OCR merged "37 Harbor" into "37/harbor", V1 saw this as a single "word" because it split on whitespace. The model could only assign that entire merged token one label. It had no way to say "the '37' part is O but the 'harbor' part is STREET_NAME."

**Why V2 works:** V2 lets the tokenizer split "37/harbor" into subwords ["37", "/", "harbor"] regardless of whitespace. Each subword gets its own label prediction. The model learns from training data (which includes synthetic merged-word examples) that "harbor" in the middle of a merged token can still be a STREET_NAME.

### Decision 2: Gap-Aware Token Joining

When the model labels a multi-word entity like "Silver Lake" as `B-STREET_NAME, I-STREET_NAME`, the postprocessor needs to reconstruct the text. A naive `"".join()` would produce "silverlake".

**The solution:** Each extracted token carries its character offsets `(start, end)` from the original text. When joining tokens, we check: does the next token's `start` position come after the previous token's `end` position? If yes, there was a gap (a space or punctuation) in the original text, so we insert a space. If they're adjacent (like subwords "brad" at positions 0-4 and "dock" at 4-8), they get concatenated directly.

```
"silver" (pos 0-6) + "lake" (pos 7-11)  → gap at 6-7 → "silver lake"
"brad" (pos 0-4)   + "dock" (pos 4-8)   → no gap    → "braddock"
```

**Why not just use the ## prefix to decide?** The ## prefix indicates a WordPiece continuation within a single word. But the model might also label two separate words as the same entity (B- then I-), and those separate words won't have ## prefixes. We need the actual character positions to know if a space existed between them. The ## prefix is also used to strip the prefix from the token text itself, but it's not a reliable space indicator.

### Decision 3: Synthetic Training Data with OCR Noise

We don't have a labeled dataset of real OCR-scanned customer records. Instead, we generate synthetic data by:

1. Building realistic text from templates using name lists, street name lists, and Faker-generated cities
2. Tracking exact character positions of each entity during construction
3. Injecting OCR noise that simulates real scanner errors

**Why synthetic data works here:** Customer records follow a limited set of patterns (name(s) + address). The variation is in the names, addresses, and formatting — all of which we can generate. The hardest part (OCR noise) is simulated directly.

**Why track character positions during noise injection?** OCR noise changes character counts. If we add a character (doubling "l" → "ll"), everything after it shifts by 1. If we drop a character, everything shifts by -1. If we merge two words by removing a space, the second word's characters shift left. The `old_to_new` mapping in `inject_ocr_noise_with_spans()` tracks these shifts so entity spans remain accurate after noise is applied.

### Decision 4: 50% Character Overlap Threshold for Labeling

When assigning BIO labels to subword tokens during training data generation, we check how much of each subword's character range overlaps with an entity span. If >=50% of the subword's characters fall within an entity, that subword gets the entity's label.

**Why 50%?** This handles boundary subwords gracefully. When a subword straddles the edge of an entity — like a subword that covers "k," from "Braddock," where the "k" is part of the entity but the comma isn't — the 50% threshold ensures the subword gets labeled if most of its content is entity text. A stricter threshold (like 90%) would miss these boundary tokens; a looser one (like 10%) would over-label non-entity tokens.

### Decision 5: First-Entity-Only Extraction

NER models typically extract *all* entities they find. But the requirement is to extract only the first person's name. The system achieves this through two layers:

1. **Training data labels only the first person**: In all templates, only the first first-name and last-name are labeled. Additional names are labeled "O". The model learns to label only the first person's name as entities.
2. **Postprocessor takes the first occurrence**: Even if the model occasionally labels a second name, `postprocess()` takes `entities["FIRST_NAME"][0]` — the first span it finds. This acts as a safety net.

**Why not just use the postprocessor?** If the model labeled every name as FIRST_NAME, the postprocessor would always return the first, yes — but the model would be confused about what the entity labels mean. Training it to label only the first person makes the model's predictions cleaner and more reliable.

### Decision 6: Street Name Filtering (Generic Word Removal)

Street addresses contain generic words like "Ave", "St", "Dr", "North" that aren't distinctive. The postprocessor filters these out because the requirement is to return the **most distinctive** part of the street name — "Braddock" from "1201 Braddock Ave", not "Ave".

The filtering pipeline works in order:
1. Join subword tokens into complete words (gap-aware, so "silver lake" stays as two words)
2. Split multi-word spans into individual words
3. Filter out words that appear in the `GENERIC_STREET_WORDS` set (45 common street suffixes and directions)
4. Filter out numeric tokens (street numbers like "1201") using a heuristic: if >=50% of a token's characters are digits, it's numeric
5. Return the first remaining word; if all words were filtered, return the first word anyway (better to return something than nothing)

**Why the >=50% digit heuristic instead of `isdigit()`?** OCR can mangle street numbers — "1201" might become "12O1" or "l201". A strict `isdigit()` check would miss these. The 50% digit threshold catches OCR-corrupted numbers while not accidentally filtering words like "1st" (which is 33% digits and would pass).

### Decision 7: ONNX Runtime with Single-Thread Configuration

The inference pipeline uses ONNX Runtime with `intra_op_num_threads=1` and `inter_op_num_threads=1`.

**Why single-threaded?** Multi-threaded inference adds parallelism overhead (thread synchronization, cache contention) that only pays off for large inputs. Our inputs are tiny (typically <64 tokens). Single-threaded execution eliminates this overhead and gives more predictable, consistent latency (~10ms p99 vs potentially more variable multi-threaded performance). In a production server handling many concurrent requests, each request should be single-threaded while the server itself handles concurrency at the request level.

### Decision 8: MAX_SEQ_LENGTH = 64

OCR-scanned customer records are short text — typically a name, an address, and maybe an email or phone number. 64 tokens is enough to cover even the longest realistic inputs while keeping inference fast.

**Why not 128 or 512 (BERT's max)?** Transformer self-attention is O(n^2) in sequence length. Doubling the sequence length roughly quadruples the attention computation. Since our inputs never exceed ~40-50 tokens in practice, padding to 128 or 512 would waste compute on padding tokens that contribute nothing.

### Decision 9: Pre-Tokenized Training Data

The training data JSON contains pre-computed `input_ids`, `attention_mask`, and `labels` rather than raw text with word-level tags. The tokenization and BIO label assignment happen in the data generation step, not during training.

**Why pre-tokenize?** The label assignment logic (character overlap computation, B vs I determination) is complex and specific to our approach. Doing it once during data generation keeps the training script simple and ensures the tokenization used for labels matches exactly what inference will use. It also means training doesn't need to re-tokenize on every epoch.

### Decision 10: Training Hyperparameters

- **Learning rate 5e-5**: Standard for BERT fine-tuning. Lower rates (1e-5) underfit; higher rates (1e-4) can overshoot and destabilize training.
- **15 epochs**: More epochs than typical BERT fine-tuning (usually 3-5) because our dataset is small (4,000 examples). The model needs more passes to learn the patterns.
- **Batch size 16**: Small enough to fit in memory, large enough for stable gradient estimates. For our small dataset, larger batches would mean fewer gradient updates per epoch.
- **Warmup ratio 0.1**: The first 10% of training steps use a linearly increasing learning rate. This prevents the randomly-initialized classification head from producing large gradients that corrupt the pre-trained weights.
- **Weight decay 0.01**: L2 regularization to prevent overfitting on our small synthetic dataset.
- **85/15 train/eval split**: 600 examples for evaluation is enough to get reliable metrics. Using more for training (3,400) gives the model more patterns to learn from.
- **`load_best_model_at_end=True` with `metric_for_best_model="f1"`**: Saves the model checkpoint with the highest F1 score across all epochs, not just the final epoch. This prevents returning an overfit model from late training.

---

## How It All Fits Together

The project has two phases: **Training** (done once or when you want to retrain) and **Inference** (done repeatedly in production).

### Training Phase

```
1. generate_training_data.py    Creates 4000 synthetic labeled examples
         |                      (with OCR noise including word merging)
         |                      Uses TextBuilder for character-span tracking
         |                      Assigns BIO labels per-subword via offset_mapping
         v
2. train.py                     Fine-tunes DistilBERT on those examples
         |                      85% train / 15% eval split
         |                      Saves best model by F1 score
         v
3. export_onnx.py               Converts PyTorch model to ONNX format
         |                      Applies INT8 dynamic quantization
         |                      265 MB → 67 MB
         v
4. evaluate.py                  Runs full inference pipeline on test examples
                                Reports per-field accuracy + sample errors
```

### Inference Phase

```
Raw OCR text (e.g., "Alex or Mary Doe, 1201 Braddock Ave, Richmond VA")
    |
    v
model.py: NameAddressParser.parse()
    |
    |-- 1. Tokenize raw text with WordPiece
    |      Input:  "Alex or Mary Doe, 1201 Braddock Ave"
    |      Output: tokens    = ["[CLS]", "alex", "or", "mary", "doe", ",", "1201", "brad", "##dock", "ave", "[SEP]"]
    |              offsets   = [(0,0),   (0,4), (5,7), (8,12), (13,16), (16,17), (18,22), (23,27), (27,31), (32,35), (0,0)]
    |
    |-- 2. Run ONNX inference (CPU, ~10ms)
    |      Output: logits matrix (1 x seq_len x 7)
    |      Argmax: [-, B-FIRST, O, O, B-LAST, O, O, B-STREET, I-STREET, O, -]
    |
    |-- 3. Post-process (postprocessor.py)
    |      |
    |      |-- extract_entities(): group consecutive B/I tokens into spans
    |      |     Each token carries its character offsets (start, end)
    |      |     STREET_NAME: [("brad", 23, 27), ("dock", 27, 31)]
    |      |
    |      |-- _join_token_infos(): reconstruct text from subwords
    |      |     Checks offset gaps: 27→27 (no gap) → "braddock" (concatenate)
    |      |     If it were 27→28 (gap) → "brad dock" (insert space)
    |      |
    |      |-- filter_street_name(): find distinctive street word
    |      |     Split multi-word results → filter generics → filter numerics → pick first
    |      |
    |      |-- Strip trailing punctuation
    |
    v
{"first_name": "alex", "last_name": "doe", "street_name": "braddock"}
```

---

## File-by-File Breakdown

### `pyproject.toml` — Project Configuration

Defines the project's package metadata and three dependency groups:

- **Runtime** (always needed): `onnxruntime` (runs the ONNX model), `transformers` (provides the tokenizer), `numpy`. These are the only dependencies needed in production.
- **Training** (`pip install -e ".[train]"`): adds `torch` (PyTorch for fine-tuning), `datasets` (HuggingFace data loading), `optimum[onnxruntime]` (ONNX export + quantization), `faker` (generates fake city names), `seqeval` (NER accuracy metrics), `accelerate` (HuggingFace training utilities).
- **Dev** (`pip install -e ".[dev]"`): adds `pytest` and `pytest-benchmark` for testing.

**Design note:** The runtime dependencies are intentionally minimal. In production, you don't need PyTorch (200+ MB) — only `onnxruntime` (~30 MB), `transformers` (for the tokenizer), and `numpy`. This keeps the production deployment small.

Uses `build-backend = "setuptools.build_meta"` with `[tool.setuptools.packages.find] where = ["src"]` for a src-layout package structure, which prevents accidental imports of the source directory during development.

---

### `src/name_parsing/__init__.py` — Package Entry Point

Exports only `NameAddressParser` — the single class users need to interact with. This keeps the public API surface minimal.

---

### `src/name_parsing/config.py` — Central Configuration

Single source of truth for all constants used across training and inference.

**Label definitions:**
```python
LABEL_LIST = ["O", "B-FIRST_NAME", "I-FIRST_NAME", "B-LAST_NAME", "I-LAST_NAME", "B-STREET_NAME", "I-STREET_NAME"]
LABEL2ID = {"O": 0, "B-FIRST_NAME": 1, ...}  # label string → integer
ID2LABEL = {0: "O", 1: "B-FIRST_NAME", ...}  # integer → label string
```

The model outputs numbers (0-6), not strings. These dictionaries convert back and forth. The order matters — "O" must be index 0 because it's the default/most common label, and -100 is used for special tokens that should be ignored during loss computation (a PyTorch convention for `CrossEntropyLoss`).

**Model paths:** `PROJECT_ROOT` is computed relative to the config file's location, making paths work regardless of the working directory. All model artifacts (finetuned, ONNX, quantized) live under `models/`.

**Training hyperparameters:** `MAX_SEQ_LENGTH = 64`, `TRAIN_EPOCHS = 15`, `LEARNING_RATE = 5e-5`, `BATCH_SIZE = 16`, `TRAIN_TEST_SPLIT = 0.15`. See [Decision 10](#decision-10-training-hyperparameters) for rationale.

**`GENERIC_STREET_WORDS`**: A set of 45 words that appear in street addresses but aren't the distinctive part of the street name. Includes suffixes (ave, st, blvd, dr...), their abbreviations, and cardinal directions (north, south, n, s...). Used by the postprocessor to filter down to the distinctive word.

**Why a hardcoded set instead of a learned filter?** The list of generic street words is finite and well-known. A learned approach would need training data where the model distinguishes "Braddock" from "Ave" — but since the model already labels both as STREET_NAME (which is correct), the filtering is better done as a deterministic postprocessing step.

---

### `training/generate_training_data.py` — Synthetic Training Data

Since we don't have a pre-existing labeled dataset of real OCR-scanned records, this script creates one by generating realistic text with known entity positions.

**`TextBuilder` class** — The core innovation for character-span tracking. It builds text piece by piece while recording the exact character range of each entity:

```python
tb = TextBuilder()
tb.add("Alex", "FIRST_NAME")  # records span {"start": 0, "end": 4, "label": "FIRST_NAME"}
tb.add_space()                  # adds " " at position 4
tb.add("Doe", "LAST_NAME")    # records span {"start": 5, "end": 8, "label": "LAST_NAME"}
```

**Why a builder pattern?** String concatenation doesn't track positions. With TextBuilder, we always know exactly where each entity is in the output string, even as the string grows. This is critical because OCR noise injection will shift these positions — and we need to know the original positions to compute the shifted ones.

**Templates** — Four patterns with weighted random selection:
- `"Alex Doe, 1201 Braddock Ave, ..."` (single name, 35%) — the most common pattern
- `"Alex or Mary Doe, ..."` (shared last name, 30%) — multiple people at the same address
- `"Alex Doe or Mary Smith, ..."` (separate full names, 25%) — two full names
- `"Alex Doe, Mary Smith, John Brown, ..."` (multiple names, 10%) — rare but exists

Each template labels only the first person's first name and last name. Additional names are added as unlabeled text ("O"). The street name is always labeled.

**Why these weights?** They roughly reflect the distribution of real customer records. Single names are most common, followed by records with shared last names. The model needs to see all patterns during training but should see the common ones more often.

**`inject_ocr_noise_with_spans()`** — Simulates OCR errors while keeping entity spans aligned.

The function builds an `old_to_new` character position mapping as it processes each character:
- **Character substitution** (60% of errors): Uses `OCR_CONFUSIONS` dictionary mapping characters to their common OCR misreadings. Includes single-character confusions (`l` → `1`, `O` → `0`) and bigram confusions (`rn` → `m`, `d` → `cl`). These are based on real OCR confusion patterns.
- **Character dropping** (20% of errors): A character is simply removed, simulating OCR failing to detect a glyph.
- **Character doubling** (20% of errors): A character is repeated, simulating OCR stuttering on a glyph.
- **Word merging** (controlled by `merge_rate`, separate from character errors): Spaces are randomly removed, optionally replaced with junk characters (`/`, `|`, `-`, `.`, `~`, `\`). This simulates OCR failing to detect the whitespace between words.

**Why track `old_to_new` instead of re-scanning for entities?** After noise injection, the text is different from the original — characters have been added, removed, or replaced. Re-scanning would require pattern matching on corrupted text, which is exactly the hard problem we're trying to train the model to solve. By tracking position shifts during noise injection, we know exactly where each entity ended up in the noisy text.

**BIO label assignment via character overlap** — After noise injection, the script tokenizes the noisy text with `return_offsets_mapping=True`, then for each subword:
1. Checks if the subword's character range overlaps with any entity span
2. If >=50% of the subword's characters fall within an entity, it gets that entity's label
3. B- vs I- is determined by checking if the *previous* subword also belonged to the same entity

**Data mix** — 30% of examples are clean (no noise), 70% have noise with randomly varying intensity. The clean examples teach the model the basic structure; the noisy examples teach robustness.

**Output format**: Pre-tokenized JSON with `text`, `input_ids`, `attention_mask`, `labels`. The `spans` field is kept during generation for debugging but stripped from the output file to reduce size.

---

### `training/train.py` — Fine-Tuning DistilBERT

The training script is deliberately simple because all the complexity lives in data generation.

**`load_data()`**: Reads the JSON file, extracts `input_ids`, `attention_mask`, and `labels`, and splits 85/15 into train/eval using HuggingFace's `Dataset.train_test_split()`. The `seed=42` ensures reproducible splits.

**Model setup**: Loads `distilbert-base-uncased` from HuggingFace with a fresh `TokenClassification` head — a single linear layer mapping 768-dim hidden states to 7 label logits. The base model weights are pre-trained; only the classification head starts random.

**`DataCollatorForTokenClassification`**: Handles dynamic padding. Since our examples have variable lengths (depending on text length), the collator pads each batch to the length of the longest example in that batch. This is more efficient than padding everything to `MAX_SEQ_LENGTH`. It also properly handles padding the `labels` tensor with -100 so padded positions don't contribute to the loss.

**`compute_metrics()`**: Uses `seqeval`, the standard NER evaluation library. `seqeval` evaluates at the entity level, not the token level — an entity is correct only if all its tokens (B- and I-) are predicted correctly. This is a stricter and more meaningful metric than per-token accuracy.

**`TrainingArguments` details**:
- `eval_strategy="epoch"`: Evaluate after every full pass through the training data. More frequent evaluation (every N steps) would slow training without much benefit given our small dataset.
- `save_strategy="epoch"` + `save_total_limit=2`: Keeps only the 2 most recent checkpoints to save disk space. With `load_best_model_at_end=True`, the best model is always available.
- `warmup_ratio=0.1`: Linear warmup over the first 10% of steps. The classification head's random initialization produces large initial gradients; warmup prevents these from corrupting the pre-trained base weights.
- `fp16=False`: We train on CPU (or MPS on Mac), where FP16 isn't beneficial. On GPU, enabling FP16 would speed up training.
- `report_to="none"`: Disables logging to Weights & Biases or TensorBoard. For a small training run, console output is sufficient.

---

### `training/export_onnx.py` — ONNX Export and Quantization

Converts the fine-tuned PyTorch model into a production-ready ONNX format in two steps.

**Step 1 — ONNX Export**: Uses the `optimum` library's `ORTModelForTokenClassification.from_pretrained(..., export=True)` to trace the PyTorch model and produce an ONNX graph. The ONNX graph is a static computation graph that ONNX Runtime can optimize (operator fusion, memory pre-allocation, etc.).

**Step 2 — INT8 Dynamic Quantization**: Uses `ORTQuantizer` with `AutoQuantizationConfig.avx2(is_static=False, per_channel=False)`. The `avx2` config generates an ONNX model optimized for x86 CPUs with AVX2 instruction support (most modern Intel/AMD processors). `per_channel=False` uses per-tensor quantization, which is simpler and sufficient for our use case.

**Why copy the tokenizer to the quantized directory?** The ONNX model needs a co-located tokenizer for inference. `NameAddressParser` loads both from the same directory. By copying the tokenizer alongside the quantized model, the quantized directory is self-contained — you can deploy it without any other files.

**Size reduction**: 265 MB (FP32 ONNX) → 67 MB (INT8 quantized), a 75% reduction. The 67 MB is mostly the embedding layer and attention weights stored as INT8.

---

### `src/name_parsing/model.py` — Inference Pipeline

The main file used in production. Contains the `NameAddressParser` class.

**`__init__(model_dir)`**:
- Finds the `.onnx` file in the model directory (flexible naming)
- Creates an `ort.InferenceSession` with CPU-optimized settings: single-threaded, all graph optimizations enabled
- Loads the tokenizer from the same directory

**Why auto-detect the .onnx file instead of hardcoding?** The ONNX export toolchain names files differently depending on the version. Globbing for `*.onnx` makes the code robust to naming changes.

**`parse(text)`** — The main method:

1. **Guard clause**: Returns empty fields for empty/whitespace input
2. **Tokenize**: Calls `self.tokenizer(text, ..., return_offsets_mapping=True, return_tensors="np")`. The `return_tensors="np"` gives NumPy arrays directly (ONNX Runtime's native format), avoiding a PyTorch dependency at inference time. The `return_offsets_mapping=True` provides the character range each subword token covers.
3. **ONNX inference**: Feeds `input_ids` and `attention_mask` to the ONNX session. The `offset_mapping` is NOT passed to the model — the model doesn't use it. It's only needed for postprocessing.
4. **Argmax**: Converts the raw logits (7 scores per token) into predicted label IDs by taking the highest score.
5. **Token reconstruction**: Converts `input_ids` back to token strings (like "brad", "##dock") using `convert_ids_to_tokens()`. These strings are needed by the postprocessor to reconstruct entity text.
6. **Postprocess**: Passes predictions, tokens, and offset_mapping to `postprocess()`.

**`parse_batch(texts)`**: Processes texts sequentially. A future optimization could batch multiple inputs into a single ONNX call for throughput (not latency) improvement.

---

### `src/name_parsing/postprocessor.py` — Entity Extraction and Filtering

Converts raw model predictions into the final JSON output. This is where the gap-aware token joining logic lives.

**`TokenInfo = tuple[str, int, int]`** — A type alias for (clean_text, char_start, char_end). Every extracted token carries its character offsets from the original text, which are essential for gap detection.

**`extract_entities(predictions, tokens, offset_mapping)`**:

Walks through predictions and groups consecutive BIO-tagged tokens into spans. Each token in a span is stored as a `TokenInfo` tuple carrying its character offsets.

Key behaviors:
- **Special tokens** (offset 0,0): Close any open entity and skip
- **B- tag**: Start a new entity span (close any previous one)
- **I- tag matching current entity**: Append to current span
- **I- tag mismatching current entity**: Close current span, discard the mismatched token (treat as noise)
- **O tag**: Close any open entity

**`_join_token_infos(token_infos)`** — The gap-aware joining function:

```python
# Example: "silver" at (0,6) + "lake" at (7,11)
# prev_end=6, curr_start=7: gap! → insert space → "silver lake"

# Example: "brad" at (0,4) + "dock" at (4,8)
# prev_end=4, curr_start=4: no gap → concatenate → "braddock"
```

This is the fix for the "silverlake" bug. Without offset-aware joining, any two tokens in the same entity span would be concatenated, losing word boundaries.

**`filter_street_name(street_spans)`**:

1. Joins subwords within each span using `_join_token_infos()` (gap-aware)
2. Splits multi-word results into individual words (so "silver lake" becomes ["silver", "lake"])
3. Filters out generic street words (`GENERIC_STREET_WORDS` set)
4. Filters out numeric tokens (>=50% digits)
5. Returns the first remaining word; falls back to the first word if all were filtered

**Why split multi-word spans before filtering?** The model might label "Silver Lake Dr" as a single STREET_NAME span. After gap-aware joining, we get "silver lake dr". We need to split this into individual words so we can filter out "dr" (generic) and return "silver" (the first distinctive word). Without splitting, "silver lake dr" would be treated as one unit and compared against the generic words set — where it wouldn't match, so the full string would be returned.

**`postprocess(predictions, tokens, offset_mapping)`** — The main entry point:

1. Calls `extract_entities()` to get all entity spans
2. Takes `[0]` of each entity type (first occurrence = first person)
3. Joins subwords within each entity using `_join_token_infos()` (gap-aware)
4. Calls `filter_street_name()` for street name
5. Strips trailing punctuation from all values

---

### `training/evaluate.py` — Model Evaluation

Measures **end-to-end accuracy** by running the full inference pipeline (model + postprocessor) against held-out test data.

**`extract_expected_from_example()`**: Reconstructs expected values from the pre-tokenized training data. Converts input_ids back to tokens, finds B/I-tagged tokens for each field, joins them, and applies the same street name filtering.

**Why evaluate end-to-end instead of just model accuracy?** The model's token-level F1 (measured by `seqeval` during training) can be high even if the postprocessor introduces errors. End-to-end evaluation catches bugs in the postprocessor, the joining logic, and the filtering logic.

The evaluation compares predicted vs expected values case-insensitively and reports per-field accuracy plus sample errors. Errors are printed so you can diagnose patterns (e.g., "the model consistently confuses X with Y").

---

### `tests/test_postprocessor.py` — Unit Tests for Post-Processing

34 tests organized into four test classes:

**`TestExtractEntities`** (6 tests): Verifies that BIO-tagged subword tokens are correctly grouped into entity spans for:
- Single names, shared last names, separate names
- Multi-subword entities (like "braddock" split into "brad" + "##dock")
- Multi-token first names (like "Mary Jane")
- Edge case: only special tokens

**`TestJoinTokenInfos`** (5 tests): Verifies the gap-aware joining logic:
- Adjacent subwords (no gap) concatenate: ("brad", 0, 4) + ("dock", 4, 8) → "braddock"
- Separate words (gap) get spaces: ("silver", 0, 6) + ("lake", 7, 11) → "silver lake"
- Empty input, single token

**`TestFilterStreetName`** (9 tests): Verifies street name filtering:
- Single distinct word, joined subwords
- Multi-word street with non-generic words
- Generic word filtering, numeric filtering, OCR-mangled number filtering
- Empty input, all-generic fallback, multiple spans

**`TestPostprocess`** (6 tests): End-to-end postprocessor tests:
- Full pipeline, punctuation stripping, empty input
- OCR-merged tokens ("37/harbor" → "harbor")
- Multi-word street names ("silver lake" → "silver")
- Multi-word first names with spaces preserved

All tests use a `_make_mock_data()` helper that builds realistic mock data with proper offset_mapping gaps (spaces between non-## tokens, no gaps between ## tokens).

---

### `tests/test_inference.py` — Integration Tests and Benchmarking

8 tests that require the trained ONNX model to be present (skip if not):

**`TestInference`** (7 tests): Tests the full parse pipeline on representative inputs:
- Single name, shared last name, separate names
- Text with email appended, middle initial
- Returns all expected keys
- Empty input handling

**`TestBenchmark`** (1 test): Measures inference latency over 100 runs after 5 warmup runs. Asserts p99 < 100ms. Currently achieves ~10ms p99.

**Why warmup runs?** The first few calls are slower due to ONNX Runtime JIT compilation, tokenizer loading, and CPU cache warming. Warmup ensures the benchmark measures steady-state performance.

---

## The V1 to V2 Evolution: Lessons Learned

### V1: Word-Level Tokenization

V1 used `is_split_into_words=True`, which meant:
1. Split text by whitespace into words
2. Tokenize each word separately into subwords
3. Use `word_ids()` to map subwords back to original words
4. Assign one label per word (using the first subword's prediction)

**V1 results**: F1 = 98.8%, first_name 100%, last_name 99.6%, street_name 94.2% (improved to 96.6% with numeric filtering). Latency ~10ms.

**V1's fatal flaw**: When OCR merged "37 Harbor" into "37/harbor", V1 treated it as a single word. The entire merged token received one label from the first subword's prediction. The model couldn't say "the '37' part is O but 'harbor' is STREET_NAME" because word boundaries were locked at whitespace before the model even saw the text.

### V2: Character-Level Tokenization

V2 feeds raw text to the tokenizer directly, letting WordPiece split based on learned patterns rather than whitespace. Each subword gets its own label.

**V2 results**: F1 = 99.75%, first_name 100%, last_name 100%, street_name 99.8%. Latency ~10ms (unchanged).

**Key insight**: The tokenizer is smarter than whitespace splitting. WordPiece knows that "/" is a natural token boundary even without spaces. By trusting the tokenizer instead of overriding it with `is_split_into_words`, we get better subword boundaries and the ability to label within merged tokens.

### The Multi-Word Street Name Bug

After V2 was working, a new bug appeared: multi-word street names like "Silver Lake" were being concatenated into "SilverLake". This happened because the postprocessor joined all tokens in an entity span with `"".join()` — which is correct for subwords of a single word ("brad" + "dock" = "braddock") but wrong for tokens that were separate words in the original text.

**The fix**: Store character offsets alongside each token, then check for gaps when joining. If consecutive tokens have a gap in their character positions, there was a space in the original text, and we insert one. This is the `_join_token_infos()` function.

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Training F1 | 99.75% |
| first_name accuracy | 100% (500/500) |
| last_name accuracy | 100% (499/499) |
| street_name accuracy | 99.8% (499/500) |
| Inference latency (p50) | ~10ms |
| Inference latency (p99) | ~10ms |
| Model size (quantized) | 67 MB |
| Model size (FP32 ONNX) | 266 MB |
| Training time | ~6 minutes (CPU) |
| Training examples | 4,000 synthetic |

---

## How to Improve the Model

If accuracy isn't sufficient for your production needs, here are the most effective improvements in order of impact:

1. **Add real labeled examples**: The model was trained on synthetic data only. Even 200-500 real OCR-scanned examples (manually labeled) would significantly improve accuracy, especially for edge cases the templates don't cover (business names, PO boxes, unusual formatting).

2. **Increase training data variety**: Add more templates to `generate_training_data.py` — business names, apartment/suite numbers, PO boxes, international addresses, multi-line OCR where lines are concatenated.

3. **Target specific failure modes**: Run `evaluate.py` and examine the errors. If the model consistently confuses certain patterns, add more training examples that emphasize the difference. The error output shows you exactly what the model predicted vs what was expected.

4. **Adjust OCR noise parameters**: If your real OCR has a different error profile than the synthetic noise, tune the `OCR_CONFUSIONS` dictionary, `error_rate`, and `merge_rate` to match your actual OCR output. You can analyze a sample of real OCR text to estimate these parameters.

5. **Try a larger model**: Swap `distilbert-base-uncased` for `bert-base-uncased` in `config.py`. This will roughly double inference time (~20ms) but may improve accuracy 1-2%. Still well under the 100ms budget.

6. **Increase training data volume**: Generate 10,000-20,000 examples instead of 4,000. More data generally helps, especially for the noisy examples where the model needs to learn many different corruption patterns.

7. **Ensemble with rules**: For the highest accuracy, combine the model's predictions with simple pattern-matching rules as a post-processing sanity check. For example, if the model outputs a street_name that matches a known last name and there's a more likely street name candidate in the text, prefer the alternative.

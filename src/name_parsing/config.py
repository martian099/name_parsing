from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Label schema for BIO tagging
LABEL_LIST = [
    "O",
    "B-FIRST_NAME",
    "I-FIRST_NAME",
    "B-LAST_NAME",
    "I-LAST_NAME",
    "B-STREET_NAME",
    "I-STREET_NAME",
]

LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

# Model config
BASE_MODEL_NAME = "answerdotai/ModernBERT-base"
FINETUNED_MODEL_DIR = PROJECT_ROOT / "models" / "finetuned"
ONNX_MODEL_DIR = PROJECT_ROOT / "models" / "onnx"
ONNX_MODEL_PATH = ONNX_MODEL_DIR / "model_quantized.onnx"

# Data paths
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Training defaults
MAX_SEQ_LENGTH = 64
TRAIN_EPOCHS = 10
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.15

# Generic street words to filter out during post-processing
GENERIC_STREET_WORDS = {
    "street", "st", "avenue", "ave", "drive", "dr", "road", "rd",
    "boulevard", "blvd", "lane", "ln", "court", "ct", "place", "pl",
    "way", "circle", "cir", "terrace", "ter", "trail", "trl",
    "parkway", "pkwy", "highway", "hwy", "loop", "run", "path",
    "north", "south", "east", "west", "n", "s", "e", "w",
    "ne", "nw", "se", "sw", "northeast", "northwest", "southeast", "southwest",
}

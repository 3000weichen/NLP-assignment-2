"""Paths and cross-validation settings."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

# Same split settings as Project 1.
SEEDS = (0,)
NUM_FOLDS = 5
INNER_VAL_FRACTION = 0.2

# GloVe archive; the loader reads it without extracting.
EMBEDDING_PATH = PROJECT_DIR / "resources" / "glove.2024.wikigiga.50d.zip"
EMBEDDING_DIM = 50
EMBEDDING_SKIP_FIRST_LINE = False

"""Paths and cross-validation settings."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

# Keep seed 0 for the Project 1 comparison; add repeated CV runs.
SEEDS = (0, 3, 5)
NUM_FOLDS = 5
INNER_VAL_FRACTION = 0.2

# GloVe archive; the loader reads it without extracting.
EMBEDDING_PATH = PROJECT_DIR / "resources" / "glove.2024.wikigiga.50d.zip"
EMBEDDING_DIM = 50
EMBEDDING_SKIP_FIRST_LINE = False

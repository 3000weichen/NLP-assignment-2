"""Load GloVe and its tokenizer.

Sources: tutorial Pre-trained word embeddings and Tokenizer.
Still to add: coverage statistics and nearest-word examples."""

from pathlib import Path

import nltk
import numpy as np

from ..clslib import BasicTokenizer, PretrainedEmbeddings
from .config import EMBEDDING_DIM, EMBEDDING_PATH, EMBEDDING_SKIP_FIRST_LINE

__all__ = ["PretrainedEmbeddings", "load_embeddings", "create_tokenizer"]


def load_embeddings(
    path: str | Path = EMBEDDING_PATH,
    num_dims: int = EMBEDDING_DIM,
    skip_first_line: bool = EMBEDDING_SKIP_FIRST_LINE,
) -> PretrainedEmbeddings:
    """Load a local txt, gz, or zip file with the tutorial loader.

    ZIP files must contain a single embedding file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Embedding file not found: {path}. Set EMBEDDING_PATH in config.py."
        )
    if isinstance(num_dims, bool) or not isinstance(num_dims, int) or num_dims <= 0:
        raise ValueError("num_dims must be a positive integer")

    model = PretrainedEmbeddings(
        str(path), num_dims=num_dims, skip_first_line=skip_first_line,
    )
    if model.vectors.shape[0] <= 2:
        raise ValueError("No word vectors loaded; check the file and num_dims")
    if not np.isfinite(model.vectors).all():
        raise ValueError("Embedding matrix contains non-finite values")
    return model


def create_tokenizer(emb_model: PretrainedEmbeddings) -> BasicTokenizer:
    """Use the embedding vocabulary for both pooling and LSTM.

    Keep stopwords and punctuation so negation is not removed."""
    # Look for NLTK data in the project resources folder.
    nltk_dir = Path(__file__).resolve().parent / "resources" / "nltk_data"
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    return BasicTokenizer(
        lowercase=True,
        remove_punct=False,
        remove_stopwords=False,
        language="english",
        vocab=emb_model.vocab,
    )

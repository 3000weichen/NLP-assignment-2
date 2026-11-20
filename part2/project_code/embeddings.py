"""Load GloVe and its tokenizer.

Sources: tutorial Pre-trained word embeddings and Tokenizer.
Coverage statistics are computed in project_code.analysis."""

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


def nearest_words(emb_model, words, top_k=8):
    """Inspect task-related cosine neighbours without using class labels.

    Source: tutorial Word Embeddings > Exercises: Similarity and Analogies.
    These examples describe the space; they do not establish classification quality.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    vectors = emb_model.vectors
    norms = np.linalg.norm(vectors, axis=1)
    pad = emb_model.vocab.index(emb_model.pad_token)
    unk = emb_model.vocab.index(emb_model.unk_token)
    rows = []
    for word in words:
        word = word.lower()
        if word not in emb_model.vocab:
            rows.append({"query": word, "rank": 0, "neighbour": "OOV", "cosine": np.nan})
            continue
        index = emb_model.vocab.index(word)
        scores = vectors @ vectors[index]
        scores = scores / np.maximum(norms * norms[index], 1e-12)
        scores[[pad, unk, index]] = -np.inf
        candidates = np.flatnonzero(np.isfinite(scores))
        best = candidates[np.argsort(scores[candidates])[-top_k:][::-1]]
        for rank, other in enumerate(best, 1):
            rows.append({"query": word, "rank": rank,
                         "neighbour": emb_model.vocab.word(int(other)),
                         "cosine": float(scores[other])})
    return rows


if __name__ == "__main__":
    import argparse
    import csv
    parser = argparse.ArgumentParser(description="Inspect task-related GloVe neighbours")
    parser.add_argument("--words", nargs="+", default=[
        "atheism", "atheist", "religion", "religious", "god", "evidence",
        "superstition", "believe", "disbelief", "not",
    ])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "embedding_neighbours.csv")
    args = parser.parse_args()
    model = load_embeddings()
    rows = nearest_words(model, args.words, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "rank", "neighbour", "cosine"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved neighbours to {args.output}", flush=True)

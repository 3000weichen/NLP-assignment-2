"""Load stance data, prepare folds, and batch token sequences.

Sources: tutorial Experiment > Data; Encoders > Embed and Pool (padding).
Fold generation is reused from Project 1."""

from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from ..clslib import BasicTokenizer, Tokenizer, Vocabulary, collate_sequences
from ..experiment import LabelledCorpus, collate_labelled_texts
from part1.previous_project.experiment_utils import make_splits

from .config import INNER_VAL_FRACTION, NUM_FOLDS, SEEDS

__all__ = [
    "LabelledCorpus", "collate_labelled_texts", "BasicTokenizer",
    "Tokenizer", "Vocabulary", "collate_sequences",
    "load_data", "create_splits", "encode_texts", "make_dataloader",
]


def load_data() -> tuple[dict[str, list[dict]], list[str]]:
    """Load the official splits and keep the original label IDs."""
    dataset = load_dataset("cardiffnlp/tweet_eval", "stance_atheism")
    label_names = list(dataset["train"].features["label"].names)
    source_splits = {
        split: [
            {"sample_id": f"{split}:{i}", "text": sample["text"], "label": sample["label"]}
            for i, sample in enumerate(dataset[split])
        ]
        for split in ("train", "validation", "test")
    }
    return source_splits, label_names


def create_splits(
    training_samples: list[dict],
    seeds: tuple[int, ...] = SEEDS,
    num_folds: int = NUM_FOLDS,
    inner_val_fraction: float = INNER_VAL_FRACTION,
) -> list[dict]:
    """Create the same nested folds as Project 1.

    Keep training samples in their original order. The returned fit, dev, and
    outer_eval lists contain indices into training_samples."""
    if not training_samples:
        raise ValueError("training_samples must not be empty")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain distinct random seeds")
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    if not 0 < inner_val_fraction < 1:
        raise ValueError("inner_val_fraction must be between 0 and 1")

    labels = [sample["label"] for sample in training_samples]
    return make_splits(labels, seeds, num_folds, inner_val_fraction)


def encode_texts(
    samples: list[dict],
    tokenizer: Tokenizer,
    unk_id: int,
    max_length: int | None = None,
) -> tuple[list[list[int]], list[int], list[str]]:
    """Return token sequences, labels, and sample IDs in matching order."""
    if max_length is not None and (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length <= 0
    ):
        raise ValueError("max_length must be None or a positive integer")

    sequences = []
    labels = []
    sample_ids = []
    for sample in samples:
        ids = list(tokenizer.ids(sample["text"]))
        if max_length is not None:
            ids = ids[:max_length]
        if not ids:
            ids = [unk_id]
        sequences.append(ids)
        labels.append(int(sample["label"]))
        sample_ids.append(sample["sample_id"])

    return sequences, labels, sample_ids


def make_dataloader(
    sequences: list[list[int]],
    labels: list[int],
    padding_idx: int,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
) -> DataLoader:
    """Pad sequences within each batch. Use shuffle=False for evaluation."""
    if len(sequences) != len(labels):
        raise ValueError("sequences and labels must have equal lengths")
    if not sequences or any(len(sequence) == 0 for sequence in sequences):
        raise ValueError("Dataset and token sequences must not be empty")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    dataset = LabelledCorpus(sequences, labels)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=partial(collate_labelled_texts, padding_value=padding_idx),
        generator=generator,
    )

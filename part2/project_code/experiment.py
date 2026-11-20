
import csv
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Source: tutorial Experiment > Training, cells 128 and 130.

from ..experiment import evaluate, training_one_epoch
from .config import RESULTS_DIR, SEEDS, NUM_FOLDS, INNER_VAL_FRACTION
from .data import create_splits, encode_texts, load_data, make_dataloader
from .embeddings import create_tokenizer, load_embeddings
# Source: tutorial Encoders > Embed and Pool / LSTM Encoder;
# Text Classification Revisited > Softmax Regressor.
from ..clslib import (
    EmbedAndPoolEncoder, LSTMEncoder, PretrainedEmbeddings, SoftmaxRegressor,
)

# All four systems use the same head settings and data splits.
SYSTEMS = {
    "pooling_frozen": {"encoder_type": "pooling", "freeze_embeddings": True},
    "pooling_finetuned": {"encoder_type": "pooling", "freeze_embeddings": False},
    "lstm_frozen": {"encoder_type": "lstm", "freeze_embeddings": True},
    "lstm_finetuned": {"encoder_type": "lstm", "freeze_embeddings": False},
}



def build_model(
    emb_model: PretrainedEmbeddings,
    num_classes: int,
    encoder_type: str = "pooling",
    freeze_embeddings: bool = True,
    pooling: str = "mean",
    hidden_size: int = 64,
    hidden_size1: int = 50,
    hidden_size2: int = 50,
    dropout: float = 0.1,
    word_dropout: float = 0.0,
    device: str | torch.device = "cpu",
) -> SoftmaxRegressor:


    """Combine a tutorial encoder with the tutorial classification head.

    Returns log probabilities for NLLLoss. Each model gets its own embedding
    copy so fine-tuning does not change the original vectors."""
    if encoder_type not in {"pooling", "embed-and-pool", "lstm"}:
        raise ValueError("encoder_type must be 'pooling', 'embed-and-pool', or 'lstm'")
    if pooling not in {"mean", "sum"}:
        raise ValueError("pooling must be 'mean' or 'sum'")
    for name, value in {
        "num_classes": num_classes, "hidden_size": hidden_size,
        "hidden_size1": hidden_size1, "hidden_size2": hidden_size2,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= dropout < 1 or not 0 <= word_dropout < 1:
        raise ValueError("dropout and word_dropout must be in [0, 1)")
    if encoder_type != "lstm" and word_dropout != 0:
        raise ValueError("word_dropout is only supported by the LSTM encoder")

    # Copy the vectors for each run; keep PAD zero for pooling.

    matrix = torch.tensor(emb_model.vectors, dtype=torch.float32)
    pad_id = emb_model.vocab.index(emb_model.pad_token)
    if matrix.ndim != 2 or matrix.shape[0] != len(emb_model.vocab):
        raise ValueError("Embedding matrix rows must match the vocabulary")
    matrix[pad_id].zero_()

    if encoder_type in {"pooling", "embed-and-pool"}:
        encoder = EmbedAndPoolEncoder(
            embedding_matrix=matrix,
            average_pool=(pooling == "mean"),
            padding_idx=pad_id,
            freeze_embeddings=freeze_embeddings,
        )
    else:
        encoder = LSTMEncoder(
            hidden_size=hidden_size,
            embedding_matrix=matrix,
            padding_idx=pad_id,
            freeze_embeddings=freeze_embeddings,
            word_dropout=word_dropout,
        )

    return SoftmaxRegressor(
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        num_classes=num_classes,
        encoder=encoder,
        dropout=dropout,
    ).to(device)

def save_csv(path, rows):
    """Save a list of records as CSV."""
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed):
    """Set random seeds before each training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def prediction_metrics(result, label_names):
    """Compute accuracy, macro-F1, and F1 over the two stance classes."""
    gold = np.array([y for batch in result["labels"] for y in batch])
    predicted = np.array([y for batch in result["predictions"] for y in batch])
    label_ids = list(range(len(label_names)))
    scores = {
        "loss": float(result["loss"]),
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(
            gold, predicted, labels=label_ids, average="macro", zero_division=0,
        )),
    }
    if "against" in label_names and "favor" in label_names:
        stance_ids = [label_names.index("against"), label_names.index("favor")]
        scores["stance_f1"] = float(f1_score(
            gold, predicted, labels=stance_ids, average="macro", zero_division=0,
        ))
    return scores


def plot_history(history, path, title):
    """Plot training and validation loss and accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "validation"):
            axis.plot([r["epoch"] for r in history],
                      [r[f"{split}_{metric}"] for r in history], label=split)
        axis.set(xlabel="Epoch", ylabel=metric, title=metric)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_fold(samples, split, emb_model, tokenizer, label_names, model_kwargs,
             output_dir, epochs=20, batch_size=64, learning_rate=1e-3,
             weight_decay=1e-4, max_length=None, device="cpu"):
    """Train on fit, select on dev, and score once on outer_eval.

    Based on tutorial Experiment > Training. Keep the epoch with the lowest
    validation NLL, as in Project 1."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    set_seed(split["training_seed"])
    started = time.perf_counter()
    pad_id = emb_model.vocab.index(emb_model.pad_token)
    unk_id = emb_model.vocab.index(emb_model.unk_token)
    loaders = {}
    outer_samples = None
    for subset in ("fit", "dev", "outer_eval"):
        selected = [samples[i] for i in split[subset]]
        sequences, labels, _ = encode_texts(selected, tokenizer, unk_id, max_length)
        loaders[subset] = make_dataloader(
            sequences, labels, pad_id, batch_size,
            shuffle=(subset == "fit"), seed=split["training_seed"],
        )
        if subset == "outer_eval":
            outer_samples = selected

    model = build_model(emb_model, len(label_names), device=device, **model_kwargs)
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate, weight_decay=weight_decay,
    )
    settings = {
        **model_kwargs, "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "max_length": max_length, "device": str(device),
        "seed": split["seed"], "fold": split["fold"],
        "training_seed": split["training_seed"],
        "checkpoint_selection": "minimum inner-validation NLL",
        "embedding_dim": emb_model.vectors.shape[1],
        "num_classes": len(label_names), "label_names": repr(label_names),
    }
    save_csv(folder / "settings.csv", [
        {"setting": key, "value": value} for key, value in settings.items()
    ])
    save_csv(folder / "split_ids.csv", [
        {"subset": subset, "position": position,
         "sample_id": samples[index]["sample_id"]}
        for subset in ("fit", "dev", "outer_eval")
        for position, index in enumerate(split[subset])
    ])
    best_loss = float("inf")
    best_epoch = None
    history = []
    checkpoint = folder / "model.pt"
    for epoch in range(1, epochs + 1):
        train_result = training_one_epoch(model, loaders["fit"], optimizer, criterion, device)
        validation = evaluate(model, loaders["dev"], criterion, device)
        row = {"epoch": epoch, "train_loss": train_result["loss"],
               "train_accuracy": train_result["accuracy"]}
        row.update({f"validation_{k}": v for k, v in
                    prediction_metrics(validation, label_names).items()})
        if not np.isfinite(row["train_loss"]) or not np.isfinite(validation["loss"]):
            raise ValueError("Loss is NaN or infinite; check the data and learning rate")
        history.append(row)
        save_csv(folder / "history.csv", history)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
        print(f"{folder.parent.name}/{folder.name} | epoch {epoch}/{epochs} | "
              f"loss {row['train_loss']:.4f} | val F1 {row['validation_macro_f1']:.4f}",
              flush=True)

    # Source: tutorial Analysis > Load and assess trained models.
    # Score the outer fold using the checkpoint selected on dev.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    result = evaluate(model, loaders["outer_eval"], criterion, device)
    scores = prediction_metrics(result, label_names)
    # Keep sample order and probabilities for error analysis.
    predictions = []
    offset = 0
    model.eval()
    with torch.no_grad():
        for x, y in loaders["outer_eval"]:
            probabilities = model(x.to(device)).exp().cpu().numpy()
            for gold, probs in zip(y.tolist(), probabilities):
                sample = outer_samples[offset]
                predicted = int(probs.argmax())
                item = {"sample_id": sample["sample_id"], "text": sample["text"],
                        "gold": gold, "predicted": predicted,
                        "correct": int(gold == predicted)}
                item.update({f"prob_{i}": float(p) for i, p in enumerate(probs)})
                predictions.append(item)
                offset += 1
    save_csv(folder / "predictions.csv", predictions)
    summary = {"seed": split["seed"], "fold": split["fold"], **scores,
               "best_epoch": best_epoch, "seconds": time.perf_counter() - started}
    save_csv(folder / "metrics.csv", [summary])
    plot_history(history, folder / "learning_curve.png", folder.parent.name)
    matrix = confusion_matrix(
        [p["gold"] for p in predictions], [p["predicted"] for p in predictions],
        labels=list(range(len(label_names))),
    )
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(label_names)), yticks=range(len(label_names)),
             xticklabels=label_names, yticklabels=label_names,
             xlabel="Predicted", ylabel="Gold")
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(folder / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return summary


def summarize_results(rows, output_dir):
    """Summarize fold scores and pair LSTM with pooling results.

    The fold standard deviation describes variation; it is not a confidence
    interval. See tutorial Training > Cross validation."""
    folder = Path(output_dir)
    systems = list(dict.fromkeys(row["system"] for row in rows))
    summary = []
    for system in systems:
        group = [r for r in rows if r["system"] == system]
        item = {"system": system, "n_scores": len(group)}
        for metric in ("macro_f1", "accuracy", "stance_f1"):
            values = [r[metric] for r in group if metric in r]
            if values:
                item[f"{metric}_mean"] = float(np.mean(values))
                item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(item)
    save_csv(folder / "summary.csv", summary)
    fig, axis = plt.subplots(figsize=(9, 4))
    for i, system in enumerate(systems):
        values = [r["macro_f1"] for r in rows if r["system"] == system]
        axis.scatter([i] * len(values), values, alpha=0.7)
        axis.scatter(i, np.mean(values), marker="_", s=250, color="black")
    axis.set(xticks=range(len(systems)), xticklabels=systems,
             ylabel="Outer-fold macro-F1", ylim=(0, 1))
    axis.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    fig.savefig(folder / "performance.png", dpi=160)
    plt.close(fig)
    paired = []
    for status in ("frozen", "finetuned"):
        a, b = f"pooling_{status}", f"lstm_{status}"
        scores_a = {(r["seed"], r["fold"]): r["macro_f1"] for r in rows if r["system"] == a}
        scores_b = {(r["seed"], r["fold"]): r["macro_f1"] for r in rows if r["system"] == b}
        for seed, fold in sorted(scores_a.keys() & scores_b.keys()):
            paired.append({"comparison": f"{b} minus {a}", "seed": seed,
                           "fold": fold, "macro_f1_difference":
                           scores_b[seed, fold] - scores_a[seed, fold]})
    save_csv(folder / "paired_differences.csv", paired)
    return summary


def run_experiments(output_dir=None, epochs=20, batch_size=64,
                    learning_rate=1e-3, weight_decay=1e-4, max_length=None,
                    device="cpu", seeds=SEEDS, num_folds=NUM_FOLDS,
                    inner_val_fraction=INNER_VAL_FRACTION):
    """Run all four systems on shared folds without using the test set.

    Based on tutorial Training > Cross validation (cells 135 and 137)."""
    if output_dir is None:
        output_dir = RESULTS_DIR / time.strftime("core_%Y%m%d_%H%M%S")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {root}")
    source_splits, label_names = load_data()
    samples = source_splits["train"]
    splits = create_splits(samples, seeds, num_folds, inner_val_fraction)
    embeddings = load_embeddings()
    tokenizer = create_tokenizer(embeddings)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for system, settings in SYSTEMS.items():
        for split in splits:
            folder = root / system / f"seed-{split['seed']}_fold-{split['fold']}"
            scores = run_fold(
                samples, split, embeddings, tokenizer, label_names, settings, folder,
                epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                weight_decay=weight_decay, max_length=max_length, device=device,
            )
            rows.append({"system": system, **scores})
            save_csv(root / "fold_metrics.csv", rows)
    summarize_results(rows, root)
    print(f"Results saved to {root}", flush=True)
    return rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    run_experiments(**vars(parser.parse_args()))

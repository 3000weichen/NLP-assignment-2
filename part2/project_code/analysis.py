"""Analyze saved outer-fold predictions without retraining.

Source: tutorial Experiment > Analysis > Load and assess trained models.
Length, OOV and negation features are project choices. Group summaries are
exploratory: models share samples, and CV folds are not independent trials.
"""

import ast
import itertools
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .embeddings import create_tokenizer, load_embeddings

NEGATION_WORDS = {"not", "n't", "no", "never", "neither", "nor", "nobody", "nothing", "nowhere"}


def get_bayesian_helpers():
    """Load the optional helpers from tutorial Experiment > Analysis."""
    from ..analysis import fit_bda_model, posterior_to_arviz
    return fit_bda_model, posterior_to_arviz


def read_predictions(results_dir):
    """Read completed folds and check labels, IDs, and comparable samples."""
    frames = []
    label_names = None
    for path in sorted(Path(results_dir).glob("*/seed-*_fold-*/predictions.csv")):
        settings = pd.read_csv(path.parent / "settings.csv", keep_default_na=False)
        settings = dict(zip(settings["setting"], settings["value"]))
        names = ast.literal_eval(settings["label_names"])
        if label_names is not None and names != label_names:
            raise ValueError("Label mappings differ between runs")
        label_names = names
        if settings.get("max_length", "None") not in {"None", ""}:
            raise ValueError("Truncated runs need features computed on the same truncated tokens")
        frame = pd.read_csv(path, keep_default_na=False)
        frame["system"] = path.parent.parent.name
        frame["seed"] = int(settings["seed"])
        frame["fold"] = int(settings["fold"])
        frame["correct"] = (frame["gold"] == frame["predicted"]).astype(int)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No fold predictions found in the results directory")
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["system", "seed", "sample_id"]).any():
        raise ValueError("A sample occurs more than once per system and CV seed")
    for column in ("gold", "predicted"):
        if not data[column].isin(range(len(label_names))).all():
            raise ValueError(f"Invalid label in {column}")
    items = data.groupby("sample_id")
    if (items["text"].nunique() > 1).any() or (items["gold"].nunique() > 1).any():
        raise ValueError("Text or labels disagree for the same sample ID")
    reference = None
    for _, group in data.groupby("system"):
        keys = set(zip(group["seed"], group["fold"], group["sample_id"]))
        if reference is not None and keys != reference:
            raise ValueError("Systems have different evaluation samples or unfinished folds")
        reference = keys
    return data, label_names


def text_features(data, emb_model, tokenizer):
    """Compute features once per text, using the training tokenizer."""
    records = []
    counts = {}
    for sample in data.drop_duplicates("sample_id").itertuples():
        tokens = tokenizer.tokens(sample.text)
        unknown = [token for token in tokens if token not in emb_model.vocab]
        for token in unknown:
            counts[token] = counts.get(token, 0) + 1
        length = len(tokens)
        rate = len(unknown) / length if length else 0.0
        records.append({
            "sample_id": sample.sample_id, "text": sample.text,
            "gold": int(sample.gold), "n_tokens": length,
            "oov_count": len(unknown), "oov_rate": rate,
            "has_negation": int(any(t in NEGATION_WORDS for t in tokens)),
            "length_group": "0-15" if length <= 15 else "16-30" if length <= 30 else "31+",
            "oov_group": "0%" if rate == 0 else "0-10%" if rate <= 0.1 else ">10%",
        })
    features = pd.DataFrame(records)
    oov = pd.DataFrame(sorted(counts.items(), key=lambda x: (-x[1], x[0])),
                       columns=["token", "count"])
    return features, oov


def marker_sensitivity(features, oov, tokenizer):
    """Recalculate OOV without the constant SemST token; keep model input unchanged."""
    features = features.copy()
    marker_is_oov = "semst" in set(oov.token)
    marker_counts = []
    for row in features.itertuples():
        tokens = tokenizer.tokens(row.text)
        if len(tokens) != row.n_tokens:
            raise ValueError("Cached features use a different tokenizer")
        marker_counts.append(tokens.count("semst"))
    features["semst_count"] = marker_counts
    features["tokens_without_semst"] = features.n_tokens - features.semst_count
    features["oov_without_semst"] = features.oov_count - features.semst_count * marker_is_oov
    denominator = features.tokens_without_semst.replace(0, np.nan)
    features["oov_rate_without_semst"] = features.oov_without_semst / denominator
    features["oov_group_without_semst"] = features.oov_rate_without_semst.map(
        lambda rate: "empty" if pd.isna(rate) else "0%" if rate == 0
        else "0-10%" if rate <= 0.1 else ">10%"
    )
    return features


def class_feature_metrics(data, label_names):
    """Compare recall within each true class to expose class-composition effects."""
    rows = []
    for feature in ("length_group", "oov_group_without_semst", "has_negation"):
        for (system, gold, value), group in data.groupby(["system", "gold", feature]):
            rows.append({"system": system, "label": label_names[gold],
                         "feature": feature, "group": str(value), "n": len(group),
                         "recall": group.correct.mean()})
    return pd.DataFrame(rows)


def favor_cases(data, label_names):
    """Put all systems' predictions side by side for the same favor examples."""
    if "favor" not in label_names:
        return pd.DataFrame()
    subset = data[data.gold == label_names.index("favor")]
    keys = ["seed", "fold", "sample_id"]
    meta = subset.drop_duplicates(keys)[keys + [
        "text", "n_tokens", "oov_rate_without_semst", "has_negation",
    ]].set_index(keys)
    predicted = subset.pivot(index=keys, columns="system", values="predicted")
    predicted = predicted.apply(lambda col: col.map(dict(enumerate(label_names))))
    correct = subset.groupby(keys).correct.sum().rename("n_systems_correct")
    return meta.join(predicted).join(correct).reset_index().sort_values(
        ["n_systems_correct", "sample_id"], ascending=[False, True]
    )


def grouped_metrics(data, label_names):
    """Save pooled and per-fold scores for fixed feature groups."""
    pooled, folds = [], []
    for feature in ("length_group", "oov_group", "oov_group_without_semst", "has_negation"):
        for (system, value), group in data.groupby(["system", feature]):
            pooled.append({"system": system, "feature": feature, "group": str(value),
                           "n_predictions": len(group),
                           "n_samples": group["sample_id"].nunique(),
                           "accuracy": accuracy_score(group.gold, group.predicted),
                           "macro_f1": f1_score(group.gold, group.predicted,
                                                labels=range(len(label_names)),
                                                average="macro", zero_division=0),
                           **{f"gold_{name}": int((group.gold == i).sum())
                              for i, name in enumerate(label_names)}})
        for (system, seed, fold, value), group in data.groupby(["system", "seed", "fold", feature]):
            folds.append({"system": system, "seed": seed, "fold": fold,
                          "feature": feature, "group": str(value), "n": len(group),
                          "accuracy": accuracy_score(group.gold, group.predicted)})
    return pd.DataFrame(pooled), pd.DataFrame(folds)


def class_metrics(data, label_names):
    """Report each class separately, including the minority favor class."""
    rows = []
    for system, group in data.groupby("system"):
        for label, name in enumerate(label_names):
            gold = group.gold == label
            predicted = group.predicted == label
            tp = int((gold & predicted).sum())
            support, predicted_count = int(gold.sum()), int(predicted.sum())
            precision = tp / predicted_count if predicted_count else 0.0
            recall = tp / support if support else 0.0
            rows.append({"system": system, "label": name, "support": support,
                         "predicted_count": predicted_count, "correct": tp,
                         "precision": precision, "recall": recall,
                         "f1": 2 * precision * recall / (precision + recall)
                         if precision + recall else 0.0})
    return pd.DataFrame(rows)


def disagreement_cases(data):
    """Pair predictions on exactly the same seed, fold, and sample."""
    keys = ["seed", "fold", "sample_id"]
    cases = []
    for a, b in itertools.combinations(sorted(data.system.unique()), 2):
        left = data[data.system == a]
        right = data[data.system == b][keys + ["predicted", "correct"]]
        paired = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one")
        paired = paired[paired.predicted_a != paired.predicted_b].copy()
        paired["system_a"], paired["system_b"] = a, b
        cases.append(paired)
    return pd.concat(cases, ignore_index=True) if cases else pd.DataFrame()


def plot_groups(groups, classes, output):
    """Plot descriptive group accuracy and per-class recall."""
    orders = {"length_group": ["0-15", "16-30", "31+"],
              "oov_group": ["0%", "0-10%", ">10%"],
              "oov_group_without_semst": ["0%", "0-10%", ">10%"],
              "has_negation": ["0", "1"]}
    for feature, order in orders.items():
        subset = groups[groups.feature == feature]
        table = subset.pivot(index="group", columns="system", values="accuracy").reindex(order)
        axis = table.plot.bar(figsize=(10, 5), ylim=(0, 1), rot=0)
        axis.set(xlabel=feature, ylabel="Accuracy", title="Outer-fold predictions")
        axis.legend(fontsize=8)
        axis.figure.tight_layout()
        axis.figure.savefig(output / f"{feature}_accuracy.png", dpi=160)
        plt.close(axis.figure)
    table = classes.pivot(index="label", columns="system", values="recall")
    axis = table.plot.bar(figsize=(10, 5), ylim=(0, 1), rot=0)
    axis.set(xlabel="Gold label", ylabel="Recall")
    axis.legend(fontsize=8)
    axis.figure.tight_layout()
    axis.figure.savefig(output / "class_recall.png", dpi=160)
    plt.close(axis.figure)


def analyze(results_dir, reuse_features=False):
    """Add feature tables, group plots, and error examples to a finished run."""
    root = Path(results_dir)
    data, names = read_predictions(root)
    output = root / "analysis"
    output.mkdir(exist_ok=True)
    if reuse_features:
        features = pd.read_csv(output / "text_features.csv", keep_default_na=False)
        oov = pd.read_csv(output / "oov_words.csv", keep_default_na=False)
        current = data[["sample_id", "text", "gold"]].drop_duplicates().sort_values("sample_id")
        cached = features[["sample_id", "text", "gold"]].sort_values("sample_id")
        if not current.reset_index(drop=True).equals(cached.reset_index(drop=True)):
            raise ValueError("Cached features do not match the current predictions")
        tokenizer = create_tokenizer(SimpleNamespace(vocab=None))
    else:
        print("Loading GloVe for vocabulary coverage...", flush=True)
        embeddings = load_embeddings()
        tokenizer = create_tokenizer(embeddings)
        features, oov = text_features(data, embeddings, tokenizer)
    features = marker_sensitivity(features, oov, tokenizer)
    table = data.merge(features.drop(columns=["text", "gold"]), on="sample_id", validate="many_to_one")
    groups, fold_groups = grouped_metrics(table, names)
    classes = class_metrics(table, names)
    cases = disagreement_cases(table)
    for filename, frame in {
        "text_features.csv": features, "prediction_features.csv": table,
        "oov_words.csv": oov, "group_metrics.csv": groups,
        "group_fold_metrics.csv": fold_groups, "class_metrics.csv": classes,
        "disagreements.csv": cases,
        "class_feature_metrics.csv": class_feature_metrics(table, names),
        "favor_cases.csv": favor_cases(table, names),
        "favor_errors.csv": table[(table.gold == names.index("favor")) & (table.correct == 0)]
        if "favor" in names else table.iloc[:0],
    }.items():
        frame.to_csv(output / filename, index=False)
    total = int(features.n_tokens.sum())
    unknown = int(features.oov_count.sum())
    coverage = pd.DataFrame([{
        "n_unique_samples": len(features), "n_tokens": total, "n_oov_tokens": unknown,
        "token_coverage": 1 - unknown / total if total else np.nan,
        "mean_text_oov_rate": features.oov_rate.mean(),
        "negation_samples": int(features.has_negation.sum()),
    }])
    coverage.to_csv(output / "coverage.csv", index=False)
    adjusted_total = int(features.tokens_without_semst.sum())
    adjusted_oov = int(features.oov_without_semst.sum())
    pd.DataFrame([
        {"version": "original", "n_tokens": total, "n_oov": unknown,
         "token_coverage": 1 - unknown / total if total else np.nan},
        {"version": "without_semst", "n_tokens": adjusted_total, "n_oov": adjusted_oov,
         "token_coverage": 1 - adjusted_oov / adjusted_total if adjusted_total else np.nan},
    ]).to_csv(output / "marker_comparison.csv", index=False)
    plot_groups(groups, classes, output)
    (output / "notes.txt").write_text(
        "Analysis uses saved outer-fold predictions, not the final test set.\n"
        "Lengths and OOV include punctuation and use the model tokenizer.\n"
        "Negation is a word-list flag, not a detector of scope or sarcasm.\n"
        "SemST exclusion is a post-hoc sensitivity analysis; model inputs are unchanged.\n"
        "The hashtag symbol remains; only the semst token is excluded in adjusted OOV.\n"
        "Bins are descriptive and missing groups are not zero accuracy.\n"
        "Group class counts may differ; inspect them before interpreting accuracy.\n"
        "Repeated seeds reuse texts; pooled prediction counts are not independent samples.\n"
        "Fold scores are descriptive; no significance claim is made.\n",
        encoding="utf-8",
    )
    print(coverage.to_string(index=False), flush=True)
    print(classes.to_string(index=False), flush=True)
    print(f"Analysis saved to {output}", flush=True)
    return table, groups, classes


def compare_runs(run_dirs, output_dir):
    """Compare completed ordinary/weighted runs on matching seeds and folds.

    Summaries are descriptive; overlapping CV training sets are not independent.
    """
    frames = []
    split_records = {}
    for run_dir in run_dirs:
        root = Path(run_dir)
        if not (root / "summary.csv").is_file():
            raise ValueError(f"Run is unfinished: {root}")
        metrics = pd.read_csv(root / "fold_metrics.csv")
        for row in metrics.itertuples():
            folder = root / row.system / f"seed-{row.seed}_fold-{row.fold}"
            settings_frame = pd.read_csv(folder / "settings.csv", keep_default_na=False)
            settings = dict(zip(settings_frame.setting, settings_frame.value))
            weighted = settings.get("class_weighted", "False") == "True"
            if settings.get("embedding_init", "pretrained") != "pretrained":
                raise ValueError("Compare initialization controls separately")
            if float(settings.get("train_fraction", 1)) != 1:
                raise ValueError("Compare data-size controls separately")
            condition = "weighted" if weighted else "ordinary"
            key = (row.system, row.seed, row.fold)
            saved = pd.read_csv(folder / "split_ids.csv").to_csv(index=False)
            shared = tuple((k, settings.get(k, default)) for k, default in {
                "epochs": "20", "batch_size": "64", "learning_rate": "0.001",
                "weight_decay": "0.0001", "max_length": "None",
                "hidden_size": "64", "hidden_size1": "50", "hidden_size2": "50",
                "dropout": "0.1", "word_dropout": "0.0", "pooling": "mean",
                "training_seed": "", "label_names": "", "embedding_dim": "50",
                "encoder_type": "", "freeze_embeddings": "",
            }.items())
            record = (saved, shared)
            if key in split_records and split_records[key] != record:
                raise ValueError(f"Settings or splits differ: {key}")
            split_records[key] = record
            item = row._asdict()
            item["condition"] = condition
            frames.append(item)
    data = pd.DataFrame(frames)
    keys = ["system", "seed", "fold"]
    if data.duplicated(["condition"] + keys).any():
        raise ValueError("Duplicate runs; provide each condition/seed/fold once")
    ordinary = data[data.condition == "ordinary"]
    weighted = data[data.condition == "weighted"]
    paired = ordinary.merge(weighted, on=keys, suffixes=("_ordinary", "_weighted"),
                            validate="one_to_one", how="outer", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Wait for all ordinary and weighted folds before comparison")
    metrics = ["macro_f1", "accuracy", "stance_f1"]
    for metric in metrics:
        paired[f"{metric}_difference"] = paired[f"{metric}_weighted"] - paired[f"{metric}_ordinary"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "all_fold_metrics.csv", index=False)
    paired.drop(columns="_merge").to_csv(output / "paired_loss_differences.csv", index=False)
    per_seed = data.groupby(["condition", "system", "seed"])[metrics].mean().reset_index()
    per_seed.to_csv(output / "seed_means.csv", index=False)
    summary = data.groupby(["condition", "system"])[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(output / "summary.csv")
    difference_summary = paired.groupby("system")[[f"{m}_difference" for m in metrics]].agg(["mean", "std", "min", "max"])
    difference_summary.to_csv(output / "difference_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    systems = list(dict.fromkeys(data.system))
    for i, system in enumerate(systems):
        for condition, offset, color in [("ordinary", -0.12, "#2563eb"), ("weighted", 0.12, "#d97706")]:
            group = data[(data.system == system) & (data.condition == condition)]
            axes[0].scatter([i + offset] * len(group), group.macro_f1, color=color,
                            alpha=0.6, label=condition if i == 0 else None)
        values = paired[paired.system == system].macro_f1_difference
        axes[1].scatter([i] * len(values), values, alpha=0.7)
    axes[0].set(ylabel="Outer-fold macro-F1", ylim=(0, 1))
    axes[0].legend()
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(ylabel="Weighted minus ordinary macro-F1")
    for axis in axes:
        axis.set_xticks(range(len(systems)), systems, rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output / "loss_comparison.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(), flush=True)
    print(f"Saved comparison to {output}", flush=True)
    return data, paired


def compare_embeddings(run_dirs, output_dir):
    """Pair pretrained and random trainable embeddings on the same folds."""
    frames, records = [], {}
    selected = {"pooling_finetuned", "lstm_finetuned"}
    for run_dir in run_dirs:
        root = Path(run_dir)
        if not (root / "summary.csv").is_file():
            raise ValueError(f"Run is unfinished: {root}")
        metrics = pd.read_csv(root / "fold_metrics.csv")
        for row in metrics[metrics.system.isin(selected)].itertuples():
            folder = root / row.system / f"seed-{row.seed}_fold-{row.fold}"
            frame = pd.read_csv(folder / "settings.csv", keep_default_na=False)
            settings = dict(zip(frame.setting, frame.value))
            condition = settings.get("embedding_init", "pretrained")
            if condition not in {"pretrained", "random"}:
                raise ValueError("Unknown embedding initialization")
            if settings.get("freeze_embeddings") != "False":
                raise ValueError("This comparison requires trainable embeddings")
            # Settings other than initialization must match within each pair.
            shared = tuple((key, settings.get(key, default)) for key, default in {
                "epochs": "20", "batch_size": "64", "learning_rate": "0.001",
                "weight_decay": "0.0001", "max_length": "None",
                "hidden_size": "64", "hidden_size1": "50", "hidden_size2": "50",
                "dropout": "0.1", "word_dropout": "0.0", "pooling": "mean",
                "training_seed": "", "label_names": "", "embedding_dim": "50",
                "encoder_type": "", "class_weighted": "False",
                "train_fraction": "1.0", "class_weights": "None",
            }.items())
            split_ids = pd.read_csv(folder / "split_ids.csv").to_csv(index=False)
            key = (row.system, row.seed, row.fold)
            record = (shared, split_ids)
            if key in records and records[key] != record:
                raise ValueError(f"Settings or sample splits differ: {key}")
            records[key] = record
            frames.append({**row._asdict(), "initialization": condition})
    data = pd.DataFrame(frames)
    if data.empty:
        raise ValueError("No trainable pooling/LSTM results found")
    keys = ["system", "seed", "fold"]
    if data.duplicated(["initialization"] + keys).any():
        raise ValueError("Duplicate initialization/system/seed/fold result")
    pretrained = data[data.initialization == "pretrained"]
    random = data[data.initialization == "random"]
    paired = pretrained.merge(random, on=keys, suffixes=("_pretrained", "_random"),
                              how="outer", indicator=True, validate="one_to_one")
    if not paired._merge.eq("both").all():
        raise ValueError("Pretrained/random pairs are incomplete")
    metrics = ["macro_f1", "accuracy", "stance_f1"]
    for metric in metrics:
        paired[f"{metric}_difference"] = paired[f"{metric}_pretrained"] - paired[f"{metric}_random"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "fold_metrics.csv", index=False)
    paired.drop(columns="_merge").to_csv(output / "paired_initialization_differences.csv", index=False)
    summary = data.groupby(["initialization", "system"])[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.to_csv(output / "summary.csv")
    data.groupby(["initialization", "system", "seed"])[metrics].mean().to_csv(output / "seed_means.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, system in zip(axes, sorted(selected)):
        values = paired[paired.system == system]
        if values.empty:
            continue
        for row in values.itertuples():
            axis.plot([0, 1], [row.macro_f1_random, row.macro_f1_pretrained],
                      color="gray", alpha=0.45, marker="o")
        axis.set(xticks=[0, 1], xticklabels=["Random", "GloVe"], ylim=(0, 1),
                 ylabel="Outer-fold macro-F1", title=system)
    fig.tight_layout()
    fig.savefig(output / "initialization_comparison.png", dpi=160)
    plt.close(fig)
    print(summary.to_string(), flush=True)
    print(f"Saved embedding comparison to {output}", flush=True)
    return data, paired


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze a completed Project 2 run")
    parser.add_argument("results_dir", type=Path, nargs="?")
    parser.add_argument("--compare-runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-factor", choices=["loss", "embedding"], default="loss")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    if args.compare_runs:
        if args.output is None:
            parser.error("--compare-runs requires --output")
        compare = compare_embeddings if args.compare_factor == "embedding" else compare_runs
        compare(args.compare_runs, args.output)
    elif args.results_dir is not None:
        analyze(args.results_dir, reuse_features=args.reuse_features)
    else:
        parser.error("Provide results_dir or --compare-runs")

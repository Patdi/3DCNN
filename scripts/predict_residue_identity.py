#!/usr/bin/env python3
"""Run residue-identity inference with a trained voxel CNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.cnn3d import CNN3DConfig, VoxelCNN3D
from protein_constants import LABEL_RES_DICT, LETTER1_3_DICT, RES_LABEL_DICT

AMINO_ACID_PROPERTIES: dict[str, str] = {
    # Positively Charged
    "Arg": "Positively Charged",
    "His": "Positively Charged",
    "Lys": "Positively Charged",
    # Negatively Charged
    "Asp": "Negatively Charged",
    "Glu": "Negatively Charged",
    # Polar Uncharged
    "Ser": "Polar Uncharged",
    "Thr": "Polar Uncharged",
    "Asn": "Polar Uncharged",
    "Gln": "Polar Uncharged",
    # Special Cases
    "Cys": "Special Case",
    "Sec": "Special Case",
    "Gly": "Special Case",
    "Pro": "Special Case",
    # Hydrophobic
    "Ala": "Hydrophobic",
    "Val": "Hydrophobic",
    "Ile": "Hydrophobic",
    "Leu": "Hydrophobic",
    "Met": "Hydrophobic",
    "Phe": "Hydrophobic",
    "Tyr": "Hydrophobic",
    "Trp": "Hydrophobic",
}

CANONICAL_AA_LABELS = [
    "Ala",
    "Arg",
    "Asn",
    "Asp",
    "Cys",
    "Gln",
    "Glu",
    "Gly",
    "His",
    "Ile",
    "Leu",
    "Lys",
    "Met",
    "Phe",
    "Pro",
    "Ser",
    "Thr",
    "Trp",
    "Tyr",
    "Val",
]

CANONICAL_AA_TYPES = [
    "Positively Charged",
    "Negatively Charged",
    "Polar Uncharged",
    "Special Case",
    "Hydrophobic",
    "Unknown",
]


class PredictionManifestDataset(Dataset):
    """Manifest dataset compatible with train_voxel_cnn.py conventions."""

    def __init__(self, manifest_path: Path, mean: np.ndarray, std: np.ndarray, task: str):
        self.manifest_path = manifest_path
        self.manifest_dir = manifest_path.parent.resolve()
        self.task = task
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-6)

        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            self.rows = list(reader)

        if not self.rows:
            raise ValueError(f"Manifest is empty: {manifest_path}")

        first = self.rows[0]
        if "sample_path" in first:
            self.path_key = "sample_path"
        elif "path" in first:
            self.path_key = "path"
        else:
            raise ValueError("Manifest must contain 'sample_path' or 'path'")

        self.has_sample_index = "sample_index" in first
        self.schema = "sharded" if self.has_sample_index else "per_example"
        self._shard_cache: dict[Path, np.ndarray | np.lib.npyio.NpzFile] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.manifest_dir / path

    @staticmethod
    def _to_channels_first(sample: np.ndarray) -> np.ndarray:
        if sample.ndim == 4 and sample.shape[-1] <= 8 and sample.shape[0] > 8:
            return np.transpose(sample, (3, 0, 1, 2))
        return sample

    @staticmethod
    def _maybe_parse_label(label_value: Any) -> int | None:
        if label_value is None:
            return None
        text = str(label_value).strip()
        if text == "":
            return None

        try:
            return int(text)
        except ValueError:
            upper_text = text.upper()
            if upper_text in RES_LABEL_DICT:
                return RES_LABEL_DICT[upper_text]
            if len(upper_text) == 1 and upper_text in LETTER1_3_DICT:
                return RES_LABEL_DICT[LETTER1_3_DICT[upper_text]]
            return None

    def _load_shard(self, path: Path) -> np.ndarray | np.lib.npyio.NpzFile:
        if path not in self._shard_cache:
            # Intentional shard caching (same behavior class as training dataset).
            self._shard_cache[path] = np.load(path, allow_pickle=False)
        return self._shard_cache[path]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        raw_sample_path = row[self.path_key]
        sample_path = self._resolve_path(raw_sample_path)

        manifest_label = self._maybe_parse_label(row.get("label"))
        npz_label: int | None = None

        if self.schema == "sharded" and row.get("sample_index", "").strip() != "":
            sample_index = int(row["sample_index"])
            sample_obj = self._load_shard(sample_path)
            sample = sample_obj[sample_index].astype(np.float32)
        else:
            with np.load(sample_path, allow_pickle=False) as sample_obj:
                if "x" not in sample_obj:
                    raise ValueError(f"Per-example file missing 'x': {sample_path}")
                sample = sample_obj["x"].astype(np.float32)
                if "y" in sample_obj:
                    npz_label = int(np.asarray(sample_obj["y"]).item())

        if manifest_label is not None and npz_label is not None and manifest_label != npz_label:
            raise ValueError(
                f"Label mismatch for {raw_sample_path}: manifest={manifest_label} npz={npz_label}"
            )

        sample = (sample - self.mean) / self.std
        sample = self._to_channels_first(sample)

        if sample.ndim == 3:
            sample = sample[np.newaxis, ...]
        elif sample.ndim != 4:
            raise ValueError(f"Expected 3D or 4D sample, got shape {sample.shape} for {raw_sample_path}")

        actual_idx = manifest_label if manifest_label is not None else npz_label

        return {
            "x": torch.from_numpy(sample),
            "actual_idx": actual_idx,
            "row": row,
            "raw_sample_path": raw_sample_path,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict residue identity from voxel manifest")
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)

    parser.add_argument("--task", choices=["residue_identity"], default="residue_identity")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--topk-values", default="1,3,5")
    parser.add_argument("--output-probs", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--aa-confusion-csv", type=Path, default=None)
    parser.add_argument("--aa-type-confusion-csv", type=Path, default=None)
    parser.add_argument("--aa-confusion-png", type=Path, default=None)
    parser.add_argument("--aa-type-confusion-png", type=Path, default=None)
    parser.add_argument("--per-class-metrics-csv", type=Path, default=None)
    parser.add_argument("--confusions-csv", type=Path, default=None)
    parser.add_argument("--confidence-hist-png", type=Path, default=None)
    parser.add_argument("--calibration-png", type=Path, default=None)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--max-confusions", type=int, default=50)
    parser.add_argument("--aa-support-csv", type=Path, default=None)
    parser.add_argument("--aa-support-png", type=Path, default=None)
    parser.add_argument("--aa-type-accuracy-csv", type=Path, default=None)
    parser.add_argument("--aa-type-accuracy-png", type=Path, default=None)
    parser.add_argument("--entropy-summary-csv", type=Path, default=None)
    parser.add_argument("--entropy-hist-png", type=Path, default=None)
    parser.add_argument("--save-embeddings-npz", type=Path, default=None)
    parser.add_argument("--embedding-plot-png", type=Path, default=None)
    parser.add_argument("--embedding-method", choices=["none", "pca", "umap"], default="none")
    parser.add_argument("--embedding-max-samples", type=int, default=5000)
    parser.add_argument(
        "--embedding-color-by",
        choices=["actual", "predicted", "actual_type", "predicted_type"],
        default="actual_type",
    )
    parser.add_argument("--embedding-layer", choices=["penultimate", "logits"], default="penultimate")
    parser.add_argument(
        "--normalize-confusion",
        choices=["none", "row", "column", "all"],
        default="row",
    )
    return parser.parse_args()


def parse_topk_values(s: str) -> list[int]:
    ks: list[int] = []
    for chunk in s.split(","):
        text = chunk.strip()
        if not text:
            continue
        value = int(text)
        if value <= 0:
            raise ValueError(f"Top-k values must be positive integers, got: {value}")
        ks.append(value)
    if not ks:
        raise ValueError("--topk-values produced no valid k values")
    return sorted(set(ks))


def compute_topk_hits(actual_idx: int, topk_indices: np.ndarray, ks: list[int]) -> dict[int, int]:
    hits: dict[int, int] = {}
    for k in ks:
        hits[k] = int(actual_idx in topk_indices[:k])
    return hits


def compute_per_class_metrics(
    actual_labels: list[str],
    predicted_labels: list[str],
    class_labels: list[str],
) -> tuple[list[dict[str, float | int | str]], dict[str, float], dict[str, float]]:
    counts = {
        label: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for label in class_labels
    }

    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual not in counts:
            continue
        counts[actual]["support"] += 1
        if actual == predicted:
            counts[actual]["tp"] += 1
        else:
            counts[actual]["fn"] += 1
            if predicted in counts:
                counts[predicted]["fp"] += 1

    rows: list[dict[str, float | int | str]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    supports: list[int] = []

    for label in class_labels:
        tp = int(counts[label]["tp"])
        fp = int(counts[label]["fp"])
        fn = int(counts[label]["fn"])
        support = int(counts[label]["support"])
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                "amino_acid": label,
                "support": support,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    total_support = float(sum(supports))
    macro = {
        "precision": float(np.mean(precisions)) if precisions else 0.0,
        "recall": float(np.mean(recalls)) if recalls else 0.0,
        "f1": float(np.mean(f1s)) if f1s else 0.0,
    }
    if total_support > 0:
        weighted = {
            "precision": float(np.average(precisions, weights=supports)),
            "recall": float(np.average(recalls, weights=supports)),
            "f1": float(np.average(f1s, weights=supports)),
        }
    else:
        weighted = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return rows, macro, weighted


def write_per_class_metrics_csv(
    metrics_rows: list[dict[str, float | int | str]],
    macro: dict[str, float],
    weighted: dict[str, float],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "amino_acid",
        "support",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)
        writer.writerow(
            {
                "amino_acid": "__macro_avg__",
                "support": "",
                "true_positive": "",
                "false_positive": "",
                "false_negative": "",
                "precision": f"{macro['precision']:.6f}",
                "recall": f"{macro['recall']:.6f}",
                "f1": f"{macro['f1']:.6f}",
            }
        )
        writer.writerow(
            {
                "amino_acid": "__weighted_avg__",
                "support": "",
                "true_positive": "",
                "false_positive": "",
                "false_negative": "",
                "precision": f"{weighted['precision']:.6f}",
                "recall": f"{weighted['recall']:.6f}",
                "f1": f"{weighted['f1']:.6f}",
            }
        )


def compute_common_confusions(
    actual_labels: list[str],
    predicted_labels: list[str],
) -> list[tuple[str, str, int]]:
    counter: Counter[tuple[str, str]] = Counter()
    for actual, predicted in zip(actual_labels, predicted_labels):
        if actual != predicted:
            counter[(actual, predicted)] += 1
    return [(a, p, c) for (a, p), c in counter.most_common()]


def write_common_confusions_csv(
    confusions: list[tuple[str, str, int]],
    out_path: Path,
    max_rows: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "actual", "predicted", "count"])
        writer.writeheader()
        for rank, (actual, predicted, count) in enumerate(confusions[:max_rows], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                }
            )


def save_confidence_histogram(
    confidences: list[float],
    correct_flags: list[bool],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct_flags, dtype=bool)

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(conf, bins=bins, alpha=0.35, label="All", color="tab:blue")
    ax.hist(conf[corr], bins=bins, alpha=0.45, label="Correct", color="tab:green")
    ax.hist(conf[~corr], bins=bins, alpha=0.45, label="Incorrect", color="tab:red")
    ax.set_xlabel("confidence")
    ax.set_ylabel("count")
    ax.set_title("Prediction Confidence Histogram")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_calibration_bins(
    confidences: list[float],
    correct_flags: list[bool],
    n_bins: int,
) -> list[dict[str, float | int]]:
    if n_bins <= 0:
        raise ValueError(f"--calibration-bins must be > 0, got {n_bins}")

    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correct_flags, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(conf, edges[1:-1], right=False)

    stats: list[dict[str, float | int]] = []
    for i in range(n_bins):
        mask = bin_ids == i
        count = int(np.sum(mask))
        if count > 0:
            mean_conf = float(np.mean(conf[mask]))
            emp_acc = float(np.mean(corr[mask]))
        else:
            mean_conf = 0.0
            emp_acc = 0.0
        stats.append(
            {
                "bin_lower": float(edges[i]),
                "bin_upper": float(edges[i + 1]),
                "count": count,
                "mean_confidence": mean_conf,
                "empirical_accuracy": emp_acc,
            }
        )
    return stats


def save_calibration_plot(bin_stats: list[dict[str, float | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [float(stat["mean_confidence"]) for stat in bin_stats if int(stat["count"]) > 0]
    ys = [float(stat["empirical_accuracy"]) for stat in bin_stats if int(stat["count"]) > 0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="Perfect calibration")
    if xs:
        ax.plot(xs, ys, marker="o", color="tab:blue", label="Model")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Calibration Plot")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_normalization(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = np.load(path)
    if "train_mean" not in stats or "train_std" not in stats:
        raise ValueError("Normalization file must contain train_mean and train_std")
    return stats["train_mean"], stats["train_std"]


def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int | None,
    task: str,
) -> tuple[torch.nn.Module, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing model_state")

    ckpt_config = checkpoint.get("config")
    if ckpt_config is None:
        if num_classes is None:
            raise ValueError("--num-classes is required when checkpoint config is unavailable")
        out_features = checkpoint["model_state"]["head.3.weight"].shape[0]
        resolved_classes = num_classes if num_classes is not None else out_features
        config = CNN3DConfig(in_channels=1, num_classes=resolved_classes, task=task)
    else:
        config_dict = dict(ckpt_config)
        config_dict["task"] = task
        if num_classes is not None:
            config_dict["num_classes"] = num_classes
        config = CNN3DConfig(**config_dict)

    model = VoxelCNN3D(config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, int(config.num_classes)


def idx_to_residue_map() -> dict[int, str]:
    return {idx: canonicalize_aa_label(label) for idx, label in LABEL_RES_DICT.items()}


def canonical_aa_labels() -> list[str]:
    return list(CANONICAL_AA_LABELS)


def canonicalize_aa_label(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()
    if text == "":
        return ""

    upper_text = text.upper()
    if len(upper_text) == 1 and upper_text in LETTER1_3_DICT:
        upper_text = LETTER1_3_DICT[upper_text]
    return upper_text[:1] + upper_text[1:].lower()


def aa_type(label: str) -> str:
    canonical = canonicalize_aa_label(label)
    if canonical == "":
        return "Unknown"
    return AMINO_ACID_PROPERTIES.get(canonical, "Unknown")


def compute_support_per_aa(actual_labels: list[str], aa_labels: list[str]) -> list[dict[str, int | str]]:
    counts = {label: 0 for label in aa_labels}
    for label in actual_labels:
        canonical = canonicalize_aa_label(label)
        if canonical in counts:
            counts[canonical] += 1
    return [{"amino_acid": label, "support": counts[label]} for label in aa_labels]


def write_support_csv(rows: list[dict[str, int | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["amino_acid", "support"])
        writer.writeheader()
        writer.writerows(rows)


def save_support_bar_chart(rows: list[dict[str, int | str]], out_path: Path) -> None:
    labels = [str(row["amino_acid"]) for row in rows]
    values = [int(row["support"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values, color="tab:blue")
    ax.set_xlabel("amino acid")
    ax.set_ylabel("count")
    ax.set_title("Test Support per Amino Acid")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_accuracy_by_type(
    actual_types: list[str],
    predicted_types: list[str],
    type_order: list[str],
) -> list[dict[str, int | float | str]]:
    stats = {t: {"support": 0, "correct": 0} for t in type_order}
    for actual, predicted in zip(actual_types, predicted_types):
        actual_t = actual if actual in stats else "Unknown"
        predicted_t = predicted if predicted in stats else "Unknown"
        stats[actual_t]["support"] += 1
        stats[actual_t]["correct"] += int(actual_t == predicted_t)
    rows: list[dict[str, int | float | str]] = []
    for t in type_order:
        support = int(stats[t]["support"])
        correct = int(stats[t]["correct"])
        rows.append(
            {
                "amino_acid_type": t,
                "support": support,
                "correct": correct,
                "accuracy": (correct / support) if support > 0 else 0.0,
            }
        )
    return rows


def write_type_accuracy_csv(rows: list[dict[str, int | float | str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["amino_acid_type", "support", "correct", "accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_type_accuracy_bar_chart(rows: list[dict[str, int | float | str]], out_path: Path) -> None:
    labels = [str(row["amino_acid_type"]) for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    supports = [int(row["support"]) for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, accuracies, color="tab:purple")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("amino acid type")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by Amino Acid Type")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    for bar, support in zip(bars, supports):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(bar.get_height() + 0.02, 1.0),
            f"n={support}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def compute_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    return -np.sum(probs * np.log(probs + eps), axis=1)


def summarize_entropy(
    entropies: list[float],
    confidences: list[float],
    correct_flags: list[bool],
) -> dict[str, float]:
    e = np.asarray(entropies, dtype=np.float64)
    c = np.asarray(confidences, dtype=np.float64)
    f = np.asarray(correct_flags, dtype=bool)
    out: dict[str, float] = {
        "num_samples": float(e.size),
        "mean_entropy": float(np.mean(e)) if e.size else float("nan"),
        "median_entropy": float(np.median(e)) if e.size else float("nan"),
        "std_entropy": float(np.std(e)) if e.size else float("nan"),
        "min_entropy": float(np.min(e)) if e.size else float("nan"),
        "max_entropy": float(np.max(e)) if e.size else float("nan"),
        "mean_confidence": float(np.mean(c)) if c.size else float("nan"),
        "mean_entropy_correct": float(np.mean(e[f])) if np.any(f) else float("nan"),
        "mean_entropy_incorrect": float(np.mean(e[~f])) if np.any(~f) else float("nan"),
        "mean_confidence_correct": float(np.mean(c[f])) if np.any(f) else float("nan"),
        "mean_confidence_incorrect": float(np.mean(c[~f])) if np.any(~f) else float("nan"),
    }
    return out


def write_entropy_summary_csv(summary: dict[str, float], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in summary.items():
            writer.writerow({"metric": metric, "value": value})


def save_entropy_histogram(
    entropies: list[float],
    correct_flags: list[bool],
    out_path: Path,
) -> None:
    entropy_arr = np.asarray(entropies, dtype=np.float64)
    flags = np.asarray(correct_flags, dtype=bool)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(entropy_arr, bins=30, alpha=0.35, color="tab:blue", label="All")
    if entropy_arr.size and flags.size == entropy_arr.size:
        ax.hist(entropy_arr[flags], bins=30, alpha=0.45, color="tab:green", label="Correct")
        ax.hist(entropy_arr[~flags], bins=30, alpha=0.45, color="tab:red", label="Incorrect")
    ax.set_title("Prediction Entropy Histogram")
    ax.set_xlabel("entropy (natural log)")
    ax.set_ylabel("count")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def extract_embeddings(
    model: torch.nn.Module,
    x: torch.Tensor,
    layer: str,
) -> torch.Tensor:
    if layer == "logits":
        return model(x)
    if layer == "penultimate":
        if hasattr(model, "features") and hasattr(model, "classifier"):
            features = model.features(x)
            flattened = features.flatten(1)
            return model.classifier(flattened)
        return model(x)
    raise ValueError(f"Unsupported embedding layer: {layer}")


def reduce_embeddings_pca(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    if embeddings.shape[1] < n_components:
        pad = np.zeros((embeddings.shape[0], n_components - embeddings.shape[1]), dtype=np.float64)
        return np.concatenate([embeddings, pad], axis=1)

    try:
        from sklearn.decomposition import PCA  # type: ignore

        return PCA(n_components=n_components, random_state=0).fit_transform(embeddings)
    except Exception:
        centered = embeddings - np.mean(embeddings, axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:n_components].T
        return centered @ basis


def reduce_embeddings_umap(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray | None:
    try:
        import umap  # type: ignore
    except ImportError:
        return None
    reducer = umap.UMAP(n_components=n_components, random_state=0)
    return reducer.fit_transform(embeddings)


def save_embedding_plot(
    embedding_2d: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
) -> None:
    unique_labels = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in unique_labels:
        mask = np.asarray([item == label for item in labels], dtype=bool)
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=14,
            alpha=0.65,
            label=label,
        )
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(title)
    if len(unique_labels) <= 12:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def init_confusion(labels: list[str]) -> dict[str, dict[str, int]]:
    return {actual: {predicted: 0 for predicted in labels} for actual in labels}


def update_confusion(
    confusion: dict[str, dict[str, int]],
    labels: list[str],
    actual: str,
    predicted: str,
) -> None:
    if actual not in confusion or predicted not in confusion[actual]:
        return
    confusion[actual][predicted] += 1


def write_confusion_csv(
    confusion: dict[str, dict[str, int]],
    labels: list[str],
    out_path: Path,
    header_label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([header_label, *labels])
        for actual in labels:
            writer.writerow([actual, *[confusion[actual][predicted] for predicted in labels]])


def confusion_to_array(confusion: dict[str, dict[str, int]], labels: list[str]) -> np.ndarray:
    return np.asarray(
        [[confusion[actual][predicted] for predicted in labels] for actual in labels],
        dtype=np.int64,
    )


def normalize_confusion_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if mode == "none":
        return matrix
    if mode == "row":
        sums = matrix.sum(axis=1, keepdims=True)
        return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)
    if mode == "column":
        sums = matrix.sum(axis=0, keepdims=True)
        return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)
    if mode == "all":
        total = matrix.sum()
        if total == 0:
            return np.zeros_like(matrix)
        return matrix / total
    raise ValueError(f"Unsupported normalization mode: {mode}")


def save_confusion_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    out_path: Path,
    title: str,
    normalize_mode: str,
) -> None:
    display_matrix = normalize_confusion_matrix(matrix, normalize_mode)
    mode_label = "raw counts" if normalize_mode == "none" else f"{normalize_mode}-normalized"
    annotate = ("type" in title.lower()) or (len(labels) <= 20)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.35), max(4, len(labels) * 0.35)))
    im = ax.imshow(display_matrix, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title} ({mode_label})")

    if annotate:
        for i in range(display_matrix.shape[0]):
            for j in range(display_matrix.shape[1]):
                text = (
                    f"{int(matrix[i, j])}"
                    if normalize_mode == "none"
                    else f"{display_matrix[i, j]:.2f}"
                )
                ax.text(j, i, text, ha="center", va="center", fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def label_to_index(label: str) -> int | None:
    upper_label = label.strip().upper()
    if upper_label in RES_LABEL_DICT:
        return RES_LABEL_DICT[upper_label]
    if len(upper_label) == 1 and upper_label in LETTER1_3_DICT:
        return RES_LABEL_DICT[LETTER1_3_DICT[upper_label]]
    return None


def normalize_residue_label(value: Any, idx_map: dict[int, str]) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return idx_map.get(int(value), str(int(value)))

    text = str(value).strip()
    if text == "":
        return ""

    maybe_int = PredictionManifestDataset._maybe_parse_label(text)
    if maybe_int is not None:
        return idx_map.get(int(maybe_int), canonicalize_aa_label(str(maybe_int)))

    upper_text = text.upper()
    if len(upper_text) == 1 and upper_text in LETTER1_3_DICT:
        return canonicalize_aa_label(LETTER1_3_DICT[upper_text])
    return canonicalize_aa_label(upper_text)


def format_site(row: dict[str, str], sample_path: Path) -> str:
    chain_id = (row.get("chain_id") or "").strip()
    res_no = (row.get("res_no") or "").strip()
    res_name = (row.get("res_name") or "").strip()

    if chain_id and res_no and res_name:
        return f"{chain_id}:{res_no}:{res_name.upper()}"

    example_id = (row.get("example_id") or "").strip()
    if example_id:
        return example_id

    stem = sample_path.stem
    # Very common fallback pattern: <structure>__<chain>:<res_no>:<res_name>
    if "__" in stem:
        return stem.split("__", 1)[1]
    return stem


def infer_structure_name(row: dict[str, str], sample_path: Path) -> str:
    structure_id = (row.get("structure_id") or "").strip()
    if structure_id:
        return structure_id

    stem = sample_path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]

    parent_name = sample_path.parent.name.strip()
    if parent_name:
        return parent_name

    return stem


def collate_prediction_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    xs = torch.stack([item["x"] for item in batch], dim=0)
    return {
        "x": xs,
        "actual_idx": [item["actual_idx"] for item in batch],
        "rows": [item["row"] for item in batch],
        "sample_paths": [item["raw_sample_path"] for item in batch],
    }


def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    idx_map: dict[int, str],
    topk_values: list[int],
    output_probs: bool,
    amp_enabled: bool,
    verbose: bool,
    collect_embedding_data: bool,
    embedding_layer: str,
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
    dict[str, Any],
]:
    rows_out: list[dict[str, str]] = []
    aa_labels = canonical_aa_labels()
    aa_types = list(CANONICAL_AA_TYPES)
    aa_confusion = init_confusion(aa_labels)
    aa_type_confusion = init_confusion(aa_types)

    with_actual = 0
    topk_hit_counts = {k: 0 for k in topk_values}
    eval_actual_labels: list[str] = []
    eval_predicted_labels: list[str] = []
    eval_confidences: list[float] = []
    eval_correct_flags: list[bool] = []
    eval_entropies: list[float] = []
    eval_actual_types: list[str] = []
    eval_predicted_types: list[str] = []
    correct_type = 0
    embedding_vectors: list[np.ndarray] = []
    embedding_structure_names: list[str] = []
    embedding_sites: list[str] = []
    embedding_actual_labels: list[str] = []
    embedding_predicted_labels: list[str] = []
    embedding_actual_types: list[str] = []
    embedding_predicted_types: list[str] = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Predicting", unit="batch"):
            x = batch["x"].to(device, non_blocking=True)

            with autocast(enabled=amp_enabled):
                logits = model(x)
                if collect_embedding_data:
                    embeddings_tensor = (
                        logits if embedding_layer == "logits" else extract_embeddings(model, x, embedding_layer)
                    )
                else:
                    embeddings_tensor = None

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            max_k = min(max(topk_values), probs.shape[1])
            topk_probs, topk_indices = torch.topk(probs, k=max_k, dim=1)

            pred_indices = preds.detach().cpu().numpy().astype(int)
            topk_indices_np = topk_indices.detach().cpu().numpy().astype(int)
            topk_probs_np = topk_probs.detach().cpu().numpy().astype(np.float32)
            probs_np = probs.detach().cpu().numpy().astype(np.float64)
            entropy_np = compute_entropy(probs_np)
            embeddings_np = (
                embeddings_tensor.detach().cpu().numpy().astype(np.float32)
                if embeddings_tensor is not None
                else None
            )

            for i, pred_idx in enumerate(pred_indices):
                row = batch["rows"][i]
                sample_path = Path(batch["sample_paths"][i])

                actual_idx = batch["actual_idx"][i]
                predicted_label = idx_map.get(pred_idx, canonicalize_aa_label(str(pred_idx)))
                predicted_type = aa_type(predicted_label)
                actual_label = normalize_residue_label(actual_idx, idx_map) if actual_idx is not None else ""
                actual_type = aa_type(actual_label) if actual_label else "Unknown"

                if actual_idx is not None:
                    with_actual += 1
                    correct_type += int(predicted_type == actual_type)
                    hits = compute_topk_hits(actual_idx, topk_indices_np[i], topk_values)
                    for k, hit in hits.items():
                        topk_hit_counts[k] += hit
                    update_confusion(aa_confusion, aa_labels, actual_label, predicted_label)
                    update_confusion(aa_type_confusion, aa_types, actual_type, predicted_type)
                    eval_actual_labels.append(actual_label)
                    eval_predicted_labels.append(predicted_label)
                    eval_confidences.append(float(topk_probs_np[i][0]))
                    eval_correct_flags.append(pred_idx == actual_idx)
                    eval_entropies.append(float(entropy_np[i]))
                    eval_actual_types.append(actual_type)
                    eval_predicted_types.append(predicted_type)

                out_row = {
                    "structure_name": infer_structure_name(row, sample_path),
                    "site": format_site(row, sample_path),
                    "predicted": predicted_label,
                    "actual": actual_label,
                    "predicted_type": predicted_type,
                    "actual_type": actual_type,
                    "entropy": f"{float(entropy_np[i]):.6f}",
                }

                if output_probs:
                    topk_labels = [idx_map.get(int(idx), str(int(idx))) for idx in topk_indices_np[i]]
                    out_row.update(
                        {
                            "predicted_index": str(pred_idx),
                            "actual_index": "" if actual_idx is None else str(int(actual_idx)),
                            "confidence": f"{float(topk_probs_np[i][0]):.6f}",
                            "topk_indices": "|".join(str(int(idx)) for idx in topk_indices_np[i]),
                            "topk_labels": "|".join(topk_labels),
                        }
                    )

                if verbose and actual_idx is None:
                    print(f"No ground truth for sample: {sample_path}")

                if embeddings_np is not None:
                    embedding_vectors.append(embeddings_np[i])
                    embedding_structure_names.append(out_row["structure_name"])
                    embedding_sites.append(out_row["site"])
                    embedding_actual_labels.append(actual_label if actual_label else "Unknown")
                    embedding_predicted_labels.append(predicted_label)
                    embedding_actual_types.append(actual_type if actual_type else "Unknown")
                    embedding_predicted_types.append(predicted_type)

                rows_out.append(out_row)

    type_accuracy = (correct_type / with_actual) if with_actual > 0 else float("nan")
    topk_accuracy = {
        k: ((topk_hit_counts[k] / with_actual) if with_actual > 0 else None)
        for k in topk_values
    }
    summary = {
        "total_samples": len(rows_out),
        "labeled_samples": with_actual,
        "topk_accuracy": topk_accuracy,
        "amino_acid_accuracy": topk_accuracy.get(1, None),
        "amino_acid_type_accuracy": type_accuracy,
        "actual_labels": eval_actual_labels,
        "predicted_labels": eval_predicted_labels,
        "actual_types": eval_actual_types,
        "predicted_types": eval_predicted_types,
        "confidences": eval_confidences,
        "entropies": eval_entropies,
        "correct_flags": eval_correct_flags,
        "embeddings": embedding_vectors,
        "embedding_structure_names": embedding_structure_names,
        "embedding_sites": embedding_sites,
        "embedding_actual_labels": embedding_actual_labels,
        "embedding_predicted_labels": embedding_predicted_labels,
        "embedding_actual_types": embedding_actual_types,
        "embedding_predicted_types": embedding_predicted_types,
    }
    return rows_out, aa_confusion, aa_type_confusion, summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    mean, std = load_normalization(args.normalization)

    dataset = PredictionManifestDataset(
        manifest_path=args.test_manifest,
        mean=mean,
        std=std,
        task=args.task,
    )

    model, num_classes = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        num_classes=args.num_classes,
        task=args.task,
    )

    idx_map = idx_to_residue_map()
    if len(idx_map) < num_classes:
        raise ValueError(
            f"Residue index map supports {len(idx_map)} classes, but model has {num_classes}"
        )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_prediction_batch,
    )

    parsed_topk_values = parse_topk_values(args.topk_values)
    if args.top_k > 1 and args.top_k not in parsed_topk_values:
        parsed_topk_values.append(args.top_k)
    parsed_topk_values = sorted(set(parsed_topk_values))

    output_stem = args.output_csv.stem
    output_parent = args.output_csv.parent
    aa_confusion_csv = args.aa_confusion_csv or (output_parent / f"{output_stem}_aa_confusion.csv")
    aa_type_confusion_csv = args.aa_type_confusion_csv or (
        output_parent / f"{output_stem}_aa_type_confusion.csv"
    )
    aa_confusion_png = args.aa_confusion_png or (output_parent / f"{output_stem}_aa_confusion.png")
    aa_type_confusion_png = args.aa_type_confusion_png or (
        output_parent / f"{output_stem}_aa_type_confusion.png"
    )
    per_class_metrics_csv = args.per_class_metrics_csv or (
        output_parent / f"{output_stem}_per_class_metrics.csv"
    )
    confusions_csv = args.confusions_csv or (output_parent / f"{output_stem}_common_confusions.csv")
    confidence_hist_png = args.confidence_hist_png or (
        output_parent / f"{output_stem}_confidence_hist.png"
    )
    calibration_png = args.calibration_png or (output_parent / f"{output_stem}_calibration.png")
    aa_support_csv = args.aa_support_csv or (output_parent / f"{output_stem}_aa_support.csv")
    aa_support_png = args.aa_support_png or (output_parent / f"{output_stem}_aa_support.png")
    aa_type_accuracy_csv = args.aa_type_accuracy_csv or (
        output_parent / f"{output_stem}_aa_type_accuracy.csv"
    )
    aa_type_accuracy_png = args.aa_type_accuracy_png or (
        output_parent / f"{output_stem}_aa_type_accuracy.png"
    )
    entropy_summary_csv = args.entropy_summary_csv or (output_parent / f"{output_stem}_entropy_summary.csv")
    entropy_hist_png = args.entropy_hist_png or (output_parent / f"{output_stem}_entropy_hist.png")
    embeddings_npz = args.save_embeddings_npz or (output_parent / f"{output_stem}_embeddings.npz")
    embedding_plot_png = args.embedding_plot_png or (output_parent / f"{output_stem}_embedding_plot.png")

    collect_embedding_data = args.embedding_method != "none" or args.save_embeddings_npz is not None
    rows_out, aa_confusion, aa_type_confusion, summary = predict(
        model=model,
        loader=loader,
        device=device,
        idx_map=idx_map,
        topk_values=parsed_topk_values,
        output_probs=args.output_probs,
        amp_enabled=args.amp and device.type == "cuda",
        verbose=args.verbose,
        collect_embedding_data=collect_embedding_data,
        embedding_layer=args.embedding_layer,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["structure_name", "site", "predicted", "actual", "predicted_type", "actual_type"]
    fieldnames.append("entropy")
    if args.output_probs:
        fieldnames.extend(["predicted_index", "actual_index", "confidence", "topk_indices", "topk_labels"])

    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    aa_labels = canonical_aa_labels()
    aa_type_labels = list(CANONICAL_AA_TYPES)

    write_confusion_csv(aa_confusion, aa_labels, aa_confusion_csv, "actual/predicted")
    write_confusion_csv(
        aa_type_confusion,
        aa_type_labels,
        aa_type_confusion_csv,
        "actual_type/predicted_type",
    )
    save_confusion_heatmap(
        matrix=confusion_to_array(aa_confusion, aa_labels),
        labels=aa_labels,
        out_path=aa_confusion_png,
        title="Amino Acid Confusion Matrix",
        normalize_mode=args.normalize_confusion,
    )
    save_confusion_heatmap(
        matrix=confusion_to_array(aa_type_confusion, aa_type_labels),
        labels=aa_type_labels,
        out_path=aa_type_confusion_png,
        title="Amino Acid Type Confusion Matrix",
        normalize_mode=args.normalize_confusion,
    )

    metrics_rows, macro_avg, weighted_avg = compute_per_class_metrics(
        actual_labels=summary["actual_labels"],
        predicted_labels=summary["predicted_labels"],
        class_labels=aa_labels,
    )
    write_per_class_metrics_csv(metrics_rows, macro_avg, weighted_avg, per_class_metrics_csv)

    common_confusions = compute_common_confusions(
        actual_labels=summary["actual_labels"],
        predicted_labels=summary["predicted_labels"],
    )
    write_common_confusions_csv(common_confusions, confusions_csv, args.max_confusions)

    if summary["labeled_samples"] > 0:
        save_confidence_histogram(summary["confidences"], summary["correct_flags"], confidence_hist_png)
        calibration_bins = compute_calibration_bins(
            summary["confidences"],
            summary["correct_flags"],
            args.calibration_bins,
        )
        save_calibration_plot(calibration_bins, calibration_png)
        support_rows = compute_support_per_aa(summary["actual_labels"], aa_labels)
        write_support_csv(support_rows, aa_support_csv)
        save_support_bar_chart(support_rows, aa_support_png)

        type_accuracy_rows = compute_accuracy_by_type(
            summary["actual_types"],
            summary["predicted_types"],
            aa_type_labels,
        )
        write_type_accuracy_csv(type_accuracy_rows, aa_type_accuracy_csv)
        save_type_accuracy_bar_chart(type_accuracy_rows, aa_type_accuracy_png)

        entropy_summary = summarize_entropy(
            summary["entropies"],
            summary["confidences"],
            summary["correct_flags"],
        )
        write_entropy_summary_csv(entropy_summary, entropy_summary_csv)
        save_entropy_histogram(summary["entropies"], summary["correct_flags"], entropy_hist_png)

    embedding_plot_written = False
    if collect_embedding_data and summary["embeddings"]:
        emb = np.asarray(summary["embeddings"], dtype=np.float32)
        n_total = emb.shape[0]
        if args.embedding_max_samples > 0 and n_total > args.embedding_max_samples:
            rng = np.random.default_rng(args.seed)
            sample_idx = rng.choice(n_total, size=args.embedding_max_samples, replace=False)
            sample_idx.sort()
            emb = emb[sample_idx]
        else:
            sample_idx = np.arange(n_total)

        actual_labels = np.asarray(summary["embedding_actual_labels"], dtype=object)
        predicted_labels = np.asarray(summary["embedding_predicted_labels"], dtype=object)
        actual_types = np.asarray(summary["embedding_actual_types"], dtype=object)
        predicted_types = np.asarray(summary["embedding_predicted_types"], dtype=object)
        structure_names = np.asarray(summary["embedding_structure_names"], dtype=object)
        sites = np.asarray(summary["embedding_sites"], dtype=object)

        if actual_labels.shape[0] == n_total:
            np.savez(
                embeddings_npz,
                embeddings=emb,
                actual_labels=actual_labels[sample_idx],
                predicted_labels=predicted_labels[sample_idx],
                actual_types=actual_types[sample_idx],
                predicted_types=predicted_types[sample_idx],
                structure_names=structure_names[sample_idx],
                sites=sites[sample_idx],
            )

        if args.embedding_method != "none":
            reduced: np.ndarray | None = None
            method = args.embedding_method
            if method == "pca":
                reduced = reduce_embeddings_pca(emb.astype(np.float64), n_components=2)
            elif method == "umap":
                reduced = reduce_embeddings_umap(emb.astype(np.float64), n_components=2)
                if reduced is None:
                    print("UMAP unavailable (umap-learn not installed); falling back to PCA.")
                    method = "pca"
                    reduced = reduce_embeddings_pca(emb.astype(np.float64), n_components=2)

            if reduced is not None:
                color_values_map = {
                    "actual": actual_labels[sample_idx].tolist(),
                    "predicted": predicted_labels[sample_idx].tolist(),
                    "actual_type": actual_types[sample_idx].tolist(),
                    "predicted_type": predicted_types[sample_idx].tolist(),
                }
                color_values = color_values_map[args.embedding_color_by]
                save_embedding_plot(
                    reduced,
                    color_values,
                    embedding_plot_png,
                    f"{method.upper()} of {args.embedding_layer.capitalize()} Embeddings "
                    f"colored by {args.embedding_color_by}",
                )
                embedding_plot_written = True

    print(f"Total samples: {summary['total_samples']}")
    print(f"Labeled samples used for evaluation: {summary['labeled_samples']}")
    top1_accuracy = summary["topk_accuracy"].get(1)
    if top1_accuracy is not None:
        print(f"Top-1 accuracy: {top1_accuracy:.4f}")
    else:
        print("Top-1 accuracy: N/A (no ground-truth labels available)")

    for k in (3, 5):
        if k in summary["topk_accuracy"]:
            if summary["topk_accuracy"][k] is not None:
                print(f"Top-{k} accuracy: {summary['topk_accuracy'][k]:.4f}")
            else:
                print(f"Top-{k} accuracy: N/A (no ground-truth labels available)")

    if np.isfinite(summary["amino_acid_type_accuracy"]):
        print(f"Amino acid type accuracy: {summary['amino_acid_type_accuracy']:.4f}")
    else:
        print("Amino acid type accuracy: N/A (no ground-truth labels available)")

    print(
        "Macro precision / recall / F1: "
        f"{macro_avg['precision']:.4f} / {macro_avg['recall']:.4f} / {macro_avg['f1']:.4f}"
    )
    print(
        "Weighted precision / recall / F1: "
        f"{weighted_avg['precision']:.4f} / {weighted_avg['recall']:.4f} / {weighted_avg['f1']:.4f}"
    )
    if summary["labeled_samples"] > 0:
        support_rows = compute_support_per_aa(summary["actual_labels"], aa_labels)
        support_values = [int(row["support"]) for row in support_rows]
        print(f"Support min/max across amino acids: {min(support_values)} / {max(support_values)}")

        type_accuracy_rows = compute_accuracy_by_type(
            summary["actual_types"],
            summary["predicted_types"],
            aa_type_labels,
        )
        print("Amino acid type accuracies:")
        for row in type_accuracy_rows:
            print(
                f"  {row['amino_acid_type']}: {float(row['accuracy']):.4f} "
                f"(n={int(row['support'])}, correct={int(row['correct'])})"
            )

        entropy_summary = summarize_entropy(
            summary["entropies"],
            summary["confidences"],
            summary["correct_flags"],
        )
        print(f"Mean entropy overall (natural log): {entropy_summary['mean_entropy']:.4f}")
        print(
            "Mean entropy (correct / incorrect): "
            f"{entropy_summary['mean_entropy_correct']:.4f} / "
            f"{entropy_summary['mean_entropy_incorrect']:.4f}"
        )
    print(
        "Embedding plot: "
        f"{'written to ' + str(embedding_plot_png) if embedding_plot_written else 'skipped'}"
    )


if __name__ == "__main__":
    main()

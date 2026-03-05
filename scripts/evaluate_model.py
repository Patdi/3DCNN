#!/usr/bin/env python3
"""Evaluate a PyTorch checkpoint on a manifest split and report metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.constants import RESIDUE_GROUP_BY_LABEL


class ManifestDataset(Dataset):
    """Dataset that reads per-example rows from a sample manifest CSV."""

    def __init__(self, manifest_path: Path, normalization_path: Path | None = None):
        self.manifest_dir = manifest_path.parent.resolve()
        self.records = self._load_manifest(manifest_path)
        self.mean = None
        self.std = None
        if normalization_path is not None:
            stats = np.load(normalization_path)
            self.mean = stats["train_mean"].astype(np.float32)
            self.std = np.maximum(stats["train_std"].astype(np.float32), 1e-12)

    @staticmethod
    def _load_manifest(manifest_path: Path) -> List[Dict[str, str]]:
        with manifest_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return []

        if "sample_path" not in rows[0] and "path" in rows[0]:
            for row in rows:
                row["sample_path"] = row["path"]

        required = {"sample_path", "label"}
        missing = required.difference(rows[0].keys())
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Manifest missing required columns: {missing_str}")
        return rows

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _to_channels_first(sample: np.ndarray) -> np.ndarray:
        if sample.ndim == 4 and sample.shape[-1] <= 8 and sample.shape[0] > 8:
            return np.transpose(sample, (3, 0, 1, 2))
        return sample

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        row = self.records[index]
        sample_path = Path(row["sample_path"])
        if not sample_path.is_absolute():
            sample_path = self.manifest_dir / sample_path
        arr = np.load(sample_path, allow_pickle=False)
        if "sample_index" in row and row["sample_index"] not in {"", None}:
            sample = arr[int(row["sample_index"])].astype(np.float32)
        else:
            if not isinstance(arr, np.lib.npyio.NpzFile):
                raise ValueError("Per-example schema requires .npz files with an 'x' array")
            sample = arr["x"].astype(np.float32)
        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
        sample = self._to_channels_first(sample)
        return {
            "inputs": torch.from_numpy(sample),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf[t, p] += 1
    return conf


def compute_balanced_accuracy(confusion: np.ndarray) -> float:
    recalls = []
    for i in range(confusion.shape[0]):
        recalls.append(_safe_divide(confusion[i, i], confusion[i].sum()))
    return float(np.mean(recalls)) if recalls else 0.0


def compute_topk_accuracy(logits: np.ndarray, y_true: np.ndarray, k: int) -> float:
    if logits.size == 0:
        return 0.0
    k = max(1, min(k, logits.shape[1]))
    topk = np.argpartition(logits, -k, axis=1)[:, -k:]
    hits = [(int(label) in topk[i]) for i, label in enumerate(y_true)]
    return float(np.mean(hits)) if hits else 0.0


def compute_residue_group_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    hits = 0
    valid = 0
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if t in RESIDUE_GROUP_BY_LABEL and p in RESIDUE_GROUP_BY_LABEL:
            valid += 1
            if RESIDUE_GROUP_BY_LABEL[t] == RESIDUE_GROUP_BY_LABEL[p]:
                hits += 1
    return _safe_divide(hits, valid)


def compute_auroc_and_prauc(y_true: np.ndarray, logits: np.ndarray, num_classes: int) -> Dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return {}

    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    if num_classes == 2:
        scores = probs[:, 1]
        return {
            "auroc": float(roc_auc_score(y_true, scores)),
            "prauc": float(average_precision_score(y_true, scores)),
        }

    y_one_hot = np.eye(num_classes)[y_true.astype(int)]
    return {
        "auroc": float(roc_auc_score(y_one_hot, probs, multi_class="ovr", average="macro")),
        "prauc": float(average_precision_score(y_one_hot, probs, average="macro")),
    }


def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            preds = torch.argmax(logits, dim=1)
            all_logits.append(logits.detach().cpu().numpy())
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    return (
        np.concatenate(all_logits, axis=0) if all_logits else np.array([]),
        np.concatenate(all_preds, axis=0) if all_preds else np.array([], dtype=np.int64),
        np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int64),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    logits: np.ndarray,
    num_classes: int,
    requested_metrics: Sequence[str],
) -> Dict[str, object]:
    metrics: Dict[str, object] = {}
    confusion = None

    for metric_name in requested_metrics:
        name = metric_name.strip().lower()
        if not name:
            continue
        if name == "accuracy":
            metrics["accuracy"] = _safe_divide(np.sum(y_true == y_pred), len(y_true))
        elif name == "balanced_accuracy":
            confusion = confusion if confusion is not None else compute_confusion_matrix(y_true, y_pred, num_classes)
            metrics["balanced_accuracy"] = compute_balanced_accuracy(confusion)
        elif name == "confusion":
            confusion = confusion if confusion is not None else compute_confusion_matrix(y_true, y_pred, num_classes)
            metrics["confusion"] = confusion.tolist()
        elif name.startswith("top"):
            k = int(name.replace("top", ""))
            metrics[name] = compute_topk_accuracy(logits, y_true, k)
        elif name in {"auroc", "prauc"}:
            auc_metrics = compute_auroc_and_prauc(y_true, logits, num_classes)
            metrics.update({k: v for k, v in auc_metrics.items() if k in requested_metrics})
        elif name == "residue_group_accuracy":
            metrics["residue_group_accuracy"] = compute_residue_group_accuracy(y_true, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric_name}")

    return metrics


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.to(device)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), torch.nn.Module):
        return checkpoint["model"].to(device)
    raise ValueError(
        "Checkpoint must contain a serialized torch.nn.Module or a dict with key 'model'. "
        "State-dict-only checkpoints are not yet supported by this script."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--normalization", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--task", required=True, type=str)

    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--metrics", type=str, default="accuracy,balanced_accuracy,top5,confusion")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset = ManifestDataset(args.manifest, args.normalization)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model_from_checkpoint(args.checkpoint, device)
    logits, preds, labels = run_inference(model, loader, device)

    requested_metrics = [metric.strip().lower() for metric in args.metrics.split(",")]
    metrics = compute_metrics(labels, preds, logits, args.num_classes, requested_metrics)
    metrics["task"] = args.task
    metrics["num_examples"] = int(labels.shape[0])

    np.savez(
        args.output_dir / "predictions.npz",
        logits=logits,
        predictions=preds,
        labels=labels,
    )
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()

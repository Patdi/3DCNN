import csv
from pathlib import Path

import numpy as np

from scripts.evaluate_model import (
    ManifestDataset,
    compute_balanced_accuracy,
    compute_confusion_matrix,
    compute_metrics,
    compute_residue_group_accuracy,
    compute_topk_accuracy,
)


def test_manifest_dataset_loads_and_normalizes(tmp_path: Path):
    sample_path = tmp_path / "sample.dat"
    data = np.array([[[[2.0, 4.0]]]], dtype=np.float32)  # (N, D, H, W, C)
    np.save(sample_path, data)
    (tmp_path / "sample.dat.npy").rename(sample_path)

    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_path", "sample_index", "label"])
        writer.writeheader()
        writer.writerow({"sample_path": str(sample_path), "sample_index": 0, "label": 1})

    stats_path = tmp_path / "stats.npz"
    np.savez(stats_path, train_mean=np.array([[[[1.0, 1.0]]]], dtype=np.float32), train_std=np.array([[[[1.0, 3.0]]]], dtype=np.float32))

    ds = ManifestDataset(manifest_path, stats_path)
    item = ds[0]

    assert tuple(item["inputs"].shape) == (2, 1, 1, 1)
    assert np.isclose(item["inputs"][0, 0, 0, 0].item(), 1.0)
    assert np.isclose(item["inputs"][1, 0, 0, 0].item(), 1.0)
    assert item["label"].item() == 1


def test_metric_helpers_return_expected_values():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    logits = np.array(
        [
            [5.0, 1.0],
            [0.2, 0.3],
            [0.1, 0.9],
            [0.4, 0.6],
        ],
        dtype=np.float32,
    )

    conf = compute_confusion_matrix(y_true, y_pred, num_classes=2)
    assert conf.tolist() == [[1, 1], [0, 2]]
    assert np.isclose(compute_balanced_accuracy(conf), 0.75)
    assert np.isclose(compute_topk_accuracy(logits, y_true, k=1), 0.75)


def test_compute_metrics_supports_residue_group_accuracy():
    y_true = np.array([0, 3, 19])
    y_pred = np.array([2, 4, 19])
    logits = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ],
        dtype=np.float32,
    )

    assert np.isclose(compute_residue_group_accuracy(y_true, y_pred), 1.0)
    metrics = compute_metrics(y_true, y_pred, logits, num_classes=20, requested_metrics=["accuracy", "residue_group_accuracy"])
    assert np.isclose(metrics["accuracy"], 1.0 / 3.0)
    assert np.isclose(metrics["residue_group_accuracy"], 1.0)

import csv
from pathlib import Path

import numpy as np

from prepare_dataset import (
    ExampleRecord,
    balance_records,
    compute_class_weights,
    compute_normalization_stats,
    prepare_dataset,
)


def make_shard(path: Path, residue: str, shard_id: int, count: int, fill: float) -> None:
    arr = np.full((count, 2, 2), fill_value=fill, dtype=np.float32)
    np.save(path / f"{residue}_{shard_id}.dat", arr)
    npy = path / f"{residue}_{shard_id}.dat.npy"
    npy.rename(path / f"{residue}_{shard_id}.dat")


def test_balance_records_downsample_reduces_to_minimum():
    records = [
        ExampleRecord("a", i, 0, "HIS", "train") for i in range(4)
    ] + [
        ExampleRecord("b", i, 1, "LYS", "train") for i in range(2)
    ]

    balanced = balance_records(records, mode="downsample", seed=1)
    labels = [r.label for r in balanced]
    assert labels.count(0) == 2
    assert labels.count(1) == 2


def test_compute_normalization_stats_from_train_records(tmp_path: Path):
    shard = tmp_path / "HIS_0.dat"
    arr = np.array([[[1.0]], [[3.0]]], dtype=np.float32)
    np.save(shard, arr)
    (tmp_path / "HIS_0.dat.npy").rename(shard)

    records = [
        ExampleRecord(str(shard), 0, 0, "HIS", "train"),
        ExampleRecord(str(shard), 1, 0, "HIS", "train"),
    ]

    mean, std = compute_normalization_stats(records)
    assert np.allclose(mean, np.array([[2.0]], dtype=np.float32))
    assert np.allclose(std, np.array([[1.0]], dtype=np.float32))


def test_prepare_dataset_creates_manifests_and_stats(tmp_path: Path):
    raw_root = tmp_path / "RAW_DATA"
    train_dir = raw_root / "train"
    test_dir = raw_root / "test"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)

    make_shard(train_dir, "HIS", 0, count=4, fill=1.0)
    make_shard(train_dir, "LYS", 0, count=4, fill=3.0)
    make_shard(test_dir, "HIS", 1, count=2, fill=5.0)

    output_dir = tmp_path / "out"
    prepare_dataset(raw_root, output_dir, val_fraction=0.25, balance_mode="weighted_loss", seed=7)

    train_manifest = output_dir / "manifest_train.csv"
    val_manifest = output_dir / "manifest_val.csv"
    test_manifest = output_dir / "manifest_test.csv"
    stats_file = output_dir / "normalization_stats.npz"

    assert train_manifest.exists()
    assert val_manifest.exists()
    assert test_manifest.exists()
    assert stats_file.exists()

    with train_manifest.open() as handle:
        train_rows = list(csv.DictReader(handle))
    with val_manifest.open() as handle:
        val_rows = list(csv.DictReader(handle))
    with test_manifest.open() as handle:
        test_rows = list(csv.DictReader(handle))

    assert len(train_rows) == 6
    assert len(val_rows) == 2
    assert len(test_rows) == 2

    stats = np.load(stats_file)
    assert "train_mean" in stats
    assert "train_std" in stats
    assert "class_weights" in stats


def test_compute_class_weights_inverse_frequency():
    records = [
        ExampleRecord("a", 0, 0, "HIS", "train"),
        ExampleRecord("b", 0, 0, "HIS", "train"),
        ExampleRecord("c", 0, 1, "LYS", "train"),
    ]
    weights = compute_class_weights(records)
    assert np.isclose(weights[0], 0.75)
    assert np.isclose(weights[1], 1.5)

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compute_normalization import main


def _write_manifest(path: Path, sample_path: Path, n: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_path", "sample_index"])
        writer.writeheader()
        for i in range(n):
            writer.writerow({"sample_path": str(sample_path), "sample_index": str(i)})


def test_compute_per_channel_normalization(tmp_path: Path) -> None:
    shard = tmp_path / "samples.npy"
    data = np.stack(
        [
            np.zeros((2, 2, 2, 2), dtype=np.float32),
            np.ones((2, 2, 2, 2), dtype=np.float32) * 2,
        ],
        axis=0,
    )
    np.save(shard, data)

    manifest = tmp_path / "train_sites.csv"
    _write_manifest(manifest, shard, n=2)
    out = tmp_path / "normalization_stats.npz"

    rc = main(["--manifest", str(manifest), "--out", str(out), "--mode", "per-channel"])
    assert rc == 0

    stats = np.load(out)
    mean = stats["train_mean"]
    std = stats["train_std"]

    assert mean.shape == (2, 1, 1, 1)
    assert std.shape == (2, 1, 1, 1)
    np.testing.assert_allclose(mean.squeeze(), np.array([1.0, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(std.squeeze(), np.array([1.0, 1.0], dtype=np.float32), atol=1e-6)


def test_compute_global_normalization_with_max_samples(tmp_path: Path) -> None:
    shard = tmp_path / "samples.npy"
    data = np.array(
        [
            np.zeros((1, 2, 2, 2), dtype=np.float32),
            np.ones((1, 2, 2, 2), dtype=np.float32),
            np.ones((1, 2, 2, 2), dtype=np.float32) * 3,
        ]
    )
    np.save(shard, data)

    manifest = tmp_path / "train_sites.csv"
    _write_manifest(manifest, shard, n=3)
    out = tmp_path / "normalization_stats.npz"

    rc = main([
        "--manifest",
        str(manifest),
        "--out",
        str(out),
        "--mode",
        "global",
        "--max-samples",
        "2",
        "--seed",
        "7",
    ])
    assert rc == 0

    stats = np.load(out)
    assert stats["train_mean"].shape == (1, 1, 1, 1)
    assert stats["train_std"].shape == (1, 1, 1, 1)
    assert int(stats["num_samples"]) == 2


def test_compute_per_channel_normalization_from_per_example_npz(tmp_path: Path) -> None:
    sample_a = tmp_path / "a.npz"
    sample_b = tmp_path / "b.npz"
    np.savez(sample_a, x=np.zeros((2, 2, 2, 2), dtype=np.float32), y=np.array(0, dtype=np.int64))
    np.savez(sample_b, x=np.ones((2, 2, 2, 2), dtype=np.float32) * 2, y=np.array(1, dtype=np.int64))

    manifest = tmp_path / "train_sites.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_path", "label"])
        writer.writeheader()
        writer.writerow({"sample_path": str(sample_a), "label": "0"})
        writer.writerow({"sample_path": str(sample_b), "label": "1"})

    out = tmp_path / "normalization_stats.npz"
    rc = main(["--manifest", str(manifest), "--out", str(out), "--mode", "per-channel"])
    assert rc == 0

    stats = np.load(out)
    np.testing.assert_allclose(stats["train_mean"].squeeze(), np.array([1.0, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(stats["train_std"].squeeze(), np.array([1.0, 1.0], dtype=np.float32), atol=1e-6)

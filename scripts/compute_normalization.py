#!/usr/bin/env python3
"""Compute normalization statistics from a training site manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="CSV manifest with sample_path (or path) and optional sample_index columns")
    parser.add_argument("--out", required=True, type=Path, help="Output .npz path for train_mean/train_std")
    parser.add_argument("--mode", choices=["per-channel", "global"], default="per-channel")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def load_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError("Manifest is empty")

    if "sample_path" not in rows[0] and "path" in rows[0]:
        for row in rows:
            row["sample_path"] = row["path"]

    required = {"sample_path"}
    missing = required.difference(rows[0].keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Manifest missing required columns: {missing_str}")
    return rows


def _to_channels_first(sample: np.ndarray) -> np.ndarray:
    if sample.ndim == 3:
        return sample[np.newaxis, ...]
    if sample.ndim == 4 and sample.shape[-1] <= 8 and sample.shape[0] > 8:
        return np.transpose(sample, (3, 0, 1, 2))
    if sample.ndim == 4:
        return sample
    raise ValueError(f"Expected 3D or 4D sample, got shape {sample.shape}")


def iter_samples(rows: Iterable[dict[str, str]], manifest_dir: Path) -> Iterable[np.ndarray]:
    cache: dict[Path, np.ndarray | np.lib.npyio.NpzFile] = {}
    for row in rows:
        sample_path = Path(row["sample_path"])
        if not sample_path.is_absolute():
            sample_path = manifest_dir / sample_path
        if sample_path not in cache:
            cache[sample_path] = np.load(sample_path, allow_pickle=False)
        arr = cache[sample_path]

        if "sample_index" in row and row["sample_index"] not in {"", None}:
            sample = arr[int(row["sample_index"])].astype(np.float64)
        else:
            if not isinstance(arr, np.lib.npyio.NpzFile):
                raise ValueError("Per-example schema requires .npz files with an 'x' array")
            sample = arr["x"].astype(np.float64)
        yield _to_channels_first(sample)


def compute_global_stats(samples: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    total_sum = 0.0
    total_sq_sum = 0.0
    count = 0

    for sample in samples:
        total_sum += float(sample.sum())
        total_sq_sum += float(np.square(sample).sum())
        count += int(sample.size)

    if count == 0:
        raise ValueError("No samples were available to compute normalization")

    mean = total_sum / count
    var = max(total_sq_sum / count - mean * mean, 0.0)
    std = max(var**0.5, 1e-6)
    return np.array([[[[mean]]]], dtype=np.float32), np.array([[[[std]]]], dtype=np.float32)


def compute_per_channel_stats(samples: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = None
    channel_sq_sum = None
    count_per_channel = None

    for sample in samples:
        c = sample.shape[0]
        if channel_sum is None:
            channel_sum = np.zeros(c, dtype=np.float64)
            channel_sq_sum = np.zeros(c, dtype=np.float64)
            count_per_channel = np.zeros(c, dtype=np.int64)
        if c != len(channel_sum):
            raise ValueError(f"Inconsistent channel count. Expected {len(channel_sum)}, got {c}")

        flat = sample.reshape(c, -1)
        channel_sum += flat.sum(axis=1)
        channel_sq_sum += np.square(flat).sum(axis=1)
        count_per_channel += flat.shape[1]

    if channel_sum is None or channel_sq_sum is None or count_per_channel is None:
        raise ValueError("No samples were available to compute normalization")

    means = channel_sum / np.maximum(count_per_channel, 1)
    variances = np.maximum(channel_sq_sum / np.maximum(count_per_channel, 1) - np.square(means), 0.0)
    stds = np.maximum(np.sqrt(variances), 1e-6)

    return means.astype(np.float32)[:, np.newaxis, np.newaxis, np.newaxis], stds.astype(np.float32)[:, np.newaxis, np.newaxis, np.newaxis]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = load_manifest_rows(args.manifest)
    if args.max_samples is not None and args.max_samples > 0 and len(rows) > args.max_samples:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(rows), size=args.max_samples, replace=False)
        rows = [rows[i] for i in sorted(indices)]

    samples = iter_samples(rows, manifest_dir=args.manifest.parent)
    if args.mode == "global":
        train_mean, train_std = compute_global_stats(samples)
    else:
        train_mean, train_std = compute_per_channel_stats(samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        train_mean=train_mean,
        train_std=train_std,
        mode=np.array(args.mode),
        num_samples=np.array(len(rows), dtype=np.int64),
        seed=np.array(args.seed, dtype=np.int64),
    )
    print(f"Wrote normalization stats to {args.out} from {len(rows)} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

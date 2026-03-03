#!/usr/bin/env python3
"""Prepare manifest CSV files and normalization stats from raw DAT shards."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

RESI_NAME_TO_LABEL = {
    "HIS": 0,
    "LYS": 1,
    "ARG": 2,
    "ASP": 3,
    "GLU": 4,
    "SER": 5,
    "THR": 6,
    "ASN": 7,
    "GLN": 8,
    "ALA": 9,
    "VAL": 10,
    "LEU": 11,
    "ILE": 12,
    "MET": 13,
    "PHE": 14,
    "TYR": 15,
    "TRP": 16,
    "PRO": 17,
    "GLY": 18,
    "CYS": 19,
}


@dataclass(frozen=True)
class ExampleRecord:
    sample_path: str
    sample_index: int
    label: int
    residue: str
    split: str
    pdb_id: str = ""
    chain: str = ""
    resnum: str = ""


def parse_filename_metadata(file_path: Path) -> Tuple[str, str, str, str]:
    stem = file_path.stem
    parts = stem.split("_")
    residue = parts[0]
    pdb_id = parts[1] if len(parts) > 1 else ""
    chain = parts[2] if len(parts) > 2 else ""
    resnum = parts[3] if len(parts) > 3 else ""
    return residue, pdb_id, chain, resnum


def load_shard(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def discover_examples(raw_split_dir: Path, split_name: str) -> List[ExampleRecord]:
    records: List[ExampleRecord] = []
    for shard in sorted(raw_split_dir.glob("*.dat")):
        residue, pdb_id, chain, resnum = parse_filename_metadata(shard)
        if residue not in RESI_NAME_TO_LABEL:
            continue
        label = RESI_NAME_TO_LABEL[residue]
        arr = load_shard(shard)
        num_samples = int(arr.shape[0]) if arr.ndim > 0 else 1
        for sample_index in range(num_samples):
            records.append(
                ExampleRecord(
                    sample_path=str(shard.resolve()),
                    sample_index=sample_index,
                    label=label,
                    residue=residue,
                    split=split_name,
                    pdb_id=pdb_id,
                    chain=chain,
                    resnum=resnum,
                )
            )
    return records


def balance_records(records: Sequence[ExampleRecord], mode: str, seed: int) -> Sequence[ExampleRecord]:
    if mode in {"none", "weighted_loss"}:
        return records

    grouped: Dict[int, List[ExampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    if not grouped:
        return []

    min_count = min(len(v) for v in grouped.values())
    rng = random.Random(seed)
    balanced: List[ExampleRecord] = []
    for label in sorted(grouped):
        label_records = grouped[label]
        balanced.extend(rng.sample(label_records, min_count))
    return balanced


def stratified_split(records: Sequence[ExampleRecord], val_fraction: float, seed: int) -> Tuple[List[ExampleRecord], List[ExampleRecord]]:
    rng = random.Random(seed)
    grouped: Dict[int, List[ExampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.label].append(record)

    train_records: List[ExampleRecord] = []
    val_records: List[ExampleRecord] = []
    for label, label_records in grouped.items():
        local = label_records[:]
        rng.shuffle(local)
        val_count = int(round(len(local) * val_fraction))
        val_records.extend([ExampleRecord(**{**asdict(r), "split": "val"}) for r in local[:val_count]])
        train_records.extend([ExampleRecord(**{**asdict(r), "split": "train"}) for r in local[val_count:]])
    return train_records, val_records


def write_manifest(records: Sequence[ExampleRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_path", "sample_index", "label", "residue", "split", "pdb_id", "chain", "resnum"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def compute_normalization_stats(records: Sequence[ExampleRecord]) -> Tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Cannot compute normalization stats on an empty training split")

    grouped: Dict[str, List[int]] = defaultdict(list)
    for record in records:
        grouped[record.sample_path].append(record.sample_index)

    running_sum = None
    running_sq_sum = None
    total_samples = 0

    for path, sample_indices in grouped.items():
        arr = load_shard(Path(path))
        sample_batch = arr[np.array(sample_indices, dtype=int)]
        batch_sum = sample_batch.sum(axis=0, dtype=np.float64)
        batch_sq_sum = np.square(sample_batch, dtype=np.float64).sum(axis=0)
        if running_sum is None:
            running_sum = batch_sum
            running_sq_sum = batch_sq_sum
        else:
            running_sum += batch_sum
            running_sq_sum += batch_sq_sum
        total_samples += sample_batch.shape[0]

    mean = running_sum / total_samples
    var = (running_sq_sum / total_samples) - np.square(mean)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_class_weights(records: Sequence[ExampleRecord]) -> np.ndarray:
    counts = Counter(r.label for r in records)
    if not counts:
        return np.array([], dtype=np.float32)

    num_classes = max(counts) + 1
    weights = np.zeros(num_classes, dtype=np.float32)
    total = sum(counts.values())
    for label, count in counts.items():
        weights[label] = total / (len(counts) * count)
    return weights


def prepare_dataset(
    raw_data_root: Path,
    output_dir: Path,
    val_fraction: float = 0.05,
    balance_mode: str = "downsample",
    seed: int = 1337,
) -> None:
    np.random.seed(seed)
    random.seed(seed)

    train_records = discover_examples(raw_data_root / "train", split_name="train")
    test_records = discover_examples(raw_data_root / "test", split_name="test")

    train_records = list(balance_records(train_records, mode=balance_mode, seed=seed))
    train_records, val_records = stratified_split(train_records, val_fraction=val_fraction, seed=seed)

    write_manifest(train_records, output_dir / "manifest_train.csv")
    write_manifest(val_records, output_dir / "manifest_val.csv")
    write_manifest(test_records, output_dir / "manifest_test.csv")

    train_mean, train_std = compute_normalization_stats(train_records)
    stats_payload = {
        "train_mean": train_mean,
        "train_std": train_std,
        "seed": np.array(seed, dtype=np.int64),
    }

    if balance_mode == "weighted_loss":
        stats_payload["class_weights"] = compute_class_weights(train_records)

    np.savez(output_dir / "normalization_stats.npz", **stats_payload)

    meta = {
        "balance_mode": balance_mode,
        "seed": seed,
        "val_fraction": val_fraction,
        "train_size": len(train_records),
        "val_size": len(val_records),
        "test_size": len(test_records),
    }
    with (output_dir / "normalization_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset manifests and normalization stats")
    parser.add_argument("--raw-data-root", type=Path, default=Path("../data/RAW_DATA"))
    parser.add_argument("--output-dir", type=Path, default=Path("../data/Sampled_Numpy"))
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--balance-mode", choices=["none", "downsample", "weighted_loss"], default="downsample")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        raw_data_root=args.raw_data_root,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        balance_mode=args.balance_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

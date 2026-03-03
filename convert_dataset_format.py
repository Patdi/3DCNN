#!/usr/bin/env python3
"""Convert legacy sampled NumPy dumps to modern dataset bundles.

Default output format is compressed ``.npz`` with a JSON sidecar containing
schema metadata. Optional HDF5 output is available when ``h5py`` is installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

TRAIN_SPLIT = "train"
DEFAULT_SPLITS = ("train", "val", "test")


class ConversionError(RuntimeError):
    """Raised when input files are missing or invalid."""


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise ConversionError(f"Missing input array: {path}")
    return np.load(path)


def _load_train_part(split_root: Path, part: int, train_shards: int) -> tuple[np.ndarray, np.ndarray]:
    data_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    for shard_idx in range(1, train_shards + 1):
        data_path = split_root / f"X_smooth{shard_idx}_{part}.dat"
        label_path = split_root / f"y{shard_idx}_{part}.dat"
        data_chunks.append(_load_array(data_path))
        labels = _load_array(label_path)
        label_chunks.append(np.ravel(labels))

    if not data_chunks:
        raise ConversionError(f"No training shards were found for part {part}.")

    data = np.concatenate(data_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    return data, labels


def _load_eval_part(split_root: Path, split: str, part: int) -> tuple[np.ndarray, np.ndarray]:
    prefix = "v" if split == "val" else "t"
    data_path = split_root / f"X{prefix}_smooth_{part}.dat"
    label_path = split_root / f"y{prefix}_{part}.dat"
    data = _load_array(data_path)
    labels = np.ravel(_load_array(label_path))
    return data, labels


def load_split_part(sampled_root: Path, split: str, part: int, train_shards: int = 19) -> tuple[np.ndarray, np.ndarray]:
    split_root = sampled_root / split
    if split == TRAIN_SPLIT:
        return _load_train_part(split_root, part, train_shards)
    return _load_eval_part(split_root, split, part)


def build_metadata(
    *,
    split: str,
    part: int,
    data: np.ndarray,
    labels: np.ndarray,
    channels: int | None,
    box_size: int | None,
    voxel_size: float | None,
    class_map: dict[str, int] | None,
    normalization: str,
    source_split_hash: str,
) -> dict[str, Any]:
    return {
        "split": split,
        "part": part,
        "sample_count": int(data.shape[0]),
        "data_shape": list(data.shape),
        "label_shape": list(labels.shape),
        "channels": channels,
        "box_size": box_size,
        "voxel_size": voxel_size,
        "class_map": class_map,
        "normalization": normalization,
        "source_split_hash": source_split_hash,
    }


def write_npz_dataset(output_file: Path, data: np.ndarray, labels: np.ndarray, metadata: dict[str, Any], *, compressed: bool) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez_compressed if compressed else np.savez
    saver(output_file, data=data, labels=labels)
    output_file.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def write_hdf5_dataset(output_file: Path, data: np.ndarray, labels: np.ndarray, metadata: dict[str, Any], *, compression: str, chunk_size: int | None) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise ConversionError("h5py is required for --format hdf5") from exc

    output_file.parent.mkdir(parents=True, exist_ok=True)
    chunks = (chunk_size, *data.shape[1:]) if chunk_size else None

    with h5py.File(output_file, "w") as h5f:
        h5f.create_dataset("data", data=data, compression=compression, chunks=chunks)
        h5f.create_dataset("labels", data=labels, compression=compression)
        for key, value in metadata.items():
            h5f.attrs[key] = json.dumps(value) if isinstance(value, (dict, list)) else value


def parse_parts(parts: str) -> list[int]:
    values: list[int] = []
    for token in parts.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    if not values:
        raise ValueError("No valid parts were provided.")
    return sorted(set(values))


def _parse_class_map(class_map_raw: str | None) -> dict[str, int] | None:
    if not class_map_raw:
        return None
    candidate = Path(class_map_raw)
    if candidate.exists():
        return json.loads(candidate.read_text())
    return json.loads(class_map_raw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampled-root", type=Path, default=Path("../data/Sampled_Numpy"))
    parser.add_argument("--output-root", type=Path, default=Path("../data/ATOM_CHANNEL_dataset"))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated list, e.g. train,val,test")
    parser.add_argument("--parts", default="1-6", help="Part indices, e.g. 1-6 or 1,2,3")
    parser.add_argument("--train-shards", type=int, default=19)
    parser.add_argument("--format", choices=("npz", "hdf5"), default="npz")
    parser.add_argument("--compressed", action="store_true", default=True)
    parser.add_argument("--no-compressed", action="store_false", dest="compressed")
    parser.add_argument("--hdf5-compression", default="gzip")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--box-size", type=int, default=None)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--class-map", default=None, help="JSON string or path to JSON file")
    parser.add_argument("--normalization", default="none")
    parser.add_argument("--source-split-hash", default="")
    return parser


def convert_dataset(args: argparse.Namespace) -> None:
    parts = parse_parts(args.parts)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    class_map = _parse_class_map(args.class_map)

    for split in splits:
        for part in parts:
            data, labels = load_split_part(args.sampled_root, split, part, train_shards=args.train_shards)
            metadata = build_metadata(
                split=split,
                part=part,
                data=data,
                labels=labels,
                channels=args.channels,
                box_size=args.box_size,
                voxel_size=args.voxel_size,
                class_map=class_map,
                normalization=args.normalization,
                source_split_hash=args.source_split_hash,
            )

            ext = "npz" if args.format == "npz" else "h5"
            output_file = args.output_root / f"{split}_data_{part}.{ext}"

            if args.format == "npz":
                write_npz_dataset(output_file, data, labels, metadata, compressed=args.compressed)
            else:
                write_hdf5_dataset(
                    output_file,
                    data,
                    labels,
                    metadata,
                    compression=args.hdf5_compression,
                    chunk_size=args.chunk_size,
                )

            print(f"Wrote {output_file}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    convert_dataset(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

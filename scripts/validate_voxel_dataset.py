#!/usr/bin/env python3
"""Validate voxel dataset manifests and sample `.npz` files.

This script is intended to run after `build_voxel_dataset.py` and before
`compute_normalization.py` / `train_voxel_cnn.py` to catch manifest and sample
corruption early.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


TASK_CHOICES = ["residue_identity", "mutation_activity", "regression"]
ERROR_PRINT_LIMIT = 10
JSON_ERROR_LIMIT = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a voxel dataset manifest and its .npz samples for corruption, "
            "missing files, label mismatches, and shape issues."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest CSV")
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="Optional root directory for resolving relative sample paths (default: manifest parent)",
    )
    parser.add_argument(
        "--task",
        choices=TASK_CHOICES,
        default=None,
        help="Optional task type used for label semantics",
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=None,
        help="Expected first dimension of x when x.ndim == 4",
    )
    parser.add_argument(
        "--expected-box-size",
        type=int,
        default=None,
        help="Expected spatial voxel size (e.g. 20 for 20x20x20)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Validate only first N rows (useful for quick checks)",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately on first validation error")
    parser.add_argument("--check-files", action="store_true", help="Check sample files exist and can be opened")
    parser.add_argument("--check-labels", action="store_true", help="Compare manifest label to npz['y'] when available")
    parser.add_argument("--check-shape", action="store_true", help="Validate tensor rank and voxel shape")
    parser.add_argument("--check-duplicates", action="store_true", help="Check duplicate identifiers and sample paths")
    parser.add_argument("--verbose", action="store_true", help="Print all errors to stderr (not only first 10)")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to write validation summary JSON")
    return parser.parse_args()


def resolve_path(raw_path: str, root_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def parse_manifest_label(label: str, task: Optional[str]) -> Any:
    text = label.strip()
    if task == "regression":
        return float(text)

    # Classification/unknown task: prefer ints, then float, then raw string.
    try:
        return int(text)
    except ValueError:
        pass
    try:
        value = float(text)
        if value.is_integer():
            return int(value)
        return value
    except ValueError:
        return text


def extract_npz_scalar(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr.tolist()


def values_match(manifest_value: Any, npz_value: Any, task: Optional[str]) -> bool:
    if task == "regression":
        try:
            return np.isclose(float(manifest_value), float(npz_value), equal_nan=True)
        except Exception:
            return False

    if isinstance(manifest_value, (int, np.integer)):
        try:
            return int(npz_value) == int(manifest_value)
        except Exception:
            return False
    return str(npz_value) == str(manifest_value)


def validate_row(
    row: List[str],
    row_num: int,
    header_len: int,
    idx: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str], Optional[str], Optional[Tuple[str, str, str, str]]]:
    errors: List[Dict[str, Any]] = []

    if len(row) != header_len:
        merged_hints: List[str] = []
        npz_hits = sum(cell.lower().count(".npz") for cell in row)
        if len(row) > header_len:
            merged_hints.append("row has more columns than header")
        if npz_hits > 1:
            merged_hints.append("multiple '.npz' substrings found")
        if any(re.match(r"^[A-Za-z0-9_.:-]+_[A-Za-z0-9_.:-]+_[A-Za-z0-9_.:-]+$", c.strip()) for c in row):
            merged_hints.append("embedded token resembles an example_id")

        msg = (
            f"row column count mismatch: expected {header_len}, got {len(row)}"
            + (f" ({'; '.join(merged_hints)})" if merged_hints else "")
        )
        errors.append({"row": row_num, "code": "corrupted_row", "message": msg})
        return errors, None, None, None, None

    path_value = row[idx["path"]].strip()
    label_value = row[idx["label"]].strip()

    if not path_value:
        errors.append({"row": row_num, "code": "empty_path", "message": "sample path is empty"})
    elif not path_value.lower().endswith(".npz"):
        errors.append(
            {
                "row": row_num,
                "code": "bad_path_extension",
                "message": f"sample path does not end with .npz: {path_value}",
            }
        )

    if not label_value:
        errors.append({"row": row_num, "code": "empty_label", "message": "label is empty"})

    example_id = row[idx["example_id"]].strip() if "example_id" in idx else None

    residue_key = None
    residue_fields = ["structure_id", "chain_id", "res_no", "res_name"]
    if all(field in idx for field in residue_fields):
        residue_key = tuple(row[idx[field]].strip() for field in residue_fields)

    return errors, path_value, label_value, example_id, residue_key


def validate_npz(
    npz_path: Path,
    manifest_label: str,
    row_num: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], bool]:
    errors: List[Dict[str, Any]] = []
    counted_unreadable = False

    try:
        with np.load(npz_path, allow_pickle=False) as obj:
            if "x" not in obj:
                errors.append({"row": row_num, "code": "npz_missing_x", "message": f"npz missing 'x': {npz_path}"})
                return errors, counted_unreadable

            x = obj["x"]

            if args.check_shape:
                if x.ndim not in (3, 4):
                    errors.append(
                        {
                            "row": row_num,
                            "code": "shape_mismatch",
                            "message": f"unexpected x.ndim={x.ndim}, expected 3 or 4 (shape={tuple(x.shape)})",
                        }
                    )
                else:
                    spatial = x.shape[-3:] if x.ndim == 4 else x.shape
                    if args.expected_channels is not None and x.ndim == 4 and x.shape[0] != args.expected_channels:
                        errors.append(
                            {
                                "row": row_num,
                                "code": "shape_mismatch",
                                "message": (
                                    f"unexpected channel count: got {x.shape[0]}, "
                                    f"expected {args.expected_channels} (shape={tuple(x.shape)})"
                                ),
                            }
                        )
                    if args.expected_box_size is not None:
                        expected_spatial = (args.expected_box_size,) * 3
                        if tuple(spatial) != expected_spatial:
                            errors.append(
                                {
                                    "row": row_num,
                                    "code": "shape_mismatch",
                                    "message": (
                                        f"unexpected spatial shape: got {tuple(spatial)}, "
                                        f"expected {expected_spatial} (full shape={tuple(x.shape)})"
                                    ),
                                }
                            )

            if args.check_labels:
                if "y" not in obj:
                    errors.append({"row": row_num, "code": "npz_missing_y", "message": f"npz missing 'y': {npz_path}"})
                else:
                    try:
                        manifest_value = parse_manifest_label(manifest_label, args.task)
                    except Exception as exc:
                        errors.append(
                            {
                                "row": row_num,
                                "code": "label_parse_error",
                                "message": f"could not parse manifest label '{manifest_label}': {exc}",
                            }
                        )
                    else:
                        npz_value = extract_npz_scalar(obj["y"])
                        if not values_match(manifest_value, npz_value, args.task):
                            errors.append(
                                {
                                    "row": row_num,
                                    "code": "label_mismatch",
                                    "message": (
                                        f"manifest label ({manifest_value}) != npz y ({npz_value}) "
                                        f"for {npz_path}"
                                    ),
                                }
                            )
    except Exception as exc:
        counted_unreadable = True
        errors.append({"row": row_num, "code": "unreadable_file", "message": f"failed to read npz '{npz_path}': {exc}"})

    return errors, counted_unreadable


def main() -> int:
    args = parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.exists():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    root_dir = args.root_dir.resolve() if args.root_dir else manifest_path.parent.resolve()

    summary: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "root_dir": str(root_dir),
        "task": args.task,
        "total_rows_scanned": 0,
        "rows_validated": 0,
        "corrupted_rows": 0,
        "missing_files": 0,
        "unreadable_files": 0,
        "npz_missing_x": 0,
        "label_mismatches": 0,
        "shape_mismatches": 0,
        "duplicate_example_ids": 0,
        "duplicate_paths": 0,
        "duplicate_residue_keys": 0,
        "validation_passed": True,
    }

    errors: List[Dict[str, Any]] = []

    def add_error(err: Dict[str, Any]) -> None:
        errors.append(err)
        if err["code"] == "corrupted_row":
            summary["corrupted_rows"] += 1
        elif err["code"] == "missing_file":
            summary["missing_files"] += 1
        elif err["code"] == "unreadable_file":
            summary["unreadable_files"] += 1
        elif err["code"] == "npz_missing_x":
            summary["npz_missing_x"] += 1
        elif err["code"] == "label_mismatch":
            summary["label_mismatches"] += 1
        elif err["code"] == "shape_mismatch":
            summary["shape_mismatches"] += 1
        elif err["code"] == "duplicate_example_id":
            summary["duplicate_example_ids"] += 1
        elif err["code"] == "duplicate_path":
            summary["duplicate_paths"] += 1
        elif err["code"] == "duplicate_residue_key":
            summary["duplicate_residue_keys"] += 1

        should_print = args.verbose or len(errors) <= ERROR_PRINT_LIMIT
        if should_print:
            print(f"[row {err['row']}] {err['code']}: {err['message']}", file=sys.stderr)
        elif len(errors) == ERROR_PRINT_LIMIT + 1:
            print(
                f"... additional errors suppressed (use --verbose to print all). Total errors so far: {len(errors)}",
                file=sys.stderr,
            )

        if args.fail_fast:
            raise RuntimeError("fail_fast_triggered")

    seen_example_ids: set[str] = set()
    seen_paths: set[str] = set()
    seen_residue_keys: set[Tuple[str, str, str, str]] = set()

    try:
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None or not header:
                print("error: manifest is empty or missing header", file=sys.stderr)
                return 1

            header = [h.strip() for h in header]
            idx = {name: i for i, name in enumerate(header)}

            path_col = "sample_path" if "sample_path" in idx else "path" if "path" in idx else None
            if path_col is None:
                print("error: manifest missing required path column ('sample_path' or 'path')", file=sys.stderr)
                return 1
            if "label" not in idx:
                print("error: manifest missing required column 'label'", file=sys.stderr)
                return 1

            idx["path"] = idx[path_col]
            header_len = len(header)

            progress_total = args.sample_limit if args.sample_limit is not None else None
            iterator = tqdm(reader, total=progress_total, desc="Validating manifest", unit=" rows")

            for row_num, row in enumerate(iterator, start=2):
                if args.sample_limit is not None and summary["total_rows_scanned"] >= args.sample_limit:
                    break

                summary["total_rows_scanned"] += 1
                row_errors, sample_path_raw, label_raw, example_id, residue_key = validate_row(row, row_num, header_len, idx)
                if row_errors:
                    for err in row_errors:
                        add_error(err)
                    continue

                assert sample_path_raw is not None
                assert label_raw is not None

                if args.check_duplicates:
                    if sample_path_raw in seen_paths:
                        add_error(
                            {
                                "row": row_num,
                                "code": "duplicate_path",
                                "message": f"duplicate sample path: {sample_path_raw}",
                            }
                        )
                    else:
                        seen_paths.add(sample_path_raw)

                    if example_id:
                        if example_id in seen_example_ids:
                            add_error(
                                {
                                    "row": row_num,
                                    "code": "duplicate_example_id",
                                    "message": f"duplicate example_id: {example_id}",
                                }
                            )
                        else:
                            seen_example_ids.add(example_id)

                    if residue_key and all(residue_key):
                        if residue_key in seen_residue_keys:
                            add_error(
                                {
                                    "row": row_num,
                                    "code": "duplicate_residue_key",
                                    "message": f"duplicate residue tuple: {residue_key}",
                                }
                            )
                        else:
                            seen_residue_keys.add(residue_key)

                if args.check_files:
                    sample_path = resolve_path(sample_path_raw, root_dir)
                    if not sample_path.exists():
                        add_error(
                            {
                                "row": row_num,
                                "code": "missing_file",
                                "message": f"sample file not found: {sample_path}",
                            }
                        )
                    elif not sample_path.is_file():
                        add_error(
                            {
                                "row": row_num,
                                "code": "missing_file",
                                "message": f"sample path is not a file: {sample_path}",
                            }
                        )
                    else:
                        npz_errors, _ = validate_npz(sample_path, label_raw, row_num, args)
                        for err in npz_errors:
                            add_error(err)

                summary["rows_validated"] += 1

    except RuntimeError as exc:
        if str(exc) != "fail_fast_triggered":
            raise

    summary["validation_passed"] = len(errors) == 0

    print("\nValidation summary:")
    print(f"  total rows scanned:      {summary['total_rows_scanned']}")
    print(f"  rows validated:          {summary['rows_validated']}")
    print(f"  corrupted rows:          {summary['corrupted_rows']}")
    print(f"  missing files:           {summary['missing_files']}")
    print(f"  unreadable files:        {summary['unreadable_files']}")
    print(f"  npz missing x:           {summary['npz_missing_x']}")
    print(f"  label mismatches:        {summary['label_mismatches']}")
    print(f"  shape mismatches:        {summary['shape_mismatches']}")
    print(f"  duplicate example IDs:   {summary['duplicate_example_ids']}")
    print(f"  duplicate paths:         {summary['duplicate_paths']}")
    print(f"  duplicate residue keys:  {summary['duplicate_residue_keys']}")
    print(f"  validation passed:       {summary['validation_passed']}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": summary,
            "errors": errors[:JSON_ERROR_LIMIT],
            "error_count": len(errors),
            "error_limit": JSON_ERROR_LIMIT,
        }
        with args.output_json.open("w", encoding="utf-8") as out_handle:
            json.dump(payload, out_handle, indent=2)

    return 0 if summary["validation_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

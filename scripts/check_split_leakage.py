#!/usr/bin/env python3
"""Check for duplicate examples and split leakage across train/val/test manifests."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

DEFAULT_CHECK_COLS = ("pdb_id", "chain_id", "sequence_hash")


@dataclass
class LeakageReport:
    check_columns: Tuple[str, ...]
    duplicate_counts: Dict[str, int]
    overlap_counts: Dict[str, int]
    missing_columns: Dict[str, List[str]]

    @property
    def has_issues(self) -> bool:
        return any(self.duplicate_counts.values()) or any(self.overlap_counts.values())


def parse_check_columns(raw: str | None) -> Tuple[str, ...]:
    if not raw:
        return DEFAULT_CHECK_COLS
    cols = tuple(col.strip() for col in raw.split(",") if col.strip())
    if not cols:
        raise ValueError("--check-cols must include at least one column")
    return cols


def load_manifest(path: Path) -> List[Dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".parquet", ".pq"}:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only if parquet used without pandas
            raise RuntimeError(
                "Parquet manifest support requires pandas. Install pandas or provide CSV manifests."
            ) from exc
        frame = pd.read_parquet(path)
        return frame.fillna("").astype(str).to_dict(orient="records")
    raise ValueError(f"Unsupported manifest format for {path}. Use CSV or Parquet.")


def _key(row: Dict[str, str], columns: Sequence[str]) -> Tuple[str, ...]:
    return tuple((row.get(col, "") or "").strip() for col in columns)


def _count_duplicates(rows: Iterable[Dict[str, str]], columns: Sequence[str]) -> int:
    seen = set()
    duplicates = 0
    for row in rows:
        key = _key(row, columns)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def evaluate_leakage(
    split_rows: Dict[str, List[Dict[str, str]]],
    check_columns: Sequence[str],
) -> LeakageReport:
    missing_columns: Dict[str, List[str]] = {}
    duplicates: Dict[str, int] = {}
    split_sets: Dict[str, set[Tuple[str, ...]]] = {}

    for split_name, rows in split_rows.items():
        header = set(rows[0].keys()) if rows else set()
        missing = [col for col in check_columns if col not in header]
        if missing:
            missing_columns[split_name] = missing
        duplicates[split_name] = _count_duplicates(rows, check_columns)
        split_sets[split_name] = {_key(row, check_columns) for row in rows}

    overlaps: Dict[str, int] = {}
    for left, right in combinations(split_rows, 2):
        pair_name = f"{left}__{right}"
        overlaps[pair_name] = len(split_sets[left].intersection(split_sets[right]))

    return LeakageReport(
        check_columns=tuple(check_columns),
        duplicate_counts=duplicates,
        overlap_counts=overlaps,
        missing_columns=missing_columns,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path, help="Path to train split manifest")
    parser.add_argument("--val", required=True, type=Path, help="Path to validation split manifest")
    parser.add_argument("--test", required=True, type=Path, help="Path to test split manifest")
    parser.add_argument(
        "--check-cols",
        default=",".join(DEFAULT_CHECK_COLS),
        help="Comma-separated key columns to check for duplicates/leakage",
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=None,
        help=(
            "Optional advisory threshold for sequence identity-based splits; "
            "currently reported only in output metadata."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    check_columns = parse_check_columns(args.check_cols)
    split_rows = {
        "train": load_manifest(args.train),
        "val": load_manifest(args.val),
        "test": load_manifest(args.test),
    }

    report = evaluate_leakage(split_rows, check_columns)
    output = {
        "check_columns": report.check_columns,
        "identity_threshold": args.identity_threshold,
        "duplicate_counts": report.duplicate_counts,
        "overlap_counts": report.overlap_counts,
        "missing_columns": report.missing_columns,
        "has_issues": report.has_issues,
    }
    print(json.dumps(output, indent=2, sort_keys=True))

    if report.missing_columns:
        parser.error(
            "One or more required columns are missing from manifests: "
            + json.dumps(report.missing_columns)
        )

    return 1 if report.has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())

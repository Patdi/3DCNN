#!/usr/bin/env python3
"""Repair or rebuild voxel manifests from on-disk .npz files.

This tool can:
- rebuild a manifest directly from discovered .npz files,
- salvage valid rows from a possibly corrupted CSV manifest,
- merge salvaged and disk-derived records while preferring disk truth,
- validate and atomically write a canonical output manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

CANONICAL_FIELDS = [
    "example_id",
    "structure_id",
    "chain_id",
    "res_no",
    "res_name",
    "label",
    "sample_path",
    "task",
    "channel_scheme",
]

TASK_CHOICES = ["residue_identity", "mutation_activity", "regression"]


@dataclass
class ValidationResult:
    ok: bool
    has_x: bool
    label: Optional[str]
    error: Optional[str] = None


@dataclass
class Summary:
    files_discovered_on_disk: int = 0
    rows_salvaged_from_manifest: int = 0
    rows_recovered_from_disk: int = 0
    duplicates_removed: int = 0
    invalid_files_skipped: int = 0
    rows_written: int = 0
    rows_missing_metadata: int = 0
    likely_concatenated_rows_detected: int = 0
    parse_failures: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "files_discovered_on_disk": self.files_discovered_on_disk,
            "rows_salvaged_from_manifest": self.rows_salvaged_from_manifest,
            "rows_recovered_from_disk": self.rows_recovered_from_disk,
            "duplicates_removed": self.duplicates_removed,
            "invalid_files_skipped": self.invalid_files_skipped,
            "rows_written": self.rows_written,
            "rows_missing_metadata": self.rows_missing_metadata,
            "likely_concatenated_rows_detected": self.likely_concatenated_rows_detected,
            "parse_failures": self.parse_failures,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair/rebuild voxel manifest CSV files")
    parser.add_argument("--input-manifest", type=Path, default=None)
    parser.add_argument("--voxel-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--task", required=True, choices=TASK_CHOICES)
    parser.add_argument("--channel-scheme", type=str, default="")
    parser.add_argument("--rebuild-only", action="store_true")
    parser.add_argument("--salvage-existing", action="store_true")
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--validate-files", action="store_true")
    parser.add_argument("--infer-labels", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def normalize_sample_path(path: Path) -> str:
    return str(path.resolve())


def discover_npz_files(voxel_root: Path) -> List[Path]:
    return sorted(p for p in voxel_root.rglob("*.npz") if p.is_file())


def _find_chain_residue_tokens(tokens: List[str]) -> Optional[Tuple[int, str, int, str]]:
    """Return (chain_idx, chain_id, res_no, res_name) from right-oriented parse."""
    for idx in range(len(tokens) - 3, -1, -1):
        chain_tok = tokens[idx]
        res_no_tok = tokens[idx + 1]
        res_name_tok = tokens[idx + 2]
        try:
            res_no_val = int(res_no_tok)
        except ValueError:
            continue
        if not chain_tok:
            continue
        if not res_name_tok:
            continue
        return idx, chain_tok, res_no_val, res_name_tok
    return None


def infer_metadata_from_npz_path(path: Path, voxel_root: Path) -> Tuple[Dict[str, str], bool]:
    """Infer metadata conservatively from path and filename.

    Returns tuple of (row, parse_clean) where parse_clean indicates whether
    key metadata (chain_id/res_no/res_name) was inferred confidently.
    """
    filename = path.name
    example_id = path.stem
    parent_structure = path.parent.name if path.parent else ""

    row: Dict[str, str] = {
        "example_id": example_id,
        "structure_id": parent_structure,
        "chain_id": "",
        "res_no": "",
        "res_name": "",
        "label": "",
        "sample_path": normalize_sample_path(path),
        "task": "",
        "channel_scheme": "",
    }

    tokens = example_id.split("_")
    parse = _find_chain_residue_tokens(tokens)
    parse_clean = False
    if parse is not None:
        chain_idx, chain_id, res_no, res_name = parse
        row["chain_id"] = chain_id
        row["res_no"] = str(res_no)
        row["res_name"] = res_name
        if not row["structure_id"]:
            row["structure_id"] = "_".join(tokens[:chain_idx]) if chain_idx > 0 else ""
        parse_clean = True
    else:
        if not row["structure_id"] and len(tokens) > 3:
            row["structure_id"] = "_".join(tokens[:-3])

    try:
        rel = path.resolve().relative_to(voxel_root.resolve())
        if len(rel.parts) >= 2:
            row["structure_id"] = rel.parts[0] or row["structure_id"]
    except Exception:
        pass

    return row, parse_clean


def _sanitize_field(value: object) -> str:
    s = "" if value is None else str(value)
    return s.replace("\n", " ").replace("\r", " ").strip()


def _coerce_label(raw: object, task: str) -> str:
    if raw is None:
        return ""
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return ""
        raw = raw.reshape(-1)[0].item()
    if isinstance(raw, np.generic):
        raw = raw.item()

    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    raw_s = str(raw).strip()
    if raw_s == "":
        return ""

    if task == "regression":
        return str(float(raw_s))
    return str(int(float(raw_s)))


def validate_npz(path: Path, task: str, infer_labels: bool) -> ValidationResult:
    try:
        with np.load(path, allow_pickle=False) as obj:
            if "x" not in obj:
                return ValidationResult(ok=False, has_x=False, label=None, error="missing_x")
            label = None
            if infer_labels and "y" in obj:
                label = _coerce_label(obj["y"], task)
            return ValidationResult(ok=True, has_x=True, label=label)
    except Exception as exc:  # noqa: BLE001
        return ValidationResult(ok=False, has_x=False, label=None, error=str(exc))


def salvage_manifest_rows(manifest_path: Path, summary: Summary, verbose: bool = False) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    npz_pattern = re.compile(r"[^,\s\"']+\.npz")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            return rows

        header = [_sanitize_field(h) for h in header]
        header_len = len(header)

        for raw_row in tqdm(reader, desc="Salvaging manifest", unit="row"):
            if not raw_row:
                continue
            row = [_sanitize_field(c) for c in raw_row]
            joined = ",".join(row)
            npz_hits = npz_pattern.findall(joined)
            if len(npz_hits) > 1:
                summary.likely_concatenated_rows_detected += 1
                if verbose:
                    print(f"[warn] likely concatenated row with multiple .npz: {npz_hits[:2]}...", file=sys.stderr)

            if len(row) == header_len:
                mapped = {k: v for k, v in zip(header, row)}
                canonical = {f: _sanitize_field(mapped.get(f, "")) for f in CANONICAL_FIELDS}
                rows.append(canonical)
                continue

            recovered_paths = []
            for hit in npz_hits:
                p = Path(hit)
                if p.suffix != ".npz":
                    p = Path(f"{hit}.npz")
                recovered_paths.append(str(p))

            if not recovered_paths:
                summary.parse_failures += 1
                continue

            for p in recovered_paths:
                canonical = {f: "" for f in CANONICAL_FIELDS}
                canonical["sample_path"] = p
                base = Path(p).stem
                canonical["example_id"] = base
                rows.append(canonical)

    return rows


def _completeness_score(row: Dict[str, str]) -> int:
    keys = ["example_id", "structure_id", "chain_id", "res_no", "res_name", "label", "sample_path"]
    return sum(1 for k in keys if _sanitize_field(row.get(k, "")) != "")


def _prefer_row(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    a_disk = a.get("_source") == "disk"
    b_disk = b.get("_source") == "disk"
    a_val = a.get("_validated") == "1"
    b_val = b.get("_validated") == "1"

    if a_val != b_val:
        return a if a_val else b
    if a_disk != b_disk:
        return a if a_disk else b

    a_score = _completeness_score(a)
    b_score = _completeness_score(b)
    if a_score != b_score:
        return a if a_score >= b_score else b

    return a


def merge_rows(existing_rows: List[Dict[str, str]], disk_rows: List[Dict[str, str]], dedupe: bool, summary: Summary) -> List[Dict[str, str]]:
    merged = list(existing_rows) + list(disk_rows)
    if not dedupe:
        return merged

    by_path: Dict[str, Dict[str, str]] = {}
    for row in merged:
        path_key = _sanitize_field(row.get("sample_path", ""))
        if not path_key:
            continue
        prev = by_path.get(path_key)
        if prev is None:
            by_path[path_key] = row
        else:
            by_path[path_key] = _prefer_row(prev, row)
            summary.duplicates_removed += 1

    rows = list(by_path.values())

    by_example: Dict[str, Dict[str, str]] = {}
    final_rows: List[Dict[str, str]] = []
    for row in rows:
        ex_id = _sanitize_field(row.get("example_id", ""))
        if not ex_id:
            final_rows.append(row)
            continue
        prev = by_example.get(ex_id)
        if prev is None:
            by_example[ex_id] = row
        else:
            chosen = _prefer_row(prev, row)
            by_example[ex_id] = chosen
            summary.duplicates_removed += 1

    # keep non-example rows then example rows
    final_rows.extend(by_example.values())
    return final_rows


def _row_missing_metadata(row: Dict[str, str]) -> bool:
    essential = ["example_id", "structure_id", "chain_id", "res_no", "res_name"]
    return any(_sanitize_field(row.get(k, "")) == "" for k in essential)


def _to_canonical_row(
    row: Dict[str, str],
    task: str,
    channel_scheme: str,
) -> Dict[str, str]:
    out = {f: _sanitize_field(row.get(f, "")) for f in CANONICAL_FIELDS}
    out["task"] = task
    out["channel_scheme"] = channel_scheme

    if out["label"] != "":
        out["label"] = _coerce_label(out["label"], task)

    if out["res_no"] != "":
        out["res_no"] = str(int(float(out["res_no"])))

    return out


def atomic_write_manifest(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CANONICAL_FIELDS})
    os.replace(tmp_path, output_path)


def build_disk_rows(args: argparse.Namespace, summary: Summary) -> List[Dict[str, str]]:
    files = discover_npz_files(args.voxel_root)
    summary.files_discovered_on_disk = len(files)

    rows: List[Dict[str, str]] = []
    parse_failures = 0

    for path in tqdm(files, desc="Scanning .npz files", unit="file"):
        row, parse_clean = infer_metadata_from_npz_path(path, args.voxel_root)
        row["task"] = args.task
        row["channel_scheme"] = args.channel_scheme
        row["_source"] = "disk"
        row["_validated"] = "0"

        if args.strict and not parse_clean:
            parse_failures += 1
            continue

        rows.append(row)

    if parse_failures and args.strict:
        raise RuntimeError(f"Strict mode enabled: failed metadata inference for {parse_failures} files")

    summary.rows_recovered_from_disk = len(rows)
    return rows


def validate_and_finalize_rows(
    rows: List[Dict[str, str]],
    args: argparse.Namespace,
    summary: Summary,
) -> List[Dict[str, str]]:
    finalized: List[Dict[str, str]] = []

    iterator: Iterable[Dict[str, str]]
    if args.validate_files:
        iterator = tqdm(rows, desc="Validating rows", unit="row")
    else:
        iterator = rows

    for row in iterator:
        sample_path_s = _sanitize_field(row.get("sample_path", ""))
        if not sample_path_s:
            summary.invalid_files_skipped += 1
            continue

        p = Path(sample_path_s)
        if p.suffix != ".npz":
            p = Path(f"{sample_path_s}.npz")

        p = p.resolve()
        if not p.exists() or not p.is_file():
            summary.invalid_files_skipped += 1
            continue

        row["sample_path"] = str(p)

        if args.validate_files:
            v = validate_npz(p, args.task, args.infer_labels)
            if not v.ok:
                summary.invalid_files_skipped += 1
                continue
            row["_validated"] = "1"
            if args.infer_labels and v.label is not None and v.label != "":
                row["label"] = v.label

        try:
            canonical = _to_canonical_row(row, args.task, args.channel_scheme)
        except Exception:
            summary.invalid_files_skipped += 1
            continue

        if _row_missing_metadata(canonical):
            summary.rows_missing_metadata += 1

        finalized.append(canonical)

    return finalized


def print_summary(summary: Summary) -> None:
    print("Repair summary:")
    for key, value in summary.as_dict().items():
        print(f"- {key.replace('_', ' ')}: {value}")


def main() -> int:
    args = parse_args()
    summary = Summary()

    if not args.voxel_root.exists() or not args.voxel_root.is_dir():
        raise FileNotFoundError(f"voxel root not found or not a directory: {args.voxel_root}")

    if args.rebuild_only and args.salvage_existing:
        raise ValueError("--rebuild-only and --salvage-existing are mutually exclusive")

    existing_rows: List[Dict[str, str]] = []
    if args.salvage_existing:
        if args.input_manifest is None:
            raise ValueError("--salvage-existing requires --input-manifest")
        if not args.input_manifest.exists():
            raise FileNotFoundError(f"input manifest not found: {args.input_manifest}")
        existing_rows = salvage_manifest_rows(args.input_manifest, summary, verbose=args.verbose)
        for row in existing_rows:
            row["_source"] = "manifest"
            row["_validated"] = "0"
        summary.rows_salvaged_from_manifest = len(existing_rows)

    disk_rows = build_disk_rows(args, summary)

    if args.rebuild_only:
        merged = disk_rows
    else:
        merged = merge_rows(existing_rows, disk_rows, dedupe=args.dedupe, summary=summary)

    finalized = validate_and_finalize_rows(merged, args, summary)

    if args.strict:
        missing = [r for r in finalized if _row_missing_metadata(r)]
        if missing:
            raise RuntimeError(f"Strict mode enabled: {len(missing)} rows missing metadata")

    atomic_write_manifest(finalized, args.output_manifest)
    summary.rows_written = len(finalized)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary.as_dict(), handle, indent=2, sort_keys=True)

    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

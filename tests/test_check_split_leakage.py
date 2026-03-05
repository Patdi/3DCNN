from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import importlib.util


module_path = Path(__file__).resolve().parents[1] / "scripts" / "check_split_leakage.py"
spec = importlib.util.spec_from_file_location("check_split_leakage", module_path)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)

evaluate_leakage = module.evaluate_leakage
parse_check_columns = module.parse_check_columns
infer_default_check_columns = module.infer_default_check_columns


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["pdb_id", "chain_id", "sequence_hash"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_parse_check_columns_defaults_and_custom() -> None:
    assert parse_check_columns(None) == ("structure_id",)
    assert parse_check_columns(None, ("example_id",)) == ("example_id",)
    assert parse_check_columns("pdb_id, chain_id") == ("pdb_id", "chain_id")


def test_infer_default_check_columns_prefers_example_id() -> None:
    split_rows = {
        "train": [{"example_id": "e1", "structure_id": "s1"}],
        "val": [{"example_id": "e2", "structure_id": "s2"}],
        "test": [{"example_id": "e3", "structure_id": "s3"}],
    }

    assert infer_default_check_columns(split_rows) == ("example_id",)


def test_infer_default_check_columns_falls_back_to_structure_id() -> None:
    split_rows = {
        "train": [{"structure_id": "s1"}],
        "val": [{"structure_id": "s2"}],
        "test": [{"structure_id": "s3"}],
    }

    assert infer_default_check_columns(split_rows) == ("structure_id",)


def test_evaluate_leakage_detects_overlaps_and_duplicates() -> None:
    split_rows = {
        "train": [
            {"pdb_id": "1abc", "chain_id": "A", "sequence_hash": "h1"},
            {"pdb_id": "1abc", "chain_id": "A", "sequence_hash": "h1"},
        ],
        "val": [{"pdb_id": "1abc", "chain_id": "A", "sequence_hash": "h1"}],
        "test": [{"pdb_id": "2xyz", "chain_id": "B", "sequence_hash": "h2"}],
    }

    report = evaluate_leakage(split_rows, ("pdb_id", "chain_id", "sequence_hash"))

    assert report.duplicate_counts["train"] == 1
    assert report.overlap_counts["train__val"] == 1
    assert report.has_issues is True


def test_cli_returns_zero_when_no_leakage(tmp_path: Path) -> None:
    train = tmp_path / "train.csv"
    val = tmp_path / "val.csv"
    test = tmp_path / "test.csv"

    _write_csv(train, [{"structure_id": "1abc", "pdb_path": "1abc.pdb", "split": "train"}])
    _write_csv(val, [{"structure_id": "2abc", "pdb_path": "2abc.pdb", "split": "val"}])
    _write_csv(test, [{"structure_id": "3abc", "pdb_path": "3abc.pdb", "split": "test"}])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_split_leakage.py",
            "--train",
            str(train),
            "--val",
            str(val),
            "--test",
            str(test),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert payload["has_issues"] is False
    assert payload["check_columns"] == ["structure_id"]

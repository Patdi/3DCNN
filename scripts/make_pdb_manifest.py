#!/usr/bin/env python3
"""Build a structure-level manifest from a folder of PDB files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass(frozen=True)
class PDBRecord:
    pdb_id: str
    pdb_path: Path
    sequence: str
    chains: tuple[str, ...]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", required=True, type=Path, help="Folder containing PDB files")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV manifest path")
    parser.add_argument("--glob", default="*.pdb", help="Glob pattern for PDB files")
    parser.add_argument("--recursive", action="store_true", help="Search folders recursively")
    parser.add_argument("--min-residues", type=int, default=0, help="Require at least this many CA residues")
    parser.add_argument("--dedupe-by-sequence", action="store_true", help="Keep only one structure per sequence")
    return parser.parse_args(list(argv) if argv is not None else None)


def find_pdb_files(pdb_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(path for path in pdb_dir.rglob(pattern) if path.is_file())
    return sorted(path for path in pdb_dir.glob(pattern) if path.is_file())


def parse_pdb(path: Path) -> tuple[str, tuple[str, ...]]:
    residues: list[tuple[str, int, str, str]] = []
    seen = set()

    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                if line.startswith("ENDMDL"):
                    break
                continue

            atom = line[12:16].strip()
            if atom != "CA":
                continue

            chain = line[21].strip() or "_"
            res_name = line[17:20].strip()
            insertion = line[26].strip() or "_"
            try:
                res_no = int(line[22:26])
            except ValueError:
                continue

            key = (chain, res_no, insertion)
            if key in seen:
                continue

            seen.add(key)
            residues.append((chain, res_no, insertion, THREE_TO_ONE.get(res_name, "X")))

    residues.sort(key=lambda item: (item[0], item[1], item[2]))
    sequence = "".join(residue[-1] for residue in residues)
    chains = tuple(sorted({residue[0] for residue in residues}))
    return sequence, chains


def load_records(args: argparse.Namespace) -> list[PDBRecord]:
    files = find_pdb_files(args.pdb_dir, args.glob, args.recursive)
    records: list[PDBRecord] = []
    seen_sequences: set[str] = set()

    for path in files:
        sequence, chains = parse_pdb(path)
        if not sequence:
            continue
        if len(sequence) < args.min_residues:
            continue
        if args.dedupe_by_sequence and sequence in seen_sequences:
            continue
        seen_sequences.add(sequence)

        records.append(
            PDBRecord(
                pdb_id=path.stem,
                pdb_path=path.resolve(),
                sequence=sequence,
                chains=chains,
            )
        )

    return records


def write_manifest(records: list[PDBRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pdb_id", "pdb_path", "num_chains", "chains", "num_residues", "sequence"]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "pdb_id": record.pdb_id,
                    "pdb_path": str(record.pdb_path),
                    "num_chains": len(record.chains),
                    "chains": ";".join(record.chains),
                    "num_residues": len(record.sequence),
                    "sequence": record.sequence,
                }
            )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    records = load_records(args)
    if not records:
        raise ValueError(f"No parseable PDB files found in {args.pdb_dir}")

    write_manifest(records, args.out)
    print(f"Wrote {len(records)} structures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create train/val/test splits directly from a folder of PDB files.

Supports random structure-level split, sequence-cluster split, and scaffold-like split.
Outputs both TXT lists and CSV manifests for downstream legacy and modern scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

AA_CLASS = {
    "A": "H", "V": "H", "I": "H", "L": "H", "M": "H", "F": "H", "W": "H", "Y": "H",
    "S": "P", "T": "P", "N": "P", "Q": "P", "C": "P", "G": "S", "P": "S",
    "K": "C", "R": "C", "H": "C", "D": "C", "E": "C",
}


@dataclass(frozen=True)
class StructureRecord:
    structure_id: str
    pdb_path: Path
    sequence: str


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb-dir", required=True, type=Path, help="Folder containing .pdb/.ent files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder to write split outputs")
    parser.add_argument("--method", choices=["random", "sequence-cluster", "scaffold"], default="sequence-cluster")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seq-identity-threshold", type=float, default=0.4)
    parser.add_argument("--write-legacy-pdb-lists", action="store_true", help="Also write PDB_train.txt/PDB_val.txt/PDB_test.txt")
    parser.add_argument("--materialize", choices=["none", "symlink", "copy"], default="none", help="Create split subfolders")
    return parser.parse_args(list(argv) if argv is not None else None)


def read_pdb_sequence(pdb_path: Path) -> str:
    residues: list[tuple[str, int, str]] = []
    seen = set()
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                if line.startswith("ENDMDL"):
                    break
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            res = line[17:20].strip()
            chain = line[21].strip() or "_"
            try:
                res_no = int(line[22:26])
            except ValueError:
                continue
            key = (chain, res_no)
            if key in seen:
                continue
            seen.add(key)
            residues.append((chain, res_no, THREE_TO_ONE.get(res, "X")))
    residues.sort(key=lambda x: (x[0], x[1]))
    return "".join(r[-1] for r in residues)


def load_structures(pdb_dir: Path) -> list[StructureRecord]:
    paths = sorted([*pdb_dir.glob("*.pdb"), *pdb_dir.glob("*.ent")])
    rows: list[StructureRecord] = []
    for path in paths:
        seq = read_pdb_sequence(path)
        if not seq:
            continue
        rows.append(StructureRecord(structure_id=path.stem, pdb_path=path.resolve(), sequence=seq))
    if not rows:
        raise ValueError(f"No parseable PDB structures found in {pdb_dir}")
    return rows


def sequence_identity(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    min_len = min(len(a), len(b))
    matches = sum(1 for i in range(min_len) if a[i] == b[i])
    return matches / max_len


def scaffold_signature(sequence: str) -> str:
    reduced = "".join(AA_CLASS.get(x, "U") for x in sequence)
    compressed = []
    for c in reduced:
        if not compressed or compressed[-1] != c:
            compressed.append(c)
    length_bin = str((len(sequence) // 25) * 25)
    digest = hashlib.sha1((length_bin + "|" + "".join(compressed[:40])).encode()).hexdigest()[:10]
    return f"{length_bin}_{digest}"


def group_indices(records: Sequence[StructureRecord], method: str, threshold: float) -> list[list[int]]:
    if method == "random":
        return [[i] for i in range(len(records))]
    if method == "scaffold":
        groups: dict[str, list[int]] = {}
        for i, rec in enumerate(records):
            groups.setdefault(scaffold_signature(rec.sequence), []).append(i)
        return list(groups.values())

    # sequence-cluster: greedy leader clusters by sequence identity.
    leaders: list[int] = []
    clusters: list[list[int]] = []
    for i, rec in enumerate(records):
        assigned = False
        for cluster_idx, leader_idx in enumerate(leaders):
            if sequence_identity(rec.sequence, records[leader_idx].sequence) >= threshold:
                clusters[cluster_idx].append(i)
                assigned = True
                break
        if not assigned:
            leaders.append(i)
            clusters.append([i])
    return clusters


def assign_groups_to_splits(groups: Sequence[Sequence[int]], n_items: int, fractions: tuple[float, float, float], seed: int) -> dict[str, list[int]]:
    train_f, val_f, test_f = fractions
    targets = {
        "train": int(round(train_f * n_items)),
        "val": int(round(val_f * n_items)),
    }
    targets["test"] = n_items - targets["train"] - targets["val"]

    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    ordered = sorted(ordered, key=len, reverse=True)

    out = {"train": [], "val": [], "test": []}
    for group in ordered:
        choice = min(out.keys(), key=lambda s: (len(out[s]) - targets[s], len(out[s])))
        out[choice].extend(group)
    return out


def write_split_outputs(records: Sequence[StructureRecord], split_map: dict[str, list[int]], out_dir: Path, write_legacy_names: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["structure_id", "pdb_path", "sequence", "split"]

    for split_name, indices in split_map.items():
        selected = [records[i] for i in sorted(indices)]

        txt_path = out_dir / f"{split_name}.txt"
        txt_path.write_text("\n".join(r.structure_id for r in selected) + ("\n" if selected else ""))

        csv_path = out_dir / f"{split_name}.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for rec in selected:
                writer.writerow({
                    "structure_id": rec.structure_id,
                    "pdb_path": str(rec.pdb_path),
                    "sequence": rec.sequence,
                    "split": split_name,
                })

        if write_legacy_names:
            (out_dir / f"PDB_{split_name}.txt").write_text("\n".join(r.structure_id for r in selected) + ("\n" if selected else ""))


def materialize_split_folders(records: Sequence[StructureRecord], split_map: dict[str, list[int]], out_dir: Path, mode: str) -> None:
    if mode == "none":
        return
    import shutil

    for split_name, indices in split_map.items():
        split_dir = out_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            src = records[idx].pdb_path
            dst = split_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if mode == "symlink":
                dst.symlink_to(src)
            else:
                shutil.copy2(src, dst)


def validate_fractions(train_fraction: float, val_fraction: float, test_fraction: float) -> tuple[float, float, float]:
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train/val/test fractions must sum to 1.0")
    for value in (train_fraction, val_fraction, test_fraction):
        if value < 0:
            raise ValueError("fractions must be >= 0")
    return train_fraction, val_fraction, test_fraction


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    fractions = validate_fractions(args.train_fraction, args.val_fraction, args.test_fraction)
    records = load_structures(args.pdb_dir)
    groups = group_indices(records, method=args.method, threshold=args.seq_identity_threshold)
    split_map = assign_groups_to_splits(groups, n_items=len(records), fractions=fractions, seed=args.seed)
    write_split_outputs(records, split_map, args.output_dir, write_legacy_names=args.write_legacy_pdb_lists)
    materialize_split_folders(records, split_map, args.output_dir, mode=args.materialize)

    print(f"Wrote splits for {len(records)} structures to {args.output_dir} using method={args.method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build activity-labeled voxel datasets from PDB structures.

Supports two example units:
- whole_structure: one aggregated voxel tensor per structure.
- mutation_site: one residue-centered voxel tensor per activity row.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from build_voxel_dataset import (
    CHANNEL_SCHEMES,
    RES_LABEL_DICT,
    build_voxel_for_center,
    parse_pdb_atoms,
    residues_from_atoms,
    structure_from_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build activity-focused voxel dataset")
    parser.add_argument("--structure-manifest", required=True, type=Path)
    parser.add_argument("--activity-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--example-manifest-out", required=True, type=Path)
    parser.add_argument("--example-unit", choices=["whole_structure", "mutation_site"], default="mutation_site")
    parser.add_argument("--box-size", type=float, default=20.0)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--channel-scheme", choices=sorted(CHANNEL_SCHEMES), default="element4")
    parser.add_argument("--include-hetero", action="store_true")
    parser.add_argument("--atom-density-threshold", type=float, default=0.0)
    parser.add_argument("--task", choices=["regression"], default="regression")
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_activity_value(row: Dict[str, str]) -> float:
    raw = row.get("activity_value") or row.get("label") or row.get("y")
    if raw is None:
        raise ValueError("activity manifest must contain activity_value (or label/y) column")
    return float(raw)


def normalize_chain(row: Dict[str, str]) -> Optional[str]:
    chain = row.get("chain_id") or row.get("chain")
    if chain is None:
        return None
    chain = chain.strip()
    return chain if chain else "_"


def resolve_structure_map(rows: Sequence[Dict[str, str]], manifest_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for row in rows:
        structure_id, pdb_path = structure_from_row(row, manifest_dir)
        out[structure_id] = pdb_path
    return out


def aggregate_structure_voxel(site_voxels: Sequence[np.ndarray]) -> np.ndarray:
    stacked = np.stack(site_voxels, axis=0)
    return stacked.mean(axis=0, dtype=np.float32)


def write_manifest(rows: Iterable[Dict[str, str]], out_path: Path) -> None:
    rows = list(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "example_id",
        "structure_id",
        "chain_id",
        "res_no",
        "mutation",
        "label",
        "task",
        "sample_path",
        "channel_scheme",
        "example_unit",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_mutation_site_examples(
    structure_paths: Dict[str, Path],
    activity_rows: Sequence[Dict[str, str]],
    cfg: argparse.Namespace,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in activity_rows:
        structure_id = row.get("structure_id") or row.get("pdb_id") or row.get("id")
        if not structure_id:
            continue
        grouped[structure_id].append(row)

    manifest_rows: List[Dict[str, str]] = []
    stats = {"missing_structure": 0, "missing_residue": 0, "invalid_label": 0, "skipped_density": 0}

    for structure_id, rows in grouped.items():
        pdb_path = structure_paths.get(structure_id)
        if pdb_path is None:
            stats["missing_structure"] += len(rows)
            continue

        atoms = parse_pdb_atoms(pdb_path, include_hetero=cfg.include_hetero)
        residue_map = residues_from_atoms(atoms)
        out_dir = cfg.output_dir / structure_id
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(rows):
            chain = normalize_chain(row)
            res_no_raw = row.get("res_no") or row.get("resi") or row.get("position")
            mutation = row.get("mutation", "")
            if res_no_raw is None:
                stats["missing_residue"] += 1
                continue
            res_no = int(res_no_raw)

            keys = [k for k in residue_map.keys() if k[1] == res_no]
            if chain is not None:
                keys = [k for k in keys if k[0] == chain]

            if len(keys) != 1:
                stats["missing_residue"] += 1
                continue
            key = keys[0]
            residue_atoms = residue_map[key]
            res_name = key[2]
            if res_name not in RES_LABEL_DICT:
                stats["invalid_label"] += 1
                continue
            res_label = RES_LABEL_DICT[res_name]
            voxel = build_voxel_for_center(
                all_atoms=atoms,
                residue_atoms=residue_atoms,
                label=res_label,
                channel_scheme=cfg.channel_scheme,
                box_size=cfg.box_size,
                voxel_size=cfg.voxel_size,
                atom_density_threshold=cfg.atom_density_threshold,
            )
            if voxel is None:
                stats["skipped_density"] += 1
                continue

            y_value = parse_activity_value(row)
            example_id = f"{structure_id}_{key[0]}_{res_no}_{i}"
            out_path = out_dir / f"{example_id}.npz"
            np.savez_compressed(out_path, x=voxel.astype(np.float32), y=np.float32(y_value))
            manifest_rows.append(
                {
                    "example_id": example_id,
                    "structure_id": structure_id,
                    "chain_id": key[0],
                    "res_no": str(res_no),
                    "mutation": mutation,
                    "label": str(y_value),
                    "task": cfg.task,
                    "sample_path": str(out_path),
                    "channel_scheme": cfg.channel_scheme,
                    "example_unit": cfg.example_unit,
                }
            )

    return manifest_rows, stats


def build_whole_structure_examples(
    structure_paths: Dict[str, Path],
    activity_rows: Sequence[Dict[str, str]],
    cfg: argparse.Namespace,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    manifest_rows: List[Dict[str, str]] = []
    stats = {"missing_structure": 0, "missing_sites": 0, "invalid_label": 0, "skipped_density": 0}

    for row in activity_rows:
        structure_id = row.get("structure_id") or row.get("pdb_id") or row.get("id")
        if not structure_id:
            continue
        pdb_path = structure_paths.get(structure_id)
        if pdb_path is None:
            stats["missing_structure"] += 1
            continue

        atoms = parse_pdb_atoms(pdb_path, include_hetero=cfg.include_hetero)
        residue_map = residues_from_atoms(atoms)
        site_voxels: List[np.ndarray] = []
        for (_, _, res_name), residue_atoms in residue_map.items():
            if res_name not in RES_LABEL_DICT:
                stats["invalid_label"] += 1
                continue
            res_label = RES_LABEL_DICT[res_name]
            voxel = build_voxel_for_center(
                all_atoms=atoms,
                residue_atoms=residue_atoms,
                label=res_label,
                channel_scheme=cfg.channel_scheme,
                box_size=cfg.box_size,
                voxel_size=cfg.voxel_size,
                atom_density_threshold=cfg.atom_density_threshold,
            )
            if voxel is not None:
                site_voxels.append(voxel)

        if not site_voxels:
            stats["missing_sites"] += 1
            continue

        x = aggregate_structure_voxel(site_voxels)
        y_value = parse_activity_value(row)
        out_dir = cfg.output_dir / structure_id
        out_dir.mkdir(parents=True, exist_ok=True)
        example_id = f"{structure_id}_whole"
        out_path = out_dir / f"{example_id}.npz"
        np.savez_compressed(out_path, x=x.astype(np.float32), y=np.float32(y_value))
        manifest_rows.append(
            {
                "example_id": example_id,
                "structure_id": structure_id,
                "chain_id": "",
                "res_no": "",
                "mutation": row.get("mutation", ""),
                "label": str(y_value),
                "task": cfg.task,
                "sample_path": str(out_path),
                "channel_scheme": cfg.channel_scheme,
                "example_unit": cfg.example_unit,
            }
        )

    return manifest_rows, stats


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    structure_rows = load_rows(args.structure_manifest)
    activity_rows = load_rows(args.activity_manifest)
    structure_paths = resolve_structure_map(structure_rows, args.structure_manifest.parent.resolve())

    if args.example_unit == "mutation_site":
        out_rows, stats = build_mutation_site_examples(structure_paths, activity_rows, args)
    else:
        out_rows, stats = build_whole_structure_examples(structure_paths, activity_rows, args)

    write_manifest(out_rows, args.example_manifest_out)
    logging.info("Wrote %d examples to %s", len(out_rows), args.example_manifest_out)
    logging.info("Stats: %s", stats)


if __name__ == "__main__":
    main()

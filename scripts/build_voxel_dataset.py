#!/usr/bin/env python3
"""Build residue-centered voxel boxes from PDB structures.

This script modernizes and unifies the legacy `generate_full_sidechain_box_20A.py`
and `generate_backbone_box_20A.py` pipelines into one configurable CLI.
"""

from __future__ import annotations

import argparse
import csv
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

# Residue label mapping from legacy code.
RES_LABEL_DICT = {
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

LABEL_ATOM_TYPE_DICT = {
    18: {"N", "CA", "C", "O"},
    19: {"N", "CA", "C", "O", "CB", "SG"},
    2: {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    5: {"N", "CA", "C", "O", "CB", "OG"},
    6: {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    1: {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    13: {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    9: {"N", "CA", "C", "O", "CB"},
    11: {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    12: {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    10: {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
    3: {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    4: {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    0: {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    7: {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    17: {"N", "CA", "C", "O", "CB", "CG", "CD"},
    8: {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    14: {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    16: {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    15: {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
}


@dataclass(frozen=True)
class AtomRecord:
    atom: str
    res: str
    chain: str
    res_no: int
    xyz: np.ndarray
    element: str
    hetatm: bool
    altloc: str


CHANNEL_SCHEMES = {
    "element4": ["O", "C", "N", "S"],
    "backbone4": ["N", "CA", "C", "O"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build voxelized residue sites from split manifest")
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--example-manifest-out", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["residue_identity", "mutation_activity"])
    parser.add_argument("--box-size", type=float, default=20.0)
    parser.add_argument("--voxel-size", type=float, default=1.0)
    parser.add_argument("--channel-scheme", choices=sorted(CHANNEL_SCHEMES), default="element4")

    parser.add_argument("--max-sites-per-structure", type=int, default=None)
    parser.add_argument("--center-mode", choices=["all_residues", "mutated_sites", "active_site"], default="all_residues")
    parser.add_argument("--include-hetero", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["npz", "zarr", "h5"], default="npz")
    parser.add_argument("--atom-density-threshold", type=float, default=0.0)
    return parser.parse_args()


def parse_pdb_atoms(pdb_path: Path, include_hetero: bool) -> List[AtomRecord]:
    atoms: List[AtomRecord] = []
    with pdb_path.open() as handle:
        for line in handle:
            record = line[0:6].strip()
            if record not in {"ATOM", "HETATM"}:
                if line.startswith("ENDMDL"):
                    break
                continue
            if record == "HETATM" and not include_hetero:
                continue
            atom = line[12:16].strip()
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            res = line[17:20].strip()
            chain = line[21].strip() or "_"
            try:
                res_no = int(line[22:26].strip())
                xyz = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            element = line[76:78].strip() or atom[0]
            atoms.append(AtomRecord(atom=atom, res=res, chain=chain, res_no=res_no, xyz=xyz, element=element, hetatm=(record == "HETATM"), altloc=altloc))
    return atoms


def center_and_transform(label: int, positions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    reference = positions["CA"]
    axis_x = positions["N"] - positions["CA"]
    pseudo_axis_y = positions["C"] - positions["CA"]
    axis_z = np.cross(axis_x, pseudo_axis_y)
    if label != 18 and "CB" in positions:
        direction = positions["CB"] - positions["CA"]
        axis_z *= np.sign(direction.dot(axis_z))
    axis_y = np.cross(axis_z, axis_x)

    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = axis_z / np.linalg.norm(axis_z)

    transform = np.array([axis_x, axis_y, axis_z], dtype=np.float32).T
    return reference, transform


def map_channel(atom: AtomRecord, scheme: str) -> Optional[int]:
    channels = CHANNEL_SCHEMES[scheme]
    if scheme == "element4":
        token = atom.element.upper()[0]
    else:
        token = atom.atom
    try:
        return channels.index(token)
    except ValueError:
        return None


def build_voxel_for_center(
    all_atoms: Sequence[AtomRecord],
    residue_atoms: Sequence[AtomRecord],
    label: int,
    channel_scheme: str,
    box_size: float,
    voxel_size: float,
    atom_density_threshold: float,
) -> Optional[np.ndarray]:
    atom_positions = {a.atom: a.xyz for a in residue_atoms}
    if set(atom_positions.keys()) != LABEL_ATOM_TYPE_DICT[label]:
        return None

    reference, transform = center_and_transform(label, atom_positions)
    num_voxels = int(box_size / voxel_size)
    min_corner = -box_size / 2.0
    max_corner = box_size / 2.0

    res_key = {(a.chain, a.res_no) for a in residue_atoms}
    sample = np.zeros((len(CHANNEL_SCHEMES[channel_scheme]), num_voxels, num_voxels, num_voxels), dtype=np.float32)
    atom_count = 0

    for atom in all_atoms:
        if (atom.chain, atom.res_no) in res_key:
            continue
        transformed = (atom.xyz - reference).dot(transform)
        if np.any(transformed <= min_corner) or np.any(transformed >= max_corner):
            continue
        channel_idx = map_channel(atom, channel_scheme)
        if channel_idx is None:
            continue
        shifted = transformed - min_corner
        b = np.floor(shifted / voxel_size).astype(int)
        b = np.clip(b, 0, num_voxels - 1)
        sample[channel_idx, b[0], b[1], b[2]] += 1.0
        atom_count += 1

    threshold = (box_size**3) * atom_density_threshold
    if atom_count <= threshold:
        return None

    smooth = np.zeros_like(sample, dtype=np.float32)
    for i in range(sample.shape[0]):
        smooth[i] = gaussian_filter(sample[i], sigma=0.6, mode="reflect", truncate=4.0)
    smooth *= 1000.0
    return smooth


def residues_from_atoms(atoms: Sequence[AtomRecord]) -> Dict[Tuple[str, int, str], List[AtomRecord]]:
    residues: Dict[Tuple[str, int, str], List[AtomRecord]] = {}
    for atom in atoms:
        key = (atom.chain, atom.res_no, atom.res)
        residues.setdefault(key, []).append(atom)
    return residues


def structure_from_row(row: Dict[str, str], manifest_dir: Path) -> Tuple[str, Path]:
    structure_id = row.get("structure_id") or row.get("pdb_id") or row.get("id")
    pdb_path = row.get("pdb_path") or row.get("path")
    if not structure_id or not pdb_path:
        raise ValueError("split manifest must contain [structure_id|pdb_id|id] and [pdb_path|path] columns")
    path = Path(pdb_path)
    if not path.is_absolute():
        path = manifest_dir / path
    return structure_id, path


def worker_process(payload: Tuple[Dict[str, str], Dict[str, object]]) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    row, cfg = payload
    structure_id, pdb_path = structure_from_row(row, Path(cfg["manifest_dir"]))
    atoms = parse_pdb_atoms(pdb_path, include_hetero=bool(cfg["include_hetero"]))
    stats = {"skipped_missing_atoms": 0, "skipped_invalid_label": 0, "skipped_density": 0}
    if not atoms:
        return [], stats

    residue_map = residues_from_atoms(atoms)
    rows: List[Dict[str, str]] = []
    centers = list(residue_map.items())
    if cfg["center_mode"] != "all_residues":
        target_positions = set()
        if cfg["center_mode"] == "mutated_sites" and row.get("mutated_sites"):
            target_positions = {int(x) for x in row["mutated_sites"].replace(";", ",").split(",") if x.strip()}
        if cfg["center_mode"] == "active_site" and row.get("active_site"):
            target_positions = {int(x) for x in row["active_site"].replace(";", ",").split(",") if x.strip()}
        centers = [c for c in centers if c[0][1] in target_positions]

    max_sites = cfg["max_sites_per_structure"]
    if max_sites is not None:
        rng = np.random.default_rng(int(cfg["seed"]))
        if len(centers) > max_sites:
            idx = rng.choice(len(centers), size=max_sites, replace=False)
            centers = [centers[i] for i in sorted(idx)]

    out_dir = Path(cfg["output_dir"]) / structure_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for (chain, res_no, res_name), residue_atoms in centers:
        if res_name not in RES_LABEL_DICT:
            stats["skipped_invalid_label"] += 1
            continue
        label = RES_LABEL_DICT[res_name]
        atom_names = {a.atom for a in residue_atoms}
        if atom_names != LABEL_ATOM_TYPE_DICT[label]:
            stats["skipped_missing_atoms"] += 1
            continue
        voxel = build_voxel_for_center(
            all_atoms=atoms,
            residue_atoms=residue_atoms,
            label=label,
            channel_scheme=str(cfg["channel_scheme"]),
            box_size=float(cfg["box_size"]),
            voxel_size=float(cfg["voxel_size"]),
            atom_density_threshold=float(cfg["atom_density_threshold"]),
        )
        if voxel is None:
            stats["skipped_density"] += 1
            continue

        example_id = f"{structure_id}_{chain}_{res_no}_{res_name}"
        out_path = out_dir / f"{example_id}.npz"
        np.savez_compressed(out_path, x=voxel.astype(np.float32), y=np.int64(label))

        rows.append(
            {
                "example_id": example_id,
                "structure_id": structure_id,
                "chain_id": chain,
                "res_no": str(res_no),
                "res_name": res_name,
                "label": str(label),
                "sample_path": str(out_path),
                "task": str(cfg["task"]),
                "channel_scheme": str(cfg["channel_scheme"]),
            }
        )

    return rows, stats


def load_split_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows: Iterable[Dict[str, str]], out_path: Path) -> None:
    rows = list(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.format != "npz":
        raise NotImplementedError("Only --format npz is currently implemented")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_split_rows(args.split_manifest)
    cfg = {
        "manifest_dir": str(args.split_manifest.parent.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "task": args.task,
        "channel_scheme": args.channel_scheme,
        "box_size": args.box_size,
        "voxel_size": args.voxel_size,
        "center_mode": args.center_mode,
        "include_hetero": args.include_hetero,
        "max_sites_per_structure": args.max_sites_per_structure,
        "atom_density_threshold": args.atom_density_threshold,
        "seed": args.seed,
    }

    payloads = [(r, cfg) for r in rows]
    all_examples: List[Dict[str, str]] = []
    agg = {"skipped_missing_atoms": 0, "skipped_invalid_label": 0, "skipped_density": 0}

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for out_rows, stats in pool.map(worker_process, payloads):
                all_examples.extend(out_rows)
                for key, value in stats.items():
                    agg[key] += value
    else:
        for payload in payloads:
            out_rows, stats = worker_process(payload)
            all_examples.extend(out_rows)
            for key, value in stats.items():
                agg[key] += value

    write_manifest(all_examples, args.example_manifest_out)
    logging.info("Wrote %d examples to %s", len(all_examples), args.example_manifest_out)
    logging.info("Skipped (missing atoms=%d, invalid label=%d, density=%d)", agg["skipped_missing_atoms"], agg["skipped_invalid_label"], agg["skipped_density"])


if __name__ == "__main__":
    main()

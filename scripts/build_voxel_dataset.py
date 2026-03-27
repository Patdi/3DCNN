#!/usr/bin/env python3
"""Build residue-centered voxel boxes from PDB structures.

This script modernizes and unifies the legacy `generate_full_sidechain_box_20A.py`
and `generate_backbone_box_20A.py` pipelines into one configurable CLI.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit
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
    parser.add_argument("--task", default="residue_identity", choices=["residue_identity"])
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
    parser.add_argument("--resume", action="store_true", help="Skip structures already marked complete.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--done-dir", type=Path, default=None, help="Directory for per-structure completion markers (default: <output-dir>/_done).")
    return parser.parse_args()


def config_fingerprint(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def get_done_dir(output_dir: Path, done_dir: Path | None) -> Path:
    d = done_dir if done_dir is not None else output_dir / "_done"
    d.mkdir(parents=True, exist_ok=True)
    return d


def structure_done_path(done_dir: Path, structure_id: str) -> Path:
    return done_dir / f"{structure_id}.json"


def is_structure_done(done_dir: Path, structure_id: str) -> bool:
    return structure_done_path(done_dir, structure_id).exists()


def mark_structure_done(done_dir: Path, structure_id: str, payload: dict) -> None:
    out = structure_done_path(done_dir, structure_id)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, out)


def atomic_savez(final_path: Path, **arrays) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp_path, final_path)

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


def chain_to_code(chain: str) -> int:
    return ord(chain[0]) if chain else 0


@njit(cache=True)
def voxelize_numba(
    coords: np.ndarray,
    resnos: np.ndarray,
    chains: np.ndarray,
    channels: np.ndarray,
    center_resno: int,
    center_chain: int,
    reference: np.ndarray,
    transform: np.ndarray,
    min_corner: float,
    max_corner: float,
    voxel_size: float,
    num_voxels: int,
    n_channels: int,
) -> tuple[np.ndarray, int]:
    sample = np.zeros((n_channels, num_voxels, num_voxels, num_voxels), dtype=np.float32)
    atom_count = 0
    for i in range(coords.shape[0]):
        if chains[i] == center_chain and resnos[i] == center_resno:
            continue

        dx0 = coords[i, 0] - reference[0]
        dx1 = coords[i, 1] - reference[1]
        dx2 = coords[i, 2] - reference[2]

        tx = dx0 * transform[0, 0] + dx1 * transform[1, 0] + dx2 * transform[2, 0]
        ty = dx0 * transform[0, 1] + dx1 * transform[1, 1] + dx2 * transform[2, 1]
        tz = dx0 * transform[0, 2] + dx1 * transform[1, 2] + dx2 * transform[2, 2]

        if tx <= min_corner or tx >= max_corner:
            continue
        if ty <= min_corner or ty >= max_corner:
            continue
        if tz <= min_corner or tz >= max_corner:
            continue

        ch = channels[i]
        if ch < 0:
            continue

        bx = int((tx - min_corner) / voxel_size)
        by = int((ty - min_corner) / voxel_size)
        bz = int((tz - min_corner) / voxel_size)

        if bx < 0:
            bx = 0
        elif bx >= num_voxels:
            bx = num_voxels - 1

        if by < 0:
            by = 0
        elif by >= num_voxels:
            by = num_voxels - 1

        if bz < 0:
            bz = 0
        elif bz >= num_voxels:
            bz = num_voxels - 1

        sample[ch, bx, by, bz] += 1.0
        atom_count += 1

    return sample, atom_count


def prepare_voxel_inputs(all_atoms: Sequence[AtomRecord], channel_scheme: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_atoms = len(all_atoms)
    coords = np.empty((n_atoms, 3), dtype=np.float32)
    resnos = np.empty(n_atoms, dtype=np.int32)
    chains = np.empty(n_atoms, dtype=np.int32)
    channels = np.empty(n_atoms, dtype=np.int16)
    for i, atom in enumerate(all_atoms):
        coords[i] = atom.xyz
        resnos[i] = atom.res_no
        chains[i] = chain_to_code(atom.chain)
        channel_idx = map_channel(atom, channel_scheme)
        channels[i] = channel_idx if channel_idx is not None else -1
    return coords, resnos, chains, channels


def build_voxel_for_center(
    all_atoms: Sequence[AtomRecord],
    residue_atoms: Sequence[AtomRecord],
    label: int,
    channel_scheme: str,
    box_size: float,
    voxel_size: float,
    atom_density_threshold: float,
    precomputed_inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Optional[np.ndarray]:
    atom_positions = {a.atom: a.xyz for a in residue_atoms}
    if set(atom_positions.keys()) != LABEL_ATOM_TYPE_DICT[label]:
        return None

    reference, transform = center_and_transform(label, atom_positions)
    num_voxels = int(box_size / voxel_size)
    min_corner = -box_size / 2.0
    max_corner = box_size / 2.0

    if precomputed_inputs is None:
        coords, resnos, chains, channels = prepare_voxel_inputs(all_atoms, channel_scheme)
    else:
        coords, resnos, chains, channels = precomputed_inputs

    center_chain = chain_to_code(residue_atoms[0].chain)
    center_resno = residue_atoms[0].res_no
    n_channels = len(CHANNEL_SCHEMES[channel_scheme])
    sample, atom_count = voxelize_numba(
        coords=coords,
        resnos=resnos,
        chains=chains,
        channels=channels,
        center_resno=center_resno,
        center_chain=center_chain,
        reference=reference,
        transform=transform,
        min_corner=min_corner,
        max_corner=max_corner,
        voxel_size=voxel_size,
        num_voxels=num_voxels,
        n_channels=n_channels,
    )

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


def worker_process(payload: Tuple[Dict[str, str], Dict[str, object]]) -> Tuple[str, List[Dict[str, str]], Dict[str, int]]:
    row, cfg = payload
    structure_id, pdb_path = structure_from_row(row, Path(cfg["manifest_dir"]))
    atoms = parse_pdb_atoms(pdb_path, include_hetero=bool(cfg["include_hetero"]))
    stats = {"skipped_missing_atoms": 0, "skipped_invalid_label": 0, "skipped_density": 0}
    if not atoms:
        return structure_id, [], stats

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
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    precomputed_inputs = prepare_voxel_inputs(atoms, str(cfg["channel_scheme"]))

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
            precomputed_inputs=precomputed_inputs,
        )
        if voxel is None:
            stats["skipped_density"] += 1
            continue

        example_id = f"{structure_id}_{chain}_{res_no}_{res_name}"
        out_path = out_dir / f"{example_id}.npz"
        atomic_savez(out_path, x=voxel.astype(np.float32), y=np.int64(label))

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

    return structure_id, rows, stats


def load_split_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

MANIFEST_FIELDS = [
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

def _rows_for_structure_from_outputs(structure_id: str, cfg: dict[str, object]) -> list[dict[str, str]]:
    structure_dir = Path(str(cfg["output_dir"])) / structure_id
    if not structure_dir.exists():
        return []

    rows: list[dict[str, str]] = []
    for npz_path in sorted(structure_dir.glob("*.npz")):
        parts = npz_path.stem.split("_")
        if len(parts) < 4:
            logging.warning("Skipping malformed sample filename for %s: %s", structure_id, npz_path.name)
            continue
        chain_id = parts[-3]
        res_no = parts[-2]
        res_name = parts[-1]
        if res_name not in RES_LABEL_DICT:
            logging.warning("Skipping sample with unknown residue label for %s: %s", structure_id, npz_path.name)
            continue
        rows.append(
            {
                "example_id": npz_path.stem,
                "structure_id": structure_id,
                "chain_id": chain_id,
                "res_no": res_no,
                "res_name": res_name,
                "label": str(RES_LABEL_DICT[res_name]),
                "sample_path": str(npz_path),
                "task": str(cfg["task"]),
                "channel_scheme": str(cfg["channel_scheme"]),
            }
        )
    return rows


def collect_manifest_rows(
    structure_ids: Sequence[str],
    done_dir: Path,
    cfg: dict[str, object],
    include_incomplete: bool = False,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for structure_id in structure_ids:
        if not include_incomplete and not is_structure_done(done_dir, structure_id):
            continue
        rows.extend(_rows_for_structure_from_outputs(structure_id, cfg))
    return rows


def validate_manifest_rows(rows: Sequence[Dict[str, str]], require_existing_paths: bool = True) -> None:
    expected_fields = set(MANIFEST_FIELDS)
    for idx, row in enumerate(rows):
        row_fields = set(row.keys())
        if row_fields != expected_fields:
            raise ValueError(
                f"Manifest row {idx} has unexpected fields: got {sorted(row_fields)}, expected {sorted(expected_fields)}"
            )
        for field in MANIFEST_FIELDS:
            value = str(row[field])
            if "\n" in value or "\r" in value:
                raise ValueError(f"Manifest row {idx} field '{field}' contains an embedded newline.")
        sample_path = str(row["sample_path"])
        if not sample_path.endswith(".npz"):
            raise ValueError(f"Manifest row {idx} has non-npz sample_path: {sample_path}")
        if require_existing_paths and not Path(sample_path).exists():
            raise FileNotFoundError(f"Manifest row {idx} sample_path does not exist: {sample_path}")


def atomic_write_manifest(rows: Iterable[Dict[str, str]], out_path: Path) -> None:
    rows = list(rows)
    validate_manifest_rows(rows, require_existing_paths=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, out_path)

def write_manifest(rows: Iterable[Dict[str, str]], out_path: Path) -> None:
    atomic_write_manifest(rows, out_path)


def main() -> None:
    args = parse_args()
    if args.format != "npz":
        raise NotImplementedError("Only --format npz is currently implemented")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = load_split_rows(args.split_manifest)
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

    done_dir = get_done_dir(args.output_dir, args.done_dir)

    pending_rows: list[Dict[str, str]] = []
    seen_structure_ids: set[str] = set()
    for row in split_rows:
        structure_id = row.get("structure_id") or row.get("pdb_id") or row.get("id")
        if not structure_id:
            raise ValueError("Missing structure_id/pdb_id/id in split manifest.")
        if structure_id in seen_structure_ids:
            logging.warning("Duplicate structure id in split manifest, skipping duplicate row: %s", structure_id)
            continue
        seen_structure_ids.add(structure_id)
        if args.resume and not args.overwrite and is_structure_done(done_dir, structure_id):
            logging.info("Skipping complete structure %s", structure_id)
            continue
        pending_rows.append(row)

    payloads = [(r, cfg) for r in pending_rows]
    agg = {"skipped_missing_atoms": 0, "skipped_invalid_label": 0, "skipped_density": 0}

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            for structure_id, out_rows, stats in pool.map(worker_process, payloads):
                mark_structure_done(
                    done_dir,
                    structure_id,
                    {
                        "structure_id": structure_id,
                        "num_examples": len(out_rows),
                        "stats": stats,
                        "channel_scheme": args.channel_scheme,
                        "task": args.task,
                        "box_size": args.box_size,
                        "voxel_size": args.voxel_size,
                        "config_fingerprint": config_fingerprint(cfg),
                    },
                )
                for key, value in stats.items():
                    agg[key] += value
    else:
        for payload in payloads:
            structure_id, out_rows, stats = worker_process(payload)
            mark_structure_done(
                done_dir,
                structure_id,
                {
                    "structure_id": structure_id,
                    "num_examples": len(out_rows),
                    "stats": stats,
                    "channel_scheme": args.channel_scheme,
                    "task": args.task,
                    "box_size": args.box_size,
                    "voxel_size": args.voxel_size,
                    "config_fingerprint": config_fingerprint(cfg),
                },
            )
            for key, value in stats.items():
                agg[key] += value

    structure_ids_in_split = [sid for sid in seen_structure_ids]
    final_rows = collect_manifest_rows(structure_ids_in_split, done_dir=done_dir, cfg=cfg, include_incomplete=False)
    final_rows.sort(key=lambda row: row["example_id"])
    atomic_write_manifest(final_rows, args.example_manifest_out)

    logging.info("Wrote %d examples to %s", len(final_rows), args.example_manifest_out)
    logging.info("Skipped (missing atoms=%d, invalid label=%d, density=%d)", agg["skipped_missing_atoms"], agg["skipped_invalid_label"], agg["skipped_density"])


if __name__ == "__main__":
    main()

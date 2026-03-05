import csv
import subprocess
import sys
from pathlib import Path

import numpy as np


def _pdb_atom_line(serial: int, atom: str, res: str, chain: str, res_no: int, x: float, y: float, z: float, element: str) -> str:
    return (
        f"ATOM  {serial:5d} {atom:>4s} {res:>3s} {chain:1s}{res_no:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {element:>2s}\n"
    )


def _write_test_pdb(path: Path) -> None:
    lines = [
        _pdb_atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
        _pdb_atom_line(2, "CA", "ALA", "A", 1, 1.0, 0.0, 0.0, "C"),
        _pdb_atom_line(3, "C", "ALA", "A", 1, 1.5, 1.0, 0.0, "C"),
        _pdb_atom_line(4, "O", "ALA", "A", 1, 1.3, 2.1, 0.0, "O"),
        _pdb_atom_line(5, "CB", "ALA", "A", 1, 1.2, -0.7, 1.1, "C"),
        _pdb_atom_line(6, "N", "GLY", "A", 2, 2.3, 0.5, 0.0, "N"),
        _pdb_atom_line(7, "CA", "GLY", "A", 2, 3.0, 1.6, 0.0, "C"),
        _pdb_atom_line(8, "C", "GLY", "A", 2, 4.1, 1.1, 0.0, "C"),
        _pdb_atom_line(9, "O", "GLY", "A", 2, 4.8, 2.0, 0.0, "O"),
        "END\n",
    ]
    path.write_text("".join(lines))


def test_build_activity_dataset_mutation_site(tmp_path: Path):
    pdb_path = tmp_path / "toy.pdb"
    _write_test_pdb(pdb_path)

    structures = tmp_path / "structures.csv"
    structures.write_text(f"structure_id,pdb_path\nS1,{pdb_path}\n")

    activity = tmp_path / "activity.csv"
    activity.write_text("structure_id,chain_id,res_no,mutation,activity_value\nS1,A,1,A1V,1.25\n")

    out_dir = tmp_path / "out"
    manifest = tmp_path / "activity_manifest.csv"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_activity_dataset.py",
            "--structure-manifest",
            str(structures),
            "--activity-manifest",
            str(activity),
            "--output-dir",
            str(out_dir),
            "--example-manifest-out",
            str(manifest),
            "--example-unit",
            "mutation_site",
        ],
        check=True,
    )

    rows = list(csv.DictReader(manifest.open()))
    assert len(rows) == 1
    assert rows[0]["task"] == "regression"
    npz = np.load(rows[0]["sample_path"])
    assert npz["x"].ndim == 4
    assert np.isclose(float(npz["y"]), 1.25)

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_pdb_manifest import main


def _write_fake_pdb(path: Path, residues: str, chain: str = "A") -> None:
    aa3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
        "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
        "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    lines = []
    atom_id = 1
    for idx, aa in enumerate(residues, start=1):
        lines.append(
            f"ATOM  {atom_id:5d}  CA  {aa3.get(aa, 'GLY')} {chain}{idx:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
        )
        atom_id += 1
    lines.append("END\n")
    path.write_text("".join(lines))


def _read_csv_rows(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def test_manifest_writes_expected_columns(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _write_fake_pdb(pdb_dir / "s1.pdb", "ACDE")

    out = tmp_path / "splits" / "pretrain_manifest.csv"
    rc = main(["--pdb-dir", str(pdb_dir), "--out", str(out)])

    assert rc == 0
    rows = _read_csv_rows(out)
    assert len(rows) == 1
    assert rows[0]["pdb_id"] == "s1"
    assert rows[0]["chains"] == "A"
    assert rows[0]["num_residues"] == "4"


def test_manifest_min_residues_and_dedupe(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _write_fake_pdb(pdb_dir / "small.pdb", "AC")
    _write_fake_pdb(pdb_dir / "dup1.pdb", "AAAA")
    _write_fake_pdb(pdb_dir / "dup2.pdb", "AAAA")

    out = tmp_path / "manifest.csv"
    main([
        "--pdb-dir", str(pdb_dir),
        "--out", str(out),
        "--min-residues", "4",
        "--dedupe-by-sequence",
    ])

    rows = _read_csv_rows(out)
    assert len(rows) == 1
    assert rows[0]["sequence"] == "AAAA"


def test_manifest_recursive_glob(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    nested = pdb_dir / "nested"
    nested.mkdir(parents=True)
    _write_fake_pdb(nested / "entry.ent", "MKT")

    out = tmp_path / "manifest.csv"
    main([
        "--pdb-dir", str(pdb_dir),
        "--out", str(out),
        "--glob", "*.ent",
        "--recursive",
    ])

    rows = _read_csv_rows(out)
    assert len(rows) == 1
    assert rows[0]["pdb_id"] == "entry"

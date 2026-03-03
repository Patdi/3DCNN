import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_splits_from_pdb_folder import main


def _write_fake_pdb(path: Path, seq: str) -> None:
    aa3 = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU",
        "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE",
        "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    }
    lines = []
    atom_id = 1
    x = 0.0
    for idx, aa in enumerate(seq, start=1):
        res = aa3.get(aa, "GLY")
        lines.append(
            f"ATOM  {atom_id:5d}  CA  {res} A{idx:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
        )
        atom_id += 1
        x += 1.2
    lines.append("END\n")
    path.write_text("".join(lines))


def test_random_split_writes_expected_files(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    for i in range(10):
        _write_fake_pdb(pdb_dir / f"s{i}.pdb", "ACDEFG")

    out = tmp_path / "splits"
    rc = main([
        "--pdb-dir", str(pdb_dir),
        "--output-dir", str(out),
        "--method", "random",
        "--train-fraction", "0.6",
        "--val-fraction", "0.2",
        "--test-fraction", "0.2",
        "--seed", "1",
        "--write-legacy-pdb-lists",
    ])
    assert rc == 0
    assert (out / "train.txt").exists()
    assert (out / "val.txt").exists()
    assert (out / "test.txt").exists()
    assert (out / "train.csv").exists()
    assert (out / "PDB_train.txt").exists()


def test_sequence_cluster_keeps_similar_sequences_together(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _write_fake_pdb(pdb_dir / "a1.pdb", "AAAAAAAAAA")
    _write_fake_pdb(pdb_dir / "a2.pdb", "AAAAAAAAAA")
    _write_fake_pdb(pdb_dir / "b1.pdb", "CCCCCCCCCC")
    _write_fake_pdb(pdb_dir / "b2.pdb", "CCCCCCCCCC")

    out = tmp_path / "splits"
    main([
        "--pdb-dir", str(pdb_dir),
        "--output-dir", str(out),
        "--method", "sequence-cluster",
        "--seq-identity-threshold", "0.9",
        "--seed", "0",
    ])

    train = set((out / "train.txt").read_text().split())
    val = set((out / "val.txt").read_text().split())
    test = set((out / "test.txt").read_text().split())

    splits = [train, val, test]
    for pair in [{"a1", "a2"}, {"b1", "b2"}]:
        assert any(pair.issubset(s) for s in splits)


def test_scaffold_materialize_symlink(tmp_path: Path):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    _write_fake_pdb(pdb_dir / "x1.pdb", "AVILMFWY")
    _write_fake_pdb(pdb_dir / "x2.pdb", "STNQCGP")
    _write_fake_pdb(pdb_dir / "x3.pdb", "KRHDE")

    out = tmp_path / "splits"
    main([
        "--pdb-dir", str(pdb_dir),
        "--output-dir", str(out),
        "--method", "scaffold",
        "--materialize", "symlink",
    ])

    materialized = list((out / "train").glob("*.pdb")) + list((out / "val").glob("*.pdb")) + list((out / "test").glob("*.pdb"))
    assert materialized
    assert all(p.is_symlink() for p in materialized)

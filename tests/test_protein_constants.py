from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atom_res_dict import res_atom_type_dict
from protein_constants import (
    AA_BOND_DICT,
    BIAS,
    LABEL_RES_DICT,
    RES_ATOM_TYPE_DICT,
    RES_LABEL_DICT,
)


def test_label_maps_are_bidirectional():
    for label, residue in LABEL_RES_DICT.items():
        assert RES_LABEL_DICT[residue] == label


def test_backbone_atoms_present_for_all_residues():
    backbone = {"N", "CA", "C", "O"}
    for atoms in RES_ATOM_TYPE_DICT.values():
        assert backbone.issubset(set(atoms))


def test_bond_dict_atoms_match_declared_atom_templates():
    for residue, bonds in AA_BOND_DICT.items():
        allowed_atoms = set(RES_ATOM_TYPE_DICT[residue])
        for left, right in bonds:
            assert left in allowed_atoms
            assert right in allowed_atoms


def test_legacy_wrapper_exports_same_mapping_object():
    assert res_atom_type_dict is RES_ATOM_TYPE_DICT


def test_bias_vector_shape():
    assert len(BIAS) == 3

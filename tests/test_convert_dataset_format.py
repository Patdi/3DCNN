from pathlib import Path
import json

import numpy as np

from convert_dataset_format import (
    build_metadata,
    load_split_part,
    parse_parts,
    write_npz_dataset,
)


def _dump(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array.dump(path)


def test_parse_parts_ranges_and_values() -> None:
    assert parse_parts("1-3,5,7-8") == [1, 2, 3, 5, 7, 8]


def test_load_split_part_train_concatenates_shards(tmp_path: Path) -> None:
    sampled_root = tmp_path / "Sampled_Numpy"
    _dump(sampled_root / "train" / "X_smooth1_1.dat", np.ones((2, 4, 2, 2, 2), dtype=np.float32))
    _dump(sampled_root / "train" / "y1_1.dat", np.array([1, 2], dtype=np.int32))
    _dump(sampled_root / "train" / "X_smooth2_1.dat", np.zeros((1, 4, 2, 2, 2), dtype=np.float32))
    _dump(sampled_root / "train" / "y2_1.dat", np.array([3], dtype=np.int32))

    data, labels = load_split_part(sampled_root, "train", part=1, train_shards=2)

    assert data.shape == (3, 4, 2, 2, 2)
    assert labels.tolist() == [1, 2, 3]


def test_load_split_part_eval_uses_expected_prefixes(tmp_path: Path) -> None:
    sampled_root = tmp_path / "Sampled_Numpy"
    _dump(sampled_root / "val" / "Xv_smooth_2.dat", np.ones((2, 1), dtype=np.float32))
    _dump(sampled_root / "val" / "yv_2.dat", np.array([[4], [5]], dtype=np.int32))
    _dump(sampled_root / "test" / "Xt_smooth_2.dat", np.ones((1, 1), dtype=np.float32))
    _dump(sampled_root / "test" / "yt_2.dat", np.array([6], dtype=np.int32))

    _, val_labels = load_split_part(sampled_root, "val", part=2)
    _, test_labels = load_split_part(sampled_root, "test", part=2)

    assert val_labels.tolist() == [4, 5]
    assert test_labels.tolist() == [6]


def test_write_npz_dataset_writes_metadata_sidecar(tmp_path: Path) -> None:
    output_file = tmp_path / "train_data_1.npz"
    data = np.ones((2, 2), dtype=np.float32)
    labels = np.array([0, 1], dtype=np.int32)
    metadata = build_metadata(
        split="train",
        part=1,
        data=data,
        labels=labels,
        channels=4,
        box_size=20,
        voxel_size=1.0,
        class_map={"ALA": 0},
        normalization="mean_center",
        source_split_hash="abc123",
    )

    write_npz_dataset(output_file, data, labels, metadata, compressed=True)

    with np.load(output_file) as loaded:
        assert loaded["data"].shape == (2, 2)
        assert loaded["labels"].tolist() == [0, 1]

    metadata_json = json.loads(output_file.with_suffix(".json").read_text())
    assert metadata_json["channels"] == 4
    assert metadata_json["class_map"] == {"ALA": 0}

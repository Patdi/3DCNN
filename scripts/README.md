# Scripts

## `build_voxel_dataset.py`

Unified voxel dataset builder for legacy `generate_full_sidechain_box_20A.py` and `generate_backbone_box_20A.py` workflows.

Example:

```bash
python scripts/build_voxel_dataset.py \
  --split-manifest data/splits/train.csv \
  --output-dir data/processed/voxel_boxes/train \
  --example-manifest-out data/splits/train_sites.csv \
  --task residue_identity \
  --box-size 20 \
  --voxel-size 1.0 \
  --channel-scheme element4
```

Minimal required args:

- `--split-manifest`
- `--output-dir`
- `--example-manifest-out`
- `--task`

Useful extras:

- `--max-sites-per-structure`
- `--center-mode`
- `--include-hetero`
- `--num-workers`
- `--seed`
- `--format`


## `compute_normalization.py`

Compute train-set normalization stats from a site-level manifest and write
`normalization_stats.npz` for training/evaluation scripts.

Example:

```bash
python scripts/compute_normalization.py \
  --manifest data/splits/train_sites.csv \
  --out data/processed/stats/normalization_stats.npz
```

Minimal required args:

- `--manifest`
- `--out`

Useful extras:

- `--mode per-channel|global`
- `--max-samples`
- `--seed`

## `evaluate_model.py`

Evaluate a PyTorch checkpoint on a split manifest and export predictions/metrics.
## `train_voxel_cnn.py`

Modern PyTorch replacement for legacy `3DCNN.py` + `layers.py`.

Example:

```bash
python scripts/evaluate_model.py \
  --checkpoint outputs/runs/voxel_cnn_pretrain/checkpoints/best_val.pt \
  --manifest data/splits/test_sites.csv \
  --normalization data/processed/stats/normalization_stats.npz \
  --output-dir outputs/runs/voxel_cnn_pretrain/eval \
  --task residue_identity
```

Minimal required args:

- `--checkpoint`
- `--manifest`
- `--normalization`
- `--output-dir`
- `--task`

Useful extras:

- `--num-classes`
- `--metrics`
- `--batch-size`
- `--device`
python scripts/train_voxel_cnn.py \
  --train-manifest data/splits/train_sites.csv \
  --val-manifest data/splits/val_sites.csv \
  --normalization data/processed/stats/normalization_stats.npz \
  --output-dir outputs/runs/voxel_cnn_pretrain \
  --task residue_identity \
  --num-classes 20 \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3
```


## `make_pdb_manifest.py`

Build a structure-level manifest from a folder of PDB files (one row per parseable structure).

Example:

```bash
python scripts/make_pdb_manifest.py \
  --pdb-dir data/raw/pdbs \
  --out data/splits/pretrain_manifest.csv
```

Minimal required args:

- `--pdb-dir`
- `--out`

Useful extras:

- `--glob "*.pdb"`
- `--recursive`
- `--min-residues 50`
- `--dedupe-by-sequence`

## `make_splits_from_pdb_folder.py`

Create `train/val/test` split files directly from a folder of PDB files.

Example:

```bash
python scripts/make_splits_from_pdb_folder.py \
  --pdb-dir data/pdbs \
  --output-dir data/splits \
  --method sequence-cluster \
  --seq-identity-threshold 0.4 \
  --write-legacy-pdb-lists
```

Outputs:

- `train.txt`, `val.txt`, `test.txt`
- `train.csv`, `val.csv`, `test.csv` (with `structure_id,pdb_path,sequence,split`)
- optional legacy lists `PDB_train.txt`, `PDB_val.txt`, `PDB_test.txt`
- optional materialized split folders via `--materialize symlink|copy`


# Scripts

This folder contains data preparation, training, evaluation, and dataset QA utilities.


## 1. `make_pdb_manifest.py`

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

## 2. `make_splits_from_pdb_folder.py`

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
- Optional legacy lists `PDB_train.txt`, `PDB_val.txt`, `PDB_test.txt`
- Optional materialized split folders via `--materialize symlink|copy`

## 3. `build_voxel_dataset.py`

Unified voxel dataset builder for legacy `generate_full_sidechain_box_20A.py` and `generate_backbone_box_20A.py` workflows. This builder is pretraining-focused and currently supports only residue identity labels.

Train Example:

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

Validate Example:
```bash
python scripts/build_voxel_dataset.py \
  --split-manifest data/splits/val.csv \
  --output-dir data/processed/voxel_boxes/val \
  --example-manifest-out data/splits/val_sites.csv \
  --task residue_identity \
  --box-size 20 \
  --voxel-size 1.0 \
  --channel-scheme element4
```

Test Example:
```bash
python scripts/build_voxel_dataset.py \
  --split-manifest data/splits/test.csv \
  --output-dir data/processed/voxel_boxes/test \
  --example-manifest-out data/splits/test_sites.csv \
  --task residue_identity \
  --box-size 20 \
  --voxel-size 1.0 \
  --channel-scheme element4
```
Minimal required args:

- `--split-manifest`
- `--output-dir`
- `--example-manifest-out`
- `--task` (`residue_identity` only)

Useful extras:

- `--max-sites-per-structure`
- `--center-mode`
- `--include-hetero`
- `--num-workers`
- `--seed`
- `--format`

## 4. `compute_normalization.py`

Compute train-set normalization stats from a site-level manifest and write `normalization_stats.npz` for training/evaluation scripts.

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

## 5. `train_voxel_cnn.py`

Modern PyTorch replacement for legacy `3DCNN.py` + `layers.py`.

Example:

```bash
python scripts/train_voxel_cnn.py \
  --train-manifest data/splits/train_sites.csv \
  --val-manifest data/splits/val_sites.csv \
  --normalization data/processed/stats/normalization_stats.npz \
  --output-dir outputs/runs/voxel_cnn_pretrain \
  --task residue_identity \
  --num-classes 20 \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --num-workers 8 \
  --persistent-workers \
  --prefetch-factor 2
```

## 6. `evaluate_model.py`

Evaluate a PyTorch checkpoint on a split manifest and export predictions/metrics.

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

## 7. `repair_voxel_manifest.py`

Repair or rebuild voxel manifest CSV files from on-disk `.npz` samples. This is useful when a manifest has corrupted/concatenated rows, stale paths, or missing metadata.

Common usage (rebuild from disk only):

```bash
python scripts/repair_voxel_manifest.py \
  --voxel-root data/processed/voxel_boxes/train \
  --output-manifest data/splits/train_sites_repaired.csv \
  --task residue_identity \
  --channel-scheme element4 \
  --rebuild-only \
  --dedupe \
  --validate-files
```

Common usage (salvage + merge with disk truth):

```bash
python scripts/repair_voxel_manifest.py \
  --input-manifest data/splits/train_sites.csv \
  --voxel-root data/processed/voxel_boxes/train \
  --output-manifest data/splits/train_sites_repaired.csv \
  --task residue_identity \
  --channel-scheme element4 \
  --salvage-existing \
  --dedupe \
  --validate-files \
  --infer-labels \
  --output-json outputs/reports/train_manifest_repair.json
```

Minimal required args:

- `--voxel-root`
- `--output-manifest`
- `--task` (`residue_identity|mutation_activity|regression`)

Useful extras:

- `--input-manifest` (required when using `--salvage-existing`)
- `--rebuild-only` or `--salvage-existing`
- `--dedupe`
- `--validate-files`
- `--infer-labels`
- `--strict`
- `--output-json`

## 8. `validate_voxel_dataset.py`

Validate a site-level voxel manifest and corresponding `.npz` files to catch malformed CSV rows, missing/unreadable files, label mismatches, duplicate IDs/paths, and shape inconsistencies.

Example:

```bash
python scripts/validate_voxel_dataset.py \
  --manifest data/splits/train_sites.csv \
  --root-dir data/processed/voxel_boxes/train \
  --task residue_identity \
  --check-files \
  --check-labels \
  --check-shape \
  --check-duplicates \
  --expected-channels 4 \
  --expected-box-size 20 \
  --output-json outputs/reports/train_dataset_validation.json
```

Minimal required args:

- `--manifest`

Useful extras:

- `--root-dir` (defaults to the manifest parent directory)
- `--task` (`residue_identity|mutation_activity|regression`)
- `--check-files`
- `--check-labels`
- `--check-shape`
- `--check-duplicates`
- `--sample-limit`
- `--fail-fast`
- `--output-json`

## Extras:

## `check_split_leakage.py`

Check for duplicate records and leakage across split manifests (`train/val/test`) using one or more key columns (default: `structure_id`).

Example:

```bash
python scripts/check_split_leakage.py \
  --train data/splits/train.csv \
  --val data/splits/val.csv \
  --test data/splits/test.csv \
  --check-cols structure_id
```

Useful extras:

- `--allow-missing-cols`
- `--json-out outputs/reports/split_leakage.json`


## 1. `build_activity_dataset.py`

Activity-focused dataset builder separated from pretraining.

Example (mutation/site activity):

```bash
python scripts/build_activity_dataset.py \
  --structure-manifest data/splits/train.csv \
  --activity-manifest data/labels/mutation_activity.csv \
  --output-dir data/processed/activity/train \
  --example-manifest-out data/splits/train_activity_sites.csv \
  --example-unit mutation_site \
  --task regression
```

Example (whole-protein activity):

```bash
python scripts/build_activity_dataset.py \
  --structure-manifest data/splits/train.csv \
  --activity-manifest data/labels/protein_activity.csv \
  --output-dir data/processed/activity/train \
  --example-manifest-out data/splits/train_activity_structures.csv \
  --example-unit whole_structure \
  --task regression
```

## 2. `predict_residue_identity.py`

Run inference for residue identity from a trained voxel CNN checkpoint and write predictions to CSV.

Example:

```bash
python scripts/predict_residue_identity.py \
  --test-manifest data/splits/test_sites.csv \
  --normalization data/processed/stats/normalization_stats.npz \
  --checkpoint outputs/runs/voxel_cnn_pretrain/best_val.pt \
  --output-csv outputs/runs/voxel_cnn_pretrain/predictions_test.csv \
  --batch-size 64 \
  --top-k 3 \
  --output-probs
```

Minimal required args:

- `--test-manifest`
- `--normalization`
- `--checkpoint`
- `--output-csv`

Useful extras:

- `--batch-size`
- `--device`
- `--num-workers`
- `--amp`
- `--top-k`
- `--output-probs`
- `--verbose`

## 3. add_solvent_accessibility_to_predictions.py

Motivation are higher solvent accessible residues are harder to predict than less solvently accessible residues since there are less microenviroment constraints for residues exposed to the solvent? Given the residue-identity prediction CSV (for example `predictions_test.csv`). Compute solvent accessibility only for the residues present in the test set / predictions. Outputs an appended solvent accessibility columns to the predictions CSV and a plot with solvent accessibility bins. 

Example:

    python scripts/add_solvent_accessibility_to_predictions.py \
      --predictions-csv ../data/outputs/runs/voxel_cnn_pretrain/test_predictions.csv \
      --test-manifest ../data/splits/test_sites.csv \
      --pdb-root ../data/pdbs \
      --output-csv ../data/outputs/runs/voxel_cnn_pretrain/test_predictions_with_sasa.csv \
      --plot-png ../data/outputs/runs/voxel_cnn_pretrain/test_predictions_sasa_accuracy.png \
      --sasa-kind relative \
      --sasa-field total \
      --binning equal_width \
      --num-bins 10

## 4. `constants.py`

Shared constants imported by evaluation/analysis code (for example, residue label-to-group mappings used by `evaluate_model.py`).

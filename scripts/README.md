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

## `train_voxel_cnn.py`

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
  --lr 1e-3
```

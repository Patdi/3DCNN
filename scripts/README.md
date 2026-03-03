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

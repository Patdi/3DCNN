# 3DCNN

Modernized 3D convolutional neural network tooling for residue-level protein micro-environment modeling.

## Overview

This repository is a modern rewrite of **Torng & Altman (2017)** for voxelized local protein environments and amino-acid environment similarity style learning. The codebase keeps the original 3D CNN spirit while replacing legacy training/data plumbing with maintainable Python + PyTorch workflows. Based on work by **Kulikova et al. (2021)** box sizes of 12 Å are equivalent to the orginal 20 Å reported by **Torng & Altman (2017)** and should greatly reduce the voxel database size. 

In addition to the original atom-channel setup, the project will eventually support additional biochemical channels commonly used in structure-based protein engineering workflows (including **hydrogen**, **partial charges**, and **solvent accessibility**) as described by **Shroff et al. (2019)** and used in later work such as **Shroff et al. (2020)** and **Kulikova et al. (2021)**.

> Original Torng et al. code repository: **(1) TBD**

## What is in this repository

### Core model

- `models/cnn3d.py`
  - `CNN3DConfig`: typed config for model construction.
  - `VoxelCNN3D`: baseline 3-block 3D CNN + dense head (classification or regression).
  - `build_loss(...)`: task-aware loss helper for classification/regression.

### Data and training pipeline (modern scripts)

- `scripts/make_pdb_manifest.py`  
  Build structure-level manifests from raw PDB folders.
- `scripts/make_splits_from_pdb_folder.py`  
  Build train/val/test splits with random, sequence-cluster, or scaffold grouping.
- `scripts/build_voxel_dataset.py`  
  Build residue-centered voxel boxes from structures.
- `scripts/compute_normalization.py`  
  Compute train-set normalization statistics.
- `scripts/train_voxel_cnn.py`  
  Train the PyTorch 3D CNN on voxel manifests.
- `scripts/evaluate_model.py`  
  Evaluate checkpoints and compute classification metrics.
- `scripts/repair_voxel_manifest.py`, `scripts/validate_voxel_dataset.py`, and other helpers  
  Dataset QA and repair utilities.

### Legacy/archived code

- `archived/` contains prior code snapshots and legacy scripts from earlier workflows.
- Root-level compatibility files (e.g., `prepare_dataset.py`, `data_util.py`, `data_utils.py`) reflect older data formats and transitional utilities.

### Tests

- `tests/` includes unit tests for model behavior and dataset/splitting/evaluation utilities.

## End-to-end workflow

A typical modern workflow is:

1. Build a PDB manifest from structures.
2. Create split manifests (`train.csv`, `val.csv`, `test.csv`).
3. Build voxelized residue-site examples for each split.
4. Compute normalization stats from training data.
5. Train the 3D CNN.
6. Evaluate saved checkpoints.

See script-level usage examples in `scripts/README.md`.

## Scientific context and references

Other 3DCNN based models:

1. Torng, Wen, and Russ B. Altman. *3D deep convolutional neural networks for amino acid environment similarity analysis.* BMC Bioinformatics 18.1 (2017): 302.
2. Shroff, Raghav, et al. *A structure-based deep learning framework for protein engineering.* bioRxiv (2019): 833905.
3. Shroff, Raghav, et al. *Discovery of novel gain-of-function mutations guided by structure-based deep learning.* ACS Synthetic Biology 9.11 (2020): 2927–2935.
4. Kulikova, Anastasiya V., et al. *Learning the local landscape of protein structures with convolutional neural networks.* Journal of Biological Physics 47.4 (2021): 435–454.
5. Sieg, Jochen, and Matthias Rarey. *Searching similar local 3D micro-environments in protein structure databases with MicroMiner.* Briefings in Bioinformatics 24.6 (2023): bbad357.

## Roadmap

Planned/desired next steps include:

- Adding a **regressor head** that consumes mutational data for activity/fitness-style prediction.
- Exploring a shift from pure 3D CNNs toward **transformer-based** modeling.
- Evaluating strategy ideas inspired by **Sieg et al. (2023)** for local 3D environment search/representation.

## Current status

The repository already contains:

- A modern PyTorch baseline model.
- A modular data preparation and voxelization pipeline.
- Train/eval CLIs and supporting quality-control scripts.

The repository is therefore ready for iterative feature expansion around richer channels, mutation-centric supervision, and architecture experimentation.

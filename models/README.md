# `models/cnn3d.py` README

This module provides a compact PyTorch 3D convolutional network for voxelized protein-environment tasks, plus task-aware loss selection helpers.

## Contents

- `CNN3DConfig`: typed configuration for model construction.
- `VoxelCNN3D`: baseline 3D CNN model with three convolution blocks and a fully-connected head.
- `build_loss(...)`: utility to construct a loss function based on the selected task.

## `CNN3DConfig`

`CNN3DConfig` is a frozen dataclass (immutable after creation) with the following fields:

- `in_channels` (**required**): number of channels in the input voxel tensor.
- `num_classes` (default: `20`): output classes for classification tasks. Use `None` for regression.
- `dropout` (default: `0.5`): dropout probability used in 3D and dense layers.
- `input_size` (default: `20`): expected cubic input edge length (`D=H=W=input_size`).
- `conv_channels` (default: `(100, 200, 400)`): output channels for the three 3D convolutions.
- `fc_hidden` (default: `1000`): hidden units in the dense layer before output.
- `task` (default: `"residue_identity"`): one of:
  - `"residue_identity"` (classification)
  - `"mutation_activity"` (classification)
  - `"regression"` (single-value output)

## `VoxelCNN3D` architecture

The model applies:

1. `Conv3d(in_channels -> conv_channels[0], kernel_size=3)` + `ReLU` + `Dropout3d`
2. `Conv3d(conv_channels[0] -> conv_channels[1], kernel_size=3)` + `ReLU` + `MaxPool3d(2)` + `Dropout3d`
3. `Conv3d(conv_channels[1] -> conv_channels[2], kernel_size=3)` + `ReLU` + `MaxPool3d(2)` + `Dropout3d`
4. Flatten
5. `Linear(flattened -> fc_hidden)` + `ReLU` + `Dropout`
6. `Linear(fc_hidden -> output_dim)`

`output_dim` is:

- `num_classes` for classification tasks
- `1` for regression

### Input and output shapes

Expected input shape:

- `x: [batch_size, in_channels, input_size, input_size, input_size]`

Output shape:

- Classification: `[batch_size, num_classes]`
- Regression: `[batch_size, 1]`

### Validation rules

If `task` is `"residue_identity"` or `"mutation_activity"`, `num_classes` must not be `None`.

## Loss helper

Use `build_loss(task, class_weights=None)` to obtain:

- `CrossEntropyLoss` for `residue_identity` and `mutation_activity`
- `MSELoss` for `regression`

A `ValueError` is raised for unsupported tasks.

## Usage examples

### Classification

```python
import torch
from models.cnn3d import CNN3DConfig, VoxelCNN3D, build_loss

config = CNN3DConfig(
    in_channels=4,
    num_classes=20,
    task="residue_identity",
    input_size=20,
)
model = VoxelCNN3D(config)
criterion = build_loss(config.task)

x = torch.randn(8, 4, 20, 20, 20)
logits = model(x)  # [8, 20]
```

### Regression

```python
import torch
from models.cnn3d import CNN3DConfig, VoxelCNN3D, build_loss

config = CNN3DConfig(
    in_channels=1,
    num_classes=None,
    task="regression",
    input_size=20,
)
model = VoxelCNN3D(config)
criterion = build_loss(config.task)

x = torch.randn(8, 1, 20, 20, 20)
pred = model(x)  # [8, 1]
```

## Notes

- Flattened feature size is inferred dynamically at initialization using a dummy tensor built from `in_channels` and `input_size`.
- Because this module provides logits directly, pair classification with `CrossEntropyLoss` (which internally applies `log_softmax`).

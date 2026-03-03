"""PyTorch 3D CNN models for voxel-based protein environment tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class CNN3DConfig:
    """Configuration for the baseline 3D CNN model."""

    in_channels: int
    num_classes: int | None = 20
    dropout: float = 0.5
    input_size: int = 20
    conv_channels: Sequence[int] = (100, 200, 400)
    fc_hidden: int = 1000
    task: str = "residue_identity"


class VoxelCNN3D(nn.Module):
    """Baseline architecture matching the legacy paper intent (3 conv blocks + FC head)."""

    def __init__(self, config: CNN3DConfig):
        super().__init__()
        if config.task in {"residue_identity", "mutation_activity"} and config.num_classes is None:
            raise ValueError("num_classes is required for classification tasks")

        channels = list(config.conv_channels)
        self.config = config
        self.features = nn.Sequential(
            nn.Conv3d(config.in_channels, channels[0], kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Dropout3d(config.dropout),
            nn.Conv3d(channels[0], channels[1], kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Dropout3d(config.dropout),
            nn.Conv3d(channels[1], channels[2], kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
            nn.Dropout3d(config.dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, config.in_channels, config.input_size, config.input_size, config.input_size)
            flattened = self.features(dummy).flatten(1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flattened, config.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )
        output_dim = config.num_classes if config.num_classes is not None else 1
        self.head = nn.Linear(config.fc_hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return self.head(x)


def build_loss(task: str, class_weights: torch.Tensor | None = None) -> nn.Module:
    """Build a loss module based on task type."""

    if task in {"residue_identity", "mutation_activity"}:
        return nn.CrossEntropyLoss(weight=class_weights)
    if task == "regression":
        return nn.MSELoss()
    raise ValueError(f"Unsupported task: {task}")

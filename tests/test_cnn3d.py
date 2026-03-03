import pytest


torch = pytest.importorskip("torch")

from models.cnn3d import CNN3DConfig, VoxelCNN3D, build_loss


def test_voxel_cnn3d_forward_shape_for_classification():
    config = CNN3DConfig(in_channels=4, num_classes=20, input_size=20, task="residue_identity")
    model = VoxelCNN3D(config)
    x = torch.randn(2, 4, 20, 20, 20)
    y = model(x)
    assert y.shape == (2, 20)


def test_voxel_cnn3d_forward_shape_for_regression():
    config = CNN3DConfig(in_channels=1, num_classes=None, input_size=20, task="regression")
    model = VoxelCNN3D(config)
    x = torch.randn(3, 1, 20, 20, 20)
    y = model(x)
    assert y.shape == (3, 1)


def test_build_loss_supports_tasks():
    assert isinstance(build_loss("residue_identity"), torch.nn.CrossEntropyLoss)
    assert isinstance(build_loss("mutation_activity"), torch.nn.CrossEntropyLoss)
    assert isinstance(build_loss("regression"), torch.nn.MSELoss)


def test_classification_requires_num_classes():
    with pytest.raises(ValueError):
        VoxelCNN3D(CNN3DConfig(in_channels=1, num_classes=None, task="residue_identity"))

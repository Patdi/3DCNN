"""Legacy layers shim.

The old Theano layer implementations were replaced by PyTorch modules in
`models/cnn3d.py`.
"""

from models.cnn3d import CNN3DConfig, VoxelCNN3D, build_loss

__all__ = ["CNN3DConfig", "VoxelCNN3D", "build_loss"]

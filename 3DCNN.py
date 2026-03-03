#!/usr/bin/env python3
"""Compatibility shim for the legacy Theano training entrypoint.

This file now dispatches to the modern PyTorch trainer in
`scripts/train_voxel_cnn.py`.
"""

from scripts.train_voxel_cnn import main


if __name__ == "__main__":
    main()

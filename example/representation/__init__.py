# -*- coding: utf-8 -*-
"""
Contrastive Representation Learning Module for EasyCarla-RL.

Exports:
    StateEncoder              - MLP encoder for 307-dim state vectors
    ContrastiveLoss           - Temperature-scaled InfoNCE loss
    HDF5Dataset               - Memory-efficient HDF5 dataset
    ContrastiveBatchConstructor - Constructs (anchor, positive, negative) triplets
"""

from representation.encoder import StateEncoder, ContrastiveLoss
from representation.contrastive_dataset import (
    HDF5Dataset,
    ContrastiveBatchConstructor,
    RandomBatchSampler,
    build_temporal_neighbor_map,
)

__all__ = [
    "StateEncoder",
    "ContrastiveLoss",
    "HDF5Dataset",
    "ContrastiveBatchConstructor",
    "RandomBatchSampler",
    "build_temporal_neighbor_map",
]
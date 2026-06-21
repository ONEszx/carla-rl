# -*- coding: utf-8 -*-
"""
Contrastive Learning Dataset for EasyCarla-RL.

This module provides:
- HDF5Dataset: memory-efficient access to HDF5 state observations
- ContrastiveBatchConstructor: constructs (anchor, positive, negative) triplets per batch
- build_temporal_neighbor_map: pre-computes episode boundaries for neighbor-aware sampling
"""

import os
import h5py
import numpy as np
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset, Sampler


class HDF5Dataset(Dataset):
    """
    Memory-efficient dataset that reads states directly from HDF5.

    Reads only the 'observations' key from the HDF5 file on demand,
    so it works with large datasets (1M+ samples) without loading everything into RAM.

    Args:
        hdf5_path:   Path to the offline HDF5 dataset.
        observations_key: HDF5 key for state observations (default 'observations').
        episode_ids:  Optional episode_id array. If provided, used to determine
                     episode boundaries for temporal neighbor construction.
        timesteps:   Optional timestep array per transition.
    """

    def __init__(
        self,
        hdf5_path: str,
        observations_key: str = "observations",
        episode_ids: Optional[np.ndarray] = None,
        timesteps: Optional[np.ndarray] = None,
    ):
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 dataset not found: {hdf5_path}")

        self.hdf5_path = hdf5_path
        self.observations_key = observations_key

        with h5py.File(hdf5_path, "r") as hf:
            self.N = hf[observations_key].shape[0]
            self.state_dim = hf[observations_key].shape[1]

        # For small datasets, load all states into memory for fast batch access
        # Threshold: 50,000 samples (~150 MB for 307-dim float32)
        LOAD_THRESHOLD = 50_000
        self._all_states = None
        if self.N <= LOAD_THRESHOLD:
            print(f"  [Hint] Loading all {self.N} states into memory for fast access...")
            with h5py.File(hdf5_path, "r") as hf:
                self._all_states = hf[observations_key][:].astype(np.float32)

        self.episode_ids = episode_ids
        self.timesteps = timesteps

        # Build episode boundary map for temporal neighbor construction
        self.episode_boundaries = self._build_episode_boundaries()

    def _build_episode_boundaries(self) -> Dict[int, Tuple[int, int]]:
        """
        Build a dict mapping episode_id -> (start_idx, end_idx) for the dataset.

        If episode_ids are not available, uses done flags to infer boundaries.
        """
        if self.episode_ids is not None:
            unique_eps = np.unique(self.episode_ids)
            boundaries = {}
            for ep in unique_eps:
                mask = self.episode_ids == ep
                indices = np.where(mask)[0]
                if len(indices) > 0:
                    boundaries[ep] = (indices[0], indices[-1])
            return boundaries

        # Fallback: infer from done flags (last step of each episode)
        try:
            with h5py.File(self.hdf5_path, "r") as hf:
                dones = hf.get("dones", hf.get("done", np.zeros(self.N)))
                if isinstance(dones, h5py.Dataset):
                    # Read in chunks to avoid memory issues
                    done_flags = np.zeros(self.N, dtype=np.uint8)
                    chunk_size = 100000
                    for i in range(0, self.N, chunk_size):
                        end = min(i + chunk_size, self.N)
                        done_flags[i:end] = dones[i:end]
                else:
                    done_flags = dones.astype(np.uint8)
        except Exception:
            # If we can't read dones, return no boundaries (treat as one long episode)
            return {0: (0, self.N - 1)}

        boundaries = {}
        ep_start = 0
        for i, done in enumerate(done_flags):
            if done > 0 or i == self.N - 1:
                boundaries[len(boundaries)] = (ep_start, i)
                ep_start = i + 1
                if ep_start >= self.N:
                    break
        return boundaries

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Read a single state from HDF5.
        """
        with h5py.File(self.hdf5_path, "r") as hf:
            state = hf[self.observations_key][idx]
        return torch.from_numpy(state.astype(np.float32))

    def get_state(self, idx: int) -> np.ndarray:
        """
        Read a single state as numpy array (non-tensor access).
        """
        with h5py.File(self.hdf5_path, "r") as hf:
            return hf[self.observations_key][idx]

    def get_states_batch(self, indices: np.ndarray) -> np.ndarray:
        """
        Read a batch of states. Handles arbitrary indices including duplicates
        (from negative sampling with replacement).

        Always returns (batch_size, state_dim) — the input indices shape
        only determines how many samples to read, not the output shape.

        Args:
            indices: numpy array of integer indices (can have duplicates)

        Returns:
            (batch_size, state_dim) numpy array
        """
        indices = np.atleast_1d(np.asarray(indices).astype(np.int64)).reshape(-1)

        if self._all_states is not None:
            valid = np.clip(indices, 0, self.N - 1)
            return self._all_states[valid]

        # Deduplicate, read unique states once, distribute back to original order
        unique_idx, inverse = np.unique(indices, return_inverse=True)

        with h5py.File(self.hdf5_path, "r") as hf:
            unique_states = hf[self.observations_key][unique_idx]

        # Flatten, apply inverse mapping, reshape to (num_indices, state_dim)
        flat = unique_states.reshape(unique_states.shape[0], -1)
        result = flat[inverse].reshape(len(indices), self.state_dim)
        return result


class ContrastiveBatchConstructor:
    """
    Constructs (anchor, positive, negative) triplets from a HDF5Dataset on-the-fly.

    Positive samples: temporally adjacent states within the same episode.
    Negative samples: states from other episodes or far-away in time.

    Args:
        dataset:         HDF5Dataset instance.
        neighbor_window: How many steps before/after to consider as positive.
                         (default 5, meaning indices [i-5, i+5] are positive candidates).
        num_negatives:   Number of negative samples per anchor (default 255).
                         When set to 0, uses in-batch negatives only.
        exclude_neighbors: Whether to exclude immediate neighbor indices from negatives pool.
    """

    def __init__(
        self,
        dataset: HDF5Dataset,
        neighbor_window: int = 5,
        num_negatives: int = 255,
        exclude_neighbors: bool = True,
    ):
        self.dataset = dataset
        self.neighbor_window = neighbor_window
        self.num_negatives = num_negatives
        self.exclude_neighbors = exclude_neighbors
        self.N = len(dataset)

        # Build index-to-episode mapping
        self._build_index_to_episode()

    def _build_index_to_episode(self):
        """Build a fast lookup from index to episode boundaries."""
        if self.dataset.episode_ids is not None:
            # Direct lookup via numpy array
            self.episode_ids_arr = self.dataset.episode_ids
            self.index_to_ep = None
        elif self.dataset.episode_boundaries:
            # Inverse map: index -> (ep_start, ep_end)
            self.index_to_ep = {}
            self.episode_ids_arr = None
            for ep_id, (start, end) in self.dataset.episode_boundaries.items():
                for idx in range(start, end + 1):
                    self.index_to_ep[idx] = (start, end)
        else:
            self.episode_ids_arr = None
            self.index_to_ep = None

    def _get_episode_range(self, idx: int) -> Tuple[int, int]:
        """Get the [start, end] index range for the episode containing idx."""
        if self.episode_ids_arr is not None:
            ep = self.episode_ids_arr[idx]
            return self.dataset.episode_boundaries.get(ep, (0, self.N - 1))
        elif self.index_to_ep is not None:
            return self.index_to_ep.get(idx, (0, self.N - 1))
        else:
            return (0, self.N - 1)

    def get_batch(
        self,
        anchor_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construct positive and negative samples for a batch of anchor indices.

        Args:
            anchor_indices: (batch_size,) array of anchor sample indices

        Returns:
            anchors:    (batch_size, state_dim)
            positives:   (batch_size, state_dim)
            negatives:   (batch_size, num_negatives, state_dim) or None
        """
        batch_size = len(anchor_indices)

        # --- Positive samples: temporal neighbors ---
        pos_indices = np.zeros(batch_size, dtype=np.int64)
        for i, idx in enumerate(anchor_indices):
            ep_start, ep_end = self._get_episode_range(idx)
            # Pick one neighbor: prefer i+1 if not at end, else i-1
            if idx < ep_end:
                pos_indices[i] = idx + 1
            elif idx > ep_start:
                pos_indices[i] = idx - 1
            else:
                # Singleton episode, use random
                pos_indices[i] = np.random.randint(ep_start, ep_end + 1)

        # --- Negative sample indices ---
        neg_indices = np.zeros((batch_size, self.num_negatives), dtype=np.int64)
        for i, idx in enumerate(anchor_indices):
            ep_start, ep_end = self._get_episode_range(idx)

            if self.exclude_neighbors:
                neighbor_start = max(ep_start, idx - self.neighbor_window)
                neighbor_end = min(ep_end, idx + self.neighbor_window)
                valid_pool = list(range(ep_start, neighbor_start)) + list(range(neighbor_end + 1, ep_end + 1))
            else:
                valid_pool = list(range(ep_start, ep_end + 1))

            if idx in valid_pool:
                valid_pool.remove(idx)

            if len(valid_pool) >= self.num_negatives:
                neg_indices[i] = np.random.choice(valid_pool, size=self.num_negatives, replace=False)
            else:
                neg_indices[i] = np.random.choice(valid_pool, size=self.num_negatives, replace=True)

        # --- Batch fetch (ONE HDF5 read per category, not one per sample) ---
        anchors_np = self.dataset.get_states_batch(anchor_indices)
        positives_np = self.dataset.get_states_batch(pos_indices)

        if self.num_negatives == 0:
            return anchors_np, positives_np, None

        # Read all negatives in ONE HDF5 read per batch
        neg_flat_indices = neg_indices.reshape(-1)
        all_neg_states = self.dataset.get_states_batch(neg_flat_indices)
        negatives_np = all_neg_states.reshape(batch_size, self.num_negatives, self.dataset.state_dim)

        return anchors_np, positives_np, negatives_np


class RandomBatchSampler(Sampler):
    """
    Random sampler that generates random batches of indices.

    Args:
        dataset_size: Total number of samples in the dataset.
        batch_size:    Number of samples per batch.
        drop_last:     Whether to drop the last incomplete batch.
    """

    def __init__(self, dataset_size: int, batch_size: int, drop_last: bool = False):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        indices = np.random.permutation(self.dataset_size).tolist()
        if self.drop_last:
            indices = indices[: -(len(indices) % self.batch_size)]
        for i in range(0, len(indices), self.batch_size):
            yield indices[i : i + self.batch_size]

    def __len__(self):
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size


def build_temporal_neighbor_map(
    hdf5_path: str,
    episode_ids: Optional[np.ndarray] = None,
) -> Dict[int, Tuple[int, int]]:
    """
    Utility: build a map from index to episode boundaries.

    This can be precomputed once and reused across training runs.

    Args:
        hdf5_path:  Path to HDF5 dataset.
        episode_ids: Optional episode_id array for faster construction.

    Returns:
        Dict mapping index -> (episode_start, episode_end)
    """
    with h5py.File(hdf5_path, "r") as hf:
        N = hf["observations"].shape[0]

    if episode_ids is not None:
        boundaries = {}
        unique_eps = np.unique(episode_ids)
        for ep in unique_eps:
            mask = episode_ids == ep
            indices = np.where(mask)[0]
            if len(indices) > 0:
                start, end = indices[0], indices[-1]
                for idx in range(start, end + 1):
                    boundaries[idx] = (start, end)
        return boundaries

    # Fallback: use done flags
    try:
        with h5py.File(hdf5_path, "r") as hf:
            done_key = "dones" if "dones" in hf else "done"
            done_flags = hf[done_key][:].astype(np.uint8)
    except Exception:
        return {i: (0, N - 1) for i in range(N)}

    boundaries = {}
    ep_start = 0
    ep_id = 0
    for i, done in enumerate(done_flags):
        if done > 0 or i == N - 1:
            ep_end = i
            for j in range(ep_start, ep_end + 1):
                boundaries[j] = (ep_start, ep_end)
            ep_start = i + 1
            ep_id += 1

    return boundaries
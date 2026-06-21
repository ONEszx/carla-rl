# -*- coding: utf-8 -*-
"""
Priority Sampler for EasyCarla-RL.

Loads precomputed priority scores (sparsity-based) from an npz file and
converts them into weighted sampling probabilities for offline RL training.

Ablation flags (to be controlled by the training entry point):
    - use_priority:     If False, fall back to uniform sampling.
    - priority_power:    Exponent alpha in p_i ∝ (score_i + eps)^alpha.
                         Higher alpha = more weight on rare samples.
    - priority_offset:   Small epsilon added to scores before power transform.

Priority scores are precomputed by train_representation.py using kNN distance
in the learned latent space. Higher score = sparser = more valuable.
"""

import os
import numpy as np
from typing import Optional


class PrioritySampler:
    """
    Priority-weighted batch sampler for offline RL.

    Loads precomputed priority scores and converts them to sampling probabilities.

    Modes:
        uniform:   All samples equally likely (baseline / ablation baseline).
        priority:  Sample proportional to (score + eps)^alpha.

    Args:
        priority_path:    Path to priority_scores.npz file.
                          If None or file doesn't exist, falls back to uniform.
        priority_power:   Exponent for priority transform (default 2.0).
                          alpha=0 → uniform sampling.
                          alpha>0 → higher weight to sparser samples.
        priority_offset:  Small epsilon added before power (default 1e-6).
                          Prevents division by zero for low scores.
        max_priority_weight: Max ratio cap to prevent degenerate sampling (default 20.0).
        device:           Device to load scores on (default cpu).
    """

    def __init__(
        self,
        priority_path: Optional[str] = None,
        priority_power: float = 2.0,
        priority_offset: float = 1e-6,
        max_priority_weight: float = 20.0,
        device: str = "cpu",
    ):
        self.priority_power = priority_power
        self.priority_offset = priority_offset
        self.max_priority_weight = max_priority_weight
        self.device = device

        self.priority_scores: Optional[np.ndarray] = None
        self.sampling_probs: Optional[np.ndarray] = None
        self._enabled = False

        if priority_path is not None and os.path.exists(priority_path):
            self._load_scores(priority_path)
        else:
            if priority_path is not None:
                print(f"[PrioritySampler] File not found: {priority_path} — falling back to uniform sampling")
            else:
                print("[PrioritySampler] No priority_path provided — using uniform sampling")

    def _load_scores(self, priority_path: str):
        """Load and preprocess priority scores."""
        data = np.load(priority_path)
        if "scores" in data:
            self.priority_scores = data["scores"].astype(np.float32)
        elif "raw_scores" in data:
            raw = data["raw_scores"].astype(np.float32)
            # Normalize
            if raw.max() > raw.min():
                self.priority_scores = (raw - raw.min()) / (raw.max() - raw.min())
            else:
                self.priority_scores = np.zeros_like(raw)
        else:
            print(f"[PrioritySampler] No valid score key found in {priority_path}")
            return

        N = len(self.priority_scores)
        print(f"[PrioritySampler] Loaded {N} priority scores from {priority_path}")
        print(f"  Score range: [{self.priority_scores.min():.4f}, {self.priority_scores.max():.4f}]")

        self._build_sampling_probs()
        self._enabled = True

    def _build_sampling_probs(self):
        """Convert priority scores to sampling probabilities via p ∝ (s+eps)^alpha."""
        scores = self.priority_scores + self.priority_offset
        weights = np.power(scores, self.priority_power).astype(np.float64)

        # Cap extreme weights to avoid degenerate sampling concentration.
        if self.max_priority_weight is not None and self.max_priority_weight > 0:
            mean_weight = weights.mean()
            if np.isfinite(mean_weight) and mean_weight > 0:
                cap_value = mean_weight * self.max_priority_weight
                weights = np.minimum(weights, cap_value)

        # Normalize to valid probability distribution (numerically safe)
        w_sum = weights.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            # Fallback to uniform if weights are degenerate
            probs = np.ones(len(weights), dtype=np.float64) / len(weights)
        else:
            probs = weights / w_sum
            # Enforce sum == 1.0 via explicit correction
            probs = probs / probs.sum()

        self.sampling_probs = probs.astype(np.float64)

        print(f"  Priority enabled: power={self.priority_power}, eps={self.priority_offset}, cap={self.max_priority_weight}")
        print(f"  Prob range: [{self.sampling_probs.min():.6f}, {self.sampling_probs.max():.6f}]")
        print(f"  Top-1 prob: {self.sampling_probs.max():.4f}, Bottom-1 prob: {self.sampling_probs.min():.6f}")

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """
        Sample a batch of indices according to priority probabilities.

        Args:
            batch_size: Number of indices to sample.

        Returns:
            (batch_size,) array of sampled indices.
        """
        if not self._enabled or self.sampling_probs is None:
            # Uniform fallback
            N = len(self.priority_scores) if self.priority_scores is not None else 1
            return np.random.randint(0, N, size=batch_size)

        return np.random.choice(
            len(self.sampling_probs),
            size=batch_size,
            replace=True,
            p=self.sampling_probs,
        )

    def get_weights(self, indices: np.ndarray) -> np.ndarray:
        """
        Get the priority weight for a set of indices (for PER-style importance sampling).

        Args:
            indices: (batch_size,) array of sample indices.

        Returns:
            (batch_size,) array of normalized importance weights.
        """
        if not self._enabled or self.priority_scores is None:
            return np.ones(len(indices), dtype=np.float32)

        scores = self.priority_scores[indices] + self.priority_offset
        weights = np.power(scores, self.priority_power)
        normalized = weights / weights.sum()
        return normalized.astype(np.float32)

    @property
    def is_enabled(self) -> bool:
        """Whether priority sampling is active (scores loaded and power > 0)."""
        return self._enabled and self.priority_power > 0

    def summary(self) -> str:
        """Return a human-readable summary string."""
        if not self._enabled:
            return "PrioritySampler: uniform (disabled)"
        return (
            f"PrioritySampler: enabled | "
            f"power={self.priority_power} | "
            f"eps={self.priority_offset} | "
            f"scores_loaded={len(self.priority_scores)}"
        )


class PriorityBuffer:
    """
    Replay buffer wrapper that applies priority sampling when fetching batches.

    Wraps an existing replay buffer (e.g. SimpleReplayBuffer) and intercepts
    the sample() method to return priority-sampled indices instead.

    This is used to integrate priority sampling into the existing
    train_diffusion_ql.py training flow without changing the buffer interface.

    Args:
        base_buffer:       The underlying replay buffer.
        priority_sampler:  PrioritySampler instance.
    """

    def __init__(
        self,
        base_buffer,  # SimpleReplayBuffer or similar
        priority_sampler: Optional[PrioritySampler] = None,
    ):
        self.base_buffer = base_buffer
        self.priority_sampler = priority_sampler

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch from the buffer.

        If priority_sampler is enabled and has loaded scores,
        samples indices according to priority probabilities.
        Otherwise falls back to the base buffer's own sampling.

        Returns:
            Same format as base_buffer.sample(batch_size)
        """
        if (
            self.priority_sampler is not None
            and self.priority_sampler.is_enabled
        ):
            N = len(self.base_buffer)
            indices = self.priority_sampler.sample_indices(batch_size)
            indices = np.clip(indices, 0, N - 1)
            return self.base_buffer.sample_indices(indices)
        else:
            return self.base_buffer.sample(batch_size)

    def __len__(self) -> int:
        return len(self.base_buffer)

    def add(self, *args, **kwargs):
        return self.base_buffer.add(*args, **kwargs)
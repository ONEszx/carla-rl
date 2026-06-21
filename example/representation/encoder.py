# -*- coding: utf-8 -*-
"""
State Representation Encoder for EasyCarla-RL.

A simple MLP-based encoder that maps raw state vectors (307-dim) into a
compact latent representation (64-dim by default).

This encoder is used in two stages:
- Offline: trained via contrastive learning, then frozen
- Online: reused as a shared representation for multiple Q heads
"""

import torch
import torch.nn as nn
from typing import Optional


class StateEncoder(nn.Module):
    """
    MLP encoder for 307-dim state vectors.

    Architecture:
        Linear(307, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, latent_dim)

    Args:
        state_dim:        Input state dimension (307 for EasyCarla).
        latent_dim:       Output embedding dimension (default 64).
        hidden_dim:       Hidden layer dimension (default 256).
        use_layer_norm:   Whether to use layer normalization (default False).
    """

    def __init__(
        self,
        state_dim: int = 307,
        latent_dim: int = 64,
        hidden_dim: int = 256,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        layers = [
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        ]

        if use_layer_norm:
            layers.insert(1, nn.LayerNorm(hidden_dim))
            layers.insert(4, nn.LayerNorm(hidden_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of state vectors.

        Args:
            x: (batch_size, state_dim) raw state tensor

        Returns:
            (batch_size, latent_dim) embedding tensor
        """
        return self.encoder(x)

    def get_config(self) -> dict:
        """Return encoder hyperparameters for logging/checkpointing."""
        return {
            "state_dim": self.state_dim,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
        }


class ContrastiveLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy (InfoNCE) loss.

    For a batch of (anchor, positive) pairs, this loss pulls anchors closer
    to their positives and pushes them away from other negatives in the batch.

    Args:
        temperature: Scaling factor for logits (default 0.07).
                     Smaller values make the distribution sharper.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            anchors:   (batch_size, latent_dim) anchor embeddings
            positives: (batch_size, latent_dim) positive embeddings
            negatives: (batch_size, num_negatives, latent_dim) optional explicit negatives.
                      If None, use all other embeddings in the batch as negatives.

        Returns:
            Scalar loss
        """
        batch_size = anchors.shape[0]

        # Normalize embeddings
        anchors = torch.nn.functional.normalize(anchors, dim=-1)
        positives = torch.nn.functional.normalize(positives, dim=-1)

        # Compute similarity between anchors and positives
        pos_sim = (anchors * positives).sum(dim=-1)  # (batch_size,)

        if negatives is None:
            # Use in-batch negatives (standard InfoNCE)
            all_embeds = torch.cat([anchors, positives], dim=0)
            all_embeds = torch.nn.functional.normalize(all_embeds, dim=-1)

            # Full similarity matrix
            sim_matrix = torch.mm(all_embeds, all_embeds.T) / self.temperature

            # Mask out self-similarity (diagonal)
            mask = torch.eye(2 * batch_size, device=sim_matrix.device).bool()
            sim_matrix.masked_fill_(mask, float("-inf"))

            # Anchor indices: 0..batch_size-1, positive indices: batch_size..2*batch_size-1
            target = torch.arange(batch_size, device=sim_matrix.device)
            positive_idx = target + batch_size

            logits = torch.cat([pos_sim.unsqueeze(-1) / self.temperature, sim_matrix[target]], dim=-1)
            target_idx = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        else:
            negatives = torch.nn.functional.normalize(negatives, dim=-1)
            neg_sim = (anchors.unsqueeze(1) * negatives).sum(dim=-1)  # (batch, num_neg)

            logits = torch.cat([pos_sim.unsqueeze(-1) / self.temperature, neg_sim / self.temperature], dim=-1)
            target_idx = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        loss = torch.nn.functional.cross_entropy(logits, target_idx)
        return loss
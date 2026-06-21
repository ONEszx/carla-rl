# -*- coding: utf-8 -*-
"""
Active sample/action selector for online finetuning.

Combines uncertainty and novelty scores without changing the core RL backbone.
"""

from typing import Dict, Optional

import numpy as np
import torch


class ActiveSelector:
    def __init__(
        self,
        lambda_uncertainty: float = 0.5,
        lambda_novelty: float = 0.5,
        retain_threshold: float = 0.0,
        max_memory_size: int = 5000,
    ):
        self.lambda_uncertainty = float(lambda_uncertainty)
        self.lambda_novelty = float(lambda_novelty)
        self.retain_threshold = float(retain_threshold)
        self.max_memory_size = int(max_memory_size)
        self._reference_embeddings = []
        self._online_embeddings = []

    def score_action(self, q_mean: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        return q_mean + self.lambda_uncertainty * uncertainty

    def should_retain(self, uncertainty: float, novelty: float) -> bool:
        return (self.lambda_uncertainty * float(uncertainty) + self.lambda_novelty * float(novelty)) >= self.retain_threshold

    def update_reference_memory(self, embeddings: np.ndarray) -> None:
        for row in np.asarray(embeddings, dtype=np.float32):
            self._reference_embeddings.append(row)
        if len(self._reference_embeddings) > self.max_memory_size:
            self._reference_embeddings = self._reference_embeddings[-self.max_memory_size:]

    def update_online_memory(self, embedding: np.ndarray) -> None:
        self._online_embeddings.append(np.asarray(embedding, dtype=np.float32))
        if len(self._online_embeddings) > self.max_memory_size:
            self._online_embeddings = self._online_embeddings[-self.max_memory_size:]

    def novelty_score(self, embedding: np.ndarray, use_online_memory: bool = True) -> float:
        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        memories = list(self._reference_embeddings)
        if use_online_memory:
            memories.extend(self._online_embeddings)
        if not memories:
            return 0.0
        memory = np.asarray(memories, dtype=np.float32)
        distances = np.linalg.norm(memory - query, axis=1)
        if distances.size == 0:
            return 0.0
        return float(np.min(distances))

    def active_score(self, q_mean: float, uncertainty: float, novelty: float) -> float:
        return float(q_mean + self.lambda_uncertainty * uncertainty + self.lambda_novelty * novelty)

    def summarize(self) -> Dict[str, int]:
        return {
            "reference_memory": len(self._reference_embeddings),
            "online_memory": len(self._online_embeddings),
        }

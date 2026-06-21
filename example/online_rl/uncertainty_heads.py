# -*- coding: utf-8 -*-
"""
Lightweight uncertainty heads for active online finetuning.

These heads reuse the current RL encoder/state representation and provide:
- multi-head Q estimates
- disagreement / std uncertainty
- offline warmup on replay samples
"""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SingleQHead(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class UncertaintyHeadEnsemble(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_heads: int = 4,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        bootstrap_prob: float = 0.8,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.bootstrap_prob = float(np.clip(bootstrap_prob, 0.05, 1.0))
        self.device = torch.device(device)
        self.heads = nn.ModuleList([
            _SingleQHead(state_dim, action_dim, hidden_dim=hidden_dim)
            for _ in range(self.num_heads)
        ])
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        values = [head(state, action) for head in self.heads]
        return torch.cat(values, dim=1)

    @torch.no_grad()
    def score_actions(self, state: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        q_values = self.forward(state, action)
        q_mean = q_values.mean(dim=1)
        q_std = q_values.std(dim=1, unbiased=False)
        q_max = q_values.max(dim=1).values
        q_min = q_values.min(dim=1).values
        disagreement = q_max - q_min
        return {
            "q_values": q_values,
            "q_mean": q_mean,
            "q_std": q_std,
            "disagreement": disagreement,
        }

    def train_step(
        self,
        encoded_state: torch.Tensor,
        action: torch.Tensor,
        target_q: torch.Tensor,
    ) -> Dict[str, float]:
        q_values = self.forward(encoded_state, action)
        losses = []
        kept_counts = []
        for head_idx in range(self.num_heads):
            mask = (torch.rand_like(target_q) < self.bootstrap_prob).float()
            kept = float(mask.sum().item())
            kept_counts.append(kept)
            if kept < 1.0:
                mask = torch.ones_like(target_q)
                kept = float(mask.sum().item())
            per_sample = F.mse_loss(q_values[:, head_idx:head_idx + 1], target_q, reduction="none")
            loss = (per_sample * mask).sum() / mask.sum().clamp_min(1.0)
            losses.append(loss)
        total_loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            q_mean = q_values.mean(dim=1)
            q_std = q_values.std(dim=1, unbiased=False)
        return {
            "loss": float(total_loss.item()),
            "q_mean": float(q_mean.mean().item()),
            "q_std": float(q_std.mean().item()),
            "bootstrap_kept": float(np.mean(kept_counts)) if kept_counts else 0.0,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "num_heads": self.num_heads,
                "bootstrap_prob": self.bootstrap_prob,
                "device": str(self.device),
            },
            path,
        )

    def load(self, path: str, strict: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)
        state_dict = payload.get("state_dict", payload)
        self.load_state_dict(state_dict, strict=strict)

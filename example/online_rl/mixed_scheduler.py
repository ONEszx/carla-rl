# -*- coding: utf-8 -*-
"""
Lightweight offline/online ratio scheduler for active finetuning.
"""

from dataclasses import dataclass


@dataclass
class MixedReplaySchedule:
    start_offline_ratio: float = 0.85
    end_offline_ratio: float = 0.45
    warmup_epochs: int = 2
    ramp_epochs: int = 4

    def ratio_for_epoch(self, epoch: int, high_value_retention: float = 0.0, reward_trend: float = 0.0) -> float:
        epoch = max(1, int(epoch))
        if epoch <= self.warmup_epochs:
            ratio = self.start_offline_ratio
        else:
            ramp_progress = min(1.0, max(0.0, (epoch - self.warmup_epochs) / max(1, self.ramp_epochs)))
            ratio = self.start_offline_ratio + (self.end_offline_ratio - self.start_offline_ratio) * ramp_progress

        if high_value_retention > 0.6:
            ratio -= 0.05
        elif high_value_retention < 0.2:
            ratio += 0.05

        if reward_trend < 0.0:
            ratio += 0.05
        elif reward_trend > 0.0:
            ratio -= 0.02

        return float(min(0.95, max(0.05, ratio)))

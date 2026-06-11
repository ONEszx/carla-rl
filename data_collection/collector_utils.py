"""Utilities for collecting EasyCarla trajectories and exporting HDF5 datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import h5py
import numpy as np

STATE_COMPONENT_SPECS = [
    ("ego_state", 9),
    ("lane_info", 2),
    ("lidar", 240),
    ("nearby_vehicles", 20),
    ("waypoints", 36),
]
STATE_DIM = sum(dim for _, dim in STATE_COMPONENT_SPECS)
ACTION_DIM = 3
SOURCE_MODE_TO_ID = {
    "random": 0,
    "autopilot": 1,
    "policy": 2,
    "mixed": 3,
}


def _to_flat_array(obs_dict: Dict[str, np.ndarray], key: str, expected_dim: int) -> np.ndarray:
    if key not in obs_dict:
        raise KeyError(f"Observation key '{key}' is missing from env output.")

    value = np.asarray(obs_dict[key], dtype=np.float32).reshape(-1)
    if value.shape[0] != expected_dim:
        raise ValueError(
            f"Observation key '{key}' has dim {value.shape[0]}, expected {expected_dim}."
        )
    return value


def flatten_obs_dict(obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """Flatten EasyCarla observation dict into the 307-dim state vector used by DiffusionQL."""
    parts = [_to_flat_array(obs_dict, key, dim) for key, dim in STATE_COMPONENT_SPECS]
    state = np.concatenate(parts, axis=0).astype(np.float32)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"Flattened state has dim {state.shape[0]}, expected {STATE_DIM}.")
    return state


@dataclass
class TransitionBuffer:
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    next_observations: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    costs: List[float] = field(default_factory=list)
    episode_ids: List[int] = field(default_factory=list)
    timesteps: List[int] = field(default_factory=list)
    source_modes: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.observations)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: bool,
        cost: float,
        episode_id: int,
        timestep: int,
        source_mode: str,
    ) -> None:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        if obs.shape[0] != STATE_DIM:
            raise ValueError(f"obs dim {obs.shape[0]} != expected {STATE_DIM}")
        if next_obs.shape[0] != STATE_DIM:
            raise ValueError(f"next_obs dim {next_obs.shape[0]} != expected {STATE_DIM}")
        if action.shape[0] != ACTION_DIM:
            raise ValueError(f"action dim {action.shape[0]} != expected {ACTION_DIM}")

        self.observations.append(obs)
        self.actions.append(action)
        self.next_observations.append(next_obs)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.costs.append(float(cost))
        self.episode_ids.append(int(episode_id))
        self.timesteps.append(int(timestep))
        self.source_modes.append(SOURCE_MODE_TO_ID.get(source_mode, SOURCE_MODE_TO_ID["mixed"]))

    def as_arrays(self) -> Dict[str, np.ndarray]:
        if len(self) == 0:
            return {
                "observations": np.zeros((0, STATE_DIM), dtype=np.float32),
                "actions": np.zeros((0, ACTION_DIM), dtype=np.float32),
                "next_observations": np.zeros((0, STATE_DIM), dtype=np.float32),
                "rewards": np.zeros((0,), dtype=np.float32),
                "done": np.zeros((0,), dtype=np.float32),
                "dones": np.zeros((0,), dtype=np.float32),
                "costs": np.zeros((0,), dtype=np.float32),
                "episode_ids": np.zeros((0,), dtype=np.int32),
                "timesteps": np.zeros((0,), dtype=np.int32),
                "source_mode": np.zeros((0,), dtype=np.int32),
            }

        observations = np.stack(self.observations).astype(np.float32)
        actions = np.stack(self.actions).astype(np.float32)
        next_observations = np.stack(self.next_observations).astype(np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        costs = np.asarray(self.costs, dtype=np.float32)
        episode_ids = np.asarray(self.episode_ids, dtype=np.int32)
        timesteps = np.asarray(self.timesteps, dtype=np.int32)
        source_mode = np.asarray(self.source_modes, dtype=np.int32)

        return {
            "observations": observations,
            "actions": actions,
            "next_observations": next_observations,
            "rewards": rewards,
            "done": dones,
            "dones": dones.copy(),
            "costs": costs,
            "episode_ids": episode_ids,
            "timesteps": timesteps,
            "source_mode": source_mode,
        }


def save_buffer_to_hdf5(buffer: TransitionBuffer, output_path: str) -> Dict[str, tuple]:
    """Write the collected transitions to HDF5 using the current EasyCarla schema."""
    arrays = buffer.as_arrays()
    with h5py.File(output_path, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=value)

    return {key: value.shape for key, value in arrays.items()}


def summarize_buffer(buffer: TransitionBuffer) -> Dict[str, object]:
    arrays = buffer.as_arrays()
    rewards = arrays["rewards"]
    dones = arrays["done"]
    return {
        "num_transitions": int(arrays["observations"].shape[0]),
        "state_dim": int(arrays["observations"].shape[1]) if arrays["observations"].ndim == 2 and arrays["observations"].shape[0] > 0 else STATE_DIM,
        "action_dim": int(arrays["actions"].shape[1]) if arrays["actions"].ndim == 2 and arrays["actions"].shape[0] > 0 else ACTION_DIM,
        "reward_mean": float(rewards.mean()) if rewards.size else 0.0,
        "reward_min": float(rewards.min()) if rewards.size else 0.0,
        "reward_max": float(rewards.max()) if rewards.size else 0.0,
        "num_terminal": int(dones.sum()) if dones.size else 0,
    }

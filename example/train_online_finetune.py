# -*- coding: utf-8 -*-
"""
Minimal offline-to-online finetuning entry for EasyCarla Diffusion QL.

This script keeps Chapter-1 online RL intentionally simple:
1. Load an offline-trained Diffusion QL checkpoint
2. Optionally load the finetuned RL encoder checkpoint
3. Interact with CARLA to collect online transitions
4. Mix offline + online replay
5. Continue finetuning the same Diffusion QL backbone

It also supports an active online mode that reuses the same backbone while
adding lightweight uncertainty heads, novelty-guided retention, and an
adaptive mixed replay schedule.
"""

import argparse
import copy
import os
import random
import re
import sys
import time
from typing import Dict, Optional, Tuple

import gym
import easycarla
import h5py
import numpy as np
import torch

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1) if hasattr(sys.stdout, 'fileno') else sys.stdout
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.ql_diffusion import Diffusion_QL
from data_collection.collector_utils import flatten_obs_dict
from online_rl.active_selector import ActiveSelector
from online_rl.mixed_scheduler import MixedReplaySchedule
from online_rl.uncertainty_heads import UncertaintyHeadEnsemble
from representation.encoder import StateEncoder

DEFAULT_CARLA_PARAMS = {
    'number_of_vehicles': 20,
    'number_of_walkers': 0,
    'dt': 0.1,
    'ego_vehicle_filter': 'vehicle.tesla.model3',
    'surrounding_vehicle_spawned_randomly': True,
    'port': 2000,
    'town': 'Town03',
    'use_current_world': True,
    'client_timeout': 30.0,
    'max_time_episode': 300,
    'max_waypoints': 12,
    'visualize_waypoints': False,
    'desired_speed': 8,
    'max_ego_spawn_times': 200,
    'view_mode': 'top',
    'traffic': 'off',
    'lidar_max_range': 50.0,
    'max_nearby_vehicles': 5,
}


class NumpyReplayBuffer:
    def __init__(self, obs, act, next_obs, rew, done):
        self.obs = np.asarray(obs, dtype=np.float32)
        self.act = np.asarray(act, dtype=np.float32)
        self.next_obs = np.asarray(next_obs, dtype=np.float32)
        self.rew = np.asarray(rew, dtype=np.float32).reshape(-1, 1)
        self.not_done = 1.0 - np.asarray(done, dtype=np.float32).reshape(-1, 1)
        self.device = None

    def __len__(self):
        return len(self.obs)

    def to(self, device: torch.device):
        self.device = device

    def sample(self, batch_size: int):
        if len(self.obs) == 0:
            raise ValueError("Replay buffer is empty.")
        idx = np.random.randint(0, len(self.obs), size=batch_size)
        return self.sample_indices(idx)

    def sample_indices(self, indices):
        if self.device is None:
            raise ValueError("Replay buffer device is not set. Call to(device) first.")
        idx = np.asarray(indices, dtype=np.int64)
        return (
            torch.from_numpy(self.obs[idx]).float().to(self.device),
            torch.from_numpy(self.act[idx]).float().to(self.device),
            torch.from_numpy(self.next_obs[idx]).float().to(self.device),
            torch.from_numpy(self.rew[idx]).float().to(self.device),
            torch.from_numpy(self.not_done[idx]).float().to(self.device),
        )

    def append(self, obs, act, next_obs, rew, done):
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        act = np.asarray(act, dtype=np.float32).reshape(1, -1)
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(1, -1)
        rew = np.asarray([[rew]], dtype=np.float32)
        not_done = np.asarray([[1.0 - float(done)]], dtype=np.float32)

        self.obs = np.concatenate([self.obs, obs], axis=0)
        self.act = np.concatenate([self.act, act], axis=0)
        self.next_obs = np.concatenate([self.next_obs, next_obs], axis=0)
        self.rew = np.concatenate([self.rew, rew], axis=0)
        self.not_done = np.concatenate([self.not_done, not_done], axis=0)


class MixedReplayBuffer:
    def __init__(self, offline_buffer: NumpyReplayBuffer, online_buffer: NumpyReplayBuffer, offline_ratio: float):
        self.offline_buffer = offline_buffer
        self.online_buffer = online_buffer
        self.offline_ratio = float(np.clip(offline_ratio, 0.0, 1.0))
        self.device = offline_buffer.device

    def __len__(self):
        return len(self.offline_buffer) + len(self.online_buffer)

    def sample(self, batch_size: int):
        if len(self.online_buffer) == 0:
            return self.offline_buffer.sample(batch_size)

        offline_batch = int(round(batch_size * self.offline_ratio))
        offline_batch = min(batch_size, max(0, offline_batch))
        online_batch = batch_size - offline_batch

        if offline_batch == 0:
            return self.online_buffer.sample(batch_size)
        if online_batch == 0:
            return self.offline_buffer.sample(batch_size)

        offline_tensors = self.offline_buffer.sample(offline_batch)
        online_tensors = self.online_buffer.sample(online_batch)
        return tuple(
            torch.cat([offline_tensor, online_tensor], dim=0)
            for offline_tensor, online_tensor in zip(offline_tensors, online_tensors)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal online finetuning entry for EasyCarla Diffusion QL.")
    parser.add_argument("--offline_data_path", type=str, default=os.path.join(PROJECT_ROOT, "data", "easycarla_offline_dataset.hdf5"))
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(CURRENT_DIR, "params_full"))
    parser.add_argument("--model_id", type=int, default=None, help="Checkpoint id to load. Defaults to the latest actor_*.pth.")
    parser.add_argument("--encoder_ckpt", type=str, default=None, help="Optional encoder checkpoint override.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--online_mode", type=str, default="baseline", choices=["baseline", "active"])
    parser.add_argument("--online_epochs", type=int, default=5)
    parser.add_argument("--episodes_per_epoch", type=int, default=2)
    parser.add_argument("--max_steps_per_episode", type=int, default=200)
    parser.add_argument("--updates_per_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--offline_ratio", type=float, default=0.7)
    parser.add_argument("--warmup_random_steps", type=int, default=100)
    parser.add_argument("--exploration_noise", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=1)

    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder_lr", type=float, default=1e-4)
    parser.add_argument("--grad_norm", type=float, default=1.0)
    parser.add_argument("--encoder_grad_norm", type=float, default=1.0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--beta_schedule", type=str, default="vp", choices=["vp", "linear", "cosine"])
    parser.add_argument("--n_timesteps", type=int, default=5)

    parser.add_argument("--uncertainty_heads", type=int, default=4)
    parser.add_argument("--uncertainty_hidden_dim", type=int, default=128)
    parser.add_argument("--uncertainty_lr", type=float, default=1e-3)
    parser.add_argument("--uncertainty_bootstrap_prob", type=float, default=0.8)
    parser.add_argument("--uncertainty_updates_per_epoch", type=int, default=50)
    parser.add_argument("--uncertainty_warmup_updates", type=int, default=100)
    parser.add_argument("--uncertainty_batch_size", type=int, default=128)
    parser.add_argument("--active_candidates", type=int, default=8)
    parser.add_argument("--active_lambda_uncertainty", type=float, default=0.5)
    parser.add_argument("--active_lambda_novelty", type=float, default=0.05)
    parser.add_argument("--retain_threshold", type=float, default=0.2)
    parser.add_argument("--novelty_memory_size", type=int, default=5000)
    parser.add_argument("--reference_memory_size", type=int, default=2000)
    parser.add_argument("--active_seed_buffer", type=int, default=32)

    parser.add_argument("--mixed_schedule_start_offline_ratio", type=float, default=0.85)
    parser.add_argument("--mixed_schedule_end_offline_ratio", type=float, default=0.45)
    parser.add_argument("--mixed_schedule_warmup_epochs", type=int, default=2)
    parser.add_argument("--mixed_schedule_ramp_epochs", type=int, default=4)

    parser.add_argument("--port", type=int, default=DEFAULT_CARLA_PARAMS['port'])
    parser.add_argument("--town", type=str, default=DEFAULT_CARLA_PARAMS['town'])
    parser.add_argument("--use_current_world", action="store_true", default=DEFAULT_CARLA_PARAMS['use_current_world'])
    parser.add_argument("--client_timeout", type=float, default=DEFAULT_CARLA_PARAMS['client_timeout'])
    parser.add_argument("--number_of_vehicles", type=int, default=DEFAULT_CARLA_PARAMS['number_of_vehicles'])
    parser.add_argument("--number_of_walkers", type=int, default=DEFAULT_CARLA_PARAMS['number_of_walkers'])
    parser.add_argument("--dt", type=float, default=DEFAULT_CARLA_PARAMS['dt'])
    parser.add_argument("--max_time_episode", type=int, default=DEFAULT_CARLA_PARAMS['max_time_episode'])
    parser.add_argument("--max_waypoints", type=int, default=DEFAULT_CARLA_PARAMS['max_waypoints'])
    parser.add_argument("--visualize_waypoints", type=int, default=0)
    parser.add_argument("--desired_speed", type=float, default=DEFAULT_CARLA_PARAMS['desired_speed'])
    parser.add_argument("--max_ego_spawn_times", type=int, default=DEFAULT_CARLA_PARAMS['max_ego_spawn_times'])
    parser.add_argument("--view_mode", type=str, default=DEFAULT_CARLA_PARAMS['view_mode'], choices=["top", "follow"])
    parser.add_argument("--traffic", type=str, default=DEFAULT_CARLA_PARAMS['traffic'], choices=["on", "off"])
    parser.add_argument("--lidar_max_range", type=float, default=DEFAULT_CARLA_PARAMS['lidar_max_range'])
    parser.add_argument("--max_nearby_vehicles", type=int, default=DEFAULT_CARLA_PARAMS['max_nearby_vehicles'])
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_id(ckpt_dir: str, model_id: Optional[int]) -> int:
    if model_id is not None:
        return model_id
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    actor_ids = []
    pattern = re.compile(r"actor_(\d+)\.pth$")
    for name in os.listdir(ckpt_dir):
        match = pattern.match(name)
        if match:
            actor_ids.append(int(match.group(1)))
    if not actor_ids:
        raise FileNotFoundError(f"No actor_*.pth checkpoints found in: {ckpt_dir}")
    return max(actor_ids)


def resolve_encoder_ckpt(ckpt_dir: str, model_id: int, encoder_ckpt: Optional[str]) -> Optional[str]:
    if encoder_ckpt and os.path.exists(encoder_ckpt):
        return encoder_ckpt
    candidate = os.path.join(ckpt_dir, f"encoder_rl_{model_id}.pth")
    if os.path.exists(candidate):
        return candidate
    fallback = os.path.join(ckpt_dir, "encoder_rl.pth")
    if os.path.exists(fallback):
        return fallback
    return encoder_ckpt


def build_env_params(args: argparse.Namespace) -> Dict[str, object]:
    return {
        'number_of_vehicles': args.number_of_vehicles,
        'number_of_walkers': args.number_of_walkers,
        'dt': args.dt,
        'ego_vehicle_filter': DEFAULT_CARLA_PARAMS['ego_vehicle_filter'],
        'surrounding_vehicle_spawned_randomly': DEFAULT_CARLA_PARAMS['surrounding_vehicle_spawned_randomly'],
        'port': args.port,
        'town': args.town,
        'use_current_world': args.use_current_world,
        'client_timeout': args.client_timeout,
        'max_time_episode': args.max_time_episode,
        'max_waypoints': args.max_waypoints,
        'visualize_waypoints': bool(args.visualize_waypoints),
        'desired_speed': args.desired_speed,
        'max_ego_spawn_times': args.max_ego_spawn_times,
        'view_mode': args.view_mode,
        'traffic': args.traffic,
        'lidar_max_range': args.lidar_max_range,
        'max_nearby_vehicles': args.max_nearby_vehicles,
    }


def load_offline_dataset(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(data_path, "r") as f:
        obs = f["observations"][:]
        act = f["actions"][:]
        next_obs = f["next_observations"][:]
        rew = f["rewards"][:]
        done = f["done"][:]
    return obs, act, next_obs, rew, done


def load_encoder(encoder_ckpt: str, device: torch.device) -> Tuple[StateEncoder, dict]:
    ckpt = torch.load(encoder_ckpt, map_location=device)
    config = ckpt.get("config", {})
    encoder = StateEncoder(
        state_dim=config.get("state_dim", 307),
        latent_dim=config.get("latent_dim", 64),
        hidden_dim=config.get("hidden_dim", 256),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    return encoder, config


def save_encoder_checkpoint(model: Diffusion_QL, save_dir: str, epoch: int, source_encoder_ckpt: Optional[str]) -> None:
    if model.encoder is None:
        return
    payload = {
        "encoder_state_dict": model.encoder.state_dict(),
        "config": model.encoder.get_config(),
        "source_encoder_ckpt": source_encoder_ckpt,
        "finetuned": True,
        "epoch": epoch,
        "ablation": "online_finetune",
    }
    torch.save(payload, os.path.join(save_dir, f"encoder_rl_{epoch}.pth"))
    torch.save(payload, os.path.join(save_dir, "encoder_rl.pth"))


def save_uncertainty_checkpoint(uncertainty_heads: Optional[UncertaintyHeadEnsemble], save_dir: str, epoch: int) -> None:
    if uncertainty_heads is None:
        return
    uncertainty_heads.save(os.path.join(save_dir, f"uncertainty_heads_{epoch}.pth"))
    uncertainty_heads.save(os.path.join(save_dir, "uncertainty_heads.pth"))


def save_online_history(save_dir: str, history: Dict[str, list]) -> None:
    payload = {key: np.asarray(values) for key, values in history.items()}
    np.savez(os.path.join(save_dir, "online_training_history.npz"), **payload)


def encode_with_module(encoder, obs_vec: np.ndarray, device: torch.device) -> np.ndarray:
    obs = np.asarray(obs_vec, dtype=np.float32)
    if encoder is None:
        return obs
    with torch.no_grad():
        tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        return encoder(tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)


def model_encode_state(model: Diffusion_QL, obs_vec: np.ndarray) -> torch.Tensor:
    state = torch.from_numpy(np.asarray(obs_vec, dtype=np.float32)).float().unsqueeze(0).to(model.device)
    with torch.no_grad():
        return model._encode_state(state, detach=True)


def sample_policy_action(model: Diffusion_QL, obs_vec: np.ndarray, exploration_noise: float) -> np.ndarray:
    action = np.asarray(model.sample_action(obs_vec), dtype=np.float32)
    if exploration_noise > 0:
        action = action + np.random.normal(0.0, exploration_noise, size=action.shape).astype(np.float32)
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def random_action() -> np.ndarray:
    throttle = np.random.uniform(0.0, 1.0)
    steer = np.random.uniform(-1.0, 1.0)
    brake = np.random.uniform(0.0, 1.0)
    return np.array([throttle, steer, brake], dtype=np.float32)


def build_reference_memory(
    reference_encoder,
    offline_buffer: NumpyReplayBuffer,
    device: torch.device,
    max_samples: int,
) -> np.ndarray:
    if len(offline_buffer) == 0 or max_samples <= 0:
        return np.zeros((0, offline_buffer.obs.shape[1] if len(offline_buffer) > 0 else 0), dtype=np.float32)

    sample_count = min(int(max_samples), len(offline_buffer))
    if sample_count == len(offline_buffer):
        indices = np.arange(len(offline_buffer))
    else:
        indices = np.random.choice(len(offline_buffer), size=sample_count, replace=False)

    embeddings = [encode_with_module(reference_encoder, offline_buffer.obs[idx], device) for idx in indices]
    return np.asarray(embeddings, dtype=np.float32)


def warmup_uncertainty_heads(
    model: Diffusion_QL,
    uncertainty_heads: Optional[UncertaintyHeadEnsemble],
    replay_buffer,
    updates: int,
    batch_size: int,
) -> list:
    if uncertainty_heads is None or updates <= 0:
        return []

    stats = []
    for _ in range(int(updates)):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            encoded_state = model._encode_state(state, detach=True)
            encoded_next_state = model._encode_state(next_state, detach=True)
            next_action = model.ema_model(encoded_next_state)
            target_q1, target_q2 = model.critic_target(encoded_next_state, next_action)
            target_q = reward + not_done * model.discount * torch.min(target_q1, target_q2)
        stats.append(uncertainty_heads.train_step(encoded_state, action, target_q.detach()))
    return stats


def select_active_action(
    model: Diffusion_QL,
    uncertainty_heads: UncertaintyHeadEnsemble,
    selector: ActiveSelector,
    obs_vec: np.ndarray,
    candidate_count: int,
    exploration_noise: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    candidate_actions = [sample_policy_action(model, obs_vec, exploration_noise) for _ in range(max(1, candidate_count))]
    action_tensor = torch.from_numpy(np.asarray(candidate_actions, dtype=np.float32)).float().to(model.device)
    state_tensor = model_encode_state(model, obs_vec).repeat(action_tensor.shape[0], 1)
    with torch.no_grad():
        scores = uncertainty_heads.score_actions(state_tensor, action_tensor)
        combined = selector.score_action(scores["q_mean"], scores["q_std"])
        best_idx = int(torch.argmax(combined).item())
    return candidate_actions[best_idx], {
        "q_mean": float(scores["q_mean"][best_idx].item()),
        "uncertainty": float(scores["q_std"][best_idx].item()),
        "score": float(combined[best_idx].item()),
    }


def train_uncertainty_heads(
    model: Diffusion_QL,
    uncertainty_heads: Optional[UncertaintyHeadEnsemble],
    replay_buffer,
    updates: int,
    batch_size: int,
) -> list:
    if uncertainty_heads is None or updates <= 0:
        return []

    stats = []
    for _ in range(int(updates)):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        with torch.no_grad():
            encoded_state = model._encode_state(state, detach=True)
            encoded_next_state = model._encode_state(next_state, detach=True)
            next_action = model.ema_model(encoded_next_state)
            target_q1, target_q2 = model.critic_target(encoded_next_state, next_action)
            target_q = reward + not_done * model.discount * torch.min(target_q1, target_q2)
        stats.append(uncertainty_heads.train_step(encoded_state, action, target_q.detach()))
    return stats


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    model_id = resolve_model_id(ckpt_dir, args.model_id)
    env_params = build_env_params(args)
    device = torch.device(args.device)

    print("\n=== [Online-Finetune] Starting ===")
    print(f"[0] seed={args.seed} | mode={args.online_mode}")
    print(f"[1] Loading offline dataset: {args.offline_data_path}")
    obs, act, next_obs, rew, done = load_offline_dataset(args.offline_data_path)
    offline_buffer = NumpyReplayBuffer(obs, act, next_obs, rew, done)
    offline_buffer.to(device)
    online_buffer = NumpyReplayBuffer(
        np.zeros((0, obs.shape[1]), dtype=np.float32),
        np.zeros((0, act.shape[1]), dtype=np.float32),
        np.zeros((0, next_obs.shape[1]), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )
    online_buffer.to(device)

    resolved_encoder_ckpt = resolve_encoder_ckpt(ckpt_dir, model_id, args.encoder_ckpt)
    encoder = None
    reference_encoder = None
    state_dim = obs.shape[1]
    if resolved_encoder_ckpt and os.path.exists(resolved_encoder_ckpt):
        encoder, encoder_config = load_encoder(resolved_encoder_ckpt, device)
        state_dim = encoder_config.get("latent_dim", 64)
        reference_encoder = copy.deepcopy(encoder).to(device)
        reference_encoder.eval()
        for param in reference_encoder.parameters():
            param.requires_grad = False
        if args.finetune_encoder:
            encoder.train()
        else:
            encoder.eval()
            for param in encoder.parameters():
                param.requires_grad = False
        print(f"[2] Encoder loaded: {resolved_encoder_ckpt}")
        print(f"    latent_dim={state_dim} | finetune={args.finetune_encoder}")
    else:
        print("[2] No encoder checkpoint found, online finetune uses raw 307-dim state.")

    model = Diffusion_QL(
        state_dim=state_dim,
        action_dim=act.shape[1],
        max_action=1.0,
        device=device,
        discount=args.discount,
        tau=args.tau,
        eta=args.eta,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        lr=args.lr,
        grad_norm=args.grad_norm,
        encoder=encoder,
        encoder_lr=args.encoder_lr,
        encoder_grad_norm=args.encoder_grad_norm,
        finetune_encoder=args.finetune_encoder,
    )
    model.load_model(ckpt_dir, id=model_id)
    print(f"[3] Loaded offline policy from {ckpt_dir} (id={model_id})")

    save_dir = os.path.join(CURRENT_DIR, "params_online_finetune")
    os.makedirs(save_dir, exist_ok=True)

    uncertainty_heads = None
    active_selector = None
    mixed_scheduler = None
    if args.online_mode == "active":
        uncertainty_heads = UncertaintyHeadEnsemble(
            state_dim=state_dim,
            action_dim=act.shape[1],
            num_heads=args.uncertainty_heads,
            hidden_dim=args.uncertainty_hidden_dim,
            lr=args.uncertainty_lr,
            bootstrap_prob=args.uncertainty_bootstrap_prob,
            device=args.device,
        )
        active_selector = ActiveSelector(
            lambda_uncertainty=args.active_lambda_uncertainty,
            lambda_novelty=args.active_lambda_novelty,
            retain_threshold=args.retain_threshold,
            max_memory_size=args.novelty_memory_size,
        )
        mixed_scheduler = MixedReplaySchedule(
            start_offline_ratio=args.mixed_schedule_start_offline_ratio,
            end_offline_ratio=args.mixed_schedule_end_offline_ratio,
            warmup_epochs=args.mixed_schedule_warmup_epochs,
            ramp_epochs=args.mixed_schedule_ramp_epochs,
        )
        reference_memory = build_reference_memory(
            reference_encoder,
            offline_buffer,
            device,
            args.reference_memory_size,
        )
        if reference_memory.size > 0:
            active_selector.update_reference_memory(reference_memory)
        warmup_stats = warmup_uncertainty_heads(
            model,
            uncertainty_heads,
            offline_buffer,
            args.uncertainty_warmup_updates,
            args.uncertainty_batch_size,
        )
        print(f"[4] Active mode enabled | reference_memory={active_selector.summarize()['reference_memory']}")
        if warmup_stats:
            warmup_mean = float(np.mean([item["loss"] for item in warmup_stats]))
            warmup_std = float(np.mean([item["q_std"] for item in warmup_stats]))
            print(f"    uncertainty warmup done | loss={warmup_mean:.4f} | q_std={warmup_std:.4f}")
    else:
        print("[4] Baseline online mode enabled (plain offline + online finetune).")

    env = gym.make('carla-v0', params=env_params)
    try:
        total_online_steps = 0
        previous_epoch_reward = None
        previous_reward_delta = 0.0
        previous_retention_rate = 0.0
        online_history = {
            "epoch": [],
            "reward": [],
            "cost": [],
            "steps": [],
            "collected": [],
            "retained": [],
            "retention_rate": [],
            "offline_ratio": [],
            "collision_rate": [],
            "offroad_rate": [],
            "bc_loss": [],
            "ql_loss": [],
            "critic_loss": [],
            "uncertainty_loss": [],
            "uncertainty_q_mean": [],
            "uncertainty_q_std": [],
            "novelty": [],
            "selection_score": [],
        }

        print("\n=== ONLINE FINETUNE START ===")
        print(f"  epochs={args.online_epochs} | episodes/epoch={args.episodes_per_epoch} | updates/epoch={args.updates_per_epoch}")
        print(f"  offline_ratio={args.offline_ratio:.2f} | warmup_random_steps={args.warmup_random_steps} | noise={args.exploration_noise}")
        sys.stdout.flush()

        for epoch in range(1, args.online_epochs + 1):
            epoch_reward = 0.0
            epoch_cost = 0.0
            epoch_steps = 0
            epoch_collected = 0
            epoch_retained = 0
            epoch_uncertainty_sum = 0.0
            epoch_novelty_sum = 0.0
            epoch_score_sum = 0.0
            epoch_collision_count = 0
            epoch_offroad_count = 0
            t0 = time.time()

            if args.online_mode == "active" and mixed_scheduler is not None:
                current_offline_ratio = mixed_scheduler.ratio_for_epoch(
                    epoch,
                    high_value_retention=previous_retention_rate,
                    reward_trend=previous_reward_delta,
                )
            else:
                current_offline_ratio = float(np.clip(args.offline_ratio, 0.0, 1.0))
            mixed_buffer = MixedReplayBuffer(offline_buffer, online_buffer, current_offline_ratio)

            for episode in range(args.episodes_per_epoch):
                obs_dict = env.reset()
                done_flag = False
                step = 0
                while not done_flag:
                    obs_vec = flatten_obs_dict(obs_dict)
                    if total_online_steps < args.warmup_random_steps:
                        action = random_action()
                        action_mode = "random"
                        action_stats = None
                    elif args.online_mode == "active" and uncertainty_heads is not None and active_selector is not None:
                        action, action_stats = select_active_action(
                            model,
                            uncertainty_heads,
                            active_selector,
                            obs_vec,
                            args.active_candidates,
                            args.exploration_noise,
                        )
                        action_mode = "active"
                    else:
                        action = sample_policy_action(model, obs_vec, args.exploration_noise)
                        action_mode = "policy"
                        action_stats = None

                    next_obs_dict, reward, cost, done_flag, info = env.step(action)
                    next_obs_vec = flatten_obs_dict(next_obs_dict)
                    epoch_collision_count += int(bool(info.get("is_collision", False)))
                    epoch_offroad_count += int(bool(info.get("is_off_road", False)))

                    retained = True
                    novelty = 0.0
                    uncertainty_value = 0.0
                    selection_score = 0.0
                    if args.online_mode == "active" and active_selector is not None:
                        embedding = encode_with_module(reference_encoder, obs_vec, device)
                        novelty = active_selector.novelty_score(embedding)
                        if action_stats is not None:
                            uncertainty_value = float(action_stats["uncertainty"])
                            selection_score = float(action_stats["score"])
                        retained = active_selector.should_retain(uncertainty_value, novelty)
                        if epoch_retained < args.active_seed_buffer:
                            retained = True
                        if retained:
                            active_selector.update_online_memory(embedding)
                        epoch_uncertainty_sum += uncertainty_value
                        epoch_novelty_sum += novelty
                        epoch_score_sum += selection_score
                    else:
                        if action_stats is not None:
                            uncertainty_value = float(action_stats["uncertainty"])
                            selection_score = float(action_stats["score"])

                    if retained:
                        online_buffer.append(obs_vec, action, next_obs_vec, reward, done_flag)
                        epoch_retained += 1

                    obs_dict = next_obs_dict
                    total_online_steps += 1
                    epoch_collected += 1
                    epoch_steps += 1
                    epoch_reward += float(reward)
                    epoch_cost += float(cost)
                    step += 1

                    if step % 50 == 0:
                        print(
                            f"[Collect] epoch={epoch} episode={episode} step={step} mode={action_mode} "
                            f"reward={epoch_reward:.2f} online_buffer={len(online_buffer)} retained={epoch_retained}"
                        )
                        sys.stdout.flush()

                    if args.max_steps_per_episode > 0 and step >= args.max_steps_per_episode:
                        done_flag = True

            uncertainty_metrics = train_uncertainty_heads(
                model,
                uncertainty_heads if args.online_mode == "active" else None,
                mixed_buffer,
                args.uncertainty_updates_per_epoch,
                args.uncertainty_batch_size,
            )

            metrics = model.train(
                replay_buffer=mixed_buffer,
                iterations=args.updates_per_epoch,
                batch_size=args.batch_size,
            )
            mean_bc = float(np.mean(metrics["bc_loss"]))
            mean_ql = float(np.mean(metrics["ql_loss"]))
            mean_cr = float(np.mean(metrics["critic_loss"]))
            mean_uncertainty_loss = float(np.mean([item["loss"] for item in uncertainty_metrics])) if uncertainty_metrics else 0.0
            mean_uncertainty_q = float(np.mean([item["q_mean"] for item in uncertainty_metrics])) if uncertainty_metrics else 0.0
            mean_uncertainty_std = float(np.mean([item["q_std"] for item in uncertainty_metrics])) if uncertainty_metrics else 0.0
            retention_rate = epoch_retained / max(1, epoch_collected)
            collision_rate = epoch_collision_count / max(1, epoch_collected)
            offroad_rate = epoch_offroad_count / max(1, epoch_collected)
            elapsed = time.time() - t0

            should_save = (epoch % args.save_every == 0) or (epoch == args.online_epochs)
            if should_save:
                model.save_model(save_dir, id=epoch)
                save_encoder_checkpoint(model, save_dir, epoch, resolved_encoder_ckpt)
                save_uncertainty_checkpoint(uncertainty_heads if args.online_mode == "active" else None, save_dir, epoch)
                history_snapshot = {key: values[:] for key, values in online_history.items()}
                history_snapshot["epoch"].append(epoch)
                history_snapshot["reward"].append(epoch_reward)
                history_snapshot["cost"].append(epoch_cost)
                history_snapshot["steps"].append(epoch_steps)
                history_snapshot["collected"].append(epoch_collected)
                history_snapshot["retained"].append(epoch_retained)
                history_snapshot["retention_rate"].append(retention_rate)
                history_snapshot["offline_ratio"].append(current_offline_ratio)
                history_snapshot["collision_rate"].append(collision_rate)
                history_snapshot["offroad_rate"].append(offroad_rate)
                history_snapshot["bc_loss"].append(mean_bc)
                history_snapshot["ql_loss"].append(mean_ql)
                history_snapshot["critic_loss"].append(mean_cr)
                history_snapshot["uncertainty_loss"].append(mean_uncertainty_loss)
                history_snapshot["uncertainty_q_mean"].append(mean_uncertainty_q)
                history_snapshot["uncertainty_q_std"].append(mean_uncertainty_std)
                history_snapshot["novelty"].append(epoch_novelty_sum / max(1, epoch_collected))
                history_snapshot["selection_score"].append(epoch_score_sum / max(1, epoch_collected))
                save_online_history(save_dir, history_snapshot)

            online_history["epoch"].append(epoch)
            online_history["reward"].append(epoch_reward)
            online_history["cost"].append(epoch_cost)
            online_history["steps"].append(epoch_steps)
            online_history["collected"].append(epoch_collected)
            online_history["retained"].append(epoch_retained)
            online_history["retention_rate"].append(retention_rate)
            online_history["offline_ratio"].append(current_offline_ratio)
            online_history["collision_rate"].append(collision_rate)
            online_history["offroad_rate"].append(offroad_rate)
            online_history["bc_loss"].append(mean_bc)
            online_history["ql_loss"].append(mean_ql)
            online_history["critic_loss"].append(mean_cr)
            online_history["uncertainty_loss"].append(mean_uncertainty_loss)
            online_history["uncertainty_q_mean"].append(mean_uncertainty_q)
            online_history["uncertainty_q_std"].append(mean_uncertainty_std)
            online_history["novelty"].append(epoch_novelty_sum / max(1, epoch_collected))
            online_history["selection_score"].append(epoch_score_sum / max(1, epoch_collected))

            if previous_epoch_reward is None:
                previous_reward_delta = 0.0
            else:
                previous_reward_delta = epoch_reward - previous_epoch_reward
            previous_epoch_reward = epoch_reward
            previous_retention_rate = retention_rate

            print(
                f"[Epoch {epoch}] mode={args.online_mode} ratio={current_offline_ratio:.2f} steps={epoch_steps} "
                f"collected={epoch_collected} retained={epoch_retained} retain_rate={retention_rate:.2f} "
                f"coll={collision_rate:.3f} offroad={offroad_rate:.3f} "
                f"reward={epoch_reward:.2f} cost={epoch_cost:.2f} bc={mean_bc:.4f} ql={mean_ql:.4f} cr={mean_cr:.4f} "
                f"unc={mean_uncertainty_loss:.4f} qstd={mean_uncertainty_std:.4f} time={elapsed:.1f}s{' [SAVED]' if should_save else ''}"
            )
            sys.stdout.flush()

        save_online_history(save_dir, online_history)
        print("\n=== ONLINE FINETUNE COMPLETE ===")
        print(f"Saved checkpoints to: {save_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

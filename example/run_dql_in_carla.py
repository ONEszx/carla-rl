# -*- coding: utf-8 -*-
"""
Run a trained diffusion policy in EasyCarla.

Supports:
- BC checkpoints (`bc`)
- OfflineRL Diffusion QL checkpoints (`baseline`)
- Encoder ablation checkpoints (`encoder_only`)
- Full method checkpoints (`full`)
"""

import argparse
import os
import re
import sys
from typing import Dict, Optional, Tuple

import gym
import easycarla
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.bc_diffusion import Diffusion_BC
from agents.ql_diffusion import Diffusion_QL
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


def convert_obs_dict_to_vector(obs_dict):
    return np.concatenate([
        obs_dict['ego_state'],
        obs_dict['lane_info'],
        obs_dict['lidar'],
        obs_dict['nearby_vehicles'],
        obs_dict['waypoints']
    ]).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained diffusion policy in EasyCarla.")
    parser.add_argument("--ckpt_dir", type=str, default=os.path.join(CURRENT_DIR, "params_dql_test"))
    parser.add_argument("--model_id", type=int, default=None, help="Checkpoint id to load. Defaults to the latest actor_*.pth in ckpt_dir.")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help="Optional per-episode cap. 0 means use env termination only.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

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

    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "bc", "dql"],
        help="Checkpoint type: auto-detect, force BC, or force Diffusion QL.",
    )
    parser.add_argument(
        "--encoder_ckpt", type=str, default=None,
        help="Path to frozen encoder checkpoint (.pth). If provided, raw 307-dim states are encoded to latent before policy.",
    )
    parser.add_argument(
        "--ablation", type=str, default=None,
        choices=["bc", "baseline", "encoder_only", "full"],
        help="Ablation mode for latent encoding / policy type. bc=Behavior Cloning, baseline=OfflineRL, encoder_only=frozen encoder ablation, full=encoder+priority.",
    )
    return parser.parse_args()


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


def load_ablation_config(ckpt_dir: str, ablation: Optional[str] = None) -> Dict[str, str]:
    candidates = []
    if ablation:
        candidates.append(os.path.join(ckpt_dir, f"ablation_config_{ablation}.txt"))
    for name in sorted(os.listdir(ckpt_dir)) if os.path.isdir(ckpt_dir) else []:
        if name.startswith("ablation_config_") and name.endswith(".txt"):
            candidates.append(os.path.join(ckpt_dir, name))

    seen = set()
    for path in candidates:
        if path in seen or not os.path.exists(path):
            continue
        seen.add(path)
        config = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
        if config:
            config["_config_path"] = path
            return config
    return {}


def str_to_bool(value: Optional[str]) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"}


def resolve_method_and_ablation(
    ckpt_dir: str,
    cli_method: str,
    cli_ablation: Optional[str],
) -> Tuple[str, Optional[str], Dict[str, str]]:
    config = load_ablation_config(ckpt_dir, cli_ablation)
    dir_name = os.path.basename(os.path.normpath(ckpt_dir)).lower()

    ablation = cli_ablation or config.get("ablation")
    if ablation is None:
        if "encoder_only" in dir_name:
            ablation = "encoder_only"
        elif "full" in dir_name:
            ablation = "full"
        elif "baseline" in dir_name:
            ablation = "baseline"
        elif "bc" in dir_name:
            ablation = "bc"

    if cli_method != "auto":
        method_type = cli_method
    else:
        method_type = config.get("method_type")
        if not method_type:
            if ablation == "bc" or "params_bc" in dir_name or dir_name.endswith("bc"):
                method_type = "bc"
            else:
                method_type = "dql"

    return method_type, ablation, config


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


def main() -> None:
    args = parse_args()
    env_params = build_env_params(args)
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    model_id = resolve_model_id(ckpt_dir, args.model_id)
    method_type, resolved_ablation, ablation_config = resolve_method_and_ablation(
        ckpt_dir, args.method, args.ablation
    )
    resolved_encoder_ckpt = resolve_encoder_ckpt(ckpt_dir, model_id, args.encoder_ckpt)
    use_encoder = str_to_bool(ablation_config.get("use_encoder"))
    if resolved_ablation in ("encoder_only", "full"):
        use_encoder = True
    if method_type == "bc":
        use_encoder = False

    env = gym.make('carla-v0', params=env_params)
    try:
        device = torch.device(args.device)

        encoder = None
        state_dim = 307

        print(f"[CKPT] dir={ckpt_dir}")
        print(f"[CKPT] model_id={model_id} | method={method_type} | ablation={resolved_ablation}")
        if ablation_config.get("_config_path"):
            print(f"[CKPT] config={ablation_config['_config_path']}")

        if use_encoder and resolved_encoder_ckpt and os.path.exists(resolved_encoder_ckpt):
            print(f"\n[ENCODER] Loading from {resolved_encoder_ckpt}")
            ckpt = torch.load(resolved_encoder_ckpt, map_location=device)
            enc_config = ckpt.get("config", {})
            latent_dim = enc_config.get("latent_dim", 64)
            state_dim = latent_dim

            encoder = StateEncoder(
                state_dim=enc_config.get("state_dim", 307),
                latent_dim=latent_dim,
                hidden_dim=enc_config.get("hidden_dim", 256),
            ).to(device)
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            print(f"[ENCODER] Loaded — encoding raw 307-dim -> {latent_dim}-dim latent")
        elif use_encoder:
            print(f"[WARN] Encoder checkpoint not found at {resolved_encoder_ckpt}, falling back to raw 307-dim")
        else:
            print(f"[POLICY] Running with raw 307-dim state (method={method_type})")

        action_dim = 3
        max_action = 1.0

        if method_type == "bc":
            model = Diffusion_BC(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                device=device,
                discount=0.99,
                tau=0.005,
                beta_schedule='vp',
                n_timesteps=5,
            )
        else:
            model = Diffusion_QL(
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
                device=device,
                discount=0.99,
                tau=0.005,
                eta=0.01,
                beta_schedule='vp',
                n_timesteps=5
            )
        model.load_model(ckpt_dir, id=model_id)
        print(f"Successfully loaded model ID {model_id} from {ckpt_dir}")
        print(f"  state_dim={state_dim} | encoder={encoder is not None} | method={method_type}")

        all_rewards = []
        all_costs = []
        all_steps = []
        all_collisions = []
        all_offroads = []

        for episode in range(args.num_episodes):
            obs = env.reset()
            done = False
            step = 0
            episode_reward = 0.0
            episode_cost = 0.0
            episode_collision = 0
            episode_offroad = 0

            while not done:
                obs_raw = convert_obs_dict_to_vector(obs)

                if encoder is not None:
                    with torch.no_grad():
                        obs_vec = encoder(
                            torch.from_numpy(obs_raw).float().unsqueeze(0).to(device)
                        ).squeeze(0).cpu().numpy()
                else:
                    obs_vec = obs_raw

                action = model.sample_action(obs_vec)

                try:
                    next_obs, reward, cost, done, info = env.step(action)
                except Exception as e:
                    print(f"[Error] Carla step failed in episode {episode}: {e}")
                    break

                obs = next_obs
                episode_reward += float(reward)
                episode_cost += float(cost)
                episode_collision += int(bool(info.get("is_collision", False)))
                episode_offroad += int(bool(info.get("is_off_road", False)))
                step += 1

                if step % 50 == 0:
                    print(
                        f"[RUN] episode={episode} step={step} reward={episode_reward:.2f} "
                        f"cost={episode_cost:.2f} coll={episode_collision} offroad={episode_offroad}"
                    )

                if args.max_steps > 0 and step >= args.max_steps:
                    done = True

            collision_rate = episode_collision / max(1, step)
            offroad_rate = episode_offroad / max(1, step)
            all_rewards.append(episode_reward)
            all_costs.append(episode_cost)
            all_steps.append(step)
            all_collisions.append(collision_rate)
            all_offroads.append(offroad_rate)
            print(
                f"[EPISODE] id={episode} reward={episode_reward:.2f} cost={episode_cost:.2f} "
                f"steps={step} collision_rate={collision_rate:.3f} offroad_rate={offroad_rate:.3f}"
            )

        if all_rewards:
            print("\n=== ROLLOUT SUMMARY ===")
            print(f"avg_reward={np.mean(all_rewards):.2f}")
            print(f"avg_cost={np.mean(all_costs):.2f}")
            print(f"avg_steps={np.mean(all_steps):.2f}")
            print(f"collision_rate={np.mean(all_collisions):.3f}")
            print(f"offroad_rate={np.mean(all_offroads):.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

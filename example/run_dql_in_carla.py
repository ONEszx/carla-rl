# -*- coding: utf-8 -*-
"""
Run a trained Diffusion_QL model in EasyCarla.
"""

import argparse
import os
import re
import sys
from typing import Dict, Optional

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

from agents.ql_diffusion import Diffusion_QL

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
    parser = argparse.ArgumentParser(description="Run a trained Diffusion_QL policy in EasyCarla.")
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

    env = gym.make('carla-v0', params=env_params)
    try:
        state_dim = 307
        action_dim = 3
        max_action = 1.0
        device = torch.device(args.device)

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

        for episode in range(args.num_episodes):
            obs = env.reset()
            done = False
            step = 0
            episode_reward = 0.0
            episode_cost = 0.0

            while not done:
                obs_vec = convert_obs_dict_to_vector(obs)
                action = model.sample_action(obs_vec)

                try:
                    next_obs, reward, cost, done, _ = env.step(action)
                except Exception as e:
                    print(f"[Error] Carla step failed in episode {episode}: {e}")
                    break

                obs = next_obs
                episode_reward += float(reward)
                episode_cost += float(cost)
                step += 1

                if step % 50 == 0:
                    print(f"[RUN] episode={episode} step={step} reward={episode_reward:.2f} cost={episode_cost:.2f}")

                if args.max_steps > 0 and step >= args.max_steps:
                    done = True

            print(f"[EPISODE] id={episode} reward={episode_reward:.2f} cost={episode_cost:.2f} steps={step}")
    finally:
        env.close()


if __name__ == "__main__":
    main()

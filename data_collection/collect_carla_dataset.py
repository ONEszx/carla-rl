# -*- coding: utf-8 -*-
"""
Collect EasyCarla trajectories and export them to the HDF5 dataset format
used by the current offline Diffusion_QL training code.
"""

import argparse
import os
import random
import sys
from typing import Dict, Optional

import gym
import easycarla
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
EXAMPLE_DIR = os.path.join(PROJECT_ROOT, "example")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, EXAMPLE_DIR)

from agents.ql_diffusion import Diffusion_QL
from data_collection.carla_launcher import DEFAULT_CARLA_EXE, launch_carla_server, stop_carla_server
from data_collection.collector_utils import (
    ACTION_DIM,
    STATE_DIM,
    TransitionBuffer,
    flatten_obs_dict,
    save_buffer_to_hdf5,
    summarize_buffer,
)

DEFAULT_CARLA_PARAMS = {
    'number_of_vehicles': 50,
    'number_of_walkers': 0,
    'dt': 0.1,
    'ego_vehicle_filter': 'vehicle.tesla.model3',
    'surrounding_vehicle_spawned_randomly': True,
    'port': 2000,
    'town': 'Town03',
    'max_time_episode': 1000,
    'max_waypoints': 12,
    'visualize_waypoints': False,
    'desired_speed': 8,
    'max_ego_spawn_times': 200,
    'view_mode': 'top',
    'traffic': 'off',
    'lidar_max_range': 50.0,
    'max_nearby_vehicles': 5,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect EasyCarla data and save it as HDF5.")
    parser.add_argument("--output_path", type=str, default=os.path.join(PROJECT_ROOT, "data", "easycarla_online_dataset.hdf5"))
    parser.add_argument("--mode", type=str, default="autopilot", choices=["random", "autopilot", "policy", "mixed"])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=0, help="Stop after this many transitions if > 0.")
    parser.add_argument("--save_every", type=int, default=0, help="If > 0, periodically overwrite the output file every N collected steps.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--launch_carla", action="store_true", help="Launch a local CARLA server before collecting data.")
    parser.add_argument("--close_carla_on_exit", action="store_true", help="Close the CARLA process started by this script when collection ends.")
    parser.add_argument("--carla_exe", type=str, default=DEFAULT_CARLA_EXE)
    parser.add_argument("--carla_wait_seconds", type=float, default=45.0)
    parser.add_argument("--carla_quality_level", type=str, default="Low")
    parser.add_argument("--carla_windowed", type=int, default=1)

    parser.add_argument("--policy_ckpt_dir", type=str, default=os.path.join(EXAMPLE_DIR, "params_dql"))
    parser.add_argument("--policy_model_id", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--random_throttle_max", type=float, default=1.0)
    parser.add_argument("--random_steer_abs", type=float, default=0.6)
    parser.add_argument("--random_brake_max", type=float, default=0.3)
    parser.add_argument("--mixed_autopilot_prob", type=float, default=0.5)
    parser.add_argument("--mixed_policy_prob", type=float, default=0.3)

    parser.add_argument("--number_of_vehicles", type=int, default=DEFAULT_CARLA_PARAMS['number_of_vehicles'])
    parser.add_argument("--number_of_walkers", type=int, default=DEFAULT_CARLA_PARAMS['number_of_walkers'])
    parser.add_argument("--dt", type=float, default=DEFAULT_CARLA_PARAMS['dt'])
    parser.add_argument("--ego_vehicle_filter", type=str, default=DEFAULT_CARLA_PARAMS['ego_vehicle_filter'])
    parser.add_argument("--surrounding_vehicle_spawned_randomly", type=int, default=1)
    parser.add_argument("--port", type=int, default=DEFAULT_CARLA_PARAMS['port'])
    parser.add_argument("--client_timeout", type=float, default=30.0)
    parser.add_argument("--town", type=str, default=DEFAULT_CARLA_PARAMS['town'])
    parser.add_argument("--use_current_world", action="store_true", help="Use the already loaded CARLA world instead of forcing a town reload.")
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


def build_env_params(args: argparse.Namespace) -> Dict[str, object]:
    return {
        'number_of_vehicles': args.number_of_vehicles,
        'number_of_walkers': args.number_of_walkers,
        'dt': args.dt,
        'ego_vehicle_filter': args.ego_vehicle_filter,
        'surrounding_vehicle_spawned_randomly': bool(args.surrounding_vehicle_spawned_randomly),
        'port': args.port,
        'client_timeout': args.client_timeout,
        'town': args.town,
        'use_current_world': args.use_current_world,
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


def load_policy_if_needed(args: argparse.Namespace, device: torch.device) -> Optional[Diffusion_QL]:
    if args.mode not in ("policy", "mixed"):
        return None

    model = Diffusion_QL(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        max_action=1.0,
        device=device,
        discount=0.99,
        tau=0.005,
        eta=0.01,
        beta_schedule='vp',
        n_timesteps=5,
    )
    model.load_model(args.policy_ckpt_dir, id=args.policy_model_id)
    print(f"[INFO] Loaded policy from {args.policy_ckpt_dir} (id={args.policy_model_id})")
    return model


def get_random_action(args: argparse.Namespace) -> np.ndarray:
    throttle = random.uniform(0.0, args.random_throttle_max)
    steer = random.uniform(-args.random_steer_abs, args.random_steer_abs)
    brake = random.uniform(0.0, args.random_brake_max)
    return np.array([throttle, steer, brake], dtype=np.float32)


def get_autopilot_action(env) -> np.ndarray:
    env.ego.set_autopilot(True)
    control = env.ego.get_control()
    return np.array([control.throttle, control.steer, control.brake], dtype=np.float32)


def get_policy_action(env, obs_vec: np.ndarray, policy: Diffusion_QL) -> np.ndarray:
    env.ego.set_autopilot(False)
    return np.asarray(policy.sample_action(obs_vec), dtype=np.float32)


def choose_action(mode: str, env, obs_vec: np.ndarray, args: argparse.Namespace, policy: Optional[Diffusion_QL]):
    if mode == "random":
        env.ego.set_autopilot(False)
        return get_random_action(args), "random"

    if mode == "autopilot":
        return get_autopilot_action(env), "autopilot"

    if mode == "policy":
        if policy is None:
            raise ValueError("Policy mode requires a loaded policy model.")
        return get_policy_action(env, obs_vec, policy), "policy"

    p = random.random()
    autopilot_prob = args.mixed_autopilot_prob
    policy_prob = args.mixed_policy_prob
    if p < autopilot_prob:
        return get_autopilot_action(env), "autopilot"
    if p < autopilot_prob + policy_prob:
        if policy is None:
            raise ValueError("Mixed mode with policy probability requires a loaded policy model.")
        return get_policy_action(env, obs_vec, policy), "policy"
    env.ego.set_autopilot(False)
    return get_random_action(args), "random"


def maybe_periodic_save(buffer: TransitionBuffer, output_path: str, save_every: int) -> None:
    if save_every <= 0:
        return
    if len(buffer) == 0 or len(buffer) % save_every != 0:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_buffer_to_hdf5(buffer, output_path)
    print(f"[INFO] Periodic save: {len(buffer)} transitions -> {output_path}")


def validate_mixed_probabilities(args: argparse.Namespace) -> None:
    if args.mode != "mixed":
        return

    if args.mixed_autopilot_prob < 0 or args.mixed_policy_prob < 0:
        raise ValueError("Mixed mode probabilities must be non-negative.")

    if args.mixed_autopilot_prob + args.mixed_policy_prob > 1.0:
        raise ValueError("mixed_autopilot_prob + mixed_policy_prob must be <= 1.0.")


def main() -> None:
    args = parse_args()
    validate_mixed_probabilities(args)
    set_seed(args.seed)

    device = torch.device(args.device)
    env_params = build_env_params(args)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    carla_process = None
    env = None
    try:
        if args.launch_carla:
            print(f"[INFO] Launching CARLA from: {args.carla_exe}")
            carla_process = launch_carla_server(
                carla_exe=args.carla_exe,
                port=args.port,
                wait_seconds=args.carla_wait_seconds,
                quality_level=args.carla_quality_level,
                windowed=bool(args.carla_windowed),
            )
            print(f"[INFO] CARLA is ready on port {args.port}")

        env = gym.make('carla-v0', params=env_params)
        policy = load_policy_if_needed(args, device)
        buffer = TransitionBuffer()

        total_steps = 0
        completed_episodes = 0

        try:
            for episode_id in range(args.num_episodes):
                obs = env.reset()
                done = False
                episode_reward = 0.0
                episode_cost = 0.0
                episode_steps = 0

                while not done:
                    obs_vec = flatten_obs_dict(obs)
                    action, source_mode = choose_action(args.mode, env, obs_vec, args, policy)

                    try:
                        next_obs, reward, cost, done, _ = env.step(action)
                    except Exception as e:
                        print(f"[WARN] Carla step failed in episode {episode_id}: {e}")
                        done = True
                        break

                    next_obs_vec = flatten_obs_dict(next_obs)
                    buffer.add(
                        obs=obs_vec,
                        action=action,
                        next_obs=next_obs_vec,
                        reward=reward,
                        done=done,
                        cost=cost,
                        episode_id=episode_id,
                        timestep=episode_steps,
                        source_mode=source_mode,
                    )

                    obs = next_obs
                    episode_reward += float(reward)
                    episode_cost += float(cost)
                    episode_steps += 1
                    total_steps += 1

                    if total_steps % 50 == 0:
                        print(
                            f"[COLLECT] total_steps={total_steps} episodes={completed_episodes} "
                            f"buffer={len(buffer)} last_mode={source_mode}"
                        )

                    maybe_periodic_save(buffer, args.output_path, args.save_every)

                    if args.num_steps > 0 and total_steps >= args.num_steps:
                        done = True
                        break

                completed_episodes += 1
                print(
                    f"[EPISODE] id={episode_id} steps={episode_steps} reward={episode_reward:.2f} "
                    f"cost={episode_cost:.2f} collected={len(buffer)}"
                )

                if args.num_steps > 0 and total_steps >= args.num_steps:
                    break

            shapes = save_buffer_to_hdf5(buffer, args.output_path)
            summary = summarize_buffer(buffer)
            print(f"[DONE] Saved dataset to {args.output_path}")
            print(f"[DONE] Shapes: {shapes}")
            print(f"[DONE] Summary: {summary}")
        finally:
            if env is not None:
                env.close()
    finally:
        if args.launch_carla and args.close_carla_on_exit:
            stop_carla_server(carla_process)


if __name__ == "__main__":
    main()

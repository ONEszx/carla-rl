# -*- coding: utf-8 -*-
"""
Batch-evaluate offline checkpoints in EasyCarla-RL.

This script supports both layouts:
    example/params_<variant>/actor_*.pth
    example/params_<variant>/seed_0/actor_*.pth

For multi-seed runs, it evaluates matching checkpoints across seeds, aggregates
mean ± std for each metric, and draws all selected variants on one figure.
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_SCRIPT = os.path.join(CURRENT_DIR, "run_dql_in_carla.py")
DEFAULT_VARIANTS = ["baseline", "encoder_only", "full"]
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "offline_actor_sweep")
SEED_DIR_PATTERN = re.compile(r"seed_(\d+)$")

SUMMARY_PATTERNS = {
    "avg_reward": re.compile(r"avg_reward=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "avg_cost": re.compile(r"avg_cost=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "avg_steps": re.compile(r"avg_steps=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "collision_rate": re.compile(r"collision_rate=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
    "offroad_rate": re.compile(r"offroad_rate=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"),
}
METRIC_KEYS = ["avg_reward", "avg_cost", "avg_steps", "collision_rate", "offroad_rate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep CARLA rollouts over saved actor checkpoints and aggregate mean±std across seeds."
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=["baseline", "encoder_only", "full", "bc"],
        help="Which checkpoint folders to evaluate.",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default=CURRENT_DIR,
        help="Root folder that contains params_<variant>/ directories.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Explicit seed ids, e.g. --seeds 0 1 2. If omitted, auto-detect seed_* dirs.",
    )
    parser.add_argument(
        "--model_ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit checkpoint ids to evaluate for every variant.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of CARLA episodes per checkpoint.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=300,
        help="Per-episode step cap passed to the rollout script.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device passed to the rollout script.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory used to save CSV/NPZ/PNG outputs.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Optional CARLA port override.",
    )
    parser.add_argument(
        "--town",
        type=str,
        default=None,
        help="Optional CARLA town override.",
    )
    parser.add_argument(
        "--use_current_world",
        action="store_true",
        help="Pass --use_current_world to the rollout script.",
    )
    parser.add_argument(
        "--traffic",
        type=str,
        choices=["on", "off"],
        default=None,
        help="Optional traffic mode override.",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip CARLA rollouts and redraw plots from existing aggregate CSV.",
    )
    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_variant_base_dir(ckpt_root: str, variant: str) -> str:
    candidate = os.path.join(ckpt_root, f"params_{variant}")
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(variant):
        return os.path.abspath(variant)
    return candidate


def list_actor_ids(ckpt_dir: str) -> List[int]:
    if not os.path.isdir(ckpt_dir):
        return []

    actor_ids = []
    pattern = re.compile(r"actor_(\d+)\.pth$")
    for name in os.listdir(ckpt_dir):
        match = pattern.match(name)
        if match:
            actor_ids.append(int(match.group(1)))
    return sorted(set(actor_ids))


def discover_seed_dirs(base_dir: str, requested_seeds: Optional[List[int]]) -> List[Tuple[str, str]]:
    seed_dirs = []

    if requested_seeds is not None:
        for seed in requested_seeds:
            candidate = os.path.join(base_dir, f"seed_{seed}")
            if os.path.isdir(candidate):
                seed_dirs.append((str(seed), candidate))
            else:
                print(f"[WARN] missing seed dir: {candidate}")
        if seed_dirs:
            return seed_dirs
        if list_actor_ids(base_dir):
            print(f"[WARN] no seed_* dirs found under {base_dir}; falling back to base directory")
            return [("base", base_dir)]
        return []

    if not os.path.isdir(base_dir):
        return []

    auto_seed_dirs = []
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        match = SEED_DIR_PATTERN.match(name)
        if os.path.isdir(path) and match:
            auto_seed_dirs.append((match.group(1), path))

    if auto_seed_dirs:
        return auto_seed_dirs

    return [("base", base_dir)]


def resolve_model_ids(seed_dirs: List[Tuple[str, str]], explicit_model_ids: Optional[List[int]]) -> List[int]:
    if explicit_model_ids is not None:
        return sorted(set(explicit_model_ids))

    actor_sets = []
    for _, ckpt_dir in seed_dirs:
        actor_ids = list_actor_ids(ckpt_dir)
        if actor_ids:
            actor_sets.append(set(actor_ids))

    if not actor_sets:
        return []

    common_ids = set.intersection(*actor_sets)
    return sorted(common_ids)


def build_command(ckpt_dir: str, model_id: int, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        RUN_SCRIPT,
        "--ckpt_dir",
        ckpt_dir,
        "--model_id",
        str(model_id),
        "--num_episodes",
        str(args.episodes),
        "--max_steps",
        str(args.max_steps),
        "--device",
        args.device,
    ]

    if args.port is not None:
        cmd.extend(["--port", str(args.port)])
    if args.town is not None:
        cmd.extend(["--town", args.town])
    if args.use_current_world:
        cmd.append("--use_current_world")
    if args.traffic is not None:
        cmd.extend(["--traffic", args.traffic])

    return cmd


def stream_command(cmd: List[str]) -> Tuple[int, str]:
    print(f"[CMD] {' '.join(cmd)}")
    sys.stdout.flush()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    captured = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        captured.append(line)
    return proc.wait(), "".join(captured)


def extract_metrics(text: str) -> Dict[str, float]:
    metrics = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = pattern.search(text)
        metrics[key] = float(match.group(1)) if match else float("nan")
    return metrics


def save_raw_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    fieldnames = [
        "variant",
        "seed",
        "ckpt_dir",
        "model_id",
        "status",
        "elapsed_s",
        "avg_reward",
        "avg_cost",
        "avg_steps",
        "collision_rate",
        "offroad_rate",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[SAVED] {path}")


def save_raw_npz(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return

    payload = {
        "variant": np.array([str(row["variant"]) for row in rows]),
        "seed": np.array([str(row["seed"]) for row in rows]),
        "ckpt_dir": np.array([str(row["ckpt_dir"]) for row in rows]),
        "model_id": np.array([int(row["model_id"]) for row in rows], dtype=np.int32),
        "status": np.array([str(row["status"]) for row in rows]),
        "elapsed_s": np.array([float(row["elapsed_s"]) for row in rows], dtype=np.float32),
    }
    for metric_key in METRIC_KEYS:
        payload[metric_key] = np.array([float(row[metric_key]) for row in rows], dtype=np.float32)
    np.savez(path, **payload)
    print(f"[SAVED] {path}")


def aggregate_rows(rows: List[Dict[str, object]], variants: List[str]) -> Dict[str, Dict[int, Dict[str, float]]]:
    aggregated = {}

    for variant in variants:
        variant_rows = [row for row in rows if row["variant"] == variant and row["status"] == "OK"]
        model_ids = sorted(set(int(row["model_id"]) for row in variant_rows))
        variant_summary = {}

        for model_id in model_ids:
            checkpoint_rows = [row for row in variant_rows if int(row["model_id"]) == model_id]
            checkpoint_summary = {
                "num_seeds": len(checkpoint_rows),
            }
            for metric_key in METRIC_KEYS:
                values = np.array([
                    float(row[metric_key])
                    for row in checkpoint_rows
                    if np.isfinite(float(row[metric_key]))
                ], dtype=np.float32)
                if values.size == 0:
                    checkpoint_summary[f"{metric_key}_mean"] = float("nan")
                    checkpoint_summary[f"{metric_key}_std"] = float("nan")
                else:
                    checkpoint_summary[f"{metric_key}_mean"] = float(values.mean())
                    checkpoint_summary[f"{metric_key}_std"] = float(values.std())
            variant_summary[model_id] = checkpoint_summary

        aggregated[variant] = variant_summary

    return aggregated


def save_aggregate_csv(path: str, aggregated: Dict[str, Dict[int, Dict[str, float]]], variants: List[str]) -> None:
    rows = []
    for variant in variants:
        for model_id in sorted(aggregated.get(variant, {}).keys()):
            summary = aggregated[variant][model_id]
            row = {
                "variant": variant,
                "model_id": model_id,
                "num_seeds": summary["num_seeds"],
            }
            for metric_key in METRIC_KEYS:
                row[f"{metric_key}_mean"] = summary[f"{metric_key}_mean"]
                row[f"{metric_key}_std"] = summary[f"{metric_key}_std"]
            rows.append(row)

    if not rows:
        return

    fieldnames = ["variant", "model_id", "num_seeds"]
    for metric_key in METRIC_KEYS:
        fieldnames.extend([f"{metric_key}_mean", f"{metric_key}_std"])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[SAVED] {path}")


def save_aggregate_npz(path: str, aggregated: Dict[str, Dict[int, Dict[str, float]]], variants: List[str]) -> None:
    payload = {}
    for variant in variants:
        model_ids = sorted(aggregated.get(variant, {}).keys())
        payload[f"{variant}_model_id"] = np.array(model_ids, dtype=np.int32)
        payload[f"{variant}_num_seeds"] = np.array(
            [aggregated[variant][model_id]["num_seeds"] for model_id in model_ids], dtype=np.int32
        )
        for metric_key in METRIC_KEYS:
            payload[f"{variant}_{metric_key}_mean"] = np.array(
                [aggregated[variant][model_id][f"{metric_key}_mean"] for model_id in model_ids], dtype=np.float32
            )
            payload[f"{variant}_{metric_key}_std"] = np.array(
                [aggregated[variant][model_id][f"{metric_key}_std"] for model_id in model_ids], dtype=np.float32
            )

    if payload:
        np.savez(path, **payload)
        print(f"[SAVED] {path}")


def load_aggregate_csv(path: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    aggregated: Dict[str, Dict[int, Dict[str, float]]] = {}
    if not os.path.exists(path):
        print(f"[WARN] aggregate CSV not found: {path}")
        return aggregated

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variant = row["variant"]
            model_id = int(row["model_id"])
            if variant not in aggregated:
                aggregated[variant] = {}
            summary = {
                "num_seeds": int(float(row["num_seeds"])),
            }
            for metric_key in METRIC_KEYS:
                summary[f"{metric_key}_mean"] = float(row[f"{metric_key}_mean"])
                summary[f"{metric_key}_std"] = float(row[f"{metric_key}_std"])
            aggregated[variant][model_id] = summary
    return aggregated


def plot_reward_overlay(output_dir: str, aggregated: Dict[str, Dict[int, Dict[str, float]]], variants: List[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "bc": "tab:purple",
        "baseline": "tab:blue",
        "encoder_only": "tab:orange",
        "full": "tab:green",
    }

    plt.figure(figsize=(9, 5.5))
    has_curve = False
    for variant in variants:
        variant_summary = aggregated.get(variant, {})
        model_ids = sorted(variant_summary.keys())
        if not model_ids:
            continue

        x = np.array(model_ids, dtype=np.int32)
        mean = np.array([variant_summary[mid]["avg_reward_mean"] for mid in model_ids], dtype=np.float32)
        std = np.array([variant_summary[mid]["avg_reward_std"] for mid in model_ids], dtype=np.float32)
        valid = np.isfinite(mean)
        if not np.any(valid):
            continue

        x = x[valid]
        mean = mean[valid]
        std = std[valid]
        color = colors.get(variant)
        plt.plot(x, mean, linewidth=2.2, label=variant, color=color)
        plt.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
        has_curve = True

    plt.title("Offline Rollout Reward Comparison (mean ± std across seeds)")
    plt.xlabel("Checkpoint ID")
    plt.ylabel("Average Reward")
    plt.grid(True, alpha=0.3)
    if has_curve:
        plt.legend()
    plt.tight_layout()

    png_path = os.path.join(output_dir, "offline_actor_sweep_reward_overlay.png")
    plt.savefig(png_path, dpi=180)
    plt.close()
    print(f"[SAVED] {png_path}")


def plot_aggregate_results(output_dir: str, aggregated: Dict[str, Dict[int, Dict[str, float]]], variants: List[str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "bc": "tab:purple",
        "baseline": "tab:blue",
        "encoder_only": "tab:orange",
        "full": "tab:green",
    }
    metric_specs = [
        ("avg_reward", "Average Reward"),
        ("avg_cost", "Average Cost"),
        ("avg_steps", "Average Steps"),
        ("collision_rate", "Collision Rate"),
        ("offroad_rate", "Offroad Rate"),
    ]

    _, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()
    has_legend = False

    for index, (metric_key, title) in enumerate(metric_specs):
        ax = axes[index]
        for variant in variants:
            variant_summary = aggregated.get(variant, {})
            model_ids = sorted(variant_summary.keys())
            if not model_ids:
                continue

            x = np.array(model_ids, dtype=np.int32)
            mean = np.array([variant_summary[mid][f"{metric_key}_mean"] for mid in model_ids], dtype=np.float32)
            std = np.array([variant_summary[mid][f"{metric_key}_std"] for mid in model_ids], dtype=np.float32)
            valid = np.isfinite(mean)
            if not np.any(valid):
                continue

            x = x[valid]
            mean = mean[valid]
            std = std[valid]
            color = colors.get(variant)
            ax.plot(x, mean, linewidth=2, label=variant, color=color)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
            if index == 0:
                has_legend = True

        ax.set_title(title)
        ax.set_xlabel("Checkpoint ID")
        ax.grid(True, alpha=0.3)

    if has_legend:
        axes[0].legend()
    axes[5].axis("off")
    plt.suptitle("Offline Rollout Comparison (mean ± std across seeds)", fontsize=14)
    plt.tight_layout()

    png_path = os.path.join(output_dir, "offline_actor_sweep_mean_std.png")
    plt.savefig(png_path, dpi=180)
    plt.close()
    print(f"[SAVED] {png_path}")


def save_outputs(output_dir: str, rows: List[Dict[str, object]], aggregated: Dict[str, Dict[int, Dict[str, float]]], variants: List[str]) -> None:
    raw_csv_path = os.path.join(output_dir, "offline_actor_sweep_raw.csv")
    raw_npz_path = os.path.join(output_dir, "offline_actor_sweep_raw.npz")
    save_raw_csv(raw_csv_path, rows)
    save_raw_npz(raw_npz_path, rows)

    aggregate_csv_path = os.path.join(output_dir, "offline_actor_sweep_aggregate.csv")
    aggregate_npz_path = os.path.join(output_dir, "offline_actor_sweep_aggregate.npz")
    save_aggregate_csv(aggregate_csv_path, aggregated, variants)
    save_aggregate_npz(aggregate_npz_path, aggregated, variants)
    plot_aggregate_results(output_dir, aggregated, variants)
    plot_reward_overlay(output_dir, aggregated, variants)


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    print("\n" + "=" * 72)
    print("Offline actor sweep")
    print(f"  variants={', '.join(args.variants)}")
    print(f"  seeds={'auto' if args.seeds is None else ', '.join(str(seed) for seed in args.seeds)}")
    print(f"  episodes={args.episodes} max_steps={args.max_steps} device={args.device}")
    print(f"  output_dir={args.output_dir}")
    print(f"  plot_only={args.plot_only}")
    print("=" * 72)

    if args.plot_only:
        aggregate_csv_path = os.path.join(args.output_dir, "offline_actor_sweep_aggregate.csv")
        aggregated = load_aggregate_csv(aggregate_csv_path)
        if not aggregated:
            print("[WARN] no aggregate data loaded; skip plotting")
            return
        plot_aggregate_results(args.output_dir, aggregated, args.variants)
        plot_reward_overlay(args.output_dir, aggregated, args.variants)
        return

    rows = []
    summary = {}

    for variant in args.variants:
        base_dir = resolve_variant_base_dir(args.ckpt_root, variant)
        if not os.path.isdir(base_dir):
            print(f"[WARN] skip missing variant dir: {base_dir}")
            continue

        seed_dirs = discover_seed_dirs(base_dir, args.seeds)
        if not seed_dirs:
            print(f"[WARN] no seed/base dirs found for {variant}: {base_dir}")
            continue

        model_ids = resolve_model_ids(seed_dirs, args.model_ids)
        if not model_ids:
            print(f"[WARN] no common actor_*.pth found for {variant}")
            continue

        print(
            f"\n--- Variant: {variant} | seeds={', '.join(seed for seed, _ in seed_dirs)} | "
            f"checkpoints={len(model_ids)} ---"
        )
        summary[variant] = []

        for seed_label, ckpt_dir in seed_dirs:
            print(f"\n[SEED] variant={variant} seed={seed_label} ckpt_dir={ckpt_dir}")
            for model_id in model_ids:
                cmd = build_command(ckpt_dir, model_id, args)
                t0 = time.time()
                code, text = stream_command(cmd)
                elapsed = time.time() - t0
                metrics = extract_metrics(text)
                status = "OK" if code == 0 else f"FAILED({code})"

                row = {
                    "variant": variant,
                    "seed": seed_label,
                    "ckpt_dir": ckpt_dir,
                    "model_id": model_id,
                    "status": status,
                    "elapsed_s": round(elapsed, 3),
                    "avg_reward": metrics["avg_reward"],
                    "avg_cost": metrics["avg_cost"],
                    "avg_steps": metrics["avg_steps"],
                    "collision_rate": metrics["collision_rate"],
                    "offroad_rate": metrics["offroad_rate"],
                }
                rows.append(row)
                summary[variant].append(row)

                print(
                    f"[DONE] {variant} seed={seed_label} id={model_id} status={status} "
                    f"reward={row['avg_reward']:.2f} cost={row['avg_cost']:.2f} "
                    f"steps={row['avg_steps']:.2f} coll={row['collision_rate']:.3f} offroad={row['offroad_rate']:.3f}"
                )

    aggregated = aggregate_rows(rows, args.variants)
    save_outputs(args.output_dir, rows, aggregated, args.variants)

    print("\n=== Summary ===")
    for variant, variant_rows in summary.items():
        ok_count = sum(1 for row in variant_rows if row["status"] == "OK")
        print(f"{variant}: {ok_count}/{len(variant_rows)} evaluations passed")


if __name__ == "__main__":
    main()

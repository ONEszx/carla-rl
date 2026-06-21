# -*- coding: utf-8 -*-
"""
Ablation Experiment Runner for EasyCarla-RL.

Runs offline ablation variants sequentially and produces a summary table.
Supports multi-seed execution and mean±std comparison plots.

Default ablation view:
    baseline      -> Original Diffusion QL (OfflineRL)
    encoder_only  -> Encoder trained but no priority sampling
    full          -> Encoder + priority-weighted sampling (Ours)

Optional paper-style extra baseline:
    bc            -> Pure Behavior Cloning

Usage:
    # Run default ablation trio
    python example/run_ablation.py

    # Run specific variants only
    python example/run_ablation.py --variants bc baseline full

    # Run multiple seeds and aggregate plots
    python example/run_ablation.py --seeds 0 1 2
"""

import argparse
import os
import sys
import subprocess
import time

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Default paths
DEFAULT_ENCODER_CKPT = os.path.join(CURRENT_DIR, "params_representation", "encoder_final.pth")
DEFAULT_PRIORITY_PATH = os.path.join(CURRENT_DIR, "params_representation", "priority_scores.npz")
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easycarla_offline_dataset.hdf5")


# ----------------------------------------------------------------------
# Variant definitions
# ----------------------------------------------------------------------
ABLATION_VARIANTS = {
    "bc": {
        "description": "Pure Behavior Cloning baseline",
        "flags": [],
        "expected_output": "params_bc/",
    },
    "baseline": {
        "description": "Original Diffusion QL (no encoder, no priority)",
        "flags": [],
        "expected_output": "params_baseline/",
    },
    "encoder_only": {
        "description": "Encoder trained but not used for priority sampling",
        "flags": [
            "--encoder_ckpt", DEFAULT_ENCODER_CKPT,
        ],
        "expected_output": "params_encoder_only/",
    },
    "full": {
        "description": "Encoder + priority-weighted sampling (full method)",
        "flags": [
            "--encoder_ckpt", DEFAULT_ENCODER_CKPT,
            "--priority_path", DEFAULT_PRIORITY_PATH,
            "--priority_power", "2.0",
        ],
        "expected_output": "params_full/",
    },
}
DEFAULT_ABLATION_VARIANTS = ["baseline", "encoder_only", "full"]


def resolve_variant_output_dir(variant_name, global_args, seed=None):
    base_dir = os.path.join(global_args.output_root, f"params_{variant_name}") if global_args.output_root else os.path.join(CURRENT_DIR, f"params_{variant_name}")
    if seed is None:
        return base_dir
    return os.path.join(base_dir, f"seed_{seed}")


def build_command(variant_name, global_args, variant_specific_flags, seed):
    """Build the full command-line for one ablation variant + seed."""
    cmd = [
        sys.executable,
        os.path.join(CURRENT_DIR, "train_diffusion_ql_priority.py"),
        "--ablation", variant_name,
        "--data_path", DEFAULT_DATA_PATH,
        "--seed", str(seed),
        "--output_dir", resolve_variant_output_dir(variant_name, global_args, seed),
    ]
    if global_args.num_epochs:
        cmd.extend(["--num_epochs", str(global_args.num_epochs)])
    if global_args.batch_size:
        cmd.extend(["--batch_size", str(global_args.batch_size)])
    if global_args.lr:
        cmd.extend(["--lr", str(global_args.lr)])
    if global_args.steps_per_epoch:
        cmd.extend(["--steps_per_epoch", str(global_args.steps_per_epoch)])
    if global_args.device:
        cmd.extend(["--device", global_args.device])
    if global_args.priority_warmup_epochs is not None:
        cmd.extend(["--priority_warmup_epochs", str(global_args.priority_warmup_epochs)])
    if global_args.finetune_encoder:
        cmd.append("--finetune_encoder")
    if global_args.encoder_lr is not None:
        cmd.extend(["--encoder_lr", str(global_args.encoder_lr)])
    if global_args.encoder_grad_norm is not None:
        cmd.extend(["--encoder_grad_norm", str(global_args.encoder_grad_norm)])
    if global_args.priority_encoder_ckpt:
        cmd.extend(["--priority_encoder_ckpt", global_args.priority_encoder_ckpt])

    cmd.extend(variant_specific_flags)
    return cmd


def run_variant(variant_name, seed, cmd):
    """Execute one variant/seed pair and capture output."""
    print("\n" + "=" * 60)
    print(f"  Running variant: {variant_name} | seed={seed}")
    print(f"  Description: {ABLATION_VARIANTS[variant_name]['description']}")
    print(f"  Command: {' '.join(cmd)}")
    print("=" * 60)

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    success = result.returncode == 0
    print(f"\n  Variant '{variant_name}' seed={seed} finished in {elapsed:.1f}s — {'OK' if success else 'FAILED'}")
    return success, elapsed


def print_summary(results):
    """Print a summary table of all variant/seed runs."""
    print("\n" + "=" * 80)
    print("  ABLATION SUMMARY")
    print("=" * 80)
    print(f"  {'Variant':<18} {'Seed':<8} {'Status':<10} {'Time':<10} {'Description'}")
    print("-" * 80)
    for variant_name, seed_results in results.items():
        desc = ABLATION_VARIANTS[variant_name]["description"]
        for seed, (ok, elapsed) in seed_results.items():
            status = "OK" if ok else "FAILED"
            print(f"  {variant_name:<18} {seed:<8} {status:<10} {elapsed:.1f}s     {desc}")
    print("-" * 80)

    total = sum(len(seed_results) for seed_results in results.values())
    passed = sum(1 for seed_results in results.values() for ok, _ in seed_results.values() if ok)
    print(f"  Passed: {passed}/{total}")
    print("=" * 80)


def load_seed_history(history_path):
    with np.load(history_path) as data:
        seed_value = np.array([-1], dtype=np.int32)
        if "seed" in data.files:
            seed_value = np.asarray(data["seed"], dtype=np.int32)
        return {
            "epoch": np.asarray(data["epoch"], dtype=np.int32),
            "bc_loss": np.asarray(data["bc_loss"], dtype=np.float32),
            "ql_loss": np.asarray(data["ql_loss"], dtype=np.float32),
            "critic_loss": np.asarray(data["critic_loss"], dtype=np.float32),
            "seed": int(seed_value.reshape(-1)[0]),
        }


def aggregate_histories(histories):
    if not histories:
        return None

    min_len = min(len(history["epoch"]) for history in histories)
    trimmed = []
    for history in histories:
        trimmed.append({key: value[:min_len] if isinstance(value, np.ndarray) else value for key, value in history.items()})

    epoch = trimmed[0]["epoch"]
    metrics = {}
    for metric_key in ["bc_loss", "ql_loss", "critic_loss"]:
        stacked = np.stack([history[metric_key] for history in trimmed], axis=0)
        metrics[metric_key] = {
            "mean": stacked.mean(axis=0),
            "std": stacked.std(axis=0),
        }

    return {
        "epoch": epoch,
        "metrics": metrics,
        "num_seeds": len(trimmed),
        "seeds": [history["seed"] for history in trimmed],
    }


def save_aggregate_history(variant_name, aggregated, output_root=None):
    if aggregated is None:
        return

    variant_dir = resolve_variant_output_dir(variant_name, argparse.Namespace(output_root=output_root), seed=None)
    os.makedirs(variant_dir, exist_ok=True)
    save_path = os.path.join(variant_dir, f"training_history_{variant_name}_aggregate.npz")
    np.savez(
        save_path,
        epoch=aggregated["epoch"],
        bc_loss_mean=aggregated["metrics"]["bc_loss"]["mean"],
        bc_loss_std=aggregated["metrics"]["bc_loss"]["std"],
        ql_loss_mean=aggregated["metrics"]["ql_loss"]["mean"],
        ql_loss_std=aggregated["metrics"]["ql_loss"]["std"],
        critic_loss_mean=aggregated["metrics"]["critic_loss"]["mean"],
        critic_loss_std=aggregated["metrics"]["critic_loss"]["std"],
        num_seeds=np.array([aggregated["num_seeds"]], dtype=np.int32),
        seeds=np.array(aggregated["seeds"], dtype=np.int32),
    )
    print(f"[SAVED] {save_path}")


def generate_comparison_plot(variant_names, seeds, output_root=None):
    """Build one comparison plot covering all selected ablation variants with mean±std shading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_dir = output_root or CURRENT_DIR
    histories = {}
    for name in variant_names:
        seed_histories = []
        for seed in seeds:
            history_path = os.path.join(base_dir, f"params_{name}", f"seed_{seed}", f"training_history_{name}.npz")
            if not os.path.exists(history_path):
                print(f"[WARN] Missing history file for {name} seed={seed}: {history_path}")
                continue
            seed_histories.append(load_seed_history(history_path))

        aggregated = aggregate_histories(seed_histories)
        if aggregated is None:
            print(f"[WARN] No training histories found for {name}, skip this variant.")
            continue
        histories[name] = aggregated
        save_aggregate_history(name, aggregated, output_root)

    if not histories:
        print("[WARN] No training histories found, skip comparison plot.")
        return

    colors = {
        "bc": "tab:purple",
        "baseline": "tab:blue",
        "encoder_only": "tab:orange",
        "full": "tab:green",
    }
    metrics = [
        ("bc_loss", "BC Loss"),
        ("ql_loss", "QL Loss"),
        ("critic_loss", "Critic Loss"),
    ]

    _, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (metric_key, title) in zip(axes, metrics):
        for name in variant_names:
            if name not in histories:
                continue
            history = histories[name]
            epoch = history["epoch"]
            mean = history["metrics"][metric_key]["mean"]
            std = history["metrics"][metric_key]["std"]
            color = colors.get(name, None)
            ax.plot(epoch, mean, label=name, color=color, linewidth=2)
            ax.fill_between(epoch, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    axes[0].legend()
    plt.suptitle(f"Ablation Comparison ({len(seeds)} seeds)", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(base_dir, "ablation_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[SAVED] {save_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for EasyCarla-RL offline stage."
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=DEFAULT_ABLATION_VARIANTS,
        choices=list(ABLATION_VARIANTS.keys()),
        help="Which offline variants to run. Default keeps the original ablation trio: baseline encoder_only full.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Random seeds to run and aggregate. Example: --seeds 0 1 2",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of epochs for all variants.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=None,
        help="Override steps per epoch for all variants.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for all variants.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate for all variants.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu).",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root directory for all output checkpoints.",
    )
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=DEFAULT_ENCODER_CKPT,
        help="Path to encoder checkpoint (used by encoder_only and full).",
    )
    parser.add_argument(
        "--priority_path",
        type=str,
        default=DEFAULT_PRIORITY_PATH,
        help="Path to priority scores (used by full).",
    )
    parser.add_argument(
        "--priority_warmup_epochs",
        type=int,
        default=5,
        help="Use uniform sampling for the first N epochs before enabling priority sampling.",
    )
    parser.add_argument(
        "--finetune_encoder",
        action="store_true",
        help="Enable RL encoder finetuning for latent ablations.",
    )
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=1e-4,
        help="Learning rate for RL encoder finetuning.",
    )
    parser.add_argument(
        "--encoder_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping max norm for RL encoder finetuning.",
    )
    parser.add_argument(
        "--priority_encoder_ckpt",
        type=str,
        default=None,
        help="Optional frozen encoder checkpoint for the priority/reference branch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for v in args.variants:
        if v in ("encoder_only", "full"):
            ckpt = args.encoder_ckpt or DEFAULT_ENCODER_CKPT
            if not os.path.exists(ckpt):
                print(f"[ERROR] Encoder checkpoint not found: {ckpt}")
                print(f"  Please run train_representation.py first to generate: {ckpt}")
                sys.exit(1)

    print("\n" + "#" * 60)
    print("#  Ablation Experiment Runner")
    print(f"#  Variants: {', '.join(args.variants)}")
    print(f"#  Seeds: {', '.join(str(seed) for seed in args.seeds)}")
    print("#" * 60)

    results = {}
    for variant_name in args.variants:
        cfg = ABLATION_VARIANTS[variant_name]
        flags = list(cfg["flags"])
        if variant_name in ("encoder_only", "full"):
            for i, f in enumerate(flags):
                if f == "--encoder_ckpt":
                    flags[i + 1] = args.encoder_ckpt or DEFAULT_ENCODER_CKPT
                if f == "--priority_path":
                    flags[i + 1] = args.priority_path or DEFAULT_PRIORITY_PATH

        results[variant_name] = {}
        for seed in args.seeds:
            cmd = build_command(variant_name, args, flags, seed)
            ok, elapsed = run_variant(variant_name, seed, cmd)
            results[variant_name][seed] = (ok, elapsed)
            print()

    print_summary(results)
    generate_comparison_plot(args.variants, args.seeds, args.output_root)

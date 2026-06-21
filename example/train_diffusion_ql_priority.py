# -*- coding: utf-8 -*-
"""
Unified Chapter-1 offline training entry with paper-style and ablation-style support.

This script supports four offline variants:
1. bc            -> pure Behavior Cloning baseline
2. baseline      -> original Diffusion QL (OfflineRL)
3. encoder_only  -> encoder ablation without priority
4. full          -> encoder + priority-weighted sampling (Ours)

Usage:
    # BC baseline
    python example/train_diffusion_ql_priority.py --ablation bc

    # OfflineRL baseline
    python example/train_diffusion_ql_priority.py --ablation baseline

    # Encoder-only ablation
    python example/train_diffusion_ql_priority.py \
        --ablation encoder_only \
        --encoder_ckpt example/params_representation/encoder_final.pth

    # Ours / full method
    python example/train_diffusion_ql_priority.py \
        --ablation full \
        --encoder_ckpt example/params_representation/encoder_final.pth \
        --priority_path example/params_representation/priority_scores.npz
"""

import sys
import os

# Ensure clean output flushing
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1) if hasattr(sys.stdout, 'fileno') else sys.stdout
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("\n=== [Priority-DiffQL] Starting ===")
sys.stdout.flush()

import argparse
import random
import h5py
import numpy as np
import torch
import time

print("[OK] Imports")

# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ----------------------------------------------------------------------
# Ablation configurations
# ----------------------------------------------------------------------
ABLATION_MODES = {
    "bc": {
        "use_encoder": False,
        "use_priority": False,
        "method_type": "bc",
        "description": "Pure Behavior Cloning baseline",
    },
    "baseline": {
        "use_encoder": False,
        "use_priority": False,
        "method_type": "dql",
        "description": "Original Diffusion QL (no encoder, no priority)",
    },
    "encoder_only": {
        "use_encoder": True,
        "use_priority": False,
        "method_type": "dql",
        "description": "Encoder trained but not used for priority sampling",
    },
    "full": {
        "use_encoder": True,
        "use_priority": True,
        "method_type": "dql",
        "description": "Encoder + priority-weighted sampling (full method)",
    },
}

DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "params_priority")


# ----------------------------------------------------------------------
# Replay Buffer
# ----------------------------------------------------------------------
class SimpleReplayBuffer:
    """
    Standard replay buffer with optional priority sampling support.

    Key addition: sample_indices() method, called by PriorityBuffer wrapper
    when priority sampling is active.
    """

    def __init__(self, obs, act, next_obs, rew, done):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.act = torch.tensor(act, dtype=torch.float32)
        self.next_obs = torch.tensor(next_obs, dtype=torch.float32)
        self.rew = torch.tensor(rew, dtype=torch.float32).unsqueeze(1)
        self.not_done = 1.0 - torch.tensor(done, dtype=torch.float32).unsqueeze(1)
        self.device = None

    def __len__(self):
        return len(self.obs)

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.obs), batch_size)
        return self._fetch(idx)

    def sample_indices(self, indices):
        """Fetch batch by explicit index array (used by PriorityBuffer)."""
        return self._fetch(np.array(indices))

    def _fetch(self, idx):
        return (
            self.obs[idx].to(self.device),
            self.act[idx].to(self.device),
            self.next_obs[idx].to(self.device),
            self.rew[idx].to(self.device),
            self.not_done[idx].to(self.device),
        )

    def to(self, device):
        self.device = device


# ----------------------------------------------------------------------
# Priority Buffer Wrapper
# ----------------------------------------------------------------------
class PriorityBuffer:
    """
    Wraps SimpleReplayBuffer with priority-based sampling.

    When priority_sampler is None or disabled, falls back to uniform sampling.
    """

    def __init__(self, base_buffer, priority_sampler=None):
        self.base = base_buffer
        self.priority_sampler = priority_sampler

    def sample(self, batch_size):
        if self.priority_sampler is not None and self.priority_sampler.is_enabled:
            N = len(self.base)
            idx = self.priority_sampler.sample_indices(batch_size)
            idx = np.clip(idx, 0, N - 1)
            return self.base.sample_indices(idx)
        return self.base.sample(batch_size)

    def __len__(self):
        return len(self.base)

    def add(self, *args, **kwargs):
        return self.base.add(*args, **kwargs)


# ----------------------------------------------------------------------
# Training Visualizer
# ----------------------------------------------------------------------
class TrainingVisualizer:
    def __init__(self, total_epochs, ablation_name, seed=None):
        self.total_epochs = total_epochs
        self.ablation_name = ablation_name
        self.seed = seed
        self.history = {
            "bc_loss": [], "ql_loss": [],
            "critic_loss": [], "epoch": [],
        }
        self.epoch_times = []
        self.start_time = time.time()
        self.best_bc_loss = float("inf")

    def update(self, epoch, metrics):
        bc_loss = np.mean(metrics["bc_loss"])
        ql_loss = np.mean(metrics["ql_loss"])
        critic_loss = np.mean(metrics["critic_loss"])

        avg_time = np.mean(self.epoch_times) if self.epoch_times else 0
        eta = (self.total_epochs - epoch) * avg_time if avg_time > 0 else 0

        self.history["bc_loss"].append(bc_loss)
        self.history["ql_loss"].append(ql_loss)
        self.history["critic_loss"].append(critic_loss)
        self.history["epoch"].append(epoch)

        is_best = bc_loss < self.best_bc_loss
        if is_best:
            self.best_bc_loss = bc_loss

        return bc_loss, ql_loss, critic_loss, eta, is_best

    def print_progress(self, epoch, bc, ql, cr, eta, is_best, saved):
        bar_len = 40
        pct = epoch / self.total_epochs
        filled = int(bar_len * pct)
        bar = "#" * filled + "-" * (bar_len - filled)

        eta_str = f"{int(eta/60)}m {int(eta%60)}s" if eta > 0 else "calculating..."
        best_mark = " <-- BEST" if is_best else ""
        save_mark = " [SAVED]" if saved else ""

        print(f"\n[{epoch}/{self.total_epochs}] [{bar}] {pct*100:.0f}% | ETA: {eta_str}")
        print(f"  BC: {bc:.4f}{best_mark} | QL: {ql:.4f} | CR: {cr:.4f}{save_mark}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        sys.stdout.flush()

    def plot_and_save(self, save_dir):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        e = self.history["epoch"]
        axes[0].plot(e, self.history["bc_loss"], "b-")
        axes[0].set_title("BC Loss")
        axes[1].plot(e, self.history["ql_loss"], "g-")
        axes[1].set_title("QL Loss")
        axes[2].plot(e, self.history["critic_loss"], "r-")
        axes[2].set_title("Critic Loss")
        plt.suptitle(f"Ablation: {self.ablation_name}", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"training_curves_{self.ablation_name}.png"))
        plt.close()
        print(f"[SAVED] training_curves_{self.ablation_name}.png")

    def save_history(self, save_dir):
        history_path = os.path.join(save_dir, f"training_history_{self.ablation_name}.npz")
        np.savez(
            history_path,
            epoch=np.array(self.history["epoch"], dtype=np.int32),
            bc_loss=np.array(self.history["bc_loss"], dtype=np.float32),
            ql_loss=np.array(self.history["ql_loss"], dtype=np.float32),
            critic_loss=np.array(self.history["critic_loss"], dtype=np.float32),
            seed=np.array([getattr(self, "seed", -1)], dtype=np.int32),
        )
        print(f"[SAVED] training_history_{self.ablation_name}.npz")


def load_encoder_from_ckpt(ckpt_path, device):
    from representation.encoder import StateEncoder

    ckpt = torch.load(ckpt_path, map_location=device)
    enc_config = ckpt.get("config", {})
    encoder = StateEncoder(
        state_dim=enc_config.get("state_dim", 307),
        latent_dim=enc_config.get("latent_dim", 64),
        hidden_dim=enc_config.get("hidden_dim", 256),
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    return encoder, enc_config


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Main training function
# ----------------------------------------------------------------------
def train_priority_diffusion_ql(args):
    """Run unified offline training for BC / OfflineRL / ablations / Ours."""

    set_random_seed(args.seed)
    ablation_cfg = ABLATION_MODES.get(args.ablation, ABLATION_MODES["baseline"])
    method_type = ablation_cfg["method_type"]

    print("\n" + "=" * 60)
    print(f"ABLATION MODE: {args.ablation} | SEED: {args.seed}")
    print(f"  Description: {ablation_cfg['description']}")
    print(f"  method_type: {method_type}")
    print(f"  use_encoder: {ablation_cfg['use_encoder']}")
    print(f"  use_priority: {ablation_cfg['use_priority']}")
    print("=" * 60)
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    save_dir = args.output_dir or os.path.join(CURRENT_DIR, f"params_{args.ablation}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("\n[1] Loading dataset...")
    sys.stdout.flush()

    dataset_path = args.data_path or os.path.join(PROJECT_ROOT, "data", "easycarla_offline_dataset.hdf5")
    with h5py.File(dataset_path, "r") as f:
        print(f"  Reading {dataset_path}...")
        obs = f["observations"][:]
        act = f["actions"][:]
        next_obs = f["next_observations"][:]
        rew = f["rewards"][:]
        done = f["done"][:]

    print(f"  Dataset: {len(obs):,} samples, obs_dim={obs.shape[1]}, act_dim={act.shape[1]}")
    if ablation_cfg["use_priority"] and not args.priority_path:
        raise ValueError(f"Ablation '{args.ablation}' requires --priority_path")
    if ablation_cfg["use_priority"]:
        print(f"  Priority path: {args.priority_path}")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Load encoder(s) if needed
    # ------------------------------------------------------------------
    encoder_rl = None
    encoder_prio = None
    encoder_config = None
    raw_state_dim = obs.shape[1]
    state_dim = raw_state_dim

    if ablation_cfg["use_encoder"]:
        if not args.encoder_ckpt:
            raise ValueError(f"Ablation '{args.ablation}' requires --encoder_ckpt")

        print("\n[2] Loading encoder checkpoint...")
        sys.stdout.flush()

        encoder_rl, encoder_config = load_encoder_from_ckpt(args.encoder_ckpt, args.device)
        state_dim = encoder_config.get("latent_dim", 64)

        if args.finetune_encoder:
            encoder_rl.train()
        else:
            encoder_rl.eval()
            for p in encoder_rl.parameters():
                p.requires_grad = False

        print(f"  RL encoder loaded from {args.encoder_ckpt}")
        print(f"  RL encoder mode: {'finetune' if args.finetune_encoder else 'frozen'}")

        if ablation_cfg["use_priority"]:
            prio_ckpt = args.priority_encoder_ckpt or args.encoder_ckpt
            encoder_prio, _ = load_encoder_from_ckpt(prio_ckpt, args.device)
            encoder_prio.eval()
            for p in encoder_prio.parameters():
                p.requires_grad = False
            print(f"  Priority encoder loaded from {prio_ckpt} (frozen)")
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Build replay buffer
    # ------------------------------------------------------------------
    print("\n[3] Building replay buffer...")
    base_buffer = SimpleReplayBuffer(obs, act, next_obs, rew, done)

    # ------------------------------------------------------------------
    # Priority sampler (only if encoder + priority ablation)
    # ------------------------------------------------------------------
    priority_sampler = None
    if ablation_cfg["use_priority"]:
        print("\n[4] Loading priority sampler...")
        sys.stdout.flush()
        from representation.priority_sampler import PrioritySampler

        priority_sampler = PrioritySampler(
            priority_path=args.priority_path,
            priority_power=args.priority_power,
            priority_offset=args.priority_offset,
            max_priority_weight=args.max_priority_weight,
            device=args.device,
        )
        print(f"  {priority_sampler.summary()}")
        if encoder_prio is not None:
            print("  Priority reference space: frozen encoder")
        sys.stdout.flush()

    # Wrap buffer with priority if enabled
    priority_buffer = PriorityBuffer(base_buffer, priority_sampler) if priority_sampler is not None else None

    # ------------------------------------------------------------------
    # Device and model
    # ------------------------------------------------------------------
    print("\n[5] Initializing model...")
    sys.stdout.flush()

    device = torch.device(args.device)
    base_buffer.to(device)

    if method_type == "bc":
        from agents.bc_diffusion import Diffusion_BC

        model = Diffusion_BC(
            state_dim=state_dim,
            action_dim=3,
            max_action=1.0,
            device=device,
            discount=args.discount,
            tau=args.tau,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.n_timesteps,
            lr=args.lr,
        )
    else:
        from agents.ql_diffusion import Diffusion_QL

        model = Diffusion_QL(
            state_dim=state_dim,
            action_dim=3,
            max_action=1.0,
            device=device,
            discount=args.discount,
            tau=args.tau,
            eta=args.eta,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.n_timesteps,
            lr=args.lr,
            grad_norm=args.grad_norm,
            encoder=encoder_rl,
            encoder_lr=args.encoder_lr,
            encoder_grad_norm=args.encoder_grad_norm,
            finetune_encoder=args.finetune_encoder,
        )
    print(f"  Device: {device}")
    print(f"  Method: {method_type}")
    print(f"  Config: epochs={args.num_epochs}, batch={args.batch_size}, lr={args.lr}, seed={args.seed}")
    print(f"  Raw state dim: {raw_state_dim} | Policy state dim: {state_dim}")
    if encoder_rl is not None:
        print(f"  RL encoder finetune: {args.finetune_encoder} | encoder_lr={args.encoder_lr}")
    if ablation_cfg["use_priority"]:
        print(f"  Priority warmup epochs: {args.priority_warmup_epochs} (uniform for first {args.priority_warmup_epochs} epochs)")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING START")
    print(f"  Ablation: {args.ablation}")
    print(f"  Epochs: {args.num_epochs} | Steps: {args.steps_per_epoch} | Batch: {args.batch_size}")
    print("=" * 60)
    sys.stdout.flush()

    visualizer = TrainingVisualizer(args.num_epochs, args.ablation, seed=args.seed)

    for epoch in range(1, args.num_epochs + 1):
        t0 = time.time()

        if ablation_cfg["use_priority"] and priority_buffer is not None and epoch > args.priority_warmup_epochs:
            current_buffer = priority_buffer
            sampling_mode = "priority"
        else:
            current_buffer = base_buffer
            sampling_mode = "uniform"

        print(f"[Epoch {epoch}] sampling={sampling_mode}")
        sys.stdout.flush()

        metrics = model.train(
            replay_buffer=current_buffer,
            iterations=args.steps_per_epoch,
            batch_size=args.batch_size,
        )

        visualizer.epoch_times.append(time.time() - t0)

        bc, ql, cr, eta, is_best = visualizer.update(epoch, metrics)
        save = (epoch % args.eval_freq == 0 or epoch == args.num_epochs)
        if save:
            model.save_model(save_dir, id=epoch)
            if encoder_rl is not None and hasattr(model, "encoder") and model.encoder is not None:
                encoder_payload = {
                    "encoder_state_dict": model.encoder.state_dict(),
                    "config": model.encoder.get_config(),
                    "source_encoder_ckpt": args.encoder_ckpt,
                    "finetuned": args.finetune_encoder,
                    "epoch": epoch,
                    "ablation": args.ablation,
                    "method_type": method_type,
                }
                encoder_save_path = os.path.join(save_dir, f"encoder_rl_{epoch}.pth")
                torch.save(encoder_payload, encoder_save_path)
                torch.save(encoder_payload, os.path.join(save_dir, "encoder_rl.pth"))

        visualizer.print_progress(epoch, bc, ql, cr, eta, is_best, save)

    print("\n=== TRAINING COMPLETE ===")
    visualizer.plot_and_save(save_dir)
    visualizer.save_history(save_dir)

    # Save ablation config for reproducibility
    config_path = os.path.join(save_dir, f"ablation_config_{args.ablation}.txt")
    with open(config_path, "w") as f:
        f.write(f"ablation={args.ablation}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"method_type={method_type}\n")
        f.write(f"description={ablation_cfg['description']}\n")
        f.write(f"use_encoder={ablation_cfg['use_encoder']}\n")
        f.write(f"use_priority={ablation_cfg['use_priority']}\n")
        f.write(f"encoder_ckpt={args.encoder_ckpt}\n")
        f.write(f"priority_encoder_ckpt={args.priority_encoder_ckpt}\n")
        f.write(f"finetune_encoder={args.finetune_encoder}\n")
        f.write(f"encoder_lr={args.encoder_lr}\n")
        f.write(f"encoder_grad_norm={args.encoder_grad_norm}\n")
        f.write(f"priority_path={args.priority_path}\n")
        f.write(f"priority_power={args.priority_power}\n")
        f.write(f"priority_warmup_epochs={args.priority_warmup_epochs}\n")
        f.write(f"num_epochs={args.num_epochs}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"lr={args.lr}\n")
        f.write(f"raw_state_dim={raw_state_dim}\n")
        f.write(f"policy_state_dim={state_dim}\n")
    print(f"[Config] saved to {config_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified offline training entry for BC, OfflineRL, encoder ablations, and Ours."
    )

    # Ablation mode
    parser.add_argument(
        "--ablation",
        type=str,
        default="baseline",
        choices=list(ABLATION_MODES.keys()),
        help="Ablation variant to run. "
             "bc=Behavior Cloning, baseline=OfflineRL, encoder_only=encoder without priority, full=encoder+priority",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints. "
             "Defaults to params_<ablation>/ in the example/ directory.",
    )

    # Data
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to offline HDF5 dataset.",
    )

    # Encoder
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="Path to trained encoder checkpoint (.pth). "
             "Used when ablation is 'encoder_only' or 'full'.",
    )
    parser.add_argument(
        "--priority_encoder_ckpt",
        type=str,
        default=None,
        help="Optional frozen encoder checkpoint for the priority/reference branch. "
             "Defaults to --encoder_ckpt.",
    )
    parser.add_argument(
        "--finetune_encoder",
        action="store_true",
        help="Enable RL encoder finetuning during offline Diffusion QL training.",
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
        help="Gradient clipping max norm for RL encoder when finetuning.",
    )

    # Priority
    parser.add_argument(
        "--priority_path",
        type=str,
        default=None,
        help="Path to priority_scores.npz generated by train_representation.py. "
             "Used when ablation is 'full'.",
    )
    parser.add_argument(
        "--priority_power",
        type=float,
        default=2.0,
        help="Exponent alpha in p_i ∝ (score_i + eps)^alpha. "
             "Higher values give more weight to rare samples. "
             "alpha=0 falls back to uniform sampling.",
    )
    parser.add_argument(
        "--priority_offset",
        type=float,
        default=1e-6,
        help="Epsilon added to scores before power transform.",
    )
    parser.add_argument(
        "--max_priority_weight",
        type=float,
        default=20.0,
        help="Maximum weight ratio cap to prevent degenerate sampling.",
    )
    parser.add_argument(
        "--priority_warmup_epochs",
        type=int,
        default=5,
        help="Use uniform sampling for the first N epochs, then enable priority sampling.",
    )

    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--grad_norm", type=float, default=1.0)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument(
        "--beta_schedule", type=str, default="vp",
        choices=["vp", "linear", "cosine"],
    )
    parser.add_argument("--n_timesteps", type=int, default=5)

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_priority_diffusion_ql(args)
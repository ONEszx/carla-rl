# -*- coding: utf-8 -*-
"""
Contrastive Representation Learning Trainer for EasyCarla-RL.

Trains a StateEncoder on the existing offline dataset using temporal contrastive learning.
Positive samples are temporally adjacent states within the same episode.
Negative samples are states from other episodes or far-away in time.

Usage:
    python example/representation/train_representation.py --data data/easycarla_offline_dataset.hdf5

Outputs:
    - Encoder checkpoint: example/params_representation/encoder_{step}.pth
    - Priority scores:    example/params_representation/priority_scores.npz
    - Training log:      example/params_representation/training_log.txt
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from representation.encoder import StateEncoder, ContrastiveLoss
from representation.contrastive_dataset import (
    HDF5Dataset,
    ContrastiveBatchConstructor,
    RandomBatchSampler,
)


# ----------------------------------------------------------------------
# Default paths
# ----------------------------------------------------------------------
DEFAULT_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "easycarla_offline_dataset.hdf5")
DEFAULT_OUTPUT_DIR = os.path.join(CURRENT_DIR, "params_representation")
DEFAULT_DATA_PATH_MANUAL = os.path.join(PROJECT_ROOT, "data", "easycarla_collect_manual_autopilot.hdf5")


# ----------------------------------------------------------------------
# Training config
# ----------------------------------------------------------------------
class TrainConfig:
    # Data
    data_path: str = DEFAULT_DATA_PATH
    use_manual_data: bool = False  # whether to also use manual dataset
    manual_data_path: str = DEFAULT_DATA_PATH_MANUAL

    # Architecture
    state_dim: int = 307
    latent_dim: int = 64
    hidden_dim: int = 256

    # Contrastive learning
    temperature: float = 0.07
    neighbor_window: int = 5
    num_negatives: int = 255  # 0 = in-batch negatives only
    exclude_neighbors_from_neg: bool = True

    # Training
    batch_size: int = 512
    num_epochs: int = 50
    max_batches_per_epoch: int = 0  # 0 = use all batches in each epoch
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0

    # Logging
    log_every: int = 10  # steps
    save_every: int = 500  # steps
    eval_every: int = 500  # steps

    # Output
    output_dir: str = DEFAULT_OUTPUT_DIR
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------
# Dataset wrapper for DataLoader compatibility
# ----------------------------------------------------------------------
class IndexDataset(torch.utils.data.Dataset):
    """
    Simple dataset that returns indices. The actual state fetching is
    handled by ContrastiveBatchConstructor outside the DataLoader.
    """

    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> int:
        return idx


# ----------------------------------------------------------------------
# Main training function
# ----------------------------------------------------------------------
def train_representation(config: TrainConfig):
    print("=" * 60)
    print("Contrastive Representation Learning")
    print("=" * 60)
    print(f"Device:          {config.device}")
    print(f"Data path:       {config.data_path}")
    print(f"State dim:       {config.state_dim}")
    print(f"Latent dim:      {config.latent_dim}")
    print(f"Batch size:      {config.batch_size}")
    print(f"Num epochs:      {config.num_epochs}")
    if config.max_batches_per_epoch > 0:
        print(f"Max batches/ep:  {config.max_batches_per_epoch}")
    else:
        print("Max batches/ep:  all")
    print(f"LR:              {config.learning_rate}")
    print(f"Temperature:     {config.temperature}")
    print(f"Neighbor window: {config.neighbor_window}")
    print(f"Num negatives:   {config.num_negatives}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------
    os.makedirs(config.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print("\n[1] Loading dataset...")
    episode_ids = None
    if config.use_manual_data and os.path.exists(config.manual_data_path):
        import h5py
        with h5py.File(config.manual_data_path, "r") as hf:
            if "episode_ids" in hf:
                episode_ids = hf["episode_ids"][:]
                print(f"  Loaded episode_ids from manual data: {len(episode_ids)} samples")
            else:
                print("  Manual data has no episode_ids, using done flags")

    dataset = HDF5Dataset(
        hdf5_path=config.data_path,
        observations_key="observations",
        episode_ids=episode_ids,
    )
    print(f"  Dataset: {dataset.N} samples, state_dim={dataset.state_dim}")
    print(f"  Episodes: {len(dataset.episode_boundaries)}")

    # ------------------------------------------------------------------
    # Build batch constructor
    # ------------------------------------------------------------------
    print("\n[2] Building batch constructor...")
    batch_constructor = ContrastiveBatchConstructor(
        dataset=dataset,
        neighbor_window=config.neighbor_window,
        num_negatives=config.num_negatives,
        exclude_neighbors=config.exclude_neighbors_from_neg,
    )
    print(f"  Neighbor window: {config.neighbor_window}")
    print(f"  Num negatives: {config.num_negatives}")

    # Create DataLoader with custom sampler for large datasets
    index_dataset = IndexDataset(dataset.N)
    sampler = RandomBatchSampler(dataset.N, config.batch_size, drop_last=True)
    dataloader = DataLoader(
        index_dataset,
        batch_sampler=sampler,
        num_workers=0,  # HDF5 access is I/O-bound, use 0 workers for simplicity
        pin_memory=(config.device == "cuda"),
    )
    print(f"  Dataloader ready: {len(dataloader)} batches per epoch")
    if config.max_batches_per_epoch > 0:
        print(f"  Effective batches per epoch: {min(len(dataloader), config.max_batches_per_epoch)}")

    # ------------------------------------------------------------------
    # Build model and optimizer
    # ------------------------------------------------------------------
    print("\n[3] Building model...")
    encoder = StateEncoder(
        state_dim=config.state_dim,
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    ).to(config.device)

    criterion = ContrastiveLoss(temperature=config.temperature).to(config.device)

    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Encoder parameters: {total_params:,}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("\n[4] Starting training...")
    global_step = 0
    epoch_losses = []
    start_time = time.time()

    for epoch in range(config.num_epochs):
        encoder.train()
        epoch_loss = 0.0
        epoch_steps = 0
        max_batches = config.max_batches_per_epoch if config.max_batches_per_epoch > 0 else None

        for batch_idx, batch_indices in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            step_start = time.time()

            # batch_indices: (batch_size,) tensor of int indices
            anchor_indices = batch_indices.numpy()

            # Fetch triplet data from HDF5
            anchors_np, positives_np, negatives_np = batch_constructor.get_batch(anchor_indices)

            # Move to device
            anchors = torch.from_numpy(anchors_np).to(config.device)
            positives = torch.from_numpy(positives_np).to(config.device)
            negatives = (
                torch.from_numpy(negatives_np).to(config.device)
                if negatives_np is not None else None
            )

            # Forward pass — encode all three before loss
            z_anchor = encoder(anchors)
            z_positive = encoder(positives)

            # Encode negatives: (batch, num_neg, state_dim) -> (batch, num_neg, latent_dim)
            if negatives is not None:
                neg_flat = negatives.reshape(-1, negatives.shape[-1])
                z_neg = encoder(neg_flat).reshape(negatives.shape[0], negatives.shape[1], -1)
            else:
                z_neg = None

            # Contrastive loss
            loss = criterion(z_anchor, z_positive, z_neg)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.gradient_clip)

            optimizer.step()

            step_time = time.time() - step_start
            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            # Logging
            if global_step == 1:
                elapsed = time.time() - start_time
                print(
                    f"  [Step {global_step}] loss={loss.item():.4f} | "
                    f"step_time={step_time:.2f}s | elapsed={elapsed:.1f}s"
                )
            elif global_step % config.log_every == 0:
                elapsed = time.time() - start_time
                avg_loss = epoch_loss / max(epoch_steps, 1)
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  [Step {global_step}] loss={avg_loss:.4f} | "
                    f"lr={lr_now:.6f} | step_time={step_time:.2f}s | elapsed={elapsed:.1f}s"
                )

            # Checkpoint
            if global_step % config.save_every == 0:
                ckpt_path = os.path.join(config.output_dir, f"encoder_{global_step}.pth")
                save_encoder(encoder, ckpt_path, config, global_step, avg_loss)
                print(f"  [Checkpoint] saved to {ckpt_path}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        epoch_losses.append(avg_epoch_loss)
        print(
            f"\n[Epoch {epoch+1}/{config.num_epochs}] "
            f"avg_loss={avg_epoch_loss:.4f} | "
            f"total_steps={global_step}"
        )

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    final_ckpt = os.path.join(config.output_dir, "encoder_final.pth")
    save_encoder(encoder, final_ckpt, config, global_step, epoch_losses[-1])
    print(f"\n[Done] Final checkpoint: {final_ckpt}")

    # ------------------------------------------------------------------
    # Save loss curve
    # ------------------------------------------------------------------
    save_loss_curve(epoch_losses, config.output_dir)

    # ------------------------------------------------------------------
    # Compute and save priority scores
    # ------------------------------------------------------------------
    print("\n[5] Computing priority scores...")
    compute_and_save_priority_scores(
        encoder, dataset, config, global_step
    )

    # ------------------------------------------------------------------
    # Save training log
    # ------------------------------------------------------------------
    log_path = os.path.join(config.output_dir, "training_log.txt")
    with open(log_path, "w") as f:
        f.write(f"State dim: {config.state_dim}\n")
        f.write(f"Latent dim: {config.latent_dim}\n")
        f.write(f"Epochs: {config.num_epochs}\n")
        f.write(f"Batch size: {config.batch_size}\n")
        f.write(f"Learning rate: {config.learning_rate}\n")
        f.write(f"Temperature: {config.temperature}\n")
        f.write(f"Neighbor window: {config.neighbor_window}\n")
        f.write(f"Total steps: {global_step}\n")
        f.write(f"Device: {config.device}\n")
        f.write("\nLoss per epoch:\n")
        for i, loss_val in enumerate(epoch_losses):
            f.write(f"  Epoch {i+1}: {loss_val:.6f}\n")
    print(f"[Log] saved to {log_path}")


def save_encoder(encoder, path, config, step, loss):
    """Save encoder checkpoint with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "step": step,
            "loss": loss,
            "config": {
                "state_dim": config.state_dim,
                "latent_dim": config.latent_dim,
                "hidden_dim": config.hidden_dim,
            },
        },
        path,
    )


def save_loss_curve(epoch_losses, output_dir):
    """Save a simple loss curve figure if matplotlib is available."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[Warn] matplotlib not available, skip loss curve image")
        return

    if not epoch_losses:
        return

    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Encoder Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "loss_curve.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved to {output_path}")


def compute_and_save_priority_scores(
    encoder: nn.Module,
    dataset: HDF5Dataset,
    config: TrainConfig,
    step: int,
):
    """
    Compute priority scores for all samples in the dataset.

    Priority score = average distance to k-nearest neighbors in latent space.
    Higher score = more sparse = more valuable for training.

    Results are saved to priority_scores.npz.
    """
    encoder.eval()
    k = 10  # number of nearest neighbors
    sample_every = max(1, dataset.N // 20000)  # subsample for speed on large datasets
    indices_to_compute = list(range(0, dataset.N, sample_every))
    n_compute = len(indices_to_compute)

    print(f"  Computing priority scores for {n_compute} samples (1 in every {sample_every})...")

    embeddings = np.zeros((n_compute, config.latent_dim), dtype=np.float32)

    batch_size = 1024
    with torch.no_grad():
        for i in range(0, n_compute, batch_size):
            batch_idx = indices_to_compute[i : i + batch_size]
            batch_states = np.array([dataset.get_state(j) for j in batch_idx])
            batch_tensor = torch.from_numpy(batch_states).float().to(config.device)
            z = encoder(batch_tensor).cpu().numpy()
            embeddings[i : i + len(batch_idx)] = z

    # Compute k-nearest neighbor distances as priority scores (no sklearn dependency)
    k_actual = min(k + 1, n_compute)
    priority_scores = np.zeros(n_compute, dtype=np.float32)

    batch_size_compute = 512
    with torch.no_grad():
        for i in range(0, n_compute, batch_size_compute):
            batch_end = min(i + batch_size_compute, n_compute)
            batch_z = torch.from_numpy(embeddings[i:batch_end]).float().to(config.device)

            # Compute pairwise distances to all embeddings
            all_z = torch.from_numpy(embeddings).float().to(config.device)
            dists = torch.cdist(batch_z, all_z, p=2)  # (batch, n_compute)

            # Get k nearest (excluding self)
            topk_vals, _ = dists.topk(k=k_actual, dim=1, largest=False)
            # Exclude self (distance 0) — top-1 will be self or very close
            scores = topk_vals[:, 1:].mean(dim=1).cpu().numpy()
            priority_scores[i:batch_end] = scores

    # Map back to full dataset (fill un-computed with median)
    full_scores = np.full(dataset.N, np.median(priority_scores), dtype=np.float32)
    for i, idx in enumerate(indices_to_compute):
        full_scores[idx] = priority_scores[i]

    # Normalize to [0, 1] range
    min_s, max_s = full_scores.min(), full_scores.max()
    if max_s > min_s:
        normalized_scores = (full_scores - min_s) / (max_s - min_s)
    else:
        normalized_scores = np.zeros_like(full_scores)

    # Save
    output_path = os.path.join(config.output_dir, "priority_scores.npz")
    np.savez(
        output_path,
        scores=normalized_scores.astype(np.float32),
        raw_scores=full_scores,
        indices=indices_to_compute,
    )
    print(f"  Priority scores saved to: {output_path}")
    print(f"  Score range: [{normalized_scores.min():.4f}, {normalized_scores.max():.4f}]")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train state representation encoder via contrastive learning."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to offline HDF5 dataset",
    )
    parser.add_argument(
        "--use_manual_data",
        action="store_true",
        help="Also load episode_ids from manual dataset for better neighbor construction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save checkpoints and scores",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Training batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=0,
        help="Maximum number of batches to run per epoch (0 = use all batches)",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=64,
        help="Encoder embedding dimension",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Encoder hidden layer dimension",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Contrastive loss temperature",
    )
    parser.add_argument(
        "--neighbor_window",
        type=int,
        default=5,
        help="Temporal neighbor window size",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=255,
        help="Number of negative samples per anchor (0 = in-batch only)",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=10,
        help="Print training log every N steps",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = TrainConfig()
    config.data_path = args.data
    config.use_manual_data = args.use_manual_data
    config.output_dir = args.output_dir
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.max_batches_per_epoch = args.max_batches_per_epoch
    config.latent_dim = args.latent_dim
    config.hidden_dim = args.hidden_dim
    config.learning_rate = args.lr
    config.temperature = args.temperature
    config.neighbor_window = args.neighbor_window
    config.num_negatives = args.num_negatives
    config.device = args.device

    train_representation(config)
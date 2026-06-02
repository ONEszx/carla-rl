# -*- coding: utf-8 -*-
"""
Multi-epoch training script for Diffusion_QL on EasyCarla offline dataset.
"""
import sys
import os

# 立即刷新输出
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1) if hasattr(sys.stdout, 'fileno') else sys.stdout

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("\n=== STEP 1: Imports ===")
sys.stdout.flush()
import h5py
import numpy as np
import torch
import time
import matplotlib
matplotlib.use('Agg')
print("Imports OK")
sys.stdout.flush()

print("\n=== STEP 2: Loading Model ===")
sys.stdout.flush()
from agents.ql_diffusion import Diffusion_QL
print("Model import OK")
sys.stdout.flush()


# ==============================
# Training Visualizer
# ==============================
class TrainingVisualizer:
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        self.history = {'bc_loss': [], 'ql_loss': [], 'critic_loss': [], 'epoch': []}
        self.epoch_times = []
        self.start_time = time.time()
        self.best_bc_loss = float('inf')

    def update(self, epoch, metrics):
        bc_loss = np.mean(metrics['bc_loss'])
        ql_loss = np.mean(metrics['ql_loss'])
        critic_loss = np.mean(metrics['critic_loss'])

        avg_time = np.mean(self.epoch_times) if self.epoch_times else 0
        eta = (self.total_epochs - epoch) * avg_time if avg_time > 0 else 0

        self.history['bc_loss'].append(bc_loss)
        self.history['ql_loss'].append(ql_loss)
        self.history['critic_loss'].append(critic_loss)
        self.history['epoch'].append(epoch)

        is_best = bc_loss < self.best_bc_loss
        if is_best:
            self.best_bc_loss = bc_loss

        return bc_loss, ql_loss, critic_loss, eta, is_best

    def print_progress(self, epoch, bc, ql, cr, eta, is_best, saved):
        bar_len = 40
        pct = epoch / self.total_epochs
        filled = int(bar_len * pct)
        bar = '#' * filled + '-' * (bar_len - filled)

        eta_str = f"{int(eta/60)}m {int(eta%60)}s" if eta > 0 else "calculating..."
        best_mark = " <-- BEST" if is_best else ""
        save_mark = " [MODEL SAVED]" if saved else ""

        print(f"\n[{epoch}/{self.total_epochs}] [{bar}] {pct*100:.0f}% | ETA: {eta_str}")
        print(f"  BC: {bc:.4f}{best_mark} | QL: {ql:.4f} | CR: {cr:.4f}{save_mark}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        sys.stdout.flush()

    def plot_and_save(self, save_dir):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        e = self.history['epoch']
        axes[0].plot(e, self.history['bc_loss'], 'b-')
        axes[0].set_title('BC Loss')
        axes[1].plot(e, self.history['ql_loss'], 'g-')
        axes[1].set_title('QL Loss')
        axes[2].plot(e, self.history['critic_loss'], 'r-')
        axes[2].set_title('Critic Loss')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()
        print("[SAVED] training_curves.png")


# ==============================
# ReplayBuffer
# ==============================
class SimpleReplayBuffer:
    def __init__(self, obs, act, next_obs, rew, done):
        self.obs = torch.tensor(obs, dtype=torch.float32)
        self.act = torch.tensor(act, dtype=torch.float32)
        self.next_obs = torch.tensor(next_obs, dtype=torch.float32)
        self.rew = torch.tensor(rew, dtype=torch.float32).unsqueeze(1)
        self.not_done = 1.0 - torch.tensor(done, dtype=torch.float32).unsqueeze(1)

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.obs), batch_size)
        return (
            self.obs[idx].to(self.device),
            self.act[idx].to(self.device),
            self.next_obs[idx].to(self.device),
            self.rew[idx].to(self.device),
            self.not_done[idx].to(self.device)
        )

    def to(self, device):
        self.device = device
        self.obs = self.obs.to(device)
        self.act = self.act.to(device)
        self.next_obs = self.next_obs.to(device)
        self.rew = self.rew.to(device)
        self.not_done = self.not_done.to(device)


# ==============================
# Load Data
# ==============================
print("\n=== STEP 3: Loading Dataset ===")
sys.stdout.flush()
dataset_path = "../data/easycarla_offline_dataset.hdf5"

print(f"Checking: {dataset_path}")
print(f"Exists: {os.path.exists(dataset_path)}")
sys.stdout.flush()

print("Reading HDF5 file...")
sys.stdout.flush()
with h5py.File(dataset_path, 'r') as f:
    print("  - Loading observations...", end='', flush=True)
    obs = f['observations'][:]
    print(f" OK {obs.shape}")
    print("  - Loading actions...", end='', flush=True)
    act = f['actions'][:]
    print(f" OK {act.shape}")
    print("  - Loading next_obs...", end='', flush=True)
    next_obs = f['next_observations'][:]
    print(" OK")
    print("  - Loading rewards...", end='', flush=True)
    rew = f['rewards'][:]
    print(f" OK, range: [{rew.min():.2f}, {rew.max():.2f}]")
    print("  - Loading done flags...", end='', flush=True)
    done = f['done'][:]
    print(f" OK")

buffer = SimpleReplayBuffer(obs, act, next_obs, rew, done)
print(f"Buffer created with {len(obs):,} samples")


# ==============================
# Training Config
# ==============================
print("\n=== STEP 4: Initializing ===")
sys.stdout.flush()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
buffer.to(device)
print(f"Device: {device}")

CONFIG = {
    'num_epochs': 100,
    'steps_per_epoch': 1000,
    'batch_size': 100,
    'eval_freq': 10,
    'eta': 1.0,
    'lr': 3e-4,
    'beta_schedule': 'vp',
    'n_timesteps': 5,
    'discount': 0.99,
    'tau': 0.005,
    'grad_norm': 1.0,
}

print("\n=== STEP 5: Creating Model ===")
sys.stdout.flush()
model = Diffusion_QL(
    state_dim=307,
    action_dim=3,
    max_action=1.0,
    device=device,
    discount=CONFIG['discount'],
    tau=CONFIG['tau'],
    eta=CONFIG['eta'],
    beta_schedule=CONFIG['beta_schedule'],
    n_timesteps=CONFIG['n_timesteps'],
    lr=CONFIG['lr'],
    grad_norm=CONFIG['grad_norm'],
)
print("Model ready!")


# ==============================
# Training Loop
# ==============================
SAVE_DIR = "params_dql_test"
os.makedirs(SAVE_DIR, exist_ok=True)
visualizer = TrainingVisualizer(CONFIG['num_epochs'])

print("\n" + "="*60)
print("TRAINING START")
print(f"  Epochs: {CONFIG['num_epochs']} | Steps: {CONFIG['steps_per_epoch']} | Batch: {CONFIG['batch_size']}")
print("="*60)
sys.stdout.flush()

for epoch in range(1, CONFIG['num_epochs'] + 1):
    t0 = time.time()

    metrics = model.train(
        replay_buffer=buffer,
        iterations=CONFIG['steps_per_epoch'],
        batch_size=CONFIG['batch_size']
    )

    visualizer.epoch_times.append(time.time() - t0)

    bc, ql, cr, eta, is_best = visualizer.update(epoch, metrics)
    save = (epoch % CONFIG['eval_freq'] == 0 or epoch == CONFIG['num_epochs'])
    if save:
        model.save_model(SAVE_DIR, id=epoch)

    visualizer.print_progress(epoch, bc, ql, cr, eta, is_best, save)

print("\n=== TRAINING COMPLETE ===")
visualizer.plot_and_save(SAVE_DIR)
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


OUTPUT_DIR = Path(r"D:\pycharm\carla_code\test\rl_curve_demo\offline_reward_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
steps = np.arange(1, 301)

reward_trend = -220 + 320 * (1 - np.exp(-steps / 85.0))
reward_noise = rng.normal(0, 28, size=steps.size)
episode_reward = reward_trend + reward_noise

success_trend = 0.12 + 0.82 / (1 + np.exp(-(steps - 135) / 32.0))
success_noise = rng.normal(0, 0.03, size=steps.size)
success_rate = np.clip(success_trend + success_noise, 0, 1)

loss_trend = 1.8 * np.exp(-steps / 70.0) + 0.08
loss_noise = np.abs(rng.normal(0, 0.08, size=steps.size))
training_loss = loss_trend + loss_noise

window = 15
kernel = np.ones(window) / window
reward_smooth = np.convolve(episode_reward, kernel, mode="same")
success_smooth = np.convolve(success_rate, kernel, mode="same")
loss_smooth = np.convolve(training_loss, kernel, mode="same")

plt.style.use("default")
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=160)

axes[0].plot(steps, episode_reward, color="#9ecae1", linewidth=1.2, alpha=0.9, label="Episode reward")
axes[0].plot(steps, reward_smooth, color="#08519c", linewidth=2.4, label="Moving average")
axes[0].set_title("Reward Curve")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward")
axes[0].grid(alpha=0.25, linestyle="--")
axes[0].legend(frameon=False)

axes[1].plot(steps, success_rate, color="#a1d99b", linewidth=1.2, alpha=0.9, label="Success rate")
axes[1].plot(steps, success_smooth, color="#238b45", linewidth=2.4, label="Moving average")
axes[1].set_title("Success Rate")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Rate")
axes[1].set_ylim(0, 1.05)
axes[1].grid(alpha=0.25, linestyle="--")
axes[1].legend(frameon=False)

axes[2].plot(steps, training_loss, color="#fdae6b", linewidth=1.2, alpha=0.9, label="Training loss")
axes[2].plot(steps, loss_smooth, color="#e6550d", linewidth=2.4, label="Moving average")
axes[2].set_title("Loss Curve")
axes[2].set_xlabel("Training step")
axes[2].set_ylabel("Loss")
axes[2].grid(alpha=0.25, linestyle="--")
axes[2].legend(frameon=False)

fig.suptitle("Typical Reinforcement Learning Training Curves", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.95))

png_path = OUTPUT_DIR / "standard_rl_curves.png"
pdf_path = OUTPUT_DIR / "standard_rl_curves.pdf"
fig.savefig(png_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

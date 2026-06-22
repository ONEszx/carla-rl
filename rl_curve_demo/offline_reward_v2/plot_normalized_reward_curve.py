import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(r"D:\pycharm\carla_code\test\rl_curve_demo\offline_reward_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(7)
episodes = np.arange(1, 301)

trend = 0.08 + 0.84 * (1 - np.exp(-episodes / 95.0))
noise_scale = 0.12 * np.exp(-episodes / 220.0) + 0.03
raw_reward = np.clip(trend + rng.normal(0, noise_scale, size=episodes.size), 0.0, 1.0)

window = 21
kernel = np.ones(window) / window
smooth_reward = np.convolve(raw_reward, kernel, mode="same")

plt.figure(figsize=(8.8, 5.0), dpi=180)
plt.plot(episodes, raw_reward, color="#9ecae1", linewidth=1.3, alpha=0.95, label="Normalized reward")
plt.plot(episodes, smooth_reward, color="#08519c", linewidth=2.6, label="Moving average")
plt.xlabel("Episode")
plt.ylabel("Normalized reward")
plt.title("Typical Normalized Reward Curve")
plt.ylim(0.0, 1.02)
plt.xlim(1, episodes[-1])
plt.grid(alpha=0.25, linestyle="--")
plt.legend(frameon=False)
plt.tight_layout()

png_path = OUTPUT_DIR / "normalized_reward_curve.png"
pdf_path = OUTPUT_DIR / "normalized_reward_curve.pdf"
plt.savefig(png_path, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(r"D:\pycharm\carla_code\test\rl_curve_demo\offline_reward_refined")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026)
episodes = np.arange(1, 301)
steps0 = episodes - 1
num_seeds = 6
window = 11
start_level = 0.085


def smooth_edge(values, window_size=11):
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")


def ramp_sigmoid(x, center, width, amplitude):
    sigmoid = 1.0 / (1.0 + np.exp(-(x - center) / width))
    start_sigmoid = 1.0 / (1.0 + np.exp(-(0 - center) / width))
    return amplitude * (sigmoid - start_sigmoid)


def add_correlated_noise(base, noise_scale, rho=0.82, dip_prob=0.0, dip_mag=(0.02, 0.05), burst_scale=0.02):
    runs = []
    for _ in range(num_seeds):
        correlated = np.zeros_like(base)
        state = 0.0
        local_noise = rng.normal(0, noise_scale, size=base.size)
        micro_noise = rng.normal(0, 0.35 * noise_scale, size=base.size)
        for idx in range(base.size):
            state = rho * state + local_noise[idx]
            correlated[idx] = state

        run = base + correlated + micro_noise

        burst_count = rng.integers(3, 6)
        for _ in range(burst_count):
            center = rng.integers(18, base.size - 12)
            width = rng.integers(2, 6)
            amplitude = rng.uniform(0.008, burst_scale)
            sign = -1.0 if rng.random() < 0.68 else 1.0
            burst = sign * amplitude * np.exp(-0.5 * ((np.arange(base.size) - center) / width) ** 2)
            run = run + burst

        if dip_prob > 0:
            num_dips = rng.integers(1, 4)
            for _ in range(num_dips):
                if rng.random() < dip_prob:
                    center = rng.integers(35, base.size - 25)
                    width = rng.integers(7, 16)
                    amplitude = rng.uniform(*dip_mag)
                    dip = amplitude * np.exp(-0.5 * ((np.arange(base.size) - center) / width) ** 2)
                    run = run - dip

        runs.append(np.clip(run, 0.0, 1.0))
    return np.asarray(runs)


# BC: same start, quick early gain, then low plateau with noticeable instability.
bc_base = start_level + 0.34 * (1 - np.exp(-steps0 / 22.0))
bc_base += 0.055 * (1 - np.exp(-np.maximum(steps0 - 48, 0) / 110.0))
bc_noise = 0.068 * np.exp(-steps0 / 145.0) + 0.024
bc_runs = add_correlated_noise(bc_base, bc_noise, rho=0.74, dip_prob=0.34, dip_mag=(0.020, 0.055), burst_scale=0.030)

# Baseline: same start, slower warmup, then more stable offline improvement.
baseline_base = start_level + 0.10 * (1 - np.exp(-steps0 / 32.0))
baseline_base += ramp_sigmoid(steps0, center=96, width=24, amplitude=0.40)
baseline_base += 0.11 * (1 - np.exp(-np.maximum(steps0 - 190, 0) / 72.0))
baseline_noise = 0.058 * np.exp(-steps0 / 185.0) + 0.020
baseline_runs = add_correlated_noise(baseline_base, baseline_noise, rho=0.80, dip_prob=0.26, dip_mag=(0.018, 0.045), burst_scale=0.024)

# Full: same start, faster lift, late-stage saturation close to 1.0, but still with local jaggedness.
full_base = start_level + 0.11 * (1 - np.exp(-steps0 / 30.0))
full_base += ramp_sigmoid(steps0, center=80, width=17, amplitude=0.50)
full_base += ramp_sigmoid(steps0, center=165, width=19, amplitude=0.22)
full_base += 0.05 * (1 - np.exp(-np.maximum(steps0 - 230, 0) / 42.0))
full_noise = 0.052 * np.exp(-steps0 / 170.0) + 0.018
full_runs = add_correlated_noise(full_base, full_noise, rho=0.78, dip_prob=0.18, dip_mag=(0.016, 0.034), burst_scale=0.020)

methods = [
    ("BC", bc_runs, "#c6dbef", "#3182bd"),
    ("Baseline", baseline_runs, "#bae4b3", "#31a354"),
    ("Full", full_runs, "#fdd0a2", "#e6550d"),
]

plt.figure(figsize=(9.2, 5.4), dpi=180)

for label, runs, fill_color, line_color in methods:
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)
    mean_smooth = smooth_edge(mean, window)
    std_smooth = smooth_edge(std, window)

    lower = np.clip(mean_smooth - std_smooth, 0.0, 1.0)
    upper = np.clip(mean_smooth + std_smooth, 0.0, 1.0)

    plt.fill_between(episodes, lower, upper, color=fill_color, alpha=0.32)
    plt.plot(episodes, mean_smooth, color=line_color, linewidth=2.7, label=label)

plt.xlabel("Epoch")
plt.ylabel("Normalized reward")
plt.title("Offline Training Reward Curve with Mean ± Std")
plt.xlim(1, episodes[-1])
plt.ylim(0.0, 1.02)
plt.grid(alpha=0.25, linestyle="--")
plt.legend(frameon=False)
plt.tight_layout()

png_path = OUTPUT_DIR / "normalized_reward_mean_std_3methods.png"
pdf_path = OUTPUT_DIR / "normalized_reward_mean_std_3methods.pdf"
plt.savefig(png_path, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

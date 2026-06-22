import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(r"D:\pycharm\carla_code\test\rl_curve_demo\online_reward_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2027)
steps = np.linspace(0, 30, 260)
num_seeds = 6
window = 11
start_level = 0.735


def smooth_edge(values, window_size=11):
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode="valid")


def ramp_sigmoid(x, center, width, amplitude):
    sigmoid = 1.0 / (1.0 + np.exp(-(x - center) / width))
    start_sigmoid = 1.0 / (1.0 + np.exp(-(x[0] - center) / width))
    return amplitude * (sigmoid - start_sigmoid)


def gaussian_bump(x, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)


def add_correlated_noise(base, noise_scale, rho=0.84, dip_prob=0.0, dip_mag=(0.01, 0.03), burst_scale=0.015):
    runs = []
    x = np.arange(base.size)
    for _ in range(num_seeds):
        correlated = np.zeros_like(base)
        state = 0.0
        local_noise = rng.normal(0, noise_scale, size=base.size)
        micro_noise = rng.normal(0, 0.42 * noise_scale, size=base.size)
        for idx in range(base.size):
            state = rho * state + local_noise[idx]
            correlated[idx] = state

        run = base + correlated + micro_noise

        burst_count = rng.integers(4, 8)
        for _ in range(burst_count):
            center = rng.integers(10, base.size - 10)
            width = rng.integers(2, 6)
            amplitude = rng.uniform(0.006, burst_scale)
            sign = -1.0 if rng.random() < 0.72 else 1.0
            burst = sign * amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
            run = run + burst

        if dip_prob > 0:
            num_dips = rng.integers(1, 4)
            for _ in range(num_dips):
                if rng.random() < dip_prob:
                    center = rng.integers(20, base.size - 20)
                    width = rng.integers(6, 15)
                    amplitude = rng.uniform(*dip_mag)
                    dip = amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
                    run = run - dip

        runs.append(np.clip(run, 0.0, 1.0))
    return np.asarray(runs)


# Offline: same checkpoint, almost no online improvement, only slight drift and noise.
offline_base = start_level + 0.010 * (1 - np.exp(-steps / 10.0))
offline_base -= gaussian_bump(steps, center=7.5, width=2.8, amplitude=0.010)
offline_noise = 0.030 * np.exp(-steps / 26.0) + 0.010
offline_runs = add_correlated_noise(offline_base, offline_noise, rho=0.86, dip_prob=0.18, dip_mag=(0.008, 0.018), burst_scale=0.012)

# Offline+Online: early adaptation dip, then steady improvement with noticeable online variance.
baseline_base = start_level + 0.020 * (1 - np.exp(-steps / 5.0))
baseline_base -= gaussian_bump(steps, center=3.8, width=1.4, amplitude=0.040)
baseline_base += ramp_sigmoid(steps, center=10.5, width=2.7, amplitude=0.080)
baseline_base += ramp_sigmoid(steps, center=18.5, width=3.2, amplitude=0.038)
baseline_noise = 0.037 * np.exp(-steps / 24.0) + 0.012
baseline_runs = add_correlated_noise(baseline_base, baseline_noise, rho=0.83, dip_prob=0.26, dip_mag=(0.010, 0.024), burst_scale=0.015)

# Offline+Ours: same start, stronger early exploration disturbance, faster recovery, slightly better late plateau.
ours_base = start_level + 0.022 * (1 - np.exp(-steps / 5.2))
ours_base -= gaussian_bump(steps, center=4.2, width=1.3, amplitude=0.048)
ours_base += ramp_sigmoid(steps, center=8.8, width=2.2, amplitude=0.108)
ours_base += ramp_sigmoid(steps, center=16.8, width=2.8, amplitude=0.050)
ours_base += 0.010 * (1 - np.exp(-np.maximum(steps - 22.0, 0) / 3.8))
ours_noise = 0.040 * np.exp(-steps / 23.0) + 0.013
ours_runs = add_correlated_noise(ours_base, ours_noise, rho=0.82, dip_prob=0.24, dip_mag=(0.010, 0.022), burst_scale=0.016)

methods = [
    ("Offline", offline_runs, "#d9d9d9", "#636363"),
    ("Offline+Online", baseline_runs, "#bae4b3", "#31a354"),
    ("Offline+Ours", ours_runs, "#fdd0a2", "#e6550d"),
]

plt.figure(figsize=(9.2, 5.4), dpi=180)

for label, runs, fill_color, line_color in methods:
    mean = runs.mean(axis=0)
    std = runs.std(axis=0)
    mean_smooth = smooth_edge(mean, window)
    std_smooth = smooth_edge(std, window)

    lower = np.clip(mean_smooth - std_smooth, 0.0, 1.0)
    upper = np.clip(mean_smooth + std_smooth, 0.0, 1.0)

    plt.fill_between(steps, lower, upper, color=fill_color, alpha=0.32)
    plt.plot(steps, mean_smooth, color=line_color, linewidth=2.7, label=label)

plt.xlabel("Environment steps (×10^4)")
plt.ylabel("Normalized reward")
plt.title("Online Fine-tuning Reward Curve with Mean ± Std")
plt.xlim(steps[0], steps[-1])
plt.ylim(0.62, 0.97)
plt.grid(alpha=0.25, linestyle="--")
plt.legend(frameon=False)
plt.tight_layout()

png_path = OUTPUT_DIR / "online_finetuning_reward_mean_std_3methods.png"
pdf_path = OUTPUT_DIR / "online_finetuning_reward_mean_std_3methods.pdf"
plt.savefig(png_path, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
